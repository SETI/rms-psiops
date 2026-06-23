# Test Suite Critique Report

**Generated:** 2026-06-22
**Scope:** `tests/` (24 test modules, `conftest.py`, `_resize_reference.py`)
**Method:** Full read of every test file plus its source module, cross-referenced against a
full-suite coverage run and the active pytest configuration.

## Executive summary

The suite is in good shape: **661 tests pass in ~13.5 s**, total coverage is **96.92 %**
(branch coverage on), all randomness is seeded, comparisons use justified tolerances against
independent NumPy/SciPy references, and the photometric "signal-conservation" property is
asserted where it matters (rotate area, resample sum-scaling, unzoom mean). The pytest config
already enables the practices the checklist asks for — `testpaths`, `--strict-markers`,
`--strict-config`, and `filterwarnings = ["error"]` are all present, and the autouse
`_restore_shortcuts` fixture correctly insulates the one global flag the suite toggles most.

The dominant weaknesses are **structural and cross-cutting**, not correctness bugs:

- **Coverage measurement:** Met and full-suite — 96.92 % total, every shipped module ≥ 90 %.
  The lone soft spot is `_validation.py` at **84 %**, and it is the project's primary
  input-validation layer (details in §18/§8).
- **Exception messages:** Practice is *uneven*. The statistics and `_utils` tests assert
  message **contents** via `str(exc_info.value)` (the correct standard); the transform,
  filter/fit, model, and camouflage tests overwhelmingly assert exception **type only**, so a
  test can pass on the *wrong* `ValueError` (§12).
- **The `shortcuts` parametrization is applied inconsistently.** The autouse machinery exists,
  but the optimized-vs-general code paths are systematically cross-checked in only a fraction
  of modules — most reduction functions, all of `stdev`, and five of six filter/fit modules
  never exercise both flag settings on identical inputs (§4, §6).
- **Large amounts of copy-pasted setup** (`image0` builders, irregular-footprint brute-force
  loops, `zero_size`/`footprint_forms` blocks) are duplicated within and across the
  near-identical statistics and transform modules, with almost nothing factored into shared
  fixtures (§5).
- **`_usable_bytes` global state is not restored by any autouse fixture**, unlike
  `_use_shortcuts` — a latent parallel-isolation hazard (§6, §14).

**High-priority fixes:** (1) direct tests for `_check_image` / `_validation.py` error paths
with message assertions; (2) systematic `shortcuts` cross-checks across all reduction/filter
modules; (3) add `match=` (message-content) assertions to the type-only `pytest.raises` calls;
(4) add an autouse fixture restoring `_usable_bytes`; (5) replace the `test_outliers.py` mock of
the real `gaussian_filter` collaborator (or track the underlying breakage).
**Nice-to-have:** extract shared image-builder fixtures, convert loop-based cases to
`@pytest.mark.parametrize`, and clean up / regularize the `ANSWERS` golden tables.

---

## 1. Return values and assertions

Generally strong. Tolerance-based comparisons are used appropriately for float math
(`np.allclose`, `np.abs(a-b).max() < 1e-14`) and exact `==` only where operations are
integer-exact (`test_mean_basic`, `test_median_basic`, zoom replication `test_zoom.py:37`,
unzoom means `test_zoom.py:137`). Specific weaknesses:

- **Existence/shape-only asserts where a value check is feasible:**
  - `test_outliers.py:137,144,151,171` assert only `result.shape == (40,40)`. With
    `weights=ones` the result should *equal* the no-weights result — that stronger equality
    is not checked.
  - `test_camouflage.py:91` `test_returns_imw_provides_three` asserts only `isinstance(list)`
    and `len == 3`; it never checks the three elements are image/mask/weights or their shapes
    (the parallel `iw`/`im` tests do check contents).
- **Mislabeled assertion:** `test_camouflage.py:151` `test_size_parameter_changes_footprint_count`
  never checks any footprint count — it only asserts `result[6,6] == approx(1.0)` per size.
  Either the name or the assertion is wrong.
- **Return arity asserted unevenly:** the tuple-vs-array contract is checked explicitly only in
  `test_minimum.py:251` / `test_maximum.py` (`returns_variants`, exact `len == 2`/`3`);
  mean/median/stdev/variance rely on implicit tuple-unpacking and never assert arity.

## 2. Success and failure conditions

Happy paths and the `size cannot be zero` empty-array path are covered consistently. Gaps:

- **`shift()` has no `pytest.raises` at all** — invalid `mode`/`returns`, sub-2-D image, and
  incompatible mask/weights shapes (documented `shift.py:75-80`) are never driven through the
  public function, whereas `ishift` (`test_ishift.py:240`) and `rotate`
  (`test_rotate.py:292-301`) do test these.
- **`nans=True` is barely tested.** Only `test_mean_nans:249` and `test_variance_nans:203`
  exercise it; median/stdev/minimum/maximum and *all six* transform modules document `nans=`
  but never test it.
- **Non-numeric `TypeError`** (every reduction docstring promises it) is tested only in
  `test_minimum.py:277` / `test_maximum.py:280`; mean/median/stdev/variance never hit it.
- **Model constructors have no invalid-arg tests:** `SummedModel` uses `zip(..., strict=True)`
  (`summedmodel.py:52`) → `ValueError` on length mismatch; `SmearedModel` indexes `smear[...]`
  (`smearedmodel.py:28`) → `IndexError`; `ArrayModel` with non-2-D input — all untested
  (`test_imagemodel.py`).
- **Documented failure modes never asserted:** `rotate`'s `ZeroDivisionError` (`rotate.py:98`),
  `fitting`'s `LinAlgError` on a singular Jacobian (`fitting.py:283-286,338`) and `remask`
  "target must be defined" (`fitting.py:161`), and `Stretch`'s degenerate-`inv` path
  (`stretch.py:326`).
- **Empty / single-pixel arrays** are not tested in any transform module.

## 3. Consistency

- **The six statistics modules are structural clones but do not test the same matrix.**
  `maskval`, `MaskedArray` input, `nans`, reduction-level `weights=`, `returns=` arity, and
  non-numeric `TypeError` are each present in some files and silently absent in others —
  **`stdev` is the most under-tested**, lacking `maskval`, `MaskedArray`, `nans`, and `weights=`
  reduction tests entirely.
- **Naming diverges across parallel files:** mean/stdev/variance use `..._axis_variants`,
  `..._mask_2d/_3d`; median/minimum/maximum use `..._axes`, `..._2d_mask/_3d_mask`. Makes
  cross-file diffing of otherwise-identical modules harder than necessary.
- **Structure style splits:** `test_shift.py`/`test_ishift.py` (and median/min/max) use
  monolithic nested-loop tests; rotate/zoom/resize/resample/mean/stdev/variance use small
  single-responsibility functions with section banners. The split style localizes failures
  better.
- **Model hierarchy not tested uniformly** (`test_imagemodel.py`): `SummedModel` and
  `SmearedModel` lack dedicated rotate-preservation tests; only `ArrayModel`/`Gaussian` test
  `expand`-integral preservation in isolation; the `no_shortcuts` fixture is applied to
  `ArrayModel` and `SummedModel` weighted-center but not `SmearedModel` (correct, since the
  latter delegates to Gaussian, but the asymmetry is undocumented).

## 4. Completeness

Coverage is high (§18), but several documented behaviors are untested:

- **`shortcuts` cross-checking is the biggest completeness gap.** The `shortcuts` fixture flips
  `_USE_SHORTCUTS`, which gates a large branch in every reduction (`mean.py:109`,
  `median.py:123`, `variance.py:141`) and filter. Yet:
  - `stdev` takes the fixture in *no* test except `zero_size`.
  - `median`/`variance` value-match tests run only with shortcuts ON.
  - The fft/gaussian/fitting/stretch/outliers group **never uses the fixture at all**
    (`test_outliers.py:65` hard-disables shortcuts), and `test_imagemodel.py` applies its local
    `no_shortcuts` only to `ArrayModel`.
  The two paths are therefore not systematically verified equal on identical inputs.
- **`_check_image` is never directly tested** (`test_validation.py` imports only
  `_check_return`). MaskedArray handling, `floats` conversion, `maskval`/`nans` masking,
  weights→mask merging, and `returns` auto-derivation are exercised only transitively.
- **Untested public-internal `_utils` helpers:** `_merge_weights`, `_pixel_area`,
  `_flatten_axes`, and the `_ImageInfo.__post_init__` ValueError path have no direct test.
- **`fitting` `zoom`/`rotate` fitting is never exercised** (flags only ever enable x/y), and the
  `corr`/`denom==0` branch is untested.
- **`gaussian_filter` `cval=None` masked path** (`gaussian_filter.py:51-54`) untested.
- **Public façade untested here:** model tests import `psiops.imagemodel.*` directly, so the
  `__init__.py` re-export wiring and the `_suppress_divide_warnings` wrapper applied to
  `Gaussian.transform` (`__init__.py:155-164`) are not exercised; fft/stretch/fitting tests
  likewise import submodules rather than the public `psiops` package / `.pyi` surface.

## 5. Redundancy

- **The `image0 = array + array[:,np.newaxis]; image = image0 * [...]` builder is duplicated
  verbatim** ~5× per statistics file and across all six (`test_mean.py:15-17,24-27,38-40`, and
  identically in stdev/median/minimum/maximum). Single biggest fixture opportunity.
- **The irregular-footprint masked-filter brute-force test is copy-pasted near
  character-for-character across all six statistics modules** (`test_mean.py:340`,
  `test_median.py:339`, `test_stdev.py:388`, `test_variance.py:336`, `test_minimum.py:299`,
  `test_maximum.py:301`), differing only in the reducer and the `len(values)` threshold — and
  variance inconsistently sub-samples with `range(0,100,7)` while the others use `range(100)`.
- **`zero_size_raises` and `footprint_forms` blocks** are identical across the statistics files.
- **Image builders duplicated across transform files:** `_make_image`/`_cube_image`/
  `_ramp_stack` recur in `test_resize.py:13`, `test_resample.py:16`, `test_zoom.py:12`,
  `test_shift` — a shared fixture would remove the repetition.
- `test_utils.py:174-200` re-lists ~10 hand-copied window tuples already asserted at
  `:129-171`; could instead assert equality against the no-mask result.
- `test_stretch.py` repeats the `Stretch(...); set_target; fit()` triple in ~15 tests with no
  fixture; `test_fitting.py` property-delegation tests (`:411-443`) are a copy-paste family.

## 6. Parallel execution

- The autouse `_restore_shortcuts` fixture (`conftest.py:13`) correctly saves/restores
  `_USE_SHORTCUTS` around every test — the right pattern under `pytest-xdist` (per-worker
  process). Sound.
- **`_usable_bytes` / `_USABLE_BYTES` is not restored by any fixture.** `test_utils.py` mutates
  it globally (`:327,357,380,…`), mostly with `try/finally`, but
  `test_apply_op_as_filter_default_memory` (`:421-438`) and `..._bad_footprint` (`:441-451`)
  set it with **no restore**. Low current risk (they set `0` early) but a real latent
  isolation hazard. **Recommend an autouse fixture mirroring `_restore_shortcuts`.**
- **Several tests bypass the conftest fixtures and toggle the global flag by hand:**
  `test_shift.py:146-173` reimplements `for status in (False,True): _use_shortcuts(status)` in
  its own `try/finally`; `test_shift.py:178-179,257-258,311-312` toggle with no local restore
  (relying solely on the autouse fixture); `test_resample.py:361,370,380,391` set
  `_use_shortcuts(True)` directly without the fixture. This duplicates the fixture machinery
  and is inconsistent with sibling tests.
- The memory-tiling layer/tile counters (`_apply_op_as_filter_info()`,
  `test_utils.py:338,351,…`) are process-global diagnostics asserted exactly (`layers == 5`);
  safe per-worker but fragile to any future intra-test concurrency.
- No shared mutable module-level arrays; image builders are per-test, so the suite is otherwise
  parallel-safe.

## 7. Mocking and dependency isolation

- **`test_outliers.py` substitutes a hand-written `_masked_gaussian_filter` (`:25-71`) for the
  real `outliers_module.gaussian_filter`** because the real masked mode is "currently broken"
  (module docstring `:7-12`). Consequence: the outlier tests validate against a *mock with
  different behavior*, never exercise the shipped masked path, and the mock returns a plain
  `ndarray` (never the `(image, mask)` tuple the real one returns), which could hide a
  return-contract bug. The patch is a module-attribute swap in a fixture `try/finally`, not via
  the `monkeypatch` fixture — fragile if an exception precedes the `try`.
- `test_fitting.py` `_BlobModel`/`_IdentityStretch` (`:23-101`) are legitimate collaborator test
  doubles, not I/O mocks; but `_IdentityStretch.model` returns `self.image`, so those tests
  verify *delegation wiring*, not real `Stretch` behavior — fine if understood as such.
- No real file/network I/O anywhere — appropriate for a pure-compute numeric library.

## 8. Security and input validation

This is the weakest area relative to its importance.

- **`_validation.py` is the input-validation layer, yet `test_validation.py` exercises none of
  its rejection paths.** Wrong-type image (`TypeError`, `_validation.py:114`),
  complex-without-comps (`:117`), mask shape mismatch (`:133`), `<2-D` mask (`:130`), weights
  shape mismatch (`:172`), `<2-D` weights (`:168`), and `returns` validation (`:72`) are all
  untested directly. The coverage gap is visible: `_validation.py` is the lowest module at
  **84 %** with uncovered lines clustered in exactly these raise branches.
- `_check_tuple` (`test_utils.py:32-80`) and `_check_axis` (`:94-119`) error paths *are*
  thoroughly tested with message assertions — the model the validation tests should follow.
- No invalid-`returns`-string or invalid-`footprint`-dtype tests in the statistics modules,
  though docstrings promise `ValueError` for both.

## 9. Parameterization and data-driven tests

- **Good examples:** `test_variance.py:23` (`vartype`,`ddof`), `test_resample.py:150` (`frac`),
  `test_rotate.py:258` (`use_3d`), `test_validation.py:52,70,81` (`@parametrize('name',
  _REDUCTIONS)`).
- **Loop-where-parametrize-belongs (loses per-case reporting, stops at first failure):**
  - dtype loops `test_median.py:131`, `test_minimum.py:129`, `test_maximum.py:133` (vs. the
    `@parametrize` form used in mean/stdev/variance for the same list);
  - radius loops `test_circle.py:33,69`; size loop `test_camouflage.py:156`; expand loop
    `test_imagemodel.py:92`;
  - the `ishift` per-mode blocks `test_ishift.py:298-336,403-482` repeat the same corner-block
    assertions for nearest/wrap/mirror;
  - the mean/stdev `factors` tests (`..._2d_multi_axis`, `_3d_axis_0_2`, `_3d_axis_0_1`) are
    four near-identical copies.
- **Missing boundary values:** `quantile=0.0`/`cutoff=0` in outliers; `omit=0` alongside ±1 in
  median; `sigma` exactly at an integer boundary in gaussian.

## 10. Async

Not applicable — the library is synchronous NumPy/SciPy compute. No async fixtures or tests.

## 11. Output and contract

- Return-shape contracts (bare array vs `(array, mask)` vs `(…, center)`) are verified well in
  the transform modules (`test_resample.py:255-259`, `test_zoom.py:21-28`,
  `test_rotate.py:332-337`) and in `test_minimum.py:251`. They are *not* asserted in
  mean/median/stdev/variance.
- The `'iw'`/`'imw'` weight-return contract is checked in `test_median.py:310` but not
  symmetrically across the other reduction modules.
- `ImageModel.transform` is abstract and tested for `NotImplementedError`
  (`test_imagemodel.py:46`), but the `.pyi` declares it as a concrete `-> np.ndarray` signature
  (`__init__.pyi:261-267`) — the abstract-vs-stub mismatch is unnoted.

## 12. Error handling and messages

- **Excellent and consistent in the statistics and `_utils` tests:** error tests assert full
  message contents via `str(exc_info.value) == '...'` (`test_mean.py:312`, `test_stdev.py:347`,
  `test_variance.py:251`, `test_minimum.py:281`, `test_utils.py:37-76,99-119`). `zero_size`
  uses `match='size cannot be zero'`. This is the standard the rest of the suite should meet.
- **Type-only elsewhere — the main §12 gap:** transform errors (`test_zoom.py`,
  `test_resize.py:106-113`, `test_rotate.py:292-301`), filter/fit errors
  (`test_gaussian_filter.py:228,235,242`, `test_stretch.py:62,356-380`, `test_reshape.py:105-118`),
  and `test_camouflage.py:171,176` assert only the exception class. Because several distinct
  `ValueError`s share a call site (e.g. `test_resize.py:106-113` raises ValueError for `-3`,
  wrong-length, *and* `None` — indistinguishable), these can pass for the wrong reason. The
  only message assertions outside the statistics/utils files are `test_ishift.py:245`,
  `test_resample.py:354`, and `test_fitting.py:216`. Add `match=` throughout.

## 13. State and workflow

- **Input-mutation / no-side-effect is rarely asserted.** Only `test_camouflage.py:181`
  (`test_does_not_mutate_input_image`), `test_fitting.py:166`
  (`test_set_target_weights_does_not_mutate_input`), and `test_ishift.py:259-262`
  (`new_weights is not weights`) verify copy semantics. The reductions defensively
  copy-or-not based on `info.image_is_copy` and then mutate in place (`image[mask]=...`); a test
  feeding a non-copy ndarray and asserting it is unchanged would be valuable and is absent in
  all six statistics modules and in the transform modules. Same gap for `gaussian_filter`'s
  non-copy branch and `camouflage`'s MaskedArray/weights inputs.
- **Idempotency/round-trip is covered where natural:** `unzoom(zoom(image,3),3) == image`
  (`test_zoom.py:27`), reflect-fold (`test_ishift.py:285`). No idempotency test for
  `camouflage` (re-filling already-filled output).

## 14. Test data and fixtures

- Data is realistic and deterministic (seeded `default_rng`, additive ramps, prime-sized arrays
  in `test_resample.py:134` to stress non-integer factors). Helper builders (`_hole_image`,
  `_ramp_image`, `_weighted_center`, `_reference_circle`) are good.
- **`conftest.py` holds only the two shortcut fixtures — appropriately minimal.** But it should
  gain an **autouse `_usable_bytes` restore** to match `_restore_shortcuts` (§6).
- `test_imagemodel.py:19-29` defines a local `no_shortcuts` fixture that overlaps conceptually
  with the conftest `shortcuts` parametrized fixture; the relationship is undocumented (it
  relies on the conftest autouse restore, correctly noted in its own docstring).
- No overly broad fixture scopes; everything is function-scoped, fine for cheap arrays. No deep
  fixture chains. The single autouse fixture (`_restore_shortcuts`) is well-justified and
  documented.

## 15. Flakiness indicators

- **Low overall risk:** all randomness is seeded, no wall-clock/`uuid` assertions, no order
  dependence given the autouse restore.
- **Brittle RNG-sequence coupling:** `test_rotate.py:118-123`
  (`test_rotate_many_masked_pixels_unweighted`) advances the RNG with a throwaway 300-iteration
  loop "to match historical sequence" before generating its mask — any change to the loop count
  silently changes the data and the golden tolerances. Use an independent seed.
- **Seed-dependent magic threshold:** `test_outliers.py:104` asserts `result.sum() < 5` on
  seeded noise — robust for the fixed seed but brittle (and it tests the mock, not production —
  §7).
- **Loose, unexplained magnitude tolerances:** `test_rotate.py:112` (`< 1`) and `:136` (`< 2`)
  are large relative to pixel values; a regression shifting values by ~0.8 would pass.

## 16. Regression and documentation

- **`filterwarnings = ["error"]` is set** (`pyproject.toml:60`) — good; unexpected runtime
  warnings already fail the suite. Tests that legitimately divide (`test_gaussian_filter.py:105`,
  the outliers mock `:51`) correctly wrap in `np.errstate(...)` to avoid spurious elevation.
- **No `pytest.warns` usage** — acceptable, as no deprecation paths are apparent in the source.
- **Latent warning risk untested:** `fitting`/`stretch` divide by `(dof-1)` /
  `(wsum - w2sum/wsum)`; with degenerate small-`dof` inputs these could emit a `RuntimeWarning`
  that `filterwarnings=error` would convert to a failure. No test guards `dof <= 1`.
- Tests are well-commented with "why" notes (e.g. the NaN corner in `test_stdev.py:366`), which
  is excellent for regression context, though none reference issue IDs.

## 17. Other good practices

- **AAA structure is clean** and section banners are used consistently in most files.
- **Single-responsibility violations:** monolithic tests bundle independent scenarios into one
  function — `test_median.py:60` (two cases), `test_minimum.py:101`/`test_maximum.py:101` (four
  setups), `test_median.py:220` (two omit signs), `test_shift.py:144-173` &
  `test_ishift.py:298-336` (triple-nested loops). A failure reports only the function name.
- **Non-trivial logic embedded in tests:** the `for i in range(100): for j in range(100)`
  footprint-comparison loops carry error-prone slice arithmetic duplicated six times;
  `brief_fmt` (`test_shift.py:107`) and `_full_pixel_weights_ok` (`test_rotate.py:13`, an O(n²)
  loop) are untested helpers — if a helper is wrong, the conservation tests are silently
  weakened.
- `test_reshape.py:8` aliases `reshape as resize` ("re-use resize tests") — harmless but can
  confuse a reader grepping for `reshape`.

## 18. Code coverage

- **Target met, full-suite measurement:** `pytest tests/ --cov=src --cov-report=term-missing`
  reports **96.92 %** total with branch coverage on; 661 passed.
- **Every shipped module ≥ 90 %.** Modules below 100 % worth attention:
  - **`_validation.py` — 84 %** (lowest; uncovered lines 84-85, 101, 117, 130, 133, 137-138,
    147-148, 168, 172, 176-177, 186-192, 273-274, 278, 287 — almost entirely error-raising
    branches). Ties directly to §8/§12.
  - `reshape.py` — 90 % (line 67); `resample.py` — 94 % (144-167 branches); `rotate.py` — 96 %
    (169, 275, 310-312, 473); `variance.py`/`ishift.py`/`_utils.py` — 97 %.
- The 90 % minimum (`fail_under = 90`, `pyproject.toml:111`) is enforced. Note that the
  *total* clears 90 % comfortably, but the per-module floor is `_validation.py` at 84 %, so a
  per-module gate would currently fail — worth considering given it is the validation layer.

## 19. Pytest markers and registration

- `markers = []` (`pyproject.toml:56`) — no custom marks defined or used; with
  `--strict-markers` enabled this is safe (a typo'd mark would error).
- **`--strict-markers` and `--strict-config` are both in `addopts`** — good.
- No `xfail`/`skip`/`skipif` anywhere (confirmed) — so no stale-skip or non-strict-xfail debt.
- **No categorization marks** (`slow`/`integration`). The whole suite runs in ~13.5 s with no
  outlier-slow test, so this is currently unnecessary; revisit only if a slow test appears.

## 20. Test boundary (public API vs internals)

- **Importing internal modules is appropriate here** — `_utils`, `_validation`, `_filter` are
  internal-by-design and the project explicitly tests them; `conftest.py:10` importing
  `_use_shortcuts` is justified white-box toggling. `_resize_reference.py` is a documented,
  `_`-prefixed test-only reference impl (header `:4-8`).
- **But the public façade is under-tested.** fft/stretch/fitting tests import submodules
  (`from psiops.fft import ...`) and model tests import `psiops.imagemodel.*` directly, rather
  than the public `psiops` package whose surface is pinned by `__init__.pyi` (the type authority
  CLAUDE.md flags as needing manual sync). Consequently the re-export wiring and the
  `_suppress_divide_warnings` wrapper applied in `__init__.py:155-164` are never exercised — a
  divide-warning regression in a wrapped `transform` would not be caught.
- **Tight coupling to private internals** in a few places couples tests to implementation:
  `rotate`'s `_debug=` dict (`test_rotate.py:39,55`), `Stretch._antimask`/`_ij_powers`/
  `_matrix3d` (`test_stretch.py:198,427,433`), `Fitting._fill_stats`/`_result`/`_params`
  (`test_fitting.py:377-396`), `_retile` (`test_fft.py:311`). Acceptable for a numerics library
  but will break on benign refactors.

## 21. Logging assertions

- None of the source modules use the `logging` module; failures surface as exceptions or
  (suppressed) warnings. No `caplog` usage exists and none is needed — correct. The strict
  `filterwarnings=error` config means any *leaked* warning already fails its test, which
  implicitly verifies the in-code `warnings.catch_warnings()` suppressions work.

## 22. Pytest configuration

- **Active config file:** `pyproject.toml` `[tool.pytest.ini_options]`. No `pytest.toml`,
  `.pytest.toml`, `pytest.ini`, `.pytest.ini`, `tox.ini[pytest]`, or `setup.cfg[tool:pytest]`
  exists, so by the fixed precedence order `pyproject.toml` is the single applied file — **no
  ignored/duplicate config.** Good.
- **`testpaths = ["tests"]`** set (avoids whole-repo collection); `pythonpath = ["src"]` set.
  `python_files`/`python_classes`/`python_functions` use defaults (standard naming) — fine.
- **`addopts = ["-n","auto","--cov=src","--strict-markers","--strict-config"]`** — strong
  defaults; `--strict-markers`/`--strict-config` and `filterwarnings=error` already cover the
  checklist's `-W error` suggestion.
- **Plugins:** `pytest-xdist` (parallelism) and `pytest-cov` are declared and used.
  **`pytest-randomly` is not installed** — the suite's order-independence (which it appears to
  have, via the autouse restore and per-test seeding) is therefore never actively stress-tested.
  Adding it would catch any future hidden order dependence (e.g. the unrestored `_usable_bytes`
  in §6).

## 23. Snapshot and golden-file testing

The `ANSWERS` regression tables in `test_shift.py:14-105` and `test_ishift.py:14-225` are the
suite's only golden data.

- **Committed:** yes, as inline dicts keyed by opaque `(m,o,k,c)` index tuples.
- **Reviewability:** poor — dense numeric/`np.nan` blobs where a reviewer cannot judge whether a
  changed value is correct. The `brief_fmt` helper (`test_shift.py:107`) keeping `1/3`/`2/3`
  exact is a small mitigation.
- **Stale detection:** only exact equality (`np.array_equal(..., equal_nan=True)`,
  `test_shift.py:344`, `test_ishift.py:525`). If the generator *inputs* (offset/mask loops)
  change, the keys silently shift with no flag.
- **Regeneration is a manual source edit** via a `PRINT_ANSWERS = False` toggle
  (`test_shift.py:12`, `test_ishift.py:12`) — not a snapshot fixture, so there is no
  `--snapshot-update` audit trail.
- **Evidence of drift / hand-editing:** `test_ishift.py:57-98` is a large commented-out block of
  old `(1,*)` answers immediately followed by live replacements (`:99-140`), plus a commented
  duplicate driver (`:531-556`) and inline "bug fix: was smask[...,:0,:] (empty, always True)"
  notes (`:374-375`). This strongly indicates the golden data was hand-patched rather than
  cleanly regenerated. Consider migrating to a real snapshot tool (e.g. `syrupy`) or at minimum
  removing the dead commented blocks and documenting the regeneration procedure.

---

## Prompt for an AI agent to fix tests

> **Task:** Improve the `psiops` pytest suite per the critique below. **Do not change any
> production code under `src/`** — only add or modify files under `tests/` (including
> `tests/conftest.py`). Preserve all currently-passing behavior; only add tests, add/strengthen
> assertions, extract fixtures, and restructure test code. After each change, run
> `scripts/run-all-checks.sh --pytest` and keep the suite green. Match the project style:
> 90-char lines, single quotes, type hints on all params/returns, Google-style docstrings, the
> existing section-banner layout.
>
> **Context (full report above).** Current state: 661 tests, 96.92 % coverage, config already
> has `testpaths`, `--strict-markers`, `--strict-config`, `filterwarnings=["error"]`.
>
> **High-priority:**
> 1. **Validation layer (§8/§12/§18):** Add direct tests for `_validation._check_image`
>    covering every raise path (wrong-type/non-numeric → `TypeError`; complex-without-comps,
>    `<2-D` image/mask/weights, mask & weights shape mismatch, bad `returns` → `ValueError`),
>    each asserting message **contents** via `pytest.raises(...) as exc_info; assert '<substr>'
>    in str(exc_info.value)`. Bring `_validation.py` to ≥ 90 % (currently 84 %, missing lines
>    listed in §18). Also add direct tests for the untested `_utils` helpers `_merge_weights`,
>    `_pixel_area`, `_flatten_axes`, and the `_ImageInfo.__post_init__` ValueError.
> 2. **Exception messages (§12):** Add `match=` (or `str(exc_info.value)`) assertions to the
>    type-only `pytest.raises` calls in `test_zoom.py`, `test_resize.py`, `test_rotate.py`,
>    `test_gaussian_filter.py`, `test_stretch.py`, `test_reshape.py`, `test_camouflage.py`,
>    `test_imagemodel.py`, so distinct errors at the same call site are distinguishable.
> 3. **`shortcuts` cross-checking (§4/§6):** Apply the `shortcuts` fixture (or explicit
>    shortcut-vs-general equality cross-checks on identical inputs) to the reduction value-match
>    and masked tests across mean/median/variance, **all of `stdev`**, and the
>    fft/gaussian/fitting/stretch group. Verify both branches of every `_use_shortcuts()` gate.
> 4. **Global-state isolation (§6/§14):** Add an autouse fixture in `conftest.py` that saves and
>    restores `_filter._usable_bytes()` (mirroring `_restore_shortcuts`). Then remove the
>    hand-rolled `_use_shortcuts` toggling in `test_shift.py`/`test_resample.py` in favor of the
>    `shortcuts` fixture, and remove the now-redundant `try/finally` blocks.
> 5. **`test_outliers.py` mock (§7):** Replace the hand-written `_masked_gaussian_filter`
>    stand-in with the real `gaussian_filter`, or — if the masked mode is genuinely broken —
>    `@pytest.mark.xfail(strict=True, reason='<issue ref>')` the affected cases so the breakage
>    is tracked rather than masked, and strengthen the shape-only asserts to value asserts
>    (e.g. `weights=ones` equals the no-weights result).
>
> **Nice-to-have:**
> 6. Extract shared image builders (`image0`, `_make_image`/`_ramp_stack`, the
>    irregular-footprint brute-force comparison, `zero_size`/`footprint_forms` blocks) into
>    `conftest.py` fixtures/helpers; reconcile variance's `range(0,100,7)` sub-sampling.
> 7. Convert loop-based cases to `@pytest.mark.parametrize`: dtype loops
>    (median/minimum/maximum), radius loops (`test_circle.py`), size loop (`test_camouflage.py`),
>    `ishift` per-mode blocks, and the mean/stdev `factors` families. Split monolithic
>    nested-loop tests (`test_shift.py`, `test_ishift.py`, median/min/max) into
>    single-responsibility functions.
> 8. Add missing-behavior tests: `nans=True` for median/stdev/min/max and the transforms;
>    non-numeric `TypeError` for mean/median/stdev/variance; `returns=` arity for the reductions;
>    model-constructor error paths (`SummedModel`/`SmearedModel`/`ArrayModel`); `fitting`
>    `LinAlgError`/non-convergence and `remask`-before-target; `rotate` `ZeroDivisionError`;
>    `gaussian_filter` `cval=None`. Add no-mutation tests feeding non-copy ndarrays.
> 9. Add public-API smoke tests that import from `psiops` directly (exercising the re-export
>    wiring and the `_suppress_divide_warnings` wrapper) rather than only from submodules.
> 10. Clean up the `ANSWERS` golden tables: remove the dead commented blocks in `test_ishift.py`
>     (`:57-98`, `:531-556`), document the `PRINT_ANSWERS` regeneration procedure, and consider
>     `syrupy`. Replace `test_rotate.py`'s throwaway 300-iteration RNG-advance with an
>     independent seed, and justify or tighten the `< 1`/`< 2` tolerances at
>     `test_rotate.py:112,136`.
>
> **Coverage:** Run coverage on the **entire** suite
> (`pytest tests/ --cov=src --cov-report=term-missing`) and keep total ≥ 90 % with every module
> ≥ 90 % (close the `_validation.py` 84 % gap). Do not modify `src/`.
