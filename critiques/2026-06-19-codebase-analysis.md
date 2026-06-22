# Codebase analysis: rms-psiops

*Generated: 2026-06-19 — Updated: 2026-06-22*

## Summary

The core algorithms are sophisticated and well-designed — the photometric-accuracy guarantees,
memory-tiling logic, and mask/weight propagation are clearly the result of careful domain
expertise. Several remediation passes have resolved all critical and high-severity bugs, added
CI, converted `_ImageInfo` to a typed dataclass, tightened the mypy config, and cleaned up
packaging.

A test-suite-and-documentation pass (2026-06-20) then **modernized and split the entire test
suite into 629 focused pytest functions, raised total coverage to ~97% (every module ≥ 89%,
most at 100%), wrote tests for all previously-untested modules, and added a Sphinx User's
Guide**. That work uncovered and fixed roughly a dozen further latent bugs (NumPy 2.0
breakage, a masked-filter axis-reduction bug that silently broke every masked `*_filter`,
weighted-filter shortcuts ignoring weights, `keepdims` dropped on the bare-image return, the
`resample` unit-zoom shortcut, and others — see §11).

A lint-and-type cleanup pass (2026-06-21) then **cleared the entire lint/type backlog: `ruff
check src tests` is fully green, `mypy src` is clean across all 30 source files, and a hand-
maintained `__init__.pyi` type stub now pins the public API signatures** (see §12).

A follow-up pass (2026-06-22) **completed and enforced test type annotations, fixed the
packaging metadata (Pyroma now rates 10/10), and made `scripts/run-all-checks.sh` fully green**
end to end. The same pass also added 2-D support to `resample`/`reshape`, simplified
`ArrayModel.transform`, fixed a `resample` shrink-path crash, and expanded the imagemodel tests
(see §13). A further pass **narrowed the divide-by-zero warning suppression to the NumPy
error-state source, centralized zero-size-input rejection in `_check_image`, and exported and
documented the `Fitting` class** (see §14), and documented the thread-safety contract (§5).
With that, every actionable finding has been addressed; the only item left open is the
`returns` DSL discoverability, a deliberate design choice marked won't-fix.

---

## 1. Structure and layout

- ~~**Finding (medium)**: `_utils.py` is 893 lines and serves as a "kitchen sink" utilities
  module combining input validation, filter dispatch, memory management, and test-control
  flags. **Evidence**: `_utils.py`. **Suggestion**: Consider splitting into `_validation.py`
  (argument checks), `_filter.py` (filter dispatch / tiling logic), and keeping `_utils.py`
  for the shared `_ImageInfo` data class.~~ ✓ **Fixed.** Split into three modules: `_utils.py`
  (shared `_ImageInfo` dataclass + 5 small helpers), `_validation.py` (`_check_image`,
  `_check_return`), and `_filter.py` (filter dispatch, tiling, memory management, test-control
  flags). All 13 source modules and 4 test files updated accordingly.

- ~~**Finding (low — partially addressed)**: Several modules are commented out in
  `__init__.py` (`fft`, `fitting`, `stretch`, `imagemodel`, `scaling`).~~ ✓ **Fixed**
  (2026-06-22). `fft`, `stretch`, and `imagemodel` were exported earlier; `Fitting` is now
  un-commented and added to `__all__` and the `__init__.pyi` stub, and the dead reference to
  the non-existent `scaling` module was deleted. No commented-out import block remains (§14).

- ~~**Finding (medium)**: `tests/resize.py` and `tests/unittester.py` are legacy test
  infrastructure not discovered by pytest.~~ ✓ **Partially fixed.** `tests/unittester.py` has
  been removed (pytest supersedes it). `tests/resize.py` is retained as a deliberate helper:
  it provides a reference `resize()` implementation that `test_resample.py` cross-checks
  against, and it carries no `test_` prefix so pytest does not collect it as a test. The
  remaining nicety would be to rename it (e.g. `_resize_reference.py`) to make the
  not-a-test-module intent unmistakable.

- ~~**Finding (low)**: File-header comments in several files refer to the old package path
  `image_ops/` (e.g. `# image_ops/tests/test_mean.py`).~~ ✓ **Fixed.** All 21 stale headers
  updated to `# psiops/<path>` or `# tests/<path>`; two files also had the wrong filename
  (`test_reshape.py` said `test_resize.py`, `test_stdev.py` said `test_variance.py`) — both
  corrected.

---

## 2. Best practices alignment

- ~~**Finding (high — likely bug)**: `_ImageInfo` has **no `__init__`** — it is a bare class
  with attributes monkey-patched on after construction. This makes it impossible to
  type-check, easy to miss attributes, and violates the project rule to use
  dataclasses/TypedDict for multi-value return objects. **Evidence**: `_utils.py:12–33`;
  attributes like `info.dtype` are set in `maximum.py:157` and `minimum.py:157` but are
  never declared.~~ ✓ **Fixed.** Converted to a `@dataclass` with explicit typed fields;
  construction in `_check_image` updated to use the constructor directly.

- ~~**Finding (high — bug)**: `temp_mask` and `temp_weights` in `_check_image` are only
  defined inside `if mask.shape != weights.shape:`, but referenced unconditionally below. If
  shapes already match, this raises `NameError`. Additionally, `mask = mask | weights == 0.`
  lacked explicit parentheses and the updated mask was never synced back to weights.
  **Evidence**: `_utils.py:355–372`.~~ ✓ **Fixed.** Inner references moved inside the same
  `if` block; explicit parentheses added.

- ~~**Finding (critical — bug)**: `mask[mask == maskval] = True` at `_utils.py:327` compares
  the **mask** to `maskval`, not the image. Mask is boolean, maskval is a float, so this is
  always False — the `maskval` feature silently did nothing. **Evidence**:
  `_utils.py:327`.~~ ✓ **Fixed.** Changed to `mask[image == maskval] = True`.

- ~~**Finding (high — bug)**: In `rotate.py:354–355`, two lines use `==` (comparison) where
  `=` (assignment) was intended. The inside-point deduplication logic silently did nothing.
  **Evidence**: `rotate.py:354–355`.~~ ✓ **Fixed.**

- ~~**Finding (medium)**: `import` order is inconsistent. `maximum.py` and `minimum.py` place
  the local `from psiops._utils import ...` before standard library imports, violating the
  project's three-group import rule. **Evidence**: `maximum.py:5–10`.~~ ✓ **Fixed.**
  (`minimum.py` was already correct; only `maximum.py` needed reordering.)

- ~~**Finding (low)**: Module-level constants `HALFPI`, `TWOPI`, `TESTING` in `rotate.py` are
  ALL_CAPS (correct), but `TESTING` doubles as a behavior flag mutated at runtime via
  `_set_rotate_testing_flag()`. The rule says avoid mutable globals; this pattern also makes
  tests non-parallel-safe. **Evidence**: `rotate.py:11–15, 22`.~~ ✓ **Fixed.** Removed
  `TESTING` and `_set_rotate_testing_flag()` entirely. Replaced with an optional `_debug: dict
  | None = None` keyword parameter on `rotate()`: callers pass an empty dict and read
  `area_list`, `imod_list`, `jmod_list`, `new_center`, `new_mask`, and `new_weights` from it
  after the call. Thread-safe by construction (each call gets its own dict) and the return type
  is unchanged. Also fixed four pre-existing bugs uncovered during this work: `three=True` in
  the `_check_image` call (rejected valid 2D images); `sum1`/`sum0` initialized with
  `image.dtype` causing a cast error for integer and float32 inputs (fixed to `float64`);
  `sum1 / new_weights` issuing a RuntimeWarning on masked pixels (fixed with `np.divide(...,
  where=~new_mask)`); `unused_below`/`unused_above` omitting `new_center` from their
  computation, shifting the output center by ~7 pixels instead of leaving it at the natural
  center.

- ~~**Finding (low)**: The `assert` at `rotate.py:461` will be silently skipped with
  `python -O`. For an internal invariant in library code, prefer `raise RuntimeError(...)`.
  **Evidence**: `rotate.py:461`.~~ ✓ **Fixed.** Replaced with `if not ...: raise RuntimeError(...)`.

---

## 3. Types and static checks

- ~~**Finding (high)**: `_check_return` accepts `pixel_area: int = 1` but never uses it.
  Instead, the undefined name `pixels_on_axes` is referenced at lines 453 and 456–457. This
  is a `NameError` whenever `'w' in info.returns and weights is None and mask is not None`.
  **Evidence**: `_utils.py:449–457`.~~ ✓ **Fixed.** All three references to `pixels_on_axes`
  replaced with `info.pixel_area`.

- ~~**Finding (medium)**: `_check_return` return type annotation is `np.ndarray | list`
  instead of `np.ndarray | list[np.ndarray]`. Similarly, all public functions return
  `np.ndarray | list[np.ndarray]` but list element types aren't specified. **Evidence**:
  `_utils.py:408`.~~ ✓ **Fixed.** Updated to `np.ndarray | list[np.ndarray]`. (All public
  modules already had the correct annotation; only `_check_return` itself was missing the
  type parameter.)

- ~~**Finding (medium)**: `mypy` is configured with `strict = false` and blanket
  `ignore_missing_imports = true` for all `psiops.*` modules, which masks type errors across
  the entire package. **Evidence**: `pyproject.toml:117–131`.~~ ✓ **Fixed.** Added
  `disallow_untyped_defs = true` and `check_untyped_defs = true` globally; removed `psiops.*`
  from `ignore_missing_imports` (first-party code analyzed from source needs no stub
  suppression); added a `tests.*` override relaxing both flags since test annotation is a
  separate open finding. Also annotated the one production function that was missing a return
  type (the `keepdims` nested helper in `_check_return`).

- ~~**Finding (low)**: No `__all__` is declared in `__init__.py` or any module.~~ ✓ **Fixed**
  (see §10).

- ~~**Finding (medium)**: Public function signatures used `npt.ArrayLike` for image
  parameters and then dereferenced `.ndim`/`.shape`, producing a few hundred `mypy` errors
  across `src`.~~ ✓ **Fixed** (2026-06-21). Image parameters are now typed as `np.ndarray`,
  and a hand-maintained `src/psiops/__init__.pyi` stub pins the full public API signatures as
  the type authority for the package. `mypy src` is clean across all 30 source files (see §12).

---

## 4. Testing

- ~~**Finding (high)**: All tests are written as `unittest.TestCase` classes with a single
  `runTest()` method (one giant test per class).~~ ✓ **Fixed.** The whole suite is now plain
  pytest. Every monolithic test was split into focused, independently-runnable `test_*`
  functions (e.g. `test_ishift` → 12, `test_shift` → 6, the stack/transform clusters into
  ~10–30 each), using `pytest.raises` for error cases and a `shortcuts` fixture (in
  `tests/conftest.py`) to sweep both the optimized and general code paths. 629 tests total,
  runnable individually and in parallel.

- ~~**Finding (high)**: Tests call `warnings.simplefilter('ignore', category=RuntimeWarning)`
  in `setUp()`, directly contradicting `filterwarnings = ["error"]`.~~ ✓ **Fixed.** The
  blanket suppression is gone; `filterwarnings = ["error"]` is honored. Warnings that are a
  genuine, expected consequence of the computation (0/0 on fully-masked pixels, all-NaN
  slices, `ddof` underflow) are now suppressed **at the source** in the library via
  `np.errstate` / `warnings.catch_warnings`, so the tests still catch *unexpected* warnings.

- ~~**Finding (high)**: Large coverage gaps: `camouflage`, `stretch`, `fft`, `fitting`,
  `outliers`, `circle`, `gaussian_filter`, and all `imagemodel/` subclasses have no test
  files.~~ ✓ **Fixed.** New test suites were written for every one of these modules. Total
  coverage is now ~97%; per-module: `circle`, `stretch`, `camouflage`, `fitting`, `mean`,
  `stdev`, `reshape`, `zoom`, `unzoom`, `maximum`, `minimum`, `median`, `_filter`, and all
  `imagemodel/*` at 100%; `gaussian_filter`/`shift` 99%; `fft`/`variance`/`ishift`/`_utils`
  97%; `rotate` 96%; `resample` 93%; `outliers` 100%; `_validation` ~86%. The 90% gate passes.

- ~~**Finding (medium)**: `np.random.seed(5965)` is called as a global side effect; with
  `pytest-xdist`, random-seed state bleeds across tests.~~ ✓ **Fixed.** All tests use
  `np.random.default_rng(seed)` scoped to each function. `tests/conftest.py` also adds an
  autouse fixture that restores the global `_use_shortcuts` flag after every test, so the
  split tests are order- and parallel-independent.

- ~~**Finding (low — largely addressed)**: Tests have no type annotations on functions;
  ~60 functions in the newer suites still omit `-> None`.~~ ✓ **Fixed** (2026-06-22). Every
  test function and helper now carries full parameter and return annotations (including
  fixture and parametrized-test parameters), and the `tests.*` mypy override that relaxed
  `disallow_untyped_defs`/`check_untyped_defs` was removed, so the test suite is now held to
  the same strict config as `src`. `mypy src tests` is clean across all 56 files (see §13).

- ✓ **(superseded)** The `tests/resize.py` discovery concern is now covered under §1: the
  file is an intentional, uncollected reference helper; `unittester.py` (which imported it) has
  been removed.

---

## 5. Performance and resource use

- ~~**Finding (medium)**: `psutil.virtual_memory().total` is called **at module import time**,
  setting `_MAX_USABLE_BYTES` once and never re-evaluating it. On long-running processes or
  containers with dynamic memory, this stale value could be misleading. **Evidence**:
  `_utils.py` (module-level).~~ ✓ **Fixed.** Removed `_MAX_USABLE_BYTES` entirely. `_USABLE_BYTES`
  is now `int | None` with `None` as a "use the live default" sentinel. `_usable_bytes()` calls
  `psutil.virtual_memory().total // 2` lazily on first read, on explicit limit-setting, and on
  reset — so each call reflects current system memory. No psutil call at import time.

- ~~**Finding (medium — thread safety)**: Module-level mutable state (`_USE_SHORTCUTS`,
  `_LAYERS_USED`, `_TILES_USED`, `_USABLE_BYTES`, in `_filter.py`) is modified via `global`
  setters with no locking.~~ ✓ **Documented** (2026-06-22). On review, the *results* are
  already thread-safe: this state is read-only during normal computation, Python global reads
  are atomic under the GIL, and `np.errstate` (the divide-by-zero decorator) is thread-local.
  The flagged globals are process-global test/tuning/diagnostic controls; the package docstring
  and a new User's Guide "Thread safety" section now document that operations may be called
  concurrently on independent arrays but those globals are not to be mutated from several
  threads at once. The one residual write during normal operation — `rotate()` briefly toggling
  `_use_shortcuts` to force `resample`'s general path — affects only which (equally correct)
  code path runs, not results; threading an explicit per-call override into `resample` to
  remove it remains an optional follow-up.

- ~~**Finding (low)**: The nested 3×3 loop in `rotate()` accumulates `area_list`,
  `imod_list`, `jmod_list` into Python lists in the hot path, only to use them in testing
  mode. **Evidence**: `rotate.py:223–258`. **Suggestion**: Guard the accumulation with
  `if TESTING:` inside the loop, or extract the testing instrumentation.~~ ✓ **Fixed.**
  List initializations and all three `append` calls moved inside `if TESTING:` guards.

---

## 6. Maintainability and extensibility

- ~~**Finding (high)**: The `returns` string mini-DSL (`'i'`, `'im'`, `'iw'`, `'imw'`, plus
  optional trailing char) is used as the central mechanism for controlling what every public
  function returns. It appears in every public function signature but is effectively
  undiscoverable without reading `_utils.py`. Consider replacing with a typed `enum` or
  explicit keyword arguments (e.g. `return_mask=False, return_weights=False`). **Evidence**:
  Every public function.~~ ✗ **Won't fix.** The `returns` DSL is a deliberate design choice
  and will be retained as-is.

- ~~**Finding (medium)**: `fft.py` contains `from gaussian_filter import gaussian_filter`
  (bare module name, missing the `psiops.` package prefix). This will fail on import. The
  module is commented out of `__init__.py`, so it's dormant — but it must be fixed before it
  can be enabled. **Evidence**: `fft.py:9`.~~ ✓ **Fixed.** Changed to
  `from psiops.gaussian_filter import gaussian_filter`.

- ~~**Finding (low)**: No `docs/` directory exists despite the `pyproject.toml` pointing to a
  ReadTheDocs URL.~~ ✓ **Fixed.** A Sphinx `docs/` tree exists (`conf.py`, `index.rst`,
  `module.rst`, Makefile) and now includes a tutorial-style **User's Guide**
  (`docs/userguide.rst`) covering image stacks, the pixel-coordinate convention, photometric
  accuracy, masking, the `returns` parameter, and every public operation family with worked
  examples. The build was also repaired (the `REPONAME`/`PSIops` placeholders, a too-short
  title underline, the missing README `start-after` marker, and orphaned pages were fixed) so
  `make -C docs html SPHINXOPTS=-W` now succeeds with zero warnings.

---

## 7. Security and robustness

- **Finding (low)**: No subprocess, `eval`, credentials, or path traversal issues found.
  Security posture is clean.

- ~~**Finding (low — partially addressed)**: `RuntimeWarning` is globally suppressed for all
  users at import time via `warnings.simplefilter('ignore', ...)` and
  `os.environ['PYTHONWARNINGS']`.~~ ✓ **Fixed** (2026-06-22). The blanket import-time
  suppression and the `PYTHONWARNINGS` env var (which leaked into subprocesses) were removed.
  A decorator now wraps every public entry point in
  `np.errstate(divide='ignore', invalid='ignore')`, suppressing only the library's intentional
  divide-by-zero/invalid results at the NumPy error-state source — which holds even when
  warnings are errors — so all other `RuntimeWarning`s and the user's own code now surface
  (§14).

---

## 8. Dependencies and tooling

- ~~**Finding (critical)**: `psutil` is imported and used in `_utils.py` but is **not listed
  in `pyproject.toml` dependencies**. Any user install will get an `ImportError` on the
  first `import psiops`. **Evidence**: `_utils.py:10`, `pyproject.toml:11–14`.~~ ✓ **Fixed.**
  Added `"psutil >= 5.9"` to `[project.dependencies]`.

- ~~**Finding (medium)**: No CI/CD pipeline exists (no `.github/workflows/`). The codecov
  config (`codecov.yml`) exists but won't be triggered without CI.~~ ✓ **Fixed.** Added
  `.github/workflows/ci.yml` running ruff, mypy, pytest, and pyroma across Python 3.11–3.13,
  with Codecov upload on 3.11.

- ~~**Finding (low)**: `pyproject.toml` description is `"TODO"`, and a license-config conflict
  made Pyroma fail to build the package metadata at all (0/10).~~ ✓ **Fixed** (2026-06-22).
  Filled in a real `description`; removed the redundant `License :: OSI Approved :: Apache
  Software License` classifier (PEP 639 forbids pairing a license expression with a license
  classifier — the actual build error); and pinned `setuptools>=77` in `[build-system]` so the
  SPDX `license = "Apache-2.0"` string is accepted. Pyroma now rates **10/10**. (The
  commented-out `[project.scripts]` placeholder remains, harmless until a console entry point
  is actually needed.)

- ~~**Finding (low)**: `dev` optional dependencies include `"psiops"` itself, which is
  redundant when installing with `pip install -e ".[dev]"`. **Evidence**:
  `pyproject.toml:78`.~~ ✓ **Fixed.**

---

## 9. Technical debt and risk

- ~~**Finding (high)**: Docstring typos: `_utils.py` says "used used instead of or in
  addition" and "muse be at least 3-D"; same "used used" in `rotate.py`.~~ ✓ **Fixed.**

- ~~**Finding (medium)**: `variance.py` header comment says `# psiops/stdev.py` — stale
  copy-paste from when the file was split.~~ ✓ **Fixed.**

- ~~**Finding (medium)**: `_ImageInfo` attribute `dtype` is set in `_maximum()` and
  `_minimum()` but `_check_return` never reads it — it uses the `return_dtype` parameter
  instead, which those callers never pass. The return dtype fix is silently skipped.
  **Evidence**: `maximum.py:157`, `_utils.py:469`.~~ ✓ **Fixed.** `_check_return` now falls
  back to `info.dtype` when `return_dtype` is not passed explicitly.

---

## 10. Packaging and distribution

- ~~**Finding (critical)**: `psutil` missing from `dependencies`.~~ ✓ **Fixed** (see §8).

- ~~**Finding (medium)**: `__all__` is absent from `__init__.py` and all modules. Without it,
  `from psiops import *` imports everything, and the declared public API surface is unclear.
  **Evidence**: No `__all__` anywhere.~~ ✓ **Fixed.** Added `__all__` to `__init__.py`
  listing all 21 currently active public symbols, grouped by category. Commented-out modules
  (`imagemodel`, `fft`, etc.) are correctly excluded until they are ready to enable.

- ~~**Finding (low)**: License uses the old `license = {text = "Apache-2.0"}` form rather
  than PEP 639 `license = "Apache-2.0"`.~~ ✓ **Fixed.**

- **Finding (low)**: `py.typed` is listed in `[tool.setuptools.package-data]` which is
  correct. ✓

---

## Additional bugs found during remediation

These were not in the original critique but were discovered while applying fixes.

- **`median.py:141` — `info.area_factor` NameError** (found during `_ImageInfo` dataclass
  conversion): `info.area_factor` was referenced but never defined on `_ImageInfo`. Any
  masked median call would raise `AttributeError`. Fixed to `info.pixel_area`. ✓

- **`_utils.py` — `returns += info.extra_char` before `info` is constructed** (found during
  dataclass conversion): In `_check_image`, line 382 referenced `info.extra_char` but `info`
  is not constructed until later. Would raise `NameError` for callers that pass
  `extra_by_default=True` with a non-empty `extra_char` (i.e., `rotate()` with no `center`
  argument). Fixed to `returns += extra_char`. ✓

- **`_utils.py` — `info.extra_char in info.returns` always `True` for empty string** (found
  during dataclass conversion): Both the validation guard and the results-building check in
  `_check_return` used `info.extra_char in info.returns` without first checking that
  `extra_char` is non-empty. In Python, `'' in any_string` is always `True`, so the check
  fired unconditionally — appending a spurious `None` to every multi-value return. Fixed both
  occurrences to `info.extra_char and info.extra_char in info.returns`. ✓

- **`test_reshape.py` and `test_stdev.py` — wrong filename in header comment** (found during
  stale-header sweep): `test_reshape.py` declared `# image_ops/tests/test_resize.py` and
  `test_stdev.py` declared `# image_ops/tests/test_variance.py`. Both corrected to match
  their actual filenames. ✓

- **`_ImageInfo` — `returns` field had no invariant check** (hardened alongside psutil fix):
  `_ImageInfo` is a plain dataclass, so `returns` could be set to any string at construction
  without validation. Added `__post_init__` which rebuilds the valid-options set from
  `extra_char` and raises `ValueError` with a descriptive message (including the list of
  valid values) if `returns` is not in that set. The existing up-front validation in
  `_check_image` remains as the user-facing guard; `__post_init__` is the invariant that
  holds for any internal construction. ✓

---

## 11. Bugs found and fixed during the test-and-docs pass (2026-06-20)

Writing the test suite and driving coverage to ~97% exercised many code paths for the first
time, surfacing these latent bugs. All are fixed with regression tests unless noted.

- **NumPy 2.0 breakage**: `median.py`/`variance.py` used `np.NaN` (removed in NumPy 2.0); the
  whole package failed under modern NumPy. Fixed to `np.nan`. ✓
- **Masked filters completely broken** (`_filter._apply_op`): after moving the footprint axis
  to `-3`, the op reduced over `axis=-1` (a spatial axis) instead of `-3`. Every masked
  `maximum/minimum/mean/median/variance/stdev _filter` returned wrong-shaped/garbage results.
  Fixed to reduce over `axis=-3`. ✓
- **Weighted filters ignored weight values** (`_mean`/`_variance`/`_median`): the unweighted
  shortcut fired whenever `mask is None`, even with `weights` present (the filter path passes
  `mask=None` with weights), so weighted `*_filter` silently returned the unweighted result.
  Fixed to require both `mask` and `weights` to be None (matching `maximum`/`minimum`). ✓
- **`keepdims` dropped on the bare-image return** (`_validation._check_return`): the
  `returns == 'i'` early return bypassed the `keepdims()` helper, so `keepdims=True` was
  silently ignored unless a mask/weights forced the multi-return path. Fixed. ✓
- **`resample` unit-zoom shortcut crashed**: the `zoom_==(1,1)` fast path referenced
  `new_weights` before assignment (no-weights case) and called `_check_return` without the
  positional `mask` arg. Fixed. ✓
- **`resample` integer-image / weighted accumulation**: integer images hit an in-place
  float→int cast error; the `weights`/`axis_weights` variable mix-up broke roundoff
  suppression; `new_mask` was undefined on the weighted path. Fixed (float accumulation
  buffer, correct variable, `new_mask=None`). ✓
- **`variance` `vartype='biased'` crash**: the weighted path never assigned `denom`. Fixed,
  and under-populated pixels are now masked consistently (`denom<=0`, count<2 for
  reliability/unbiased, count==0 for biased). ✓
- **`ishift` with both mask and weights**: left `new_mask`/`new_weights` unbound
  (`UnboundLocalError`). Fixed by prioritizing the weights branch (weights encode the mask). ✓
- **`unzoom`/`shift` integer division & dtype**: in-place int division crashes and integer
  results from fractional shifts; fixed to use float output dtypes. ✓
- **Stack-op 3-D enforcement**: `maximum`/`median` lacked `three=True`, so 2-D inputs did not
  raise like the other reductions. Fixed. ✓
- **Multi-dimensional `factors`** (`_merge_weights`): factors were reshaped as
  `[:, newaxis, newaxis]` (insert after axis 0) instead of appending trailing spatial axes,
  breaking multi-axis `factors`. Fixed to `[..., newaxis, newaxis]`. ✓
- **`gaussian_filter` masked path** (entirely broken): forwarded the literal `'masked'` mode
  to SciPy, rounded boolean weights, ignored `returns=`, and propagated NaNs. Fixed. ✓
- **`median` `omit`**, **`maximum`/`minimum` weights-only & integer-overflow fill**, **`fft`
  N-D iteration / `ifft real=`**, **`fitting`/`stretch`/`camouflage`/`outliers`/`imagemodel`
  import and attribute bugs**: all fixed by the per-module test agents. ✓
- `_utils._check_tuple` now renders NumPy scalars as plain Python scalars in error messages
  (NumPy 2.0 `repr` change). ✓

---

## 12. Lint and type cleanup pass (2026-06-21)

A focused pass cleared the lint/type backlog from item #7 and added a public-API type stub.

- **Lint backlog cleared**: `ruff check src tests` now passes with zero findings. The ~184
  prior findings (`E501` line-length, `I001` import order, `E701` one-line
  `with pytest.raises(...):`, `E712` `== True/False`, `RUF010`, `B904`, `B007`, `RUF059`)
  were resolved across `src/` and `tests/`; `RUF005` was disabled globally as a deliberate
  style choice. ✓
- **Source mypy clean**: image parameters were retyped from `npt.ArrayLike` to `np.ndarray`,
  eliminating the bulk of the `.ndim`/`.shape`-deref errors. `MYPYPATH=src mypy src` now
  reports success across all 30 source files. ✓
- **Public API type stub added** (`src/psiops/__init__.pyi`): a hand-maintained stub mirrors
  every exported signature (transforms, stack ops, filters, `ImageModel` family, `Stretch`,
  and the FFT/correlation ops) and is the type authority for the package. It is committed and
  tracked, and is ruff- and mypy-clean. Because it is not auto-validated against the
  implementation, its signatures must be kept in sync when public functions change (per
  `CLAUDE.md`). ✓
- **Signature/docstring formatting** (`src/psiops/*.py`): `def` signatures were rewrapped to
  the 90-column limit (no trailing comma after the last parameter), and `, optional` was
  added to the docstring type of every defaulted parameter. Ruff-clean; tests unaffected.
- **Remaining**: ~33 `mypy` errors confined to `tests/` (see #7) and the `-> None` test
  annotations (#8). ✓ **Both since resolved on 2026-06-22 — see §13.**

---

## 13. Test-annotation enforcement, packaging, and 2-D resampling (2026-06-22)

This pass brought `scripts/run-all-checks.sh` to fully green and added a user-facing capability.

- **Test annotations completed and enforced.** Every test function, helper, and fixture now
  carries full parameter and return annotations (fixture parameters like `shortcuts: bool`,
  parametrized parameters, and the `resize()` reference helper via `@overload`). The residual
  ~33 test-only `mypy` errors were fixed (e.g. `pytest.ExceptionInfo[Exception]` declarations
  for reused `exc_info`, `np.ndarray` annotations for reassigned arrays, `assert x is not None`
  narrowing). The `tests.*` mypy override relaxing `disallow_untyped_defs`/`check_untyped_defs`
  was removed, so tests are now held to the same strict config as `src`; `mypy src tests` is
  clean across all 56 files. ✓
- **Packaging metadata fixed → Pyroma 10/10.** Real `description`; removed the redundant Apache
  license classifier (PEP 639 forbids pairing a license expression with a classifier); and
  `setuptools>=77` pinned so the SPDX `license = "Apache-2.0"` string is accepted (§8).
  `scripts/run-all-checks.sh` now passes end to end (ruff, mypy, pytest, Sphinx `-W`, Pyroma,
  PyMarkdown). ✓
- **`resample`/`reshape` accept 2-D images.** The `three=True` requirement was removed from
  `resample` (and `reshape`, which delegates to it), so a plain 2-D image is now valid; output
  is identical to the previous single-layer 3-D path. `ArrayModel.transform` was simplified to
  drop its temporary leading-axis wrap/unwrap. The User's Guide was updated to match. ✓
- **`resample` shrink-path crash fixed.** Shrinking a small source onto a much larger output
  grid raised a `ValueError` because the index-uniqueness reassignment ran out of spare source
  indices; out-of-range, zero-weighted read indices are now clamped instead. A faster,
  size-agnostic rewrite of the accumulation core (separable sparse weight-matrix matmul, ~5–11×
  faster in benchmarks) is tracked separately as GitHub issue #1. ✓
- **Imagemodel tests expanded.** New `test_{gaussian,arraymodel,smearedmodel,summedmodel}_`
  `weighted_center` verify both the weighted centroid and integral conservation to tight
  tolerances (1e-12 for the exact-positioning models, 3e-9 for the truncation-limited Gaussian
  ones), plus a `resample` shrink regression test. Total coverage remains ~97%.

---

## 14. Warning policy, zero-size inputs, and the Fitting export (2026-06-22)

- **Divide-by-zero warnings narrowed at the source** (item #10). The blanket import-time
  `warnings.simplefilter('ignore', RuntimeWarning)` and the `PYTHONWARNINGS` env var (which
  leaked into subprocesses) were removed from `__init__.py`. A decorator now wraps every public
  entry point in `np.errstate(divide='ignore', invalid='ignore')`, so the library's intentional
  0/0 and x/0 results are suppressed at the NumPy error-state source — which holds even under
  `-W error` / pytest `filterwarnings=["error"]` — without an `errstate` at every division. All
  other `RuntimeWarning`s, and the user's own code, now surface. The remaining non-divide
  warnings are handled locally where they legitimately occur: `Degrees of freedom <= 0` for a
  single-frame unbiased variance, and masked all-NaN slices in the `nanmedian`/`nanvar` paths.
- **Zero-size inputs rejected centrally.** `_check_image` now raises
  `ValueError('invalid image shape …; size cannot be zero')`, so every operation — reductions,
  filters, and transforms — rejects a zero-size array consistently. This replaced scattered
  per-op size checks and removed earlier inconsistencies (e.g. `minimum_filter`/`maximum_filter`
  silently accepting a zero-size image, and a `median` `IndexError`). The zero-size `ValueError`
  is documented in each reduction's `Raises` section and covered by regression tests.
- **`Fitting` exported and documented** (item #9). The commented-out `Fitting` import was
  enabled and added to `__all__` and the `__init__.pyi` stub; the dead reference to the
  non-existent `scaling` module was deleted. The `Fitting` docstring was corrected (typo, a
  phantom property), and the User's Guide gained worked-example sections on FFT/correlation,
  image models, `Stretch`, and `Fitting`.

---

## Recommended priorities (remaining)

1. ~~**Fix the critical bugs.**~~ ✓ Done.
2. ~~**Convert `_ImageInfo` to a `dataclass`.**~~ ✓ Done.
3. ~~**Modernize the test suite** and **add tests for the untested modules.**~~ ✓ Done
   (2026-06-20): full pytest conversion + split, ~97% coverage, all modules covered.
4. ~~**Add CI.**~~ ✓ Done (`.github/workflows/ci.yml`).
5. ~~**Tighten mypy config and add `__all__`.**~~ ✓ Done.
6. ~~**Documentation infrastructure / User's Guide.**~~ ✓ Done (2026-06-20).

**Still open (medium/low):**

7. ~~**Lint and type debt**: `ruff check src tests` reports ~184 findings and `mypy` reports a
   few hundred errors.~~ ✓ **Done.** `ruff check src tests` is fully green (2026-06-21) and, as
   of 2026-06-22, `mypy src tests` is clean across all 56 files (the residual test-only errors
   were fixed and the `tests.*` relaxation removed). `scripts/run-all-checks.sh` now passes end
   to end (see §12, §13).
8. ~~**Add `-> None` to the ~60 remaining test functions** in the newer test suites (§4).~~
   ✓ **Done** (2026-06-22): all test functions, helpers, fixtures, and parametrized parameters
   are now annotated, enforced by the strict mypy config (§13).
9. ~~**Decide the fate of the commented-out `__init__.py` block** (§1).~~ ✓ **Done**
   (2026-06-22): `Fitting` exported (and added to `__all__`/stub), the non-existent `scaling`
   reference deleted; no commented-out block remains (§1, §14).
10. ~~**Narrow or remove the import-time `RuntimeWarning` suppression** in `__init__.py`.~~
    ✓ **Done** (2026-06-22): replaced by a source-level `np.errstate` decorator at the public
    entry points; only divide-by-zero/invalid are suppressed (§7, §14).
11. ~~**Thread safety** (§5): the module-level mutable flags are unguarded.~~ ✓ **Done**
    (2026-06-22): results are thread-safe by construction; the process-global test/tuning flags
    are now documented as not for concurrent mutation (§5). An optional `rotate`/`resample`
    refactor to remove the one internal global toggle remains available but is not required.
12. ~~**Packaging metadata** (§8): `pyproject.toml` `description` is still `"TODO"`.~~ ✓ **Done**
    (2026-06-22): real description added, license-classifier conflict removed, and
    `setuptools>=77` pinned; Pyroma now rates 10/10 (§8, §13). The commented-out
    `[project.scripts]` placeholder remains, harmless until an entry point is needed.
