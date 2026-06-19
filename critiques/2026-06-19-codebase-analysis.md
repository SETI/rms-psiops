# Codebase analysis: rms-psiops

*Generated: 2026-06-19 — Updated: 2026-06-19*

## Summary

The core algorithms are sophisticated and well-designed — the photometric-accuracy guarantees,
memory-tiling logic, and mask/weight propagation are clearly the result of careful domain
expertise. Two remediation passes have resolved all critical and high-severity bugs, added CI,
converted `_ImageInfo` to a typed dataclass, tightened the mypy config, and cleaned up
packaging. The remaining work is primarily test-suite modernization, documentation
infrastructure, and a handful of medium/low structural items.

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

- **Finding (low)**: Several modules are permanently commented out in `__init__.py` (`fft`,
  `fitting`, `stretch`, `imagemodel`). **Evidence**: `__init__.py:67–76`. **Suggestion**: If
  these aren't ready, consider a feature flag or separate install extra rather than dead
  commented-out code in the public init. At minimum, remove the block so the intent is clear.

- **Finding (medium)**: `tests/resize.py` and `tests/unittester.py` are legacy test
  infrastructure not discovered by pytest. `resize.py` contains test logic with no `test_`
  prefix; `unittester.py` is a manual runner using wildcard imports. **Evidence**:
  `tests/unittester.py`, `tests/resize.py`. **Suggestion**: Rename `resize.py` to
  `test_resize_helpers.py` or integrate it into `test_resize.py`; remove `unittester.py`
  since pytest supersedes it.

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

---

## 4. Testing

- **Finding (high)**: All tests are written as `unittest.TestCase` classes with a single
  `runTest()` method (one giant test per class). The project rule mandates pytest style,
  `pytest.raises`, and independent, parallelizable tests. With `runTest()`, a failure in the
  middle of the method stops all remaining assertions. **Evidence**: Every test file, e.g.
  `test_mean.py:11–16`. **Suggestion**: Refactor to top-level `test_*` functions (or
  `@pytest.mark.parametrize`) with one focused assertion cluster per function.

- **Finding (high)**: Tests call `warnings.simplefilter('ignore', category=RuntimeWarning)`
  in `setUp()`, directly contradicting the `filterwarnings = ["error"]` setting in
  `pyproject.toml`. This suppresses real bugs in the production code. **Evidence**:
  `test_mean.py:14`.

- **Finding (high)**: Large coverage gaps: `camouflage`, `stretch`, `fft`, `fitting`,
  `outliers`, `circle`, `gaussian_filter`, and all `imagemodel/` subclasses have no test
  files. Given the 90% coverage threshold, these modules must be hit incidentally (or not at
  all) — coverage likely passes today only because many of these modules are commented out of
  `__init__.py`. **Evidence**: `tests/` directory listing.

- **Finding (medium)**: `np.random.seed(5965)` is called at the top of `runTest()` methods
  as a global side effect. With `pytest-xdist` parallel execution, workers share a process
  so random seed state bleeds across tests. **Evidence**: `test_mean.py:19`. **Suggestion**:
  Use `rng = np.random.default_rng(seed)` scoped to each test.

- **Finding (medium)**: Tests have no type annotations on functions, violating the project
  rule (§7: "Include type annotations on test functions"). **Evidence**: All test files.

- **Finding (low)**: `tests/resize.py` is not `test_resize.py`, so pytest doesn't
  auto-discover it. It's imported by `unittester.py` via wildcard but is invisible to the
  standard `pytest` invocation. **Evidence**: `tests/resize.py`, `unittester.py:20`.

---

## 5. Performance and resource use

- ~~**Finding (medium)**: `psutil.virtual_memory().total` is called **at module import time**,
  setting `_MAX_USABLE_BYTES` once and never re-evaluating it. On long-running processes or
  containers with dynamic memory, this stale value could be misleading. **Evidence**:
  `_utils.py` (module-level).~~ ✓ **Fixed.** Removed `_MAX_USABLE_BYTES` entirely. `_USABLE_BYTES`
  is now `int | None` with `None` as a "use the live default" sentinel. `_usable_bytes()` calls
  `psutil.virtual_memory().total // 2` lazily on first read, on explicit limit-setting, and on
  reset — so each call reflects current system memory. No psutil call at import time.

- **Finding (medium — thread safety)**: Module-level mutable state (`_USE_SHORTCUTS`,
  `_LAYERS_USED`, `_TILES_USED`, `_USABLE_BYTES`) is modified via setter functions using
  `global` statements with no locking. Concurrent calls from multiple threads would race.
  **Evidence**: `_utils.py`. **Suggestion**: Document that the library is not thread-safe, or
  use `threading.local()` for test-control state.

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

- **Finding (low)**: No `docs/` directory exists despite the `pyproject.toml` pointing to a
  ReadTheDocs URL. Sphinx infrastructure is in the dev dependencies but there's no `conf.py`
  or source tree. **Evidence**: No `docs/` directory; `pyproject.toml:45`.

---

## 7. Security and robustness

- **Finding (low)**: No subprocess, `eval`, credentials, or path traversal issues found.
  Security posture is clean.

- **Finding (low)**: `RuntimeWarning` is globally suppressed for all users at import time
  via `warnings.simplefilter('ignore', ...)` and `os.environ['PYTHONWARNINGS']`. This hides
  NumPy divide-by-zero warnings that can legitimately indicate bugs in calling code.
  **Evidence**: `__init__.py:43–46`. **Suggestion**: Consider suppressing only within the
  specific operations that generate expected warnings (using `warnings.catch_warnings()`
  locally), rather than globally.

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

- **Finding (low)**: `pyproject.toml` description is `"TODO"` and `[project.scripts]` has a
  commented-out placeholder. These must be resolved before publishing to PyPI. **Evidence**:
  `pyproject.toml:8, 99`.

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

## Recommended priorities

1. ~~**Fix the critical bugs.**~~ ✓ Done.

2. ~~**Convert `_ImageInfo` to a `dataclass`.**~~ ✓ Done (also surfaced and fixed three
   additional bugs).

3. **Modernize the test suite**: Convert `unittest.TestCase` + `runTest` to proper pytest
   functions; remove the `warnings.simplefilter('ignore')` suppression in `setUp`; add tests
   for the untested modules (`camouflage`, `outliers`, `gaussian_filter`, `circle`).

4. ~~**Add CI.**~~ ✓ Done (`.github/workflows/ci.yml`).

5. ~~**Tighten mypy config and add `__all__`.**~~ ✓ Done. `disallow_untyped_defs` and
   `check_untyped_defs` enabled; `psiops.*` removed from `ignore_missing_imports`; `__all__`
   added to `__init__.py`.
