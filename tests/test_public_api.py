##########################################################################################
# tests/test_public_api.py
##########################################################################################
#
# Smoke tests for the public `psiops` package surface. The rest of the suite imports each
# function from its submodule; these go through the top-level package instead, so they
# exercise the re-export wiring in `psiops/__init__.py` and the
# `_suppress_divide_warnings` wrapper it applies to every public entry point -- neither of
# which a submodule import reaches.

import warnings

import numpy as np
import pytest

import psiops

# `__all__` is a runtime attribute the type stub does not declare; read it once here.
_ALL: list[str] = psiops.__all__   # type: ignore[attr-defined]

# Split `__all__` into the module-level functions (wrapped by rebinding the package name)
# and the modeling classes (whose specific methods are wrapped on the class). Derived from
# `__all__` so the lists cannot drift from the package. The `None` sentinel keeps a stale
# `__all__` entry from crashing collection here; `test_public_api_all_names_importable`
# reports it cleanly instead.
_FUNCTION_NAMES = [n for n in _ALL if not isinstance(getattr(psiops, n, None), type)]
_CLASS_NAMES = [n for n in _ALL if isinstance(getattr(psiops, n, None), type)]

# (class, method) pairs wrapped in `psiops/__init__.py`; kept in sync with the loop there.
_WRAPPED_METHODS = [
    ('ArrayModel', 'transform'),
    ('Gaussian', 'transform'),
    ('SmearedModel', 'transform'),
    ('SummedModel', 'transform'),
    ('Stretch', 'set_image'),
    ('Stretch', 'set_target'),
    ('Stretch', 'fit'),
    ('Fitting', 'set_target'),
    ('Fitting', 'fit'),
]

##########################################################################################
# Re-export wiring
##########################################################################################

def test_public_api_all_names_importable() -> None:
    assert _ALL
    for name in _ALL:
        assert hasattr(psiops, name), f'{name} is in __all__ but missing from psiops'
        assert callable(getattr(psiops, name)), f'psiops.{name} is not callable'


def test_public_api_classes_are_types() -> None:
    # The modeling names must resolve to the actual classes, not stale/rebound objects.
    assert set(_CLASS_NAMES) == {'ArrayModel', 'Fitting', 'Gaussian', 'ImageModel',
                                 'SmearedModel', 'Stretch', 'SummedModel'}


def test_public_api_function_forwards_to_submodule() -> None:
    # A package-level call must forward to the real submodule implementation and match it.
    from psiops.mean import mean as raw_mean
    image = np.arange(24, dtype=float).reshape(2, 3, 4)
    assert np.array_equal(psiops.mean(image), raw_mean(image))
    assert psiops.mean.__wrapped__ is raw_mean   # type: ignore[attr-defined]


def test_public_api_class_call_through_package() -> None:
    # Exercise a modeling class via the package surface end to end.
    psf = psiops.Gaussian(sigma=2.0).transform((21, 21), center=(10.5, 10.5))
    assert psf.shape == (21, 21)
    assert psf.sum() == pytest.approx(1.0)

##########################################################################################
# Divide-warning suppression (_suppress_divide_warnings)
##########################################################################################

@pytest.mark.parametrize('name', _FUNCTION_NAMES)
def test_public_function_is_divide_wrapped(name: str) -> None:
    # Every re-exported function is wrapped, so a documented 0/0 or x/0 cannot leak a
    # RuntimeWarning to a `psiops`-importing user (or this suite's filterwarnings=error).
    assert hasattr(getattr(psiops, name), '__wrapped__'), f'psiops.{name} is not wrapped'


@pytest.mark.parametrize(('cls_name', 'method'), _WRAPPED_METHODS)
def test_public_method_is_divide_wrapped(cls_name: str, method: str) -> None:
    bound = getattr(getattr(psiops, cls_name), method)
    assert hasattr(bound, '__wrapped__'), f'psiops.{cls_name}.{method} is not wrapped'


def test_public_api_suppresses_divide_warning() -> None:
    # Gaussian(sigma=0).transform divides by zero in its exponent; through the package the
    # wrapper must keep that from surfacing as a warning even under warnings-as-errors.
    model = psiops.Gaussian(sigma=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        result = model.transform((11, 11), (5.5, 5.5))
    assert result.shape == (11, 11)


def test_public_api_wrapper_is_what_suppresses() -> None:
    # Sanity check that the wrapper is doing the work: the unwrapped implementation does
    # raise a divide warning on the same input, so the test above is meaningful.
    model = psiops.Gaussian(sigma=0.0)
    raw = psiops.Gaussian.transform.__wrapped__   # type: ignore[attr-defined]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        with pytest.raises(RuntimeWarning, match='divide by zero'):
            raw(model, (11, 11), (5.5, 5.5))
