##########################################################################################
# tests/test_validation.py
##########################################################################################

import numpy as np
import pytest

import psiops
from psiops._utils import _ImageInfo
from psiops._validation import _check_return

# Every stack reduction that supports `keepdims`. Each is exercised on its bare-image
# (no mask/weights) return path, which is where `keepdims` was previously dropped.
_REDUCTIONS = ['mean', 'median', 'minimum', 'maximum', 'variance', 'stdev']


def test_check_return_keepdims_bare_image() -> None:
    # Regression: when returns == 'i' (no mask/weights), _check_return must still
    # re-insert the reduced axes recorded in info.axis. Previously it returned the
    # array unchanged, silently ignoring keepdims.
    reduced = np.arange(12, dtype=float).reshape(3, 4)
    info = _ImageInfo(returns='i', axis=(0,))

    result = _check_return(reduced, None, None, info)

    assert result.shape == (1, 3, 4)
    assert np.all(result[0] == reduced)


def test_check_return_keepdims_bare_image_multiple_axes() -> None:
    # The reduced axes are re-inserted at each recorded position, in order.
    reduced = np.arange(12, dtype=float).reshape(3, 4)
    info = _ImageInfo(returns='i', axis=(0, 2))

    result = _check_return(reduced, None, None, info)

    assert result.shape == (1, 3, 1, 4)
    assert np.all(result[0, :, 0, :] == reduced)


def test_check_return_no_keepdims_bare_image_unchanged() -> None:
    # With no recorded axes, the bare-image path returns the array unchanged.
    reduced = np.arange(12, dtype=float).reshape(3, 4)
    info = _ImageInfo(returns='i')

    result = _check_return(reduced, None, None, info)

    assert result.shape == (3, 4)
    assert np.all(result == reduced)


@pytest.mark.parametrize('name', _REDUCTIONS)
def test_reduction_keepdims_bare_image_shape(name) -> None:
    # The public reductions must honor keepdims on the default (no-mask) return,
    # which uses the returns == 'i' path.
    rng = np.random.default_rng(7)
    stack = rng.random((4, 3, 8, 10))
    fn = getattr(psiops, name)

    kept = fn(stack, axis=0, keepdims=True)
    plain = fn(stack, axis=0)

    assert isinstance(kept, np.ndarray)         # bare-image path, not a list
    assert kept.shape == (1, 3, 8, 10)
    assert plain.shape == (3, 8, 10)
    # keepdims only changes shape, not values
    assert np.allclose(np.squeeze(kept, axis=0), plain, equal_nan=True)


@pytest.mark.parametrize('name', _REDUCTIONS)
def test_reduction_keepdims_multiple_axes(name) -> None:
    rng = np.random.default_rng(11)
    stack = rng.random((4, 3, 8, 10))
    fn = getattr(psiops, name)

    kept = fn(stack, axis=(0, 1), keepdims=True)

    assert kept.shape == (1, 1, 8, 10)


@pytest.mark.parametrize('name', _REDUCTIONS)
def test_reduction_keepdims_matches_numpy_shape(name) -> None:
    # The kept shape matches what numpy produces for the same reduction.
    rng = np.random.default_rng(13)
    stack = rng.random((5, 2, 6, 6))
    fn = getattr(psiops, name)

    kept = fn(stack, axis=1, keepdims=True)

    assert kept.shape == np.mean(stack, axis=1, keepdims=True).shape


def test_mean_keepdims_with_mask_still_works() -> None:
    # The multi-return (mask) path already applied keepdims; confirm it is unaffected.
    rng = np.random.default_rng(17)
    stack = rng.random((4, 3, 8, 10))
    mask = rng.random((4, 3, 8, 10)) < 0.2

    kept, kmask = psiops.mean(stack, axis=0, mask=mask, keepdims=True)

    assert kept.shape == (1, 3, 8, 10)
    assert kmask.shape == (1, 3, 8, 10)

##########################################################################################
