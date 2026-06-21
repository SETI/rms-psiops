##########################################################################################
# tests/test_gaussian_filter.py
##########################################################################################

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter as _sgf

from psiops.gaussian_filter import gaussian_filter


def _scipy_ref(image: np.ndarray, sigma, mode: str, cval: float = 0.,
               order=0) -> np.ndarray:
    """SciPy reference with leading (non-spatial) axes left unfiltered."""
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    if not isinstance(order, tuple):
        order = (order, order)
    isigma = (image.ndim - 2) * (0,) + sigma
    iorder = (image.ndim - 2) * (0,) + order
    return _sgf(image, isigma, mode=mode, cval=cval, order=iorder)


##########################################################################################
# Unmasked behavior: equivalent to scipy on the spatial axes only
##########################################################################################

def test_gaussian_no_mask_constant_matches_scipy() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    a = gaussian_filter(image, 1.5, mode='constant', cval=0.5)
    b = _scipy_ref(image, 1.5, mode='constant', cval=0.5)
    assert np.abs(a - b).max() < 1.e-14


def test_gaussian_no_mask_tuple_sigma() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((2,20,20))
    a = gaussian_filter(image, (1.0, 2.0), mode='nearest')
    b = _scipy_ref(image, (1.0, 2.0), mode='nearest')
    assert np.abs(a - b).max() < 1.e-14


@pytest.mark.parametrize('mode', ['constant', 'nearest', 'wrap', 'reflect', 'mirror'])
def test_gaussian_no_mask_modes(mode: str) -> None:
    rng = np.random.default_rng(7)
    image = rng.random((2,16,16))
    a = gaussian_filter(image, 1.0, mode=mode)
    b = _scipy_ref(image, 1.0, mode=mode)
    assert np.abs(a - b).max() < 1.e-14


def test_gaussian_no_mask_order_tuple() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((2,20,20))
    a = gaussian_filter(image, 1.5, mode='reflect', order=(0, 1))
    b = _scipy_ref(image, 1.5, mode='reflect', order=(0, 1))
    assert np.abs(a - b).max() < 1.e-14


def test_gaussian_sigma_zero_is_identity() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((2,16,16))
    a = gaussian_filter(image, 0.0, mode='constant')
    assert np.abs(a - image).max() < 1.e-15


def test_gaussian_2d_image() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((20,20))
    a = gaussian_filter(image, 1.0, mode='nearest')
    b = _sgf(image, 1.0, mode='nearest')
    assert a.shape == (20,20)
    assert np.abs(a - b).max() < 1.e-14


##########################################################################################
# Masked mode: ratio of two scipy filters with constant-zero padding
##########################################################################################

def test_gaussian_masked_mode_no_mask_is_normalized() -> None:
    # With the default "masked" mode and no mask, the result equals the ratio of the
    # filtered image and the filtered ones-weights (both padded with zeros), which
    # renormalizes the boundary.
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    a = gaussian_filter(image, 1.5)
    w = np.ones((20,20))
    fw = _sgf(w, 1.5, mode='constant', cval=0.)
    fi = _scipy_ref(image, 1.5, mode='constant', cval=0.)
    assert np.abs(a - fi / fw).max() < 1.e-13


def test_gaussian_masked_mode_with_mask() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    mask = rng.random((20,20)) < 0.3
    a, amask = gaussian_filter(image, 1.5, mask=mask)
    assert a.shape == (3,20,20)
    assert amask.shape == (3,20,20)

    w = np.logical_not(mask).astype(float)
    fw = _sgf(w, 1.5, mode='constant', cval=0.)
    fi = _scipy_ref(image * w, 1.5, mode='constant', cval=0.)
    with np.errstate(invalid='ignore', divide='ignore'):
        expect = fi / fw
    good = np.broadcast_to(np.isfinite(expect), a.shape)
    assert np.abs((np.asarray(a) - np.broadcast_to(expect, a.shape))[good]).max() < 1.e-12


def test_gaussian_mask_3d() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    mask = rng.random((3,20,20)) < 0.3
    a, amask = gaussian_filter(image, 1.0, mask=mask)
    assert a.shape == (3,20,20)
    assert amask.shape == (3,20,20)

    w = np.logical_not(mask).astype(float)
    fw = _scipy_ref(w, 1.0, mode='constant', cval=0.)
    fi = _scipy_ref(image * w, 1.0, mode='constant', cval=0.)
    expect = fi / fw
    good = np.isfinite(expect)
    assert np.abs((a - expect)[good]).max() < 1.e-12


def test_gaussian_fully_masked() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    mask = np.ones((20,20), dtype=bool)
    a, amask = gaussian_filter(image, 1.5, mask=mask)
    assert a.shape == (3,20,20)
    assert amask.shape == (3,20,20)
    assert amask.all()


##########################################################################################
# Weighted filtering
##########################################################################################

def test_gaussian_weights_constant_mode() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    weights = rng.random((20,20)) + 0.5
    a, aw = gaussian_filter(image, 1.5, weights=weights, mode='nearest')
    assert a.shape == (3,20,20)
    assert aw.shape == (3,20,20)

    fw = _sgf(weights, 1.5, mode='nearest', cval=np.max(weights))
    fi = _scipy_ref(image * weights, 1.5, mode='nearest', cval=0.)
    assert np.abs(a - fi / fw).max() < 1.e-12


def test_gaussian_weights_masked_mode() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((2,20,20))
    weights = rng.random((20,20)) + 0.5
    a, _ = gaussian_filter(image, 1.0, weights=weights)
    fw = _sgf(weights, 1.0, mode='constant', cval=0.)
    fi = _scipy_ref(image * weights, 1.0, mode='constant', cval=0.)
    assert np.abs(a - fi / fw).max() < 1.e-12


##########################################################################################
# maskval, nans, MaskedArray inputs
##########################################################################################

def test_gaussian_maskval() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    image[0, 5, 5] = 42.
    a = gaussian_filter(image, 1.0, maskval=42.)
    # mask was None originally, so the default return is the image only.
    assert isinstance(a, np.ndarray)
    assert a.shape == (3,20,20)
    assert np.isfinite(a).all()


def test_gaussian_nans() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    image[0, 5, 5] = np.nan
    a = gaussian_filter(image, 1.0, nans=True)
    assert isinstance(a, np.ndarray)
    assert np.isfinite(a).all()


def test_gaussian_maskedarray_input() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    m = rng.random((3,20,20)) < 0.2
    ma = np.ma.MaskedArray(image, mask=m)
    a = gaussian_filter(ma, 1.0)
    assert isinstance(a, np.ma.MaskedArray)
    assert a.shape == (3,20,20)


##########################################################################################
# returns overrides
##########################################################################################

def test_gaussian_returns_override_iw() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    mask = rng.random((20,20)) < 0.3
    a, aw = gaussian_filter(image, 1.0, mask=mask, returns='iw')
    assert a.shape == (3,20,20)
    assert aw.shape == (3,20,20)


def test_gaussian_returns_override_imw() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((3,20,20))
    weights = rng.random((3,20,20)) + 0.5
    a, amask, aw = gaussian_filter(image, 1.0, weights=weights, returns='imw')
    assert a.shape == (3,20,20)
    assert amask.shape == (3,20,20)
    assert aw.shape == (3,20,20)


##########################################################################################
# Error paths
##########################################################################################

def test_gaussian_negative_sigma() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((2,16,16))
    with pytest.raises(ValueError):
        _ = gaussian_filter(image, -1.0, mode='constant')


def test_gaussian_negative_order() -> None:
    rng = np.random.default_rng(7)
    image = rng.random((2,16,16))
    with pytest.raises(ValueError):
        _ = gaussian_filter(image, 1.0, order=-1, mode='constant')


def test_gaussian_non_numeric_dtype() -> None:
    # A non-numeric array cannot be converted to floats for filtering.
    image = np.array([[['a', 'b'], ['c', 'd']]])
    with pytest.raises((TypeError, ValueError)):
        _ = gaussian_filter(image, 1.0, mode='constant')

##########################################################################################
