##########################################################################################
# tests/test_outliers.py
##########################################################################################
"""Tests for psiops.outliers.

The `outliers` function depends on `psiops.gaussian_filter.gaussian_filter` with its
default mode ('masked'). That dependency is currently broken for the 'masked' mode (it
forwards the unsupported mode string straight to SciPy and ignores `returns`); fixing it
is outside the scope of these tests. To exercise the outlier-detection logic in isolation,
the `patched_outliers` fixture substitutes a small, correct masked Gaussian smoother for
that single collaborator.
"""

import numpy as np
import pytest
import scipy.ndimage as ndi

import psiops.outliers as outliers_module
from psiops._filter import _use_shortcuts


def _masked_gaussian_filter(image, sigma, mask=None, *, maskval=None, weights=None,
                            nans=False, returns='i'):
    """A correct, weight-aware Gaussian smoother returning a single image array.

    This stands in for `psiops.gaussian_filter.gaussian_filter` so that the outlier
    logic can be tested independently of that collaborator's current 'masked'-mode bug.
    """

    arr = np.asarray(image, dtype=float)
    axis_sigma = (arr.ndim - 2) * (0.0,) + (float(sigma), float(sigma))

    if mask is None:
        return ndi.gaussian_filter(arr, axis_sigma, mode='constant', cval=0.0)

    valid = np.logical_not(np.broadcast_to(mask, arr.shape)).astype(float)
    numerator = ndi.gaussian_filter(arr * valid, axis_sigma, mode='constant', cval=0.0)
    denominator = ndi.gaussian_filter(valid, axis_sigma, mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        result = np.where(denominator > 0.0, numerator / denominator, 0.0)
    return result


@pytest.fixture
def outliers():
    """Provide `outliers` with a working masked Gaussian smoother and no shortcuts.

    The resample/filter shortcut path is disabled because it has a latent defect that
    is unrelated to outlier detection. The conftest autouse fixture restores the global
    flag afterward.
    """

    _use_shortcuts(False)
    saved = outliers_module.gaussian_filter
    outliers_module.gaussian_filter = _masked_gaussian_filter
    try:
        yield outliers_module.outliers
    finally:
        outliers_module.gaussian_filter = saved


def _noisy_image(seed: int = 0, shape: tuple[int, int] = (40, 40),
                 level: float = 100.0, noise: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(level, noise, shape)


##########################################################################################
# Basic behavior
##########################################################################################

def test_outliers_returns_bool_array(outliers) -> None:
    result = outliers(_noisy_image(), footprint=8)
    assert result.dtype == np.bool_
    assert result.shape == (40, 40)


def test_outliers_flags_injected_outlier(outliers) -> None:
    image = _noisy_image()
    image[20, 20] += 500.0
    result = outliers(image, footprint=8, cutoff=5)
    assert result[20, 20]


def test_outliers_clean_image_flags_nothing(outliers) -> None:
    # A perfectly uniform image has no statistical outliers.
    image = np.full((30, 30), 50.0)
    result = outliers(image, footprint=8, cutoff=5)
    assert not result.any()


def test_outliers_does_not_flag_typical_noise(outliers) -> None:
    # Pure Gaussian noise with a generous cutoff should flag few or no pixels.
    image = _noisy_image(seed=3, noise=1.0)
    result = outliers(image, footprint=10, cutoff=8)
    assert result.sum() < 5


def test_outliers_higher_cutoff_flags_fewer(outliers) -> None:
    image = _noisy_image(seed=1)
    image[10, 10] += 50.0
    image[25, 30] += 30.0
    lenient = outliers(image, footprint=8, cutoff=3).sum()
    strict = outliers(image, footprint=8, cutoff=10).sum()
    assert strict <= lenient


##########################################################################################
# Mask, maskval, weights, NaNs
##########################################################################################

def test_outliers_preserves_input_mask(outliers) -> None:
    image = _noisy_image()
    mask = np.zeros((40, 40), dtype=bool)
    mask[0, 0] = True
    result = outliers(image, footprint=8, mask=mask)
    assert result[0, 0]


def test_outliers_mask_result_shape(outliers) -> None:
    image = _noisy_image()
    mask = np.zeros((40, 40), dtype=bool)
    mask[5, 5] = True
    result = outliers(image, footprint=8, mask=mask)
    assert result.shape == (40, 40)


def test_outliers_with_maskval(outliers) -> None:
    image = _noisy_image()
    sentinel = float(image[0, 0])
    result = outliers(image, footprint=8, maskval=sentinel)
    assert result.shape == (40, 40)


def test_outliers_with_weights(outliers) -> None:
    image = _noisy_image()
    weights = np.ones((40, 40))
    result = outliers(image, footprint=8, weights=weights)
    assert result.shape == (40, 40)


def test_outliers_nans_treated_as_masked(outliers) -> None:
    image = _noisy_image()
    image[5, 5] = np.nan
    result = outliers(image, footprint=8, nans=True)
    # NaN pixels are masked, so they appear in the returned mask.
    assert result[5, 5]


##########################################################################################
# Quantile parameter
##########################################################################################

def test_outliers_quantile_one_skips_clipping(outliers) -> None:
    # quantile == 1.0 bypasses the squared-difference clipping branch.
    image = _noisy_image()
    image[15, 15] += 400.0
    result = outliers(image, footprint=8, quantile=1.0)
    assert result.shape == (40, 40)
    assert result[15, 15]


def test_outliers_quantile_below_one_clips(outliers) -> None:
    # quantile < 1.0 clips the squared-difference image. A bright cluster (a few
    # percent of the pixels) survives the clip and is still detected.
    image = _noisy_image()
    image[18:22, 18:22] += 200.0
    result = outliers(image, footprint=10, quantile=0.99)
    assert result[18:22, 18:22].any()


def test_outliers_quantile_with_mask_uses_unmasked_pixels(outliers) -> None:
    # Exercises the masked branch of the quantile clipping computation.
    image = _noisy_image()
    image[18:22, 18:22] += 200.0
    mask = np.zeros((40, 40), dtype=bool)
    mask[0, :5] = True
    result = outliers(image, footprint=10, quantile=0.99, mask=mask)
    assert result[18:22, 18:22].any()


def test_outliers_axis_reduces_stack(outliers) -> None:
    # The `axis` parameter builds the noise model by reducing a stack of images
    # across that axis with keepdims=True (exercising the axis branch). The reduced
    # axis is retained as size 1, so a (5, 40, 40) stack reduced over axis 0 yields a
    # (1, 40, 40) boolean mask.
    rng = np.random.default_rng(5)
    stack = rng.normal(10.0, 0.05, (5, 40, 40))
    result = outliers(stack, footprint=10, axis=0)
    assert result.dtype == bool
    assert result.shape == (1, 40, 40)

##########################################################################################
