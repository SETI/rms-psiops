##########################################################################################
# tests/test_outliers.py
##########################################################################################
"""Tests for psiops.outliers.

These tests exercise the shipped `psiops.outliers.outliers` against the real
`psiops.gaussian_filter.gaussian_filter` collaborator in its default ('masked') mode.
Both shortcut settings are exercised via the parametrized `shortcuts` fixture from
conftest.py.
"""

import numpy as np

from psiops.outliers import outliers


def _noisy_image(seed: int = 0, shape: tuple[int, int] = (40, 40),
                 level: float = 100.0, noise: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(level, noise, shape)


##########################################################################################
# Basic behavior
##########################################################################################

def test_outliers_returns_bool_array(shortcuts: bool) -> None:
    result = outliers(_noisy_image(), footprint=8)
    assert result.dtype == np.bool_
    assert result.shape == (40, 40)


def test_outliers_flags_injected_outlier(shortcuts: bool) -> None:
    image = _noisy_image()
    image[20, 20] += 500.0
    result = outliers(image, footprint=8, cutoff=5)
    assert result[20, 20]


def test_outliers_clean_image_flags_nothing(shortcuts: bool) -> None:
    # A perfectly uniform image has no statistical outliers.
    image = np.full((30, 30), 50.0)
    result = outliers(image, footprint=8, cutoff=5)
    assert not result.any()


def test_outliers_does_not_flag_typical_noise(shortcuts: bool) -> None:
    # Pure Gaussian noise with a generous cutoff should flag few or no pixels.
    image = _noisy_image(seed=3, noise=1.0)
    result = outliers(image, footprint=10, cutoff=8)
    assert result.sum() < 5


def test_outliers_higher_cutoff_flags_fewer(shortcuts: bool) -> None:
    image = _noisy_image(seed=1)
    image[10, 10] += 50.0
    image[25, 30] += 30.0
    lenient = outliers(image, footprint=8, cutoff=3).sum()
    strict = outliers(image, footprint=8, cutoff=10).sum()
    assert strict <= lenient


##########################################################################################
# Mask, maskval, weights, NaNs
##########################################################################################

def test_outliers_preserves_input_mask(shortcuts: bool) -> None:
    image = _noisy_image()
    mask = np.zeros((40, 40), dtype=bool)
    mask[0, 0] = True
    result = outliers(image, footprint=8, mask=mask)
    assert result[0, 0]


def test_outliers_mask_result_shape(shortcuts: bool) -> None:
    image = _noisy_image()
    mask = np.zeros((40, 40), dtype=bool)
    mask[5, 5] = True
    result = outliers(image, footprint=8, mask=mask)
    assert result.shape == (40, 40)
    # The masked pixel is folded into the returned mask.
    assert result[5, 5]


def test_outliers_with_maskval(shortcuts: bool) -> None:
    image = _noisy_image()
    sentinel = float(image[0, 0])
    result = outliers(image, footprint=8, maskval=sentinel)
    assert result.shape == (40, 40)
    # The sentinel value is masked wherever it appears, so it shows up in the result.
    assert result[0, 0]


def test_outliers_with_weights(shortcuts: bool) -> None:
    image = _noisy_image()
    # Uniform unit weights must reproduce the no-weights result exactly: a weight of one
    # everywhere carries the same information as supplying no weights at all.
    weights = np.ones((40, 40))
    result = outliers(image, footprint=8, weights=weights)
    baseline = outliers(image, footprint=8)
    assert result.shape == (40, 40)
    assert np.array_equal(result, baseline)


def test_outliers_nans_treated_as_masked(shortcuts: bool) -> None:
    image = _noisy_image()
    image[5, 5] = np.nan
    result = outliers(image, footprint=8, nans=True)
    # NaN pixels are masked, so they appear in the returned mask.
    assert result[5, 5]


##########################################################################################
# Quantile parameter
##########################################################################################

def test_outliers_quantile_one_skips_clipping(shortcuts: bool) -> None:
    # quantile == 1.0 bypasses the squared-difference clipping branch.
    image = _noisy_image()
    image[15, 15] += 400.0
    result = outliers(image, footprint=8, quantile=1.0)
    assert result.shape == (40, 40)
    assert result[15, 15]


def test_outliers_quantile_below_one_clips(shortcuts: bool) -> None:
    # quantile < 1.0 clips the squared-difference image. A bright cluster (a few
    # percent of the pixels) survives the clip and is still detected.
    image = _noisy_image()
    image[18:22, 18:22] += 200.0
    result = outliers(image, footprint=10, quantile=0.99)
    assert result[18:22, 18:22].any()


def test_outliers_quantile_with_mask_uses_unmasked_pixels(shortcuts: bool) -> None:
    # Exercises the masked branch of the quantile clipping computation.
    image = _noisy_image()
    image[18:22, 18:22] += 200.0
    mask = np.zeros((40, 40), dtype=bool)
    mask[0, :5] = True
    result = outliers(image, footprint=10, quantile=0.99, mask=mask)
    assert result[18:22, 18:22].any()


def test_outliers_axis_reduces_stack(shortcuts: bool) -> None:
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
