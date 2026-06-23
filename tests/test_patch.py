##########################################################################################
# tests/test_patch.py
##########################################################################################
"""Tests for psiops.patch."""

import numpy as np
import pytest

from psiops.patch import patch


def _hole_image(shape: tuple[int, int] = (20, 20), value: float = 1.0,
                badval: float = -999.0, lo: int = 8,
                hi: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Return a uniform image with a square hole and its boolean mask."""

    image = np.full(shape, value, dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    mask[lo:hi, lo:hi] = True
    image[mask] = badval
    return image, mask


def test_no_mask_returns_image_unchanged() -> None:
    image = np.arange(100.0).reshape(10, 10)
    result = patch(image)
    assert np.array_equal(result, image)
    assert not isinstance(result, list)


def test_no_mask_returns_im_gives_list() -> None:
    image = np.arange(100.0).reshape(10, 10)
    result = patch(image, returns='im')
    assert isinstance(result, list)
    assert len(result) == 2
    assert np.array_equal(result[0], image)
    assert not result[1].any()


def test_masked_hole_is_filled() -> None:
    image, mask = _hole_image()
    result = patch(image, mask)
    # The bad value must no longer appear anywhere.
    assert not np.any(result == -999.0)
    assert np.all(np.isfinite(result))


def test_filled_value_blends_with_surroundings() -> None:
    # A uniform surround of 1.0 means the filled hole should also be ~1.0.
    image, mask = _hole_image(value=1.0)
    result = patch(image, mask)
    assert np.allclose(result[mask], 1.0)


def test_unmasked_pixels_are_preserved() -> None:
    image, mask = _hole_image(value=5.0)
    result = patch(image, mask)
    antimask = ~mask
    assert np.allclose(result[antimask], 5.0)


def test_fill_value_interpolates_gradient() -> None:
    # On a gradient, the filled center should lie between its neighbors' extremes.
    ramp = np.add.outer(np.arange(20.0), np.arange(20.0))
    mask = np.zeros((20, 20), dtype=bool)
    mask[9:11, 9:11] = True
    result = patch(ramp, mask)
    filled = result[mask]
    # Neighbors around the 2x2 hole span roughly 16..22; filled must be within.
    assert np.all(filled > 14.0)
    assert np.all(filled < 24.0)


def test_returns_im_mask_clean_for_small_hole() -> None:
    image, mask = _hole_image()
    image_out, new_mask = patch(image, mask, returns='im')
    # A small hole inside a large image is fully fillable.
    assert not new_mask.any()
    assert not np.any(image_out == -999.0)


def test_returns_iw_provides_weights() -> None:
    image, mask = _hole_image()
    _image_out, weights = patch(image, mask, returns='iw')
    assert weights.shape == image.shape
    # Unmasked pixels keep weight 1; filled pixels get positive weight.
    assert np.all(weights[~mask] == 1.0)
    assert np.all(weights[mask] > 0.0)


def test_returns_imw_provides_three() -> None:
    image, mask = _hole_image()
    result = patch(image, mask, returns='imw')
    assert isinstance(result, list)
    assert len(result) == 3


def test_maskval_path() -> None:
    image = np.ones((15, 15), dtype=np.float64)
    image[7, 7] = -5.0
    result = patch(image, maskval=-5.0)
    assert result[7, 7] == pytest.approx(1.0)
    assert not np.any(result == -5.0)


def test_nans_path_fills_without_leaving_nan() -> None:
    image = np.ones((15, 15), dtype=np.float64)
    image[7, 7] = np.nan
    image_out, new_mask = patch(image, nans=True, returns='im')
    assert np.isfinite(image_out[7, 7])
    assert image_out[7, 7] == pytest.approx(1.0)
    assert not new_mask[7, 7]


def test_weights_input_treated_like_mask() -> None:
    image = np.ones((15, 15), dtype=np.float64)
    weights = np.ones((15, 15))
    weights[7, 7] = 0.0
    image_out, _new_mask = patch(image, weights=weights, returns='im')
    assert image_out[7, 7] == pytest.approx(1.0)


def test_maskedarray_input_returns_maskedarray() -> None:
    mask = np.zeros((15, 15), dtype=bool)
    mask[7, 7] = True
    marr = np.ma.MaskedArray(np.ones((15, 15)), mask=mask)
    result = patch(marr)
    assert isinstance(result, np.ma.MaskedArray)
    assert result.data[7, 7] == pytest.approx(1.0)


def test_integer_image_with_maskval_keeps_dtype() -> None:
    image = np.ones((10, 10), dtype=int)
    image[5, 5] = 99
    result = patch(image, maskval=99)
    assert result.dtype == image.dtype
    assert result[5, 5] != 99


def test_large_hole_leaves_residual_mask() -> None:
    # A hole much larger than `size` cannot be fully filled (Gaussian underflow).
    image = np.ones((60, 60), dtype=np.float64)
    mask = np.zeros((60, 60), dtype=bool)
    mask[5:55, 5:55] = True
    _, new_mask = patch(image, mask, size=10, returns='im')
    assert new_mask.any()
    # Residual mask must lie within the original hole.
    assert np.all(new_mask <= mask)


def test_size_parameter_changes_footprint_count() -> None:
    # Both sizes should fill a single isolated pixel to the surround value.
    image = np.ones((12, 12), dtype=np.float64)
    mask = np.zeros((12, 12), dtype=bool)
    mask[6, 6] = True
    for size in (3, 5, 30):
        result = patch(image, mask, size=size)
        assert result[6, 6] == pytest.approx(1.0)


def test_multiple_separate_holes_all_filled() -> None:
    image = np.ones((20, 20), dtype=np.float64)
    mask = np.zeros((20, 20), dtype=bool)
    mask[3, 3] = True
    mask[15, 16] = True
    mask[10:12, 10:12] = True
    result = patch(image, mask)
    assert np.allclose(result[mask], 1.0)


def test_invalid_returns_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        patch(np.ones((5, 5)), returns='x')
    assert 'invalid `returns` value "x"' in str(exc_info.value)


def test_one_dimensional_image_raises() -> None:
    with pytest.raises(ValueError) as exc_info:
        patch(np.ones(5))
    assert 'must be at least 2-D' in str(exc_info.value)


def test_does_not_mutate_input_image() -> None:
    image, mask = _hole_image()
    snapshot = image.copy()
    _ = patch(image, mask)
    assert np.array_equal(image, snapshot)

##########################################################################################
