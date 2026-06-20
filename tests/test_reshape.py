##########################################################################################
# tests/test_reshape.py
##########################################################################################

import numpy as np
import pytest
from psiops.reshape import reshape as resize    # rename and then re-use resize tests
from psiops.unzoom  import unzoom
from psiops.zoom    import zoom

EPS = 2.e-14


def _make_image() -> np.ndarray:
    array = np.arange(10)
    return array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]


def test_reshape_vs_zoom_no_mask() -> None:
    image = _make_image()
    zoomed = zoom(image, (3, 2))
    resized = resize(image, (30, 20))
    assert np.all(zoomed == resized)


def test_reshape_vs_zoom_3d_mask() -> None:
    rng = np.random.default_rng(4163)
    image = _make_image()
    mask = rng.random((10, 10, 10)) < 0.2
    zoomed, zmask = zoom(image, (3, 2), mask)
    resized, rmask = resize(image, (30, 20), mask)
    assert np.all(zmask == rmask)
    assert np.all(zoomed[~zmask] == resized[~zmask])


def test_reshape_vs_zoom_2d_mask() -> None:
    rng = np.random.default_rng(4163)
    image = _make_image()
    mask = rng.random((10, 10)) < 0.2
    zoomed, zmask = zoom(image, (3, 2), mask)
    resized, rmask = resize(image, (30, 20), mask)
    # zoom() keeps the 2-D mask shape; reshape() broadcasts to the image's 3-D shape.
    zmask_b = np.broadcast_to(zmask, zoomed.shape)
    assert np.all(zmask_b == rmask)
    assert np.all(zoomed[~zmask_b] == resized[~zmask_b])


def test_reshape_vs_unzoom_no_mask() -> None:
    array = np.arange(12)
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    unzoomed = unzoom(image, (3, 2))
    resized = resize(image, (4, 6))
    assert np.abs(unzoomed - resized).max() < EPS


def test_reshape_vs_unzoom_3d_mask() -> None:
    rng = np.random.default_rng(4163)
    array = np.arange(12)
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    mask = rng.random((12, 12, 12)) < 0.2
    unzoomed, zmask = unzoom(image, (3, 2), mask)
    resized, rmask = resize(image, (4, 6), mask)
    assert np.all(zmask == rmask)
    assert np.abs(unzoomed[~zmask] - resized[~zmask]).max() < EPS


def test_reshape_vs_unzoom_2d_mask() -> None:
    rng = np.random.default_rng(4163)
    array = np.arange(12)
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    mask = rng.random((12, 12)) < 0.2
    unzoomed, zmask = unzoom(image, (3, 2), mask)
    resized, rmask = resize(image, (4, 6), mask)
    # unzoom() keeps the 2-D mask shape; reshape() broadcasts to the image's 3-D shape.
    zmask_b = np.broadcast_to(zmask, unzoomed.shape)
    assert np.all(zmask_b == rmask)
    assert np.abs(unzoomed[~zmask_b] - resized[~zmask_b]).max() < EPS


def test_reshape_mixed_mean_preserved() -> None:
    image = _make_image()
    resized = resize(image, (7, 11))
    assert np.all(image.mean(axis=(1, 2)) == resized.mean(axis=(1, 2)))


def test_reshape_mixed_with_mask() -> None:
    image = _make_image()
    resized = resize(image, (7, 11))

    mask = np.zeros(image.shape, dtype='bool')
    mask[:3, :3, :3] = True
    resized2, rmask = resize(image, (7, 11), mask)
    assert np.abs(resized[3:] - resized2[3:]).max() < EPS
    assert np.abs(resized[:3, 3:] - resized2[:3, 3:]).max() < EPS
    assert np.abs(resized[:3, :3, 4:] - resized2[:3, :3, 4:]).max() < EPS
    # The fully-masked corner is flagged in the mask; reshape() leaves masked output
    # pixels as NaN (no fill value was requested).
    assert np.all(rmask[:3, :2, :3])
    assert np.all(np.isnan(resized2[:3, :2, :3]))


def test_reshape_errors() -> None:
    image = _make_image()
    with pytest.raises(TypeError): resize(image, 3.2)
    with pytest.raises(TypeError): resize(image, (3.2, 1))
    with pytest.raises(TypeError): resize(image, '')
    with pytest.raises(ValueError): resize(image, (-3, 1))
    with pytest.raises(ValueError): resize(image, (1, 2, 3))
    with pytest.raises(ValueError): resize(image, (2,))
    with pytest.raises(ValueError): resize(image, None)


def test_reshape_dtype_float32_preserved() -> None:
    array = np.arange(10).astype('float32')
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    resized = resize(image, (12, 9))
    assert image.dtype == resized.dtype


def test_reshape_image_only_return() -> None:
    image = _make_image()
    # Without a mask, reshape() returns just the image array (not a tuple)
    resized = resize(image, (30, 20))
    assert isinstance(resized, np.ndarray)
    assert resized.shape == (10, 30, 20)


def test_reshape_returns_iw() -> None:
    image = _make_image()
    resized, weights = resize(image, (5, 5), returns='iw')
    assert resized.shape == (10, 5, 5)
    assert weights.shape == (10, 5, 5)


def test_reshape_maskedarray() -> None:
    image = _make_image().astype(np.float64)
    mask = np.zeros(image.shape, dtype='bool')
    mask[:, 0, 0] = True
    marray = np.ma.MaskedArray(image, mask=mask)
    resized = resize(marray, (5, 5))
    assert isinstance(resized, np.ma.MaskedArray)


def test_reshape_2d_image_requires_3d() -> None:
    # reshape() delegates to resample(), which requires at least 3 dimensions.
    array = np.arange(10)
    image = array + array[:, np.newaxis]
    with pytest.raises(ValueError):
        resize(image, (12, 9))

##########################################################################################
