##########################################################################################
# tests/test_resize.py
##########################################################################################

import numpy as np
import pytest

from psiops.unzoom import unzoom
from psiops.zoom import zoom
from tests.resize import resize  # removed from image_ops but retained for cross-testing


def _make_image() -> np.ndarray:
    array = np.arange(10)
    return array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]


def test_resize_vs_zoom_no_mask() -> None:
    image = _make_image()
    zoomed = zoom(image, (3, 2))
    resized = resize(image, (30, 20))
    assert np.all(zoomed == resized)


def test_resize_vs_zoom_3d_mask() -> None:
    rng = np.random.default_rng(4163)
    image = _make_image()
    mask = rng.random((10, 10, 10)) < 0.2
    zoomed, zmask = zoom(image, (3, 2), mask)
    resized, rmask = resize(image, (30, 20), mask)
    assert np.all(zmask == rmask)
    assert np.all(zoomed[~zmask] == resized[~zmask])


def test_resize_vs_zoom_2d_mask() -> None:
    rng = np.random.default_rng(4163)
    image = _make_image()
    mask = rng.random((10, 10)) < 0.2
    zoomed, zmask = zoom(image, (3, 2), mask)
    resized, rmask = resize(image, (30, 20), mask)
    # zoom() preserves the 2-D mask shape; broadcast to compare with the 3-D image
    zmask_b = np.broadcast_to(zmask, zoomed.shape)
    assert np.all(zmask_b == rmask)
    assert np.all(zoomed[~zmask_b] == resized[~zmask_b])


def test_resize_vs_unzoom_no_mask() -> None:
    array = np.arange(12)
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    unzoomed = unzoom(image, (3, 2))
    resized = resize(image, (4, 6))
    assert np.all(unzoomed == resized)


def test_resize_vs_unzoom_3d_mask() -> None:
    rng = np.random.default_rng(4163)
    array = np.arange(12)
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    mask = rng.random((12, 12, 12)) < 0.2
    unzoomed, zmask = unzoom(image, (3, 2), mask)
    resized, rmask = resize(image, (4, 6), mask)
    assert np.all(zmask == rmask)
    assert np.all(unzoomed[~zmask] == resized[~zmask])


def test_resize_vs_unzoom_2d_mask() -> None:
    rng = np.random.default_rng(4163)
    array = np.arange(12)
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    mask = rng.random((12, 12)) < 0.2
    unzoomed, zmask = unzoom(image, (3, 2), mask)
    resized, rmask = resize(image, (4, 6), mask)
    # unzoom() preserves the 2-D mask shape; broadcast to compare with the 3-D image
    zmask_b = np.broadcast_to(zmask, unzoomed.shape)
    assert np.all(zmask_b == rmask)
    assert np.all(unzoomed[~zmask_b] == resized[~zmask_b])


def test_resize_mixed_mean_preserved() -> None:
    image = _make_image()
    resized = resize(image, (7, 11))
    assert np.all(image.mean(axis=(1, 2)) == resized.mean(axis=(1, 2)))


def test_resize_mixed_with_mask() -> None:
    image = _make_image()
    resized = resize(image, (7, 11))

    mask = np.zeros(image.shape, dtype='bool')
    mask[:3, :3, :3] = True
    resized2, _ = resize(image, (7, 11), mask)
    assert np.all(resized[3:] == resized2[3:])
    assert np.all(resized[:3, 3:] == resized2[:3, 3:])
    assert np.all(resized[:3, :3, 4:] == resized2[:3, :3, 4:])
    assert np.all(resized2[:3, :2, :3] == 0)


def test_resize_errors() -> None:
    image = _make_image()
    with pytest.raises(TypeError):
        resize(image, 3.2)
    with pytest.raises(TypeError):
        resize(image, (3.2, 1))
    with pytest.raises(TypeError):
        resize(image, '')
    with pytest.raises(ValueError):
        resize(image, (-3, 1))
    with pytest.raises(ValueError):
        resize(image, (1, 2, 3))
    with pytest.raises(ValueError):
        resize(image, (2,))
    with pytest.raises(ValueError):
        resize(image, None)


def test_resize_dtype_float32_preserved() -> None:
    array = np.arange(10)
    image = (array + array[:, np.newaxis]).astype('float32')
    resized = resize(image, (12, 9))
    assert image.dtype == resized.dtype


def test_resize_scalar_mask_broadcast() -> None:
    array = np.arange(10)
    image = (array + array[:, np.newaxis]).astype('float32')
    # A scalar mask should be broadcast and converted to bool
    resized, rmask = resize(image, (12, 9), mask=7.)
    assert rmask.dtype == np.dtype('bool')
    assert rmask.shape == resized.shape
    assert np.all(rmask)

##########################################################################################
