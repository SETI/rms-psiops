##########################################################################################
# tests/test_zoom.py
##########################################################################################

import numpy as np
import pytest

from psiops.unzoom import unzoom
from psiops.zoom import zoom


def _make_image() -> np.ndarray:
    array = np.arange(10)
    return array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]


##########################################################################################
# zoom() tests
##########################################################################################

def test_zoom_up_and_down_no_mask() -> None:
    image = _make_image()
    zoomed = zoom(image, 3)
    assert zoomed.shape == (10, 30, 30)
    unzoomed = unzoom(zoomed, (3, 1))
    assert unzoomed.shape == (10, 10, 30)
    unzoomed = unzoom(zoomed, 3)
    assert np.all(unzoomed == image)


def test_zoom_up_with_mask() -> None:
    image = _make_image()
    mask = np.zeros(image.shape, dtype='bool')
    mask[:3, :3, :3] = True

    zoomed, zmask = zoom(image, 2, mask)
    assert np.all(image == zoomed[:, ::2, ::2])
    assert np.all(image == zoomed[:, 1::2, ::2])
    assert np.all(image == zoomed[:, ::2, 1::2])
    assert np.all(image == zoomed[:, 1::2, 1::2])

    assert np.all(zmask[:, ::2, ::2] == mask)
    assert np.all(zmask[:, 1::2, ::2] == mask)
    assert np.all(zmask[:, ::2, 1::2] == mask)
    assert np.all(zmask[:, 1::2, 1::2] == mask)


def test_zoom_up_asymmetric_with_mask() -> None:
    image = _make_image()
    mask = np.zeros(image.shape, dtype='bool')
    mask[:3, :3, :3] = True

    zoomed, zmask = zoom(image, (3, 1), mask)
    assert np.all(image == zoomed[:, ::3])
    assert np.all(image == zoomed[:, 1::3])
    assert np.all(image == zoomed[:, 2::3])

    assert np.all(zmask[:, ::3] == mask)
    assert np.all(zmask[:, 1::3] == mask)
    assert np.all(zmask[:, 2::3] == mask)


def test_zoom_with_weights() -> None:
    image = _make_image()
    weights = np.ones(image.shape, dtype=np.float64)
    weights[:, 0, 0] = 0.
    weights[:, 1, 1] = 2.

    zoomed, zweights = zoom(image, 2, weights=weights, returns='iw')
    assert zoomed.shape == (10, 20, 20)
    # Replicated weights match the source weights at every replicated position
    assert np.all(zweights[:, ::2, ::2] == weights)
    assert np.all(zweights[:, 1::2, 1::2] == weights)
    # Zero-weight pixels propagate to all their replicas
    assert np.all(zweights[:, 0:2, 0:2] == 0.)


def test_zoom_errors() -> None:
    image = _make_image()
    exc_info: pytest.ExceptionInfo[Exception]
    with pytest.raises(TypeError) as exc_info:
        zoom(image, 3.2)
    assert 'two integers required' in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        zoom(image, (3.2, 1))
    assert 'two integers required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        zoom(image, (-3, 1))
    assert 'positive values required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        zoom(image, (1, 2, 3))
    assert 'zoom factor (1, 2, 3); two values required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        zoom(image, (2,))
    assert 'zoom factor (2,); two values required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        zoom(image, None)
    assert 'missing zoom factor' in str(exc_info.value)


def test_zoom_dtype_preserved() -> None:
    array = np.arange(10)
    image = array + array[:, np.newaxis]
    for dtype in ('int', 'float', 'float32', 'bool', 'uint8', 'uint16', 'int16'):
        typed_image = image.astype(dtype)
        zoomed = zoom(typed_image, 2)
        assert typed_image.dtype == zoomed.dtype


def test_zoom_maskedarray() -> None:
    array = np.arange(10)
    image = array + array[:, np.newaxis]
    mask = np.zeros(image.shape, dtype='bool')
    mask[0, 0] = mask[3, 4] = True
    marray = np.ma.MaskedArray(image, mask=mask)

    zoomed = zoom(marray, 2)
    assert isinstance(zoomed, np.ma.MaskedArray)
    zmask = np.asarray(zoomed.mask)
    assert np.all(zmask[::2, ::2] == mask)
    assert np.all(zmask[1::2, 1::2] == mask)


def test_zoom_maskval() -> None:
    array = np.arange(10)
    image = (array + array[:, np.newaxis]).astype(np.float64)
    image[2, 2] = -999.

    zoomed, zmask = zoom(image, 2, maskval=-999., returns='im')
    assert np.all(zmask[4:6, 4:6])
    assert np.all(zoomed[4:6, 4:6] == -999.)


##########################################################################################
# unzoom() tests
##########################################################################################

def test_unzoom_no_mask_averaging() -> None:
    image = _make_image()
    array = np.arange(10)
    unzoomed = unzoom(image, (1, 2))
    answer = (np.arange(0.5, 9, 2) + array[:, np.newaxis]
              + array[:, np.newaxis, np.newaxis])
    assert np.all(unzoomed == answer)


def test_unzoom_with_mask() -> None:
    image = _make_image()
    unzoomed = unzoom(image, (1, 2))

    mask = np.zeros(image.shape, dtype='bool')
    mask[:3, :3, :3] = True
    unzoomed2, umask = unzoom(image, (1, 2), mask=mask)

    new_mask = np.zeros(unzoomed.shape, dtype='bool')
    new_mask[:3, :3, :1] = True
    assert np.all(new_mask == umask)

    assert np.all(unzoomed2[umask] == 0)
    assert np.all(unzoomed2[:3, :3, 1] == image[:3, :3, 3])
        # because [:3,:3,2] is masked
    assert np.sum(unzoomed2 != unzoomed) == 18


def test_unzoom_with_weights() -> None:
    image = _make_image()
    weights = np.ones(image.shape, dtype=np.float64)
    weights[:, :, ::2] = 0.     # zero-weight the first column of every 2-pixel group

    unzoomed, new_weights = unzoom(image, (1, 2), weights=weights, returns='iw')
    # The first column of each group is zero-weight, so the result equals the second
    assert np.all(unzoomed == image[:, :, 1::2])
    assert np.all(new_weights == 1.)


def test_unzoom_fully_masked_pixel_fills_zero() -> None:
    image = _make_image()
    mask = np.zeros(image.shape, dtype='bool')
    mask[:, 0, :2] = True       # fully mask a 1x2 group -> 0/0

    unzoomed, umask = unzoom(image, (1, 2), mask=mask, returns='im')
    assert np.all(umask[:, 0, 0])
    assert np.all(unzoomed[:, 0, 0] == 0.)


def test_unzoom_maskval_sets_fill_value() -> None:
    array = np.arange(10)
    image = (array + array[:, np.newaxis]).astype(np.float64)
    image[0, 0] = image[0, 1] = -999.       # fully mask one 1x2 group

    unzoomed, umask = unzoom(image, (1, 2), maskval=-999., returns='im')
    assert umask[0, 0]
    assert unzoomed[0, 0] == -999.


def test_unzoom_errors() -> None:
    image = _make_image()
    exc_info: pytest.ExceptionInfo[Exception]
    with pytest.raises(TypeError) as exc_info:
        unzoom(image, 3.2)
    assert 'two integers required' in str(exc_info.value)
    with pytest.raises(TypeError) as exc_info:
        unzoom(image, (3.2, 1))
    assert 'two integers required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        unzoom(image, (-3, 1))
    assert 'positive values required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        unzoom(image, (1, 2, 3))
    assert 'unzoom factor (1, 2, 3); two values required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        unzoom(image, (2,))
    assert 'unzoom factor (2,); two values required' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        unzoom(image, None)
    assert 'missing unzoom factor' in str(exc_info.value)
    with pytest.raises(ValueError) as exc_info:
        unzoom(image, 3)     # shape is (10,10)
    assert 'is not divisible by unzoom factor' in str(exc_info.value)


def test_unzoom_dtype_float32_preserved() -> None:
    array = np.arange(10)
    image = (array + array[:, np.newaxis]).astype('float32')
    unzoomed = unzoom(image, 2)
    assert image.dtype == unzoomed.dtype


def test_unzoom_maskedarray() -> None:
    array = np.arange(10)
    image = (array + array[:, np.newaxis]).astype(np.float64)
    mask = np.zeros(image.shape, dtype='bool')
    mask[0, 0] = mask[0, 1] = True      # fully mask one 1x2 group
    marray = np.ma.MaskedArray(image, mask=mask)

    unzoomed = unzoom(marray, (1, 2))
    assert isinstance(unzoomed, np.ma.MaskedArray)

    # The fully-masked group is reflected in the mask only when explicitly requested.
    _, umask = unzoom(marray, (1, 2), returns='im')
    assert umask[0, 0]

##########################################################################################
