##########################################################################################
# tests/test_utils.py
##########################################################################################

import numpy as np
import pytest
import scipy.ndimage

from psiops.maximum import maximum_filter
from psiops.median import median_filter
from psiops._filter import (_apply_op_as_filter_info, _image_to_4d, _usable_bytes,
                            _use_shortcuts)
from psiops._utils  import _check_tuple, _check_axis


def test_check_tuple() -> None:

    assert _check_tuple(None,     'index', nones=True)       == None
    assert _check_tuple(None,     'index', default='abc')    == 'abc'
    assert _check_tuple(1,        'index')                   == (1,1)
    assert _check_tuple([2,3],    'index')                   == (2,3)
    assert _check_tuple([2,3.],   'index')                   == (2,3.)
    assert _check_tuple([2,-3.],  'index', negs=True)        == (2,-3.)
    assert _check_tuple(-1,       'index', negs=True)        == (-1,-1)
    assert _check_tuple(np.arange(7,9),'index')              == (7,8)


def test_check_tuple_errors() -> None:

    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple(None, 'index')
    assert str(exc_info.value) == 'missing index'

    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple([], 'index')
    assert str(exc_info.value) == 'invalid index (); two values required'

    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple(np.arange(3), 'index')
    assert str(exc_info.value) == 'invalid index (0, 1, 2); two values required'

    with pytest.raises(TypeError) as exc_info:
        _ = _check_tuple((3,4.), 'index', floats=False)
    assert str(exc_info.value) == 'invalid index (3, 4.0); two integers required'

    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple(3, 'index', shape=(100,100))
    assert str(exc_info.value) == 'shape (100, 100) is not divisible by index (3, 3)'


def test_check_tuple_negs_zeros() -> None:

    # Zeros disallowed by default
    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple(0, 'index')
    assert str(exc_info.value) == 'invalid index (0, 0); positive values required'

    # Negatives disallowed by default
    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple(-1, 'index')
    assert str(exc_info.value) == 'invalid index (-1, -1); positive values required'

    # Negatives disallowed but zeros allowed
    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple(-1, 'index', zeros=True)
    assert str(exc_info.value) == 'invalid index (-1, -1); non-negative values required'

    # Zeros disallowed but negatives allowed
    with pytest.raises(ValueError) as exc_info:
        _ = _check_tuple(0, 'index', negs=True, zeros=False)
    assert str(exc_info.value) == 'invalid index (0, 0); non-zero values required'

    # Both allowed (zeros defaults to negs)
    assert _check_tuple(0, 'index', negs=True) == (0, 0)
    assert _check_tuple(0, 'index', negs=True, zeros=True) == (0, 0)

##########################################################################################

def test_check_axis() -> None:

    assert _check_axis(None, (2,3,4,5,100,100)) == (0,1,2,3)
    assert _check_axis(1, (2,3,4,5,100,100)) == (1,)
    assert _check_axis(-1, (2,3,4,5,100,100)) == (3,)
    assert _check_axis((3,2,1), (2,3,4,5,100,100)) == (3,2,1)
    assert _check_axis([3,1,2], (2,3,4,5,100,100)) == (3,1,2)
    assert _check_axis(np.arange(1,4), (2,3,4,5,100,100)) == (1,2,3)


def test_check_axis_errors() -> None:

    with pytest.raises(TypeError) as exc_info:
        _ = _check_axis(1.5, (2,3,4,100,100))
    assert str(exc_info.value) == 'invalid axis of type float: 1.5'

    with pytest.raises(TypeError) as exc_info:
        _ = _check_axis((0,'abc'), (2,3,4,100,100))
    assert str(exc_info.value) == "invalid axis item 'abc' of type str"

    with pytest.raises(TypeError) as exc_info:
        _ = _check_axis((0,1.), (2,3,4,100,100))
    assert str(exc_info.value) == 'invalid axis item 1.0 of type float'

    with pytest.raises(IndexError) as exc_info:
        _ = _check_axis((0,5), (2,3,4,100,100))
    assert str(exc_info.value) == 'axis value 5 out of range for shape (2, 3, 4)'

    with pytest.raises(IndexError) as exc_info:
        _ = _check_axis(-6, (2,3,4,100,100))
    assert str(exc_info.value) == 'axis value -6 out of range for shape (2, 3, 4)'

    with pytest.raises(ValueError) as exc_info:
        _ = _check_axis((2,-1), (2,3,4,100,100))
    assert str(exc_info.value) == 'duplicated array axis -1 for shape (2, 3, 4)'

##########################################################################################

def _ramp_image() -> np.ndarray:
    """A 10x10 image where pixel (i,j) has value 100*i + j."""

    return 100 * np.arange(10)[:,np.newaxis] + np.arange(10)


def test_image_to_4d_no_mask() -> None:

    image = _ramp_image()
    footprint = np.ones((3,3), dtype='bool')
    test, mask, weights = _image_to_4d(image, footprint, None, None)

    assert weights is None

    assert tuple(test[0,0][~mask[0,0]]) == (0, 1, 100, 101)
    assert tuple(test[0,1][~mask[0,1]]) == (0, 1, 2, 100, 101, 102)
    assert tuple(test[0,-2][~mask[0,-2]]) == (7, 8, 9, 107, 108, 109)
    assert tuple(test[0,-1][~mask[0,-1]]) == (8, 9, 108, 109)

    assert tuple(test[1,0][~mask[1,0]]) == (0, 1, 100, 101, 200, 201)
    assert tuple(test[1,1][~mask[1,1]]) == (0, 1, 2, 100, 101, 102,
                                                     200, 201, 202)
    assert tuple(test[1,-2][~mask[1,-2]]) == (7, 8, 9, 107, 108, 109,
                                                       207, 208, 209)
    assert tuple(test[1,-1][~mask[1,-1]]) == (8, 9, 108, 109, 208, 209)

    assert tuple(test[2,0][~mask[2,0]]) == (100, 101, 200, 201, 300, 301)
    assert tuple(test[2,1][~mask[2,1]]) == (100, 101, 102, 200, 201, 202,
                                                           300, 301, 302)
    assert tuple(test[2,-2][~mask[2,-2]]) == (107, 108, 109, 207, 208, 209,
                                                             307, 308, 309)
    assert tuple(test[2,-1][~mask[2,-1]]) == (108, 109, 208, 209, 308, 309)

    assert tuple(test[-2,0][~mask[-2,0]]) == (700, 701, 800, 801, 900, 901)
    assert tuple(test[-2,1][~mask[-2,1]]) == (700, 701, 702, 800, 801, 802,
                                                             900, 901, 902)
    assert tuple(test[-2,-2][~mask[-2,-2]]) == (707, 708, 709, 807, 808, 809,
                                                               907, 908, 909)
    assert tuple(test[-2,-1][~mask[-2,-1]]) == (708, 709, 808, 809, 908, 909)

    assert tuple(test[-1,0][~mask[-1,0]]) == (800, 801, 900, 901)
    assert tuple(test[-1,1][~mask[-1,1]]) == (800, 801, 802, 900, 901, 902)
    assert tuple(test[-1,-2][~mask[-1,-2]]) == (807, 808, 809, 907, 908, 909)
    assert tuple(test[-1,-1][~mask[-1,-1]]) == (808, 809, 908, 909)

    assert tuple(test[5,7][~mask[5,7]]) == (406, 407, 408, 506, 507, 508,
                                                           606, 607, 608)
    assert tuple(test[6,7][~mask[6,7]]) == (506, 507, 508, 606, 607, 608,
                                                           706, 707, 708)


def test_image_to_4d_all_unmasked() -> None:

    # An explicit all-False mask matches the no-mask result
    image = _ramp_image()
    footprint = np.ones((3,3), dtype='bool')
    imask = np.zeros(image.shape, dtype='bool')
    test, mask, weights = _image_to_4d(image, footprint, imask, None)

    assert weights is None

    assert tuple(test[0,0][~mask[0,0]]) == (0, 1, 100, 101)
    assert tuple(test[0,-2][~mask[0,-2]]) == (7, 8, 9, 107, 108, 109)
    assert tuple(test[0,-1][~mask[0,-1]]) == (8, 9, 108, 109)
    assert tuple(test[1,1][~mask[1,1]]) == (0, 1, 2, 100, 101, 102,
                                                     200, 201, 202)
    assert tuple(test[1,-1][~mask[1,-1]]) == (8, 9, 108, 109, 208, 209)
    assert tuple(test[2,0][~mask[2,0]]) == (100, 101, 200, 201, 300, 301)
    assert tuple(test[2,-2][~mask[2,-2]]) == (107, 108, 109, 207, 208, 209,
                                                             307, 308, 309)
    assert tuple(test[-2,1][~mask[-2,1]]) == (700, 701, 702, 800, 801, 802,
                                                             900, 901, 902)
    assert tuple(test[-2,-1][~mask[-2,-1]]) == (708, 709, 808, 809, 908, 909)
    assert tuple(test[-1,1][~mask[-1,1]]) == (800, 801, 802, 900, 901, 902)
    assert tuple(test[-1,-1][~mask[-1,-1]]) == (808, 809, 908, 909)
    assert tuple(test[5,7][~mask[5,7]]) == (406, 407, 408, 506, 507, 508,
                                                           606, 607, 608)


def test_image_to_4d_all_masked() -> None:

    # An all-True mask masks every window pixel
    image = _ramp_image()
    footprint = np.ones((3,3), dtype='bool')
    imask = np.ones(image.shape, dtype='bool')
    test, mask, weights = _image_to_4d(image, footprint, imask, None)
    assert weights is None
    assert np.all(mask)


def test_image_to_4d_isolated_mask() -> None:

    image = _ramp_image()
    footprint = np.ones((3,3), dtype='bool')
    imask = np.zeros(image.shape, dtype='bool')
    imask[5,7] = True
    imask[0,0] = True
    test, mask, weights = _image_to_4d(image, footprint, imask, None)

    assert weights is None

    assert tuple(test[0,0][~mask[0,0]]) == (1, 100, 101)
    assert tuple(test[0,1][~mask[0,1]]) == (1, 2, 100, 101, 102)
    assert tuple(test[1,0][~mask[1,0]]) == (1, 100, 101, 200, 201)
    assert tuple(test[1,1][~mask[1,1]]) == (1, 2, 100, 101, 102,
                                                     200, 201, 202)
    assert tuple(test[5,7][~mask[5,7]]) == (406, 407, 408, 506,      508,
                                                           606, 607, 608)
    assert tuple(test[6,7][~mask[6,7]]) == (506,      508, 606, 607, 608,
                                                           706, 707, 708)


def test_image_to_4d_incomplete_footprint() -> None:

    image = _ramp_image()
    imask = np.zeros(image.shape, dtype='bool')
    imask[5,7] = True
    imask[0,0] = True

    footprint = np.ones((3,3), dtype='bool')
    footprint[1,0] = False
    test, mask, weights = _image_to_4d(image, footprint, imask, None)
    assert weights is None
    test = test[..., footprint]
    mask = mask[..., footprint]

    assert tuple(test[0,0][~mask[0,0]]) == (1, 100, 101)
    assert tuple(test[0,1][~mask[0,1]]) == (1, 2, 100, 101, 102)
    assert tuple(test[0,-2][~mask[0,-2]]) == (8, 9, 107, 108, 109)
    assert tuple(test[0,-1][~mask[0,-1]]) == (9, 108, 109)

    assert tuple(test[1,0][~mask[1,0]]) == (1, 100, 101, 200, 201)
    assert tuple(test[1,1][~mask[1,1]]) == (1, 2,      101, 102,
                                                  200, 201, 202)
    assert tuple(test[1,-2][~mask[1,-2]]) == (7, 8, 9,      108, 109,
                                                       207, 208, 209)
    assert tuple(test[1,-1][~mask[1,-1]]) == (8, 9, 109, 208, 209)

    assert tuple(test[2,0][~mask[2,0]]) == (100, 101, 200, 201, 300, 301)
    assert tuple(test[2,1][~mask[2,1]]) == (100, 101, 102,      201, 202,
                                                           300, 301, 302)
    assert tuple(test[2,-2][~mask[2,-2]]) == (107, 108, 109,      208, 209,
                                                             307, 308, 309)
    assert tuple(test[2,-1][~mask[2,-1]]) == (108, 109, 209, 308, 309)

    assert tuple(test[-2,0][~mask[-2,0]]) == (700, 701, 800, 801, 900, 901)
    assert tuple(test[-2,1][~mask[-2,1]]) == (700, 701, 702,      801, 802,
                                                             900, 901, 902)
    assert tuple(test[-2,-2][~mask[-2,-2]]) == (707, 708, 709,      808, 809,
                                                               907, 908, 909)
    assert tuple(test[-2,-1][~mask[-2,-1]]) == (708, 709, 809, 908, 909)

    assert tuple(test[-1,0][~mask[-1,0]]) == (800, 801, 900, 901)
    assert tuple(test[-1,1][~mask[-1,1]]) == (800, 801, 802, 901, 902)
    assert tuple(test[-1,-2][~mask[-1,-2]]) == (807, 808, 809, 908, 909)
    assert tuple(test[-1,-1][~mask[-1,-1]]) == (808, 809, 909)

    assert tuple(test[5,7][~mask[5,7]]) == (406, 407, 408,           508,
                                                           606, 607, 608)
    assert tuple(test[6,7][~mask[6,7]]) == (506, 508, 607,           608,
                                                           706, 707, 708)


def test_image_to_4d_weights() -> None:

    # With weights, the mask output is None and zero weights mark ignored pixels
    image = _ramp_image()
    footprint = np.ones((3,3), dtype='bool')
    wts = np.ones(image.shape, dtype=np.float64)
    wts[0,0] = 0.
    wts[5,7] = 0.

    test, mask, weights = _image_to_4d(image, footprint, None, wts)
    assert mask is None
    assert weights is not None
    assert weights.dtype == np.float64

    # The valid (nonzero weight) values around (0,0) exclude the (0,0) pixel itself
    assert tuple(test[0,0][weights[0,0] > 0]) == (1, 100, 101)
    assert tuple(test[5,7][weights[5,7] > 0]) == (406, 407, 408, 506,      508,
                                                                 606, 607, 608)
    # Off-edge buffer pixels also carry zero weight
    assert np.sum(weights[0,0] > 0) == 3


def test_image_to_4d_3d_image() -> None:

    # Leading axes are preserved by the sliding window
    image = np.stack([_ramp_image(), _ramp_image() + 1000])
    footprint = np.ones((3,3), dtype='bool')
    test, mask, weights = _image_to_4d(image, footprint, None, None)

    assert test.shape == (2, 10, 10, 3, 3)
    assert weights is None
    assert tuple(test[0,0,0][~mask[0,0]]) == (0, 1, 100, 101)
    assert tuple(test[1,0,0][~mask[0,0]]) == (1000, 1001, 1100, 1101)

##########################################################################################

def test_apply_op_as_filter_masked() -> None:

    rng = np.random.default_rng(4484)
    mask = rng.random((100,100)) < 0.6

    # Shrink memory footprint to the minimum, forcing tiling
    _usable_bytes(1)
    try:
        image = rng.random((100,100))
        a, _ = maximum_filter(image, footprint=11, mask=mask)
        assert a.shape == image.shape

        image[mask] = 0.
        b = scipy.ndimage.maximum_filter(image, size=11, mode='constant', cval=0.)
        assert np.all(a == b)

        layers, tiles = _apply_op_as_filter_info()
        assert layers == 1
        assert tiles > 3

        image = rng.random((5,100,100))
        a, _ = maximum_filter(image, footprint=11, mask=mask)
        assert a.shape == image.shape

        image[..., mask] = 0.
        for k in range(5):
            b = scipy.ndimage.maximum_filter(image[k], size=11, mode='constant', cval=0.)
            assert np.all(a[k] == b)

        layers, tiles = _apply_op_as_filter_info()
        assert layers == 5
        assert tiles > 1

        # Set available memory to just less than half what's needed
        nbytes = image.size * 11**2 * image.dtype.itemsize // 2
        _usable_bytes(nbytes)

        image = rng.random((5,100,100))
        a, _ = maximum_filter(image, footprint=11, mask=mask)
        image[..., mask] = 0.
        for k in range(5):
            b = scipy.ndimage.maximum_filter(image[k], size=11, mode='constant', cval=0.)
            assert np.all(a[k] == b)

        layers, tiles = _apply_op_as_filter_info()
        assert layers == 3
        assert tiles == 1
    finally:
        _usable_bytes(0)


def test_apply_op_as_filter_unmasked(shortcuts) -> None:

    # With shortcuts disabled, the unmasked path also exercises _apply_op_as_filter.
    # With shortcuts enabled, scipy handles it directly (so the layer/tile counts do not
    # apply); we only check correctness there.
    rng = np.random.default_rng(4485)

    _usable_bytes(1)
    try:
        image = rng.random((100,100))
        a = maximum_filter(image, footprint=11)
        assert a.shape == image.shape
        b = scipy.ndimage.maximum_filter(image, size=11, mode='constant', cval=0.)
        assert np.all(a == b)

        if not shortcuts:
            layers, tiles = _apply_op_as_filter_info()
            assert layers == 1
            assert tiles > 3

        image = rng.random((5,100,100))
        a = maximum_filter(image, footprint=11)
        assert a.shape == image.shape
        for k in range(5):
            b = scipy.ndimage.maximum_filter(image[k], size=11, mode='constant', cval=0.)
            assert np.all(a[k] == b)

        if not shortcuts:
            layers, tiles = _apply_op_as_filter_info()
            assert layers == 5
            assert tiles > 1

        nbytes = image.size * 11**2 * image.dtype.itemsize // 2
        _usable_bytes(nbytes)
        image = rng.random((5,100,100))
        a = maximum_filter(image, footprint=11)
        for k in range(5):
            b = scipy.ndimage.maximum_filter(image[k], size=11, mode='constant', cval=0.)
            assert np.all(a[k] == b)

        if not shortcuts:
            layers, tiles = _apply_op_as_filter_info()
            assert layers == 3
            assert tiles == 1
    finally:
        _usable_bytes(0)


def test_apply_op_as_filter_default_memory() -> None:

    rng = np.random.default_rng(4486)

    # Force the full path, then restore default memory; everything fits in memory
    _use_shortcuts(False)
    _usable_bytes(0)

    image = rng.random((100,100))
    a = maximum_filter(image, footprint=11)
    assert a.shape == image.shape

    b = scipy.ndimage.maximum_filter(image, size=11, mode='constant', cval=0.)
    assert np.all(a == b)

    layers, tiles = _apply_op_as_filter_info()
    assert layers == 1
    assert tiles == 1


def test_apply_op_as_filter_bad_footprint() -> None:

    # The footprint dtype is validated inside _apply_op_as_filter, so disable the scipy
    # shortcut to reach it.
    _use_shortcuts(False)
    rng = np.random.default_rng(4487)
    image = rng.random((10,10))
    footprint = np.zeros((3,3), dtype='object')
    with pytest.raises(ValueError) as exc_info:
        _ = maximum_filter(image, footprint)
    assert str(exc_info.value) == 'invalid footprint dtype: object'


def test_apply_op_as_filter_weights_tiled() -> None:

    # A weighted filter forces the weights branches of the layered and tiled paths.
    rng = np.random.default_rng(4488)
    _use_shortcuts(False)
    try:
        # Reference against the full (in-memory) computation
        _usable_bytes(0)
        image = rng.random((3,40,40))
        weights = rng.random((3,40,40)) + 0.1
        ref_img, ref_w = maximum_filter(image, footprint=7, weights=weights)

        # Layered path: enough memory per layer but not for the whole stack. The
        # per-layer cost includes both the image and the weights array bytes.
        itemsize = image.dtype.itemsize + weights.dtype.itemsize
        layer_bytes = 40 * 40 * 7**2 * itemsize
        _usable_bytes(layer_bytes)
        a_img, a_w = maximum_filter(image, footprint=7, weights=weights)
        layers, tiles = _apply_op_as_filter_info()
        assert layers > 1
        assert tiles == 1
        assert np.allclose(a_img, ref_img)
        assert np.allclose(a_w, ref_w)

        # Tiled path: not even one layer fits, forcing per-tile processing
        _usable_bytes(1)
        b_img, b_w = maximum_filter(image, footprint=7, weights=weights)
        layers, tiles = _apply_op_as_filter_info()
        assert tiles > 1
        assert np.allclose(b_img, ref_img)
        assert np.allclose(b_w, ref_w)
    finally:
        _usable_bytes(0)


def test_apply_op_as_filter_median_weights_tiled() -> None:

    # Same idea for the weighted median filter, exercising the median op through tiling.
    rng = np.random.default_rng(4489)
    _use_shortcuts(False)
    try:
        _usable_bytes(0)
        image = rng.random((2,30,30))
        weights = rng.random((2,30,30)) + 0.1
        ref_img, ref_w = median_filter(image, footprint=5, weights=weights)

        _usable_bytes(1)
        a_img, a_w = median_filter(image, footprint=5, weights=weights)
        layers, tiles = _apply_op_as_filter_info()
        assert tiles > 1
        assert np.allclose(a_img, ref_img)
        assert np.allclose(a_w, ref_w)
    finally:
        _usable_bytes(0)

##########################################################################################
