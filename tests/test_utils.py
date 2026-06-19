##########################################################################################
# tests/test_utils.py
##########################################################################################

import numpy as np
import pytest
import scipy.ndimage

from psiops.maximum import maximum_filter
from psiops._filter import _apply_op_as_filter_info, _image_to_4d, _usable_bytes
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

##########################################################################################

def test_check_axis() -> None:

    assert _check_axis(None, (2,3,4,5,100,100)) == (0,1,2,3)
    assert _check_axis(1, (2,3,4,5,100,100)) == (1,)
    assert _check_axis(-1, (2,3,4,5,100,100)) == (3,)
    assert _check_axis((3,2,1), (2,3,4,5,100,100)) == (3,2,1)
    assert _check_axis([3,1,2], (2,3,4,5,100,100)) == (3,1,2)
    assert _check_axis(np.arange(1,4), (2,3,4,5,100,100)) == (1,2,3)

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

def test_image_to_4d() -> None:

    # No mask
    image = 100 * np.arange(10)[:,np.newaxis] + np.arange(10)
    footprint = np.ones((3,3), dtype='bool')
    test, mask = _image_to_4d(image, footprint)

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

    # mask is False, spot-checks
    test, mask = _image_to_4d(image, footprint, mask=False)

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

    # mask is True
    test, mask = _image_to_4d(image, footprint, mask=True)
    assert np.all(mask)

    # Mask of isolated samples
    imask = np.zeros(image.shape, dtype='bool')
    imask[5,7] = True
    imask[0,0] = True
    test, mask = _image_to_4d(image, footprint, imask)

    assert tuple(test[0,0][~mask[0,0]]) == (1, 100, 101)
    assert tuple(test[0,1][~mask[0,1]]) == (1, 2, 100, 101, 102)

    assert tuple(test[1,0][~mask[1,0]]) == (1, 100, 101, 200, 201)
    assert tuple(test[1,1][~mask[1,1]]) == (1, 2, 100, 101, 102,
                                                     200, 201, 202)

    assert tuple(test[5,7][~mask[5,7]]) == (406, 407, 408, 506,      508,
                                                           606, 607, 608)
    assert tuple(test[6,7][~mask[6,7]]) == (506,      508, 606, 607, 608,
                                                           706, 707, 708)

    # Incomplete footprint
    footprint = np.ones((3,3), dtype='bool')
    footprint[1,0] = False
    test, mask = _image_to_4d(image, footprint, imask)
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

##########################################################################################

def test_apply_op_as_filter() -> None:

    rng = np.random.default_rng(4484)

    # With and without mask...

    for mask in (None, rng.random((100,100)) < 0.6):

        # Shrink memory footprint to the minimum
        _usable_bytes(1)

        image = rng.random((100,100))
        a = maximum_filter(image, footprint=11, mask=mask)
        if mask is not None:
            a = a[0]
        assert a.shape == image.shape

        if mask is not None:
            image[mask] = 0.
        b = scipy.ndimage.maximum_filter(image, size=11, mode='constant', cval=0.)
        assert np.all(a == b)

        layers, tiles = _apply_op_as_filter_info()
        assert layers == 1
        assert tiles > 3

        image = rng.random((5,100,100))
        a = maximum_filter(image, footprint=11, mask=mask)
        if mask is not None:
            a = a[0]
        assert a.shape == image.shape

        if mask is not None:
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
        a = maximum_filter(image, footprint=11, mask=mask)
        if mask is not None:
            a = a[0]

        if mask is not None:
            image[..., mask] = 0.
        for k in range(5):
            b = scipy.ndimage.maximum_filter(image[k], size=11, mode='constant', cval=0.)
            assert np.all(a[k] == b)

        layers, tiles = _apply_op_as_filter_info()
        assert layers == 3
        assert tiles == 1

    # Restore default behavior
    _usable_bytes(0)

    image = rng.random((100,100))
    a = maximum_filter(image, footprint=11)
    assert a.shape == image.shape

    b = scipy.ndimage.maximum_filter(image, size=11, mode='constant', cval=0.)
    assert np.all(a == b)

    layers, tiles = _apply_op_as_filter_info()
    assert layers == 1
    assert tiles == 1

    # Weird input
    image = rng.random((10,10))
    footprint = np.zeros((3,3), dtype='object')
    with pytest.raises(ValueError) as exc_info:
        _ = maximum_filter(image, footprint)
    assert str(exc_info.value) == 'invalid footprint dtype: object'

##########################################################################################
