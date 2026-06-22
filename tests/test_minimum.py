##########################################################################################
# tests/test_minimum.py
##########################################################################################

import numpy as np
import pytest

from psiops.minimum import minimum, minimum_filter


def test_minimum_basic() -> None:

    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([3,-1,-2])[:,np.newaxis,np.newaxis]

    a = minimum(image)
    assert np.all(a == -2 * image0)


def test_minimum_2d_mask() -> None:

    rng = np.random.default_rng(3711)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([3,-1,-2])[:,np.newaxis,np.newaxis]

    a = minimum(image)
    mask = rng.random((10,10)) < 0.3
    b, bmask = minimum(image, mask=mask)
    assert np.all(mask == bmask)
    assert np.shape(bmask) == (10,10)
    assert np.all(a[~mask] == b[~mask])


def test_minimum_3d_mask() -> None:

    rng = np.random.default_rng(3711)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([3,-1,-2])[:,np.newaxis,np.newaxis]

    mask = np.zeros(image.shape, dtype='bool')
    mask[2] = True
    b, bmask = minimum(image, mask=mask)
    assert np.all(b == -image0)
    assert not np.any(bmask)

    mask = rng.random((3,10,10)) < 0.3
    mask[2] = True
    mask[1] = mask[0]
    b, bmask = minimum(image, mask=mask)
    assert np.all(bmask == mask[1])
    assert np.all((b == -image0)[~bmask])

    mask = rng.random((3,10,10)) < 0.3
    mask[0] = False
    mask[1] = True
    b, bmask = minimum(image, mask=mask)
    assert not np.any(bmask)
    assert np.all((b == -2 * image0)[~mask[2]])
    assert np.all((b == 3 * image0)[mask[2]])


def test_minimum_axes() -> None:

    rng = np.random.default_rng(3711)
    image = rng.random((3,4,5,10,10))
    a = minimum(image, axis=1)
    assert a.shape == (3,5,10,10)
    assert np.all(a == np.min(image,axis=1))

    a = minimum(image, axis=(1,-1))
    assert a.shape == (3,10,10)
    assert np.all(a == np.min(image,axis=(1,2)))

    a = minimum(image, axis=None)
    assert a.shape == (10,10)
    assert np.all(a == np.min(image,axis=(0,1,2)))

    image = rng.random((3,4,5,10,10))     # range is 0 to 1
    image[:,0] -= 2                        # range is -2 to -1 at this axis
    a = minimum(image, axis=1)
    assert np.all(a == image[:,0])


def test_minimum_keepdims() -> None:

    # keepdims is only honored on the multi-return path (see report on
    # `_validation._check_return`).
    rng = np.random.default_rng(31)
    image = rng.random((3,4,5,10,10))
    mask = rng.random((10,10)) < 0.3
    a, amask = minimum(image, axis=1, keepdims=True, mask=mask)
    assert a.shape == (3,1,5,10,10)
    assert amask.shape == (3,1,5,10,10)
    b = minimum(image, axis=1, mask=mask)[0]
    assert np.all(a[:,0] == b)


def test_minimum_axes_mask() -> None:

    rng = np.random.default_rng(3711)
    image = rng.random((3,4,5,10,10))
    image[:,0] -= 2                        # range is -2 to -1 at this axis
    mask = rng.random((10,10)) < 0.3
    a, amask = minimum(image, axis=1, mask=mask)
    b = minimum(image, axis=1)
    assert a.shape == (3,5,10,10)
    assert amask.shape == (3,5,10,10)
    assert np.all(amask == mask)
    assert np.all(a[~amask] == b[~amask])

    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.9        # mostly masked
    a, amask = minimum(image, axis=2, mask=mask)
    assert a.shape == (5,4,10,10)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == np.all(mask, axis=2))

    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.2
    a, amask = minimum(image, axis=(0,2), mask=mask)
    image[mask] = 2
    b = minimum(image, axis=(0,-1))
    assert np.all(a[~amask] == b[~amask])


def test_minimum_dtypes() -> None:

    image0 = (np.arange(10) + np.arange(10)[:,np.newaxis]
                            + np.arange(10)[:,np.newaxis,np.newaxis])
    for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                  'int64', 'float32', 'float64'):
        image = image0.astype(dtype)
        test = minimum(image)
        assert np.all(test == np.min(image,axis=0))
        assert test.dtype == image.dtype

        if image.dtype.kind == 'i' and dtype != 'int64':
            maxval = np.ma.minimum_fill_value(image)
            image[0,0,0] = maxval
            assert np.all(minimum(image) == np.min(image,axis=0))


def test_minimum_dtypes_masked() -> None:

    # Masked path across the special-case dtypes (bool, small/large ints, floats)
    rng = np.random.default_rng(7)
    image0 = (np.arange(10) + np.arange(10)[:,np.newaxis]
                            + np.arange(10)[:,np.newaxis,np.newaxis])
    mask = rng.random((10,10,10)) < 0.3
    mask[:, 0, 0] = True            # one fully masked pixel column
    for dtype in ('bool', 'uint8', 'int8', 'int64', 'float32', 'float64'):
        image = image0.astype(dtype)
        a, amask = minimum(image, mask=mask)
        assert a.dtype == image.dtype

        floats = image.astype(np.float64)
        floats[mask] = np.inf
        b = np.min(floats, axis=0)
        assert np.all(amask == np.all(mask, axis=0))
        assert np.all(a.astype(np.float64)[~amask] == b[~amask])


def test_minimum_masked_list_input() -> None:

    # A list input is converted to an array (image_is_copy=True), so the masked path skips
    # the defensive copy.
    image = [[[1., 9.], [3., 4.]], [[5., 6.], [7., 2.]]]
    mask = np.array([[[True, False], [False, True]],
                     [[False, True], [False, False]]])
    a, amask = minimum(image, mask=mask)
    assert a[0,0] == 5.     # layer 0 masked -> only layer 1 = 5
    assert a[0,1] == 9.     # layer 1 masked -> only layer 0 = 9
    assert a[1,0] == 3.     # neither masked -> min(3, 7)
    assert a[1,1] == 2.     # layer 0 masked -> only layer 1 = 2
    assert not np.any(amask)


def test_minimum_masked_fill_value_int() -> None:

    # Masked path where an unmasked value equals the type's fill value (its maximum),
    # exercising both the promote-to-int64 branch (small ints) and the int64 fallback.
    mask = np.zeros((3,4,4), dtype='bool')
    mask[0] = True                          # mask the first layer everywhere

    for dtype in ('int8', 'int16', 'int32'):
        info = np.iinfo(dtype)
        image = np.zeros((3,4,4), dtype=dtype)
        image[1] = info.max                 # unmasked value at the type maximum
        image[2] = 7
        a, amask = minimum(image, mask=mask)
        assert a.dtype == np.dtype(dtype)
        assert np.all(a == 7)               # min over unmasked layers 1 and 2
        assert not np.any(amask)

    # int64: cannot promote further, so the fallback `ignore = maxval` branch runs
    image = np.zeros((3,4,4), dtype='int64')
    image[1] = np.iinfo('int64').max
    image[2] = 7
    a, amask = minimum(image, mask=mask)
    assert a.dtype == np.dtype('int64')
    assert np.all(a == 7)


def test_minimum_weights() -> None:

    rng = np.random.default_rng(11)
    image = rng.random((3,10,10))
    w = rng.random((3,10,10)) + 0.1

    _, aw = minimum(image, weights=w)
    assert np.allclose(aw, np.sum(w, axis=0))

    # Weight of zero acts like a mask
    w2 = w.copy()
    w2[0] = 0.
    a2, _ = minimum(image, weights=w2)
    expected = np.min(np.where(w2 == 0, np.inf, image), axis=0)
    assert np.all(a2 == expected)


def test_minimum_maskval() -> None:

    image = np.full((3,10,10), 5)
    image[0,0,0] = 2
    a, amask = minimum(image, maskval=5, returns='im')
    assert a[0,0] == 2
    assert np.all(a[1:,1:] == 5)
    assert np.all(amask[1:,1:])


def test_minimum_maskedarray() -> None:

    rng = np.random.default_rng(13)
    data = rng.random((3,10,10))
    mask = rng.random((3,10,10)) < 0.3
    ma = np.ma.MaskedArray(data, mask=mask)
    a = minimum(ma)
    assert isinstance(a, np.ma.MaskedArray)
    assert np.all(a.mask == np.all(mask, axis=0))


def test_minimum_returns_variants() -> None:

    rng = np.random.default_rng(17)
    image = rng.random((3,10,10))
    mask = rng.random((10,10)) < 0.3

    assert isinstance(minimum(image, returns='i'), np.ndarray)
    assert len(minimum(image, mask=mask, returns='im')) == 2
    assert len(minimum(image, mask=mask, returns='iw')) == 2
    assert len(minimum(image, mask=mask, returns='imw')) == 3


def test_minimum_list_input() -> None:

    rng = np.random.default_rng(3711)
    image = rng.random((5,4,3,10,10))
    assert np.all(minimum(image) == np.min(image, axis=(0,1,2)))
    assert np.all(minimum(image, axis=1) == np.min(image, axis=1))
    assert np.all(minimum(image, axis=(0,-1)) == np.min(image, axis=(0,2)))

    image_list = list(image)     # works for non-arrays?
    assert np.all(minimum(image_list) == np.min(image_list, axis=(0,1,2)))


def test_minimum_errors() -> None:

    exc_info: pytest.ExceptionInfo[Exception]
    image: np.ndarray = np.arange(12).reshape(4,3)
    with pytest.raises(ValueError) as exc_info:
        _ = minimum(image)
    assert str(exc_info.value) == 'invalid image shape (4, 3); must be at least 3-D'

    image = np.array([None, 1, 4., 'str', np.dtype('float'), None, None, None])
    image = image.reshape(2,2,2)
    with pytest.raises(TypeError) as exc_info:
        _ = minimum(image)
    assert str(exc_info.value) == 'image dtype object is not numeric'

##########################################################################################

def test_minimum_filter_no_mask(shortcuts: bool) -> None:

    image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
    a = minimum_filter(image, 2, returns='i')
    b = np.empty((4,10,10))
    b[:,0,0] = image[:,0,0]
    b[:,0,1:] = np.minimum(image[:,0,:-1], image[:,0,1:])
    b[:,1:,0] = np.minimum(image[:,:-1,0], image[:,1:,0])
    for i in range(1,10):
        for j in range(1,10):
            b[:,i,j] = np.min(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1))
    assert np.abs(a - b).max() < 1.e-15


def test_minimum_filter_mask(shortcuts: bool) -> None:

    rng = np.random.default_rng(8063)
    image = rng.random((100,100))
    mask = rng.random((100,100)) < 0.6
    footprint = np.ones((3,3), dtype='bool')
    footprint[0,0] = False
    footprint[0,2] = False
    footprint[2,0] = False
    a, amask = minimum_filter(image, footprint=footprint, mask=mask)

    for i in range(100):
        islice1 = slice(max(i-1, 0), min(i+2, 100))
        islice2 = slice(1 if i==0 else 0, 2 if i==99 else 3)
        for j in range(100):
            jslice1 = slice(max(j-1, 0), min(j+2, 100))
            jslice2 = slice(1 if j==0 else 0, 2 if j==99 else 3)
            values = image[islice1,jslice1]
            vmask = mask[islice1,jslice1]

            fp = footprint[islice2,jslice2]
            values = values[~vmask & fp]
            if len(values) == 0:
                assert amask[i,j]
                continue

            assert not amask[i,j]
            assert np.abs(a[i,j] - np.min(values)) < 1.e-14


def test_minimum_filter_footprint_forms(shortcuts: bool) -> None:

    rng = np.random.default_rng(21)
    image = rng.random((20,20))

    a_int = minimum_filter(image, 3, returns='i')
    a_tuple = minimum_filter(image, (3,3), returns='i')
    a_bool = minimum_filter(image, np.ones((3,3), dtype='bool'), returns='i')

    assert np.all(a_int == a_tuple)
    assert np.all(a_int == a_bool)


def test_minimum_filter_weights(shortcuts: bool) -> None:

    rng = np.random.default_rng(23)
    image = rng.random((30,30))
    weights = rng.random((30,30)) + 0.1
    a, aw = minimum_filter(image, 3, weights=weights)
    assert a.shape == image.shape
    assert aw.shape == image.shape
    assert np.isclose(aw[10,10], np.sum(weights[9:12, 9:12]))


def test_minimum_zero_size_raises(shortcuts: bool) -> None:
    # A reduction over a zero-size array is undefined and must raise, not NaN.
    empty = np.ones((0, 4, 4))
    with pytest.raises(ValueError, match='size cannot be zero'):
        minimum(empty)
    with pytest.raises(ValueError, match='size cannot be zero'):
        minimum(empty, mask=np.zeros((0, 4, 4), dtype=bool))

##########################################################################################
