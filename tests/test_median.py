##########################################################################################
# tests/test_median.py
##########################################################################################

import numpy as np
import pytest
from psiops.median import median, median_filter


def _omit_image_and_mask() -> tuple[np.ndarray, np.ndarray]:
    """The 3x3x3 image and mask used by several median tests."""

    image = np.array([[[3, 9, 2],
                       [1, 4, 8],
                       [0, 6, 1]],
                      [[9, 2, 2],
                       [9, 9, 3],
                       [8, 1, 2]],
                      [[0, 8, 3],
                       [5, 8, 8],
                       [2, 7, 5]]])
    mask = np.array( [[[0, 1, 0],
                       [0, 1, 0],
                       [1, 1, 0]],
                      [[0, 0, 1],
                       [1, 0, 0],
                       [1, 1, 1]],
                      [[0, 1, 0],
                       [0, 0, 0],
                       [1, 1, 1]]]).astype('bool')
    return image, mask


def test_median_basic() -> None:

    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

    a = median(image)
    assert np.all(a == image0)


def test_median_2d_mask() -> None:

    rng = np.random.default_rng(3863)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

    a = median(image)
    mask = rng.random((10,10)) < 0.3
    b, bmask = median(image, mask=mask)
    assert np.all(mask == bmask)
    assert np.shape(bmask) == (10,10)
    assert np.all(a[~mask] == b[~mask])


def test_median_3d_mask() -> None:

    rng = np.random.default_rng(3863)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

    mask = np.zeros(image.shape, dtype='bool')
    mask[0] = True
    b, bmask = median(image, mask=mask)
    assert np.all(b == 1.5 * image0)
    assert not np.any(bmask)

    mask = rng.random((3,10,10)) < 0.3
    mask[0] = True
    mask[2] = mask[1]
    b, bmask = median(image, mask=mask)
    assert np.all(bmask == mask[1])
    assert np.all((b == 1.5 * image0)[~bmask])


def test_median_axes() -> None:

    rng = np.random.default_rng(3863)
    image = rng.random((3,4,5,10,10))
    a = median(image, axis=1)
    assert a.shape == (3,5,10,10)
    assert np.all(a == np.median(image,axis=1))

    a = median(image, axis=(1,-1))
    assert a.shape == (3,10,10)
    assert np.all(a == np.median(image,axis=(1,2)))

    a = median(image, axis=None)
    assert a.shape == (10,10)
    assert np.all(a == np.median(image,axis=(0,1,2)))

    assert np.all(median(image, axis=(0,-1)) == np.median(image, axis=(0,2)))


def test_median_axes_mask() -> None:

    rng = np.random.default_rng(3863)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7        # mostly masked
    a, amask = median(image, axis=2, mask=mask)
    assert a.shape == (5,4,10,10)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == np.all(mask, axis=2))


def test_median_masked_values() -> None:

    rng = np.random.default_rng(3863)
    image = rng.random((5,4,100,200))
    mask = rng.random((5,4,100,200)) < 0.6        # mostly masked
    a, amask = median(image, axis=0, mask=mask)

    sorted = image.copy()
    sorted[mask] = 10
    sorted = np.sort(sorted, axis=0)

    k = np.sum(np.logical_not(mask), axis=0)
    assert np.all((k==0) == amask)
    assert np.all((a == sorted[0])[k==1])
    assert np.all((a == np.mean(sorted[:2], axis=0))[k==2])
    assert np.all((a == sorted[1])[k==3])
    assert np.all((a == np.mean(sorted[1:3], axis=0))[k==4])
    assert np.all((a == sorted[2])[k==5])


def test_median_dtypes() -> None:

    for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                  'int64', 'float32', 'float64'):
        image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
        test = median(image.astype(dtype))
        if dtype == 'float32':
            assert test.dtype == np.float32
        else:
            assert test.dtype == np.float64


def test_median_list_input() -> None:

    rng = np.random.default_rng(3863)
    image = rng.random((3,5,7,10,10))
    assert np.all(median(image) == np.median(image, axis=(0,1,2)))
    assert np.all(median(image, axis=1) == np.median(image, axis=1))
    assert np.all(median(image, axis=(0,-1)) == np.median(image, axis=(0,2)))

    image = list(image)     # works for non-arrays?
    assert np.all(median(image) == np.median(image, axis=(0,1,2)))


def test_median_keepdims() -> None:

    # keepdims is only honored on the multi-return path (see report on
    # `_validation._check_return`).
    rng = np.random.default_rng(41)
    image = rng.random((3,4,5,10,10))
    mask = rng.random((10,10)) < 0.3
    a, amask = median(image, axis=1, keepdims=True, mask=mask)
    assert a.shape == (3,1,5,10,10)
    assert amask.shape == (3,1,5,10,10)
    b = median(image, axis=1, mask=mask)[0]
    assert np.all(a[:,0] == b)


def test_median_error() -> None:

    image = np.arange(12).reshape(4,3)
    with pytest.raises(ValueError) as exc_info:
        _ = median(image)
    assert str(exc_info.value) == 'invalid image shape (4, 3); must be at least 3-D'


def test_median_factors() -> None:

    rng = np.random.default_rng(3863)
    image = rng.random((5,10,10))
    image[3:] = image[2]    # last layer repeat three times
    assert np.all(median(image) == image[2])

    test = median(image[:3], factors=[1,1,2.1])
    assert np.all(test == image[2])

    test = median(image[:3], factors=[1,1,2])
    test2 = median(image[:4])
    assert np.all(test == test2)


def test_median_masked_factors() -> None:

    image, mask = _omit_image_and_mask()

    test, tmask = median(image, mask)
    assert np.all(test == np.array([[3. , 2. , 2.5],
                                    [3. , 8.5, 8. ],
                                    [0. , 0. , 1. ]]))
    assert np.all(tmask == np.array([[0 , 0  , 0  ],
                                     [0 , 0  , 0  ],
                                     [1 , 1  , 0  ]]).astype('bool'))

    image2 = np.empty((7,3,3)).astype('int')
    image2[:3]  = image
    image2[3:6] = image
    image2[6]   = image[0]

    mask2 = np.empty((7,3,3)).astype('bool')
    mask2[:3]  = mask
    mask2[3:6] = mask
    mask2[6]   = mask[0]

    test1, tmask1 = median(image, mask, factors=[1.5,1,1])
    test2, tmask2 = median(image2, mask2)
    assert np.all(test1 == test2)
    assert np.all(tmask1 == tmask2)


def test_median_omit() -> None:

    image, mask = _omit_image_and_mask()

    image2 = np.empty((7,3,3)).astype('int')
    image2[:3]  = image
    image2[3:6] = image
    image2[6]   = image[0]

    mask2 = np.empty((7,3,3)).astype('bool')
    mask2[:3]  = mask
    mask2[3:6] = mask
    mask2[6]   = mask[0]

    test2, tmask2 = median(image2, mask2)

    # Omit the single largest unmasked value (which we inject as a huge value)
    image3 = np.empty((8,3,3)).astype('int')
    image3[:7] = image2
    image3[-1] = 99999999

    mask3 = np.empty((8,3,3)).astype('bool')
    mask3[:7] = mask2
    mask3[-1] = False

    test3, tmask3 = median(image3, mask3, omit=-1)
    assert np.all(test3 == test2)
    assert np.all(tmask3 == tmask2)

    # Omit the single smallest unmasked value (injected as a hugely negative value)
    image3[-1] = -99999999
    test3, tmask3 = median(image3, mask3, omit=1)
    assert np.all(test3 == test2)
    assert np.all(tmask3 == tmask2)


def test_median_omit_unmasked() -> None:

    # The omit path with no mask uses unit weights for every pixel.
    rng = np.random.default_rng(43)
    image = rng.random((5,8,8))

    a = median(image, omit=-1)
    ref = np.median(np.sort(image, axis=0)[:4], axis=0)
    assert np.allclose(a, ref)

    a = median(image, omit=1)
    ref = np.median(np.sort(image, axis=0)[1:], axis=0)
    assert np.allclose(a, ref)


def test_median_weights() -> None:

    rng = np.random.default_rng(45)
    image = rng.random((4,10,10))
    weights = rng.random((4,10,10)) + 0.1

    a, aw = median(image, weights=weights)
    assert np.allclose(aw, np.sum(weights, axis=0))

    # A weight far larger than the rest pins the median to that layer's value
    weights2 = np.ones((3,6,6))
    weights2[2] = 100.
    image2 = rng.random((3,6,6))
    a2 = median(image2, weights=weights2)[0]
    assert np.allclose(a2, image2[2])


def test_median_maskval() -> None:

    image = np.full((3,8,8), 5.)
    image[0,0,0] = 2.
    image[1,0,0] = 8.
    a, amask = median(image, maskval=5., returns='im')
    assert a[0,0] == 5.       # median of [2, 8]
    assert np.all(amask[1:,1:])
    assert np.all(a[1:,1:] == 5.)   # fully masked -> filled with maskval


def test_median_maskedarray() -> None:

    rng = np.random.default_rng(47)
    data = rng.random((3,10,10))
    mask = rng.random((3,10,10)) < 0.3
    ma = np.ma.MaskedArray(data, mask=mask)
    a = median(ma)
    assert isinstance(a, np.ma.MaskedArray)
    assert np.all(a.mask == np.all(mask, axis=0))


def test_median_masked_weights_return() -> None:

    # The masked nanmedian shortcut returns per-pixel weights when 'w' is requested.
    rng = np.random.default_rng(49)
    image = rng.random((4,10,10))
    mask = rng.random((4,10,10)) < 0.3
    a, aw = median(image, mask=mask, returns='iw')
    assert np.all(aw == np.sum(np.logical_not(mask), axis=0))

    a, am, aw = median(image, mask=mask, returns='imw')
    assert np.all(am == np.all(mask, axis=0))
    assert np.all(aw == np.sum(np.logical_not(mask), axis=0))

##########################################################################################

def test_median_filter_no_mask(shortcuts) -> None:

    image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
    a = median_filter(image, 2)
    b = np.empty((4,10,10))
    b[:,0,0] = image[:,0,0]
    b[:,0,1:] = (image[:,0,:-1] + image[:,0,1:]) / 2.
    b[:,1:,0] = (image[:,:-1,0] + image[:,1:,0]) / 2.
    for i in range(1,10):
        for j in range(1,10):
            b[:,i,j] = np.median(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1))
    assert np.abs(a - b).max() < 1.e-15


def test_median_filter_mask(shortcuts) -> None:

    rng = np.random.default_rng(8063)
    image = rng.random((100,100))
    mask = rng.random((100,100)) < 0.6
    footprint = np.ones((3,3), dtype='bool')
    footprint[0,0] = False
    footprint[0,2] = False
    footprint[2,0] = False
    a, amask = median_filter(image, footprint=footprint, mask=mask)

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
            assert np.abs(a[i,j] - np.median(values)) < 1.e-14


def test_median_filter_footprint_forms(shortcuts) -> None:

    rng = np.random.default_rng(21)
    image = rng.random((20,20))

    a_int = median_filter(image, 3)
    a_tuple = median_filter(image, (3,3))
    a_bool = median_filter(image, np.ones((3,3), dtype='bool'))

    assert np.allclose(a_int, a_tuple)
    assert np.allclose(a_int, a_bool)


def test_median_filter_omit(shortcuts) -> None:

    rng = np.random.default_rng(51)
    image = rng.random((10,10))
    # An omit-trimmed median filter still returns the right shape and stays finite
    a = median_filter(image, 3, omit=1)
    assert a.shape == image.shape
    assert np.all(np.isfinite(a))

##########################################################################################
