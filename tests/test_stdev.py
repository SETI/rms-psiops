##########################################################################################
# tests/test_stdev.py
##########################################################################################

import numpy as np
import pytest
from psiops.stdev import stdev, stdev_filter, variance_filter


def test_stdev() -> None:

    rng = np.random.default_rng(3534)

    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]

    a = stdev(image)
    assert np.abs(a - image0).max() < 1.e-15

    # 2-D mask
    mask = rng.random((10,10)) < 0.3
    b, bmask = stdev(image, mask=mask)
    assert np.all(mask == bmask)
    assert np.shape(bmask) == (10,10)
    assert np.all(a[~mask] == b[~mask])

    # 3-D mask
    mask = np.zeros(image.shape, dtype='bool')
    mask[0] = True
    b, bmask = stdev(image, mask=mask)
    assert np.abs(b - np.sqrt(0.5) * image0).max() < 1.e-14
    assert not np.any(bmask)

    mask = rng.random((3,10,10)) < 0.3
    mask[0] = True
    mask[2] = mask[1]
    b, bmask = stdev(image, mask=mask)
    assert np.all(bmask == mask[1])
    assert np.abs(b - np.sqrt(0.5) * image0)[~bmask].max() < 1.e-14

    mask = rng.random((3,10,10)) < 0.3
    mask[1] = False
    mask[2] = False
    b, bmask = stdev(image, mask=mask)
    assert not np.any(bmask)
    assert np.abs(b - image0)[~mask[0]].max() < 1.e-14
    assert np.abs(b - np.sqrt(0.5) * image0)[mask[0]].max() < 1.e-14

    # More complicated shapes
    image = rng.random((3,4,5,10,10))
    a = stdev(image, axis=1)
    assert a.shape == (3,5,10,10)
    assert np.abs(a - np.std(image,axis=1,ddof=1)).max() < 1.e-15

    a = stdev(image, axis=(1,-1))
    assert a.shape == (3,10,10)
    assert np.abs(a - np.std(image,axis=(1,2),ddof=1)).max() < 1.e-15

    a = stdev(image, axis=None)
    assert a.shape == (10,10)
    assert np.abs(a - np.std(image,axis=(0,1,2),ddof=1)).max() < 1.e-15

    # More complicated shape, axes, mask
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7
    a, amask = stdev(image, axis=2, mask=mask)
    b = stdev(image, axis=-1, stdtype='biased')
    assert a.shape == (5,4,10,10)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == (np.sum(np.logical_not(mask), axis=2) == 0))

    b = stdev(image, axis=-1, stdtype='unbiased')
    assert a.shape == (5,4,10,10)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == (np.sum(np.logical_not(mask), axis=2) < 2))

    image = rng.random((5,4,100,200))
    mask = rng.random((5,4,100,200)) < 0.6        # mostly masked
    a, amask = stdev(image, axis=0, mask=mask)

    sorted = image.copy()
    sorted[mask] = 10
    sorted = np.sort(sorted, axis=0)

    k = np.sum(np.logical_not(mask), axis=0)
    assert np.all(((k==0) | (k==1)) == amask)
    assert np.abs(a - np.std(sorted[:2], axis=0, ddof=1))[k==2].max() < 1.e-14
    assert np.abs(a - np.std(sorted[:3], axis=0, ddof=1))[k==3].max() < 1.e-14
    assert np.abs(a - np.std(sorted[:4], axis=0, ddof=1))[k==4].max() < 1.e-14
    assert np.abs(a - np.std(sorted[:5], axis=0, ddof=1))[k==5].max() < 1.e-14

    # Weights, stdtype='frequency', no mask

    # See https://www.itl.nist.gov/div898/software/dataplot/refman2/ch2/weightsd.pdf
    # Example from p. 2-67, correct value is 5.82
    values = np.array([2,3,5,7,11,13,17,19,23])
    factors = np.array([1,1,0,0,4,1,2,1,0])
    image = np.empty((9,3,3))
    image[...] = values[:,np.newaxis,np.newaxis]
    a = stdev(image, factors=factors)
    assert abs(a[0,0] - 5.82) < 0.005

    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2]).reshape(5,1,1)
    a = stdev(image[:5], axis=0, factors=factors)
    b = stdev(image, axis=0)
    wlen = factors.size
    wsum = np.sum(factors)
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.abs(a - b * factor).max() < 1.e-15

    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([0,0,1,1,2]).reshape(5,1,1)      # check zero weights
    a = stdev(image[:5], axis=0, factors=factors)
    b = stdev(image[2:], axis=0)
    wlen = np.sum(factors != 0)
    wsum = np.sum(factors)
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.abs(a - b * factor).max() < 1.e-15

    image = rng.random((5,2,6,10,10))
    image[:,0,4] = image[:,0,3]
    image[:,0,5] = image[:,0,3]
    image[:,1,4] = image[:,1,0]
    image[:,1,5] = image[:,1,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]])
    a = stdev(image[:,:,:4], axis=(1,2), factors=factors)
    b = stdev(image, axis=(1,2))
    wlen = factors.size
    wsum = np.sum(factors)
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.abs(a - b * factor).max() < 1.e-15

    a = stdev(image[:,:,:4], axis=2, factors=factors)
    b = stdev(image, axis=2)
    wlen = factors.shape[-1]
    wsum = np.sum(factors) // 2
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.abs(a - b * factor).max() < 1.e-15

    image = rng.random((2,5,6,10,10))
    image[0,:,4] = image[0,:,3]
    image[0,:,5] = image[0,:,3]
    image[1,:,4] = image[1,:,0]
    image[1,:,5] = image[1,:,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,1,4)
    a = stdev(image[:,:,:4], axis=(0,2), factors=factors)
    b = stdev(image, axis=(0,2))
    wlen = factors.size
    wsum = np.sum(factors)
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.abs(a - b * factor).max() < 1.e-15

    a = stdev(image[:,:,:4], axis=2, factors=factors)
    b = stdev(image, axis=2)
    wlen = factors.shape[-1]
    wsum = np.sum(factors) // 2
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.abs(a - b * factor).max() < 1.e-15

    image = rng.random((2,6,5,10,10))
    image[0,4] = image[0,3]
    image[0,5] = image[0,3]
    image[1,4] = image[1,0]
    image[1,5] = image[1,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,4,1)
    a = stdev(image[:,:4], axis=(0,1), factors=factors)
    b = stdev(image, axis=(0,1))
    wlen = factors.size
    wsum = np.sum(factors)
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.abs(a - b * factor).max() < 1.e-15

#     a = stdev(image[:,:4], axis=1, factors=factors)
#     b = stdev(image, axis=1)
#     wlen = factors.shape[-1]
#     wsum = np.sum(factors) // 2
#     factor = np.sqrt(wlen/wsum * (wsum-1)/np.maximum(wlen-1,1))
#     assert np.abs(a - b * factor).max() < 1.e-15

    # Weights, stdtype='reliability', no mask
    image = rng.random((8,4,4))
    factors = np.array([1,0,1,1,2,3,0,2])
    a = stdev(image, axis=0, factors=factors, stdtype='reliability')
    b = np.empty((4,4))
    for i in range(4):
        for j in range(4):
            b[i,j] = np.sqrt(np.cov(image[:,i,j], aweights=factors))
    assert np.abs(a - b).max() < 1.e-15

    image = rng.random((8,4,4))
    factors = np.array([1,1,2,1,1,0,1,20])
    a = stdev(image, axis=0, factors=factors, stdtype='reliability')
    b = np.empty((4,4))
    for i in range(4):
        for j in range(4):
            b[i,j] = np.sqrt(np.cov(image[:,i,j], aweights=factors))
    assert np.abs(a - b).max() < 1.e-15

    # Weights, mask
    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2]).reshape(5,1,1)
    mask = rng.random((10,10)) < 0.3
    a, amask = stdev(image[:5], axis=0, factors=factors, mask=mask)
    b, bmask = stdev(image, axis=0, mask=mask)
    c = stdev(image, axis=0)
    wlen = factors.size
    wsum = np.sum(factors)
    factor = np.sqrt(wlen/wsum * (wsum-1)/(wlen-1))
    assert np.all(amask == mask)
    assert np.all(amask == bmask)
    assert np.abs(a - b * factor)[~amask].max() < 1.e-15
    assert np.abs(a - c * factor)[~amask].max() < 1.e-15

    image = rng.random((8,10,10))
    factors = np.array([1,1,2,1,1,0,1,20])
    mask = rng.random((8,10,10)) < 0.3
    mask[:] = mask[0]
    mask[1:3] = True
    a, amask = stdev(image, factors=factors, mask=mask)
    b = stdev(image, factors=[1,0,0,1,1,0,1,20])
    assert np.abs(a - b)[~amask].max() < 1.e-15

    # Check dtypes
    for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                  'int64', 'float32', 'float64'):
        image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
        test = stdev(image.astype(dtype))
        if dtype == 'float32':
            assert test.dtype == np.float32
        else:
            assert test.dtype == np.float64

    image = rng.random((2,3,4,10,10))
    assert np.abs(stdev(image)
                  - np.std(image, axis=(0,1,2), ddof=1)).max() < 1.e-14
    assert np.abs(stdev(image, axis=1)
                  - np.std(image, axis=1, ddof=1)).max() < 1.e-14
    assert np.abs(stdev(image, axis=(0,-1))
                  - np.std(image, axis=(0,2), ddof=1)).max() < 1.e-14

    # Check dtypes
    for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                  'int64', 'float32', 'float64'):
        image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
        test = stdev(image.astype(dtype))
        if dtype == 'float32':
            assert test.dtype == np.float32
        else:
            assert test.dtype == np.float64

    # Weird input
    image = rng.random((2,3,4,10,10))
    image = list(image)     # works for non-arrays?
    assert np.abs(stdev(image)
                  - np.std(image, axis=(0,1,2), ddof=1)).max() < 1.e-14

    with pytest.raises(ValueError) as exc_info:
        _ = stdev(image, stdtype='huh?')
    assert str(exc_info.value) == "unrecognized value for wtype: 'huh?'"

    image = np.arange(12).reshape(4,3)
    with pytest.raises(ValueError) as exc_info:
        _ = stdev(image)
    assert str(exc_info.value) == 'illegal image shape; ndim >= 3 required: (4, 3)'

##########################################################################################

def test_stdev_filter() -> None:

    rng = np.random.default_rng(8063)

    # No mask
    image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
    a = stdev_filter(image, 2)
    b = np.empty((4,10,10))
    b[:,0,0] = 0
    b[:,0,1:] = np.abs(image[:,0,:-1] - image[:,0,1:]) / np.sqrt(2)
    b[:,1:,0] = np.abs(image[:,:-1,0] - image[:,1:,0]) / np.sqrt(2)
    for i in range(1,10):
        for j in range(1,10):
            b[:,i,j] = np.std(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1), ddof=1)
    assert np.abs(a - b).max() < 1.e-14

    # Mask, irregular footprint
    image = rng.random((100,100))
    mask = rng.random((100,100)) < 0.4
    footprint = np.ones((3,3), dtype='bool')
    footprint[0,0] = False
    footprint[0,2] = False
    footprint[2,0] = False
    a, amask = stdev_filter(image, footprint=footprint, mask=mask)
    b, bmask = variance_filter(image, footprint=footprint, mask=mask)

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
            if len(values) < 2:
                assert amask[i,j]
                assert bmask[i,j]
                continue

            assert not amask[i,j]
            assert not bmask[i,j]
            assert np.abs(a[i,j] - np.std(values, ddof=1)) < 1.e-14
            assert np.abs(b[i,j] - np.var(values, ddof=1)) < 1.e-14

##########################################################################################
