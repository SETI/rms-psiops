##########################################################################################
# tests/test_minimum.py
##########################################################################################

import numpy as np
import pytest
from psiops.minimum import minimum, minimum_filter


def test_minimum() -> None:

    rng = np.random.default_rng(3711)

    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([3,-1,-2])[:,np.newaxis,np.newaxis]

    a = minimum(image)
    assert np.all(a == -2 * image0)

    # 2-D mask
    mask = rng.random((10,10)) < 0.3
    b, bmask = minimum(image, mask=mask)
    assert np.all(mask == bmask)
    assert np.shape(bmask) == (10,10)
    assert np.all(a[~mask] == b[~mask])

    # 3-D mask
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

    # More complicated shapes
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

    # More complicated shape, axes, mask
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

    # Check dtypes, limits
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

    image = rng.random((5,4,3,10,10))
    assert np.all(minimum(image) == np.min(image, axis=(0,1,2)))
    assert np.all(minimum(image, axis=1) == np.min(image, axis=1))
    assert np.all(minimum(image, axis=(0,-1)) == np.min(image, axis=(0,2)))

    # Weird input
    image = list(image)     # works for non-arrays?
    assert np.all(minimum(image) == np.min(image, axis=(0,1,2)))

    image = np.arange(12).reshape(4,3)
    with pytest.raises(ValueError) as exc_info:
        _ = minimum(image)
    assert str(exc_info.value) == 'illegal image shape; ndim >= 3 required: (4, 3)'

    image = np.array([None, 1, 4., 'str', np.dtype('float'), None, None, None])
    image = image.reshape(2,2,2)
    with pytest.raises(TypeError) as exc_info:
        _ = minimum(image)
    assert str(exc_info.value) == 'image dtype=object is not numeric'

##########################################################################################

def test_minimum_filter() -> None:

    rng = np.random.default_rng(8063)

    # No mask
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

    # Mask, irregular footprint
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

##########################################################################################
