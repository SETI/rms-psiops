##########################################################################################
# tests/test_resize.py
##########################################################################################

import numpy as np
import pytest
from psiops.unzoom import unzoom
from psiops.zoom   import zoom

from tests.resize import resize # removed from image_ops but retained for cross-testing


def test_resize() -> None:

    rng = np.random.default_rng(4163)

    # Compared to zoom
    array = np.arange(10)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    zoomed = zoom(image, (3,2))
    resized = resize(image, (30,20))
    assert np.all(zoomed == resized)

    mask = rng.random((10,10,10)) < 0.2
    zoomed, zmask = zoom(image, (3,2), mask)
    resized, rmask = resize(image, (30,20), mask)
    assert np.all(zmask == rmask)
    assert np.all(zoomed[~zmask] == resized[~zmask])

    mask = rng.random((10,10)) < 0.2
    zoomed, zmask = zoom(image, (3,2), mask)
    resized, rmask = resize(image, (30,20), mask)
    assert np.all(zmask == rmask)
    assert np.all(zoomed[~zmask] == resized[~zmask])

    # Compared to unzoom
    array = np.arange(12)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    unzoomed = unzoom(image, (3,2))
    resized = resize(image, (4,6))
    assert np.all(unzoomed == resized)

    mask = rng.random((12,12,12)) < 0.2
    unzoomed, zmask = unzoom(image, (3,2), mask)
    resized, rmask = resize(image, (4,6), mask)
    assert np.all(zmask == rmask)
    assert np.all(unzoomed[~zmask] == resized[~zmask])

    mask = rng.random((12,12)) < 0.2
    unzoomed, zmask = unzoom(image, (3,2), mask)
    resized, rmask = resize(image, (4,6), mask)
    assert np.all(zmask == rmask)
    assert np.all(unzoomed[~zmask] == resized[~zmask])

    # Mixed
    array = np.arange(10)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    resized = resize(image, (7,11))
    assert np.all(image.mean(axis=(1,2)) == resized.mean(axis=(1,2)))

    mask = np.zeros(image.shape, dtype='bool')
    mask[:3,:3,:3] = True
    resized2, rmask = resize(image, (7,11), mask)
    assert np.all(resized[3:] == resized2[3:])
    assert np.all(resized[:3,3:] == resized2[:3,3:])
    assert np.all(resized[:3,:3,4:] == resized2[:3,:3,4:])
    assert np.all(resized2[:3,:2,:3] == 0)

    # Errors
    with pytest.raises(TypeError): resize(image, 3.2)
    with pytest.raises(TypeError): resize(image, (3.2,1))
    with pytest.raises(TypeError): resize(image, '')
    with pytest.raises(ValueError): resize(image, (-3,1))
    with pytest.raises(ValueError): resize(image, (1,2,3))
    with pytest.raises(ValueError): resize(image, (2,))
    with pytest.raises(ValueError): resize(image, None)

    # Make sure dtype float32 is preserved
    array = np.arange(10)
    image = (array + array[:,np.newaxis]).astype('float32')
    resized = resize(image, (12,9))
    assert image.dtype == resized.dtype

    # Make sure masks are broadcasted and converted to bool
    resized, rmask = resize(image, (12,9), mask=7.)
    assert rmask.dtype == np.dtype('bool')
    assert rmask.shape == resized.shape
    assert np.all(rmask)

##########################################################################################
