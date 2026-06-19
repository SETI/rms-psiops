##########################################################################################
# tests/test_resample.py
##########################################################################################

import numpy as np
import pytest
from psiops.resample import resample
from psiops.unzoom   import unzoom
from psiops.zoom     import zoom
from psiops._filter  import _use_shortcuts

from tests.resize import resize # removed from image_ops but retained for cross-testing


def test_resample() -> None:

    rng = np.random.default_rng(8107)

    for status in (False, True):
        _use_shortcuts(status)

        array = np.arange(10)
        image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
        mask = rng.standard_normal((10,10)) < 0.3

        zoomed = zoom(image, 3)
        resampled, rmask, rcenter = resample(image, 3)
        assert np.all(zoomed == resampled)
        assert np.all(rmask == False)
        assert rcenter == (15,15)

        zoomed = zoom(image, (2,3))
        resampled, rmask, rcenter = resample(image, (2,3))
        assert np.all(rmask == False)
        assert np.all(zoomed == resampled)
        assert rcenter == (10,15)

        zoomed, zmask = zoom(image, (2,3), mask)
        resampled, rmask, rcenter = resample(image, (2,3), mask)
        assert np.all(zmask == rmask)
        assert np.all(zoomed[~zmask] == resampled[~zmask])
        assert rcenter == (10,15)

        zoomed, zmask = unzoom(image, 2, mask)
        resampled, rmask, rcenter = resample(image, 0.5, mask)
        assert np.all(zmask == rmask)
        assert np.all(zoomed[~zmask] == resampled[~zmask])
        assert rcenter == (2.5,2.5)

        zoomed, zmask = zoom(image, (3,1), mask)
        zoomed, zmask = unzoom(zoomed, (1,2), zmask)
        resampled, rmask, rcenter = resample(image, (3,0.5), mask)
        assert np.all(zmask == rmask)
        assert np.all(zoomed[~zmask] == resampled[~zmask])
        assert rcenter == (15,2.5)

        EPS = 1.e-14

        array1 = np.arange(4)
        array2 = 2 * np.arange(4)
        array3 = 3 * np.arange(3)
        image = array1 + array2[:,np.newaxis] + array3[:,np.newaxis,np.newaxis]
        resized = resize(image, (3,3))
        resampled, rmask, rcenter = resample(image, (3/4.,3/4.))
        assert np.all(rmask == False)
        assert np.max(np.abs(resized - resampled)) < EPS
        assert rcenter == (1.5,1.5)

        mask = rng.standard_normal((4,4)) < 0.3
        resized, zmask = resize(image, (3,3), mask)
        resampled, rmask, rcenter = resample(image, (3/4.,3/4.), mask)
        assert np.all(rmask == zmask)
        assert np.max(np.abs(resized[~rmask] - resampled[~rmask])) < EPS
        assert rcenter == (1.5,1.5)

        resized = resize(image, (3,5))
        resampled, rmask, rcenter = resample(image, (3/4.,5/4.))
        assert np.all(rmask == False)
        assert np.all(np.abs(resized - resampled) < EPS)
        assert rcenter == (1.5,2.5)

        resized, zmask = resize(image, (3,5), mask)
        resampled, rmask, rcenter = resample(image, (3/4.,5/4.), mask)
        assert np.all(rmask == zmask)
        assert np.max(np.abs(resized[~rmask] - resampled[~rmask])) < EPS
        assert rcenter == (1.5,2.5)

        # prime number dimensions (13,47) to (97,11)
        array1 = np.arange(47)
        array2 = 2 * np.arange(13)
        array3 = 3 * np.arange(6).reshape(2,3)
        image = array1 + array2[:,np.newaxis] + array3[...,np.newaxis,np.newaxis]

        EPS = 1.e-12

        resized = resize(image, (97,11))
        resampled, rmask, rcenter = resample(image, (97/13.,11/47.))
        assert np.all(rmask == False)
        assert np.max(np.abs(resized - resampled)) < EPS
        assert rcenter == (97/2.,11/2.)

        for frac in (0.02, 0.3, 0.7, 0.9, 0.98):
            mask = rng.standard_normal(image.shape) < frac

            resized, zmask = resize(image, (97,11), mask)
            resampled, rmask, rcenter = resample(image, (97/13.,11/47.), mask)
            assert np.all(rmask == zmask)
            assert np.max(np.abs(resized[~zmask] - resampled[~zmask])) < EPS
            assert rcenter == (97/2.,11/2.)

        # Tests of additional input parameters
        image = np.arange(4) + 2 * np.arange(1,7)[:,np.newaxis]

        ref, rmask, rcenter = resample(image, 1.5)
        assert np.all(rmask == False)
        assert ref.shape == (9,6)
        assert np.mean(image) == np.mean(ref)
        assert rcenter == (4.5,3)

        resampled, rmask, rcenter = resample(image, 1.5, origin=(1,1))
        assert np.all(resampled == ref)
        assert np.all(rmask == False)
        assert rcenter == (1.5,1.5)

        resampled, rmask, rcenter = resample(image, 1.5, origin=(2,2), center=(3,3))
        assert np.all(resampled == ref)
        assert np.all(rmask == False)
        assert rcenter == (3,3)

        resampled, rmask, rcenter = resample(image, 1.5, origin=(2,2), center=(4,4))
        assert np.all(resampled[1:,1:] == ref)
        assert np.all(rmask[1:,1:] == False)
        assert np.all(rmask[0] == True)
        assert np.all(rmask[:,0] == True)
        assert rcenter == (4,4)

        resampled, rmask, rcenter = resample(image, 1.5, shape=(11,8))
        assert np.all(resampled[:-2,:-2] == ref)
        assert np.all(rmask[:-2,:-2] == False)
        assert np.all(rmask[-2:] == True)
        assert np.all(rmask[:,-2:] == True)
        assert rcenter == (4.5,3)

        resampled, rmask, rcenter = resample(image, 1.5, shape=(11,8), center=(4.5,3))
        assert np.all(resampled[:-2,:-2] == ref)
        assert np.all(rmask[:-2,:-2] == False)
        assert np.all(rmask[-2:] == True)
        assert np.all(rmask[:,-2:] == True)
        assert rcenter == (4.5,3)

        resampled, rmask, rcenter = resample(image, 1.5, origin=(0,2), shape=(6,6))
        assert np.all(resampled == ref[:6,:6])
        assert np.all(rmask == False)
        assert rcenter == (0,3)

        # Make sure dtype float32 is preserved
        array = np.arange(10)
        image = (array + array[:,np.newaxis]).astype('float32')
        resampled = resample(image, (12,9))[0]
        assert image.dtype == resampled.dtype

        # Weird input
        image = rng.random((100,100))
        with pytest.raises(ValueError) as exc_info:
            _ = resample(image, 2, origin=(0,0), center=(-1000,0))
        assert str(exc_info.value).partition(':')[0] == 'negative dimensions are not allowed'

##########################################################################################
