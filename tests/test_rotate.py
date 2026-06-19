##########################################################################################
# tests/test_rotate.py
##########################################################################################

import numpy as np
from psiops.resample import resample
from psiops.rotate   import rotate
from psiops._filter  import _use_shortcuts


def test_rotate() -> None:

    rng = np.random.default_rng(9676)

    for status in (False, True):
        _use_shortcuts(status)

        #### Square, rotation by 45 degrees
        image = np.arange(10) + np.arange(10)[:,np.newaxis]
        debug = {}
        result = rotate(image, np.pi/4, _debug=debug)
        rotated = result[0]
        center = debug['new_center']
        weight = debug['new_weights']
        assert rotated.shape == (16,16)
        assert center == (8,8)
        assert np.max(np.abs(weight[2:14, 7:9] - 1)) < 1.e-8
        assert np.max(np.abs(weight[3:13,6:10] - 1)) < 1.e-8
        assert np.max(np.abs(weight[4:12,5:11] - 1)) < 1.e-8
        assert np.max(np.abs(weight[5:11,4:12] - 1)) < 1.e-8
        assert np.max(np.abs(weight[6:10,3:13] - 1)) < 1.e-8
        assert np.max(np.abs(weight[7: 9,2:14] - 1)) < 1.e-8

        # Make sure every original pixel was fully weighted
        alist = np.array(debug['area_list'])
        ilist = np.array(debug['imod_list'])
        jlist = np.array(debug['jmod_list'])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ijmask = (ilist == i) & (jlist == j)
                assert np.abs(np.sum(alist[ijmask]) - 1) < 1.e-8

        #### Not square, rotation by 60 degrees
        image = np.arange(10) + 3 * np.arange(20)[:,np.newaxis]
        debug = {}
        result = rotate(image, np.pi/3, _debug=debug)
        rotated = result[0]
        center = debug['new_center']
        weight = debug['new_weights']
        assert rotated.shape == (20,24)
        assert center == (10,12)
        assert np.sum(weight > 0.99) == 166

        alist = np.array(debug['area_list'])
        ilist = np.array(debug['imod_list'])
        jlist = np.array(debug['jmod_list'])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ijmask = (ilist == i) & (jlist == j)
                assert np.abs(np.sum(alist[ijmask]) - 1) < 1.e-8

        #### Not square, rotation by random angles
        image = np.arange(10) + 2 * np.arange(20)[:,np.newaxis]
        for k in range(300):
            angle = 2 * np.pi * rng.random()
            debug = {}
            rotate(image, angle, _debug=debug)
            if debug['area_list'] is None:
                continue  # exact pi/2 multiple: area list not available
            alist = np.array(debug['area_list'])
            ilist = np.array(debug['imod_list'])
            jlist = np.array(debug['jmod_list'])
            max_error = 0.
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    ijmask = (ilist == i) & (jlist == j)
                    max_error = max(np.abs(np.sum(alist[ijmask]) - 1), max_error)

            assert max_error <= 3.e-5

        #### Square, rotation by 45 degrees, isolated masked pixels
        image = np.arange(10) + 2 * np.arange(10)[:,np.newaxis]
        debug0 = {}
        result0 = rotate(image, np.pi/4, _debug=debug0)
        rotated0 = result0[0]
        rmask0 = debug0['new_mask']

        mask = np.zeros(image.shape, dtype='bool')
        mask[3,3] = mask[5,5] = mask[7,7] = True
        image[mask] = -999

        debug = {}
        result = rotate(image, np.pi/4, mask, _debug=debug)
        rotated = result[0]
        rmask = debug['new_mask']
        weight = debug['new_weights']
        assert np.all(rmask0 == rmask)
        assert np.max(np.abs(rotated - rotated0)) < 1
        assert abs(np.sum(weight) - np.sum(~mask)) < 5e-9

        #### Many masked pixels, prove masked pixels are unweighted
        image = np.arange(10) + np.arange(10)[:,np.newaxis]
        debug0 = {}
        result0 = rotate(image, np.pi/5, _debug=debug0)
        rotated0 = result0[0]
        rmask0 = debug0['new_mask']

        mask = rng.random((10,10)) < 0.8
        image[mask] = -9999
        debug = {}
        result = rotate(image, np.pi/5, mask, _debug=debug)
        rotated = result[0]
        rmask = debug['new_mask']
        weight = debug['new_weights']
        assert np.max(np.abs(rotated[~rmask] - rotated0[~rmask])) < 2
        assert abs(np.sum(weight) - np.sum(~mask)) < 5e-9

        #### Rotation by 0 degrees, three layers, no mask
        image = (np.arange(10) + 3 * np.arange(20)[:,np.newaxis]
                 + np.arange(3)[:,np.newaxis,np.newaxis])
        rotated = rotate(image, 0.)[0]
        assert np.max(np.abs(image - rotated)) < 1.e-7

        #### Rotation by 0 degrees, three layers, random mask
        mask = rng.random(image.shape[-2:]) < 0.3
        result = rotate(image, 0., mask)
        rotated, rmask = result[0], result[1]
        assert np.all(mask == rmask)
        assert np.max(np.abs(image[~rmask] - rotated[~rmask])) < 1.e-7

        #### Rotation by 90 degrees, three layers, random mask
        mask = rng.random(image.shape) < 0.3
        result = rotate(image, np.pi/2, mask)
        rotated, rmask = result[0], result[1]

        test = image.swapaxes(-2,-1)[:,::-1]
        mtest = mask.swapaxes(-2,-1)[:,::-1]
        assert np.all(rmask == mtest)
        assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7

        result = rotate(image, np.pi/2 * (1 - 1.e-16), mask)
        rotated, rmask = result[0], result[1]
        assert np.all(rmask == mtest)
        assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7

        #### Rotation by 180 degrees, random mask
        image = np.arange(10) + 3 * np.arange(20)[:,np.newaxis]
        mask = rng.random(image.shape) < 0.3
        result = rotate(image, np.pi, mask)
        rotated, rmask = result[0], result[1]

        test = image[::-1,::-1]
        mtest = mask[::-1,::-1]
        assert np.all(rmask == mtest)
        assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7

        #### Rotation by 270 degrees, random mask
        result = rotate(image, 1.5*np.pi, mask)
        rotated, rmask = result[0], result[1]

        test = image.swapaxes(-2,-1)[:,::-1]
        mtest = mask.swapaxes(-2,-1)[:,::-1]
        assert np.all(rmask == mtest)
        assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7

        # Make sure dtype float32 is preserved
        array = np.arange(10)
        image = (array + array[:,np.newaxis]).astype('float32')
        rotated = rotate(image, np.pi/3)[0]
        assert image.dtype == rotated.dtype

        # Half-integer center
        image = rng.random((10,10))
        rotated, new_center = rotate(image, 0.3, origin=(3.5,3.5))
        assert new_center[0] % 1 == 0.5
        assert new_center[1] % 1 == 0.5

        rotated, new_center = rotate(image, 0.4, origin=(3.,4.))
        assert new_center[0] % 1 == 0.
        assert new_center[1] % 1 == 0.

        # New shape but not new center
        image = rng.random((10,10))
        rotated, new_center = rotate(image, 0.3, shape=(20,20))
        assert new_center == (10,10)
        assert rotated.shape == (20,20)

    # Check rotation angle zero
#     image0 = (np.arange(4)[:,np.newaxis] + np.arange(10))[:,np.newaxis] + np.arange(10)
    image0 = np.arange(10)[:,np.newaxis] + np.arange(10)
    image1 = rng.random((4,10,10))
    for image in (image0, image1):
        for mask in (None, rng.random((10,10)) < 0.05, rng.random((10,10)) < 0.45):
            for center in ((4.5,5.25), None, (4.5 + rng.random(), 4.5 + rng.random())):
                _use_shortcuts(False)
                result1 = rotate(image, 0., mask=mask, center=center, returns='imc')
                rotated1, mask1, center1 = result1

                _use_shortcuts(True)
                result2 = rotate(image, 0., mask=mask, center=center, returns='imc')
                rotated2, mask2, center2 = result2

                assert np.all(mask1 == mask2)
                assert np.abs((rotated1 - rotated2)[~mask1]).max() < 1.e-9
                assert abs(center1[0] - center2[0]) == 0
                assert abs(center1[1] - center2[1]) == 0
                if center is not None:
                    assert abs(center1[0] - center[0]) == 0
                    assert abs(center1[1] - center[1]) == 0

##########################################################################################
