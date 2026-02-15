##########################################################################################
# image_ops/tests/test_rotate.py
##########################################################################################

import numpy as np
import unittest
from psiops.resample import resample
from psiops.rotate   import rotate, _set_rotate_testing_flag
from psiops._utils   import _use_shortcuts


class Test_rotate(unittest.TestCase):

  def runTest(self):

    np.random.seed(9676)

    for status in (False, True):
        _use_shortcuts(status)

        # Force functions to return additional info
        _set_rotate_testing_flag(True)

        #### Square, rotation by 45 degrees
        image = np.arange(10) + np.arange(10)[:,np.newaxis]
        rotated, rmask, center, weight, alist, ilist, jlist = rotate(image, np.pi/4)
        self.assertEqual(rotated.shape, (16,16))
        self.assertEqual(center, (8,8))
        self.assertLess(np.max(np.abs(weight[2:14, 7:9] - 1)), 1.e-8)
        self.assertLess(np.max(np.abs(weight[3:13,6:10] - 1)), 1.e-8)
        self.assertLess(np.max(np.abs(weight[4:12,5:11] - 1)), 1.e-8)
        self.assertLess(np.max(np.abs(weight[5:11,4:12] - 1)), 1.e-8)
        self.assertLess(np.max(np.abs(weight[6:10,3:13] - 1)), 1.e-8)
        self.assertLess(np.max(np.abs(weight[7: 9,2:14] - 1)), 1.e-8)

        # Make sure every original pixel was fully weighted
        alist = np.array(alist)
        ilist = np.array(ilist)
        jlist = np.array(jlist)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ijmask = (ilist == i) & (jlist == j)
                self.assertLess(np.abs(np.sum(alist[ijmask]) - 1), 1.e-8)

        #### Not square, rotation by 60 degrees
        image = np.arange(10) + 3 * np.arange(20)[:,np.newaxis]
        rotated, rmask, center, weight, alist, ilist, jlist = rotate(image, np.pi/3)
        self.assertEqual(rotated.shape, (20,24))
        self.assertEqual(center, (10,12))
        self.assertEqual(np.sum(weight > 0.99), 166)

        alist = np.array(alist)
        ilist = np.array(ilist)
        jlist = np.array(jlist)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                ijmask = (ilist == i) & (jlist == j)
                self.assertLess(np.abs(np.sum(alist[ijmask]) - 1), 1.e-8)

        #### Not square, rotation by random angles
        image = np.arange(10) + 2 * np.arange(20)[:,np.newaxis]
        for k in range(300):
            angle = 2 * np.pi * np.random.rand()
            rotated, rmask, center, weight, alist, ilist, jlist = rotate(image,angle)

            alist = np.array(alist)
            ilist = np.array(ilist)
            jlist = np.array(jlist)
            max_error = 0.
            for i in range(image.shape[0]):
                for j in range(image.shape[1]):
                    ijmask = (ilist == i) & (jlist == j)
                    max_error = max(np.abs(np.sum(alist[ijmask]) - 1), max_error)

            self.assertTrue(max_error <= 3.e-5)

        #### Square, rotation by 45 degrees, isolated masked pixels
        image = np.arange(10) + 2 * np.arange(10)[:,np.newaxis]
        rotated0, rmask0, center, weight, _, _, _ = rotate(image, np.pi/4)

        mask = np.zeros(image.shape, dtype='bool')
        mask[3,3] = mask[5,5] = mask[7,7] = True
        image[mask] = -999

        rotated, rmask, center, weight, _, _, _ = rotate(image, np.pi/4, mask)
        self.assertTrue(np.all(rmask0 == rmask))
        self.assertLess(np.max(np.abs(rotated - rotated0)), 1)
        self.assertAlmostEqual(np.sum(weight), np.sum(~mask), 8)

        #### Many masked pixels, prove masked pixels are unweighted
        image = np.arange(10) + np.arange(10)[:,np.newaxis]
        rotated0, rmask0, center, weight, _, _, _ = rotate(image, np.pi/5)

        mask = np.random.rand(10,10) < 0.8
        image[mask] = -9999
        rotated, rmask, center, weight, _, _, _ = rotate(image, np.pi/5, mask)
        self.assertLess(np.max(np.abs(rotated[~rmask] - rotated0[~rmask])), 2)
        self.assertAlmostEqual(np.sum(weight), np.sum(~mask), 8)

        #### Rotation by 0 degrees, three layers, no mask
        image = (np.arange(10) + 3 * np.arange(20)[:,np.newaxis]
                 + np.arange(3)[:,np.newaxis,np.newaxis])
        rotated = rotate(image, 0.)[0]
        self.assertLess(np.max(np.abs(image - rotated)), 1.e-7)

        #### Rotation by 0 degrees, three layers, random mask
        mask = np.random.rand(*image.shape[-2:]) < 0.3
        rotated, rmask, center, weight, _, _, _ = rotate(image, 0., mask)
        self.assertTrue(np.all(mask == rmask))
        self.assertLess(np.max(np.abs(image[~rmask] - rotated[~rmask])), 1.e-7)

        #### Rotation by 90 degrees, three layers, random mask
        mask = np.random.rand(*image.shape) < 0.3
        rotated, rmask, center, weight, _, _, _ = rotate(image, np.pi/2, mask)

        test = image.swapaxes(-2,-1)[:,::-1]
        mtest = mask.swapaxes(-2,-1)[:,::-1]
        self.assertTrue(np.all(rmask == mtest))
        self.assertLess(np.max(np.abs(test[~rmask] - rotated[~rmask])), 1.e-7)

        rotated, rmask, center, weight, _, _, _ = rotate(image, np.pi/2 * (1 - 1.e-16), mask)
        self.assertTrue(np.all(rmask == mtest))
        self.assertLess(np.max(np.abs(test[~rmask] - rotated[~rmask])), 1.e-7)

        #### Rotation by 180 degrees, random mask
        image = np.arange(10) + 3 * np.arange(20)[:,np.newaxis]
        mask = np.random.rand(*image.shape) < 0.3
        rotated, rmask, center, weight, _, _, _ = rotate(image, np.pi, mask)

        test = image[::-1,::-1]
        mtest = mask[::-1,::-1]
        self.assertTrue(np.all(rmask == mtest))
        self.assertLess(np.max(np.abs(test[~rmask] - rotated[~rmask])), 1.e-7)

        #### Rotation by 270 degrees, random mask
        rotated, rmask, center, weight, _, _, _ = rotate(image, 1.5*np.pi, mask)

        test = image.swapaxes(-2,-1)[:,::-1]
        mtest = mask.swapaxes(-2,-1)[:,::-1]
        self.assertTrue(np.all(rmask == mtest))
        self.assertLess(np.max(np.abs(test[~rmask] - rotated[~rmask])), 1.e-7)

        # Restore default behavior
        _set_rotate_testing_flag(False)

        # Make sure dtype float32 is preserved
        array = np.arange(10)
        image = (array + array[:,np.newaxis]).astype('float32')
        rotated = rotate(image, np.pi/3)[0]
        self.assertEqual(image.dtype, rotated.dtype)

        # Half-integer center
        image = np.random.rand(10,10)
        rotated, _, new_center = rotate(image, 0.3, origin=(3.5,3.5))
        self.assertEqual(new_center[0] % 1, 0.5)
        self.assertEqual(new_center[1] % 1, 0.5)

        rotated, _, new_center = rotate(image, 0.4, origin=(3.,4.))
        self.assertEqual(new_center[0] % 1, 0.)
        self.assertEqual(new_center[1] % 1, 0.)

        # New shape but not new center
        image = np.random.rand(10,10)
        rotated, _, new_center = rotate(image, 0.3, shape=(20,20))
        self.assertEqual(new_center, (10,10))
        self.assertEqual(rotated.shape, (20,20))

    # Check rotation angle zero
#     image0 = (np.arange(4)[:,np.newaxis] + np.arange(10))[:,np.newaxis] + np.arange(10)
    image0 = np.arange(10)[:,np.newaxis] + np.arange(10)
    image1 = np.random.rand(4,10,10)
    for image in (image0, image1):
      print('image0' if image is image0 else 'image1')
      for mask in (None, np.random.rand(10,10) < 0.05, np.random.rand(10,10) < 0.45):
        print('mask', 'None' if mask is None else 'random')
        for center in ((4.5,5.25), None, (4.5+np.random.rand(), 4.5+np.random.rand())):
            print('center', repr(center))
            _use_shortcuts(False)
            rotated1, mask1, center1 = rotate(image, 0., mask=mask, center=center)

            _use_shortcuts(True)
            rotated2, mask2, center2 = rotate(image, 0., mask=mask, center=center)

            self.assertTrue(np.all(mask1 == mask2))
            self.assertLess(np.abs((rotated1 - rotated2)[~mask1]).max(), 1.e-9)
            self.assertEqual(abs(center1[0] - center2[0]), 0)
            self.assertEqual(abs(center1[1] - center2[1]), 0)
            if center is not None:
                self.assertEqual(abs(center1[0] - center[0]), 0)
                self.assertEqual(abs(center1[1] - center[1]), 0)

##########################################################################################
