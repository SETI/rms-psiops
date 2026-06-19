##########################################################################################
# tests/test_mean.py
##########################################################################################

import numpy as np
import unittest
import warnings
from psiops.mean import mean, mean_filter


class Test_mean(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=RuntimeWarning)

    def runTest(self):

        np.random.seed(5965)

        array = np.arange(10)
        image0 = array + array[:,np.newaxis]
        image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

        a = mean(image)
        self.assertTrue(np.all(a == 0.))

        # 2-D mask
        mask = np.random.rand(10,10) < 0.3
        b, bmask = mean(image, mask=mask)
        self.assertTrue(np.all(mask == bmask))
        self.assertEqual(np.shape(bmask), (10,10))
        self.assertTrue(np.all(a[~mask] == b[~mask]))

        # 3-D mask
        mask = np.zeros(image.shape, dtype='bool')
        mask[0] = True
        b, bmask = mean(image, mask=mask)
        self.assertTrue(np.all(b == 1.5 * image0))
        self.assertTrue(not np.any(bmask))

        mask = np.random.rand(3,10,10) < 0.3
        mask[0] = True
        mask[2] = mask[1]
        b, bmask = mean(image, mask=mask)
        self.assertTrue(np.all(bmask == mask[1]))
        self.assertTrue(np.all((b == 1.5 * image0)[~bmask]))

        # More complicated shape, axes
        image = np.random.rand(3,4,5,10,10)
        a = mean(image, axis=1)
        self.assertEqual(a.shape, (3,5,10,10))
        self.assertLess(np.abs(a - np.mean(image,axis=1)).max(), 1.e-15)

        a = mean(image, axis=(1,-1))
        self.assertEqual(a.shape, (3,10,10))
        self.assertTrue(np.abs(a - np.mean(image,axis=(1,2))).max() < 1.e-15)

        a = mean(image, axis=None)
        self.assertEqual(a.shape, (10,10))
        self.assertLess(np.abs(a - np.mean(image,axis=(0,1,2))).max(), 1.e-15)

        # More complicated shape, axes, mask
        image = np.random.rand(5,4,3,10,10)
        mask = np.random.rand(5,4,3,10,10) < 0.9        # mostly masked
        a, amask = mean(image, axis=2, mask=mask)
        self.assertEqual(a.shape, (5,4,10,10))
        self.assertEqual(amask.shape, (5,4,10,10))
        self.assertTrue(np.all(amask == np.all(mask, axis=2)))

        image = np.random.rand(5,4,100,200)
        mask = np.random.rand(5,4,100,200) < 0.6        # mostly masked
        a, amask = mean(image, axis=0, mask=mask)

        sorted = image.copy()
        sorted[mask] = 10
        sorted = np.sort(sorted, axis=0)

        k = np.sum(np.logical_not(mask), axis=0)
        self.assertTrue(np.all((k==0) == amask))
        self.assertTrue(np.all((a == sorted[0])[k==1]))
        self.assertLess(np.abs(a - np.mean(sorted[:2], axis=0))[k==2].max(), 1.e-15)
        self.assertLess(np.abs(a - np.mean(sorted[:3], axis=0))[k==3].max(), 1.e-15)
        self.assertLess(np.abs(a - np.mean(sorted[:4], axis=0))[k==4].max(), 1.e-15)
        self.assertLess(np.abs(a - np.mean(sorted[:5], axis=0))[k==5].max(), 1.e-15)

        # Factors, no mask
        image = np.random.rand(6,5,4,10,10)
        image = np.random.rand(6,1,1,4,4)
        image[5] = image[4]
        factors = np.array([1,1,1,1,2]).reshape(5,1,1)
        a = mean(image[:5], axis=0, factors=factors)
        b = mean(image, axis=0)
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        image = np.random.rand(5,2,6,10,10)
        image[:,0,4] = image[:,0,3]
        image[:,0,5] = image[:,0,3]
        image[:,1,4] = image[:,1,0]
        image[:,1,5] = image[:,1,1]
        factors = np.array([[1,1,1,3],[2,2,1,1]])
        a = mean(image[:,:,:4], axis=(1,2), factors=factors)
        b = mean(image, axis=(1,2))
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        a = mean(image[:,:,:4], axis=2, factors=factors)
        b = mean(image, axis=2)
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        image = np.random.rand(2,5,6,10,10)
        image[0,:,4] = image[0,:,3]
        image[0,:,5] = image[0,:,3]
        image[1,:,4] = image[1,:,0]
        image[1,:,5] = image[1,:,1]
        factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,1,4)
        a = mean(image[:,:,:4], axis=(0,2), factors=factors)
        b = mean(image, axis=(0,2))
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        a = mean(image[:,:,:4], axis=2, factors=factors.reshape(2,1,4))
        b = mean(image, axis=2)
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        image = np.random.rand(2,6,5,10,10)
        image[0,4] = image[0,3]
        image[0,5] = image[0,3]
        image[1,4] = image[1,0]
        image[1,5] = image[1,1]
        factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,4,1)
        a = mean(image[:,:4], axis=(0,1), factors=factors)
        b = mean(image, axis=(0,1))
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        a = mean(image[:,:4], axis=1, factors=factors)
        b = mean(image, axis=1)
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        # Weights, mask
        image = np.random.rand(6,5,4,10,10)
        image[5] = image[4]
        factors = np.array([1,1,1,1,2])
        mask = np.random.rand(10,10) < 0.3
        a, amask = mean(image[:5], axis=0, factors=factors.reshape(5,1,1), mask=mask)
        b, bmask = mean(image, axis=0, mask=mask)
        c = mean(image, axis=0)
        self.assertTrue(amask.shape == (5,4,10,10))
        self.assertTrue(np.all(amask == mask))
        self.assertTrue(np.all(amask == bmask))
        self.assertLess(np.abs(a - b)[~amask].max(), 1.e-15)

        image = np.random.rand(6,5,4,10,10)
        image[5] = image[4]
        factors = np.array([1,1,1,1,2])
        mask = np.random.rand(5,4,10,10) < 0.3
        a, amask = mean(image[:5], axis=0, factors=factors.reshape(5,1,1), mask=mask)
        b, bmask = mean(image, axis=0, mask=mask)
        c = mean(image, axis=0)
        self.assertTrue(amask.shape == (5,4,10,10))
        self.assertTrue(np.all(amask == bmask))
        self.assertLess(np.abs(a - b)[~amask].max(), 1.e-15)

        # Check dtypes
        for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                      'int64', 'float32', 'float64'):
            image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
            test = mean(image.astype(dtype))
            if dtype == 'float32':
                self.assertEqual(test.dtype, np.float32)
            else:
                self.assertEqual(test.dtype, np.float64)

        image = np.random.rand(2,4,8,10,10)
        self.assertTrue(np.all(mean(image) == np.mean(image, axis=(0,1,2))))
        self.assertTrue(np.all(mean(image, axis=1) == np.mean(image, axis=1)))
        self.assertTrue(np.all(mean(image, axis=(0,-1)) == np.mean(image, axis=(0,2))))

        # Check dtypes
        for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                      'int64', 'float32', 'float64'):
            image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
            test = mean(image.astype(dtype))
            if dtype == 'float32':
                self.assertEqual(test.dtype, np.float32)
            else:
                self.assertEqual(test.dtype, np.float64)

        # Weird input
        image = np.random.rand(2,4,8,10,10)
        image = list(image)     # works for non-arrays?
        self.assertTrue(np.all(mean(image) == np.mean(image, axis=(0,1,2))))

        image = np.arange(12).reshape(4,3)
        with self.assertRaises(ValueError) as err:
            _ = mean(image)
        self.assertEqual(str(err.exception), 'illegal image shape; ndim >= 3 required: (4, 3)')

##########################################################################################

class Test_mean_filter(unittest.TestCase):

    def runTest(self):

        np.random.seed(8063)

        # No mask
        image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
        a = mean_filter(image, (2,2))
        b = np.empty((4,10,10))
        b[:,0,0] = image[:,0,0]
        b[:,0,1:] = (image[:,0,:-1] + image[:,0,1:]) / 2.
        b[:,1:,0] = (image[:,:-1,0] + image[:,1:,0]) / 2.
        for i in range(1,10):
            for j in range(1,10):
                b[:,i,j] = np.mean(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1))
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        # Mask, irregular footprint
        image = np.random.rand(100,100)
        mask = np.random.rand(100,100) < 0.6
        footprint = np.ones((3,3), dtype='bool')
        footprint[0,0] = False
        footprint[0,2] = False
        footprint[2,0] = False
        a, amask = mean_filter(image, footprint=footprint, mask=mask)

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
                    self.assertTrue(amask[i,j])
                    continue

                self.assertFalse(amask[i,j])
                self.assertLess(np.abs(a[i,j] - np.mean(values)), 1.e-14)

##########################################################################################
