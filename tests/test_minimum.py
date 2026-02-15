##########################################################################################
# image_ops/tests/test_minimum.py
##########################################################################################

import numpy as np
import unittest
from psiops.minimum import minimum, minimum_filter


class Test_minimum(unittest.TestCase):

    def runTest(self):

        np.random.seed(3711)

        array = np.arange(10)
        image0 = array + array[:,np.newaxis]
        image = image0 * np.array([3,-1,-2])[:,np.newaxis,np.newaxis]

        a = minimum(image)
        self.assertTrue(np.all(a == -2 * image0))

        # 2-D mask
        mask = np.random.rand(10,10) < 0.3
        b, bmask = minimum(image, mask=mask)
        self.assertTrue(np.all(mask == bmask))
        self.assertEqual(np.shape(bmask), (10,10))
        self.assertTrue(np.all(a[~mask] == b[~mask]))

        # 3-D mask
        mask = np.zeros(image.shape, dtype='bool')
        mask[2] = True
        b, bmask = minimum(image, mask=mask)
        self.assertTrue(np.all(b == -image0))
        self.assertTrue(not np.any(bmask))

        mask = np.random.rand(3,10,10) < 0.3
        mask[2] = True
        mask[1] = mask[0]
        b, bmask = minimum(image, mask=mask)
        self.assertTrue(np.all(bmask == mask[1]))
        self.assertTrue(np.all((b == -image0)[~bmask]))

        mask = np.random.rand(3,10,10) < 0.3
        mask[0] = False
        mask[1] = True
        b, bmask = minimum(image, mask=mask)
        self.assertTrue(not np.any(bmask))
        self.assertTrue(np.all((b == -2 * image0)[~mask[2]]))
        self.assertTrue(np.all((b == 3 * image0)[mask[2]]))

        # More complicated shapes
        image = np.random.rand(3,4,5,10,10)
        a = minimum(image, axis=1)
        self.assertEqual(a.shape, (3,5,10,10))
        self.assertTrue(np.all(a == np.min(image,axis=1)))

        a = minimum(image, axis=(1,-1))
        self.assertEqual(a.shape, (3,10,10))
        self.assertTrue(np.all(a == np.min(image,axis=(1,2))))

        a = minimum(image, axis=None)
        self.assertEqual(a.shape, (10,10))
        self.assertTrue(np.all(a == np.min(image,axis=(0,1,2))))

        image = np.random.rand(3,4,5,10,10)     # range is 0 to 1
        image[:,0] -= 2                         # range is -2 to -1 at this axis
        a = minimum(image, axis=1)
        self.assertTrue(np.all(a == image[:,0]))

        # More complicated shape, axes, mask
        image = np.random.rand(3,4,5,10,10)
        image[:,0] -= 2                         # range is -2 to -1 at this axis
        mask = np.random.rand(10,10) < 0.3
        a, amask = minimum(image, axis=1, mask=mask)
        b = minimum(image, axis=1)
        self.assertEqual(a.shape, (3,5,10,10))
        self.assertEqual(amask.shape, (3,5,10,10))
        self.assertTrue(np.all(amask == mask))
        self.assertTrue(np.all(a[~amask] == b[~amask]))

        image = np.random.rand(5,4,3,10,10)
        mask = np.random.rand(5,4,3,10,10) < 0.9        # mostly masked
        a, amask = minimum(image, axis=2, mask=mask)
        self.assertEqual(a.shape, (5,4,10,10))
        self.assertEqual(amask.shape, (5,4,10,10))
        self.assertTrue(np.all(amask == np.all(mask, axis=2)))

        image = np.random.rand(5,4,3,10,10)
        mask = np.random.rand(5,4,3,10,10) < 0.2
        a, amask = minimum(image, axis=(0,2), mask=mask)
        image[mask] = 2
        b = minimum(image, axis=(0,-1))
        self.assertTrue(np.all(a[~amask] == b[~amask]))

        # Check dtypes, limits
        image0 = (np.arange(10) + np.arange(10)[:,np.newaxis]
                                + np.arange(10)[:,np.newaxis,np.newaxis])
        for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                      'int64', 'float32', 'float64'):
            image = image0.astype(dtype)
            test = minimum(image)
            self.assertTrue(np.all(test == np.min(image,axis=0)))
            self.assertEqual(test.dtype, image.dtype)

            if image.dtype.kind == 'i' and dtype != 'int64':
                maxval = np.ma.minimum_fill_value(image)
                image[0,0,0] = maxval
                self.assertTrue(np.all(minimum(image) == np.min(image,axis=0)))

        image = np.random.rand(5,4,3,10,10)
        self.assertTrue(np.all(minimum(image) == np.min(image, axis=(0,1,2))))
        self.assertTrue(np.all(minimum(image, axis=1) == np.min(image, axis=1)))
        self.assertTrue(np.all(minimum(image, axis=(0,-1)) == np.min(image, axis=(0,2))))

        # Weird input
        image = list(image)     # works for non-arrays?
        self.assertTrue(np.all(minimum(image) == np.min(image, axis=(0,1,2))))

        image = np.arange(12).reshape(4,3)
        with self.assertRaises(ValueError) as err:
            _ = minimum(image)
        self.assertEqual(str(err.exception), 'illegal image shape; ndim >= 3 required: (4, 3)')

        image = np.array([None, 1, 4., 'str', np.dtype('float'), None, None, None])
        image = image.reshape(2,2,2)
        with self.assertRaises(TypeError) as err:
            _ = minimum(image)
        self.assertEqual(str(err.exception), 'image dtype=object is not numeric')

##########################################################################################

class Test_minimum_filter(unittest.TestCase):

    def runTest(self):

        np.random.seed(8063)

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
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        # Mask, irregular footprint
        image = np.random.rand(100,100)
        mask = np.random.rand(100,100) < 0.6
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
                    self.assertTrue(amask[i,j])
                    continue

                self.assertFalse(amask[i,j])
                self.assertLess(np.abs(a[i,j] - np.min(values)), 1.e-14)

##########################################################################################
