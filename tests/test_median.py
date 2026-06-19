##########################################################################################
# tests/test_median.py
##########################################################################################

import numpy as np
import unittest
import warnings
from psiops.median import median, median_filter


class Test_median(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter('ignore', category=RuntimeWarning)

    def runTest(self):

        np.random.seed(3863)

        array = np.arange(10)
        image0 = array + array[:,np.newaxis]
        image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

        a = median(image)
        self.assertTrue(np.all(a == image0))

        # 2-D mask
        mask = np.random.rand(10,10) < 0.3
        b, bmask = median(image, mask=mask)
        self.assertTrue(np.all(mask == bmask))
        self.assertEqual(np.shape(bmask), (10,10))
        self.assertTrue(np.all(a[~mask] == b[~mask]))

        # 3-D mask
        mask = np.zeros(image.shape, dtype='bool')
        mask[0] = True
        b, bmask = median(image, mask=mask)
        self.assertTrue(np.all(b == 1.5 * image0))
        self.assertTrue(not np.any(bmask))

        mask = np.random.rand(3,10,10) < 0.3
        mask[0] = True
        mask[2] = mask[1]
        b, bmask = median(image, mask=mask)
        self.assertTrue(np.all(bmask == mask[1]))
        self.assertTrue(np.all((b == 1.5 * image0)[~bmask]))

        # More complicated shape
        image = np.random.rand(3,4,5,10,10)
        a = median(image, axis=1)
        self.assertEqual(a.shape, (3,5,10,10))
        self.assertTrue(np.all(a == np.median(image,axis=1)))

        a = median(image, axis=(1,-1))
        self.assertEqual(a.shape, (3,10,10))
        self.assertTrue(np.all(a == np.median(image,axis=(1,2))))

        a = median(image, axis=None)
        self.assertEqual(a.shape, (10,10))
        self.assertTrue(np.all(a == np.median(image,axis=(0,1,2))))

        # More complicated shape, axes, mask
        image = np.random.rand(5,4,3,10,10)
        mask = np.random.rand(5,4,3,10,10) < 0.7        # mostly masked
        a, amask = median(image, axis=2, mask=mask)
        self.assertEqual(a.shape, (5,4,10,10))
        self.assertEqual(amask.shape, (5,4,10,10))
        self.assertTrue(np.all(amask == np.all(mask, axis=2)))

        image = np.random.rand(5,4,100,200)
        mask = np.random.rand(5,4,100,200) < 0.6        # mostly masked
        a, amask = median(image, axis=0, mask=mask)

        sorted = image.copy()
        sorted[mask] = 10
        sorted = np.sort(sorted, axis=0)

        k = np.sum(np.logical_not(mask), axis=0)
        self.assertTrue(np.all((k==0) == amask))
        self.assertTrue(np.all((a == sorted[0])[k==1]))
        self.assertTrue(np.all((a == np.mean(sorted[:2], axis=0))[k==2]))
        self.assertTrue(np.all((a == sorted[1])[k==3]))
        self.assertTrue(np.all((a == np.mean(sorted[1:3], axis=0))[k==4]))
        self.assertTrue(np.all((a == sorted[2])[k==5]))

        # Check dtypes
        for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                      'int64', 'float32', 'float64'):
            image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
            test = median(image.astype(dtype))
            if dtype == 'float32':
                self.assertEqual(test.dtype, np.float32)
            else:
                self.assertEqual(test.dtype, np.float64)

        image = np.random.rand(3,5,7,10,10)
        self.assertTrue(np.all(median(image) == np.median(image, axis=(0,1,2))))
        self.assertTrue(np.all(median(image, axis=1) == np.median(image, axis=1)))
        self.assertTrue(np.all(median(image, axis=(0,-1)) == np.median(image, axis=(0,2))))

        # Check dtypes
        for dtype in ('bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                      'int64', 'float32', 'float64'):
            image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
            test = median(image.astype(dtype))
            if dtype == 'float32':
                self.assertEqual(test.dtype, np.float32)
            else:
                self.assertEqual(test.dtype, np.float64)

        # Weird input
        image = np.random.rand(3,5,7,10,10)
        image = list(image)     # works for non-arrays?
        self.assertTrue(np.all(median(image) == np.median(image, axis=(0,1,2))))

        image = np.arange(12).reshape(4,3)
        with self.assertRaises(ValueError) as err:
            _ = median(image)
        self.assertEqual(str(err.exception), 'illegal image shape; ndim >= 3 required: (4, 3)')

        # Unequal factors
        image = np.random.rand(5,10,10)
        image[3:] = image[2]    # last layer repeat three times
        self.assertTrue(np.all(median(image) == image[2]))

        test = median(image[:3], factors=[1,1,2.1])
        self.assertTrue(np.all(test == image[2]))

        test = median(image[:3], factors=[1,1,2])
        test2 = median(image[:4])
        self.assertTrue(np.all(test == test2))

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

        test, tmask = median(image, mask)
        self.assertTrue(np.all(test == np.array([[3. , 2. , 2.5],
                                                 [3. , 8.5, 8. ],
                                                 [0. , 0. , 1. ]])))
        self.assertTrue(np.all(tmask == np.array([[0 , 0  , 0  ],
                                                  [0 , 0  , 0  ],
                                                  [1 , 1  , 0  ]]).astype('bool')))

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
        self.assertTrue(np.all(test1 == test2))
        self.assertTrue(np.all(tmask1 == tmask2))

        # Omit
        image3 = np.empty((8,3,3)).astype('int')
        image3[:7] = image2
        image3[-1] = 99999999

        mask3 = np.empty((8,3,3)).astype('bool')
        mask3[:7] = mask2
        mask3[-1] = False

        test3, tmask3 = median(image3, mask3, omit=-1)
        self.assertTrue(np.all(test3 == test2))
        self.assertTrue(np.all(tmask3 == tmask2))

        image3[-1] = -99999999
        test3, tmask3 = median(image3, mask3, omit=1)
        self.assertTrue(np.all(test3 == test2))
        self.assertTrue(np.all(tmask3 == tmask2))

##########################################################################################

class Test_median_filter(unittest.TestCase):

    def runTest(self):

        np.random.seed(8063)

        # No mask
        image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
        a = median_filter(image, 2)
        b = np.empty((4,10,10))
        b[:,0,0] = image[:,0,0]
        b[:,0,1:] = (image[:,0,:-1] + image[:,0,1:]) / 2.
        b[:,1:,0] = (image[:,:-1,0] + image[:,1:,0]) / 2.
        for i in range(1,10):
            for j in range(1,10):
                b[:,i,j] = np.median(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1))
        self.assertLess(np.abs(a - b).max(), 1.e-15)

        # Mask, irregular footprint
        image = np.random.rand(100,100)
        mask = np.random.rand(100,100) < 0.6
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
                    self.assertTrue(amask[i,j])
                    continue

                self.assertFalse(amask[i,j])
                self.assertLess(np.abs(a[i,j] - np.median(values)), 1.e-14)

##########################################################################################
