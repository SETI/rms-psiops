##########################################################################################
# tests/test_reshape.py
##########################################################################################

import numpy as np
import unittest
from psiops.reshape import reshape as resize    # rename and then re-use resize tests
from psiops.unzoom  import unzoom
from psiops.zoom    import zoom

EPS = 2.e-14

class Test_reshape(unittest.TestCase):

  def runTest(self):

    np.random.seed(4163)

    # Compared to zoom
    array = np.arange(10)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    zoomed = zoom(image, (3,2))
    resized = resize(image, (30,20))
    self.assertTrue(np.all(zoomed == resized))

    mask = np.random.rand(10,10,10) < 0.2
    zoomed, zmask = zoom(image, (3,2), mask)
    resized, rmask = resize(image, (30,20), mask)
    self.assertTrue(np.all(zmask == rmask))
    self.assertTrue(np.all(zoomed[~zmask] == resized[~zmask]))

    mask = np.random.rand(10,10) < 0.2
    zoomed, zmask = zoom(image, (3,2), mask)
    resized, rmask = resize(image, (30,20), mask)
    self.assertTrue(np.all(zmask == rmask))
    self.assertTrue(np.all(zoomed[~zmask] == resized[~zmask]))

    # Compared to unzoom
    array = np.arange(12)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    unzoomed = unzoom(image, (3,2))
    resized = resize(image, (4,6))
    self.assertLess(np.abs(unzoomed - resized).max(), EPS)

    mask = np.random.rand(12,12,12) < 0.2
    unzoomed, zmask = unzoom(image, (3,2), mask)
    resized, rmask = resize(image, (4,6), mask)
    self.assertTrue(np.all(zmask == rmask))
    self.assertLess(np.abs(unzoomed[~zmask] - resized[~zmask]).max(), EPS)

    mask = np.random.rand(12,12) < 0.2
    unzoomed, zmask = unzoom(image, (3,2), mask)
    resized, rmask = resize(image, (4,6), mask)
    self.assertTrue(np.all(zmask == rmask))
    self.assertLess(np.abs(unzoomed[~zmask] - resized[~zmask]).max(), EPS)

    # Mixed
    array = np.arange(10)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    resized = resize(image, (7,11))
    self.assertTrue(np.all(image.mean(axis=(1,2)) == resized.mean(axis=(1,2))))

    mask = np.zeros(image.shape, dtype='bool')
    mask[:3,:3,:3] = True
    resized2, rmask = resize(image, (7,11), mask)
    self.assertLess(np.abs(resized[3:] - resized2[3:]).max(), EPS)
    self.assertTrue(np.abs(resized[:3,3:] - resized2[:3,3:]).max(), EPS)
    self.assertTrue(np.abs(resized[:3,:3,4:] - resized2[:3,:3,4:]).max(), EPS)
    self.assertTrue(np.all(resized2[:3,:2,:3] == 0))

    # Errors
    self.assertRaises(TypeError, resize, image, 3.2)
    self.assertRaises(TypeError, resize, image, (3.2,1))
    self.assertRaises(TypeError, resize, image, '')
    self.assertRaises(ValueError, resize, image, (-3,1))
    self.assertRaises(ValueError, resize, image, (1,2,3))
    self.assertRaises(ValueError, resize, image, (2,))
    self.assertRaises(ValueError, resize, image, None)

    # Make sure dtype float32 is preserved
    array = np.arange(10)
    image = (array + array[:,np.newaxis]).astype('float32')
    resized = resize(image, (12,9))
    self.assertEqual(image.dtype, resized.dtype)

##########################################################################################
