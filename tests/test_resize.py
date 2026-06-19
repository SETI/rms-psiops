##########################################################################################
# tests/test_resize.py
##########################################################################################

import numpy as np
import unittest
from psiops.unzoom import unzoom
from psiops.zoom   import zoom

from tests.resize import resize # removed from image_ops but retained for cross-testing

class Test_resize(unittest.TestCase):

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
    self.assertTrue(np.all(unzoomed == resized))

    mask = np.random.rand(12,12,12) < 0.2
    unzoomed, zmask = unzoom(image, (3,2), mask)
    resized, rmask = resize(image, (4,6), mask)
    self.assertTrue(np.all(zmask == rmask))
    self.assertTrue(np.all(unzoomed[~zmask] == resized[~zmask]))

    mask = np.random.rand(12,12) < 0.2
    unzoomed, zmask = unzoom(image, (3,2), mask)
    resized, rmask = resize(image, (4,6), mask)
    self.assertTrue(np.all(zmask == rmask))
    self.assertTrue(np.all(unzoomed[~zmask] == resized[~zmask]))

    # Mixed
    array = np.arange(10)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    resized = resize(image, (7,11))
    self.assertTrue(np.all(image.mean(axis=(1,2)) == resized.mean(axis=(1,2))))

    mask = np.zeros(image.shape, dtype='bool')
    mask[:3,:3,:3] = True
    resized2, rmask = resize(image, (7,11), mask)
    self.assertTrue(np.all(resized[3:] == resized2[3:]))
    self.assertTrue(np.all(resized[:3,3:] == resized2[:3,3:]))
    self.assertTrue(np.all(resized[:3,:3,4:] == resized2[:3,:3,4:]))
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

    # Make sure masks are broadcasted and converted to bool
    resized, rmask = resize(image, (12,9), mask=7.)
    self.assertEqual(rmask.dtype, np.dtype('bool'))
    self.assertEqual(rmask.shape, resized.shape)
    self.assertTrue(np.all(rmask))

##########################################################################################
