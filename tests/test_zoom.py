##########################################################################################
# tests/test_zoom.py
##########################################################################################

import numpy as np
import unittest
from psiops.zoom   import zoom
from psiops.unzoom import unzoom


class Test_zoom(unittest.TestCase):

  def runTest(self):

    # 3-D array, zoomed up and then down, no mask
    array = np.arange(10)
    image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    zoomed = zoom(image, 3)
    self.assertEqual(zoomed.shape, (10,30,30))
    unzoomed = unzoom(zoomed, (3,1))
    self.assertEqual(unzoomed.shape, (10,10,30))
    unzoomed = unzoom(zoomed, 3)
    self.assertTrue(np.all(unzoomed == image))

    # Zoom down to force averaging
    unzoomed = unzoom(image, (1,2))
    answer = np.arange(0.5,9,2) + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
    self.assertTrue(np.all(unzoomed == answer))

    # Zoom down with a mask
    mask = np.zeros(image.shape, dtype='bool')
    mask[:3,:3,:3] = True
    unzoomed2, umask = unzoom(image, (1,2), mask=mask)

    new_mask = np.zeros(unzoomed.shape, dtype='bool')
    new_mask[:3,:3,:1] = True
    self.assertTrue(np.all(new_mask == umask))

    self.assertTrue(np.all(unzoomed2[umask] == 0))
    self.assertTrue(np.all(unzoomed2[:3,:3,1] == image[:3,:3,3]))
        # because [:3,:3,2] is masked
    self.assertTrue(np.sum(unzoomed2 != unzoomed) == 18)

    # Zoom up with a mask
    zoomed, zmask = zoom(image, 2, mask)
    self.assertTrue(np.all(image == zoomed[:, ::2, ::2]))
    self.assertTrue(np.all(image == zoomed[:,1::2, ::2]))
    self.assertTrue(np.all(image == zoomed[:, ::2,1::2]))
    self.assertTrue(np.all(image == zoomed[:,1::2,1::2]))

    self.assertTrue(np.all(zmask[:, ::2, ::2] == mask))
    self.assertTrue(np.all(zmask[:,1::2, ::2] == mask))
    self.assertTrue(np.all(zmask[:, ::2,1::2] == mask))
    self.assertTrue(np.all(zmask[:,1::2,1::2] == mask))

    zoomed, zmask = zoom(image, (3,1), mask)
    self.assertTrue(np.all(image == zoomed[:, ::3]))
    self.assertTrue(np.all(image == zoomed[:,1::3]))
    self.assertTrue(np.all(image == zoomed[:,2::3]))

    self.assertTrue(np.all(zmask[:, ::3] == mask))
    self.assertTrue(np.all(zmask[:,1::3] == mask))
    self.assertTrue(np.all(zmask[:,2::3] == mask))

    # Errors
    self.assertRaises(TypeError, zoom, image, 3.2)
    self.assertRaises(TypeError, zoom, image, (3.2,1))
    self.assertRaises(ValueError, zoom, image, (-3,1))
    self.assertRaises(ValueError, zoom, image, (1,2,3))
    self.assertRaises(ValueError, zoom, image, (2,))
    self.assertRaises(ValueError, zoom, image, None)

    self.assertRaises(TypeError, unzoom, image, 3.2)
    self.assertRaises(TypeError, unzoom, image, (3.2,1))
    self.assertRaises(ValueError, unzoom, image, (-3,1))
    self.assertRaises(ValueError, unzoom, image, (1,2,3))
    self.assertRaises(ValueError, unzoom, image, (2,))
    self.assertRaises(ValueError, unzoom, image, None)
    self.assertRaises(ValueError, unzoom, image, 3)     # shape is (10,10)

    # Make sure dtype is preserved on zoom
    array = np.arange(10)
    image = array + array[:,np.newaxis]
    for dtype in ('int', 'float', 'float32', 'bool', 'uint8', 'uint16', 'int16'):
        typed_image = image.astype(dtype)
        zoomed = zoom(typed_image, 2)
        self.assertEqual(typed_image.dtype, zoomed.dtype)

    # Make sure dtype float32 is preserved on unzoom
    image = (array + array[:,np.newaxis]).astype('float32')
    unzoomed = unzoom(image, 2)
    self.assertEqual(image.dtype, unzoomed.dtype)

##########################################################################################
