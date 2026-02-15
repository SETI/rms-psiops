##########################################################################################
# image_ops/robust_mean.py
##########################################################################################

import numpy as np
from image_ops.utils import _check_tuple, _check_axis


def robust_mean(image, mask=None, *, axis=None, ignore=(0,0), sigma=None, stdev=None,
                       iterations=3):
    """The robust mean of an array of images, excluding masked pixels.

    At each pixel, the set of images is sorted and "ignore" pixels are excluded from the
    top and bottom of the sort.

    Among those pixels that remain, any that are more than "sigma" standard deviations
    different from median are also ignored.

    Inputs:
        image       an image array, at least 3-D. The last array two axes are always the
                    spatial axes of the image.
        mask        an optional boolean array broadcastable to the shape of the image,
                    containing True where image values should be ignored.
        axis        the axis over which to operate, or a tuple of two or more axes;
                    None to operate over all but the last two axes. Note that negative
                    axes count backwards from the first image axis, so axis=-1 actually
                    refers to the third-to-last axis of the image array.
        ignore      an optional number of pixels to ignore at the top and bottom of the
                    sorted pixels. Use a tuple of two integers for different bottom and
                    top values.
        sigma       an optional upper limit on the standard deviation among the pixels not
                    ignored; values more than this different from the local median are
                    excluded.
        stdev       an optional image array of standard deviations to use for the sigma
                    test; if not provided, the local standard deviation is calculated.
        iterations  if stdev is None, this is the number of iterations in which the sigma
                    test is applied and a new standard deviation is calculated.

    Return          mean_image or (mean_image, new_mask)
        mean_image  the mean image array.
        new_mask    the mask for the mean image, provided if the input mask is not None.
    """

    # Make a new copy, must be floating-point
    if isinstance(image, np.ndarray) and image.dtype.kind == 'f':
        image = image.copy()
    else:
        image = np.asarray(image, dtype=np.float64)

    # Identify the axes
    axis = _check_axis(axis, image.shape)

    # Prep the mask
    if mask is None:
        mask = False
    mask = np.broadcast_to(mask, image.shape)

    # Move the selected axes to the front, then flatten them
    naxes = len(axis)
    before = axis
    after = tuple(range(naxes))
    image = np.moveaxis(image, before, after)
    mask  = np.moveaxis(mask,  before, after)

    image = image.reshape((-1,) + image.shape[naxes:])
    mask  = mask.reshape( (-1,) + image.shape[naxes:])

    # Replace all masked pixels with a value above the maximum in the image array
    image_max = image.max()
    HI = max(image_max * 0.999, image_max * 1.001)
    image[mask] = HI

    # Interpret the "ignore" input
    ignore = _check_tuple(ignore, 'ignore counts', floats=False, negs=False,
                          default=(0,0))

    # Slice away the ignored low pixels, if necessary; update mask
    image_max = image.max()
    HI = max(image_max * 0.999, image_max * 1.001)
    if ignore[0] != 0:
        image[mask] = HI
        image = np.sort(image, axis=0)
        image = image[:ignore[0]]
        mask = (image == HI)

    # Slice away the ignored low pixels, if necessary
    image_min = image.min()
    LO = min(image_min * 0.999, image_min * 1.001)
    if ignore[1] != 0.:
        image[mask] = LO
        image = np.sort(image, axis=0)
        image = image[-ignore[1]:]
        mask = (image == LO)

    # Count the unmasked pixels
    unmasked = image.shape[0] - np.sum(mask, axis=0)

    # Select the midpoint values
    index_hi = unmasked // 2
    index_lo = np.maximum((unmasked - 1) // 2, 0)
        # For odd counts of unmasked pixels, index_lo and index_hi are the same;
        # otherwise, they are adjacent and we must take their mean.

    indices = np.indices(image.shape[1:])
    values_lo = image[(index_lo,) + indices]
    values_hi = image[(index_hi,) + indices]

    # Define the median
    median_image = 0.5 * (values_lo + values_hi)
    new_mask = (median_image == LO)

    # Without the sigma test, we're done
    if not sigma:
        median_image[new_mask] = 0.
        return (median_image, new_mask)

    # Check the number of iterations
    if stdev is not None:
        iterations = 1
        var = stdev**2

    # Iterate...
    new_mask = (image == LO)
    for k in iterations:
        if stdev is None:
            weight = 1 - new_mask
            n = np.sum(weight, axis=0)
            new_mask |= n < 2
            n[new_mask] = 2
            sum1 = np.sum(weight * image, axis=0)
            sum2 = np.sum(weight * image**2, axis=0)
            var = (n * sum2 - sum1**2) / (n-1)

        new_mask |= (image - median_image)**2 > sigma**2 * var

    # Determine the mean of remaining pixels
    image[new_mask] = 0.

    n = image.shape[0] - np.sum(new_mask, axis=0)
    new_mask = (n == 0)
    n[new_mask] = 1
    mean_image = np.sum(image, axis=0) / n

    return (mean_image, new_mask)

##########################################################################################

import unittest

class Test_robust_mean(unittest.TestCase):

    def runTest(self):
        pass        #### TBD

############################################
# Execute from command line...
############################################
if __name__ == '__main__':
    unittest.main(verbosity=2)
##########################################################################################
