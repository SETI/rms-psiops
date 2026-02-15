##########################################################################################
# psiops/gaussian_filter.py
##########################################################################################

import numpy as np
from psiops._utils import _check_image, _check_tuple, _check_return
from scipy.ndimage import gaussian_filter as _unmasked_gaussian_filter
from scipy.ndimage import minimum_filter as _unmasked_minimum_filter


def gaussian_fill(image, mask, sigma=0.5, min_weight=1.e-8, nans=False, infs=False):
    """Image with all masked pixels filled in smoothly.

    Although masked pixels should never be used for data analysis, it can be convenient to
    have images in which major visible artifacts are subdued.

    This procedure has the property that small areas of filled pixels are nearly
    indistinguishable by eye from their nearby unmasked pixels. Also, larger masked areas
    become progressively smoother toward their centers without deviating from the local
    mean.

    The procedure works as follows:
        1. Select all masked pixels that are adjacent to an unmasked pixel.
        2. Set d = 1.
        3. Blur `image` by d * sigma, taking the original mask into consideration.
        4. Replace the selected pixels by their blurred counterparts.
        5. Select all the masked pixels that are adjacent to a recently updated pixel.
        6. Increment d.
        7. Repeat steps 3-6 until there are no more unmasked pixels in step 5.

    Args:
        image:      The image as a NumPy array or MaskedArray, at least 2-D. The last two
                    array axes are treated as the spatial axes of the image.
        mask:       Used to define an optional mask or weight array:
                    - If `image` is a MaskedArray, this input is ignored.
                    - A boolean array is interpreted as a mask that is equal to True where
                      the values of `image` should be ignored;
                    - An array of integers or floats is interpreted as a weight value
                      between zero and one, where zero is unweighted (equivalent to
                      masked) and one is fully weighted;
                    - A single integer or float defines a flag value that should be masked
                      wherever it occurs within the image array;
                    - None (the default) indicates that the image array is unmasked and
                      weighted uniformly. This also prevents a new mask from being
                      returned; use mask=False if you do not wish to define an input mask
                      but do wish for a mask to be returned.
                    If an array is provided, it must be broadcastable to the shape of
                    `image`.
        sigma:      Standard deviation to use for the Gaussian filter. This is the sigma
                    value used when filling in masked pixels that are adjacent to unmasked
                    pixels. The default value of 0.5 generally provides pleasing results.
        min_weight  In the case where the mask contains weight values, the nonzero weight
                    to apply to the newly filled pixels upon return.
        nans        True to mask NaN values in `image`.
        infs        True to mask infinite values in `image`.

    Returns:
        filled      A copy of `image` with all masked pixels filled in by interpolation
                    from the nearest unmasked pixels. This array always contains
                    floating-point values and is a standard (unmasked) NumPy array even if
                    `image` is a MaskedArray.
        new_mask    A new mask or weight array, included if `mask` is not None. If `mask`
                    is a boolean array, a boolean array will be returned, but it will be
                    updated based on `nans`, `infs`, and/or a numeric flag value. If
                    `mask` is a floating-point array, then it will be returned with all
                    previously unweighted pixel locations replaced by `min_weight`, which
                    must be a small but nonzero positive value.
    """

    # Interpret the image and mask
    image, mask, weights, info = _check_image(image, mask, maskval, nans=nans, infs=infs,
                                              floats=True)

    # Without a mask, there's nothing to do
    if mask is None:
        return image

    # Define a footprint array including the four adjacent pixels
    footprint = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype='bool')

    # Initialize an array of distances to the nearest unmasked pixel
    distance = np.zeros(mask.shape, dtype=np.uint16)

    # Convert mask to boolean
    m = mask if mask.dtype.kind == 'b' else (mask == 0)

    # While there are still masked pixels...
    d = 0
    while np.any(m):

        # Select the pixels at the next greater distance
        m = _unmasked_minimum_filter(m, footprint=footprint)
        d += 1
        selection = (distance == 0) & (mask ^ m)
        distance[selection] = d

        # Blur the image by this many sigma and insert the blurred values at the selection
        blurred = gaussian_filter(image, sigma=d*sigma, mask=mask)[0]
        image[selection] = blurred[selection]

    # If there are no floating-point weights, just return the filled image
    if mask.dtype.kind == 'b':
        return (image, mask)

    # Otherwise, update the weight array with nonzero weights
    weights = mask
    zeros = weights == 0
    new_weights = weights.copy()
    new_weights[zeros] = min_weight

    return (image, new_weights)

##########################################################################################
