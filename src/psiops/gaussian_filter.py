##########################################################################################
# psiops/gaussian_filter.py
##########################################################################################

import numpy as np
from scipy.ndimage import gaussian_filter as _unmasked_gaussian_filter

from psiops._utils import _check_tuple
from psiops._validation import _check_image, _check_return


def gaussian_filter(
    image: np.ndarray,
    sigma: float | tuple[float, float],
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    mode: str = 'masked',
    cval: float = 0.,
    order: int | tuple[int, int] = 0,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Gaussian filter of an array of images, allowing for masked and/or non-uniformly
    weighted pixels.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        sigma: The standard deviation to use for the Gaussian filter. Provide two values
            to use different standard deviations along the two trailing (spatial) axes.
        mask: Boolean mask array, equal to True where the values in `image` are to be
            ignored. It is broadcasted to the shape of `image` if necessary.
        maskval: A value that should be masked wherever it appears in `image`. This can
            be used instead of or in addition to the `mask`.
        weights: Weight array specifying the possibly unequal weights associated with
            the pixels in `image`. A weight of zero is equivalent to a `mask` value of
            True. This can be provided in addition to or instead of the `mask` or
            `maskval`. It is broadcasted to the shape of `image` if necessary. Values
            should never be negative.
        nans: True to check `image` for NaNs and interpret them as masked values.
        mode: The method for handling locations outside the input image boundary, one
            of:

            * "masked": Values outside the boundary are masked.
            * "constant" (`k k k k | a b c d | k k k k`): Assume all exterior values
              equal a constant defined by `cval`.
            * "nearest" (`a a a a | a b c d | d d d d`): Duplicate the nearest edge
              values.
            * "wrap" (`a b c d | a b c d | a b c d`): Wrap values from one edge of
              the image to the other.
            * "reflect" (`d c b a | a b c d | d c b a`): Reflect pixels near each
              edge of the image, where pixels at the edge appear twice ("whole-sample
              symmetric").
            * "mirror" (`c d c b | a b c d | c b a b`): Reflect pixels near each
              edge of the image, where pixels at the edge appear only once
              ("half-sample symmetric").
        cval: If mode is "constant", the numeric value to fill in for areas outside
            the boundaries of the input array. The value is cast to the dtype of
            `image`; use None to indicate that values outside the boundaries are masked.
        order: The order of the filter along each axis, given as a single value or a
            tuple of two values if the order is different across the two image axes. An
            order of 0 corresponds to convolution with a Gaussian kernel. A positive
            order corresponds to convolution with that derivative of a Gaussian.
        returns: Used to override the default quantity or quantities to return, one of
            "i" (image only), "im" (image and mask), "iw" (image and weight array), or
            "imw" (image, mask, and weight array).

    Returns:
        `filtered` or (`filtered`[, `new_mask`][, `new_weights`]):

        * `filtered`: The floating-point, Gaussian-filtered image array, with the same
          shape as `image`. If `image` is a MaskedArray, this will also be a MaskedArray.
          If `maskval` is specified, any masked elements in this array will be filled with
          this value. Otherwise, if `nans` is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to the
          image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the Gaussian-weighted mean of the
          weights of the elements that contributed to each element in `filtered`. By
          default, this is returned if `weights` was provided; use `returns` to override
          this default.

    Raises:
        ValueError: If `sigma` or `order` values are invalid.
        ValueError: If any image inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.

    Notes:
        Let `I` be an image and `F` be the result of applying a Gaussian filter to
        `I`. `F` is the result of convolving `I` over function `G`. Written as a sum
        over indices `ii` and `jj`::

            F[i,j] = sum[ii,jj]( G[ii,jj] * I[i-ii,j-jj] ) /
                     sum[ii,jj]( G[ii,jj] )

        Each pixel in `F` can be understood as a weighted mean of the nearby pixels of
        `I`, where `G` is the array defining the Gaussian weights.

        For a weighted image, we need to adjust the formula for `F` for the weighted
        mean. Let `W` be the weight array. The new weighted mean is::

            F[i,j] = sum[ii,jj](G[ii,jj] * (I*W)[i-ii,j-jj]) /
                     sum[ii,jj](G[ii,jj] *     W[i-ii,j-jj])

        This can be obtained by taking the ratio of the results of two
        Gaussian-filtered images, `I*W` and `W`.
    """

    # Interpret array and mask
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              floats=True, returns=returns)
    if info.fill_value is None:  # don't leave NaNs in the array unless that's intended
        info.fill_value = 0

    # Interpret sigma and order
    sigma = _check_tuple(sigma, 'sigma', floats=True,  negs=False, zeros=True)
    order = _check_tuple(order, 'order', floats=False, negs=False, zeros=True)

    # Leading axes are not filtered, so set sigma and order to zero
    isigma = (image.ndim-2) * (0,) + sigma
    iorder = (image.ndim-2) * (0,) + order

    # Without a mask, use the SciPy.ndimage version
    if mask is None and mode != 'masked':
        filtered_image = _unmasked_gaussian_filter(image, isigma, mode=mode, cval=cval,
                                                   order=iorder)
        return _check_return(filtered_image, None, None, info)

    # If the image is completely masked, there's nothing to do
    if mask is not None and np.all(mask):
        new_mask = np.ones(image.shape, dtype=np.bool_)
        return _check_return(image, new_mask, None, info)

    # Masked locations carry zero weight, but a masked NaN (or maskval) value would still
    # propagate through the `image * weights` product as NaN * 0 == NaN. Replace masked
    # values with zero so they have no effect on the filtered result.
    if mask is not None:
        bmask = np.broadcast_to(mask, image.shape)
        if np.any(bmask) and not np.all(np.isfinite(image[bmask])):
            if not info.image_is_copy:
                image = image.copy()
                info.image_is_copy = True
            image[bmask] = 0.

    # Filter the weights. A weight array must be floating-point; a boolean array would be
    # rounded back to 0/1 by the filter and corrupt the weighted normalization.
    if weights is None:
        if mask is None:
            weights = np.ones(image.shape[-2:])
        else:
            weights = np.logical_not(mask).astype(np.float64)
    wsigma = (weights.ndim-2) * (0,) + sigma
    worder = (weights.ndim-2) * (0,) + order
    # In "masked" mode, locations outside the boundary contribute no weight, which is
    # equivalent to filtering both the weights and the weighted image with constant zeros
    # outside the boundary.
    if mode == 'masked':
        wmode = imode = 'constant'
        wval = 0.
        ival = 0.
    else:
        wmode = imode = mode
        wval = np.max(weights)
        ival = cval or 0.
    filtered_weights = _unmasked_gaussian_filter(weights, sigma=wsigma, mode=wmode,
                                                 cval=wval, order=worder)

    # Filter the weighted image
    isigma = (image.ndim-2) * (0,) + sigma
    iorder = (image.ndim-2) * (0,) + order
    filtered_image = _unmasked_gaussian_filter(image * weights, sigma=isigma, mode=imode,
                                               cval=ival, order=iorder)
    # Locations with no contributing weight produce expected 0/0 -> NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        filtered_image /= filtered_weights

    # The weights may have fewer leading axes than the image (e.g. a 2-D mask over a
    # stack of images); broadcast them so the returned weights/mask match the result.
    if filtered_weights.shape != filtered_image.shape:
        filtered_weights = np.broadcast_to(filtered_weights,
                                           filtered_image.shape).copy()

    return _check_return(filtered_image, None, filtered_weights, info)

##########################################################################################
