##########################################################################################
# psiops/gaussian_filter.py
##########################################################################################

import numpy as np
from psiops._utils import _check_image, _check_tuple, _check_return
from scipy.ndimage import gaussian_filter as _unmasked_gaussian_filter


def gaussian_filter(image, sigma, mask=None, *, maskval=None, weights=None, nans=False,
                    mode='masked', cval=0., order=0, returns=None):
    """Gaussian filter of an array of images, allowing for masked and/or non-uniformly
    weighted pixels.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
            This can be a MaskedArray.
        sigma (scalar): The standard deviation to use for the Gaussian filter. Provide two
            values to use different standard deviations along the two trailing (spatial)
            axes.
        mask (array, optional): Boolean mask array, equal to True where the values in
            `image` are to be ignored. It is broadcasted to the shape of `image` if
            necessary.
        maskval (scalar, optional): A value that should be masked wherever it appears in
            `image`. This can be used used instead of or in addition to the `mask`.
        weights (array, optional): Weight array specifying the possibly unequal weights
            associated with the pixels in `image`. A weight of zero is equivalent to a
            `mask` value of True. This can be provided in addition to or instead of the
            `mask` or `maskval`. It is broadcasted to the shape of `image` if necessary.
            Values should never be negative.
        nans (bool, optional): True to check `image` for NaNs and interpret them as masked
            values.
        mode (str, optional):
            The method for handling locations outside the input image boundary, one of:

            * "masked": Values outside the boundary are masked.
            * "constant" (`k k k k | a b c d | k k k k`): Assume all exterior values equal
              a constant defined by `cval`.
            * "nearest" (`a a a a | a b c d | d d d d`): Duplicate the nearest edge
              values.
            * "wrap" (`a b c d | a b c d | a b c d`): Wrap values from one edge of the
              image to the other.
            * "reflect" (`d c b a | a b c d | d c b a`): Reflect pixels near each edge of
              the image, where pixels at the edge appear twice ("whole-sample symmetric").
            * "mirror" (`c d c b | a b c d | c b a b`): Reflect pixels near each edge of
              the image, where pixels at the edge appear only once ("half-sample
              symmetric").
        cval (float, int, bool, complex, or None):
            If mode is "constant", the numeric value to fill in for areas outside the
            boundaries of the input array. The value is casted to the dtype of `image`;
            use None to indicate that values outside the boundaries are masked.
        order (int or tuple[int], optional):
            The order of the filter along each axis, given as a single value or a tuple of
            two values if the order is different across the two image axes. An order of 0
            corresponds to convolution with a Gaussian kernel. A positive order
            corresponds to convolution with that derivative of a Gaussian.
        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and weight
            array), or "imw" (image, mask, and weight array).

    Returns:
        array or tuple: `filtered` or (`filtered`[, `new_mask`][, `new_weights`]):

        * `filtered` (array): The floating-point, Gaussian-filtered image array, with the
          same shape as `image`. If `image` is a MaskedArray, this will also be a
          MaskedArray. If `maskval` is specified, any masked elements in this array will
          be filled with this value. Otherwise, if `nans` is True, masked pixels will be
          filled with NaN.
        * `new_mask` (array): The new mask array, True wherever all the pixels contibuting
          to the image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights` (np.ndarray[float]): The weight array, equal to the
          gaussian-weighted mean of the weights of the elements that contributed to each
          element in `filtered`. By default, this is returned if `weights` was provided;
          use the `returns` input to override this default.

    Notes:
        Let `I` be an image and `F` be the result of applying a Gaussian filter to `I`.
        `F` is the result of convolving `I` over function `G`. Written as a sum over
        indices `ii` and `jj`::

            F[i,j] = sum[ii,jj]( G[ii,jj] * I[i-ii,j-jj] ) /
                     sum[ii,jj]( G[ii,jj] )

        Each pixel in `F` can be understood as a weighted mean of the nearby pixels of
        `I`, where `G` is the array defining the Gaussian weights.

        For a weighted image, we need to adjust the formula for `F` for the weighted mean.
        Let `W` be the weight array. The new weighted mean is::

            F[i,j] = sum[ii,jj](G[ii,jj] * (I*W)[i-ii,j-jj]) /
                     sum[ii,jj](G[ii,jj] *     W[i-ii,j-jj])

        This can be obtained by taking the ratio of the results of two Gaussian-filtered
        images, `I*W` and `W`.
    """

    # Interpret array and mask
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              floats=True)
    if info.fill_value is None:  # don't leave NaNs in the array unless that's intended
        info.fill_value = 0

    # Interpret sigma and order
    sigma = _check_tuple(sigma, 'sigma', floats=True,  negs=False, zeros=True)
    order = _check_tuple(order, 'order', floats=False, negs=False, zeros=True)

    # Leading axes are not filtered, so set sigma and order to zero
    isigma = (image.ndim-2) * (0,) + sigma
    iorder = (image.ndim-2) * (0,) + order

    # Without a mask, use the SciPy.ndimage version
    if mask is None and not mode == 'masked':
        filtered_image = _unmasked_gaussian_filter(image, isigma, mode=mode, cval=cval,
                                                   order=iorder)
        return _check_return(filtered_image, None, None, info)

    # If the image is completely masked, there's nothing to do
    if np.all(mask):
        new_mask = np.ones(mask.shape, dtype=np.bool_)
        return _check_return(image, new_mask, None, info)

    # Filter the weights
    if weights is None:
        weights = np.ones(image.shape[-2:]) if mask is None else np.logical_not(mask)
    wsigma = (weights.ndim-2) * (0,) + sigma
    worder = (weights.ndim-2) * (0,) + order
    if mode == 'masked':
        wmode = 'constant'
        wval = 0.
    else:
        wmode = mode
        wval = np.max(weights)
    filtered_weights = _unmasked_gaussian_filter(weights, sigma=wsigma, mode=wmode,
                                                 cval=wval, order=worder)

    # Filter the weighted image
    isigma = (image.ndim-2) * (0,) + sigma
    iorder = (image.ndim-2) * (0,) + order
    filtered_image = _unmasked_gaussian_filter(image * weights, sigma=isigma, mode=mode,
                                               cval=cval or 0., order=iorder)
    filtered_image /= filtered_weights

    return _check_return(filtered_image, None, filtered_weights, info)

##########################################################################################
