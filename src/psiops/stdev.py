##########################################################################################
# psiops/stdev.py
##########################################################################################

import numpy as np
from psiops.variance import variance, variance_filter


def stdev(image, mask=None, *, maskval=None, weights=None, nans=False, axis=None,
          keepdims=False, factors=None, stdtype='reliability', returns=None):
    """Standard deviation among the pixels in an array of images, accounting for masked
    pixels and/or weights.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
            This can be a MaskedArray. Must be at least 3-D.
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
        axis (int or tuple, optional): The axis or axes over which to operate, given  as
            an integer or tuple of integers; None to operate over all but the last two
            (spatial) axes. Note that negative axes count backwards from the first spatial
            axis, so axis=-1 actually refers to the third-to-last axis of the image array.
        keepdims (bool, optional): If True, the axes that are reduced are left in the
            result as dimensions with size one. With this option, the results will
            broadcast correctly against the input arrays.
        factors (array or array-like, optional):  Array of weights to be applied to the
            non-spatial axes of the `image` array. It is broadcasted to the shape of the
            image after excluding the image's trailing two (spatial) axes.
        stdtype (str, optional): The type of the standard deviation estimate:

            * "biased": The divisor used in the standard deviation calculation is the
              number of unmasked items or the sum of the weights. This is equivalent to
              using `ddof=0` in `numpy.std()`.
            * "unbiased" or "frequency": The divisor is one less than that for the
              "biased" standard deviation, providing an unbiased estimate. For uniform
              weights, this is equivalent to using `ddof=1` in `numpy.std()`. For
              non-uniform weights, this treats each weight factor is equivalent to the
              frequency or number of measurements with the associated value.
            * "reliability": This provides an unbiased estimate of the standard deviation
              if `weights` and `factors` are interpreted as describing the relative
              trustworthiness or uncertainty in individual measurements. For an unweighted
              standard deviation, this is equivalent to "unbiased".

            See further info here:
                https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Related_concepts

        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and weight
            array), or "imw" (image, mask, and weight array).

    Returns:
        (array or tuple): `stdev_image` or (`stdev_image`[, `new_mask`][, `new_weights`]):

        * `stdev_image` (array): The floating-point standard deviation array. If `image`
          is a MaskedArray, this will also be a MaskedArray. If `maskval` was specified,
          any masked pixels in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask` (array): The new mask array, True wherever all the image pixels along
          the specified `axis` are masked. By default, this is returned if `mask` is
          provided; use `returns` to override the default behavior.
        * `new_weights` (array): The weight array, equal to the sum of the weights of the
          elements that contributed to each pixel in `stdev_image`. By default, this is
          returned if `weights` is provided; use the `returns` to override the default
          behavior. If `weights` is not provided, this is the integer number of unmasked
          pixels that contributed to each new pixel's value.
    """

    if stdtype not in {'biased', 'unbiased', 'frequency', 'reliability'}:
        raise ValueError(f'unrecognized value for stdtype: {stdtype!r}')

    results = variance(image, mask=mask, maskval=maskval, weights=weights, nans=nans,
                       axis=axis, factors=factors, vartype=stdtype, returns=returns)

    if isinstance(results, np.ndarray):
        return np.sqrt(results)

    return (np.sqrt(results[0]),) + results[1:]


def stdev_filter(image, footprint, *, mask=None, maskval=None, weights=None, nans=False,
                 stdtype='reliability', returns=None):
    """Filter this image such that each new pixel contains the standard deviation among
    the pixels within the specified footprint.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
            This can be a MaskedArray.
        footprint (array, int, or tuple): The 2-D boolean footprint array or else an
            integer or tuple of two integers defining the rectangular shape of the
            footprint.
        mask (array, optional): Boolean mask array with the same shape as `image` and
            equal to True where the values in `image` are to be ignored.
        maskval (scalar, optional): A value that should be masked wherever it appears in
            `image`. This can be used used instead of or in addition to the `mask`.
        weights (array, optional): Weight array specifying the possibly unequal weights
            associated with the pixels in `image`. A weight of zero is equivalent to a
            `mask` value of True. This can be provided in addition to or instead of the
            `mask` or `maskval`.
        nans (bool, optional): True to check `image` for NaNs and interpret them as masked
            values.
        stdtype (str, optional): The type of the standard deviation estimate:

            * "biased": The divisor used in the standard deviation calculation is the
              number of unmasked items or the sum of the weights. This is equivalent to
              using `ddof=0` in `numpy.std()`.
            * "unbiased" or "frequency": The divisor is one less than that for the
              "biased" standard deviation, providing an unbiased estimate. For uniform
              weights, this is equivalent to using `ddof=0` in `numpy.std()`. For
              non-uniform weights, this treats each weight factor is equivalent to the
              frequency or number of measurements with the associated value.
            * "reliability": This provides an unbiased estimate of the standard deviation
              if `weights` and `factors` are interpreted as describing the relative
              trustworthiness or uncertainty in individual measurements. For an unweighted
              standard deviation, this is equivalent to "unbiased".

            See further info here:
                https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Related_concepts

        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and weight
            array), or "imw" (image, mask, and weight array).

    Returns:
        (array or tuple): `stdev_image` or (`stdev_image`[, `new_mask`][, `new_weights`]):

        * `stdev_image` (array): The floating-point standard deviation array. If `image`
          is a MaskedArray, this will also be a MaskedArray. If `maskval` was specified,
          any masked pixels in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask` (array): The new mask array, True wherever all the pixels contibuting
          to the image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights` (array): The weight array, equal to the sum of the weights of the
          elements that contributed to each pixel in `stdev_image`. By default, this is
          returned if `weights` is provided; use the `returns` to override the default
          behavior. If `weights` is not provided, this is the integer number of unmasked
          pixels that contributed to each new pixel's value.
    """

    if stdtype not in {'biased', 'unbiased', 'frequency', 'reliability'}:
        raise ValueError(f'unrecognized value for stdtype: {stdtype!r}')

    results = variance_filter(image, footprint, mask=mask, maskval=maskval,
                              weights=weights, nans=nans, vartype=stdtype,
                              returns=returns)

    if isinstance(results, np.ndarray):
        return np.sqrt(results)

    return (np.sqrt(results[0]),) + results[1:]

##########################################################################################
