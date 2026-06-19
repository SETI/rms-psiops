##########################################################################################
# psiops/mean.py
##########################################################################################

import numpy as np
from psiops._utils import (_apply_op_as_filter, _check_axis, _check_image, _check_return,
                           _merge_weights, _pixel_area, _use_shortcuts)


def mean(image, mask=None, *, maskval=None, weights=None, nans=False, axis=None,
         keepdims=False, factors=None, returns=None):
    """Mean or weighted mean of an array of images, excluding masked pixels.

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
        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and weight
            array), or "imw" (image, mask, and weight array).

    Returns:
        (array or tuple): `mean_image` or (`mean_image`[, `new_mask`][, `new_weights`]):

        * `mean_image` (array): The floating-point mean image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` is specified, any
          masked pixels in this array will be filled with this value. Otherwise, if `nans`
          is True, masked pixels will be filled with NaN.
        * `new_mask` (array): The new mask array, True wherever all the image pixels along
          the specified `axis` are masked. By default, this is returned if `mask` is
          provided; use `returns` to override the default behavior.
        * `new_weights` (array): The weight array, equal to the sum of the weights of the
          elements that contributed to each pixel in `max_image`. By default, this is
          returned if `weights` is provided; use the `returns` to override the default
          behavior. If `weights` is not provided, this is the integer number of unmasked
          pixels that contributed to each new pixel's value.
    """

    # Interpret the image inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=False, floats=True, three=True,
                                              returns=returns)

    # Identify the axes
    axis = _check_axis(axis, image.shape)
    if keepdims:
        info.axis = axis
    info.pixel_area = _pixel_area(axis, image.shape)

    # Evaluate and return
    (mean_image, new_mask, new_weights) = _mean(image, mask, weights, info, axis=axis,
                                                factors=factors)
    return _check_return(mean_image, new_mask, new_weights, info)


def _mean(image, mask, weights, info, axis, factors=None):
    """Internal implementation of mean()."""

    # Never leave NaNs in the array unless that's intended
    if info.fill_value is None:
        info.fill_value = 0

    # Handle the unweighted cases
    if factors is None and _use_shortcuts():
        if mask is None:
            mean_image = np.mean(image, axis=axis)
            return (mean_image, None, None)

        if weights is None:
            if not info.image_is_copy:
                image = image.copy()
            mask = np.broadcast_to(mask, image.shape)
            image[..., mask] = 0
            image_sum = np.sum(image, axis=axis)
            new_weights = info.pixel_area - np.sum(mask, axis=axis)
            mean_image = image_sum / new_weights
            return (mean_image, None, new_weights)

    # Combine weights, mask, and factors
    weights = _merge_weights(mask, weights, factors)
    weights = np.broadcast_to(weights, image.shape)

    # Calculate the mean and weights
    new_weights = np.sum(weights, axis=axis)
    mean_image = np.sum(weights * image, axis=axis) / new_weights

    return (mean_image, None, new_weights)


def mean_filter(image, footprint, *, mask=None, maskval=None, weights=None, nans=False,
                returns=None):
    """Filter this image such that each new pixel is the mean over the specified
    footprint.

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
        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and weight
            array), or "imw" (image, mask, and weight array).

    Returns:
        (array or tuple): `mean_image` or (`mean_image`[, `new_mask`][, `new_weights`]):

        * `mean_image` (array): The floating-point mean-filtered image array, with the
          same shape as `image`. If `image` contains integers or bools, then this array
          will have dtype `np.float64`; otherwise, the dtype of `image` is preserved. If
          `image` is a MaskedArray, this will also be a MaskedArray. If `maskval` is
          specified, any masked elements in this array will be filled with this value.
          Otherwise, if `nans` is True, masked pixels will be filled with NaN.
        * `new_mask` (array): The new mask array, True wherever all the pixels contibuting
          to the image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights` (array): The weight array, equal to the sum of the weights of the
          elements that contributed to each pixel in `mean_image`. By default, this is
          returned if `weights` is provided; use the `returns` to override this default
          behavior. If `weights` is not provided, this is the integer number of unmasked
          pixels that contributed to each new pixel's value.
    """

    # Interpret the image inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, floats=True, returns=returns)

    # Filter
    mean_image, new_mask, new_weights = _apply_op_as_filter(_mean, image, footprint,
                                                            mask=mask, weights=weights,
                                                            info=info)
    return _check_return(mean_image, new_mask, new_weights, info)

##########################################################################################
