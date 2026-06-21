##########################################################################################
# psiops/mean.py
##########################################################################################

import numpy as np
import numpy.typing as npt

from psiops._filter import _apply_op_as_filter, _use_shortcuts
from psiops._utils import _check_axis, _ImageInfo, _merge_weights, _pixel_area
from psiops._validation import _check_image, _check_return


def mean(
    image: npt.ArrayLike,
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    factors: npt.ArrayLike | None = None,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Mean or weighted mean of an array of images, excluding masked pixels.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray. Must be at least 3-D.
        mask: Boolean mask array, equal to True where the values in `image` are to be
            ignored. It is broadcasted to the shape of `image` if necessary.
        maskval: A value that should be masked wherever it appears in `image`. This can
            be used instead of or in addition to the `mask`.
        weights: Weight array specifying the possibly unequal weights associated with the
            pixels in `image`. A weight of zero is equivalent to a `mask` value of True.
            This can be provided in addition to or instead of the `mask` or `maskval`. It
            is broadcasted to the shape of `image` if necessary. Values should never be
            negative.
        nans: True to check `image` for NaNs and interpret them as masked values.
        axis: The axis or axes over which to operate, given as an integer or tuple of
            integers; None to operate over all but the last two (spatial) axes. Note that
            negative axes count backwards from the first spatial axis, so axis=-1 actually
            refers to the third-to-last axis of the image array.
        keepdims: If True, the axes that are reduced are left in the result as dimensions
            with size one. With this option, the results will broadcast correctly against
            the input arrays.
        factors: Array of weights to be applied to the non-spatial axes of the `image`
            array. It is broadcasted to the shape of the image after excluding the
            image's trailing two (spatial) axes.
        returns: Used to override the default quantity or quantities to return, one of "i"
            (image only), "im" (image and mask), "iw" (image and weight array), or "imw"
            (image, mask, and weight array).

    Returns:
        `mean_image` or (`mean_image`[, `new_mask`][, `new_weights`]):

        * `mean_image`: The floating-point mean image array. If `image` is a MaskedArray,
          this will also be a MaskedArray. If `maskval` is specified, any masked pixels in
          this array will be filled with this value. Otherwise, if `nans` is True, masked
          pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the image pixels along the
          specified `axis` are masked. By default, this is returned if `mask` is provided;
          use `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `mean_image`. By default, this is returned if
          `weights` is provided; use `returns` to override the default behavior. If
          `weights` is not provided, this is the integer number of unmasked pixels that
          contributed to each new pixel's value.

    Raises:
        ValueError: If `image` has fewer than three dimensions, or if `axis` is invalid.
        ValueError: If `mask` or `weights` have shapes incompatible with `image`.
        ValueError: If `returns` is not a valid option.
        TypeError: If `image` dtype is not numeric.
        IndexError: If an `axis` value is out of range.
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


def _mean(
    image: np.ndarray,
    mask: np.ndarray | None,
    weights: np.ndarray | None,
    info: _ImageInfo,
    axis: tuple[int, ...],
    factors: npt.ArrayLike | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Internal implementation of mean().

    Parameters:
        image: Image array.
        mask: Boolean mask array, or None if unmasked.
        weights: Weight array, or None.
        info: The _ImageInfo object returned by _check_image.
        axis: Tuple of axes over which to compute the mean.
        factors: Optional array of per-image weight factors.

    Returns:
        Tuple of (`mean_image`, `new_mask`, `new_weights`), where `new_mask` and
        `new_weights` may each be None depending on the requested return type.
    """

    # Never leave NaNs in the array unless that's intended
    if info.fill_value is None:
        info.fill_value = 0

    # Handle the unweighted cases
    if factors is None and _use_shortcuts():
        if mask is None and weights is None:
            mean_image = np.mean(image, axis=axis)
            return (mean_image, None, None)

        if weights is None:
            if not info.image_is_copy:
                image = image.copy()
            mask = np.broadcast_to(mask, image.shape)
            image[..., mask] = 0
            image_sum = np.sum(image, axis=axis)
            new_weights = info.pixel_area - np.sum(mask, axis=axis)
            # Fully masked pixels produce 0/0 -> NaN, which is expected
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_image = image_sum / new_weights
            return (mean_image, None, new_weights)

    # Combine weights, mask, and factors
    weights = _merge_weights(mask, weights, factors)

    # With no mask, weights, or factors, every pixel has uniform weight: a plain mean.
    if weights is None:
        mean_image = np.mean(image, axis=axis)
        return (mean_image, None, None)

    weights = np.broadcast_to(weights, image.shape)

    # Calculate the mean and weights (fully masked pixels yield expected 0/0 NaNs)
    new_weights = np.sum(weights, axis=axis)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean_image = np.sum(weights * image, axis=axis) / new_weights

    return (mean_image, None, new_weights)


def mean_filter(
    image: npt.ArrayLike,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = None,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Filter this image such that each new pixel is the mean over the specified
    footprint.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        footprint: The 2-D boolean footprint array or else an integer or tuple of two
            integers defining the rectangular shape of the footprint.
        mask: Boolean mask array with the same shape as `image` and equal to True where
            the values in `image` are to be ignored.
        maskval: A value that should be masked wherever it appears in `image`. This can
            be used instead of or in addition to the `mask`.
        weights: Weight array specifying the possibly unequal weights associated with the
            pixels in `image`. A weight of zero is equivalent to a `mask` value of True.
            This can be provided in addition to or instead of the `mask` or `maskval`.
        nans: True to check `image` for NaNs and interpret them as masked values.
        returns: Used to override the default quantity or quantities to return, one of "i"
            (image only), "im" (image and mask), "iw" (image and weight array), or "imw"
            (image, mask, and weight array).

    Returns:
        `mean_image` or (`mean_image`[, `new_mask`][, `new_weights`]):

        * `mean_image`: The floating-point mean-filtered image array, with the same shape
          as `image`. If `image` contains integers or bools, then this array will have
          dtype `np.float64`; otherwise, the dtype of `image` is preserved. If `image` is
          a MaskedArray, this will also be a MaskedArray. If `maskval` is specified, any
          masked elements in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to the
          image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `mean_image`. By default, this is returned if
          `weights` is provided; use `returns` to override this default behavior. If
          `weights` is not provided, this is the integer number of unmasked pixels that
          contributed to each new pixel's value.

    Raises:
        ValueError: If `footprint` has invalid shape or dtype.
        ValueError: If `mask` or `weights` have shapes incompatible with `image`.
        ValueError: If `returns` is not a valid option.
        TypeError: If `image` dtype is not numeric.
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
