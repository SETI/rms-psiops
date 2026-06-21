##########################################################################################
# psiops/median.py
##########################################################################################

import warnings

import numpy as np
import numpy.typing as npt

from psiops._filter import _apply_op_as_filter, _use_shortcuts
from psiops._utils import (
    _check_axis,
    _flatten_axes,
    _ImageInfo,
    _merge_weights,
    _pixel_area,
)
from psiops._validation import _check_image, _check_return


def median(
    image: npt.ArrayLike,
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    factors: npt.ArrayLike | None = None,
    omit: int = 0,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Median or weighted median of an array of images, excluding masked pixels.

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
        omit: The index identifying extreme pixels to ignore at one end of the
            distribution before taking the median. A positive value indicates the number
            to exclude at the lower end of the sorted list of pixel values along the
            selected axes; a negative value indicates the number to exclude at the upper
            end of the sorted list.
        returns: Used to override the default quantity or quantities to return, one of "i"
            (image only), "im" (image and mask), "iw" (image and weight array), or "imw"
            (image, mask, and weight array).

    Returns:
        `median_image` or (`median_image`[, `new_mask`][, `new_weights`]):

        * `median_image`: The floating-point median image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` is specified, any
          masked pixels in this array will be filled with this value. Otherwise, if `nans`
          is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the image pixels along the
          specified `axis` are masked. By default, this is returned if `mask` is provided;
          use `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `median_image`. By default, this is returned
          if `weights` is provided; use `returns` to override the default behavior. If
          `weights` is not provided, this is the integer number of unmasked pixels that
          contributed to each new pixel's value.

    Raises:
        ValueError: If `image` has fewer than two dimensions, or if `axis` is invalid.
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
    median_image, new_mask, new_weights = _median(image, mask, weights, info, axis=axis,
                                                  factors=factors, omit=omit)
    return _check_return(median_image, new_mask, new_weights, info)


def _median(
    image: np.ndarray,
    mask: np.ndarray | None,
    weights: np.ndarray | None,
    info: _ImageInfo,
    axis: tuple[int, ...],
    factors: npt.ArrayLike | None = None,
    omit: int = 0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Internal implementation of median().

    Parameters:
        image: Image array.
        mask: Boolean mask array, or None if unmasked.
        weights: Weight array, or None.
        info: The _ImageInfo object returned by _check_image.
        axis: Tuple of axes over which to compute the median.
        factors: Optional array of per-image weight factors.
        omit: Number of extreme values to exclude before taking the median.

    Returns:
        Tuple of (`median_image`, `new_mask`, `new_weights`), where `new_mask` and
        `new_weights` may each be None depending on the requested return type.
    """

    # Never leave NaNs in the array unless that's intended
    if info.fill_value is None:
        info.fill_value = 0

    # Handle the case without leading factors
    if factors is None and omit == 0 and _use_shortcuts():
        if mask is None and weights is None:
            median_image = np.median(image, axis=axis)
            return (median_image, None, None)

        if weights is None:
            if not info.image_is_copy:
                image = image.copy()
            mask = np.broadcast_to(mask, image.shape)
            image[mask] = np.nan
            # Fully masked slices yield NaN (expected); suppress the warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                median_image = np.nanmedian(image, axis=axis)
            new_weights = info.pixel_area - np.sum(mask, axis=axis)
            # Report the per-pixel count as weights when requested; otherwise report a
            # mask that is True wherever every contributing pixel was masked, so that
            # fully masked pixels are flagged in the result (e.g. a returned MaskedArray).
            if 'w' in info.returns:
                return (median_image, None, new_weights)
            return (median_image, new_weights == 0, None)

    # Combine weights and factors. This is None only when there is no mask, no weights,
    # and no factors, in which case every pixel carries an implicit unit weight.
    weights = _merge_weights(mask, weights, factors)
    if weights is not None:
        weights = np.broadcast_to(weights, image.shape)
        weights = _flatten_axes(weights, axis, shape=image.shape)

    # Broadcast and flatten the mask the same way as the image, so it always aligns with
    # the flattened image even when it has fewer leading axes (e.g. a footprint-boundary
    # mask supplied by the filter path).
    if mask is not None:
        mask = np.broadcast_to(mask, image.shape)
        mask = _flatten_axes(mask, axis, shape=image.shape)

    # Move the selected axes to the front, then flatten them
    image = _flatten_axes(image, axis)

    # Replace all masked pixels using a flag value that is out of range, above or below
    if mask is not None:
        ignore = -np.inf if omit < 0 else np.inf
        if not info.image_is_copy:
            image = image.copy()
        image[mask] = ignore

    # Sort the image pixels and associated weights
    if weights is None:
        sorted_image = np.sort(image, axis=0)
        sorted_weights = np.ones(sorted_image.shape[0], dtype=np.int_)
    else:
        args = np.argsort(image, axis=0)
        sorted_image = np.take_along_axis(image, args, axis=0)
        sorted_weights = np.take_along_axis(weights, args, axis=0)

    # Locate the midpoint. When `omit` is nonzero, trim that many weight units from one
    # end of the sorted distribution before taking the (weighted) median: a positive value
    # trims the lower end, a negative value trims the upper end. Masked pixels already
    # carry zero weight, so they never count toward the trimmed total.
    cumweights = np.cumsum(sorted_weights, axis=0)
    total = cumweights[-1]
    lo = max(omit, 0)
    hi = total + min(omit, 0)
    new_weights = np.maximum(hi - lo, 0)
    midweight = (lo + hi) / 2.
    index_below = np.sum(cumweights <  midweight, axis=0)
    index_above = np.sum(cumweights <= midweight, axis=0)
    index_above = np.minimum(index_above, image.shape[0] - 1)

    # Interpolate the median
    indices = tuple(np.indices(image.shape[1:]))
    value_below = sorted_image[(index_below,) + indices]
    value_above = sorted_image[(index_above,) + indices]
    median_image = 0.5 * (value_below + value_above)

    return (median_image, None, new_weights)


def median_filter(
    image: npt.ArrayLike,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = None,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    omit: int = 0,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Filter this image such that each new pixel equals the median over the specified
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
        omit: The index identifying extreme pixels to ignore at one end of the
            distribution before taking the median. A positive value indicates the number
            to exclude at the lower end of the sorted list of pixel values along the
            selected axes; a negative value indicates the number to exclude at the upper
            end of the sorted list.
        returns: Used to override the default quantity or quantities to return, one of "i"
            (image only), "im" (image and mask), "iw" (image and weight array), or "imw"
            (image, mask, and weight array).

    Returns:
        `median_image` or (`median_image`[, `new_mask`][, `new_weights`]):

        * `median_image`: The floating-point median-filtered image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` is specified, any
          masked pixels in this array will be filled with this value. Otherwise, if `nans`
          is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to the
          image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `median_image`. By default, this is returned
          if `weights` is provided; use `returns` to override this default behavior. If
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
    median_image, new_mask, new_weights = _apply_op_as_filter(_median, image, footprint,
                                                              mask=mask, weights=weights,
                                                              info=info)

    return _check_return(median_image, new_mask, new_weights, info)

##########################################################################################
