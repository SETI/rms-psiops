##########################################################################################
# psiops/maximum.py
##########################################################################################

import numpy as np
import numpy.typing as npt
import scipy.ndimage

from psiops._filter import _apply_op_as_filter, _use_shortcuts
from psiops._utils import _ImageInfo, _check_axis, _pixel_area
from psiops._validation import _check_image, _check_return


def maximum(
    image: npt.ArrayLike,
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Maximum of an array of images, excluding masked pixels.

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
        returns: Used to override the default quantity or quantities to return, one of "i"
            (image only), "im" (image and mask), "iw" (image and weight array), or "imw"
            (image, mask, and weight array).

    Returns:
        `max_image` or (`max_image`[, `new_mask`][, `new_weights`]):

        * `max_image`: The maximum image array, with the same dtype as `image`. If `image`
          is a MaskedArray, this will also be a MaskedArray. If `maskval` was specified,
          any masked pixels in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the image pixels along the
          specified `axis` are masked. By default, this is returned if `mask` is provided;
          use `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `max_image`. By default, this is returned if
          `weights` is provided; use `returns` to override the default behavior. If
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
                                              comps=False, three=True, returns=returns)

    # Identify the axes
    axis = _check_axis(axis, image.shape)
    if keepdims:
        info.axis = axis
    info.pixel_area = _pixel_area(axis, image.shape)

    # Evaluate and return
    max_image, new_mask, new_weights = _maximum(image, mask, weights, info, axis=axis)
    return _check_return(max_image, new_mask, new_weights, info)


def _maximum(
    image: np.ndarray,
    mask: np.ndarray | None,
    weights: np.ndarray | None,
    info: _ImageInfo,
    axis: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Internal implementation of maximum().

    Parameters:
        image: Image array.
        mask: Boolean mask array, or None if unmasked.
        weights: Weight array, or None.
        info: The _ImageInfo object returned by _check_image.
        axis: Tuple of axes over which to compute the maximum.

    Returns:
        Tuple of (`max_image`, `new_mask`, `new_weights`), where `new_mask` and
        `new_weights` may each be None depending on the requested return type.
    """

    # Handle the unmasked case
    if mask is None:
        new_image = np.max(image, axis=axis)
        return (new_image, None, None)

    # Define an `ignore` value less than the minimum in the image array
    original_dtype = image.dtype
    if image.dtype.kind == 'b':
        ignore = -1
        image = image.view(np.int8)     # convert bools to bytes so -1 can be represented
    elif image.dtype.kind == 'f':
        ignore = -np.inf
    else:
        minval = image[..., ~mask].min()
        ignore = minval - 1
        if minval <= np.ma.maximum_fill_value(image):
            if image.dtype.itemsize < 8:
                image = image.astype(np.int64)
            else:
                ignore = minval         # calculation might be impossible; use best option

    mask = np.broadcast_to(mask, image.shape)
    weights = weights and np.broadcast_to(weights, image.shape)

    # Replace all unweighted values with the `ignore` flag
    if not info.image_is_copy:
        image = image.copy()
    image[mask] = ignore

    # Evaluate the new arrays
    new_image = np.max(image, axis=axis)

    if 'w' in info.returns:
        new_mask = None
        if weights is None:
            new_weights = np.sum(np.logical_not(mask), axis=axis)
        else:
            new_weights = np.sum(weights, axis=axis)
    else:
        new_mask = new_image == ignore
        new_weights = None

    # Use zero as a fill value if not otherwise specified
    if info.fill_value is None:
        info.fill_value = 0

    # Indicate the return dtype
    info.dtype = original_dtype

    return (new_image, new_mask, new_weights)


def maximum_filter(
    image: npt.ArrayLike,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = None,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Filter this image such that each new pixel is the maximum over the specified
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
        `max_image` or (`max_image`[, `new_mask`][, `new_weights`]):

        * `max_image`: The maximum-filtered image array, with the same shape and dtype as
          `image`. If `image` is a MaskedArray, this will also be a MaskedArray. If
          `maskval` is specified, any masked elements in this array will be filled with
          this value. Otherwise, if `nans` is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to the
          image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `max_image`. By default, this is returned if
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
                                              comps=False, returns=returns)

    # Handle unmasked case
    if mask is None and _use_shortcuts():
        if np.asarray(footprint).dtype.kind != 'b':
            size = footprint
            footprint = None
        else:
            size = None

        max_image = scipy.ndimage.maximum_filter(image, size=size, footprint=footprint,
                                                 mode='nearest', axes=(-2, -1))
        return _check_return(max_image, None, None, info)

    # Otherwise, perform the full evaluation
    max_image, new_mask, new_weights = _apply_op_as_filter(_maximum, image, footprint,
                                                           mask=mask, weights=weights,
                                                           info=info)
    return _check_return(max_image, new_mask, new_weights, info)

##########################################################################################
