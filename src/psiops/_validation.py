##########################################################################################
# psiops/_validation.py
##########################################################################################

import numpy as np
import numpy.typing as npt

from psiops._utils import _ImageInfo


def _check_image(
    image: npt.ArrayLike,
    mask: npt.ArrayLike | None = None,
    maskval: float | None = None,
    weights: npt.ArrayLike | None = None,
    *,
    nans: bool = False,
    returns: str | None = None,
    extra_char: str = '',
    extra_by_default: bool = False,
    floats: bool = False,
    comps: bool = False,
    two: bool = False,
    three: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, _ImageInfo]:
    """Interpret function inputs. Return (image, mask, weight array, flags).

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be an MaskedArray.
        mask: Boolean mask array, equal to True where the values in `image` are to be
            ignored. It is broadcasted to the shape of `image` if necessary.
        maskval: A value that should be masked wherever it appears in `image`. This can be
            used instead of or in addition to the `mask`.
        weights: Weight array specifying the possibly unequal weights associated with the
            pixels in `image`. A weight of zero is equivalent to a `mask` value of True.
            This can be provided in addition to or instead of the `mask` or `maskval`. It
            is broadcasted to the shape of `image` if necessary.
        nans: True to check `image` for NaNs and interpret them as masked values.
        returns: The quantities for `_check_return` to return, indicated by "i" (image
            only), "im" (image and mask), "iw" (image and weights), or "imw" (image, mask,
            and weights). Optionally, it can end with the `extra_char`.
        extra_char: A character that can be appended to the `returns` string to indicate
            that one extra quantity is to be appended to the tuple returned by
            `_check_return`.
        extra_by_default: If True and `extra_char` is present, the extra quantity will be
            included by default; otherwise it will be omitted by default.
        floats: True to convert the dtype of `image` to float64 if it does not already
            contain floats. An array of dtype float32 or complex64 is not changed.
        comps: True to allow `image` to contain complex values.
        two: True if the image must have exactly two dimensions.
        three: True if at least three image dimensions are required.

    Returns:
        A tuple (`image`, `mask`, `weights`, `info`), where:

        * `image`: The image array, which is never a MaskedArray.
        * `mask`: The mask array. This is None if all of `mask`, `maskval`, `weights` are
          None, `nans` is False, and `image` is not a MaskedArray. Otherwise, this boolean
          array will have at least two dimensions and its shape will be broadcastable to
          the shape of `image`.
        * `weights`: The floating-point weight array. This is None if `weights` is None.
          If not None, this array will have at least two dimensions and will be
          broadcastable to the shape of `image`.
        * `info`: An `_ImageInfo` object.

    Raises:
        ValueError: If `returns` is not a valid option.
        ValueError: If `image` has fewer than two dimensions.
        ValueError: If `image` has complex values and `comps` is False.
        ValueError: If `mask` or `weights` have shapes incompatible with `image`.
        TypeError: If `image` dtype is not numeric.
    """

    # Check returns value
    options = {'i', 'im', 'iw', 'imw', None}
    if extra_char:
        options |= {option + extra_char for option in options if option is not None}
    if returns not in options:
        raise ValueError(f'invalid `returns` value "{returns}"')

    # Interpret `image`, including if a MaskedArray
    image_is_copy = False
    mask_was_none = mask is None
    fill_value = None
    is_maskedarray = isinstance(image, np.ma.MaskedArray)
    if is_maskedarray:
        fill_value = image.fill_value
        if mask is None:
            mask = image.mask
        else:
            mask = image.mask | np.asarray(mask, dtype=np.bool_)
            mask_is_copy = True
        image = image.data

    if floats:
        if not isinstance(image, np.ndarray):
            image = np.asarray(image, dtype=np.float64)
            image_is_copy = True
        elif image.dtype.kind not in 'fc':      # don't convert float32
            image = np.asarray(image, dtype=np.float64)
            image_is_copy = True
    elif not isinstance(image, np.ndarray):
        image = np.asarray(image)
        image_is_copy = True

    # Check the image shape
    if two and image.ndim != 2:
        raise ValueError(f'invalid image shape {image.shape}; must be 2-D')

    if three and image.ndim < 3:
        raise ValueError(f'invalid image shape {image.shape}; must be at least 3-D')

    if image.ndim < 2:
        raise ValueError(f'invalid image shape {image.shape}; must be at least 2-D')

    # Check the image dtype
    if image.dtype.kind not in 'fuicb':
        raise TypeError(f'image dtype {image.dtype} is not numeric')

    if image.dtype.kind == 'c' and not comps:
        raise ValueError('complex image values are not supported')

    # Convert mask to array if not None
    mask_is_copy = False
    if mask is not None:
        if not isinstance(mask, np.ndarray) or mask.dtype.kind != 'b':
            mask = np.asarray(mask, dtype=np.bool_)
            mask_is_copy = True

        if mask.ndim == 0:
            mask = np.full(image.shape[-2:], bool(mask), dtype=np.bool_)
            mask_is_copy = True
        elif mask.ndim < 2:
            raise ValueError(f'illegal mask shape {mask.shape}; must be at least 2-D')

        if mask.shape[-2:] != image.shape[-2:]:
            raise ValueError(f'mask and image have incompatible shapes: {mask.shape}, '
                             f'{image.shape}')
        try:
            _ = np.broadcast_to(mask, image.shape)
        except ValueError:
            raise ValueError(f'mask and image have incompatible shapes: {mask.shape}, '
                             f'{image.shape}')

    # Update the mask based on maskval and nans
    if maskval is not None or nans:
        if mask is None:
            mask = np.zeros(image.shape, dtype=np.bool_)
            mask_is_copy = True
        elif mask.shape != image.shape:
            mask = np.broadcast_to(mask, image.shape).copy()
            mask_is_copy = True

        if not mask_is_copy:
            mask = mask.copy()
            mask_is_copy = True
        if nans:
            mask[np.isnan(image)] = True
            fill_value = np.nan
        if maskval is not None:
            mask[image == maskval] = True
            fill_value = maskval        # overrides fill_value from MaskedArray and NaN

    # Construct weight array and update mask if necessary
    weights_is_copy = False
    if weights is not None:
        if not isinstance(weights, np.ndarray) or weights.dtype.kind != 'f':
            weights = np.asarray(weights, dtype=np.float64)
            weights_is_copy = True

        if weights.ndim < 2:
            raise ValueError(f'illegal weights shape {weights.shape}; must be at least '
                             '2-D')

        if weights.shape[-2:] != image.shape[-2:]:
            raise ValueError('weights and image have incompatible shapes: '
                             f'{weights.shape}, {image.shape}')
        try:
            _ = np.broadcast_to(weights, image.shape)
        except ValueError:
            raise ValueError('weights and image have incompatible shapes: '
                             f'{weights.shape}, {image.shape}')

        # Force mask and weights to the same shape
        if mask is None:
            mask = weights == 0
            mask_is_copy = True
        else:
            if mask.shape != weights.shape:
                temp_mask, temp_weights = np.broadcast_arrays(mask, weights)
                if temp_mask.shape != mask.shape:
                    mask = temp_mask.copy()
                    mask_is_copy = True
                if temp_weights.shape != weights.shape:
                    weights = temp_weights.copy()
                    weights_is_copy = True

            # Insert zero-valued weights into mask
            mask = mask | (weights == 0.)
            mask_is_copy = True

            # Zero-out masked locations in weight array
            if not weights_is_copy:
                weights = weights.copy()
                weights_is_copy = True
            weights[mask] = 0.

    # Update the value of returns
    if returns is None:
        returns = 'i'
        if not (mask_was_none or is_maskedarray):
            returns += 'm'
        if weights is not None:
            returns += 'w'
        if extra_char and extra_by_default:
            returns += extra_char

    info = _ImageInfo(
        extra_char=extra_char,
        fill_value=fill_value,
        image_is_copy=image_is_copy,
        is_maskedarray=is_maskedarray,
        mask_is_copy=mask_is_copy,
        returns=returns,
        weights_is_copy=weights_is_copy,
    )

    return (image, mask, weights, info)


def _check_return(
    image: np.ndarray,
    mask: np.ndarray | None,
    weights: np.ndarray | None,
    info: _ImageInfo,
    *,
    extra: object = None,
    return_dtype: np.dtype | None = None,
    pixel_area: int = 1,
) -> np.ndarray | list[np.ndarray]:
    """Construct the return value for a function based on the original input info.

    Parameters:
        image: Image to return, which cannot be a MaskedArray.
        mask: Mask to return, if any. If provided, it must be a new copy.
        weights: Floating-point weight array to return, if any. If both `mask` and
            `weights` are provided, they must be compatible. In other words, `weights` is
            zero where `mask` is True, and vice-versa. This is not tested. If provided,
            it must be a new copy.
        info: The `_ImageInfo` object returned by `_check_image`.
        extra: Anything to append to the return object if `returns` contains
            `info.extra_char`.
        return_dtype: The dtype of the image array to return.
        pixel_area: The number of pixels being combined into a new result.

    Returns:
        `image` or (`image`[, `mask`][, `weights`][, `extra`]):

        * `image`: The image array, as a MaskedArray if `info.is_maskedarray` is True.
          Any masked values are replaced by `info.fill_value`.
        * `mask`: The mask array, returned if `info.returns` contains "m".
        * `weights`: The floating-point weight array, returned if `info.returns` contains
          "w".
        * `extra`: The contents of the `extra` input, returned if `returns` contains
          `info.extra_char`.

    Raises:
        ValueError: if the value of `extra` is missing.
    """

    def keepdims(array: np.ndarray | None) -> np.ndarray | None:
        if array is None or not info.axis:
            return array

        shape = list(array.shape)
        for k in info.axis:
            shape.insert(k, 1)
        return array.reshape(shape)

    # Check the inputs
    unweighted = (weights is None)
    if 'w' in info.returns and unweighted:
        if mask is None:
            weights = np.empty(image.shape[-2:], dtype=np.float64)
            weights.fill(info.pixel_area)
        else:
            weights = 1. - np.asarray(mask)
            if info.pixel_area != 1:
                weights *= info.pixel_area

    if 'm' in info.returns and mask is None:
        if unweighted:
            mask = np.zeros(image.shape[-2:], dtype=np.bool_)
        else:
            mask = (weights == 0.)

    if info.extra_char and info.extra_char in info.returns and extra is None:
        raise ValueError('missing extra quantity')

    # Fix the image dtype if necessary
    effective_dtype = return_dtype if return_dtype is not None else info.dtype
    if effective_dtype is not None:
        image = np.asarray(image, dtype=effective_dtype)

    # Update the image using the fill_value
    if info.fill_value is not None and mask is not None:
        image[mask] = info.fill_value

    # Convert the image to a MaskedArray if necessary
    if info.is_maskedarray:
        if mask is None:
            mask = False
        image = np.ma.MaskedArray(data=image, mask=mask, fill_value=info.fill_value)

    # Construct results and return
    if info.returns == 'i':
        return image

    results = [keepdims(image)]
    if 'm' in info.returns:
        results.append(keepdims(mask))
    if 'w' in info.returns:
        results.append(keepdims(weights))
    if info.extra_char and info.extra_char in info.returns:
        results.append(extra)

    return results

##########################################################################################
