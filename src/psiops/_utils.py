##########################################################################################
# psiops/_utils.py
##########################################################################################

import numbers
import numpy as np
import psutil

class _ImageInfo:
    """Structure to hold info about the function inputs and what `_check_returns` should
    return.

    Attributes:
        axis (tuple): Indices of any unit-length axes that are to be restored to the
            returned array, supporting the `keepdims` option.
        fill_value (scalar): Value to appear in the array where it is masked; otherwise,
            None. This could be NaN.
        is_maskedarray (bool): True if `image` was a MaskedArray.
        image_is_copy (bool): True if the returned image is a copy of `image`.
        mask_is_copy (bool): True if the returned mask is a copy of `mask`.
        weights_is_copy (bool): True if the returned weights array is a copy of `weights`.
        pixel_area (int): The number of pixels being combined during processsing; default
            is 1.
        extra_char (str): One extra character that can be appended to the `returns` string
            to indicate that extra information is to be appended.
        returns (str): String describing what `_check_returns` should return. One of
            "i" (image only), "im" (image and mask), "iw" (image and weights), or "imw"
            (image, mask, and weights). It might end with a trailing `extra_char`,
            indicating that `_check_returns` should also return its `extra` value.
    """

##########################################################################################
# Input argument checks
##########################################################################################

def _check_tuple(item, title, *, default=None, floats=True, negs=False, zeros=None,
                 nones=False, shape=None):
    """Validate an input as a single value or tuple of two values.

    Parameters:
        item (any): The value to check.
        title (str): Name of the parameter, used in error messages.
        default (tuple, optional): The default value to return as a tuple of two values.
        floats (bool, optional): True to allow floating-point values.
        negs (bool, optional): True to allow negative values.
        zeros (bool, optional): True to allow zero values; by default this has the same
            value as `negs`.
        nones (bool, optional): True to allow None as an input value. If `item` is None
            and `default` is specified, `default` will be returned; if `item` is None but
            no `default` is specified, a ValueError is raised.
        shape (tuple of two ints, optional): If specified, each element must be an integer
            multiple of the corresponding element in `item`.

    Returns (tuple):
        The validated, two-element tuple.

    Raises:
        ValueError: If `item` is None and `default` is None.
    """

    zeros = negs if zeros is None else zeros

    if item is None:
        if nones or default is not None:
            return default
        raise ValueError(f'missing {title}')

    if isinstance(item, (tuple, list, np.ndarray)):
        if len(item) != 2:
            raise ValueError(f'invalid {title} {tuple(item)}; two values required')
        item = tuple(item)
    else:
        item = (item, item)

    if not floats:
        if (not isinstance(item[0], numbers.Integral) or
            not isinstance(item[1], numbers.Integral)):
                raise TypeError(f'invalid {title} {item}; two integers required')

    if not negs and (item[0] < 0 or item[1] < 0):
        if zeros:
            raise ValueError(f'invalid {title} {item}; non-negative values required')
        else:
            raise ValueError(f'invalid {title} {item}; positive values required')

    if not zeros and (item[0] == 0 or item[1] == 0):
        if negs:
            raise ValueError(f'invalid {title} {item}; non-zero values required')
        else:
            raise ValueError(f'invalid {title} {item}; positive values required')

    if shape is not None and (shape[0] % item[0] != 0 or shape[1] % item[1] != 0):
        raise ValueError(f'shape {shape} is not divisible by {title} {item}')

    return item


def _check_axis(axis, shape):
    """Validate an input value for axis, given the shape of an image.

    Parameters:
        axis (int, tuple[int], or None): Axis index or a tuple of axis indices, which can
            be positive or negative. None is replaced by tuple(range(len(shape))).
        shape (tuple[int]): Overall shape of an image array. All elements must be
            positive.

    Returns:
        tuple: Revised tuple of non-negative integers in the order given. Note that the
            last two elements of the shape cannot be referenced.
    """

    shape = shape[:-2]

    if axis is None:
        return tuple(range(len(shape)))

    if isinstance(axis, numbers.Integral):
        axis = (axis,)

    try:
        axis = tuple(axis)
    except TypeError:
        raise TypeError(f'invalid axis of type {type(axis).__name__}: {repr(axis)}')

    new_axis = []
    for k in axis:
        original_k = k
        if not isinstance(k, numbers.Integral):
            raise TypeError(f'invalid axis item {repr(k)} of type {type(k).__name__}')
        if k < 0:
            k += len(shape)
        if k < 0 or k >= len(shape):
            raise IndexError(f'axis value {original_k} out of range for shape {shape}')
        if k in new_axis:
            raise ValueError(f'duplicated array axis {original_k} for shape {shape}')
        new_axis.append(k)

    return tuple(new_axis)


def _pixel_area(axes, shape):
    """The number of pixels along the specified axes."""

    lengths = [shape[k] for k in axes]
    return int(np.prod(lengths))


def _check_image(image, mask=None, maskval=None, weights=None, *, nans=False,
                 returns=None, extra_char='', extra_by_default=False, floats=False,
                 comps=False, two=False, three=False):
    """Interpret function inputs. Return (image, mask, weight array, flags).

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
            This can be an MaskedArray.
        mask (array, optional): Boolean mask array, equal to True where the values in
            `image` are to be ignored. It is broadcasted to the shape of `image` if
            necessary.
        maskval (scalar, optional): A value that should be masked wherever it appears in
            `image`. This can be used used instead of or in addition to the `mask`.
        weights (array, optional): Weight array specifying the possibly unequal weights
            associated with the pixels in `image`. A weight of zero is equivalent to a
            `mask` value of True. This can be provided in addition to or instead of the
            `mask` or `maskval`. It is broadcasted to the shape of `image` if necessary.
        nans (bool, optional): True to check `image` for NaNs and interpret them as masked
            values.
        returns (str, optional): The quantities for `_check_return` to return, indicated
            by "i" (image only), "im" (image and mask), "iw" (image and weights), or "imw"
            (image, mask, and weights). Optionally, it can end with the `extra_char`.
        extra_char (str, optional): A character that can be appended to the `returns`
            string to indicate that one extra quantity is to be appended to the tuple
            returned by `_check_return`.
        extra_by_default (bool, optional): If True and `extra_char` is present, the extra
            quantity will be included by default; otherwise it will be omitted by
            default.
        floats (bool, optional): True to convert the dtype of `image` to float64 if it
            does not already contain floats. An array of dtype float32 or complex64 is not
            changed.
        comps (bool, optional): True to allow `image` to contain complex values.
        two (bool, optional): True if the image must have exactly two dimensions.
        three (bool, optional): True if at least three image dimensions are required.

    Returns:
        A tuple (`image`, `mask`, `weights`, `info`), where:

        * `image`: The image array, which is never a MaskedArray.
        * `mask`: The mask array. This is None if all of `mask`, `maskval`, `weights`
          are None, `nans` is False, and `image` is not a MaskedArray. Otherwise, this
          boolean array will have at least two dimensions and its shape will be
          broadcastable to the shape of `image`.
        * `weights`: The floating-point weight array. This is None if `weights` is None.
          If not None, this array will have at least two dimensions and will be
          broadcastable to the shape of `image`.
        * `info`: An `_ImageInfo` object.

    Raises:
        ValueError: If any inputs are invalid.
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
        raise ValueError(f'invalid image shape {image.shape}; muse be at least 3-D')

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

        if mask.ndim < 2:
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
            mask[mask == maskval] = True
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
            mask = mask | weights == 0.
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
            returns += info.extra_char

    # Return the _ImageInfo
    info = _ImageInfo()
    info.axis = ()
    info.fill_value = fill_value
    info.is_maskedarray = is_maskedarray
    info.image_is_copy = image_is_copy
    info.mask_is_copy = mask_is_copy
    info.weights_is_copy = weights_is_copy
    info.pixel_area = 1     # default value
    info.extra_char = extra_char
    info.returns = returns

    return (image, mask, weights, info)


def _check_return(image, mask, weights, info, *,  extra=None, return_dtype=None,
                  pixel_area=1):
    """Construct the return value for a function based on the original input info.

    Parameters:
        image (array): Image to return, which cannot be a MaskedArray.
        mask (array, optional): Mask to return, if any. If provided, it must be a new
            copy.
        weights (array, optional): Floating-point weight array to return, if any. If
            both `mask` and `weights` are provided, they must be compatible. In other
            words, `weights` is zero where `mask` is True, and vice-versa. This is not
            tested. If provided, it must be a new copy.
        extra (any, optional): Anything to append to the return object if `returns`
            contains `info.extra_char`.
        return_dtype (np.dtype, optional): The dtype of the image array to return.
        pixel_area (int, optional): The number of pixels being combined into a new result.

    Returns:
        (array or tuple): `image` or (`image`[, `mask`][, `weights`][, `extra`]):

        * `image` (array): The image array, as a MaskedArray if `info.is_maskedarray` is
          True. Any masked values are replaced by `info.fill_value`.
        * `mask` (array): The mask array, returned if `info.returns` contains "m".
        * `weights` (array): The floating-point weight array, returned if `info.returns`
          contains "w".
        * `extra` (any): The contents of the `extra` input, returned if `returns` contains
          `info.extra_char`.

    Raises:
        ValueError: if the value of `extra` is missing.
    """

    def keepdims(array):
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
            weights.fill(pixels_on_axes)
        else:
            weights = 1. - np.asarray(mask)
            if pixels_on_axes != 1:
                weights *= pixels_on_axes

    if 'm' in info.returns and mask is None:
        if unweighted:
            mask = np.zeros(image.shape[-2:], dtype=np.bool_)
        else:
            mask = (weights == 0.)

    if info.extra_char in info.returns and extra is None:
        raise ValueError('missing extra quantity')

    # Fix the image dtype if necessary
    if return_dtype is not None:
        image = np.asarray(image, dtype=return_dtype)

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
    if info.extra_char in info.returns:
        results.append(extra)

    return results


def _flatten_axes(image, axes, shape=None):
    """Move selected axes to the front and flatten them. None returns None."""

    if image is None:
        return None

    if shape is not None and image.shape != shape:
        image = np.broadcast_to(image, shape)

    if isinstance(axes, numbers.Integral):
        axes = (axes,)

    naxes = len(axes)
    image = np.moveaxis(image, axes, tuple(range(naxes)))   # move to front
    image = image.reshape((-1,) + image.shape[naxes:])
    return image


def _merge_weights(mask, weights, factors):
    """Merge an optional mask, weights, and factors into a single weight array.

    Parameters:
        mask (array or None): A mask array as returned by `_check_image`.
        weights (array or None): A weight array as returned by `_check_image`.
        factors (array-like or None): An extra weight factor that applies to entire 2-D
            images.

    Returns:
        (array or none): None if all the inputs are None; otherwise, a weight array that
            can be broadcasted to the shape of the image.
    """

    if factors is not None:
        factors = np.asarray(factors)[:, np.newaxis, np.newaxis]

    if weights is None:
        if mask is None:
            return factors      # could be None
        weights = np.logical_not(mask)

    if factors is None:
        return weights

    return weights * factors

##########################################################################################
# Filter support
##########################################################################################

# Unit test support
_LAYERS_USED = 0
_TILES_USED = 0

def _apply_op_as_filter_info():
    """For unit testing. Call this after _apply_op_as_filter() to find out how many layers
    and tiles were used.
    """

    global _LAYERS_USED, _TILES_USED
    return (_LAYERS_USED, _TILES_USED)


def _apply_op_as_filter(op, image, footprint, *, mask, weights, info, **kwargs):
    """Apply this operation as a filter on the image over the specified footprint.

    Parameters:
        op (callable): The function performing the image operation.
        image (array): Image array, in which the last two axes are spatial dimensions.
            This can be a MaskedArray.
        footprint (array, int, or tuple): The 2-D boolean footprint array, or else an
            integer or tuple of two integers defining the rectangular shape of the
            footprint.
        mask (array or None): A boolean mask array, equal to True where image values
            are to be ignored. The shape is broadcasted to the shape of `image` if
            necessary.
        weights (array or None): Array of weight values where a weight of zero is
            equivalent to a mask value of True. Values should never be negative.
        info (_ImageInfo): The `_ImageInfo` object returned by `_check_image`.
        **kwargs: Any other inputs to be passed to `op`.

    Returns:
        tuple: (`filtered_image`, `mask`, `weights`), where:

        * `filtered_image` (array): The filtered image array.
        * `mask` (array or None): The mask array if `weights` is None, otherwise None.
        * `weights` (array or None): The weight array if `weights` is not None, otherwise
          None.

    Raises:
        ValueError: If any input is invalid.
    """

    global _LAYERS_USED, _TILES_USED
    _LAYERS_USED = 0
    _TILES_USED = 0

    # Interpret the image and mask
    shape = np.shape(image)[-2:]    # the 2-D shape

    # Interpret the footprint
    if isinstance(footprint, (tuple, list, np.ndarray)):
        footprint = np.asarray(footprint)
        if footprint.ndim == 2:
            fshape = footprint.shape
            if footprint.dtype.kind in 'buif':
                footprint = footprint.astype(np.bool_)
            else:
                raise ValueError(f'invalid footprint dtype: {footprint.dtype}')
        else:
            fshape = _check_tuple(footprint, 'footprint shape', floats=False, negs=False)
            footprint = np.ones(fshape, dtype=np.bool_)
    else:
        fshape = _check_tuple(footprint, 'footprint shape', floats=False, negs=False)
        footprint = np.ones(fshape, dtype=np.bool_)

    info.pixel_area = np.sum(footprint)

    # The filtering operation will employ a "sliding window array"; see
    #   https://numpy.org/doc/stable/reference/generated/
    #       numpy.lib.stride_tricks.sliding_window_view.html
    # However, it will then expand this array to its full 4-D size, possibly occupying
    # massive amounts of memory if the image is large and the footprint is large. If this
    # will require too much system memory, we need to break down the process into smaller
    # chunks.

    mask_bytes = 0 if mask is None else 1 if weights is None else weights.dtype.itemsize
    itemsize = image.dtype.itemsize + mask_bytes
    needed_bytes = image.size * footprint.size * itemsize
    usable_bytes = _usable_bytes()

    # If we have enough memory, go ahead
    if needed_bytes <= usable_bytes:
        _LAYERS_USED = 1
        _TILES_USED = 1
        return _apply_op(op, image, footprint, mask=mask, weights=weights, info=info,
                         **kwargs)

    # First, break down the image by layers
    old_shape = image.shape
    image = np.reshape(image, (-1,) + shape)    # collapse front axes to one
    filtered = np.empty(image.shape, dtype=image.dtype)

    if weights is not None:
        weights = np.broadcast_to(weights, old_shape)
        weights = np.reshape(weights, image.shape)
        new_weights = np.empty(image.shape, dtype=np.bool_)
        new_mask = None
    elif mask is not None:
        mask = np.broadcast_to(mask, old_shape)
        mask = np.reshape(mask, image.shape)
        new_mask = np.empty(image.shape, dtype=np.bool_)
        new_weights = None

    # If the layers are sufficiently small, handle them in sequence
    layer_bytes = needed_bytes // image.shape[0]
    if layer_bytes <= usable_bytes:
        lstep = usable_bytes // layer_bytes
        l0 = 0
        l1 = lstep
        while l0 < image.shape[0]:
            mslice = None if mask is None else mask[l0:l1]
            wslice = None if weights is None else weights[l0:l1]
            (temp_image, temp_mask,
             temp_weights) = _apply_op(op, image[l0:l1], footprint, mask=mslice,
                                       weights=wslice, info=info, **kwargs)
            filtered[l0:l1] = temp_image
            if new_mask is not None:
                new_mask[l0:l1] = temp_mask
            elif new_weights is not None:
                new_weights[l0:l1] = temp_weights

            l0 += lstep
            l1 = min(l0 + lstep, image.shape[0])
            _LAYERS_USED += 1

        _TILES_USED = 1
        if mask is None or isinstance(filtered, np.ma.MaskedArray):
            return filtered.reshape(old_shape)
        else:
            return (filtered.reshape(old_shape), new_mask.reshape(old_shape))

    # Otherwise, we have to break down the images into smaller tiles too
    _LAYERS_USED = image.shape[0]
    istep = max(usable_bytes // layer_bytes, fshape[0])
        # Set the lower limit on the tile size to the filter shape, because smaller values
        # become very inefficient. We hope the computer can handle it.

    ibelow = fshape[0] // 2             # number of footprint pixels below center
    iabove = fshape[0] - ibelow - 1     # number of footprint pixels above center
    imin = 0
    imax = istep
    while imin < shape[0]:
        i0 = max(imin - ibelow, 0)
        i1 = min(imax + iabove, shape[0])
        for l in range(image.shape[0]):
            mslice = None if mask is None else mask[l,i0:i1]
            wslice = None if weights is None else weights[l,i0:i1]
            (temp_image, temp_mask,
             temp_weights) = _apply_op(op, image[l,i0:i1], footprint, mask=mslice,
                                       weights=wslice, info=info, **kwargs)
            filtered[l,imin:imax] = temp_image[imin-i0:imax-i0]
            if new_mask is not None:
                new_mask[l,imin:imax] = temp_mask[imin-i0:imax-i0]
            elif new_weights is not None:
                new_weights[l,imin:imax] = temp_weights[imin-i0:imax-i0]

        imin += istep
        imax = min(imin + istep, shape[0])
        _TILES_USED += 1

    filtered = filtered.reshape(old_shape)
    new_mask = None if new_mask is None else new_mask.reshape(old_shape)
    new_weights = None if new_weights is None else new_weights.reshape(old_shape)
    return (filtered, new_mask, new_weights)


def _apply_op(op, image, footprint, mask, weights, info, **kwargs):
    """Same as _apply_op_as_filter() but with no memory check, layers, or tiles."""

    # Convert to 4-D using stride tricks
    image4d, mask4d, weights4d = _image_to_4d(image, footprint, mask, weights)

    # Apply the footprint (now possibly requiring lots of memory)
    if weights4d is None:
        image3d = image4d[..., footprint]
        mask3d = mask4d[..., footprint]
        weights3d = None
    else:
        image3d = image4d[..., footprint]
        weights3d = weights4d[..., footprint]
        mask3d = None

    # At this point, the new axis is last. Move it to the slot before the image axes.
    image3d = np.moveaxis(image3d, -1, -3)
    mask3d = None if mask3d is None else np.moveaxis(mask3d, -1, -3)
    weights3d = None if weights3d is None else np.moveaxis(weights3d, -1, -3)

    # Finally, apply the operation
    return op(image3d, mask=mask3d, weights=weights3d, info=info, axis=-1, **kwargs)


def _image_to_4d(image, footprint, mask, weights):
    """The image and mask or weights array re-strided with a sliding window in the shape
    of the footprint.

    This is needed for applying a footprint-based operation on a 2-D image. It is
    accomplished without using the memory that would otherwise be needed for a 4-D image.

    Parameters:
        image (array): The image array.
        footprint (array): The 2-D boolean footprint array.
        mask (array or None): The mask array if any.
        weights (array or None): The weights array if any.

    Returns:
        tuple: (`image4d`, `mask4d`, `weights4d`):

        * `image4d`: The image as 4-D, re-strided with a sliding window. Where
          `image[...,i,j]` refers to a single pixel, `image4d[...,i,j,:,:]` iS A 2-D
          array containing the pixel values in the shape of the footprint and centered on
          (i,j). The new shape is:
          `(..., image.shape[0], image.shape[1], footprint.shape[0], footprint.shape[1])`
        * `mask4d` (array or None): A mask equal to True where values should be ignored.
          Indexing is the same as for `image4d`. Provided only if `weights` is None.
        * `weights4d (array or None): A weights array equal to 0 where values should be
          ignored. Indexing is the same as for `image4d`. Provided only if `weights` is
          not None.
    """

    front = image.shape[:-2]
    (im0, im1) = image.shape[-2:]
    fshape = footprint.shape
    (fp0, fp1) = fshape

    imin = fp0 // 2
    jmin = fp1 // 2
    imax = imin + im0
    jmax = jmin + im1

    # Re-stride the image to 4-D with sliding window
    buffer_shape = front + (im0 + fp0 - 1, im1 + fp1 - 1)
    buffer = np.zeros(buffer_shape, dtype=image.dtype)
    buffer[..., imin:imax, jmin:jmax] = image
    image4d = np.lib.stride_tricks.sliding_window_view(buffer, fshape, axis=(-2,-1))

    # Create or re-stride the mask with a sliding window if `weights` is None
    if weights is None:
        if mask is None:
            buffer = np.ones(buffer_shape[-2:], dtype=np.bool_)
            buffer[..., imin:imax, jmin:jmax] = False
        else:
            buffer = np.ones(mask.shape[:-2] + buffer_shape[-2:], dtype=np.bool_)
            buffer[..., imin:imax, jmin:jmax] = mask
        mask4d = np.lib.stride_tricks.sliding_window_view(buffer, fshape, axis=(-2,-1))
        weights4d = None

    # Otherwise, re-stride the weights with a sliding window
    else:
        buffer = np.zeros(weights.shape[:-2] + buffer_shape[-2:], dtype=weights.dtype)
        buffer[..., imin:imax, jmin:jmax] = weights
        weights4d = np.lib.stride_tricks.sliding_window_view(buffer, fshape, axis=(-2,-1))
        mask4d = None

    return (image4d, mask4d, weights4d)

##########################################################################################
# Unit test support
##########################################################################################

_USE_SHORTCUTS = True

def _use_shortcuts(status=None):
    """To get status:
        status = _use_shortcuts()

    To set status:
        _use_shortcuts(True)
    or
        _use_shortcuts(False)
    """

    global _USE_SHORTCUTS

    if status is None:
        return _USE_SHORTCUTS

    _USE_SHORTCUTS = bool(status)


# Unless otherwise specified, use up to half the total system memory.
_MAX_USABLE_BYTES = psutil.virtual_memory().total // 2
_USABLE_BYTES = _MAX_USABLE_BYTES

def _usable_bytes(nbytes=None):
    """To get usable bytes:
        nbytes = _usable_bytes()

    To set usable bytes to a small number for testing:
        _usable_bytes(100000)

    To reset usable bytes to the default:
        _usable_bytes(0)
    """

    global _MAX_USABLE_BYTES, _USABLE_BYTES

    if nbytes is None:
        return _USABLE_BYTES

    if nbytes:
        _USABLE_BYTES = min(nbytes, _MAX_USABLE_BYTES)
    else:
        _USABLE_BYTES = _MAX_USABLE_BYTES

##########################################################################################

