##########################################################################################
# psiops/_filter.py
##########################################################################################

import numpy as np
import psutil

from psiops._utils import _check_tuple

##########################################################################################
# Filter support
##########################################################################################

# Unit test support
_LAYERS_USED = 0
_TILES_USED = 0

def _apply_op_as_filter_info():
    """For unit testing: call this after _apply_op_as_filter() to find out how many
    layers and tiles were used.
    """

    global _LAYERS_USED, _TILES_USED
    return (_LAYERS_USED, _TILES_USED)


def _apply_op_as_filter(op, image, footprint, *, mask, weights, info, **kwargs):
    """Apply this operation as a filter on the image over the specified footprint.

    Parameters:
        op (callable): The function performing the image operation.
        image (array): Image array, in which the last two axes are spatial dimensions.
            This can be a MaskedArray.
        footprint (array-like or int or tuple of two ints): The 2-D boolean footprint
            array, or else an integer or tuple of two integers defining the rectangular
            shape of the footprint.
        mask (array, optional): A boolean mask array, equal to True where image values
            are to be ignored. The shape is broadcasted to the shape of `image` if
            necessary.
        weights (array, optional): Array of weight values where a weight of zero is
            equivalent to a mask value of True. Values should never be negative.
        info (_ImageInfo): The `_ImageInfo` object returned by `_check_image`.
        **kwargs: Any other inputs to be passed to `op`.

    Returns:
        A tuple (`filtered_image`, `mask`, `weights`), where:

        * `filtered_image`: The filtered image array.
        * `mask`: The mask array if `weights` is None, otherwise None.
        * `weights`: The weight array if `weights` is not None, otherwise None.

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

    # The auxiliary outputs (`new_mask` and `new_weights`) are allocated lazily from the
    # first computed slice, so their dtype always matches what `op` actually returns. They
    # remain None when `op` produces no mask or weights (e.g. the fully unmasked case).
    new_mask = None
    new_weights = None

    if weights is not None:
        weights = np.broadcast_to(weights, old_shape)
        weights = np.reshape(weights, image.shape)
    elif mask is not None:
        mask = np.broadcast_to(mask, old_shape)
        mask = np.reshape(mask, image.shape)

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
            if temp_mask is not None:
                if new_mask is None:
                    new_mask = np.empty(image.shape, dtype=temp_mask.dtype)
                new_mask[l0:l1] = temp_mask
            if temp_weights is not None:
                if new_weights is None:
                    new_weights = np.empty(image.shape, dtype=temp_weights.dtype)
                new_weights[l0:l1] = temp_weights

            l0 += lstep
            l1 = min(l0 + lstep, image.shape[0])
            _LAYERS_USED += 1

        _TILES_USED = 1
        filtered = filtered.reshape(old_shape)
        new_mask = None if new_mask is None else new_mask.reshape(old_shape)
        new_weights = None if new_weights is None else new_weights.reshape(old_shape)
        return (filtered, new_mask, new_weights)

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
            if temp_mask is not None:
                if new_mask is None:
                    new_mask = np.empty(image.shape, dtype=temp_mask.dtype)
                new_mask[l,imin:imax] = temp_mask[imin-i0:imax-i0]
            if temp_weights is not None:
                if new_weights is None:
                    new_weights = np.empty(image.shape, dtype=temp_weights.dtype)
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

    # At this point, the new axis is last. Move it to the slot before the image axes so
    # that the trailing two axes remain the spatial axes expected by the operations.
    image3d = np.moveaxis(image3d, -1, -3)
    mask3d = None if mask3d is None else np.moveaxis(mask3d, -1, -3)
    weights3d = None if weights3d is None else np.moveaxis(weights3d, -1, -3)

    # Finally, apply the operation, reducing over the footprint axis (now at -3)
    return op(image3d, mask=mask3d, weights=weights3d, info=info, axis=-3, **kwargs)


def _image_to_4d(image, footprint, mask, weights):
    """The image and mask or weights array re-strided with a sliding window in the shape
    of the footprint.

    This is needed for applying a footprint-based operation on a 2-D image. It is
    accomplished without using the memory that would otherwise be needed for a 4-D image.

    Parameters:
        image (array): The image array.
        footprint (array): The 2-D boolean footprint array.
        mask (array, optional): The mask array if any.
        weights (array, optional): The weights array if any.

    Returns:
        A tuple (`image4d`, `mask4d`, `weights4d`):

        * `image4d`: The image as 4-D, re-strided with a sliding window. Where
          `image[...,i,j]` refers to a single pixel, `image4d[...,i,j,:,:]` is a 2-D
          array containing the pixel values in the shape of the footprint and centered on
          (i,j). The new shape is:
          `(..., image.shape[0], image.shape[1], footprint.shape[0], footprint.shape[1])`
        * `mask4d`: A mask equal to True where values should be ignored. Indexing is the
          same as for `image4d`. Provided only if `weights` is None.
        * `weights4d`: A weights array equal to 0 where values should be ignored.
          Indexing is the same as for `image4d`. Provided only if `weights` is not None.
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
    """Get or set whether shortcut optimizations are used.

    Parameters:
        status (bool, optional): If None, the current setting is left unchanged. If True
            or False, the shortcut status is updated to this value.

    Returns:
        The current shortcut status, after applying any update.
    """

    global _USE_SHORTCUTS

    if status is not None:
        _USE_SHORTCUTS = bool(status)

    return _USE_SHORTCUTS


# None means "use the default": half of current total system memory, evaluated lazily.
_USABLE_BYTES = None

def _usable_bytes(nbytes=None):
    """Get or set the memory limit for array operations.

    Parameters:
        nbytes (int, optional): If None, the memory limit is left unchanged. If a
            positive integer, sets the limit to `min(nbytes, half of current total system
            memory)`. If zero, resets to the default (half of total system memory,
            re-queried at that time).

    Returns:
        The current memory limit in bytes, after applying any update. At its default, this
        is half of the current total system memory.
    """

    global _USABLE_BYTES

    if nbytes is not None:
        if nbytes:
            _USABLE_BYTES = min(nbytes, psutil.virtual_memory().total // 2)
        else:
            _USABLE_BYTES = None  # next call re-queries system memory

    if _USABLE_BYTES is None:
        return psutil.virtual_memory().total // 2

    return _USABLE_BYTES

##########################################################################################
