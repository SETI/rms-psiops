##########################################################################################
# image_ops/resize.py
##########################################################################################

import numpy as np
from image_ops.utils import _check_tuple, _usable_bytes

# Unit test support
_METHOD = 0
_LAYERS_USED = 0

def _set_method(method=0):
    """For unit testing; define the method of the resize operation."""

    global _METHOD
    _METHOD = method


def _get_resize_info():
    """For unit testing. Call this after resize() to find out method and how many layers
    were used.
    """

    global _METHOD, _LAYERS_USED

    return (_METHOD, _LAYERS_USED)

##########################################################################################
# Low-level support functions
##########################################################################################

def _resize_2d(image, shape, mask, lcm):

    shape1 = image.shape[:-2] + (image.shape[-2], 1, image.shape[-1], 1)
    shape2 = image.shape[:-2] + (image.shape[-2], lcm[0] // image.shape[-2],
                                 image.shape[-1], lcm[1] // image.shape[-1])
    shape3 = image.shape[:-2] + (lcm[0] // shape[0], shape[0],
                                 lcm[1] // shape[1], shape[1])

    image1 = image.reshape(shape1)              # insert two axes; conserve memory layout
    image2 = np.broadcast_to(image1, shape2)    # use a stride trick on the inserted axes
    image3 = image2.reshape(shape3)             # expand into new memory array

    if mask is None:
        return image3.mean(axis=(-4,-2))

    else:
        mask1 = mask.reshape(shape1)
        mask2 = np.broadcast_to(mask1, shape2)
        mask3 = mask2.reshape(shape3)

        image3[mask3] = 0.
        area = lcm[0] * lcm[1] // (shape[0] * shape[1])
        wsum = area - np.sum(mask3, axis=(-4,-2))

        return (np.sum(image3, axis=(-4,-2)) / np.maximum(wsum, 1), wsum == 0)


def _sum_x(image, shape, lcm):
    """Expand the x-axis over the LCM and then sum, returning array shape
        (shape[0], shape0[1])
    and requiring
        shape0[0] * shape0[1] * LCM[0]
    array elements in memory at once.
    """"

    shape0 = image.shape
    shape1 = shape0[:-2] + (shape0[-2], 1                   , shape0[-1])
    shape2 = shape0[:-2] + (shape0[-2], lcm[0] // shape0[-2], shape0[-1])
    shape3 = shape0[:-2] + (shape [-2], lcm[0] // shape [-2], shape0[-1])

    image1 = image.reshape(shape1)              # insert one axis; conserve memory layout
    image2 = np.broadcast_to(image1, shape2)    # use a stride trick on the inserted axis
    image3 = image2.reshape(shape3)             # expand into new memory array
    return image3.sum(axis=-2)                  # sum over axis to obtain new shape


def _sum_y(image, shape, lcm):
    """Expand the y-axis over the LCM and then sum, returning array shape
        (shape0[0], shape[1])
    and requiring
        shape0[0] * shape0[1] * LCM[1]
    array elements in memory at once.
    """"

    shape0 = image.shape
    shape1 = shape0[:-1] + (shape0[-1], 1)
    shape2 = shape0[:-1] + (shape0[-1], lcm[1] // shape0[-1])
    shape3 = shape0[:-1] + (shape [-1], lcm[1] // shape [-1])

    image1 = image.reshape(shape1)              # insert one axis; conserve memory layout
    image2 = np.broadcast_to(image1, shape2)    # use a stride trick on the inserted axis
    image3 = image2.reshape(shape3)             # expand into new memory array
    return image3.sum(axis=-1)                  # sum over axis to obtain new shape


_SUM_XY_FUNCS = [_sum_x, _sum_y]

##########################################################################################
# Main function
##########################################################################################

def resize(image, shape, mask=None):
    """This entire image resized to a new 2-D shape.

    This function is more general than zoom(), supporting zoom enlargement or reduction by
    any rational factor. It is generally more efficient than resample() if the goal is to
    process the entire image.

    Args:
        image:      The image array, at least 2-D. The last two array axes are treated as
                    the spatial axes of the image.
        shape:      The new shape for the last two axes.
        mask:       An optional boolean array broadcastable to the shape of `image` and
                    containing True where values should be ignored.

    Returns:
        resized:    The resized array, always containing floating-point values.
        new_mask:   The boolean mask for `resized`, included if `mask` is not None.
    """

    shape = _check_tuple(shape, 'new shape', floats=False, negs=False)

    image = np.asarray(image, dtype=np.float64)
    front = image.shape[:-2]
    fsize = np.prod(front)
    shape0 = image.shape[-2:]
    new_shape = front + shape

    if mask is not None:
        mask = np.broadcast_to(mask, image.shape)

    # Least common multiple along each axis
    lcm = np.lcm(shape0, shape)

    #### Method 1: 2-D

    usable_bytes = _usable_bytes()
    itemsize1 = image.dtype.itemsize + (0 if mask is None else 1)
    layer_bytes = np.prod(shape0) * np.prod(lcm) * itemsize1
    if _METHOD == 1 or layer_bytes <= usable_bytes:
        _METHOD = 1
        _LAYERS_USED = 0

        image = image.reshape((fsize,) + shape0)
        resized = np.empty((fsize,) + shape)
        if mask is not None:
            new_mask = np.empty((fsize,) + shape, dtype='bool')

        lstep = usable_bytes // layer_bytes
        l0 = 0
        l1 = lstep
        while l0 < image.shape[0]:
            _LAYERS_USED += 1

            mslice = None if mask is None else mask[l0:l1]
            result = _resize_2d(image, shape, mask=mslice, lcm=lcm)
            if mask is None:
                resized[l0:l1] = result
            else:
                resized[l0:l1] = result[0]
                new_mask[l0:l1] = result[1]

            l0 += lstep
            l1 = min(l0 + lstep, image.shape[0])

        if mask is None:
            return resized.reshape(new_shape)

        return (resized.reshape(new_shape), new_mask.reshape(new_shape))

    #### Method 2: Sequential axes

    elements_xy = max(shape0[0] * shape0[1] * lcm[0], shape[0] * shape0[1] * lcm[1])
        # trailing elements needed for a sum over x and then y
    elements_yx = max(shape0[0] * shape0[1] * lcm[1], shape0[0] * shape[1] * lcm[0])
        # trailing elements needed for a sum over y and then x
    elements = np.array([elements_xy, elements_yx])

    layer_bytes = np.min(elements) * image.dtype.itemsize
    if _METHOD == 2 or needed_bytes <= usable_bytes:
        _METHOD = 2
        _LAYERS_USED = 0

        imin = np.argmin(elements)
        image = image.reshape((fsize,) + shape0)
        resized = np.empty((fsize,) + shape)
        if mask is None:
            counts = np.prod(lcm) // np.prod(shape0)
        else:
            new_mask = np.empty((fsize,) + shape, dtype='bool')

        lstep = usable_bytes // layer_bytes
        l0 = 0
        l1 = lstep
        while l0 < image.shape[0]:
            _LAYERS_USED += 1

            image_sum = _SUM_XY_FUNCS[imin](image[l0:l1], shape, lcm)
            image_sum = _SUM_XY_FUNCS[1-imin](image_sum, shape, lcm)
            if mask is None:
                resized[l0:l1] = image_sum / counts
            else:
                counts = _SUM_XY_FUNCS[imin](1 - mask[l0:l1], shape, lcm)
                counts = _SUM_XY_FUNCS[1-imin](counts, shape, lcm)
                resized[l0:l1] = image_sum / np.maximum(counts, 1)
                new_mask[l0:l1] = counts == 0

            l0 += lstep
            l1 = min(l0 + lstep, image.shape[0])

        if mask is None:
            return resized.reshape(new_shape)

        return (resized.reshape(new_shape), new_mask.reshape(new_shape))

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
            result = _apply_op(op, image[l,i0:i1], footprint, mask=mslice, **args)
            if mask is None:
                filtered[l,imin:imax] = result[imin-i0:imax-i0]
            else:
                filtered[l,imin:imax] = result[0][imin-i0:imax-i0]
                new_mask[l,imin:imax] = result[1][imin-i0:imax-i0]

        imin += istep
        imax = min(imin + istep, shape[0])
        _LAYERS_USED += 1

    if mask is None:
        return filtered.reshape(old_shape)
    else:
        return (filtered.reshape(old_shape), new_mask.reshape(old_shape))

    #### Method 3: Use resample()

    _METHOD = 3
    _LAYERS_USED = 9

    zoom = (shape[0] / shape0[0], shape[1] / shape0[1])
    result = resample(image, zoom=zoom, mask=mask, shape=shape)

    if mask is None:
        return result[0]

    return result[:2]

##########################################################################################
