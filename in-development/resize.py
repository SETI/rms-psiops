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

def _resize_x(image, shape, lcm):
    """Expand the x-axis over the LCM and then sum, returning array shape
        (shape[0], shape0[1])
    and requiring
        shape0[0] * shape0[1] * LCM[0]
    array elements in memory at once.
    """

    shape0 = image.shape
    shape1 = shape0[:-2] + (shape0[-2], 1                   , shape0[-1])
    shape2 = shape0[:-2] + (shape0[-2], lcm[0] // shape0[-2], shape0[-1])
    shape3 = shape0[:-2] + (shape [-2], lcm[0] // shape [-2], shape0[-1])

    image1 = image.reshape(shape1)              # insert one axis; conserve memory layout
    image2 = np.broadcast_to(image1, shape2)    # use a stride trick on the inserted axis
    image3 = image2.reshape(shape3)             # expand into new memory array
    return image3.sum(axis=-2)                  # sum over axis to obtain new shape


def _resize_y(image, shape, lcm):
    """Expand the y-axis over the LCM and then sum, returning array shape
        (shape0[0], shape[1])
    and requiring
        shape0[0] * shape0[1] * LCM[1]
    array elements in memory at once.
    """

    shape0 = image.shape
    shape1 = shape0[:-1] + (shape0[-1], 1)
    shape2 = shape0[:-1] + (shape0[-1], lcm[1] // shape0[-1])
    shape3 = shape0[:-1] + (shape [-1], lcm[1] // shape [-1])

    image1 = image.reshape(shape1)              # insert one axis; conserve memory layout
    image2 = np.broadcast_to(image1, shape2)    # use a stride trick on the inserted axis
    image3 = image2.reshape(shape3)             # expand into new memory array
    return image3.sum(axis=-1)                  # sum over axis to obtain new shape


_RESIZE_XY_FUNCS = [_resize_x, _resize_y]

##########################################################################################
# Main function
##########################################################################################

def resize(image, shape, mask=None):
    """This entire image resized to a new 2-D shape.

    This function is more general than zoom(), supporting zoom enlargement or reduction by
    any rational factor. It is generally more efficient than resample() if the intent is
    to process the entire image; resample() is generally preferred if only a portion of an
    image is to be resized.

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

    new_xy_shape = _check_tuple(shape, 'new shape', floats=False, negs=False)

    image = np.asarray(image)
    if image.dtype.kind != 'f':
        image = np.asarray(image, dtype=np.float64)

    front = image.shape[:-2]
    old_xy_shape = image.shape[-2:]

    if mask is not None:
        mask = np.asarray(mask, dtype='bool')
        if mask.ndim == 0:
            mask = np.broadcast_to(mask, old_xy_shape)

    # Least common multiple along each axis
    lcm = np.lcm(old_xy_shape, new_xy_shape)

    if _METHOD == 1:
        return _resize_2d(image, new_xy_shape, mask, lcm)

    elif mask is None:
        imin = 0
        resized = _RESIZE_XY_FUNCS[imin](image, new_xy_shape, lcm)
        resized = _RESIZE_XY_FUNCS[1-imin](resized, new_xy_shape, lcm)
        counts = np.prod(lcm) // np.prod(new_xy_shape)
        resized /= counts
        return resized

    zoom = (new_xy_shape[0] / old_xy_shape[0], new_xy_shape[1] / old_xy_shape[1])
    result = resample(image, zoom=zoom, mask=mask, shape=new_xy_shape)

    if mask is None:
        return result[0]

    return result[:2]

##########################################################################################
