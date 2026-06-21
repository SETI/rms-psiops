##########################################################################################
# tests/resize.py
#
# This was originally part of image_ops but it has no particular advantages over
# resample() so it was removed. I have left it here because it is useful for
# cross-testing against several other functions, especially resample().
##########################################################################################

import numpy as np

from psiops._utils import _check_tuple


def resize(image, shape, mask=None):
    """This image resized to a new 2-D shape.

    If the image array has more than two dimensions, the leading dimensions are preserved
    and only the last two dimensions are resized.

    Mean pixel values are preserved.

    Inputs:
        image       an image array, at least 2-D.
        shape       new shape for the last two axes.
        mask        an optional boolean array broadcastable to the shape of the image,
                    containing True where image values should be ignored.

    Return:         resized or (resized, new_mask)
        resized     the new, resized array. Note that it will be floating-point even if
                    the input array contained integers.
        new_mask    the resized mask, provided if the input mask is not None.
    """

    new_xy_shape = _check_tuple(shape, 'new shape', floats=False, negs=False)

    image = np.asarray(image)
    front = image.shape[:-2]
    old_xy_shape = image.shape[-2:]

    if image.dtype.kind != 'f':
        image = np.asarray(image, dtype=np.float64)

    shape1 = (old_xy_shape[0],               1, old_xy_shape[1],               1)
    shape2 = (old_xy_shape[0], new_xy_shape[0], old_xy_shape[1], new_xy_shape[1])
    shape3 = (new_xy_shape[0], old_xy_shape[0], new_xy_shape[1], old_xy_shape[1])

    if mask is None:
        reshaped = image.reshape(front + shape1)
        reshaped = np.broadcast_to(reshaped, front + shape2)
        reshaped = reshaped.reshape(front + shape3)
        return reshaped.mean(axis=(-3,-1))

    mask = np.asarray(mask, dtype='bool')
    if mask.ndim == 0:
        mask = np.broadcast_to(mask, old_xy_shape)

    weights = 1 - np.asarray(mask, dtype='bool')
    image = image * weights.astype(image.dtype)

    wfront = weights.shape[:-2]
    weights = weights.reshape(wfront + shape1)
    weights = np.broadcast_to(weights, wfront + shape2)
    weights = weights.reshape(wfront + shape3)
    weights = np.sum(weights, axis=(-3,-1))

    new_mask = (weights == 0)
    weights[new_mask] = 1

    reshaped = image.reshape(front + shape1)
    reshaped = np.broadcast_to(reshaped, front + shape2)
    reshaped = reshaped.reshape(front + shape3)
    resized = np.sum(reshaped, axis=(-3,-1)) / weights

    if new_mask.shape != resized.shape:
        new_mask = np.broadcast_to(new_mask, resized.shape)

    return (resized, new_mask)

##########################################################################################
