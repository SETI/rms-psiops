##########################################################################################
# psiops/unzoom.py
##########################################################################################

import numpy as np
from psiops._utils import _check_image, _check_tuple, _check_return


def unzoom(image, unzoom_, mask=None, *, maskval=None, weights=None, nans=False,
           returns=None):
    """Zoom down by an integer factor.

    The unzoom factor must be an integer divisor of the image shape. Use `resample` or
    `reshape` for non-integer unzoom factors.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
            This can be a MaskedArray.
        unzoom_ (int or tuple): The positive integer unzoom factor or tuple of two integer
            unzoom factors. Each spatial dimension of `image` must be an integer multiple
            of the associated factor.
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
        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and weight
            array), or "imw" (image, mask, and weight array).

    Returns:
        (array or tuple): `unzoomed` or (`unzoomed`[, `new_mask`][, `new_weights`]):

        * `unzoomed` (array): The floating-point, unzoomed image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` is specified, any
          masked elements in the unzoomed array will be filled with this value. Otherwise,
          if `nans` is True, masked values will be filled with NaN.
        * `new_mask` (array): The zoomed boolean mask array. By default, this is returned
          if `mask` is specified; use `returns` to override this default behavior.
        * `new_weights` (array): The zoomed floating-point weight array. By default, this
          is returned if `weights` is specified; use `returns` to override this default
          behavior.
    """

    # Interpret the standard inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, returns=returns)

    # Interpret shape and unzoom factor
    shape = image.shape[-2:]
    unzoom_ = _check_tuple(unzoom_, 'unzoom factor', negs=False, floats=False, shape=shape)

    # Perform the unzoom using stride tricks
    reshaped_image = _reshaped_unzoom_array(image, unzoom_)
    info.pixel_area = unzoom_[0] * unzoom_[1]

    # Without a mask, this is easy
    if mask is None:
        unzoomed = reshaped_image.mean(axis=(-3,-1))
        return _check_return(unzoomed, None, None, info)

    # Unzoom including the mask or weight array
    if weights is None:
        weights = np.logical_not(mask)

    reshaped_weights = _reshaped_unzoom_array(weights, unzoom_)
    unzoomed_sum = np.sum(reshaped_image * reshaped_weights, axis=(-3,-1))
    new_weights = np.sum(reshaped_weights, axis=(-3,-1))
    unzoomed_sum /= new_weights

    if info.fill_value is None:     # don't leave NaNs in the array unless it's intended
        info.fill_value = 0

    return _check_return(unzoomed_sum, None, new_weights, info)


def _reshaped_unzoom_array(array, unzoom_):
    """Unzoom a single array using stride tricks, with newly grouped pixels along the new
    -3 and -1 axes.
    """

    front = array.shape[:-2]
    shape = array.shape[-2:]
    new_shape = (shape[0] // unzoom_[0], shape[1] // unzoom_[1])
    tail_shape = (new_shape[0], unzoom_[0], new_shape[1], unzoom_[1])
    reshaped = array.reshape(front + tail_shape)
    return reshaped

##########################################################################################
