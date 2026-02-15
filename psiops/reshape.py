##########################################################################################
# psiops/reshape.py
##########################################################################################

from psiops.resample import resample
from psiops._utils   import _check_tuple


def reshape(image, shape, mask=None, *, maskval=None, weights=None, nans=False,
            returns=None):
    """This image resized to a new 2-D shape.

    This function employs resample() but with simplified inputs.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
            This can be a MaskedArray.
        shape (tuple or int): New shape for the last two (spatial) axes.
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
        (array or tuple): `reshaped` or
            (`reshaped`[, `new_mask`][, `new_weights`][, `new_center`]):

        * `reshaped` (array): The floating-point, reshaped image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` was specified, any
          masked elements in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask` (array): The new mask array, True wherever all the pixels contibuting
          to the image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights` (array): The weight array, equal to the sum of the weights of the
          elements that contributed to each pixel in `reshaped`. By default, this is
          returned if `weights` is provided; use the `returns` to override this default
          behavior. If `weights` is not provided, this is the number of unmasked pixels
          that contributed to each new pixel's value.
    """

    shape = _check_tuple(shape, 'new shape', floats=False, negs=False, zeros=False,
                         default=None)
    zoom_ = (shape[0] / image.shape[-2], shape[1] / image.shape[-1])
    result = resample(image, zoom_=zoom_, mask=mask,  maskval=maskval, weights=weights,
                      nans=nans, shape=shape, center=None, returns=returns)

    if len(result) == 2:
        return result[0]
    else:
        return result[:-1]

##########################################################################################
