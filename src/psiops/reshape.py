##########################################################################################
# psiops/reshape.py
##########################################################################################

import numpy as np
import numpy.typing as npt

from psiops.resample import resample
from psiops._utils import _check_tuple


def reshape(
    image: npt.ArrayLike,
    shape: int | tuple[int, int],
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """This image resized to a new 2-D shape.

    This function employs resample() but with simplified inputs.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        shape: New shape for the last two (spatial) axes.
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
        returns: Used to override the default quantity or quantities to return, one of
            "i" (image only), "im" (image and mask), "iw" (image and weight array), or
            "imw" (image, mask, and weight array).

    Returns:
        `reshaped` or (`reshaped`[, `new_mask`][, `new_weights`]):

        * `reshaped`: The floating-point, reshaped image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` was specified, any
          masked elements in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to
          the image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the
          elements that contributed to each pixel in `reshaped`. By default, this is
          returned if `weights` is provided; use the `returns` to override this default
          behavior. If `weights` is not provided, this is the number of unmasked pixels
          that contributed to each new pixel's value.

    Raises:
        ValueError: If `shape` is invalid.
        ValueError: If any inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.
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
