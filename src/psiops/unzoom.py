##########################################################################################
# psiops/unzoom.py
##########################################################################################

import numpy as np
import numpy.typing as npt

from psiops._utils import _check_tuple
from psiops._validation import _check_image, _check_return


def unzoom(
    image: npt.ArrayLike,
    unzoom_: int | tuple[int, int],
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Zoom down by an integer factor.

    The unzoom factor must be an integer divisor of the image shape. Use `resample` or
    `reshape` for non-integer unzoom factors.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        unzoom_: The positive integer unzoom factor or tuple of two integer unzoom
            factors. Each spatial dimension of `image` must be an integer multiple of
            the associated factor.
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
        `unzoomed` or (`unzoomed`[, `new_mask`][, `new_weights`]):

        * `unzoomed`: The floating-point, unzoomed image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` is specified, any
          masked elements in the unzoomed array will be filled with this value.
          Otherwise, if `nans` is True, masked values will be filled with NaN.
        * `new_mask`: The zoomed boolean mask array. By default, this is returned if
          `mask` is specified; use `returns` to override this default behavior.
        * `new_weights`: The zoomed floating-point weight array. By default, this is
          returned if `weights` is specified; use `returns` to override this default
          behavior.

    Raises:
        ValueError: If `unzoom_` is invalid or not a divisor of the image shape.
    """

    # Interpret the standard inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, returns=returns)

    # Interpret shape and unzoom factor
    shape = image.shape[-2:]
    unzoom_ = _check_tuple(unzoom_, 'unzoom factor', negs=False, floats=False,
                           shape=shape)

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


def _reshaped_unzoom_array(
    array: np.ndarray,
    unzoom_: tuple[int, int],
) -> np.ndarray:
    """Reshape an array for unzooming, grouping pixels along new -3 and -1 axes.

    Parameters:
        array: The array to reshape.
        unzoom_: The two-element unzoom factor tuple.

    Returns:
        The reshaped array with grouped pixels along the new -3 and -1 axes.
    """

    front = array.shape[:-2]
    shape = array.shape[-2:]
    new_shape = (shape[0] // unzoom_[0], shape[1] // unzoom_[1])
    tail_shape = (new_shape[0], unzoom_[0], new_shape[1], unzoom_[1])
    reshaped = array.reshape(front + tail_shape)
    return reshaped

##########################################################################################
