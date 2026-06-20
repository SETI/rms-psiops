##########################################################################################
# psiops/shift.py
##########################################################################################

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from psiops.ishift import ishift
from psiops._filter import _use_shortcuts
from psiops._utils import _check_tuple
from psiops._validation import _check_image, _check_return


def shift(
    image: npt.ArrayLike,
    offset: float | Sequence[float],
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    mode: str = 'masked',
    cval: float | None = 0,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Apply an integral or non-integral shift to an image while retaining photometric
    precision.

    Linear interpolation is performed.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        offset: Two offsets to apply along the last two axes.
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
        mode: The method for handling locations outside the boundary of `image`, one of:

            * "masked": Values outside the boundary are masked.
            * "constant" (`k k k k | a b c d | k k k k`): Assume all exterior values
              equal a constant defined by `cval`.
            * "nearest" (`a a a a | a b c d | d d d d`): Duplicate the nearest edge
              values.
            * "wrap" (`a b c d | a b c d | a b c d`): Wrap values from one edge of the
              image to the other.
            * "reflect" (`d c b a | a b c d | d c b a`): Reflect pixels near each edge
              of the image, where pixels at the edge appear twice ("whole-sample
              symmetric").
            * "mirror" (`c d c b | a b c d | c b a b`): Reflect pixels near each edge
              of the image, where pixels at the edge appear only once ("half-sample
              symmetric").
        cval: If mode is "constant", the numeric value to fill in for areas outside the
            boundaries of the input array.
        returns: Used to override the default quantity or quantities to return, one of
            "i" (image only), "im" (image and mask), "iw" (image and weight array), or
            "imw" (image, mask, and weight array).

    Returns:
        `shifted` or (`shifted`[, `new_mask`][, `new_weights`]):

        * `shifted`: The floating-point, shifted image array with the same shape as
          `image`. If `image` is a MaskedArray, this will also be a MaskedArray. If
          `maskval` was specified, any masked pixels in the shifted array will be filled
          with this value. Otherwise, if `nans` is True, masked pixels will be filled
          with NaN.
        * `new_mask`: The shifted boolean mask array. By default, this is returned if
          `mask` is provided or if `mode` is "masked". Use `returns` to override this
          default behavior.
        * `new_weights`: The shifted floating-point weight array. By default, this is
          returned if `weights` is provided. Use `returns` to override this default
          behavior.

    Raises:
        ValueError: If `mode` is not one of the valid options.
        ValueError: If `image` has fewer than two dimensions.
        ValueError: If `mask` or `weights` have shapes incompatible with `image`.
        ValueError: If `returns` is not a valid option.
        TypeError: If `image` dtype is not numeric.
    """

    # cval=None means mask boundary pixels rather than fill with a constant
    if cval is None:
        mode = 'masked'
        cval = 0.

    # Interpret the offset
    offset = _check_tuple(offset, 'offset', floats=True, negs=True)

    # Split the offsets into integer and fractional parts
    i = int(offset[0] // 1)
    j = int(offset[1] // 1)
    ifrac = offset[0] - i
    jfrac = offset[1] - j

    # For integer offsets, use `ishift` instead
    use_shortcuts = _use_shortcuts()
    if (ifrac, jfrac) == (0, 0) and use_shortcuts:
        image = np.asarray(image)
        if image.dtype.kind not in 'fc':        # make sure the image is floating-point
            image = image.astype(np.float64)
        return ishift(image, (i, j), mask=mask, maskval=maskval, weights=weights,
                      nans=nans, mode=mode, cval=cval, returns=returns)

    # Interpret the standard inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, returns=returns)

    # Special case where a mask is needed even if not provided
    if mask is None and weights is None and mode == 'masked':
        mask = np.zeros(image.shape[-2:], dtype=np.bool_)
        info.mask_is_copy = True
        if info.returns == 'i' and not info.is_maskedarray:
            info.returns = 'im'

    # Determine what we need for the integer shifts
    if mask is None:
        local_returns = 'i'
    elif weights is None:
        local_returns = 'im'
    else:
        local_returns = 'iw'

    # Construct the four nearest integer-shifted arrays and masks or weights
    shifts = np.empty((2,2), dtype='object')
    shifts[0,0] = ishift(image, (i, j), mask=mask, maskval=maskval, weights=weights,
                         mode=mode, cval=cval, returns=local_returns)
    if jfrac == 0 and use_shortcuts:
        shifts[0,1] = shifts[0,0]
    else:
        shifts[0,1] = ishift(image, (i, j+1), mask=mask, maskval=maskval, weights=weights,
                             mode=mode, cval=cval, returns=local_returns)

    if ifrac == 0 and use_shortcuts:
        shifts[1,0] = shifts[0,0]   # avoid unnecessary shifts
        shifts[1,1] = shifts[0,1]
    elif jfrac == 0 and use_shortcuts:
        shifts[1,0] = ishift(image, (i+1, j), mask=mask, maskval=maskval, weights=weights,
                             mode=mode, cval=cval, returns=local_returns)
        shifts[1,1] = shifts[1,0]
    else:
        shifts[1,0] = ishift(image, (i+1, j), mask=mask, maskval=maskval, weights=weights,
                             mode=mode, cval=cval, returns=local_returns)
        shifts[1,1] = ishift(image, (i+1, j+1), mask=mask, maskval=maskval,
                             weights=weights, mode=mode, cval=cval, returns=local_returns)

    # Define fractional weights based on linear interpolation between the shifted arrays
    f00 = (1. - ifrac) * (1. - jfrac)
    f01 = (1. - ifrac) * jfrac
    f10 = ifrac * (1. - jfrac)
    f11 = ifrac * jfrac

    # Without masks or weights, just return the weighted sum (as float32 if necessary)
    if local_returns == 'i':
        out_dtype = image.dtype if image.dtype.kind in 'fc' else np.float64
        if (ifrac and jfrac) or not use_shortcuts:
            return (shifts[0,0] * f00 + shifts[0,1] * f01 +
                    shifts[1,0] * f10 + shifts[1,1] * f11).astype(out_dtype)
        elif ifrac:
            return (shifts[0,0] * f00 + shifts[1,0] * f10).astype(out_dtype)
        else:
            return (shifts[0,0] * f00 + shifts[0,1] * f01).astype(out_dtype)
        # the integer shift case was handled above

    # Split the mask/weight arrays from the shifted images
    shift00, mask00 = shifts[0,0]
    shift01, mask01 = shifts[0,1]
    shift10, mask10 = shifts[1,0]
    shift11, mask11 = shifts[1,1]

    # Convert masks to weight arrays
    if local_returns == 'im':
        w00 = f00 * np.logical_not(mask00)
        w10 = f10 * np.logical_not(mask10)
        w01 = f01 * np.logical_not(mask01)
        w11 = f11 * np.logical_not(mask11)
    else:
        w00 = f00 * mask00
        w10 = f10 * mask10
        w01 = f01 * mask01
        w11 = f11 * mask11

    # Construct the weighted sum
    if (ifrac and jfrac) or not use_shortcuts:
        shifted = shift00 * w00 + shift01 * w01 + shift10 * w10 + shift11 * w11
        new_weights = w00 + w01 + w10 + w11
    elif ifrac:
        shifted = shift00 * w00 + shift10 * w10
        new_weights = w00 + w10
    else:
        shifted = shift00 * w00 + shift01 * w01
        new_weights = w00 + w01

    # Re-normalize the sums (fully masked pixels produce 0/0 → NaN, which is expected)
    out_dtype = image.dtype if image.dtype.kind in 'fc' else np.float64
    with np.errstate(invalid='ignore', divide='ignore'):
        shifted /= new_weights
    return _check_return(shifted.astype(out_dtype), None, new_weights, info)

##########################################################################################
