##########################################################################################
# psiops/ishift.py
##########################################################################################

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from psiops._filter import _use_shortcuts
from psiops._utils import _check_tuple
from psiops._validation import _check_image, _check_return


def ishift(
    image: npt.ArrayLike,
    offset: int | Sequence[int],
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    mode: str = 'masked',
    cval: float | None = 0,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Apply an integer shift to an image.

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

        * `shifted`: The shifted image array with the same shape and dtype as `image`. If
          `image` is a MaskedArray, this will also be a MaskedArray. If `maskval` was
          specified, any masked pixels in the shifted array will be filled with this
          value. Otherwise, if `nans` is True, masked pixels will be filled with NaN.
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
        TypeError: If `offset` values are not integers.
    """

    if mode not in {'masked', 'constant', 'nearest', 'wrap', 'reflect', 'mirror'}:
        raise ValueError(f'invalid mode "{mode}"')

    # Interpret the standard inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              returns=returns, comps=True)

    # Special case where a mask is needed even if not provided
    if mask is None and weights is None and (mode == 'masked' or cval is None):
        mask = np.zeros(image.shape[-2:], dtype=np.bool_)
        info.mask_is_copy = True
        if info.returns == 'i' and not info.is_maskedarray:
            info.returns = 'im'

    # Interpret the offset
    offset = _check_tuple(offset, 'offset', floats=False, negs=True)
    if offset == (0, 0) and _use_shortcuts():
        # Never return an input array; always a copy
        image = image if info.image_is_copy else image.copy()
        if mask is not None and not info.mask_is_copy:
            mask = mask.copy()
        if weights is not None and not info.weights_is_copy:
            weights = weights.copy()
        return _check_return(image, mask, weights, info)

    # Shift the image one axis at a time; 'masked' and cval=None both use a zero fill
    image_mode = 'constant' if mode == 'masked' else mode
    shifted_image = _ishift_array(image, offset, mode=image_mode,
                                  cval=0 if cval is None else cval)

    # Without any need for a mask or weights, we're done
    if mask is None and weights is None and cval is not None:
        return _check_return(shifted_image, None, None, info)

    # Shift the mask or weight array one axis at a time. When weights are present they
    # already encode the masking (weights == 0 wherever masked), so shifting the weights
    # alone is sufficient; the new mask is derived from them by `_check_return`.
    if weights is not None:
        if mode == 'masked' or cval is None:
            temp_mode = 'constant'
            temp_cval = 0
        else:
            temp_mode = mode
            temp_cval = np.max(weights)
        new_weights = _ishift_array(weights, offset, mode=temp_mode, cval=temp_cval)
        new_mask = None             # filled in by _check_return()
    else:
        if mode == 'masked' or cval is None:
            temp_mode = 'constant'
            temp_cval = True
        else:
            temp_mode = mode
            temp_cval = False
        new_mask = _ishift_array(mask, offset, mode=temp_mode, cval=temp_cval)
        new_weights = None

    # Clean up and return the results
    return _check_return(shifted_image, new_mask, new_weights, info)


def _ishift_array(
    image: np.ndarray,
    offset: tuple[int, int],
    *,
    mode: str,
    cval: float | int,
) -> np.ndarray:
    """Perform integer shifts of an image along both spatial axes.

    Parameters:
        image: Array to shift.
        offset: Offsets along the last two axes.
        mode: Boundary handling mode.
        cval: Fill value for "constant" mode.

    Returns:
        Shifted copy of `image`.
    """

    image = _ishift_axis0(image, offset[0], mode=mode, cval=cval)
    image = _ishift_axis1(image, offset[1], mode=mode, cval=cval)
    return image


def _ishift_axis0(
    image: np.ndarray,
    offset: int,
    *,
    mode: str,
    cval: float | int,
) -> np.ndarray:
    """Apply an integer shift along the second-to-last axis.

    Parameters:
        image: Array to shift.
        offset: Offset along the second-to-last axis.
        mode: Boundary handling mode.
        cval: Fill value for "constant" mode.

    Returns:
        Shifted copy of `image`.
    """

    swapped = image.swapaxes(-2, -1)
    return _ishift_axis1(swapped, offset, mode=mode, cval=cval).swapaxes(-2, -1)


def _ishift_axis1(
    image: np.ndarray,
    offset: int,
    *,
    mode: str,
    cval: float | int,
) -> np.ndarray:
    """Apply an integer shift along the last axis.

    Parameters:
        image: Array to shift.
        offset: Offset along the last axis.
        mode: Boundary handling mode.
        cval: Fill value for "constant" mode.

    Returns:
        Shifted copy of `image`.

    Raises:
        ValueError: If `mode` is not recognized.
    """

    if offset == 0:
        return image.copy()

    if offset < 0:
        return _ishift_axis1(image[..., ::-1], -offset, mode=mode, cval=cval)[..., ::-1]

    width = image.shape[-1]
    shifted = np.empty(image.shape, dtype=image.dtype)

    # Constant case
    if mode == 'constant':
        if offset < width:
            shifted[..., offset:] = image[..., :-offset]
            shifted[..., :offset] = cval
        else:
            shifted[...] = cval

    # Nearest case
    elif mode == 'nearest':
        if offset < width:
            shifted[..., offset:] = image[..., :-offset]
            shifted[..., :offset] = image[..., :1]
        else:
            shifted[...] = image[..., :1]

    # Wrap case
    elif mode == 'wrap':
        offset %= width
        if offset == 0:
            return image.copy()

        shifted[..., offset:] = image[..., :-offset]
        shifted[..., :offset] = image[..., -offset:]

    # Mirror case
    elif mode == 'mirror':
        w_minus_1 = width - 1
        repeat = 2 * w_minus_1
        offset = offset % repeat
        if offset > w_minus_1:
            return _ishift_axis1(image[..., ::-1], offset - w_minus_1, mode=mode, cval=cval)
        if offset == 0:
            return image.copy()
        if offset == w_minus_1:
            return image[..., ::-1].copy()

        shifted[..., offset:] = image[..., :-offset]
        shifted[..., :offset] = image[..., 1:offset+1][..., ::-1]

    # Reflect case
    elif mode == 'reflect':
        repeat = 2 * width
        offset = offset % repeat
        if offset > width:
            return _ishift_axis1(image[..., ::-1], offset - width, mode=mode, cval=cval)
        if offset == 0:
            return image.copy()
        if offset == width:
            return image[..., ::-1].copy()

        shifted[..., offset:] = image[..., :-offset]
        shifted[..., :offset] = image[..., :offset][..., ::-1]

    else:
        raise ValueError(f'unrecognized mode "{mode}"')

    return shifted

##########################################################################################
