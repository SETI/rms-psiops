##########################################################################################
# psiops/zoom.py
##########################################################################################

import numpy as np

from psiops._utils import _check_tuple
from psiops._validation import _check_image, _check_return


def zoom(
    image: np.ndarray,
    zoom_: int | tuple[int, int],
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Zoom up an image array by an integer factor.

    This algorithm uses pixel replication without interpolation in order to ensure that
    photometric values are preserved. Use `resample` or `reshape` for non-integer zoom
    factors.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        zoom_: The positive integer zoom factor or tuple of two integer zoom factors.
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
        `zoomed` or (`zoomed`[, `new_mask`][, `new_weights`]):

        * `zoomed`: The zoomed image array with the same dtype as `image`. If `image` is
          a MaskedArray, this will also be a MaskedArray. If `maskval` is specified, any
          masked elements in the zoomed array will be filled with this value. Otherwise,
          if `nans` is True, masked values will be filled with NaN.
        * `new_mask`: The zoomed boolean mask array. By default, this is returned if
          `mask` is specified; use `returns` to override this default behavior.
        * `new_weights`: The zoomed floating-point weight array. By default, this is
          returned if `weights` is specified; use `returns` to override this default
          behavior.

    Raises:
        ValueError: If `zoom_` is not a positive integer or valid tuple.
        ValueError: If any inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.
    """

    # Interpret the standard inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, returns=returns)

    # Interpret shape and zoom factor
    zoom_ = _check_tuple(zoom_, 'zoom factor', negs=False, floats=False)

    # Zoom the array
    zoomed = _zoom_array(image, zoom_)

    # Without a mask, we're done
    if mask is None:
        return _check_return(zoomed, None, None, info)

    # Interpret the mask or weights
    if weights is None:
        new_mask = _zoom_array(mask, zoom_)
        new_weights = None
    else:
        new_weights = _zoom_array(weights, zoom_)
        new_mask = None

    return _check_return(zoomed, new_mask, new_weights, info)


def _zoom_array(
    array: np.ndarray,
    zoom_: tuple[int, int],
) -> np.ndarray:
    """Zoom a single array using stride tricks.

    Parameters:
        array: Array to zoom.
        zoom_: Two-element tuple of positive integer zoom factors.

    Returns:
        A zoomed copy of `array`.
    """

    front = array.shape[:-2]
    shape = array.shape[-2:]
    reshaped = array.reshape(front + (shape[0], 1, shape[1], 1))
    zoomed = np.broadcast_to(reshaped, front + (shape[0], zoom_[0], shape[1], zoom_[1]))
    zoomed = zoomed.reshape(front + (shape[0]*zoom_[0], shape[1]*zoom_[1]))
    return zoomed.copy()

##########################################################################################
