##########################################################################################
# ops/camouflage.py
##########################################################################################

import numpy as np
import numpy.typing as npt
import scipy.ndimage

from ._utils import _check_image, _check_return, _merge_weights
from .gaussian_filter import gaussian_filter

_EPS = 1.e-8


def camouflage(
    image: npt.ArrayLike,
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    size: float = 30,
    returns: str = 'i',
) -> np.ndarray | list[np.ndarray]:
    """Replace the masked pixels of an image based on the the nearby, unmasked pixels.

    Replacement values are obtained by Gaussian-filtering the unmasked pixels of the
    image. The sigma of this Gaussian is scaled to the size of the masked area being
    filled, which helps the replaced pixels to blend in with their surroundings.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        mask: Boolean mask array, equal to True where the values in `image` are to be
            ignored. It is broadcasted to the shape of `image` if necessary.
        maskval: A value that should be masked wherever it appears in `image`. This can
            be used instead of or in addition to the `mask`.
        weights: Weight array specifying the possibly unequal weights associated with
            the pixels in `image`. A weight of zero is equivalent to a `mask` value of
            True. This can be provided in addition to or instead of the `mask` or
            `maskval`. It is broadcasted to the shape of `image` if necessary. Values
            should never be negative.
        nans: True to check `image` for NaNs and interpret them as masked values.
        size: The approximate upper limit on the size in pixels of the masked areas to
            be camouflaged. Areas that are somewhat larger will still be filled, but
            with less accuracy. Areas that are much larger could remain masked due to
            the underflow of the Gaussian. Note that camouflaging very large areas
            (larger than 50-100 pixels) is time-consuming and does not always yield
            satisfactory results.
        returns: Used to override the default quantity or quantities to return, one of
            "i" (image only), "im" (image and mask), "iw" (image and weight array), or
            "imw" (image, mask, and weight array). Default is "i".

    Returns:
        `filled` or (`filled`[, `new_mask`][, `new_weights`]):

        * `filled`: The floating-point, filled image array. If `image` is a MaskedArray,
          this will also be a MaskedArray. If `maskval` was specified, any remaining
          unmasked elements in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True where underflow of the Gaussian has caused
          some pixels to still be masked.
        * `new_weights`: The weight array, equal to the original weights where `image` is
          unmasked, and equal to the weight of the Gaussian-filtered image at locations
          where masked values have been filled.

    Raises:
        ValueError: If any inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.

    Notes:
        This function categorizes each "hole" of masked pixels in the image by its
        approximate size. It then replaces the pixels in each hole with values from a
        Gaussian-filtered version of the nearby, unmasked pixels. It scales the "sigma"
        of the Gaussian filter to the size of the hole, so larger holes receive
        contributions from more distant unmasked pixels. This ensures that each hole is
        filled with new pixels that vary smoothly on the scale of that hole, so that all
        filled holes blend in well with their surroundings.

        Note that, if `image` contains a masked area that is significantly larger than
        the specified `size`, it is possible for pixels in the middle of that area to
        remain masked. This is due to underflow of the Gaussian filter, which is based
        on the nearest unmasked pixels of `image`.
    """

    # Interpret the image inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, returns=returns)

    # If the image has no mask, return it as is
    if mask is None:
        return _check_return(image, None, None, info)

    # Build a sequence of circular footprints that increase in size geometrically by
    # factors of sqrt(2)
    footprints = []
    radii = []
    radius_sq = 2
    while True:
        halfsize = int(np.sqrt(radius_sq) - _EPS)    # ints need to round down by 1
        fullsize = 2 * halfsize + 1
        if fullsize > size:
            break
        x = np.arange(-halfsize, halfsize+1)
        dist_sq = x**2 + x[:,np.newaxis]**2
        footprint = (dist_sq < radius_sq)
        footprints.append(footprint)
        radii.append(np.sqrt(radius_sq))
        radius_sq *= 2

    # The `holes` array equals -1 where the image is unmasked; its value increases with
    # increasing distance from the nearest unmasked pixel.
    holes = np.array(mask, dtype=np.int8) - 1
    unfilled = mask
    max_k = -1
    for k, footprint in enumerate(footprints):
        filtered = scipy.ndimage.minimum_filter(holes, footprint=footprint,
                                                mode='constant', cval=127)
        in_range = (filtered == -1)
        holes[in_range & unfilled] = k
        if np.all(in_range):
            max_k = k
            break
        unfilled = np.logical_not(in_range)

    if np.any(unfilled):
        max_k = len(footprints)
        holes[unfilled] = max_k + 1
        radii.append(radii[-1] * np.sqrt(2))

    # Replace every non-negative value with the largest value within the same hole
    antimask = np.logical_not(mask)
    for _iter in range(radii):
        new_holes = scipy.ndimage.maximum_filter(holes, footprint=footprints[0],
                                                 mode='constant', cval=0)
        new_holes[antimask] = -1
        if np.all(new_holes == holes):
            break
        holes = new_holes

    # Fill holes using using a Gaussian-filtered image with a sigma defined by the hole
    # size.
    filled = image.copy()
    weights = _merge_weights(mask, weights)
    new_weights = weights.copy()
    for k, rad in enumerate(radii[:max_k+1]):
        filtered, filtered_weights = gaussian_filter(image, rad/2., weights=weights,
                                                     returns='iw')
        temp_mask = (holes == k) & filtered_weights
        filled[temp_mask] = filtered[temp_mask]
        new_weights[temp_mask] = filtered_weights[temp_mask]

    # Return the filled image and its possible mask
    info.fill_value = info.fill_value or 0.
    return _check_return(image, None, new_weights, info)

##########################################################################################
