##########################################################################################
# ops/outliers.py
##########################################################################################

import numpy as np
import numpy.typing as npt

from ._utils import _check_axis
from ._validation import _check_image
from .circle import circle
from .gaussian_filter import gaussian_filter
from .median import median, median_filter


def outliers(
    image: npt.ArrayLike,
    footprint: float = 10,
    *,
    cutoff: float = 5,
    quantile: float = 0.999,
    axis: int | tuple[int, ...] | None = None,
    mask: np.ndarray | None = None,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
) -> np.ndarray:
    """A mask identifying pixels of the given image that are excessively bright compared
    to their surroundings.

    This function obtains a median-filtered "baseline" version of the image. It then
    infers a 2-D model for the local noise amplitude from mean-square difference between
    the image and the baseline. Pixels that differ from the baseline by more than a
    specified number of standard deviations are identified as outliers.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        footprint: The diameter in pixels of the median filter to be applied to the
            image. This should be a few times larger than the size of the largest clump
            of pixels to be identified as outliers.
        cutoff: The number of standard deviations by which a pixel must deviate from
            the local median for it to be identified as an outlier.
        quantile: The fractional cutoff in the histogram of inferred standard deviations
            above which values are truncated. This prevents bright objects from
            contributing excessively to the noise amplitude analysis.
        axis: One or more axes along which to assume that the images have identical
            properties, given as an integer or tuple of integers; None to operate over
            all but the last two (spatial) axes. When images have identical properties,
            the image-to-image variation can contribute to the analysis by providing a
            more robust estimate of the noise amplitude. Note that negative axes count
            backwards from the first spatial axis, so axis=-1 actually refers to the
            third-to-last axis of the image array. Default is to treat every 2-D image
            as independent.
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

    Returns:
        A new boolean mask that incorporates the original `mask` but is also True where
        pixels are identified as statistical outliers.

    Raises:
        ValueError: If any image inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.
    """

    # Interpret the image inputs
    image, mask, weights, _info = _check_image(image, mask, maskval, weights, nans=nans,
                                               comps=False)

    # Identify the axes
    if axis is not None:
        axis = _check_axis(axis, image.shape)

    # Infer a baseline image via median filtering
    radius = footprint / 2.
    circle_footprint = circle(radius)
    baseline = median_filter(image, circle_footprint, mask=mask, maskval=maskval,
                             weights=weights, nans=nans, returns='i')

    # Clip the squared difference
    diff_sq = (image - baseline)**2
    if quantile < 1.:
        if mask is None:
            clip = np.quantile(diff_sq, quantile)
        else:
            clip = np.quantile(diff_sq[np.logical_not(mask)], quantile)
        diff_sq[diff_sq > clip] = clip

    # Construct the smoothed standard deviation model
    if axis is not None:
        diff_sq = median(diff_sq, mask=mask, weights=weights, axis=axis, keepdims=True)

    mean_sq = median_filter(diff_sq, circle_footprint, mask=mask, maskval=maskval,
                            weights=weights, nans=nans, returns='i')
    smoothed = gaussian_filter(mean_sq, sigma=radius, mask=mask, maskval=maskval,
                               weights=weights, nans=nans, returns='i')

    # Return the mask of residuals above the cutoff
    new_mask = diff_sq > cutoff**2 * smoothed
    if mask is not None:
        new_mask = mask | new_mask
    return new_mask

##########################################################################################
