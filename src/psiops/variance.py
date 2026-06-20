##########################################################################################
# psiops/variance.py
##########################################################################################

import warnings

import numpy as np
import numpy.typing as npt

from psiops._filter import _apply_op_as_filter, _use_shortcuts
from psiops._utils import _ImageInfo, _check_axis, _flatten_axes, _merge_weights, _pixel_area
from psiops._validation import _check_image, _check_return


def variance(
    image: npt.ArrayLike,
    mask: np.ndarray | None = None,
    *,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    factors: npt.ArrayLike | None = None,
    vartype: str = 'reliability',
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Variance among the pixels in an array of images, accounting for masked pixels
    and/or weights.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray. Must be at least 3-D.
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
        axis: The axis or axes over which to operate, given as an integer or tuple of
            integers; None to operate over all but the last two (spatial) axes. Note that
            negative axes count backwards from the first spatial axis, so axis=-1 actually
            refers to the third-to-last axis of the image array.
        keepdims: If True, the axes that are reduced are left in the result as dimensions
            with size one. With this option, the results will broadcast correctly against
            the input arrays.
        factors: Array of weights to be applied to the non-spatial axes of the `image`
            array. It is broadcasted to the shape of the image after excluding the
            image's trailing two (spatial) axes.
        vartype: The type of the variance estimate:

            * "biased": The divisor used in the variance calculation is the number of
              unmasked items or the sum of the weights. This is equivalent to using
              `ddof=0` in `numpy.std()`.
            * "unbiased" or "frequency": The divisor is one less than that for the
              "biased" variance, providing an unbiased estimate. For uniform weights, this
              is equivalent to using `ddof=1` in `numpy.std()`. For non-uniform weights,
              this treats each weight factor as equivalent to the frequency or number of
              measurements with the associated value.
            * "reliability": This provides an unbiased estimate of the variance if
              `weights` and `factors` are interpreted as describing the relative
              trustworthiness or uncertainty in individual measurements. For an unweighted
              variance, this is equivalent to "unbiased".

            See further info here:
                https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Related_concepts

        returns: Used to override the default quantity or quantities to return, one of
            "i" (image only), "im" (image and mask), "iw" (image and weight array), or
            "imw" (image, mask, and weight array).

    Returns:
        `var_image` or (`var_image`[, `new_mask`][, `new_weights`]):

        * `var_image`: The floating-point variance array. If `image` is a MaskedArray,
          this will also be a MaskedArray. If `maskval` was specified, any masked pixels in
          this array will be filled with this value. Otherwise, if `nans` is True, masked
          pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the image pixels along the
          specified `axis` are masked. By default, this is returned if `mask` is provided;
          use `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `var_image`. By default, this is returned if
          `weights` is provided; use `returns` to override the default behavior. If
          `weights` is not provided, this is the integer number of unmasked pixels that
          contributed to each new pixel's value.

    Raises:
        ValueError: If `vartype` is not one of "biased", "unbiased", "frequency", or
            "reliability".
        ValueError: If any inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.
    """

    # Interpret the image inputs
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=False, floats=True, three=True,
                                              returns=returns)
    # Identify the axes
    axis = _check_axis(axis, image.shape)
    if keepdims:
        info.axis = axis
    info.pixel_area = _pixel_area(axis, image.shape)

    # Evaluate and return
    (var_image, new_mask, new_weights) = _variance(image, mask, weights, info=info,
                                                   axis=axis, factors=factors,
                                                   vartype=vartype)
    return _check_return(var_image, new_mask, new_weights, info)


def _variance(
    image: np.ndarray,
    mask: np.ndarray | None,
    weights: np.ndarray | None,
    info: _ImageInfo,
    axis: tuple[int, ...],
    *,
    factors: npt.ArrayLike | None = None,
    vartype: str = 'reliability',
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Internal implementation of variance().

    Parameters:
        image: Image array.
        mask: Boolean mask array, or None.
        weights: Weight array, or None.
        info: The `_ImageInfo` object returned by `_check_image`.
        axis: Resolved tuple of axis indices over which to operate.
        factors: Optional extra weight factors applied along non-spatial axes.
        vartype: The type of variance estimate; one of "biased", "unbiased",
            "frequency", or "reliability".

    Returns:
        A tuple (`var_image`, `new_mask`, `new_weights`).

    Raises:
        ValueError: If `vartype` is not a recognized value.
    """

    if vartype not in {'biased', 'unbiased', 'frequency', 'reliability'}:
        raise ValueError(f'unrecognized value for vartype: {vartype!r}')

    if info.fill_value is None:  # don't leave NaNs in the array unless that's intended
        info.fill_value = 0

    # Handle the case without leading factors
    if factors is None and _use_shortcuts():
        ddof = 0 if vartype == 'biased' else 1

        if mask is None:
            var_image = np.var(image, axis=axis, ddof=ddof)
            return (var_image, None, None)

        if weights is None:
            if not info.image_is_copy:
                image = image.copy()
            mask = np.broadcast_to(mask, image.shape)
            image[mask] = np.nan
            # Fully/under-populated slices yield NaN (expected); suppress the warning
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                var_image = np.nanvar(image, axis=axis, ddof=ddof)
            new_weights = info.pixel_area - np.sum(mask, axis=axis)
            return (var_image, None, new_weights)

    # Combine weights and factors, case by case...
    weights = _merge_weights(mask, weights, factors)

    # Handle the unweighted case
    if weights is None:
        denom = info.pixel_area
        mean = np.sum(image, axis=axis, keepdims=True) / denom
        sumsq = np.sum((image - mean)**2, axis=axis)
        if vartype != 'biased':
            denom -= 1
        if denom < 0:
            new_mask = np.ones(image.shape[-2:], dtype=np.bool_)
            return (sumsq, new_mask, None)
        return (sumsq / denom, None, None)

    # Move the selected axes to the front, then flatten them
    weights = _flatten_axes(weights, axis, shape=image.shape)
    image = _flatten_axes(image, axis)

    # Determined the weighted variance. Zero-weight elements produce expected NaNs from
    # 0/0 divisions, so suppress the associated warnings.
    with np.errstate(invalid='ignore', divide='ignore'):
        new_weights = np.sum(weights, axis=0)
        image_sum = np.sum(image * weights, axis=0)
        mean = image_sum / new_weights              # zero-weight elements are NaN
        sumsq = np.sum(weights * (image - mean)**2, axis=0)

        if vartype == 'reliability':
            wsumsq = np.sum(weights**2, axis=0)
            denom = new_weights - wsumsq/new_weights
        elif vartype != 'biased':
            denom = new_weights - 1

        new_weights[denom == 0] = 0
        var_image = sumsq / denom
    return (var_image, None, new_weights)


def variance_filter(
    image: npt.ArrayLike,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = None,
    maskval: float | None = None,
    weights: np.ndarray | None = None,
    nans: bool = False,
    vartype: str = 'reliability',
    returns: str | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Filter this image such that each new pixel contains the variance among the pixels
    within the specified footprint.

    Parameters:
        image: Image array, in which the last two axes are the spatial dimensions. This
            can be a MaskedArray.
        footprint: The 2-D boolean footprint array or else an integer or tuple of two
            integers defining the rectangular shape of the footprint.
        mask: Boolean mask array with the same shape as `image` and equal to True where
            the values in `image` are to be ignored.
        maskval: A value that should be masked wherever it appears in `image`. This can
            be used instead of or in addition to the `mask`.
        weights: Weight array specifying the possibly unequal weights associated with the
            pixels in `image`. A weight of zero is equivalent to a `mask` value of True.
            This can be provided in addition to or instead of the `mask` or `maskval`.
        nans: True to check `image` for NaNs and interpret them as masked values.
        vartype: The type of the variance estimate:

            * "biased": The divisor used in the variance calculation is the number of
              unmasked items or the sum of the weights. This is equivalent to using
              `ddof=0` in `numpy.std()`.
            * "unbiased" or "frequency": The divisor is one less than that for the
              "biased" variance, providing an unbiased estimate. For uniform weights, this
              is equivalent to using `ddof=0` in `numpy.std()`. For non-uniform weights,
              this treats each weight factor as equivalent to the frequency or number of
              measurements with the associated value.
            * "reliability": This provides an unbiased estimate of the variance if
              `weights` and `factors` are interpreted as describing the relative
              trustworthiness or uncertainty in individual measurements. For an unweighted
              variance, this is equivalent to "unbiased".

            See further info here:
                https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Related_concepts

        returns: Used to override the default quantity or quantities to return, one of
            "i" (image only), "im" (image and mask), "iw" (image and weight array), or
            "imw" (image, mask, and weight array).

    Returns:
        `var_image` or (`var_image`[, `new_mask`][, `new_weights`]):

        * `var_image`: The floating-point variance array. If `image` is a MaskedArray,
          this will also be a MaskedArray. If `maskval` was specified, any masked pixels in
          this array will be filled with this value. Otherwise, if `nans` is True, masked
          pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to the
          image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `var_image`. By default, this is returned if
          `weights` is provided; use `returns` to override the default behavior. If
          `weights` is not provided, this is the integer number of unmasked pixels that
          contributed to each new pixel's value.

    Raises:
        ValueError: If `vartype` is not one of "biased", "unbiased", "frequency", or
            "reliability".
        ValueError: If any inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.
    """

    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=False, floats=True, three=False,
                                              returns=returns)

    var_image, new_mask, new_weights = _apply_op_as_filter(_variance, image, footprint,
                                                           mask=mask, weights=weights,
                                                           info=info, vartype=vartype)
    return _check_return(var_image, new_mask, new_weights, info)

##########################################################################################
