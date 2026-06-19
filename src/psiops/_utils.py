##########################################################################################
# psiops/_utils.py
##########################################################################################

import numbers
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class _ImageInfo:
    """State threaded from `_check_image` through internal ops to `_check_return`.

    Attributes:
        axis: Indices of any unit-length axes to be restored in the returned array,
            supporting the `keepdims` option.
        dtype: Original dtype of the image, used by `_check_return` to restore it after
            internal promotion (e.g. bool → int8). None means no conversion needed.
        extra_char: One extra character that can be appended to the `returns` string to
            indicate that extra information is to be appended.
        fill_value: Value to fill masked pixels in the returned array; None means no
            fill. May be NaN.
        image_is_copy: True if the image returned by `_check_image` is a copy.
        is_maskedarray: True if the original `image` argument was a MaskedArray.
        mask_is_copy: True if the mask returned by `_check_image` is a copy.
        pixel_area: Number of pixels being combined per output pixel; used by
            `_check_return` to build a weight array when one is requested but not
            computed. Default is 1.
        returns: Controls what `_check_return` returns. One of "i" (image only), "im"
            (image and mask), "iw" (image and weights), or "imw" (image, mask, and
            weights). May end with `extra_char` to append the `extra` value.
        weights_is_copy: True if the weights array returned by `_check_image` is a copy.
    """

    axis: tuple[int, ...] = field(default_factory=tuple)
    dtype: np.dtype | None = None
    extra_char: str = ''
    fill_value: float | None = None
    image_is_copy: bool = False
    is_maskedarray: bool = False
    mask_is_copy: bool = False
    pixel_area: int = 1
    returns: str = 'i'
    weights_is_copy: bool = False

    def __post_init__(self) -> None:
        valid = {'i', 'im', 'iw', 'imw'}
        if self.extra_char:
            valid |= {v + self.extra_char for v in valid}
        if self.returns not in valid:
            raise ValueError(
                f'invalid `returns` value {self.returns!r}; '
                f'valid values are {sorted(valid)}'
            )

##########################################################################################
# Shared argument helpers
##########################################################################################

def _check_tuple(
    item: object,
    title: str,
    *,
    default: tuple[float, float] | None = None,
    floats: bool = True,
    negs: bool = False,
    zeros: bool | None = None,
    nones: bool = False,
    shape: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Validate an input as a single value or tuple of two values.

    Parameters:
        item: The value to check.
        title: Name of the parameter, used in error messages.
        default: The default value to return as a tuple of two values.
        floats: True to allow floating-point values.
        negs: True to allow negative values.
        zeros: True to allow zero values; by default this has the same value as `negs`.
        nones: True to allow None as an input value. If `item` is None and `default` is
            specified, `default` will be returned; if `item` is None but no `default` is
            specified, a ValueError is raised.
        shape: If specified, each element must be an integer multiple of the corresponding
            element in `item`.

    Returns:
        The validated, two-element tuple.

    Raises:
        ValueError: If `item` is None and `default` is None.
        TypeError: If the values are not integers when `floats` is False.
        ValueError: If the value violates the `negs` or `zeros` constraints.
        ValueError: If `shape` is not divisible by `item`.
    """

    zeros = negs if zeros is None else zeros

    if item is None:
        if nones or default is not None:
            return default
        raise ValueError(f'missing {title}')

    if isinstance(item, (tuple, list, np.ndarray)):
        if len(item) != 2:
            raise ValueError(f'invalid {title} {tuple(item)}; two values required')
        item = tuple(item)
    else:
        item = (item, item)

    if not floats:
        if (not isinstance(item[0], numbers.Integral) or
            not isinstance(item[1], numbers.Integral)):
                raise TypeError(f'invalid {title} {item}; two integers required')

    if not negs and (item[0] < 0 or item[1] < 0):
        if zeros:
            raise ValueError(f'invalid {title} {item}; non-negative values required')
        else:
            raise ValueError(f'invalid {title} {item}; positive values required')

    if not zeros and (item[0] == 0 or item[1] == 0):
        if negs:
            raise ValueError(f'invalid {title} {item}; non-zero values required')
        else:
            raise ValueError(f'invalid {title} {item}; positive values required')

    if shape is not None and (shape[0] % item[0] != 0 or shape[1] % item[1] != 0):
        raise ValueError(f'shape {shape} is not divisible by {title} {item}')

    return item


def _check_axis(
    axis: int | tuple[int, ...] | None,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    """Validate an input value for axis, given the shape of an image.

    Parameters:
        axis: Axis index or a tuple of axis indices, which can be positive or negative.
            None is replaced by tuple(range(len(shape))).
        shape: Overall shape of an image array. All elements must be positive.

    Returns:
        Revised tuple of non-negative integers in the order given. Note that the last two
        elements of the shape cannot be referenced.

    Raises:
        TypeError: If `axis` or any axis element is not an integer.
        IndexError: If any axis value is out of range.
        ValueError: If any axis value is duplicated.
    """

    shape = shape[:-2]

    if axis is None:
        return tuple(range(len(shape)))

    if isinstance(axis, numbers.Integral):
        axis = (axis,)

    try:
        axis = tuple(axis)
    except TypeError:
        raise TypeError(f'invalid axis of type {type(axis).__name__}: {repr(axis)}')

    new_axis = []
    for k in axis:
        original_k = k
        if not isinstance(k, numbers.Integral):
            raise TypeError(f'invalid axis item {repr(k)} of type {type(k).__name__}')
        if k < 0:
            k += len(shape)
        if k < 0 or k >= len(shape):
            raise IndexError(f'axis value {original_k} out of range for shape {shape}')
        if k in new_axis:
            raise ValueError(f'duplicated array axis {original_k} for shape {shape}')
        new_axis.append(k)

    return tuple(new_axis)


def _pixel_area(
    axes: tuple[int, ...],
    shape: tuple[int, ...],
) -> int:
    """The number of pixels along the specified axes.

    Parameters:
        axes: Axis indices.
        shape: Overall shape of the image array.

    Returns:
        Product of the lengths along the specified axes.
    """

    lengths = [shape[k] for k in axes]
    return int(np.prod(lengths))


def _flatten_axes(
    image: np.ndarray | None,
    axes: int | tuple[int, ...],
    shape: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    """Move selected axes to the front and flatten them.

    Parameters:
        image: Array to rearrange, or None.
        axes: Axis index or tuple of axis indices to move to the front.
        shape: If provided and `image` has a different shape, broadcast `image` to this
            shape first.

    Returns:
        The rearranged array with the selected axes moved to the front and flattened into
        one axis, or None if `image` is None.
    """

    if image is None:
        return None

    if shape is not None and image.shape != shape:
        image = np.broadcast_to(image, shape)

    if isinstance(axes, numbers.Integral):
        axes = (axes,)

    naxes = len(axes)
    image = np.moveaxis(image, axes, tuple(range(naxes)))   # move to front
    image = image.reshape((-1,) + image.shape[naxes:])
    return image


def _merge_weights(
    mask: np.ndarray | None,
    weights: np.ndarray | None,
    factors: npt.ArrayLike | None = None,
) -> np.ndarray | None:
    """Merge an optional mask, weights, and factors into a single weight array.

    Parameters:
        mask: A mask array as returned by `_check_image`.
        weights: A weight array as returned by `_check_image`.
        factors: An extra weight factor that applies to entire 2-D images.

    Returns:
        None if all the inputs are None; otherwise, a weight array that can be
        broadcasted to the shape of the image.
    """

    if factors is not None:
        factors = np.asarray(factors)[:, np.newaxis, np.newaxis]

    if weights is None:
        if mask is None:
            return factors      # could be None
        weights = np.logical_not(mask)

    if factors is None:
        return weights

    return weights * factors

##########################################################################################
