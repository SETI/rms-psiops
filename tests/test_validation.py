##########################################################################################
# tests/test_validation.py
##########################################################################################

import numpy as np
import pytest

import psiops
from psiops._utils import _ImageInfo
from psiops._validation import _check_image, _check_return

# Every stack reduction that supports `keepdims`. Each is exercised on its bare-image
# (no mask/weights) return path, which is where `keepdims` was previously dropped.
_REDUCTIONS = ['mean', 'median', 'minimum', 'maximum', 'variance', 'stdev']


def test_check_return_keepdims_bare_image() -> None:
    # Regression: when returns == 'i' (no mask/weights), _check_return must still
    # re-insert the reduced axes recorded in info.axis. Previously it returned the
    # array unchanged, silently ignoring keepdims.
    reduced = np.arange(12, dtype=float).reshape(3, 4)
    info = _ImageInfo(returns='i', axis=(0,))

    result = _check_return(reduced, None, None, info)

    assert result.shape == (1, 3, 4)
    assert np.all(result[0] == reduced)


def test_check_return_keepdims_bare_image_multiple_axes() -> None:
    # The reduced axes are re-inserted at each recorded position, in order.
    reduced = np.arange(12, dtype=float).reshape(3, 4)
    info = _ImageInfo(returns='i', axis=(0, 2))

    result = _check_return(reduced, None, None, info)

    assert result.shape == (1, 3, 1, 4)
    assert np.all(result[0, :, 0, :] == reduced)


def test_check_return_no_keepdims_bare_image_unchanged() -> None:
    # With no recorded axes, the bare-image path returns the array unchanged.
    reduced = np.arange(12, dtype=float).reshape(3, 4)
    info = _ImageInfo(returns='i')

    result = _check_return(reduced, None, None, info)

    assert result.shape == (3, 4)
    assert np.all(result == reduced)


@pytest.mark.parametrize('name', _REDUCTIONS)
def test_reduction_keepdims_bare_image_shape(name: str) -> None:
    # The public reductions must honor keepdims on the default (no-mask) return,
    # which uses the returns == 'i' path.
    rng = np.random.default_rng(7)
    stack = rng.random((4, 3, 8, 10))
    fn = getattr(psiops, name)

    kept = fn(stack, axis=0, keepdims=True)
    plain = fn(stack, axis=0)

    assert isinstance(kept, np.ndarray)         # bare-image path, not a list
    assert kept.shape == (1, 3, 8, 10)
    assert plain.shape == (3, 8, 10)
    # keepdims only changes shape, not values
    assert np.allclose(np.squeeze(kept, axis=0), plain, equal_nan=True)


@pytest.mark.parametrize('name', _REDUCTIONS)
def test_reduction_keepdims_multiple_axes(name: str) -> None:
    rng = np.random.default_rng(11)
    stack = rng.random((4, 3, 8, 10))
    fn = getattr(psiops, name)

    kept = fn(stack, axis=(0, 1), keepdims=True)

    assert kept.shape == (1, 1, 8, 10)


@pytest.mark.parametrize('name', _REDUCTIONS)
def test_reduction_keepdims_matches_numpy_shape(name: str) -> None:
    # The kept shape matches what numpy produces for the same reduction.
    rng = np.random.default_rng(13)
    stack = rng.random((5, 2, 6, 6))
    fn = getattr(psiops, name)

    kept = fn(stack, axis=1, keepdims=True)

    assert kept.shape == np.mean(stack, axis=1, keepdims=True).shape


def test_mean_keepdims_with_mask_still_works() -> None:
    # The multi-return (mask) path already applied keepdims; confirm it is unaffected.
    rng = np.random.default_rng(17)
    stack = rng.random((4, 3, 8, 10))
    mask = rng.random((4, 3, 8, 10)) < 0.2

    kept, kmask = psiops.mean(stack, axis=0, mask=mask, keepdims=True)

    assert kept.shape == (1, 3, 8, 10)
    assert kmask.shape == (1, 3, 8, 10)

##########################################################################################
# _check_image: error paths
##########################################################################################

def test_check_image_invalid_returns() -> None:
    # An unrecognized `returns` string is rejected before anything else.
    rng = np.random.default_rng(101)
    image = rng.random((4, 5))

    exc_info: pytest.ExceptionInfo[Exception]
    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, returns='bogus')
    assert str(exc_info.value) == 'invalid `returns` value "bogus"'


def test_check_image_returns_extra_char_not_allowed_without_flag() -> None:
    # Appending `extra_char` is only valid when `extra_char` is supplied.
    rng = np.random.default_rng(102)
    image = rng.random((4, 5))

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, returns='ix')
    assert str(exc_info.value) == 'invalid `returns` value "ix"'


def test_check_image_non_numeric_dtype() -> None:
    # A string array has dtype kind 'U', which is not numeric.
    image = np.array([['a', 'b'], ['c', 'd']])

    with pytest.raises(TypeError) as exc_info:
        _ = _check_image(image)
    assert str(exc_info.value) == f'image dtype {image.dtype} is not numeric'


def test_check_image_object_dtype_not_numeric() -> None:
    # An object-dtype array also fails the numeric-dtype check.
    image = np.empty((2, 2), dtype=object)

    with pytest.raises(TypeError) as exc_info:
        _ = _check_image(image)
    assert str(exc_info.value) == 'image dtype object is not numeric'


def test_check_image_complex_without_comps() -> None:
    # Complex values are rejected unless comps=True.
    image = np.ones((3, 3), dtype=np.complex128)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image)
    assert str(exc_info.value) == 'complex image values are not supported'


def test_check_image_complex_with_comps_ok() -> None:
    # With comps=True the complex image passes through unchanged.
    image = np.ones((3, 3), dtype=np.complex128)

    out_image, mask, weights, info = _check_image(image, comps=True)

    assert out_image.dtype.kind == 'c'
    assert mask is None
    assert weights is None
    assert info.returns == 'i'


def test_check_image_less_than_2d() -> None:
    # A 1-D image fails the minimum-dimension check.
    image = np.arange(5, dtype=float)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image)
    assert str(exc_info.value) == 'invalid image shape (5,); must be at least 2-D'


def test_check_image_two_flag_requires_exactly_2d() -> None:
    # With two=True, anything other than 2-D is rejected.
    image = np.ones((2, 3, 4), dtype=float)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, two=True)
    assert str(exc_info.value) == 'invalid image shape (2, 3, 4); must be 2-D'


def test_check_image_three_flag_requires_at_least_3d() -> None:
    # With three=True, a 2-D image is rejected.
    image = np.ones((3, 4), dtype=float)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, three=True)
    assert str(exc_info.value) == 'invalid image shape (3, 4); must be at least 3-D'


def test_check_image_zero_size() -> None:
    # A 2-D but empty image is rejected.
    image = np.zeros((0, 4), dtype=float)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image)
    assert str(exc_info.value) == 'invalid image shape (0, 4); size cannot be zero'


def test_check_image_mask_less_than_2d() -> None:
    # A 1-D mask (and not a scalar) is rejected.
    rng = np.random.default_rng(103)
    image = rng.random((4, 5))
    mask = np.zeros(5, dtype=bool)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, mask=mask)
    assert str(exc_info.value) == 'illegal mask shape (5,); must be at least 2-D'


def test_check_image_mask_shape_mismatch() -> None:
    # A mask whose trailing two axes differ from the image is rejected.
    rng = np.random.default_rng(104)
    image = rng.random((4, 5))
    mask = np.zeros((4, 6), dtype=bool)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, mask=mask)
    assert str(exc_info.value) == ('mask and image have incompatible shapes: (4, 6), '
                                   '(4, 5)')


def test_check_image_mask_broadcast_mismatch() -> None:
    # Trailing axes match but the leading axes cannot broadcast to the image.
    rng = np.random.default_rng(105)
    image = rng.random((3, 4, 5))
    mask = np.zeros((2, 4, 5), dtype=bool)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, mask=mask)
    assert str(exc_info.value) == ('mask and image have incompatible shapes: (2, 4, 5), '
                                   '(3, 4, 5)')


def test_check_image_weights_less_than_2d() -> None:
    # A 1-D weights array is rejected.
    rng = np.random.default_rng(106)
    image = rng.random((4, 5))
    weights = np.ones(5, dtype=np.float64)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, weights=weights)
    assert str(exc_info.value) == ('illegal weights shape (5,); must be at least 2-D')


def test_check_image_weights_shape_mismatch() -> None:
    # Weights whose trailing two axes differ from the image are rejected.
    rng = np.random.default_rng(107)
    image = rng.random((4, 5))
    weights = np.ones((4, 6), dtype=np.float64)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, weights=weights)
    assert str(exc_info.value) == ('weights and image have incompatible shapes: (4, 6), '
                                   '(4, 5)')


def test_check_image_weights_broadcast_mismatch() -> None:
    # Trailing axes match but the leading axes cannot broadcast to the image.
    rng = np.random.default_rng(108)
    image = rng.random((3, 4, 5))
    weights = np.ones((2, 4, 5), dtype=np.float64)

    with pytest.raises(ValueError) as exc_info:
        _ = _check_image(image, weights=weights)
    assert str(exc_info.value) == ('weights and image have incompatible shapes: '
                                   '(2, 4, 5), (3, 4, 5)')

##########################################################################################
# _check_image: happy paths reaching mask/weights merging
##########################################################################################

def test_check_image_maskedarray_mask_none() -> None:
    # A MaskedArray with no explicit mask adopts the array's own mask and fill_value.
    rng = np.random.default_rng(109)
    data = rng.random((4, 5))
    amask = rng.random((4, 5)) < 0.3
    image = np.ma.MaskedArray(data=data, mask=amask, fill_value=-1.0)

    out_image, mask, _weights, info = _check_image(image)

    assert not isinstance(out_image, np.ma.MaskedArray)
    assert info.is_maskedarray is True
    assert info.fill_value == -1.0
    assert mask is not None
    assert np.array_equal(mask, amask)
    # A MaskedArray with mask_was_none keeps 'i' (the 'm' is suppressed).
    assert info.returns == 'i'


def test_check_image_maskedarray_merges_explicit_mask() -> None:
    # An explicit mask is OR-ed with the MaskedArray's own mask.
    rng = np.random.default_rng(110)
    data = rng.random((4, 5))
    amask = np.zeros((4, 5), dtype=bool)
    amask[0, 0] = True
    image = np.ma.MaskedArray(data=data, mask=amask)
    extra = np.zeros((4, 5), dtype=bool)
    extra[1, 1] = True

    _out_image, mask, _weights, info = _check_image(image, extra)

    assert mask[0, 0]
    assert mask[1, 1]
    assert int(np.sum(mask)) == 2
    # An explicit mask on a MaskedArray still suppresses 'm' (is_maskedarray).
    assert info.returns == 'i'


def test_check_image_scalar_mask_expands() -> None:
    # A scalar (0-D) mask is expanded to the image's trailing two axes.
    rng = np.random.default_rng(111)
    image = rng.random((4, 5))

    _out_image, mask, _weights, info = _check_image(image, mask=True)

    assert mask.shape == (4, 5)
    assert np.all(mask)
    assert info.returns == 'im'


def test_check_image_maskval() -> None:
    # `maskval` masks every pixel equal to that value and records it as fill_value.
    image = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])

    _out_image, mask, _weights, info = _check_image(image, maskval=3.0)

    assert mask[0, 2]
    assert mask[1, 0]
    assert int(np.sum(mask)) == 2
    assert info.fill_value == 3.0
    # `mask` was None on input, so 'm' is NOT added to `returns` even though a mask is
    # now built from `maskval`.
    assert info.returns == 'i'


def test_check_image_nans() -> None:
    # With nans=True, NaN pixels are masked and fill_value becomes NaN.
    image = np.array([[1.0, np.nan], [3.0, 4.0]])

    _out_image, mask, _weights, info = _check_image(image, nans=True)

    assert mask[0, 1]
    assert int(np.sum(mask)) == 1
    assert np.isnan(info.fill_value)
    # As with `maskval`, a derived NaN mask does not add 'm' to `returns`.
    assert info.returns == 'i'


def test_check_image_maskval_overrides_nan_fill_value() -> None:
    # maskval is applied after nans, so its value wins as the fill_value.
    image = np.array([[1.0, np.nan], [7.0, 4.0]])

    _out_image, mask, _weights, info = _check_image(image, maskval=7.0, nans=True)

    assert mask[0, 1]       # NaN masked
    assert mask[1, 0]       # maskval masked
    assert info.fill_value == 7.0


def test_check_image_maskval_broadcasts_existing_mask() -> None:
    # An existing mask whose shape differs from (but broadcasts to) the image is
    # broadcast to the full image shape so maskval can be applied element-wise. The mask
    # has matching trailing two axes but a unit leading axis.
    image = np.array([[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]],
                      [[3.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])   # shape (2, 2, 3)
    mask = np.array([[[True, False, False], [False, False, False]]])  # shape (1, 2, 3)

    _out_image, out_mask, _weights, _info = _check_image(image, mask=mask, maskval=3.0)

    assert out_mask.shape == (2, 2, 3)
    assert out_mask[0, 0, 0]    # from the original mask (broadcast to both layers)
    assert out_mask[1, 0, 0]    # broadcast original mask
    assert out_mask[0, 0, 2]    # maskval in layer 0
    assert out_mask[0, 1, 0]    # maskval in layer 0
    assert out_mask[1, 0, 0]    # maskval in layer 1


def test_check_image_weights_into_mask_no_mask() -> None:
    # With weights but no mask, the mask is derived from zero weights.
    weights = np.array([[1.0, 0.0], [2.0, 3.0]])
    image = np.ones((2, 2), dtype=float)

    _out_image, mask, _out_weights, info = _check_image(image, weights=weights)

    assert mask is not None
    assert mask[0, 1]
    assert int(np.sum(mask)) == 1
    assert info.returns == 'iw'


def test_check_image_weights_merge_with_mask() -> None:
    # With both mask and weights of the same shape, zero weights are OR-ed into the
    # mask and masked locations are zeroed in the weights.
    image = np.ones((2, 2), dtype=float)
    mask = np.array([[True, False], [False, False]])
    weights = np.array([[2.0, 0.0], [3.0, 4.0]])

    _out_image, out_mask, out_weights, info = _check_image(image, mask=mask,
                                                          weights=weights)

    assert out_mask[0, 0]       # from the original mask
    assert out_mask[0, 1]       # from zero weight
    assert out_weights[0, 0] == 0.0     # masked location zeroed
    assert out_weights[0, 1] == 0.0
    assert out_weights[1, 1] == 4.0
    assert info.returns == 'imw'


def test_check_image_weights_broadcast_mask() -> None:
    # When mask and weights have different but broadcast-compatible shapes, they are
    # broadcast against one another before merging. The mask has a unit leading axis.
    image = np.ones((2, 2, 3), dtype=float)
    mask = np.array([[[True, False, False],
                      [False, False, False]]])          # shape (1, 2, 3)
    weights = np.array([[[1.0, 2.0, 0.0],
                         [1.0, 2.0, 3.0]],
                        [[1.0, 2.0, 3.0],
                         [1.0, 2.0, 3.0]]])             # shape (2, 2, 3)

    _out_image, out_mask, out_weights, _info = _check_image(image, mask=mask,
                                                          weights=weights)

    assert out_mask.shape == (2, 2, 3)
    assert out_weights.shape == (2, 2, 3)
    assert out_mask[0, 0, 0]                            # broadcast original mask
    assert out_mask[1, 0, 0]                            # broadcast original mask
    assert out_mask[0, 0, 2]                            # zero weight
    assert out_weights[0, 0, 0] == 0.0                  # masked, zeroed
    assert out_weights[1, 1, 2] == 3.0


def test_check_image_floats_converts_int() -> None:
    # With floats=True an integer image is promoted to float64 and flagged as a copy.
    image = np.arange(6, dtype=np.int32).reshape(2, 3)

    out_image, _mask, _weights, info = _check_image(image, floats=True)

    assert out_image.dtype == np.float64
    assert info.image_is_copy is True


def test_check_image_floats_keeps_float32() -> None:
    # float32 is left unchanged by floats=True.
    rng = np.random.default_rng(112)
    image = rng.random((3, 3)).astype(np.float32)

    out_image, _mask, _weights, _info = _check_image(image, floats=True)

    assert out_image.dtype == np.float32


def test_check_image_floats_from_list() -> None:
    # A non-ndarray input with floats=True is converted via np.asarray(float64).
    image = [[1, 2, 3], [4, 5, 6]]

    out_image, _mask, _weights, info = _check_image(image, floats=True)

    assert isinstance(out_image, np.ndarray)
    assert out_image.dtype == np.float64
    assert info.image_is_copy is True

##########################################################################################
