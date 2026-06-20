##########################################################################################
# tests/test_rotate.py
##########################################################################################

import numpy as np
import pytest
from psiops.rotate import rotate


def _full_pixel_weights_ok(debug: dict, shape: tuple[int, int],
                           tol: float = 1.e-8) -> float:
    """Return the maximum per-source-pixel total-overlap-area error.

    Each original pixel should distribute a total overlap area of exactly 1 across the
    new, rotated grid.
    """

    alist = np.array(debug['area_list'])
    ilist = np.array(debug['imod_list'])
    jlist = np.array(debug['jmod_list'])
    max_error = 0.
    for i in range(shape[0]):
        for j in range(shape[1]):
            ijmask = (ilist == i) & (jlist == j)
            max_error = max(np.abs(np.sum(alist[ijmask]) - 1), max_error)
    return max_error


##########################################################################################
# Area-conservation tests
##########################################################################################

def test_rotate_45_square_shape_and_weights(shortcuts) -> None:
    image = np.arange(10) + np.arange(10)[:, np.newaxis]
    debug = {}
    result = rotate(image, np.pi/4, _debug=debug)
    rotated = result[0]
    center = debug['new_center']
    weight = debug['new_weights']
    assert rotated.shape == (16, 16)
    assert center == (8, 8)
    assert np.max(np.abs(weight[2:14, 7:9] - 1)) < 1.e-8
    assert np.max(np.abs(weight[3:13, 6:10] - 1)) < 1.e-8
    assert np.max(np.abs(weight[4:12, 5:11] - 1)) < 1.e-8
    assert np.max(np.abs(weight[5:11, 4:12] - 1)) < 1.e-8
    assert np.max(np.abs(weight[6:10, 3:13] - 1)) < 1.e-8
    assert np.max(np.abs(weight[7:9, 2:14] - 1)) < 1.e-8


def test_rotate_45_square_area_conserved(shortcuts) -> None:
    image = np.arange(10) + np.arange(10)[:, np.newaxis]
    debug = {}
    rotate(image, np.pi/4, _debug=debug)
    assert _full_pixel_weights_ok(debug, image.shape) < 1.e-8


def test_rotate_60_nonsquare_shape_and_weights(shortcuts) -> None:
    image = np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
    debug = {}
    result = rotate(image, np.pi/3, _debug=debug)
    rotated = result[0]
    center = debug['new_center']
    weight = debug['new_weights']
    assert rotated.shape == (20, 24)
    assert center == (10, 12)
    assert np.sum(weight > 0.99) == 166


def test_rotate_60_nonsquare_area_conserved(shortcuts) -> None:
    image = np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
    debug = {}
    rotate(image, np.pi/3, _debug=debug)
    assert _full_pixel_weights_ok(debug, image.shape) < 1.e-8


def test_rotate_random_angles_area_conserved(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = np.arange(10) + 2 * np.arange(20)[:, np.newaxis]
    for k in range(300):
        angle = 2 * np.pi * rng.random()
        debug = {}
        rotate(image, angle, _debug=debug)
        if debug['area_list'] is None:
            continue  # exact pi/2 multiple: area list not available
        assert _full_pixel_weights_ok(debug, image.shape) <= 3.e-5


##########################################################################################
# Masked-pixel handling
##########################################################################################

def test_rotate_45_isolated_masked_pixels(shortcuts) -> None:
    image = np.arange(10) + 2 * np.arange(10)[:, np.newaxis]
    debug0 = {}
    result0 = rotate(image, np.pi/4, _debug=debug0)
    rotated0 = result0[0]
    rmask0 = debug0['new_mask']

    mask = np.zeros(image.shape, dtype='bool')
    mask[3, 3] = mask[5, 5] = mask[7, 7] = True
    image[mask] = -999

    debug = {}
    result = rotate(image, np.pi/4, mask, _debug=debug)
    rotated = result[0]
    rmask = debug['new_mask']
    weight = debug['new_weights']
    assert np.all(rmask0 == rmask)
    assert np.max(np.abs(rotated - rotated0)) < 1
    # The total weight equals the number of unmasked pixels to full precision here, since
    # no sub-minweight slivers are discarded for these isolated masked pixels.
    assert abs(np.sum(weight) - np.sum(~mask)) < 1.e-9


def test_rotate_many_masked_pixels_unweighted(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = np.arange(10) + 2 * np.arange(20)[:, np.newaxis]
    for k in range(300):                # advance the RNG to match historical sequence
        _ = 2 * np.pi * rng.random()

    image = np.arange(10) + np.arange(10)[:, np.newaxis]
    debug0 = {}
    result0 = rotate(image, np.pi/5, _debug=debug0)
    rotated0 = result0[0]
    rmask0 = debug0['new_mask']

    mask = rng.random((10, 10)) < 0.8
    image[mask] = -9999
    debug = {}
    result = rotate(image, np.pi/5, mask, _debug=debug)
    rotated = result[0]
    rmask = debug['new_mask']
    weight = debug['new_weights']
    assert np.max(np.abs(rotated[~rmask] - rotated0[~rmask])) < 2
    # The total weight should equal the number of unmasked pixels, except for tiny
    # slivers below `minweight` (1.e-6) that are intentionally discarded. The achievable
    # precision is therefore bounded by `minweight` times the number of new pixels.
    assert abs(np.sum(weight) - np.sum(~mask)) < 1.e-4


##########################################################################################
# Rotations by exact multiples of pi/2
##########################################################################################

def test_rotate_zero_three_layers_no_mask(shortcuts) -> None:
    image = (np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
             + np.arange(3)[:, np.newaxis, np.newaxis])
    rotated = rotate(image, 0.)[0]
    assert np.max(np.abs(image - rotated)) < 1.e-7


def test_rotate_zero_three_layers_random_mask(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = (np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
             + np.arange(3)[:, np.newaxis, np.newaxis])
    mask = rng.random(image.shape[-2:]) < 0.3
    result = rotate(image, 0., mask)
    rotated, rmask = result[0], result[1]
    assert np.all(mask == rmask)
    assert np.max(np.abs(image[~rmask] - rotated[~rmask])) < 1.e-7


def test_rotate_90_three_layers_random_mask(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = (np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
             + np.arange(3)[:, np.newaxis, np.newaxis])
    mask = rng.random(image.shape) < 0.3
    result = rotate(image, np.pi/2, mask)
    rotated, rmask = result[0], result[1]

    test = image.swapaxes(-2, -1)[:, ::-1]
    mtest = mask.swapaxes(-2, -1)[:, ::-1]
    assert np.all(rmask == mtest)
    assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7


def test_rotate_near_90_three_layers_random_mask(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = (np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
             + np.arange(3)[:, np.newaxis, np.newaxis])
    mask = rng.random(image.shape) < 0.3
    test = image.swapaxes(-2, -1)[:, ::-1]
    mtest = mask.swapaxes(-2, -1)[:, ::-1]

    result = rotate(image, np.pi/2 * (1 - 1.e-16), mask)
    rotated, rmask = result[0], result[1]
    assert np.all(rmask == mtest)
    assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7


def test_rotate_180_random_mask(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
    mask = rng.random(image.shape) < 0.3
    result = rotate(image, np.pi, mask)
    rotated, rmask = result[0], result[1]

    test = image[::-1, ::-1]
    mtest = mask[::-1, ::-1]
    assert np.all(rmask == mtest)
    assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7


def test_rotate_270_random_mask(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = np.arange(10) + 3 * np.arange(20)[:, np.newaxis]
    mask = rng.random(image.shape) < 0.3
    result = rotate(image, 1.5 * np.pi, mask)
    rotated, rmask = result[0], result[1]

    test = image.swapaxes(-2, -1)[:, ::-1]
    mtest = mask.swapaxes(-2, -1)[:, ::-1]
    assert np.all(rmask == mtest)
    assert np.max(np.abs(test[~rmask] - rotated[~rmask])) < 1.e-7


##########################################################################################
# dtype, center, and shape options
##########################################################################################

def test_rotate_float32_preserved(shortcuts) -> None:
    array = np.arange(10)
    image = (array + array[:, np.newaxis]).astype('float32')
    rotated = rotate(image, np.pi/3)[0]
    assert image.dtype == rotated.dtype


def test_rotate_half_integer_center(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = rng.random((10, 10))
    rotated, new_center = rotate(image, 0.3, origin=(3.5, 3.5))
    assert new_center[0] % 1 == 0.5
    assert new_center[1] % 1 == 0.5


def test_rotate_integer_center(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = rng.random((10, 10))
    rotated, new_center = rotate(image, 0.4, origin=(3., 4.))
    assert new_center[0] % 1 == 0.
    assert new_center[1] % 1 == 0.


def test_rotate_new_shape_no_center(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = rng.random((10, 10))
    rotated, new_center = rotate(image, 0.3, shape=(20, 20))
    assert new_center == (10, 10)
    assert rotated.shape == (20, 20)


##########################################################################################
# Cross-check of the shortcut and general code paths for zero rotation
##########################################################################################

@pytest.mark.parametrize('use_3d', [False, True])
def test_rotate_zero_shortcuts_match(use_3d) -> None:
    from psiops._filter import _use_shortcuts

    rng = np.random.default_rng(9676)
    if use_3d:
        image = rng.random((4, 10, 10))
    else:
        image = np.arange(10)[:, np.newaxis] + np.arange(10)

    masks = (None, rng.random((10, 10)) < 0.05, rng.random((10, 10)) < 0.45)
    centers = ((4.5, 5.25), None, (4.5 + rng.random(), 4.5 + rng.random()))
    for mask in masks:
        for center in centers:
            _use_shortcuts(False)
            rotated1, mask1, center1 = rotate(image, 0., mask=mask, center=center,
                                              returns='imc')
            _use_shortcuts(True)
            rotated2, mask2, center2 = rotate(image, 0., mask=mask, center=center,
                                              returns='imc')

            assert np.all(mask1 == mask2)
            assert np.abs((rotated1 - rotated2)[~mask1]).max() < 1.e-9
            assert abs(center1[0] - center2[0]) == 0
            assert abs(center1[1] - center2[1]) == 0
            if center is not None:
                assert abs(center1[0] - center[0]) == 0
                assert abs(center1[1] - center[1]) == 0


##########################################################################################
# Error handling and additional coverage
##########################################################################################

def test_rotate_invalid_returns() -> None:
    image = np.arange(10) + np.arange(10)[:, np.newaxis]
    with pytest.raises(ValueError):
        rotate(image, 0.3, returns='qq')


def test_rotate_non_numeric_dtype() -> None:
    image = np.array([['a', 'b'], ['c', 'd']])
    with pytest.raises(TypeError):
        rotate(image, 0.3)


def test_rotate_maskedarray() -> None:
    rng = np.random.default_rng(9676)
    image = rng.random((10, 10))
    mask = np.zeros(image.shape, dtype='bool')
    mask[2, 2] = mask[7, 8] = True
    marray = np.ma.MaskedArray(image, mask=mask)
    rotated = rotate(marray, 0.3)
    assert isinstance(rotated[0], np.ma.MaskedArray)


def test_rotate_with_weights(shortcuts) -> None:
    rng = np.random.default_rng(9676)
    image = rng.random((10, 10))
    weights = np.ones(image.shape)
    weights[0, 0] = 0.
    weights[5, 5] = 3.
    rotated, new_weights, center = rotate(image, 0.3, weights=weights, returns='iwc')
    assert rotated.shape == new_weights.shape


def test_rotate_maskval(shortcuts) -> None:
    image = (np.arange(10) + np.arange(10)[:, np.newaxis]).astype(np.float64)
    image[4, 4] = -999.
    rotated, rmask, center = rotate(image, 0.3, maskval=-999., returns='imc')
    # The masked input value never bleeds into the unmasked output region as -999.
    assert np.all(rotated[~rmask] != -999.)


def test_rotate_explicit_center_skips_new_center_default(shortcuts) -> None:
    image = np.arange(10) + np.arange(10)[:, np.newaxis]
    # With an explicit center and no mask, the default return is image only.
    rotated = rotate(image, 0.3, center=(5, 5))
    assert isinstance(rotated, np.ndarray)
    assert rotated.ndim == 2

##########################################################################################
