##########################################################################################
# tests/test_resample.py
##########################################################################################

import numpy as np
import pytest

from psiops._filter import _use_shortcuts
from psiops.resample import resample
from psiops.unzoom import unzoom
from psiops.zoom import zoom
from tests._resize_reference import resize  # reference impl, kept for cross-testing

# resample() requires at least three dimensions, so every test uses a 3-D image.

def _cube_image() -> np.ndarray:
    array = np.arange(10)
    return array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]


##########################################################################################
# Cross-checks against zoom() and unzoom()
##########################################################################################

def test_resample_vs_zoom_isotropic(shortcuts: bool) -> None:
    image = _cube_image()
    zoomed = zoom(image, 3)
    resampled, rcenter = resample(image, 3)
    assert np.all(zoomed == resampled)
    assert rcenter == (15, 15)


def test_resample_vs_zoom_anisotropic(shortcuts: bool) -> None:
    image = _cube_image()
    zoomed = zoom(image, (2, 3))
    resampled, rcenter = resample(image, (2, 3))
    assert np.all(zoomed == resampled)
    assert rcenter == (10, 15)


def test_resample_vs_zoom_with_mask(shortcuts: bool) -> None:
    rng = np.random.default_rng(8107)
    image = _cube_image()
    mask = rng.standard_normal((10, 10)) < 0.3

    zoomed, zmask = zoom(image, (2, 3), mask)
    resampled, rmask, rcenter = resample(image, (2, 3), mask)
    zmask_b = np.broadcast_to(zmask, zoomed.shape)
    assert np.all(zmask_b == rmask)
    assert np.all(zoomed[~zmask_b] == resampled[~rmask])
    assert rcenter == (10, 15)


def test_resample_vs_unzoom_with_mask(shortcuts: bool) -> None:
    rng = np.random.default_rng(8107)
    image = _cube_image()
    mask = rng.standard_normal((10, 10)) < 0.3

    zoomed, zmask = unzoom(image, 2, mask)
    resampled, rmask, rcenter = resample(image, 0.5, mask)
    zmask_b = np.broadcast_to(zmask, zoomed.shape)
    assert np.all(zmask_b == rmask)
    assert np.all(zoomed[~zmask_b] == resampled[~rmask])
    assert rcenter == (2.5, 2.5)


def test_resample_vs_zoom_unzoom_mixed(shortcuts: bool) -> None:
    rng = np.random.default_rng(8107)
    image = _cube_image()
    mask = rng.standard_normal((10, 10)) < 0.3

    zoomed, zmask = zoom(image, (3, 1), mask)
    zoomed, zmask = unzoom(zoomed, (1, 2), zmask)
    resampled, rmask, rcenter = resample(image, (3, 0.5), mask)
    zmask_b = np.broadcast_to(zmask, zoomed.shape)
    assert np.all(zmask_b == rmask)
    assert np.all(zoomed[~zmask_b] == resampled[~rmask])
    assert rcenter == (15, 2.5)


##########################################################################################
# Cross-checks against the resize() helper (non-integer zoom factors)
##########################################################################################

def _resize_image() -> np.ndarray:
    array1 = np.arange(4)
    array2 = 2 * np.arange(4)
    array3 = 3 * np.arange(3)
    return array1 + array2[:, np.newaxis] + array3[:, np.newaxis, np.newaxis]


def test_resample_vs_resize_shrink_square(shortcuts: bool) -> None:
    eps = 1.e-14
    image = _resize_image()
    resized = resize(image, (3, 3))
    resampled, rcenter = resample(image, (3/4., 3/4.))
    assert np.max(np.abs(resized - resampled)) < eps
    assert rcenter == (1.5, 1.5)


def test_resample_vs_resize_shrink_square_masked(shortcuts: bool) -> None:
    eps = 1.e-14
    rng = np.random.default_rng(8107)
    image = _resize_image()
    mask = rng.standard_normal((4, 4)) < 0.3
    resized, zmask = resize(image, (3, 3), mask)
    resampled, rmask, rcenter = resample(image, (3/4., 3/4.), mask)
    assert np.all(rmask == zmask)
    assert np.max(np.abs(resized[~rmask] - resampled[~rmask])) < eps
    assert rcenter == (1.5, 1.5)


def test_resample_vs_resize_rectangular(shortcuts: bool) -> None:
    eps = 1.e-14
    image = _resize_image()
    resized = resize(image, (3, 5))
    resampled, rcenter = resample(image, (3/4., 5/4.))
    assert np.all(np.abs(resized - resampled) < eps)
    assert rcenter == (1.5, 2.5)


def test_resample_vs_resize_rectangular_masked(shortcuts: bool) -> None:
    eps = 1.e-14
    rng = np.random.default_rng(8107)
    image = _resize_image()
    mask = rng.standard_normal((4, 4)) < 0.3
    resized, zmask = resize(image, (3, 5), mask)
    resampled, rmask, rcenter = resample(image, (3/4., 5/4.), mask)
    assert np.all(rmask == zmask)
    assert np.max(np.abs(resized[~rmask] - resampled[~rmask])) < eps
    assert rcenter == (1.5, 2.5)


def _prime_image() -> np.ndarray:
    array1 = np.arange(47)
    array2 = 2 * np.arange(13)
    array3 = 3 * np.arange(6).reshape(2, 3)
    return array1 + array2[:, np.newaxis] + array3[..., np.newaxis, np.newaxis]


def test_resample_vs_resize_primes(shortcuts: bool) -> None:
    eps = 1.e-12
    image = _prime_image()
    resized = resize(image, (97, 11))
    resampled, rcenter = resample(image, (97/13., 11/47.))
    assert np.max(np.abs(resized - resampled)) < eps
    assert rcenter == (97/2., 11/2.)


@pytest.mark.parametrize('frac', [0.02, 0.3, 0.7, 0.9, 0.98])
def test_resample_vs_resize_primes_masked(shortcuts: bool, frac: float) -> None:
    eps = 1.e-12
    rng = np.random.default_rng(8107)
    image = _prime_image()
    mask = rng.standard_normal(image.shape) < frac
    resized, zmask = resize(image, (97, 11), mask)
    resampled, rmask, rcenter = resample(image, (97/13., 11/47.), mask)
    assert np.all(rmask == zmask)
    assert np.max(np.abs(resized[~zmask] - resampled[~zmask])) < eps
    assert rcenter == (97/2., 11/2.)


##########################################################################################
# Additional input parameters (origin, center, shape)
##########################################################################################

def _param_image() -> np.ndarray:
    return (np.arange(4) + 2 * np.arange(1, 7)[:, np.newaxis])[np.newaxis]


def test_resample_default_origin(shortcuts: bool) -> None:
    image = _param_image()
    ref, rmask, rcenter = resample(image, 1.5, returns='imc')
    assert not np.any(rmask)
    assert ref.shape == (1, 9, 6)
    assert np.mean(image) == np.mean(ref)
    assert rcenter == (4.5, 3)


def test_resample_explicit_origin(shortcuts: bool) -> None:
    image = _param_image()
    ref = resample(image, 1.5)[0]
    resampled, rmask, rcenter = resample(image, 1.5, origin=(1, 1), returns='imc')
    assert np.all(resampled == ref)
    assert not np.any(rmask)
    assert rcenter == (1.5, 1.5)


def test_resample_origin_and_center_aligned(shortcuts: bool) -> None:
    image = _param_image()
    ref = resample(image, 1.5)[0]
    resampled, rmask, rcenter = resample(image, 1.5, origin=(2, 2), center=(3, 3),
                                         returns='imc')
    assert np.all(resampled == ref)
    assert not np.any(rmask)
    assert rcenter == (3, 3)


def test_resample_origin_and_center_shifted(shortcuts: bool) -> None:
    image = _param_image()
    ref = resample(image, 1.5)[0]
    resampled, rmask, rcenter = resample(image, 1.5, origin=(2, 2), center=(4, 4),
                                         returns='imc')
    assert np.all(resampled[0][1:, 1:] == ref[0])
    assert not np.any(rmask[0][1:, 1:])
    assert np.all(rmask[0][0])
    assert np.all(rmask[0][:, 0])
    assert rcenter == (4, 4)


def test_resample_explicit_shape(shortcuts: bool) -> None:
    image = _param_image()
    ref = resample(image, 1.5)[0]
    resampled, rmask, rcenter = resample(image, 1.5, shape=(11, 8), returns='imc')
    assert np.all(resampled[0][:-2, :-2] == ref[0])
    assert not np.any(rmask[0][:-2, :-2])
    assert np.all(rmask[0][-2:])
    assert np.all(rmask[0][:, -2:])
    assert rcenter == (4.5, 3)


def test_resample_shape_and_center(shortcuts: bool) -> None:
    image = _param_image()
    ref = resample(image, 1.5)[0]
    resampled, rmask, rcenter = resample(image, 1.5, shape=(11, 8), center=(4.5, 3),
                                         returns='imc')
    assert np.all(resampled[0][:-2, :-2] == ref[0])
    assert not np.any(rmask[0][:-2, :-2])
    assert np.all(rmask[0][-2:])
    assert np.all(rmask[0][:, -2:])
    assert rcenter == (4.5, 3)


def test_resample_origin_and_shape(shortcuts: bool) -> None:
    image = _param_image()
    ref = resample(image, 1.5)[0]
    resampled, rmask, rcenter = resample(image, 1.5, origin=(0, 2), shape=(6, 6),
                                         returns='imc')
    assert np.all(resampled[0] == ref[0][:6, :6])
    assert not np.any(rmask)
    assert rcenter == (0, 3)


##########################################################################################
# dtype, returns, and error handling
##########################################################################################

def test_resample_float32_preserved(shortcuts: bool) -> None:
    array = np.arange(10).astype('float32')
    image = array + array[:, np.newaxis] + array[:, np.newaxis, np.newaxis]
    resampled = resample(image, (12, 9))[0]
    assert image.dtype == resampled.dtype


def test_resample_returns_i_is_bare_array(shortcuts: bool) -> None:
    image = _cube_image()
    resampled = resample(image, 2, returns='i')
    assert isinstance(resampled, np.ndarray)
    assert resampled.shape == (10, 20, 20)


def test_resample_returns_iwc(shortcuts: bool) -> None:
    image = _cube_image()
    mask = np.zeros((10, 10), dtype='bool')
    mask[0, 0] = True
    resampled, weights, center = resample(image, 2, mask, returns='iwc')
    assert resampled.shape == (10, 20, 20)
    assert weights.shape == (10, 20, 20)
    assert center == (10, 10)


def test_resample_maskedarray(shortcuts: bool) -> None:
    image = _cube_image().astype(np.float64)
    mask = np.zeros(image.shape, dtype='bool')
    mask[:, 0, 0] = True
    marray = np.ma.MaskedArray(image, mask=mask)
    resampled = resample(marray, 2)[0]
    assert isinstance(resampled, np.ma.MaskedArray)


def test_resample_maskval(shortcuts: bool) -> None:
    image = _cube_image().astype(np.float64)
    image[:, 2, 2] = -999.
    # Use a zoom factor != 1 so the general resampling path is exercised; source pixel
    # (2,2) maps to the 2x2 output block at (4:6, 4:6).
    resampled, rmask, _ = resample(image, 2, maskval=-999., returns='imc')
    assert np.all(rmask[:, 4:6, 4:6])
    assert np.all(resampled[:, 4:6, 4:6] == -999.)


def test_resample_invalid_returns(shortcuts: bool) -> None:
    image = _cube_image()
    with pytest.raises(ValueError):
        resample(image, 2, returns='xyz')


def test_resample_2d_matches_3d(shortcuts: bool) -> None:
    # A bare 2-D image is resampled directly (no leading axes are required). The result
    # must match the single-layer 3-D path with the unit axis squeezed off.
    rng = np.random.default_rng(4521)
    image = rng.random((9, 11))
    for factor in (2, 0.5, 1.7, 1):
        flat = resample(image, factor, returns='i')
        stacked = resample(image[np.newaxis], factor, returns='i')
        assert flat.shape == stacked.shape[1:]
        assert np.allclose(flat, stacked[0], equal_nan=True)


def test_resample_2d_with_mask(shortcuts: bool) -> None:
    rng = np.random.default_rng(99)
    image = rng.random((8, 10))
    mask = rng.random((8, 10)) < 0.3
    flat, fmask = resample(image, 1.5, mask=mask, returns='im')
    stacked, smask = resample(image[np.newaxis], 1.5, mask=mask[np.newaxis], returns='im')
    assert flat.shape == stacked.shape[1:]
    assert np.allclose(flat, stacked[0], equal_nan=True)
    assert np.array_equal(fmask, smask[0])


def test_resample_2d_returns_modes(shortcuts: bool) -> None:
    # Every `returns` mode yields 2-D outputs for a 2-D input.
    image = np.arange(100.0).reshape(10, 10)
    for ret, count in (('i', 1), ('im', 2), ('iw', 2), ('imw', 3)):
        out = resample(image, 2, returns=ret)
        results = out if isinstance(out, list) else [out]
        assert len(results) == count
        assert all(r.shape == (20, 20) for r in results)


def test_resample_shrink_tiny_source_onto_large_grid(shortcuts: bool) -> None:
    # Regression: shrinking a small source onto a much larger output grid used to raise in
    # the index-uniqueness reassignment, because there were no spare source indices to
    # absorb the many out-of-range output pixels. The out-of-range read indices are now
    # clamped (their weight is zero), so the source is placed and the rest is masked.
    image = np.zeros((7, 9))
    image[2:5, 2:7] = 1.0   # 3x5 ones with a 2-pixel zero border, wide enough that the
                            # shrink window never reaches the array edge
    result, mask = resample(image, 0.5, origin=(3.5, 4.5), center=(25., 35.),
                            shape=(60, 80), returns='im')
    assert result.shape == (60, 80)
    assert mask.shape == (60, 80)
    assert mask.any()           # the small source covers only part of the large grid
    assert not mask.all()
    # The whole shrunk source lands inside the grid, so signal scales by exactly zoom**2.
    # The zero border keeps the edge-replicating boundary from inflating the integral.
    assert result[~mask].sum() == pytest.approx(image.sum() * 0.5**2, rel=1e-6)


def test_resample_negative_dimensions(shortcuts: bool) -> None:
    rng = np.random.default_rng(8107)
    image = rng.random((3, 100, 100))
    with pytest.raises(ValueError) as exc_info:
        resample(image, 2, origin=(0, 0), center=(-1000, 0))
    assert (str(exc_info.value).partition(':')[0]
            == 'negative dimensions are not allowed')


def test_resample_unit_zoom_integer_offset() -> None:
    # The zoom_==(1,1) fast path relocates the image with shift(). With an integer
    # offset it is a pure integer shift; the origin defaults to shape/2 = (4, 5).
    _use_shortcuts(True)
    image = np.arange(80, dtype=float).reshape(1, 8, 10)

    shifted = resample(image, 1, center=(6, 8))     # offset (2, 3)
    assert shifted.shape == (1, 10, 13)
    assert np.allclose(shifted[:, 2:8, 3:10], image[:, 0:6, 0:7])


def test_resample_unit_zoom_fractional_offset() -> None:
    _use_shortcuts(True)
    image = np.arange(80, dtype=float).reshape(1, 8, 10)

    # A fractional offset enlarges the buffer by one pixel before trimming
    frac, fmask = resample(image, 1, center=(6.5, 8.0), returns='im')
    assert frac.shape == (1, 11, 13)
    assert fmask.shape[-2:] == frac.shape[-2:]


def test_resample_unit_zoom_with_mask() -> None:
    _use_shortcuts(True)
    image = np.arange(80, dtype=float).reshape(1, 8, 10)
    mask = np.zeros((1, 8, 10), dtype=bool)
    mask[:, 0, 0] = True

    _, mmask = resample(image, 1, center=(6, 8), mask=mask, returns='im')
    # The masked source pixel (0, 0) moves to (2, 3) in the output
    assert mmask[:, 2, 3].all()


def test_resample_unit_zoom_with_weights() -> None:
    _use_shortcuts(True)
    image = np.arange(80, dtype=float).reshape(1, 8, 10)
    weights = np.full((1, 8, 10), 2.0)

    weighted, new_weights = resample(image, 1, center=(6, 8), weights=weights,
                                     returns='iw')
    assert weighted.shape == (1, 10, 13)
    assert np.isclose(new_weights.max(), 2.0)


def test_resample_unit_zoom_unweighted_boundary_mask(shortcuts: bool) -> None:
    # Regression: the zoom_==(1,1) shortcut used to drop the boundary mask when no weights
    # were supplied, leaving output pixels with no source coverage unmasked (and the mask
    # came back 2-D). Both the shortcut and general paths must flag the exposed boundary.
    image = np.arange(80, dtype=float).reshape(1, 8, 10)

    # Integer offset (2, 3): the source fills output rows 2:10 and cols 3:13; the rest of
    # the (10, 13) output has no coverage and must be masked.
    _, mask = resample(image, 1, center=(6, 8), returns='im')
    assert mask.shape == (1, 10, 13)
    assert mask[:, :2, :].all()             # top rows have no source coverage
    assert mask[:, :, :3].all()             # left columns have no source coverage
    assert not mask[:, 2:, 3:].any()        # the covered region is unmasked


def test_resample_unit_zoom_shortcut_matches_general() -> None:
    # The zoom_==(1,1) shortcut must give the same answer as the general path. Both
    # renormalize partially-covered boundary pixels so edge pixels keep their intensity
    # (off-edge == nearest in-image pixel). Compare both paths across integer/fractional
    # offsets and unweighted, weighted, and masked input.
    rng = np.random.default_rng(2024)
    image = rng.random((1, 6, 7))
    for center in [(3.3, 4.7), (3.0, 4.0), (3.5, 3.0)]:
        for kwargs in ({}, {'weights': rng.random((1, 6, 7)) + 0.1},
                       {'mask': rng.random((1, 6, 7)) < 0.25}):
            results = []
            for use in (False, True):
                _use_shortcuts(use)
                results.append(resample(image, 1, origin=(3., 3.5), center=center,
                                        shape=(6, 7), returns='imw', **kwargs))
            (gi, gm, gw), (si, sm, sw) = results
            assert np.array_equal(sm, gm)
            assert np.allclose(si, gi, equal_nan=True)
            assert np.allclose(sw, gw, equal_nan=True)

##########################################################################################
