##########################################################################################
# tests/test_stretch.py
##########################################################################################
"""Tests for psiops.stretch."""

from typing import Any, cast

import numpy as np
import pytest

from psiops.stretch import Stretch


def _ij(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized (i, j) coordinate arrays matching Stretch's internal convention.

    Stretch maps each axis onto [-1, 1] via (index - half) / half.
    """

    ni, nj = shape
    half_i = 0.5 * (ni - 1)
    half_j = 0.5 * (nj - 1)
    i = (np.arange(ni) - half_i)[:, np.newaxis] / half_i
    j = (np.arange(nj) - half_j) / half_j
    return i, j


def _image(shape: tuple[int, int] = (8, 8), seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(shape)


##########################################################################################
# Construction and rank bookkeeping
##########################################################################################

def test_rank_from_order_sequence() -> None:
    expected = [0, 1, 3, 6, 10, 15]
    got = [Stretch._rank_from_order(o) for o in [-1, 0, 1, 2, 3, 4]]
    assert got == expected


def test_ranks_and_ncoeffs_background_scale() -> None:
    s = Stretch([2, 0])
    assert s.ranks == (6, 1)
    assert s.ncoeffs == 7
    assert s.orders == (2, 0)


def test_ranks_scale_only() -> None:
    s = Stretch([-1, 0])
    assert s.ranks == (0, 1)
    assert s.ncoeffs == 1


def test_set_coeffs_valid() -> None:
    s = Stretch([0, 0])
    s.set_coeffs([3.0, 2.0])
    assert np.array_equal(cast(Any, s.coeffs), [3.0, 2.0])


def test_set_coeffs_wrong_count_raises() -> None:
    s = Stretch([0, 0])
    with pytest.raises(ValueError) as exc_info:
        s.set_coeffs([1.0])
    assert 'incorrect number of parameters' in str(exc_info.value)


##########################################################################################
# Basic fitting: constant background + constant scale
##########################################################################################

def test_fit_recovers_offset_and_scale() -> None:
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(cast(Any, s.coeffs), [3.0, 2.0])


def test_fit_model_matches_target_exactly() -> None:
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.model, target)


def test_background_and_scaling_properties() -> None:
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.background, 3.0)
    assert np.allclose(s.scaling, 2.0)


def test_scale_only_fit() -> None:
    image = _image()
    target = 2.5 * image
    s = Stretch([-1, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(cast(Any, s.coeffs), [2.5])
    assert np.allclose(s.model, target)


def test_perfect_fit_has_negligible_rms() -> None:
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert cast(Any, s.rms) < 1e-10
    assert cast(Any, s.chi_sq) < 1e-20


def test_dof_and_weight_sum_unweighted() -> None:
    image = _image((8, 8))
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert s.weight_sum == 64
    assert s.dof == 64 - 2


##########################################################################################
# Spatially varying backgrounds and scalings
##########################################################################################

def test_order1_background_recovered() -> None:
    shape = (12, 12)
    image = _image(shape)
    i, j = _ij(shape)
    bg = 1.0 + 0.5 * i + 0.3 * j
    target = bg + 2.0 * image
    s = Stretch([1, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(cast(Any, s.coeffs), [1.0, 0.5, 0.3, 2.0])
    assert np.allclose(s.background, bg)
    assert np.allclose(s.scaling, 2.0)


def test_order2_background_recovered() -> None:
    shape = (14, 14)
    image = _image(shape)
    i, j = _ij(shape)
    bg = 1.0 + 0.5 * i + 0.3 * j + 0.2 * i**2 + 0.1 * i * j + 0.4 * j**2
    target = bg + 2.0 * image
    s = Stretch([2, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(cast(Any, s.coeffs), [1.0, 0.5, 0.3, 0.2, 0.1, 0.4, 2.0])
    assert np.allclose(s.background, bg)


def test_spatially_varying_scale_recovered() -> None:
    shape = (12, 12)
    image = _image(shape)
    i, j = _ij(shape)
    scale = 2.0 + 0.5 * i + 0.3 * j
    target = 1.0 + scale * image
    s = Stretch([0, 1], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(cast(Any, s.coeffs), [1.0, 2.0, 0.5, 0.3])
    assert np.allclose(s.scaling, scale)
    assert np.allclose(s.model, target)


def test_quadratic_image_term_recovered() -> None:
    image = _image((10, 10))
    target = 1.0 + 2.0 * image + 0.5 * image**2
    s = Stretch([0, 0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(cast(Any, s.coeffs), [1.0, 2.0, 0.5])
    assert np.allclose(s.model, target)


##########################################################################################
# Masks and weights
##########################################################################################

def test_target_mask_excludes_pixels() -> None:
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = mask[5, 5] = True
    s = Stretch([0, 0], image=image)
    s.set_target(target, mask=mask)
    s.fit()
    assert s.mask is not None
    assert cast(Any, s._antimask).sum() == 98
    assert np.allclose(cast(Any, s.coeffs), [3.0, 2.0])


def test_image_mask_excludes_pixels() -> None:
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    imask = np.zeros((10, 10), dtype=bool)
    imask[1, 1] = True
    s = Stretch([0, 0], image=image, mask=imask)
    s.set_target(target)
    s.fit()
    assert cast(Any, s._antimask).sum() == 99
    assert np.allclose(cast(Any, s.coeffs), [3.0, 2.0])


def test_combined_image_and_target_masks() -> None:
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    imask = np.zeros((10, 10), dtype=bool)
    imask[1, 1] = True
    tmask = np.zeros((10, 10), dtype=bool)
    tmask[2, 2] = True
    s = Stretch([0, 0], image=image, mask=imask)
    s.set_target(target, mask=tmask)
    s.fit()
    assert cast(Any, s._antimask).sum() == 98
    assert cast(Any, s.mask)[1, 1]
    assert cast(Any, s.mask)[2, 2]


def test_weighted_fit_recovers_coeffs() -> None:
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    weights = np.ones((10, 10))
    weights[0, 0] = 0.0
    s = Stretch([0, 0], image=image)
    s.set_target(target, weights=weights)
    s.fit()
    assert np.allclose(cast(Any, s.coeffs), [3.0, 2.0])
    assert s.target_weights is not None


def test_weighted_fit_normalizes_weights() -> None:
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    weights = np.full((10, 10), 4.0)
    s = Stretch([0, 0], image=image)
    s.set_target(target, weights=weights)
    assert np.max(cast(Any, s.target_weights)) == pytest.approx(1.0)


##########################################################################################
# Residuals
##########################################################################################

def test_residuals_are_target_minus_model() -> None:
    image = _image((10, 10))
    rng = np.random.default_rng(1)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.01, (10, 10))
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.residuals, target - s.model)


def test_residuals_zeroed_at_masked_pixels() -> None:
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = True
    s = Stretch([0, 0], image=image)
    s.set_target(target, mask=mask)
    s.fit()
    assert s.residuals[4, 4] == 0.0


def test_residuals_1d_excludes_masked() -> None:
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = mask[7, 2] = True
    s = Stretch([0, 0], image=image)
    s.set_target(target, mask=mask)
    s.fit()
    assert s.residuals_1d.shape == (98,)


def test_rms_matches_residuals_unweighted() -> None:
    image = _image((12, 12))
    rng = np.random.default_rng(7)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.05, (12, 12))
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    expected = np.sqrt(np.sum(s.residuals_1d**2) / (cast(Any, s.dof) - 1))
    assert s.rms == pytest.approx(expected)


##########################################################################################
# Uncertainties
##########################################################################################

def test_sigma_properties_are_positive() -> None:
    image = _image((20, 20))
    rng = np.random.default_rng(3)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.01, (20, 20))
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.all(s.m_sigma > 0)
    assert np.all(s.b_sigma > 0)
    assert np.all(s.s_sigma > 0)
    assert s.m_sigma.shape == (20, 20)


def test_covariance_diagonal_positive() -> None:
    image = _image((20, 20))
    rng = np.random.default_rng(4)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.02, (20, 20))
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.all(np.diag(cast(Any, s.covar)) > 0)


def test_lower_noise_gives_smaller_sigma() -> None:
    image = _image((20, 20))
    rng = np.random.default_rng(5)
    base = 3.0 + 2.0 * image
    s_lo = Stretch([0, 0], image=image)
    s_lo.set_target(base + rng.normal(0, 0.001, (20, 20)))
    s_lo.fit()
    s_hi = Stretch([0, 0], image=image)
    s_hi.set_target(base + rng.normal(0, 0.1, (20, 20)))
    s_hi.fit()
    assert np.mean(s_lo.b_sigma) < np.mean(s_hi.b_sigma)


def test_weighted_sigma_positive() -> None:
    image = _image((16, 16))
    rng = np.random.default_rng(6)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.01, (16, 16))
    weights = np.full((16, 16), 0.5)
    weights[0, 0] = 0.0
    s = Stretch([0, 0], image=image)
    s.set_target(target, weights=weights)
    s.fit()
    assert np.all(s.m_sigma > 0)
    assert cast(Any, s.rms) > 0


##########################################################################################
# Error paths
##########################################################################################

def test_set_target_without_image_raises() -> None:
    s = Stretch([0, 0])
    with pytest.raises(ValueError) as exc_info:
        s.set_target(np.ones((5, 5)))
    assert 'no image has been assigned' in str(exc_info.value)


def test_set_target_shape_mismatch_raises() -> None:
    s = Stretch([0, 0], image=np.ones((5, 5)))
    with pytest.raises(ValueError) as exc_info:
        s.set_target(np.ones((6, 6)))
    assert 'shape mismatch' in str(exc_info.value)


def test_model_without_coeffs_raises() -> None:
    s = Stretch([0, 0], image=np.ones((5, 5)))
    with pytest.raises(ValueError) as exc_info:
        _ = s.model
    assert 'no coefficients have been defined' in str(exc_info.value)


def test_eval_without_image_raises() -> None:
    s = Stretch([0, 0], coeffs=[1.0, 2.0])
    with pytest.raises(ValueError) as exc_info:
        _ = s.model
    assert 'has not yet been applied to an image' in str(exc_info.value)


def test_sigma_before_fit_raises() -> None:
    s = Stretch([0, 0], coeffs=[1.0, 2.0], image=np.ones((4, 4)))
    with pytest.raises(ValueError) as exc_info:
        _ = s.m_sigma
    assert 'has not yet been fitted' in str(exc_info.value)


def test_set_image_updates_shape() -> None:
    s = Stretch([0, 0])
    s.set_image(np.ones((6, 7)))
    assert s.shape == (6, 7)


def test_background_is_zero_for_scale_only() -> None:
    image = _image()
    target = 2.5 * image
    s = Stretch([-1, 0], image=image)
    s.set_target(target)
    s.fit()
    # orders[0] == -1 means rank 0, so the background evaluates to a scalar zero.
    assert s.background == 0.0


def test_b_sigma_is_zero_array_for_scale_only() -> None:
    image = _image()
    target = 2.5 * image
    s = Stretch([-1, 0], image=image)
    s.set_target(target)
    s.fit()
    b_sigma = s.b_sigma
    assert b_sigma.shape == image.shape
    assert np.all(b_sigma == 0.0)


def test_sigma_without_coeffs_raises() -> None:
    s = Stretch([0, 0], image=np.ones((4, 4)))
    # No coeffs assigned and no fit performed.
    with pytest.raises(ValueError) as exc_info:
        _ = s.m_sigma
    assert 'no coefficients have been defined' in str(exc_info.value)


def test_sigma_without_image_raises() -> None:
    s = Stretch([0, 0], coeffs=[1.0, 2.0])
    with pytest.raises(ValueError) as exc_info:
        _ = s.m_sigma
    assert 'has not yet been applied to an image' in str(exc_info.value)


def test_set_image_same_shape_keeps_grid() -> None:
    s = Stretch([1, 0], image=_image((8, 8), seed=20))
    grid_before = s._ij_powers
    s.set_image(_image((8, 8), seed=21))
    # Same shape: the (i, j) power grid object is reused, not rebuilt.
    assert s._ij_powers is grid_before
    assert s.shape == (8, 8)


def test_set_image_resize_rebuilds_grid() -> None:
    s = Stretch([1, 0], image=_image((6, 6), seed=10))
    image8 = _image((8, 8), seed=11)
    s.set_image(image8)
    assert s.shape == (8, 8)
    i, j = _ij((8, 8))
    bg = 1.0 + 0.5 * i + 0.3 * j
    target = bg + 2.0 * image8
    s.set_target(target)
    s.fit()
    assert np.allclose(s.model, target)


##########################################################################################
# Feature images stretched by a polynomial background and scaling (with noise)
##########################################################################################

def _poly_basis(i: np.ndarray, j: np.ndarray, order: int) -> list[np.ndarray]:
    """The 2-D polynomial basis terms for the given order, in Stretch's coefficient order.

    Stretch orders the terms as 1, i, j, i**2, i*j, j**2, i**3, i**2*j, ... — that is, by
    ascending total degree d, and within each degree as i**(d-k) * j**k for k in 0..d.
    """

    ni, nj = i.shape[0], j.shape[-1]
    basis = [np.ones((ni, nj))]
    for d in range(1, order + 1):
        for k in range(d + 1):
            basis.append(np.broadcast_to(i**(d - k) * j**k, (ni, nj)))
    return basis


def _eval_poly(coeffs: list[float], basis: list[np.ndarray]) -> np.ndarray:
    """Evaluate a 2-D polynomial given its coefficients and basis terms."""

    out = np.zeros(basis[0].shape)
    for c, term in zip(coeffs, basis, strict=True):
        out = out + c * term
    return out


def _feature_image(shape: tuple[int, int] = (500, 500)) -> np.ndarray:
    """A blank image containing a few filled circles and rectangles.

    The features are spread across the frame so that the (i, j) polynomial scaling, which
    is constrained only where the image is non-zero, is well determined everywhere.
    """

    img = np.zeros(shape)
    ii, jj = np.ogrid[:shape[0], :shape[1]]
    for ci, cj, r in [(120, 120, 40), (120, 380, 55), (380, 120, 50),
                      (250, 250, 60), (390, 390, 45)]:
        img[(ii - ci)**2 + (jj - cj)**2 <= r**2] = 1.0
    img[60:110, 200:320] = 1.0      # rectangle
    img[300:360, 380:470] = 1.0     # rectangle
    return img


# (background order, scaling order, background coeffs, scaling coeffs)
_POLY_CASES = [
    (0, 0, [5.0], [2.0]),
    (1, 1, [5.0, 1.0, -2.0], [2.0, 0.5, -0.3]),
    (2, 1, [4.0, 1.0, -2.0, 0.7, -0.4, 0.9], [2.0, 0.5, -0.3]),
    (2, 2, [4.0, 1.0, -2.0, 0.7, -0.4, 0.9], [2.0, 0.5, -0.3, 0.2, -0.1, 0.15]),
]


@pytest.mark.parametrize(('bg_order', 'sc_order', 'bg_coeffs', 'sc_coeffs'), _POLY_CASES)
def test_fit_recovers_polynomial_background_and_scaling(
    bg_order: int, sc_order: int, bg_coeffs: list[float], sc_coeffs: list[float],
) -> None:
    """Stretch.fit() recovers known background/scaling coefficients to good accuracy.

    The image is blank except for a few features. The target is that same image with a
    polynomial background added and a polynomial scaling applied (both with known
    coefficients), plus Gaussian noise. A Stretch of the matching orders should recover
    the coefficients.
    """

    shape = (500, 500)
    image = _feature_image(shape)
    i, j = _ij(shape)
    background = _eval_poly(bg_coeffs, _poly_basis(i, j, bg_order))
    scaling = _eval_poly(sc_coeffs, _poly_basis(i, j, sc_order))

    rng = np.random.default_rng(8421)
    target = background + scaling * image + rng.normal(0.0, 0.01, shape)

    s = Stretch([bg_order, sc_order], image=image)
    s.set_target(target)
    s.fit()

    expected = np.array(bg_coeffs + sc_coeffs)
    assert s.ncoeffs == len(expected)
    # Statistical recovery error for these inputs is well under 1.e-3; 5.e-3 leaves a wide
    # margin so the test is not flaky.
    assert np.allclose(cast(Any, s.coeffs), expected, atol=5.e-3)


def test_fit_reconstructs_background_scaling_and_model_arrays() -> None:
    """The fitted background, scaling, and model arrays reproduce the noiseless inputs."""

    shape = (500, 500)
    image = _feature_image(shape)
    i, j = _ij(shape)
    bg_coeffs = [3.0, 0.8, -1.5, 0.4, -0.2, 0.6]    # order 2
    sc_coeffs = [2.5, 0.4, -0.25]                   # order 1
    background = _eval_poly(bg_coeffs, _poly_basis(i, j, 2))
    scaling = _eval_poly(sc_coeffs, _poly_basis(i, j, 1))

    rng = np.random.default_rng(2718)
    clean = background + scaling * image
    target = clean + rng.normal(0.0, 0.01, shape)

    s = Stretch([2, 1], image=image)
    s.set_target(target)
    s.fit()

    assert np.allclose(cast(Any, s.coeffs), np.array(bg_coeffs + sc_coeffs), atol=5.e-3)
    assert np.allclose(s.background, background, atol=1.e-2)
    assert np.allclose(s.scaling, scaling, atol=1.e-2)
    assert np.abs(s.model - clean).max() < 2.e-2

##########################################################################################
