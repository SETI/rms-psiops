##########################################################################################
# tests/test_stretch.py
##########################################################################################
"""Tests for psiops.stretch."""

import numpy as np
import pytest

from psiops.stretch import Stretch


def _ij(shape):
    """Return normalized (i, j) coordinate arrays matching Stretch's internal convention.

    Stretch maps each axis onto [-1, 1] via (index - half) / half.
    """

    ni, nj = shape
    half_i = 0.5 * (ni - 1)
    half_j = 0.5 * (nj - 1)
    i = (np.arange(ni) - half_i)[:, np.newaxis] / half_i
    j = (np.arange(nj) - half_j) / half_j
    return i, j


def _image(shape=(8, 8), seed=0):
    rng = np.random.default_rng(seed)
    return rng.random(shape)


##########################################################################################
# Construction and rank bookkeeping
##########################################################################################

def test_rank_from_order_sequence():
    expected = [0, 1, 3, 6, 10, 15]
    got = [Stretch._rank_from_order(o) for o in [-1, 0, 1, 2, 3, 4]]
    assert got == expected


def test_ranks_and_ncoeffs_background_scale():
    s = Stretch([2, 0])
    assert s.ranks == (6, 1)
    assert s.ncoeffs == 7
    assert s.orders == (2, 0)


def test_ranks_scale_only():
    s = Stretch([-1, 0])
    assert s.ranks == (0, 1)
    assert s.ncoeffs == 1


def test_set_coeffs_valid():
    s = Stretch([0, 0])
    s.set_coeffs([3.0, 2.0])
    assert np.array_equal(s.coeffs, [3.0, 2.0])


def test_set_coeffs_wrong_count_raises():
    s = Stretch([0, 0])
    with pytest.raises(ValueError):
        s.set_coeffs([1.0])


##########################################################################################
# Basic fitting: constant background + constant scale
##########################################################################################

def test_fit_recovers_offset_and_scale():
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.coeffs, [3.0, 2.0])


def test_fit_model_matches_target_exactly():
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.model, target)


def test_background_and_scaling_properties():
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.background, 3.0)
    assert np.allclose(s.scaling, 2.0)


def test_scale_only_fit():
    image = _image()
    target = 2.5 * image
    s = Stretch([-1, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.coeffs, [2.5])
    assert np.allclose(s.model, target)


def test_perfect_fit_has_negligible_rms():
    image = _image()
    target = 3.0 + 2.0 * image
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert s.rms < 1e-10
    assert s.chi_sq < 1e-20


def test_dof_and_weight_sum_unweighted():
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

def test_order1_background_recovered():
    shape = (12, 12)
    image = _image(shape)
    i, j = _ij(shape)
    bg = 1.0 + 0.5 * i + 0.3 * j
    target = bg + 2.0 * image
    s = Stretch([1, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.coeffs, [1.0, 0.5, 0.3, 2.0])
    assert np.allclose(s.background, bg)
    assert np.allclose(s.scaling, 2.0)


def test_order2_background_recovered():
    shape = (14, 14)
    image = _image(shape)
    i, j = _ij(shape)
    bg = 1.0 + 0.5 * i + 0.3 * j + 0.2 * i**2 + 0.1 * i * j + 0.4 * j**2
    target = bg + 2.0 * image
    s = Stretch([2, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.coeffs, [1.0, 0.5, 0.3, 0.2, 0.1, 0.4, 2.0])
    assert np.allclose(s.background, bg)


def test_spatially_varying_scale_recovered():
    shape = (12, 12)
    image = _image(shape)
    i, j = _ij(shape)
    scale = 2.0 + 0.5 * i + 0.3 * j
    target = 1.0 + scale * image
    s = Stretch([0, 1], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.coeffs, [1.0, 2.0, 0.5, 0.3])
    assert np.allclose(s.scaling, scale)
    assert np.allclose(s.model, target)


def test_quadratic_image_term_recovered():
    image = _image((10, 10))
    target = 1.0 + 2.0 * image + 0.5 * image**2
    s = Stretch([0, 0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.coeffs, [1.0, 2.0, 0.5])
    assert np.allclose(s.model, target)


##########################################################################################
# Masks and weights
##########################################################################################

def test_target_mask_excludes_pixels():
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = mask[5, 5] = True
    s = Stretch([0, 0], image=image)
    s.set_target(target, mask=mask)
    s.fit()
    assert s.mask is not None
    assert s._antimask.sum() == 98
    assert np.allclose(s.coeffs, [3.0, 2.0])


def test_image_mask_excludes_pixels():
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    imask = np.zeros((10, 10), dtype=bool)
    imask[1, 1] = True
    s = Stretch([0, 0], image=image, mask=imask)
    s.set_target(target)
    s.fit()
    assert s._antimask.sum() == 99
    assert np.allclose(s.coeffs, [3.0, 2.0])


def test_combined_image_and_target_masks():
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    imask = np.zeros((10, 10), dtype=bool)
    imask[1, 1] = True
    tmask = np.zeros((10, 10), dtype=bool)
    tmask[2, 2] = True
    s = Stretch([0, 0], image=image, mask=imask)
    s.set_target(target, mask=tmask)
    s.fit()
    assert s._antimask.sum() == 98
    assert s.mask[1, 1]
    assert s.mask[2, 2]


def test_weighted_fit_recovers_coeffs():
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    weights = np.ones((10, 10))
    weights[0, 0] = 0.0
    s = Stretch([0, 0], image=image)
    s.set_target(target, weights=weights)
    s.fit()
    assert np.allclose(s.coeffs, [3.0, 2.0])
    assert s.target_weights is not None


def test_weighted_fit_normalizes_weights():
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    weights = np.full((10, 10), 4.0)
    s = Stretch([0, 0], image=image)
    s.set_target(target, weights=weights)
    assert np.max(s.target_weights) == pytest.approx(1.0)


##########################################################################################
# Residuals
##########################################################################################

def test_residuals_are_target_minus_model():
    image = _image((10, 10))
    rng = np.random.default_rng(1)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.01, (10, 10))
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.allclose(s.residuals, target - s.model)


def test_residuals_zeroed_at_masked_pixels():
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = True
    s = Stretch([0, 0], image=image)
    s.set_target(target, mask=mask)
    s.fit()
    assert s.residuals[4, 4] == 0.0


def test_residuals_1d_excludes_masked():
    image = _image((10, 10))
    target = 3.0 + 2.0 * image
    mask = np.zeros((10, 10), dtype=bool)
    mask[4, 4] = mask[7, 2] = True
    s = Stretch([0, 0], image=image)
    s.set_target(target, mask=mask)
    s.fit()
    assert s.residuals_1d.shape == (98,)


def test_rms_matches_residuals_unweighted():
    image = _image((12, 12))
    rng = np.random.default_rng(7)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.05, (12, 12))
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    expected = np.sqrt(np.sum(s.residuals_1d**2) / (s.dof - 1))
    assert s.rms == pytest.approx(expected)


##########################################################################################
# Uncertainties
##########################################################################################

def test_sigma_properties_are_positive():
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


def test_covariance_diagonal_positive():
    image = _image((20, 20))
    rng = np.random.default_rng(4)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.02, (20, 20))
    s = Stretch([0, 0], image=image)
    s.set_target(target)
    s.fit()
    assert np.all(np.diag(s.covar) > 0)


def test_lower_noise_gives_smaller_sigma():
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


def test_weighted_sigma_positive():
    image = _image((16, 16))
    rng = np.random.default_rng(6)
    target = 3.0 + 2.0 * image + rng.normal(0, 0.01, (16, 16))
    weights = np.full((16, 16), 0.5)
    weights[0, 0] = 0.0
    s = Stretch([0, 0], image=image)
    s.set_target(target, weights=weights)
    s.fit()
    assert np.all(s.m_sigma > 0)
    assert s.rms > 0


##########################################################################################
# Error paths
##########################################################################################

def test_set_target_without_image_raises():
    s = Stretch([0, 0])
    with pytest.raises(ValueError):
        s.set_target(np.ones((5, 5)))


def test_set_target_shape_mismatch_raises():
    s = Stretch([0, 0], image=np.ones((5, 5)))
    with pytest.raises(ValueError):
        s.set_target(np.ones((6, 6)))


def test_model_without_coeffs_raises():
    s = Stretch([0, 0], image=np.ones((5, 5)))
    with pytest.raises(ValueError):
        _ = s.model


def test_eval_without_image_raises():
    s = Stretch([0, 0], coeffs=[1.0, 2.0])
    with pytest.raises(ValueError):
        _ = s.model


def test_sigma_before_fit_raises():
    s = Stretch([0, 0], coeffs=[1.0, 2.0], image=np.ones((4, 4)))
    with pytest.raises(ValueError):
        _ = s.m_sigma


def test_set_image_updates_shape():
    s = Stretch([0, 0])
    s.set_image(np.ones((6, 7)))
    assert s.shape == (6, 7)


def test_background_is_zero_for_scale_only():
    image = _image()
    target = 2.5 * image
    s = Stretch([-1, 0], image=image)
    s.set_target(target)
    s.fit()
    # orders[0] == -1 means rank 0, so the background evaluates to a scalar zero.
    assert s.background == 0.0


def test_b_sigma_is_zero_array_for_scale_only():
    image = _image()
    target = 2.5 * image
    s = Stretch([-1, 0], image=image)
    s.set_target(target)
    s.fit()
    b_sigma = s.b_sigma
    assert b_sigma.shape == image.shape
    assert np.all(b_sigma == 0.0)


def test_sigma_without_coeffs_raises():
    s = Stretch([0, 0], image=np.ones((4, 4)))
    # No coeffs assigned and no fit performed.
    with pytest.raises(ValueError):
        _ = s.m_sigma


def test_sigma_without_image_raises():
    s = Stretch([0, 0], coeffs=[1.0, 2.0])
    with pytest.raises(ValueError):
        _ = s.m_sigma


def test_set_image_same_shape_keeps_grid():
    s = Stretch([1, 0], image=_image((8, 8), seed=20))
    grid_before = s._ij_powers
    s.set_image(_image((8, 8), seed=21))
    # Same shape: the (i, j) power grid object is reused, not rebuilt.
    assert s._ij_powers is grid_before
    assert s.shape == (8, 8)


def test_set_image_resize_rebuilds_grid():
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
