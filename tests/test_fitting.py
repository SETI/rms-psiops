##########################################################################################
# tests/test_fitting.py
##########################################################################################
"""Tests for psiops.fitting.

The Fitting class orchestrates an ImageModel and a Stretch. These tests drive the
real Fitting logic using lightweight stand-ins for those two collaborators, which
mirror the public interfaces that Fitting relies on. The model is a smooth Gaussian
blob whose center is recoverable by least squares, and the stretch performs an
identity comparison so that the residuals equal target minus model.
"""

import numpy as np
import pytest

from psiops.fitting import Fitting
from psiops.imagemodel import ImageModel
from psiops.imagemodel.gaussian import Gaussian
from psiops.imagemodel.smearedmodel import SmearedModel
from psiops.stretch import Stretch

##########################################################################################
# Test doubles mirroring the ImageModel and Stretch public interfaces
##########################################################################################

class _BlobModel(ImageModel):
    """A Gaussian blob whose center is the (x, y) transform parameters."""

    def __init__(self, sigma: float = 1.5) -> None:
        self.sigma = sigma

    def transform(
        self,
        shape: tuple[int, int],
        center: tuple[float, float],
        expand: float = 1.,
        rotate: float = 0.,
    ) -> np.ndarray:
        i = np.arange(shape[0])[:, np.newaxis]
        j = np.arange(shape[1])[np.newaxis, :]
        s = self.sigma * expand
        di = i - center[0]
        dj = j - center[1]
        return np.exp(-(di**2 + dj**2) / (2. * s * s))


class _IdentityStretch:
    """A minimal Stretch stand-in whose model is exactly the assigned image."""

    ncoeffs = 1

    def set_target(
        self,
        target: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        maskval: float | None = None,
        weights: np.ndarray | None = None,
        nans: bool = False,
    ) -> None:
        self.target = np.asarray(target, dtype=float)
        self.mask = mask
        self.weights = weights

    def set_image(self, image: np.ndarray) -> None:
        self.image = np.asarray(image, dtype=float)

    def fit(self) -> None:
        self._diff = self.target - self.image

    @property
    def residuals_1d(self) -> np.ndarray:
        if self.mask is None:
            return self._diff.ravel()
        return self._diff[np.logical_not(self.mask)]

    @property
    def model(self) -> np.ndarray:
        return self.image

    @property
    def residuals(self) -> np.ndarray:
        return self._diff

    @property
    def background(self) -> np.ndarray:
        return np.zeros_like(self.image)

    @property
    def scaling(self) -> np.ndarray:
        return np.ones_like(self.image)

    @property
    def m_sigma(self) -> np.ndarray:
        return np.full_like(self.image, 0.1)

    @property
    def b_sigma(self) -> np.ndarray:
        return np.full_like(self.image, 0.2)

    @property
    def s_sigma(self) -> np.ndarray:
        return np.full_like(self.image, 0.3)


def _make_fitting(sigma: float = 1.5) -> Fitting:
    return Fitting(_BlobModel(sigma), _IdentityStretch())


##########################################################################################
# __init__
##########################################################################################

def test_init_stores_collaborators() -> None:
    model = _BlobModel()
    stretch = _IdentityStretch()
    f = Fitting(model, stretch)
    assert f.imagemodel is model
    assert f.stretch is stretch


##########################################################################################
# set_target
##########################################################################################

def test_set_target_no_weights() -> None:
    f = _make_fitting()
    target = np.arange(25, dtype=float).reshape(5, 5)
    f.set_target(target)
    assert f.target.shape == (5, 5)
    assert f.mask is None
    assert f.weights is None
    assert np.array_equal(f.target, target)
    assert f.corner == (0, 0)
    assert f.shape == (5, 5)


def test_set_target_stores_full_arrays() -> None:
    f = _make_fitting()
    target = np.ones((6, 6))
    f.set_target(target)
    assert np.array_equal(f._full_target, target)
    assert f._full_shape == (6, 6)


def test_set_target_with_mask_is_honored() -> None:
    f = _make_fitting()
    target = np.ones((5, 5))
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    f.set_target(target, mask=mask)
    assert f.mask is not None
    assert f.mask[0, 0]
    assert f.mask.sum() == 1


def test_set_target_with_weights_normalizes() -> None:
    f = _make_fitting()
    target = np.ones((5, 5))
    weights = np.full((5, 5), 2.0)
    weights[1, 1] = 4.0
    f.set_target(target, weights=weights)
    # Weights are normalized so that the maximum is 1.
    assert f.weights is not None
    assert f.weights.max() == pytest.approx(1.0)
    assert f.weights[1, 1] == pytest.approx(1.0)


def test_set_target_weights_does_not_mutate_input() -> None:
    f = _make_fitting()
    weights = np.full((5, 5), 4.0)
    original = weights.copy()
    f.set_target(np.ones((5, 5)), weights=weights)
    assert np.array_equal(weights, original)


def test_set_target_maskval() -> None:
    f = _make_fitting()
    target = np.ones((5, 5))
    target[2, 3] = -99.0
    f.set_target(target, maskval=-99.0)
    assert f.mask is not None
    assert f.mask[2, 3]


def test_set_target_slice() -> None:
    f = _make_fitting()
    target = np.arange(64, dtype=float).reshape(8, 8)
    f.set_target(target, corner=(2, 3), shape=(4, 5))
    assert f.corner == (2, 3)
    assert f.shape == (4, 5)
    assert f.target.shape == (4, 5)
    assert np.array_equal(f.target, target[2:6, 3:8])


def test_set_target_slice_default_shape() -> None:
    f = _make_fitting()
    target = np.arange(64, dtype=float).reshape(8, 8)
    f.set_target(target, corner=(2, 2))
    # With no shape, the slice extends to the array edges.
    assert f.shape == (6, 6)
    assert f.target.shape == (6, 6)


def test_set_target_passes_target_to_stretch() -> None:
    f = _make_fitting()
    target = np.arange(25, dtype=float).reshape(5, 5)
    f.set_target(target)
    assert np.array_equal(f.stretch.target, f.target)


##########################################################################################
# remask
##########################################################################################

def test_remask_wrong_shape_raises() -> None:
    f = _make_fitting()
    f.set_target(np.ones((5, 5)))
    with pytest.raises(ValueError, match='new mask shape'):
        f.remask(np.zeros((4, 4), dtype=bool))


def test_remask_when_no_existing_mask() -> None:
    f = _make_fitting()
    f.set_target(np.ones((5, 5)))
    new = np.zeros((5, 5), dtype=bool)
    new[1, 1] = True
    f.remask(new)
    assert f.mask is new


def test_remask_overlays_existing_mask() -> None:
    f = _make_fitting()
    target = np.ones((5, 5))
    base = np.zeros((5, 5), dtype=bool)
    base[0, 0] = True
    f.set_target(target, mask=base)
    new = np.zeros((5, 5), dtype=bool)
    new[4, 4] = True
    f.remask(new)
    # Both the original and the new masked pixels remain masked.
    assert f.mask is not None
    assert f.mask[0, 0]
    assert f.mask[4, 4]
    assert f.mask.sum() == 2


##########################################################################################
# fit -- correctness
##########################################################################################

def test_fit_recovers_center() -> None:
    f = _make_fitting()
    true_center = (5.3, 6.1)
    target = f.imagemodel.transform((12, 12), center=true_center)
    f.set_target(target)
    f.fit([5.0, 6.0, 1.0, 0.0])
    assert f.x == pytest.approx(true_center[0], abs=1e-4)
    assert f.y == pytest.approx(true_center[1], abs=1e-4)


def test_fit_sets_param_aliases() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(4.5, 5.5))
    f.set_target(target)
    f.fit([4.0, 5.0, 1.0, 0.0])
    assert (f.x, f.y, f.zoom, f.rotate) == tuple(f.params)
    assert f.params.shape == (4,)


def test_fit_perfect_model_zero_rms() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((12, 12), center=(5.3, 6.1))
    f.set_target(target)
    f.fit([5.0, 6.0, 1.0, 0.0])
    assert f.rms == pytest.approx(0.0, abs=1e-6)


def test_fit_noisy_positive_rms_and_uncertainty() -> None:
    f = _make_fitting()
    rng = np.random.default_rng(0)
    target = (f.imagemodel.transform((12, 12), center=(5.3, 6.1))
              + 0.01 * rng.standard_normal((12, 12)))
    f.set_target(target)
    f.fit([5.0, 6.0, 1.0, 0.0])
    assert f.rms > 0.0
    assert f.dx > 0.0
    assert f.dy > 0.0
    assert f.covar.shape == (4, 4)


def test_fit_dof_no_weights() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 5.0))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0])
    # 100 pixels, 2 fitted params, 1 stretch coefficient.
    assert f.weight_sum == 100
    assert f.dof == 100 - 2 - 1


def test_fit_flags_count_nparams() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 5.0))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0], flags=(True, False, False, False))
    assert f.nparams == 1


def test_fit_unfitted_param_unchanged() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 4.0))
    f.set_target(target)
    f.fit([5.0, 8.0, 1.0, 0.0], flags=(False, True, False, False))
    # x is not fitted, so it stays at its guess.
    assert f.x == pytest.approx(5.0)
    assert f.y == pytest.approx(4.0, abs=1e-4)


def test_fit_without_limits_uses_identity_branch() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.3, 4.7))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0], limits=(0., 0., 0., 0.))
    assert f.x == pytest.approx(5.3, abs=1e-4)
    assert f.y == pytest.approx(4.7, abs=1e-4)


def test_fit_with_limits_uses_sin_branch() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.3, 4.7))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0], limits=(2., 2., 0.1, 0.2))
    assert f.x == pytest.approx(5.3, abs=1e-3)
    assert f.y == pytest.approx(4.7, abs=1e-3)


def test_fit_lsq_dict_passthrough() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 5.0))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0], lsq_dict={'method': 'lm'})
    assert f.lsq_dict == {'method': 'lm'}


def test_fit_weighted_branch() -> None:
    f = _make_fitting()
    rng = np.random.default_rng(1)
    target = (f.imagemodel.transform((12, 12), center=(5.3, 6.1))
              + 0.01 * rng.standard_normal((12, 12)))
    weights = np.ones((12, 12))
    weights[0, 0] = 0.5
    f.set_target(target, weights=weights)
    f.fit([5.0, 6.0, 1.0, 0.0])
    assert f.x == pytest.approx(5.3, abs=2e-2)
    assert f.weight_sum > 0.0
    assert f.rms > 0.0


def test_fit_masked_branch_reduces_dof() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 5.0))
    mask = np.zeros((10, 10), dtype=bool)
    mask[0, 0] = True
    f.set_target(target, mask=mask)
    f.fit([5.0, 5.0, 1.0, 0.0])
    # One pixel masked out of 100.
    assert f.dof == 99 - 2 - 1


def test_fit_transformed_attribute_set() -> None:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 5.0))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0])
    assert f.transformed.shape == (10, 10)
    assert np.allclose(f.transformed, target, atol=1e-4)


def test_fill_stats_recomputes_when_x_changed() -> None:
    # _fill_stats re-evaluates _func when the result's x differs from the last
    # cached evaluation. Drive a normal fit, then re-fill with a perturbed result.
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 5.0))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0])

    class _Result:
        x: np.ndarray
        cost: float
        jac: np.ndarray

    result = _Result()
    result.x = f._result.x + 0.01           # differs from cached self._x
    result.cost = f._result.cost
    result.jac = f._result.jac
    f._fill_stats(result)
    # After recompute, the cached x matches the new result.
    assert np.all(f._x == result.x)  # type: ignore[attr-defined]


##########################################################################################
# Property delegation to the stretch
##########################################################################################

def _fitted() -> Fitting:
    f = _make_fitting()
    target = f.imagemodel.transform((10, 10), center=(5.0, 5.0))
    f.set_target(target)
    f.fit([5.0, 5.0, 1.0, 0.0])
    return f


def test_model_property() -> None:
    f = _fitted()
    assert np.array_equal(f.model, f.stretch.model)


def test_background_property() -> None:
    f = _fitted()
    assert np.array_equal(f.background, f.stretch.background)


def test_scaling_property() -> None:
    f = _fitted()
    assert np.array_equal(f.scaling, f.stretch.scaling)


def test_residuals_property() -> None:
    f = _fitted()
    assert np.array_equal(f.residuals, f.stretch.residuals)


def test_m_sigma_property() -> None:
    f = _fitted()
    assert np.array_equal(f.m_sigma, f.stretch.m_sigma)


def test_b_sigma_property() -> None:
    f = _fitted()
    assert np.array_equal(f.b_sigma, f.stretch.b_sigma)


def test_s_sigma_property() -> None:
    f = _fitted()
    assert np.array_equal(f.s_sigma, f.stretch.s_sigma)


##########################################################################################
# End-to-end fits with the real Gaussian ImageModel and a real Stretch
#
# Each target is a 50x50 image holding a 2-D Gaussian (centered within 6 pixels of the
# middle), plus a second-order polynomial background, plus noise. A Fitting using the same
# Gaussian as the model and a Stretch of orders (2, 0) should recover the Gaussian center
# to a small fraction of a pixel, and the transformed model should conserve the Gaussian's
# integral.
##########################################################################################

_FIT_SHAPE = (50, 50)
_FIT_INTEGRAL = 10000.0      # bright enough that the source dominates the background


def _background(shape: tuple[int, int]) -> np.ndarray:
    """A second-order polynomial background on the (i, j) grid normalized to [-1, 1]."""

    ni, nj = shape
    i = (np.arange(ni) - 0.5 * (ni - 1))[:, np.newaxis] / (0.5 * (ni - 1))
    j = (np.arange(nj) - 0.5 * (nj - 1))[np.newaxis, :] / (0.5 * (nj - 1))
    return 5.0 + 1.0 * i - 2.0 * j + 0.7 * i**2 - 0.4 * i * j + 0.9 * j**2


def _peak_guess(target: np.ndarray) -> tuple[float, float]:
    """A rough source-center guess: the peak of the background-subtracted target.

    Returns pixel-center coordinates (index + 0.5), which is how a Fitting would be seeded
    in practice (detect the source, then refine).
    """

    resid = target - np.median(target)
    pi, pj = np.unravel_index(int(np.argmax(resid)), target.shape)
    return (pi + 0.5, pj + 0.5)


# (sigma, x0, y0, seed); every center is within 6 pixels of the middle (25, 25).
_GAUSSIAN_CASES = [
    (2.0, 31.0, 19.0, 0),
    (2.5, 19.5, 30.5, 1),
    (3.0, 21.0, 29.0, 2),
    (4.0, 30.5, 19.5, 3),
    (5.0, 22.0, 28.0, 4),
    (6.0, 28.0, 24.0, 5),
]


# Both the bounded default and an unbounded (no-limit) parameterization should converge
# from a good initial guess.
_FIT_LIMITS = [(10., 10., 0.1, 0.2), (0., 0., 0., 0.)]


@pytest.mark.parametrize('limits', _FIT_LIMITS, ids=['bounded', 'unbounded'])
@pytest.mark.parametrize(('sigma', 'x0', 'y0', 'seed'), _GAUSSIAN_CASES)
def test_fit_recovers_gaussian_center_and_integral(
    sigma: float, x0: float, y0: float, seed: int,
    limits: tuple[float, float, float, float],
) -> None:
    """Fitting recovers the true Gaussian center and conserves its integral.

    Run with both the bounded default `limits` and an unbounded `(0, 0, 0, 0)`; with a
    good initial guess the optimizer stays well inside the bounds, so both converge.
    """

    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(seed)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    fitting.fit(guesses=(*_peak_guess(target), 1.0, 0.0), limits=limits)

    # Center recovered to a small fraction of a pixel (observed worst case ~0.05 px).
    assert abs(fitting.x - x0) < 0.2
    assert abs(fitting.y - y0) < 0.2

    # The transformed model (no stretch) conserves the Gaussian's integral; the only loss
    # is the small tail that falls outside the 50x50 frame.
    assert abs(fitting.transformed.sum() - _FIT_INTEGRAL) < 2.e-3 * _FIT_INTEGRAL


def test_fit_reconstructs_background_and_model() -> None:
    """The fitted background and model also reproduce the noiseless inputs."""

    sigma, x0, y0 = 3.5, 27.0, 22.0
    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    background = _background(_FIT_SHAPE)
    rng = np.random.default_rng(99)
    target = gaussian + background + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    fitting.fit(guesses=(*_peak_guess(target), 1.0, 0.0))

    assert abs(fitting.x - x0) < 0.2
    assert abs(fitting.y - y0) < 0.2
    # The recovered background matches the injected polynomial away from the noise.
    assert np.abs(fitting.background - background).max() < 0.5
    # The model reproduces the noiseless target (Gaussian + background) closely.
    assert np.abs(fitting.model - (gaussian + background)).max() < 1.0


def test_median_abs_deviation_matches_unmasked_median() -> None:
    """With no mask, the property equals the median of |target - model|."""

    sigma, x0, y0 = 3.0, 26.0, 24.0
    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(5)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    fitting.fit(guesses=(*_peak_guess(target), 1.0, 0.0))

    expected = np.median(np.abs(fitting.target - fitting.model))
    assert fitting.median_abs_deviation == pytest.approx(expected)
    # Residuals are essentially the unit-sigma noise: median |N(0,1)| ~ 0.6745.
    assert 0.5 < fitting.median_abs_deviation < 0.9


def test_median_abs_deviation_skips_masked_pixels() -> None:
    """Heavily corrupted but masked pixels do not affect the property."""

    sigma, x0, y0 = 3.0, 26.0, 24.0
    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(6)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    # Corrupt a block far from the source, then mask exactly those pixels.
    mask = np.zeros(_FIT_SHAPE, dtype=bool)
    mask[40:50, 40:50] = True
    target[mask] += 1000.0

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target, mask=mask)
    # Seed at the frame center (the source is ~1.4 px away); a peak detector would lock
    # onto the +1000 corrupted block instead.
    fitting.fit(guesses=(25.0, 25.0, 1.0, 0.0))

    diff = np.abs(fitting.target - fitting.model)
    # The property equals the median over the unmasked pixels only...
    assert fitting.median_abs_deviation == pytest.approx(np.median(diff[~mask]))
    # ...the masked block really is heavily corrupted (would dominate if included)...
    assert np.median(diff[mask]) > 100.0
    # ...yet the property stays at the noise level, proving it skips the masked pixels.
    assert fitting.median_abs_deviation < 1.0


def test_median_abs_deviation_is_cached_per_fit() -> None:
    """The deviation is evaluated only once per fit and cached, and a re-fit resets it."""

    sigma, x0, y0 = 3.0, 26.0, 24.0
    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(7)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    fitting.fit(guesses=(*_peak_guess(target), 1.0, 0.0))

    # Not evaluated until first access, then cached.
    assert fitting._median_abs_deviation is None
    first = fitting.median_abs_deviation
    assert fitting._median_abs_deviation == first

    # A later change to the inputs is ignored because the value is evaluated only once.
    fitting.target = fitting.target + 1000.0
    assert fitting.median_abs_deviation == first

    # The next fit invalidates the cache.
    fitting.set_target(target)
    fitting.fit(guesses=(*_peak_guess(target), 1.0, 0.0))
    assert fitting._median_abs_deviation is None


##########################################################################################
# fit() grid search over multiple initial guesses
##########################################################################################

def test_fit_grid_of_guesses_recovers_far_source() -> None:
    """A grid of center guesses finds a narrow source a single center guess would miss."""

    sigma, x0, y0 = 2.0, 31.0, 19.0
    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(0)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    # A single guess at the middle stalls (sigma 2, ~8 px away); the grid includes a
    # starting point near the true center, so the best-residual combination converges.
    fitting.fit(guesses=([19., 25., 31.], [19., 25., 31.], 1.0, 0.0))

    assert abs(fitting.x - x0) < 0.2
    assert abs(fitting.y - y0) < 0.2
    assert fitting.rms < 1.2


def test_fit_grid_searches_a_fixed_parameter() -> None:
    """Guess values are tried for a parameter even when its flag is False."""

    sigma, x0, y0 = 3.0, 26.0, 24.0
    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(1)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    # zoom is held fixed (flag False) but offered three values. The grid must try each
    # and keep zoom == 1.0, which matches the unexpanded model and gives the best fit.
    fitting.fit(guesses=(25.0, 25.0, [0.5, 1.0, 2.0], 0.0),
                flags=(True, True, False, False))

    assert fitting.zoom == 1.0
    assert abs(fitting.x - x0) < 0.2
    assert abs(fitting.y - y0) < 0.2


@pytest.mark.parametrize(('sigma', 'x0', 'y0', 'seed'), _GAUSSIAN_CASES)
def test_central_grid_guess_recovers_peak(
    sigma: float, x0: float, y0: float, seed: int,
) -> None:
    """A fixed 3x3 central grid of guesses recovers the source without peak detection.

    In place of the peak-seeded guess, this uses guesses=((20, 25, 30), (20, 25, 30), 1,
    0): a 3x3 grid of starting centers spanning the middle of the frame, with zoom and
    rotate held fixed. Every center within 6 px of the middle is then within reach of at
    least one grid start, so the best-residual combination converges. (The integer guess
    values exercise the float-cast in `_fit1`, without which the fractional fit updates
    would be truncated.)
    """

    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(seed)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    fitting.fit(guesses=((20, 25, 30), (20, 25, 30), 1, 0))

    assert abs(fitting.x - x0) < 0.2
    assert abs(fitting.y - y0) < 0.2
    assert abs(fitting.transformed.sum() - _FIT_INTEGRAL) < 2.e-3 * _FIT_INTEGRAL


@pytest.mark.parametrize('zoom', [0.0, -1.0, -3.5])
def test_fit_rejects_nonpositive_zoom(zoom: float) -> None:
    """A zoom guess <= 0 is invalid (it collapses or inverts the model) and raises."""

    sigma, x0, y0 = 3.0, 26.0, 24.0
    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(3)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    with pytest.raises(ValueError, match='zoom must be positive'):
        fitting.fit(guesses=(25.0, 25.0, zoom, 0.0))


@pytest.mark.parametrize(('sigma', 'x0', 'y0', 'seed'), _GAUSSIAN_CASES)
def test_fit_recovers_zoom_near_one(
    sigma: float, x0: float, y0: float, seed: int,
) -> None:
    """Fitting the zoom as well recovers a value very close to 1.

    The model has the same sigma as the target, so the true zoom is 1.0. Using the
    peak-seeded x, y guess plus flags=(True, True, True, False) and a zoom guess of 1.1
    (rotate held fixed), the fit drives zoom back toward 1. Relies on the default
    limits=(10, 10, 0.4, 1.), which allow zoom to move +/-0.4 from the guess.
    """

    model = Gaussian(sigma=sigma, integral=_FIT_INTEGRAL)
    gaussian = model.transform(_FIT_SHAPE, center=(x0, y0))
    rng = np.random.default_rng(seed)
    target = gaussian + _background(_FIT_SHAPE) + rng.normal(0.0, 1.0, _FIT_SHAPE)

    fitting = Fitting(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), Stretch([2, 0]))
    fitting.set_target(target)
    fitting.fit(guesses=(*_peak_guess(target), 1.1, 0.0), flags=(True, True, True, False))

    assert abs(fitting.x - x0) < 0.2
    assert abs(fitting.y - y0) < 0.2
    # The fitted zoom returns to ~1; worst observed |zoom - 1| across these cases ~0.003.
    assert abs(fitting.zoom - 1.0) < 0.02


@pytest.mark.parametrize(('sigma', 'x0', 'y0', 'seed'), _GAUSSIAN_CASES)
def test_fit_recovers_smear_rotation(
    sigma: float, x0: float, y0: float, seed: int,
) -> None:
    """With all four parameters free, fit() recovers the rotation aligning the smears.

    The target is a Gaussian smeared along (3, 4); the model is smeared along (0, 5). Both
    smears have length 5, so a pure rotation (with zoom ~ 1) maps the model smear onto the
    target's. With flags=(True, True, True, True) and guesses zoom=1, rotate=0, fit()
    recovers rotate = atan2(-3, 4) ~ -0.6435 rad.
    """

    smeared = SmearedModel(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), smear=(3, 4))
    target = (smeared.transform(_FIT_SHAPE, center=(x0, y0)) + _background(_FIT_SHAPE)
              + np.random.default_rng(seed).normal(0.0, 1.0, _FIT_SHAPE))

    model = SmearedModel(Gaussian(sigma=sigma, integral=_FIT_INTEGRAL), smear=(0, 5))
    fitting = Fitting(model, Stretch([2, 0]))
    fitting.set_target(target)
    fitting.fit(guesses=(*_peak_guess(target), 1.0, 0.0), flags=(True, True, True, True))

    expected_rotate = np.arctan2(-3.0, 4.0)
    assert abs(fitting.x - x0) < 0.2
    assert abs(fitting.y - y0) < 0.2
    assert abs(fitting.zoom - 1.0) < 0.05
    # Rotation recovered to within ~0.08 rad worst case (broad sigma=6); much tighter for
    # narrower Gaussians, where the 5-px smear is better resolved.
    assert abs(fitting.rotate - expected_rotate) < 0.15
    assert fitting.rms < 1.2

##########################################################################################
