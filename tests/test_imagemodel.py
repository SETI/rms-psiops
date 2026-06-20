##########################################################################################
# tests/test_imagemodel.py
##########################################################################################
"""Tests for the psiops.imagemodel package: ImageModel, ArrayModel, Gaussian,
SummedModel, and SmearedModel.
"""

import numpy as np
import pytest

from psiops._filter import _use_shortcuts
from psiops.imagemodel import ImageModel
from psiops.imagemodel.arraymodel import ArrayModel
from psiops.imagemodel.gaussian import Gaussian
from psiops.imagemodel.smearedmodel import SmearedModel
from psiops.imagemodel.summedmodel import SummedModel


@pytest.fixture
def no_shortcuts():
    """Disable filter/resample shortcut paths for resample-based models.

    The optimized resample shortcut path has a latent defect when no weights are
    supplied (out of test scope), so models that rely on resample()/rotate() are
    exercised through the general code path. The autouse fixture in conftest.py
    restores the global flag afterward.
    """

    _use_shortcuts(False)


##########################################################################################
# ImageModel (abstract base)
##########################################################################################

def test_imagemodel_transform_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        ImageModel().transform((3, 3), (1.5, 1.5))


def test_imagemodel_error_names_subclass() -> None:
    class MyModel(ImageModel):
        pass

    with pytest.raises(NotImplementedError, match='MyModel'):
        MyModel().transform((3, 3), (1.5, 1.5))


##########################################################################################
# Gaussian
##########################################################################################

def test_gaussian_transform_shape() -> None:
    result = Gaussian().transform((7, 9), (3.5, 4.5))
    assert result.shape == (7, 9)


def test_gaussian_preserves_integral() -> None:
    # A wide enough grid captures essentially the entire unit-volume Gaussian.
    g = Gaussian(sigma=2.0, integral=7.0)
    result = g.transform((41, 41), (20.5, 20.5))
    assert result.sum() == pytest.approx(7.0, abs=1e-3)


def test_gaussian_default_integral_is_one() -> None:
    result = Gaussian(sigma=1.5).transform((41, 41), (20.5, 20.5))
    assert result.sum() == pytest.approx(1.0, abs=1e-3)


def test_gaussian_peak_at_center_pixel() -> None:
    # Centered at the middle of pixel (5,5); that pixel should hold the maximum.
    result = Gaussian(sigma=1.5).transform((11, 11), (5.5, 5.5))
    assert np.unravel_index(result.argmax(), result.shape) == (5, 5)


def test_gaussian_is_symmetric_about_center() -> None:
    result = Gaussian(sigma=2.0).transform((11, 11), (5.5, 5.5))
    assert np.allclose(result, result[::-1, :])
    assert np.allclose(result, result[:, ::-1])


def test_gaussian_expand_preserves_integral() -> None:
    g = Gaussian(sigma=1.0, integral=3.0)
    for expand in [0.5, 1.0, 2.0, 3.0]:
        result = g.transform((61, 61), (30.5, 30.5), expand=expand)
        assert result.sum() == pytest.approx(3.0, abs=1e-3)


def test_gaussian_expand_broadens_peak() -> None:
    g = Gaussian(sigma=1.0, integral=1.0)
    narrow = g.transform((61, 61), (30.5, 30.5), expand=1.0)
    wide = g.transform((61, 61), (30.5, 30.5), expand=3.0)
    assert wide.max() < narrow.max()


def test_gaussian_rotate_is_noop_for_symmetric() -> None:
    # A symmetric Gaussian is unchanged by rotation (the rotate argument is ignored).
    g = Gaussian(sigma=2.0)
    plain = g.transform((21, 21), (10.5, 10.5))
    rotated = g.transform((21, 21), (10.5, 10.5), rotate=0.7)
    assert np.allclose(plain, rotated)


def test_gaussian_all_values_nonnegative() -> None:
    result = Gaussian(sigma=1.5, integral=4.0).transform((15, 15), (7.5, 7.5))
    assert np.all(result >= 0.0)


def test_gaussian_integral_full_range() -> None:
    # The integral over (-inf, inf) of a unit Gaussian is 1.
    assert Gaussian._gaussian_integral(-50.0, 50.0, 0.0, 1.0) == pytest.approx(1.0)


def test_gaussian_integral_half_range() -> None:
    assert Gaussian._gaussian_integral(0.0, 50.0, 0.0, 1.0) == pytest.approx(0.5)


def test_gaussian_integral_symmetric_about_center() -> None:
    left = Gaussian._gaussian_integral(-3.0, 0.0, 0.0, 1.5)
    right = Gaussian._gaussian_integral(0.0, 3.0, 0.0, 1.5)
    assert left == pytest.approx(right)


def test_gaussian_integral_array_inputs() -> None:
    xmin = np.array([-1.0, 0.0, 1.0])
    result = Gaussian._gaussian_integral(xmin, xmin + 1.0, 0.0, 1.0)
    assert result.shape == (3,)
    assert np.all(result > 0.0)


def test_gaussian_psf_offset_shifts_peak() -> None:
    # A unit PSF centered at (0.5, 0.5) peaks in pixel (0, 0).
    i = np.arange(5)[:, np.newaxis]
    j = np.arange(5)
    psf = Gaussian._gaussian_psf(i, j, 0.5, 0.5, 1.0)
    assert np.unravel_index(psf.argmax(), psf.shape) == (0, 0)


##########################################################################################
# ArrayModel
##########################################################################################

def _delta(value: float = 1.0, n: int = 5) -> np.ndarray:
    arr = np.zeros((n, n))
    arr[n // 2, n // 2] = value
    return arr


def test_arraymodel_default_origin_is_midpoint() -> None:
    model = ArrayModel(np.zeros((6, 8)))
    assert np.array_equal(model._origin, np.array([3.0, 4.0]))


def test_arraymodel_explicit_origin() -> None:
    model = ArrayModel(np.zeros((5, 5)), origin=(1.0, 2.0))
    assert np.array_equal(model._origin, np.array([1.0, 2.0]))


def test_arraymodel_transform_shape(no_shortcuts) -> None:
    model = ArrayModel(_delta())
    assert model.transform((9, 11), (4.5, 5.5)).shape == (9, 11)


def test_arraymodel_preserves_integral(no_shortcuts) -> None:
    model = ArrayModel(_delta(value=3.0))
    result = model.transform((11, 11), (5.5, 5.5))
    assert result.sum() == pytest.approx(3.0)


def test_arraymodel_expand_preserves_integral(no_shortcuts) -> None:
    model = ArrayModel(_delta(value=4.0))
    result = model.transform((15, 15), (7.5, 7.5), expand=2.0)
    assert result.sum() == pytest.approx(4.0)


def test_arraymodel_rotate_preserves_integral(no_shortcuts) -> None:
    model = ArrayModel(_delta(value=5.0))
    result = model.transform((15, 15), (7.5, 7.5), rotate=np.pi / 4)
    assert result.sum() == pytest.approx(5.0, abs=1e-6)


def test_arraymodel_rotate_with_expand_preserves_integral(no_shortcuts) -> None:
    model = ArrayModel(_delta(value=2.0))
    result = model.transform((17, 17), (8.5, 8.5), rotate=0.6, expand=2.0)
    assert result.sum() == pytest.approx(2.0, abs=1e-6)


def test_arraymodel_outside_fill(no_shortcuts) -> None:
    # A small source on a large grid leaves the corners filled with `outside`.
    model = ArrayModel(_delta(value=1.0), outside=9.0)
    result = model.transform((9, 9), (4.5, 4.5))
    assert result[0, 0] == pytest.approx(9.0)
    assert result[-1, -1] == pytest.approx(9.0)


def test_arraymodel_outside_default_is_zero(no_shortcuts) -> None:
    model = ArrayModel(_delta())
    result = model.transform((9, 9), (4.5, 4.5))
    assert result[0, 0] == pytest.approx(0.0)


def test_arraymodel_accepts_list_input(no_shortcuts) -> None:
    model = ArrayModel([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    result = model.transform((7, 7), (3.5, 3.5))
    assert result.sum() == pytest.approx(1.0)


##########################################################################################
# SummedModel
##########################################################################################

def test_summedmodel_transform_shape() -> None:
    model = SummedModel([Gaussian(), Gaussian()], [1.0, 1.0])
    assert model.transform((9, 9), (4.5, 4.5)).shape == (9, 9)


def test_summedmodel_integral_is_weighted_sum() -> None:
    g1 = Gaussian(sigma=1.0, integral=2.0)
    g2 = Gaussian(sigma=3.0, integral=5.0)
    model = SummedModel([g1, g2], [2.0, 1.0])
    result = model.transform((61, 61), (30.5, 30.5))
    assert result.sum() == pytest.approx(2.0 * 2.0 + 1.0 * 5.0, abs=1e-3)


def test_summedmodel_equals_manual_combination() -> None:
    g1 = Gaussian(sigma=1.0, integral=2.0)
    g2 = Gaussian(sigma=2.5, integral=4.0)
    model = SummedModel([g1, g2], [1.5, 0.5])
    combined = model.transform((31, 31), (15.5, 15.5))
    manual = (1.5 * g1.transform((31, 31), (15.5, 15.5))
              + 0.5 * g2.transform((31, 31), (15.5, 15.5)))
    assert np.allclose(combined, manual)


def test_summedmodel_zero_factor_drops_component() -> None:
    g1 = Gaussian(sigma=1.0, integral=3.0)
    g2 = Gaussian(sigma=2.0, integral=7.0)
    model = SummedModel([g1, g2], [1.0, 0.0])
    only_first = g1.transform((31, 31), (15.5, 15.5))
    assert np.allclose(model.transform((31, 31), (15.5, 15.5)), only_first)


def test_summedmodel_three_components() -> None:
    parts = [Gaussian(sigma=1.0, integral=1.0) for _ in range(3)]
    model = SummedModel(parts, [1.0, 2.0, 3.0])
    result = model.transform((41, 41), (20.5, 20.5))
    assert result.sum() == pytest.approx(6.0, abs=1e-3)


##########################################################################################
# SmearedModel
##########################################################################################

def test_smearedmodel_transform_shape() -> None:
    model = SmearedModel(Gaussian(sigma=1.0), [4.0, 0.0])
    assert model.transform((21, 21), (10.5, 10.5)).shape == (21, 21)


def test_smearedmodel_preserves_integral() -> None:
    g = Gaussian(sigma=1.0, integral=8.0)
    model = SmearedModel(g, [6.0, 0.0])
    result = model.transform((41, 41), (20.5, 20.5))
    assert result.sum() == pytest.approx(8.0, abs=1e-3)


def test_smearedmodel_nsteps_scales_with_distance() -> None:
    near = SmearedModel(Gaussian(), [2.0, 0.0], maxstep=0.5)
    far = SmearedModel(Gaussian(), [10.0, 0.0], maxstep=0.5)
    assert far._nsteps > near._nsteps


def test_smearedmodel_offsets_match_nsteps() -> None:
    model = SmearedModel(Gaussian(), [5.0, 0.0])
    assert model._offsets.shape[0] == model._nsteps


def test_smearedmodel_lowers_peak_vs_unsmeared() -> None:
    g = Gaussian(sigma=1.0, integral=10.0)
    unsmeared = g.transform((41, 41), (20.5, 20.5))
    smeared = SmearedModel(g, [8.0, 0.0]).transform((41, 41), (20.5, 20.5))
    assert smeared.max() < unsmeared.max()


def test_smearedmodel_broadens_along_smear_axis() -> None:
    # Smearing along the first axis should widen the profile along that axis.
    g = Gaussian(sigma=1.0, integral=1.0)
    smeared = SmearedModel(g, [10.0, 0.0]).transform((61, 61), (30.5, 30.5))
    center_row = smeared[30, :]
    center_col = smeared[:, 30]
    # The smeared (first) axis spreads more than the unsmeared (second) axis.
    assert center_col.std() > center_row.std()


def test_smearedmodel_zero_smear_matches_original() -> None:
    g = Gaussian(sigma=1.5, integral=4.0)
    model = SmearedModel(g, [0.0, 0.0])
    assert model._nsteps == 1
    smeared = model.transform((31, 31), (15.5, 15.5))
    plain = g.transform((31, 31), (15.5, 15.5))
    assert np.allclose(smeared, plain)


def test_smearedmodel_accepts_array_smear() -> None:
    model = SmearedModel(Gaussian(integral=2.0), np.array([3.0, 4.0]))
    result = model.transform((41, 41), (20.5, 20.5))
    assert result.sum() == pytest.approx(2.0, abs=1e-3)

##########################################################################################
