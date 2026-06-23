##########################################################################################
# tests/test_fft.py
##########################################################################################
"""Tests for psiops.fft."""

import numpy as np
import pytest
from scipy import fftpack

from psiops.fft import (
    _retile,
    autocorrelate,
    correlate,
    fft,
    fft_power,
    ialign,
    ifft,
)
from psiops.ishift import ishift

##########################################################################################
# fft
##########################################################################################

def test_fft_matches_fftpack_2d() -> None:
    rng = np.random.default_rng(0)
    img = rng.standard_normal((8, 8))
    assert np.allclose(fft(img), fftpack.fft2(img))


def test_fft_returns_complex_by_default() -> None:
    img = np.ones((4, 4))
    result = fft(img)
    assert result.dtype == np.complex128
    assert result.shape == (4, 4)


def test_fft_real_returns_real_component() -> None:
    rng = np.random.default_rng(1)
    img = rng.standard_normal((6, 6))
    result = fft(img, real=True)
    assert result.dtype == np.float64
    assert np.allclose(result, fftpack.fft2(img).real)


def test_fft_retile_even_equals_fftshift() -> None:
    rng = np.random.default_rng(2)
    img = rng.standard_normal((8, 10))
    assert np.allclose(fft(img, retile=True), np.fft.fftshift(fftpack.fft2(img)))


def test_fft_retile_real_combo() -> None:
    rng = np.random.default_rng(3)
    img = rng.standard_normal((8, 8))
    result = fft(img, retile=True, real=True)
    expected = np.fft.fftshift(fftpack.fft2(img).real)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_fft_3d_matches_per_slice() -> None:
    rng = np.random.default_rng(4)
    img = rng.standard_normal((3, 8, 8))
    result = fft(img)
    assert result.shape == (3, 8, 8)
    assert result.dtype == np.complex128
    for k in range(3):
        assert np.allclose(result[k], fftpack.fft2(img[k]))


def test_fft_3d_real() -> None:
    rng = np.random.default_rng(5)
    img = rng.standard_normal((2, 8, 8))
    result = fft(img, real=True)
    assert result.dtype == np.float64
    for k in range(2):
        assert np.allclose(result[k], fftpack.fft2(img[k]).real)


def test_fft_3d_retile() -> None:
    rng = np.random.default_rng(6)
    img = rng.standard_normal((2, 8, 8))
    result = fft(img, retile=True)
    for k in range(2):
        assert np.allclose(result[k], np.fft.fftshift(fftpack.fft2(img[k])))


def test_fft_4d_iterates_all_leading_indices() -> None:
    rng = np.random.default_rng(7)
    img = rng.standard_normal((2, 3, 4, 4))
    result = fft(img)
    assert result.shape == (2, 3, 4, 4)
    for i in range(2):
        for j in range(3):
            assert np.allclose(result[i, j], fftpack.fft2(img[i, j]))


##########################################################################################
# ifft
##########################################################################################

def test_ifft_matches_fftpack_2d() -> None:
    rng = np.random.default_rng(8)
    img = rng.standard_normal((8, 8))
    assert np.allclose(ifft(img), fftpack.ifft2(img))


def test_fft_ifft_round_trip() -> None:
    rng = np.random.default_rng(9)
    img = rng.standard_normal((8, 8))
    assert np.allclose(ifft(fft(img)), img)


def test_ifft_real() -> None:
    rng = np.random.default_rng(10)
    img = rng.standard_normal((6, 6))
    result = ifft(img, real=True)
    assert result.dtype == np.float64
    assert np.allclose(result, fftpack.ifft2(img).real)


def test_ifft_3d_matches_per_slice() -> None:
    rng = np.random.default_rng(11)
    img = rng.standard_normal((3, 8, 8))
    result = ifft(img)
    assert result.shape == (3, 8, 8)
    for k in range(3):
        assert np.allclose(result[k], fftpack.ifft2(img[k]))


def test_ifft_3d_real() -> None:
    rng = np.random.default_rng(12)
    img = rng.standard_normal((2, 8, 8))
    result = ifft(img, real=True)
    assert result.dtype == np.float64
    for k in range(2):
        assert np.allclose(result[k], fftpack.ifft2(img[k]).real)


def test_ifft_retile_even_equals_fftshift() -> None:
    rng = np.random.default_rng(13)
    img = rng.standard_normal((8, 8))
    assert np.allclose(ifft(img, retile=True), np.fft.fftshift(fftpack.ifft2(img)))


##########################################################################################
# fft_power
##########################################################################################

def test_fft_power_matches_reference() -> None:
    rng = np.random.default_rng(14)
    img = rng.standard_normal((8, 8))
    raw = fftpack.fft2(img)
    expected = (raw * np.conj(raw)).real
    result = fft_power(img)
    assert result.dtype == np.float64
    assert np.allclose(result, expected)


def test_fft_power_is_nonnegative() -> None:
    rng = np.random.default_rng(15)
    img = rng.standard_normal((8, 8))
    assert np.all(fft_power(img) >= -1e-9)


def test_fft_power_dc_equals_sum_squared() -> None:
    # The (0,0) power equals the squared sum of the image.
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    result = fft_power(img)
    assert np.isclose(result[0, 0], img.sum() ** 2)


def test_fft_power_retile() -> None:
    rng = np.random.default_rng(16)
    img = rng.standard_normal((8, 8))
    raw = fftpack.fft2(img)
    pw = (raw * np.conj(raw)).real
    assert np.allclose(fft_power(img, retile=True), np.fft.fftshift(pw))


def test_fft_power_accepts_list_input() -> None:
    img = [[1.0, 2.0], [3.0, 4.0]]
    result = fft_power(img)
    assert result[0, 0] == pytest.approx(100.0)


##########################################################################################
# correlate
##########################################################################################

def test_correlate_peak_at_offset() -> None:
    # A shifted delta correlated against the original peaks at the shift offset.
    ref = np.zeros((16, 16))
    ref[4, 6] = 1.0
    image = ishift(ref, (3, -2), mode='wrap')
    corr = correlate(image, ref)
    peak = np.unravel_index(np.argmax(corr), corr.shape)
    assert peak == (3, 14)  # -2 wraps to 14


def test_correlate_matches_reference_formula() -> None:
    rng = np.random.default_rng(17)
    image = rng.standard_normal((8, 8))
    ref = rng.standard_normal((8, 8))
    image_fft = fftpack.fft2(image)
    ref_fft = fftpack.fft2(ref)
    expected = fftpack.ifft2(image_fft * np.conj(ref_fft)).real
    assert np.allclose(correlate(image, ref), expected)


def test_correlate_normalize_self_peaks_at_one() -> None:
    rng = np.random.default_rng(18)
    img = rng.standard_normal((8, 8))
    corr = correlate(img, img, normalize=True)
    # Normalized self-correlation peaks at 1 at zero lag.
    assert corr[0, 0] == pytest.approx(1.0)
    assert corr.max() == pytest.approx(1.0)


def test_correlate_normalize_bounds() -> None:
    rng = np.random.default_rng(19)
    image = rng.standard_normal((8, 8))
    ref = rng.standard_normal((8, 8))
    corr = correlate(image, ref, normalize=True)
    assert corr.max() <= 1.0 + 1e-9
    assert corr.min() >= -1.0 - 1e-9


def test_correlate_retile() -> None:
    rng = np.random.default_rng(20)
    image = rng.standard_normal((8, 8))
    ref = rng.standard_normal((8, 8))
    base = correlate(image, ref)
    retiled = correlate(image, ref, retile=True)
    assert np.allclose(retiled, np.fft.fftshift(base))


##########################################################################################
# autocorrelate
##########################################################################################

def test_autocorrelate_normalized_to_one_at_origin() -> None:
    rng = np.random.default_rng(21)
    img = rng.standard_normal((8, 8))
    result = autocorrelate(img)
    assert result[0, 0] == pytest.approx(1.0)


def test_autocorrelate_matches_reference() -> None:
    rng = np.random.default_rng(22)
    img = rng.standard_normal((8, 8))
    raw = fftpack.fft2(img)
    expected = fftpack.ifft2(raw * np.conj(raw)).real
    expected /= expected[0, 0]
    assert np.allclose(autocorrelate(img), expected)


def test_autocorrelate_retile_moves_peak_to_center() -> None:
    rng = np.random.default_rng(23)
    img = rng.standard_normal((8, 8))
    result = autocorrelate(img, retile=True)
    # After retiling, the zero-lag peak (value 1) lands at the center.
    assert result[4, 4] == pytest.approx(1.0)


##########################################################################################
# ialign
##########################################################################################

@pytest.mark.parametrize('offset', [(0, 0), (2, 3), (-2, -3), (4, -1), (-4, 1)])
def test_ialign_recovers_shift(offset: tuple[int, int]) -> None:
    rng = np.random.default_rng(24)
    ref = rng.standard_normal((10, 10))
    image = ishift(ref, offset, mode='wrap')
    found = ialign(image, ref, 0)
    recovered = ishift(image, found, mode='wrap')
    assert np.allclose(recovered, ref)


@pytest.mark.parametrize('offset', [(0, 0), (2, 3), (-3, 1)])
def test_ialign_with_sigma_recovers_shift(offset: tuple[int, int]) -> None:
    # A non-zero sigma unsharp-masks both images before correlating; the recovered
    # offset must still align the image to the reference.
    rng = np.random.default_rng(28)
    ref = rng.standard_normal((12, 12))
    image = ishift(ref, offset, mode='wrap')
    found = ialign(image, ref, 1.5)
    recovered = ishift(image, found, mode='wrap')
    assert np.allclose(recovered, ref)


def test_ialign_zero_shift() -> None:
    rng = np.random.default_rng(25)
    ref = rng.standard_normal((12, 12))
    assert ialign(ref, ref, 0) == (0, 0)


def test_ialign_returns_python_pair() -> None:
    rng = np.random.default_rng(26)
    ref = rng.standard_normal((8, 8))
    image = ishift(ref, (1, 1), mode='wrap')
    di, dj = ialign(image, ref, 0)
    assert isinstance(di, (int, np.integer))
    assert isinstance(dj, (int, np.integer))


##########################################################################################
# _retile (internal)
##########################################################################################

def test_retile_no_op_without_flags() -> None:
    img = np.arange(16).reshape(4, 4).astype(np.complex128)
    out = _retile(img, retile=False, real=False, dest=None)
    assert out is img


def test_retile_real_only_converts_dtype() -> None:
    img = (np.arange(16).reshape(4, 4) + 1j).astype(np.complex128)
    out = _retile(img, retile=False, real=True, dest=None)
    assert out.dtype.kind == 'f'
    assert np.allclose(out, img.real)


def test_retile_even_is_self_inverse() -> None:
    rng = np.random.default_rng(27)
    img = rng.standard_normal((6, 8))
    once = _retile(img.copy(), retile=True)
    twice = _retile(once.copy(), retile=True)
    assert np.allclose(twice, img)


def test_retile_writes_into_dest() -> None:
    img = np.arange(16, dtype=np.float64).reshape(4, 4)
    dest = np.empty_like(img)
    out = _retile(img, retile=True, dest=dest)
    assert out is dest
    assert np.allclose(out, np.fft.fftshift(img))


def test_retile_dest_copy_without_retile() -> None:
    # dest provided but retile False: data is copied through.
    img = np.arange(9, dtype=np.float64).reshape(3, 3)
    dest = np.empty_like(img)
    out = _retile(img, retile=False, dest=dest)
    assert out is dest
    assert np.allclose(out, img)

##########################################################################################
