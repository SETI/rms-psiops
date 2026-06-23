##########################################################################################
# tests/test_circle.py
##########################################################################################
"""Tests for psiops.circle."""

import numpy as np

from psiops.circle import circle


def _reference_circle(radius: float) -> np.ndarray:
    """An independent reference implementation of a circular footprint."""

    halfsize = int(radius)
    coords = np.arange(-halfsize, halfsize + 1)
    distance_sq = coords**2 + coords[:, np.newaxis]**2
    return distance_sq <= radius**2


def test_circle_returns_bool_array() -> None:
    result = circle(3.0)
    assert isinstance(result, np.ndarray)
    assert result.dtype == np.bool_


def test_circle_is_square() -> None:
    result = circle(4.0)
    assert result.shape[0] == result.shape[1]


def test_circle_shape_is_odd() -> None:
    # An odd shape avoids introducing an offset when used as a filter footprint.
    for radius in [0.5, 1.0, 2.0, 3.7, 10.0]:
        result = circle(radius)
        assert result.shape[0] % 2 == 1
        assert result.shape[1] % 2 == 1


def test_circle_shape_matches_radius() -> None:
    # The half-size is int(radius), so the full size is 2*int(radius) + 1.
    assert circle(3.0).shape == (7, 7)
    assert circle(3.9).shape == (7, 7)
    assert circle(4.0).shape == (9, 9)


def test_circle_center_is_true() -> None:
    result = circle(5.0)
    center = result.shape[0] // 2
    assert result[center, center]


def test_circle_corners_are_false_for_large_radius() -> None:
    # For radius 3, the array is 7x7; the corners are at distance sqrt(18) > 3.
    result = circle(3.0)
    assert not result[0, 0]
    assert not result[0, -1]
    assert not result[-1, 0]
    assert not result[-1, -1]


def test_circle_is_symmetric() -> None:
    result = circle(4.3)
    assert np.array_equal(result, result[::-1, :])
    assert np.array_equal(result, result[:, ::-1])
    assert np.array_equal(result, result.T)


def test_circle_matches_reference() -> None:
    for radius in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.7, 8.0]:
        assert np.array_equal(circle(radius), _reference_circle(radius))


def test_circle_count_within_radius() -> None:
    # Count exactly the pixels whose center is within the radius, computed independently.
    result = circle(5.0)
    center = result.shape[0] // 2
    coords = np.arange(result.shape[0]) - center
    grid_sq = coords**2 + coords[:, np.newaxis]**2
    expected_count = int((grid_sq <= 25.0).sum())
    assert int(result.sum()) == expected_count


def test_circle_radius_one() -> None:
    # Radius 1 includes the center and its four edge neighbors (a plus shape).
    result = circle(1.0)
    assert result.shape == (3, 3)
    assert int(result.sum()) == 5
    assert result[1, 1]
    assert not result[0, 0]


def test_circle_zero_radius() -> None:
    # int(0) gives a 1x1 array; the single pixel is at distance 0 <= 0.
    result = circle(0.0)
    assert result.shape == (1, 1)
    assert result[0, 0]


def test_circle_pixels_outside_radius_excluded() -> None:
    # Every False pixel must be strictly farther than the radius.
    radius = 4.6
    result = circle(radius)
    center = result.shape[0] // 2
    coords = np.arange(result.shape[0]) - center
    grid_sq = coords**2 + coords[:, np.newaxis]**2
    assert np.all(grid_sq[result] <= radius**2)
    assert np.all(grid_sq[~result] > radius**2)

##########################################################################################
