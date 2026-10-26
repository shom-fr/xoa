"""
Test the :mod:`xoa.core.poly` module
"""

import numpy as np

from xoa.core import poly


def _square(lon0, lon1, lat0, lat1):
    return np.array([lat0, lat0, lat1, lat1]), np.array([lon0, lon1, lon1, lon0])


def test_spherical_area():
    area = poly.spherical_area(*_square(0.0, 1.0, 0.0, 1.0))
    np.testing.assert_allclose(area, np.deg2rad(1.0) ** 2 * poly.EARTH_RADIUS**2, rtol=1e-2)


def test_compute_overlap_area():
    ref = poly.spherical_area(*_square(0.0, 1.0, 0.0, 1.0))
    half = poly.compute_overlap_area(*_square(0.0, 1.0, 0.0, 1.0), *_square(0.5, 1.5, 0.0, 1.0))
    np.testing.assert_allclose(half, 0.5 * ref, rtol=1e-2)
    assert (
        poly.compute_overlap_area(*_square(0.0, 1.0, 0.0, 1.0), *_square(2.0, 3.0, 0.0, 1.0)) == 0.0
    )


def test_compute_overlap_area_across_the_dateline():
    lats = np.array([0.0, 0.0, 2.0, 2.0])
    ref = poly.compute_overlap_area(
        lats, np.array([-12.0, -8.0, -8.0, -12.0]), lats + 1.0, np.array([-10.0, -6.0, -6.0, -10.0])
    )
    assert ref > 0
    for shift in (170.0, -170.0, 360.0):
        for sa, sb in ((0, 0), (0, 360.0), (360.0, 0)):
            lon0 = np.array([-12.0, -8.0, -8.0, -12.0]) + shift
            lon1 = np.array([-10.0, -6.0, -6.0, -10.0]) + shift
            wrap = lambda lon: ((lon + 180.0) % 360.0) - 180.0
            area = poly.compute_overlap_area(lats, wrap(lon0 + sa), lats + 1.0, wrap(lon1 + sb))
            np.testing.assert_allclose(area, ref, rtol=1e-9)


RECT = np.array([[1.5, 1.5], [5.5, 1.5], [5.5, 4.5], [1.5, 4.5]])


def test_point_in_polygon():
    concave = np.array([[0, 0], [4, 0], [4, 4], [2, 1], [0, 4]], dtype="d")
    assert poly.point_in_polygon(1.0, 1.0, concave)
    assert poly.point_in_polygon(3.5, 1.0, concave)
    assert not poly.point_in_polygon(2.0, 3.0, concave)  # In the notch
    assert not poly.point_in_polygon(5.0, 1.0, concave)
    # Same result for both orientations
    assert poly.point_in_polygon(1.0, 1.0, concave[::-1])


def test_points_in_polygon():
    x, y = np.meshgrid(np.arange(8.0), np.arange(6.0))
    inside = poly.points_in_polygon(x, y, RECT)
    assert inside.shape == x.shape
    assert inside.sum() == 4 * 3
    assert inside[2, 2] and not inside[0, 0]
