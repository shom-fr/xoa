# -*- coding: utf-8 -*-
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


def _grid(lon0=0.0):
    lon, lat = np.meshgrid(np.arange(10.0) + 0.5, np.arange(8.0) + 0.5)
    lon_bounds = np.stack([lon - 0.5, lon + 0.5, lon + 0.5, lon - 0.5], axis=-1)
    lat_bounds = np.stack([lat - 0.5, lat - 0.5, lat + 0.5, lat + 0.5], axis=-1)
    wrap = lambda x: (x + lon0 + 180) % 360 - 180  # noqa
    return wrap(lon), lat, wrap(lon_bounds), lat_bounds, wrap


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


def test_polygon_cell_fraction():
    lon, lat, lonb, latb, _ = _grid()
    frac = poly.polygon_cell_fraction(RECT, lon, lat, lonb, latb)
    assert frac.shape == lon.shape
    np.testing.assert_allclose(frac[2:4, 2:5], 1.0)  # Fully covered
    np.testing.assert_allclose(frac[2:4, 1], 0.5)  # Half covered edge cells
    np.testing.assert_allclose(frac[1, 2:5], 0.5)
    np.testing.assert_allclose(frac[1, 1], 0.25)  # Corner
    assert frac[0].sum() == 0
    # Total covered area
    np.testing.assert_allclose(frac.sum(), 4 * 3)


def test_polygon_mask_threshold():
    lon, lat, lonb, latb, _ = _grid()
    assert poly.polygon_mask(RECT, lon, lat, lonb, latb, 1.0).sum() == 2 * 3
    assert poly.polygon_mask(RECT, lon, lat, lonb, latb, 0.5).sum() == 4 * 5 - 4
    assert poly.polygon_mask(RECT, lon, lat, lonb, latb, 0.3).sum() == 4 * 5 - 4
    assert poly.polygon_mask(RECT, lon, lat, lonb, latb, 0.2).sum() == 4 * 5
    assert poly.polygon_mask(RECT, lon, lat, lonb, latb, 0.0).sum() == 4 * 5
    # Centers only
    mask = poly.polygon_mask(RECT, lon, lat)
    assert mask.sum() == 4 * 3
    assert mask.dtype == bool


def test_polygon_mask_concave():
    lon, lat, lonb, latb, _ = _grid()
    concave = np.array([[0, 0], [8, 0], [8, 6], [4, 2], [0, 6]], dtype="d")
    frac = poly.polygon_cell_fraction(concave, lon, lat, lonb, latb)
    # Area of the polygon: rectangle minus the notch triangle
    np.testing.assert_allclose(frac.sum(), 8 * 6 - 0.5 * 8 * 4)
    assert frac[5, 4] == 0.0  # In the notch


def test_polygon_mask_across_the_dateline():
    ref = poly.polygon_cell_fraction(RECT, *_grid()[:4])
    lon, lat, lonb, latb, wrap = _grid(175.0)
    assert (lon < 0).any() and (lon > 0).any()
    shifted = RECT.copy()
    shifted[:, 0] = wrap(shifted[:, 0])
    frac = poly.polygon_cell_fraction(shifted, lon, lat, lonb, latb)
    np.testing.assert_allclose(frac, ref)


def test_polygon_mask_missing_centers():
    lon, lat, lonb, latb, _ = _grid()
    lon[2, 2] = np.nan
    mask = poly.polygon_mask(RECT, lon, lat, lonb, latb, 0.5)
    assert not mask[2, 2]
