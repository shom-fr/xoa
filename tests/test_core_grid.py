# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.grid` module
"""

import numpy as np

from xoa.core import grid


def test_centers2edges_1d():
    np.testing.assert_allclose(grid.centers2edges(np.array([0.0, 1.0, 2.0])), [-0.5, 0.5, 1.5, 2.5])


def test_centers2edges_2d():
    xx, yy = np.meshgrid(np.arange(3.0), np.arange(2.0))
    assert grid.centers2edges(xx).shape == (3, 4)
    assert grid.centers2edges(xx, axis=1).shape == (2, 4)


def test_edges2bounds():
    assert grid.edges2bounds(np.arange(4.0)).shape == (3, 2)
    xx, yy = np.meshgrid(np.arange(4.0), np.arange(3.0))
    bounds = grid.edges2bounds(xx)
    assert bounds.shape == (2, 3, 4)
    np.testing.assert_array_equal(bounds[0, 0], [0.0, 1.0, 1.0, 0.0])


def test_check_grid_type():
    xx, yy = np.meshgrid(np.arange(4.0), np.arange(3.0))
    assert grid.check_grid_type({"lon": xx, "lat": yy}) == "regular"
    xx, yy = np.meshgrid(np.array([0.0, 1.0, 3.0, 4.0]), np.arange(3.0))
    assert grid.check_grid_type({"lon": xx, "lat": yy}) == "rectangular"
    assert grid.check_grid_type({"lon": xx, "lat": yy + 0.1 * xx}) == "curvilinear"
    assert grid.check_grid_type({"lon": xx, "lat": yy, "type": "regular"}) == "regular"


def test_compute_resolution():
    lon, lat = np.meshgrid(np.linspace(0, 2, 3), np.linspace(0, 3, 4))
    dx, dy = grid.compute_resolution(lon, lat)
    assert dx.shape == (4, 2)
    assert dy.shape == (3, 3)
    # One degree along the equator and a meridian
    ref = np.deg2rad(1.0) * 6371e3
    np.testing.assert_allclose(dx[0], ref)
    np.testing.assert_allclose(dy, ref)
    # Smaller x resolution at high latitudes
    assert dx[-1, 0] < dx[0, 0]
    np.testing.assert_allclose(dx[-1, 0], ref * np.cos(np.deg2rad(3.0)), rtol=1e-3)
    # Radius
    dx2, dy2 = grid.compute_resolution(lon, lat, radius=1.0)
    np.testing.assert_allclose(dx2, dx / 6371e3)


def test_median_resolution_deg():
    lon, lat = np.meshgrid(np.arange(0, 5, 0.5), np.arange(0, 4, 0.25))
    np.testing.assert_allclose(grid.median_resolution_deg(lon, lat), 0.5)
    # A rotated grid spans more than its step along each axis
    rot = np.deg2rad(30.0)
    rlon = lon * np.cos(rot) - lat * np.sin(rot)
    rlat = lon * np.sin(rot) + lat * np.cos(rot)
    assert grid.median_resolution_deg(rlon, rlat) > 0.25


def test_check_grid_type_with_a_single_row_or_column():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xx, yy = np.meshgrid(np.arange(5.0), np.zeros(1))
        assert grid.check_grid_type({"lon": xx, "lat": yy}) == "regular"
        xx, yy = np.meshgrid(np.zeros(1), np.arange(4.0))
        assert grid.check_grid_type({"lon": xx, "lat": yy}) == "regular"
        xx, yy = np.meshgrid(np.array([0.0, 1.0, 3.0]), np.zeros(1))
        assert grid.check_grid_type({"lon": xx, "lat": yy}) == "rectangular"


def test_unwrap_grid_longitudes():
    lon = np.array([[178.0, 180.0, -178.0, -176.0]] * 2)
    out = grid.unwrap_grid_longitudes(lon)
    np.testing.assert_allclose(out[0], [178.0, 180.0, 182.0, 184.0])
    assert out.shape == lon.shape
    # Already continuous: unchanged
    lon = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(grid.unwrap_grid_longitudes(lon), lon)
