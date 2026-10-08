# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.geo` module
"""

import numpy as np
import pytest

from xoa.core import geo


class TestHaversine:
    """Test haversine distance calculation on unit sphere"""

    def test_haversine_scalar(self):
        """Test haversine with scalar inputs"""
        # Distance from (0,0) to (180,0) should be pi on unit sphere
        dist = geo.haversine(0.0, 0.0, 180.0, 0.0)
        np.testing.assert_allclose(dist, np.pi)

    def test_haversine_poles(self):
        """Test haversine between poles"""
        dist = geo.haversine(0.0, -90.0, 0.0, 90.0)
        np.testing.assert_allclose(dist, np.pi)

    def test_haversine_same_point(self):
        """Test haversine for same point"""
        dist = geo.haversine(10.0, 20.0, 10.0, 20.0)
        np.testing.assert_allclose(dist, 0.0)

    def test_haversine_array(self):
        """Test haversine with array inputs"""
        lon0 = np.array([0.0, 0.0, 10.0])
        lat0 = np.array([0.0, -90.0, 20.0])
        lon1 = np.array([180.0, 0.0, 10.0])
        lat1 = np.array([0.0, 90.0, 20.0])

        dists = geo.haversine(lon0, lat0, lon1, lat1)
        assert dists.shape == (3,)
        np.testing.assert_allclose(dists[0], np.pi)
        np.testing.assert_allclose(dists[1], np.pi)
        np.testing.assert_allclose(dists[2], 0.0)


class TestBearing:
    """Test bearing angle calculation"""

    def test_bearing_north(self):
        """Test bearing pointing north (90° in math convention)"""
        angle = geo.bearing(0.0, 0.0, 0.0, 90.0)
        np.testing.assert_allclose(angle, 90.0, atol=1e-10)

    def test_bearing_east(self):
        """Test bearing pointing east (0° in math convention)"""
        angle = geo.bearing(0.0, 0.0, 90.0, 0.0)
        np.testing.assert_allclose(angle, 0.0, atol=1e-10)

    def test_bearing_south(self):
        """Test bearing pointing south (-90° in math convention)"""
        angle = geo.bearing(0.0, 90.0, 0.0, 0.0)
        np.testing.assert_allclose(angle, -90.0, atol=1e-10)

    def test_bearing_west(self):
        """Test bearing pointing west (180° in math convention)"""
        angle = geo.bearing(0.0, 0.0, -90.0, 0.0)
        np.testing.assert_allclose(angle, 180.0, atol=1e-10)

    def test_bearing_array(self):
        """Test bearing with array inputs"""
        lon0 = np.array([0.0, 0.0, 0.0, 0.0])
        lat0 = np.array([0.0, 0.0, 90.0, 0.0])
        lon1 = np.array([0.0, 90.0, 0.0, -90.0])
        lat1 = np.array([90.0, 0.0, 0.0, 0.0])

        angles = geo.bearing(lon0, lat0, lon1, lat1)
        assert angles.shape == (4,)
        np.testing.assert_allclose(angles[0], 90.0, atol=1e-10)  # North
        np.testing.assert_allclose(angles[1], 0.0, atol=1e-10)   # East
        np.testing.assert_allclose(angles[2], -90.0, atol=1e-10) # South
        np.testing.assert_allclose(angles[3], 180.0, atol=1e-10) # West

    def test_bearing_same_point(self):
        """Test bearing for same point (undefined but should not error)"""
        angle = geo.bearing(10.0, 20.0, 10.0, 20.0)
        # Just check it returns a value without error
        assert isinstance(angle, (float, np.floating))

    def test_bearing_diagonal(self):
        """Test bearing for diagonal direction"""
        # From (0,0) to (45,45) should give a specific angle
        angle = geo.bearing(0.0, 0.0, 45.0, 45.0)
        # Just verify it's in reasonable range and consistent
        assert -180 <= angle <= 180
        assert isinstance(angle, (float, np.floating))


class TestGridHelpers:
    """Test the grid search and cell location helpers"""

    def test_diff_lon(self):
        assert geo.diff_lon(170.0, -170.0) == 20.0
        assert geo.diff_lon(-170.0, 170.0) == -20.0

    def test_normalize_longitude(self):
        assert geo.normalize_longitude(190.0) == -170.0
        assert geo.normalize_longitude(-190.0) == 170.0

    def test_closest_point_fast(self):
        for n in (5, 120):
            lons, lats = np.meshgrid(np.arange(n, dtype="d"), np.arange(n, dtype="d"))
            assert geo.closest_point_fast(lons, lats, 2.1, 3.2) == (2, 3)

    def test_relative_cell_coords(self):
        p, q = geo.relative_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.25, 0.75)
        np.testing.assert_allclose((p, q), (0.25, 0.75))
        assert geo.relative_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0) == (
            -1.0,
            -1.0,
        )


class TestClosestPointFastOnLargeGrids:
    """The search for large grids must find the closest node of the grid"""

    @staticmethod
    def get_grid(n):
        x, y = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 8, n))
        return x + 0.5 * np.sin(0.7 * y) + 0.05 * y, y + 0.4 * np.sin(0.6 * x) - 0.03 * x

    @staticmethod
    def get_exact(lon, lat, x, y):
        dist = geo.haversine(x, y, lon, lat)
        j, i = np.unravel_index(np.argmin(dist), dist.shape)
        return i, j

    @pytest.mark.parametrize("n", [60, 300, 600])
    def test_same_node_as_the_full_search(self, n):
        lon, lat = self.get_grid(n)
        rng = np.random.default_rng(1)
        jj, ii = rng.integers(2, n - 3, 150), rng.integers(2, n - 3, 150)
        # Points inside the cells, which are not on the nodes
        x = lon[jj, ii] + rng.uniform(-0.5, 0.5, 150) * (lon[jj, ii + 1] - lon[jj, ii])
        y = lat[jj, ii] + rng.uniform(-0.5, 0.5, 150) * (lat[jj + 1, ii] - lat[jj, ii])
        for xi, yi in zip(x, y):
            assert geo.closest_point_fast(lon, lat, xi, yi) == self.get_exact(lon, lat, xi, yi)

    def test_points_that_are_far_from_the_grid(self):
        lon, lat = self.get_grid(200)
        i, j = geo.closest_point_fast(lon, lat, -5.0, -5.0)
        assert (i, j) == self.get_exact(lon, lat, -5.0, -5.0) == (0, 0)
        i, j = geo.closest_point_fast(lon, lat, 20.0, 20.0)
        assert (i, j) == self.get_exact(lon, lat, 20.0, 20.0)

    def test_nan_nodes_are_ignored(self):
        lon, lat = self.get_grid(200)
        x, y = lon[100, 100], lat[100, 100]
        lon, lat = lon.copy(), lat.copy()
        lon[95:106, 95:106] = np.nan
        lat[95:106, 95:106] = np.nan
        i, j = geo.closest_point_fast(lon, lat, x, y)
        assert np.isfinite(lon[j, i])
        dist = geo.haversine(x, y, lon, lat)
        jx, ix = np.unravel_index(np.nanargmin(dist), dist.shape)
        assert (i, j) == (ix, jx)

    def test_all_the_subsampled_nodes_are_nan(self):
        lon, lat = self.get_grid(200)
        mask = np.ones(lon.shape, bool)
        mask[::10, ::10] = False  # the subsampled nodes (step 10) are the only valid ones... inverted
        lon = np.where(mask, lon, np.nan)
        lat = np.where(mask, lat, np.nan)
        x, y = lon[51, 52], lat[51, 52]
        assert geo.closest_point_fast(lon, lat, x, y) == (52, 51)
