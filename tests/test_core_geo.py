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
