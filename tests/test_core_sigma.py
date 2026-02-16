# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.sigma` module
"""

import numpy as np
import pytest

from xoa.core import sigma


class TestAtmosphereSigma:
    """Test atmosphere_sigma gufunc"""

    def test_basic_computation(self):
        """Test basic atmosphere sigma to pressure conversion"""
        sigma_levels = np.array([0.0, 0.5, 1.0])
        ps = 101325.0  # Surface pressure in Pa
        ptop = 0.0  # Top pressure

        result = sigma.atmosphere_sigma_coordinate(sigma_levels, ps, ptop)

        expected = np.array([0.0, 50662.5, 101325.0])
        np.testing.assert_allclose(result, expected)

    def test_with_top_pressure(self):
        """Test with non-zero top pressure"""
        sigma_levels = np.array([0.0, 0.5, 1.0])
        ps = 101325.0
        ptop = 10000.0  # 100 hPa at top

        result = sigma.atmosphere_sigma_coordinate(sigma_levels, ps, ptop)

        expected = np.array([10000.0, 55662.5, 101325.0])
        np.testing.assert_allclose(result, expected)

    def test_broadcasting_2d(self):
        """Test broadcasting over 2D surface pressure"""
        sigma_levels = np.array([0.0, 0.5, 1.0])
        ps = np.array([[100000.0, 101000.0], [102000.0, 103000.0]])
        ptop = 0.0

        result = sigma.atmosphere_sigma_coordinate(sigma_levels, ps, ptop)

        assert result.shape == (2, 2, 3)
        # Check first point
        np.testing.assert_allclose(result[0, 0, :], [0.0, 50000.0, 100000.0])
        # Check last point
        np.testing.assert_allclose(result[1, 1, :], [0.0, 51500.0, 103000.0])


class TestOceanSigma:
    """Test ocean_sigma gufunc"""

    def test_basic_computation(self):
        """Test basic ocean sigma to depth conversion"""
        sigma_levels = np.array([-1.0, -0.5, 0.0])
        eta = 0.5  # Sea surface height in m
        depth = 100.0  # Bottom depth in m

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        # z = eta + sigma * (eta + depth)
        expected = np.array([-100.0, -49.75, 0.5])
        np.testing.assert_allclose(result, expected)

    def test_zero_sea_level(self):
        """Test with zero sea surface height"""
        sigma_levels = np.array([-1.0, -0.5, 0.0])
        eta = 0.0
        depth = 100.0

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        expected = np.array([-100.0, -50.0, 0.0])
        np.testing.assert_allclose(result, expected)

    def test_broadcasting_2d(self):
        """Test broadcasting over 2D bathymetry and SSH"""
        sigma_levels = np.array([-1.0, -0.5, 0.0])
        eta = np.array([[0.5, 1.0], [0.0, -0.5]])
        depth = np.array([[100.0, 150.0], [200.0, 50.0]])

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        assert result.shape == (2, 2, 3)
        # Check first point: eta=0.5, depth=100
        np.testing.assert_allclose(result[0, 0, :], [-100.0, -49.75, 0.5])
        # Check point with different values: eta=1.0, depth=150
        np.testing.assert_allclose(result[0, 1, :], [-150.0, -74.5, 1.0])


class TestOceanS:
    """Test ocean_s gufunc"""

    def test_basic_computation(self):
        """Test ocean s-coordinate to depth conversion"""
        s_levels = np.array([-1.0, -0.5, 0.0])
        eta = 0.5
        depth = 100.0
        depth_c = 10.0
        C = np.array([0.0, 0.5, 1.0])

        result = sigma.ocean_s_coordinate(s_levels, eta, depth, depth_c, C)

        # z = eta*(1+s) + depth_c*s + (depth-depth_c)*C
        # For s=-1: 0.5*0 + 10*(-1) + 90*0 = -10
        # For s=-0.5: 0.5*0.5 + 10*(-0.5) + 90*0.5 = 0.25 - 5 + 45 = 40.25
        # For s=0: 0.5*1 + 10*0 + 90*1 = 0.5 + 0 + 90 = 90.5
        expected = np.array([-10.0, 40.25, 90.5])
        np.testing.assert_allclose(result, expected)

    def test_optimization_invariants(self):
        """Test that pre-computed invariants are used correctly"""
        s_levels = np.array([-1.0, 0.0])
        eta = 1.0
        depth = 200.0
        depth_c = 20.0
        C = np.array([0.0, 1.0])

        result = sigma.ocean_s_coordinate(s_levels, eta, depth, depth_c, C)

        # For s=-1: eta*(1+s) + depth_c*s + (depth-depth_c)*C
        #         = 1*0 + 20*(-1) + 180*0 = -20
        # For s=0: 1*1 + 20*0 + 180*1 = 181
        expected = np.array([-20.0, 181.0])
        np.testing.assert_allclose(result, expected)

    def test_broadcasting_2d(self):
        """Test broadcasting over 2D fields"""
        s_levels = np.array([-1.0, 0.0])
        eta = np.array([[0.0, 1.0]])
        depth = np.array([[100.0, 200.0]])
        depth_c = 10.0
        C = np.array([0.0, 1.0])

        result = sigma.ocean_s_coordinate(s_levels, eta, depth, depth_c, C)

        assert result.shape == (1, 2, 2)


class TestOceanSG1:
    """Test ocean_s_g1 gufunc"""

    def test_basic_computation(self):
        """Test ocean s-coordinate g1 to depth conversion"""
        s_levels = np.array([-1.0, -0.5, 0.0])
        eta = 1.0
        depth = 100.0
        depth_c = 10.0
        C = np.array([-1.0, -0.5, 0.0])

        result = sigma.ocean_s_coordinate_g1(s_levels, eta, depth, depth_c, C)

        # S = depth_c*s + (depth-depth_c)*C
        # z = S + eta*(1 + S/depth)
        # For s=-1, C=-1: S = 10*(-1) + 90*(-1) = -100
        #                 z = -100 + 1*(1 + (-100)/100) = -100 + 1*0 = -100
        # For s=-0.5, C=-0.5: S = 10*(-0.5) + 90*(-0.5) = -50
        #                     z = -50 + 1*(1 + (-50)/100) = -50 + 0.5 = -49.5
        # For s=0, C=0: S = 0, z = 0 + 1*1 = 1
        expected = np.array([-100.0, -49.5, 1.0])
        np.testing.assert_allclose(result, expected)

    def test_optimization_invariants(self):
        """Test that division is pre-computed outside loop"""
        s_levels = np.array([-1.0, 0.0])
        eta = 2.0
        depth = 150.0
        depth_c = 15.0
        C = np.array([0.0, 1.0])

        result = sigma.ocean_s_coordinate_g1(s_levels, eta, depth, depth_c, C)

        # The function should pre-compute 1/depth = 1/150
        # For s=-1, C=0: S = -15, z = -15 + 2*(1-15/150) = -15 + 2*0.9 = -13.2
        # For s=0, C=1: S = 135, z = 135 + 2*(1+135/150) = 135 + 2*1.9 = 138.8
        expected = np.array([-13.2, 138.8])
        np.testing.assert_allclose(result, expected)

    def test_broadcasting_3d(self):
        """Test broadcasting over 3D fields (time, y, x)"""
        s_levels = np.array([-1.0, 0.0])
        eta = np.random.uniform(-1, 1, (2, 3, 4))
        depth = np.random.uniform(50, 200, (3, 4))
        depth_c = 10.0
        C = np.array([0.0, 1.0])

        result = sigma.ocean_s_coordinate_g1(s_levels, eta, depth, depth_c, C)

        assert result.shape == (2, 3, 4, 2)


class TestOceanSG2:
    """Test ocean_s_g2 gufunc"""

    def test_basic_computation(self):
        """Test ocean s-coordinate g2 to depth conversion"""
        s_levels = np.array([-1.0, -0.5, 0.0])
        eta = 0.5
        depth = 100.0
        depth_c = 10.0
        C = np.array([-1.0, -0.5, 0.0])

        result = sigma.ocean_s_coordinate_g2(s_levels, eta, depth, depth_c, C)

        # S = (depth_c*s + depth*C) / (depth_c + depth)
        # z = S*(depth + eta) + eta
        # For s=-1, C=-1: S = (10*(-1) + 100*(-1))/(110) = -110/110 = -1
        #                 z = -1*100.5 + 0.5 = -100
        # For s=-0.5, C=-0.5: S = (10*(-0.5) + 100*(-0.5))/110 = -55/110 = -0.5
        #                     z = -0.5*100.5 + 0.5 = -49.75
        # For s=0, C=0: S = 0, z = 0 + 0.5 = 0.5
        expected = np.array([-100.0, -49.75, 0.5])
        np.testing.assert_allclose(result, expected)

    def test_optimization_invariants(self):
        """Test that divisions are pre-computed outside loop"""
        s_levels = np.array([-1.0, 0.0])
        eta = 1.0
        depth = 200.0
        depth_c = 20.0
        C = np.array([0.0, 1.0])

        result = sigma.ocean_s_coordinate_g2(s_levels, eta, depth, depth_c, C)

        # Pre-computed: depth+depth_c=220, 1/220, depth+eta=201
        # For s=-1, C=0: S = -20/220 = -0.090909..
        #                z = -0.090909*201 + 1 = -18.272727 + 1 = -17.272727
        # For s=0, C=1: S = 200/220 = 0.909090...
        #               z = 0.909090*201 + 1 = 182.727272 + 1 = 183.727272
        expected = np.array([-17.272727272727, 183.727272727272])
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_broadcasting_with_varying_depth_c(self):
        """Test broadcasting when depth_c varies spatially"""
        s_levels = np.array([-1.0, 0.0])
        eta = np.array([[0.0, 1.0]])
        depth = np.array([[100.0, 200.0]])
        depth_c = 10.0  # Scalar depth_c
        C = np.array([0.0, 1.0])

        result = sigma.ocean_s_coordinate_g2(s_levels, eta, depth, depth_c, C)

        assert result.shape == (1, 2, 2)


class TestEdgeCases:
    """Test edge cases and numerical stability"""

    def test_very_shallow_water(self):
        """Test with very shallow water"""
        sigma_levels = np.array([-1.0, 0.0])
        eta = 0.1
        depth = 1.0  # Very shallow

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        # Should still work correctly
        expected = np.array([-1.0, 0.1])
        np.testing.assert_allclose(result, expected)

    def test_very_deep_water(self):
        """Test with very deep water"""
        sigma_levels = np.array([-1.0, 0.0])
        eta = 0.5
        depth = 10000.0  # Very deep

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        expected = np.array([-10000.0, 0.5])
        np.testing.assert_allclose(result, expected)

    def test_negative_ssh(self):
        """Test with negative sea surface height"""
        sigma_levels = np.array([-1.0, -0.5, 0.0])
        eta = -0.5
        depth = 100.0

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        # z = eta + sigma*(eta + depth)
        # For sigma=-1: -0.5 + (-1)*99.5 = -100
        # For sigma=-0.5: -0.5 + (-0.5)*99.5 = -50.25
        # For sigma=0: -0.5
        expected = np.array([-100.0, -50.25, -0.5])
        np.testing.assert_allclose(result, expected)

    def test_single_level(self):
        """Test with single vertical level"""
        sigma_levels = np.array([0.0])
        eta = 1.0
        depth = 100.0

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        assert result.shape == (1,)
        np.testing.assert_allclose(result, [1.0])

    def test_many_levels(self):
        """Test with many vertical levels"""
        n_levels = 50
        sigma_levels = np.linspace(-1, 0, n_levels)
        eta = 0.5
        depth = 100.0

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        assert result.shape == (n_levels,)
        # Check surface and bottom
        np.testing.assert_allclose(result[0], -100.0)
        np.testing.assert_allclose(result[-1], 0.5)


class TestDataTypes:
    """Test different data types"""

    def test_float32(self):
        """Test with float32 arrays"""
        sigma_levels = np.array([-1.0, 0.0], dtype=np.float32)
        eta = np.float32(0.5)
        depth = np.float32(100.0)

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        # Result should work even if inputs are float32
        expected = np.array([-100.0, 0.5])
        np.testing.assert_allclose(result, expected, rtol=1e-6)

    def test_mixed_types(self):
        """Test with mixed input types"""
        sigma_levels = np.array([-1.0, 0.0])
        eta = 0.5  # Python float
        depth = 100  # Python int

        result = sigma.ocean_sigma_coordinate(sigma_levels, eta, depth)

        expected = np.array([-100.0, 0.5])
        np.testing.assert_allclose(result, expected)
