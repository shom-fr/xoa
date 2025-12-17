# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.sigma` module
"""
import numpy as np
import xarray as xr

from xoa import sigma


class TestSigmaCoefficients:
    """Test sigma coordinate coefficient functions"""

    def test_get_cs(self):
        nz = 3
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        thetas = 7.0
        thetab = 2.0
        cs = sigma.get_cs(sig, thetas, thetab, cs_type=None)
        assert cs.dims == sig.dims
        np.testing.assert_allclose(cs, [-1.0, -0.96983013, 0.0])


class TestAtmosphereSigma:
    """Test atmosphere sigma coordinate functions"""

    def test_atmosphere_sigma_coordinate(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ptop = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
        ps = xr.DataArray(150 * np.ones(nx), dims="nx")
        sigma.atmosphere_sigma_coordinate(sig, ptop, ps)


class TestOceanSigma:
    """Test ocean sigma coordinate functions"""

    def test_ocean_sigma_coordinate(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
        sigma.ocean_sigma_coordinate(sig, ssh, bathy)

    def test_ocean_s_coordinate(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
        hc = 10.0 + bathy * 0
        thetas = 7.0
        thetab = 2.0
        sigma.ocean_s_coordinate(sig, ssh, bathy, hc, thetas, thetab)

    def test_ocean_s_coordinate_g1(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
        hc = 10.0 + bathy * 0
        thetas = 7.0
        thetab = 2.0
        sigma.ocean_s_coordinate_g1(sig, ssh, bathy, hc, thetas, thetab)

    def test_ocean_s_coordinate_g2(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
        hc = 10.0 + bathy * 0
        thetas = 7.0
        thetab = 2.0
        sigma.ocean_s_coordinate_g2(sig, ssh, bathy, hc, thetas, thetab)
