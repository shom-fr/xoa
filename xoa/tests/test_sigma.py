# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.sigma` module
"""
import numpy as np
import xarray as xr

import xoa
from xoa import sigma


def test_sigma_get_cs():

    nz = 3
    sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
    thetas = 7.0
    thetab = 2.0
    cs = sigma.get_cs(sig, thetas, thetab, cs_type=None)
    assert cs.dims == sig.dims
    np.testing.assert_allclose(cs, [-1.0, -0.96983013, 0.0])


def test_sigma_atmosphere_sigma_coordinate():
    nt, nz, nx = 2, 3, 5
    sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
    ptop = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
    ps = xr.DataArray(150 * np.ones(nx), dims="nx")
    sigma.atmosphere_sigma_coordinate(sig, ptop, ps)


def test_ocean_sigma_coordinate():
    nt, nz, nx = 2, 3, 5
    sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
    ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
    bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
    sigma.ocean_sigma_coordinate(sig, ssh, bathy)


def test_ocean_s_coordinate():
    nt, nz, nx = 2, 3, 5
    sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
    ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
    bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
    hc = 10.0 + bathy * 0
    thetas = 7.0
    thetab = 2.0
    sigma.ocean_s_coordinate(sig, ssh, bathy, hc, thetas, thetab)


def test_ocean_s_coordinate_g1():
    nt, nz, nx = 2, 3, 5
    sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
    ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
    bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
    hc = 10.0 + bathy * 0
    thetas = 7.0
    thetab = 2.0
    sigma.ocean_s_coordinate_g1(sig, ssh, bathy, hc, thetas, thetab)


def test_ocean_s_coordinate_g2():
    nt, nz, nx = 2, 3, 5
    sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
    ssh = xr.DataArray(np.ones((nt, nx)), dims=("nt", "nx"))
    bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
    hc = 10.0 + bathy * 0
    thetas = 7.0
    thetab = 2.0
    sigma.ocean_s_coordinate_g2(sig, ssh, bathy, hc, thetas, thetab)


def test_sigma_decode_cf_sigma():

    ds = xoa.open_data_sample("croco.south-africa.meridional.nc")
    dsd = sigma.decode_cf_sigma(ds)

    assert "depth" in dsd
    assert dsd.depth.dims == ('time', 's_rho', 'eta_rho', 'xi_rho')
    assert dsd.depth.shape == (1, 32, 56, 1)
