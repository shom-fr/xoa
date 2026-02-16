# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.sigma` module
"""
import numpy as np
import xarray as xr
import pytest

from xoa import sigma


def test_get_cs():
    """Test sigma coordinate coefficient functions"""
    nz = 3
    sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
    thetas = 7.0
    thetab = 2.0
    cs = sigma.get_cs(sig, thetas, thetab, cs_type=None)
    assert cs.dims == sig.dims
    np.testing.assert_allclose(cs, [-1.0, -0.96983013, 0.0])


def test_atmosphere_sigma_coordinate():
    """Test atmosphere sigma coordinate functions"""
    nt, nz, nx = 2, 3, 5
    sig = xr.DataArray(np.linspace(0, 1, nz), dims="sig")
    ptop = xr.DataArray(10 * np.ones((nt, nx)), dims=("nt", "nx"))
    ps = xr.DataArray(1000 * np.ones(nx), dims="nx")
    p = sigma.atmosphere_sigma_coordinate(sig, ps, ptop)
    assert "sig" in p.dims
    assert p.sizes["sig"] == nz
    # At sigma=0: p = ptop = 10
    np.testing.assert_allclose(p.isel(sig=0), 10.)
    # At sigma=1: p = ps = 1000
    np.testing.assert_allclose(p.isel(sig=-1), 1000.)


class TestOceanSigma:
    """Test ocean sigma coordinate functions"""

    def test_ocean_sigma_coordinate(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.zeros((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(100 * np.ones(nx), dims="nx")
        depth = sigma.ocean_sigma_coordinate(sig, ssh, bathy)
        assert "sig" in depth.dims
        assert depth.sizes["sig"] == nz
        # At sigma=-1 with ssh=0: z = -bathy = -100
        np.testing.assert_allclose(depth.isel(sig=0), -100.)
        # At sigma=0 with ssh=0: z = 0
        np.testing.assert_allclose(depth.isel(sig=-1), 0.)

    def test_ocean_s_coordinate(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.zeros((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
        hc = 10.0 + bathy * 0
        thetas = 7.0
        thetab = 2.0
        depth = sigma.ocean_s_coordinate(sig, ssh, bathy, hc, thetas, thetab)
        assert "sig" in depth.dims
        assert depth.sizes["sig"] == nz
        # At surface (sigma=0): z should be 0 (with ssh=0)
        np.testing.assert_allclose(depth.isel(sig=-1), 0., atol=1e-10)

    def test_ocean_s_coordinate_g1(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.zeros((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
        hc = 10.0 + bathy * 0
        thetas = 7.0
        thetab = 2.0
        depth = sigma.ocean_s_coordinate_g1(sig, ssh, bathy, hc, thetas, thetab)
        assert "sig" in depth.dims
        assert depth.sizes["sig"] == nz
        np.testing.assert_allclose(depth.isel(sig=-1), 0., atol=1e-10)

    def test_ocean_s_coordinate_g2(self):
        nt, nz, nx = 2, 3, 5
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.zeros((nt, nx)), dims=("nt", "nx"))
        bathy = xr.DataArray(150 * np.ones(nx), dims="nx")
        hc = 10.0 + bathy * 0
        thetas = 7.0
        thetab = 2.0
        depth = sigma.ocean_s_coordinate_g2(sig, ssh, bathy, hc, thetas, thetab)
        assert "sig" in depth.dims
        assert depth.sizes["sig"] == nz
        np.testing.assert_allclose(depth.isel(sig=-1), 0., atol=1e-10)

    def test_ocean_s_with_cs(self):
        """Test providing a pre-computed stretching curve instead of thetas/thetab"""
        nz, nx = 4, 3
        sig = xr.DataArray(np.linspace(-1, 0, nz), dims="sig")
        ssh = xr.DataArray(np.zeros(nx), dims="nx")
        bathy = xr.DataArray(100 * np.ones(nx), dims="nx")
        hc = 10.0 + bathy * 0
        cs = sigma.get_cs(sig, 7.0, 2.0)
        depth = sigma.ocean_s_coordinate_g2(sig, ssh, bathy, hc, cs=cs)
        assert depth.sizes["sig"] == nz
        np.testing.assert_allclose(depth.isel(sig=-1), 0., atol=1e-10)


def test_decode_formula_terms():
    """Test parsing of formula_terms attribute"""
    terms = sigma.decode_formula_terms(
        "s: sc_r C: Cs_r eta: zeta depth: h depth_c: hc"
    )
    assert terms == {
        "s": "sc_r",
        "C": "Cs_r",
        "eta": "zeta",
        "depth": "h",
        "depth_c": "hc",
    }

    # Simple case
    terms = sigma.decode_formula_terms("sigma: sig eta: ssh depth: bathy")
    assert terms == {"sigma": "sig", "eta": "ssh", "depth": "bathy"}

    # Malformed
    with pytest.raises(Exception):
        sigma.decode_formula_terms("bad format here")


def test_decode_sigma():
    """Test full sigma decoding from a dataset"""
    nz, nx = 5, 3
    sig_values = np.linspace(-1, 0, nz)
    ssh_values = np.zeros(nx)
    bathy_values = 100 * np.ones(nx)
    data = np.random.randn(nz, nx)

    ds = xr.Dataset(
        {
            "temp": (("sig", "nx"), data),
            "sig": (
                "sig",
                sig_values,
                {
                    "standard_name": "ocean_sigma_coordinate",
                    "formula_terms": "sigma: sig eta: ssh depth: bathy",
                },
            ),
        },
        coords={
            "ssh": ("nx", ssh_values),
            "bathy": ("nx", bathy_values),
        },
    )

    ds_out = sigma.decode_sigma(ds)
    # Should have a depth coordinate added
    assert "temp" in ds_out
    depth_coord = [c for c in ds_out["temp"].coords if c not in ("sig",)]
    assert len(depth_coord) > 0
    assert ds_out.encoding.get("xoa_sigma_type") == "ocean_sigma_coordinate"
