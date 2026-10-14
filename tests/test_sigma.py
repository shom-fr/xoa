# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.sigma` module
"""
import numpy as np
import xarray as xr
import pytest

import xoa
from xoa import meta, sigma


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
    assert ds_out.z.attrs["positive"] == "up"
    np.testing.assert_allclose(ds_out.z.isel(nx=0), np.linspace(-100, 0, nz))
    assert "depth" not in ds_out
    assert ds_out.encoding.get("xoa_sigma_type") == "ocean_sigma_coordinate"


def get_staggered_sigma_dataset(nzr=4, nx=3, zeta=0.0):
    """CROCO-like dataset with s coordinates at rho and w levels"""
    s_w = np.linspace(-1, 0, nzr + 1)
    s_rho = 0.5 * (s_w[1:] + s_w[:-1])
    return xr.Dataset(
        {
            "temp": (("s_rho", "xi_rho"), np.random.randn(nzr, nx)),
            "s_rho": (
                "s_rho",
                s_rho,
                {
                    "standard_name": "ocean_s_coordinate_g2",
                    "formula_terms": "s: s_rho C: Cs_rho eta: zeta depth: h depth_c: hc",
                },
            ),
            "s_w": (
                "s_w",
                s_w,
                {
                    "standard_name": "ocean_s_coordinate_g2_at_w_location",
                    "formula_terms": "s: s_w C: Cs_w eta: zeta depth: h depth_c: hc",
                },
            ),
            "Cs_rho": ("s_rho", s_rho),
            "Cs_w": ("s_w", s_w),
            "hc": 10.0,
        },
        coords={
            "zeta": ("xi_rho", np.full(nx, zeta)),
            "h": ("xi_rho", 100 * np.ones(nx)),
        },
    )


def test_decode_sigma_staggered_vertical_locations():
    """Heights at rho and w levels must not overwrite each other"""
    ds = meta.assign_meta_specs(get_staggered_sigma_dataset(), "croco")
    s_rho, s_w = ds.s_rho.values, ds.s_w.values

    ds_out = sigma.decode_sigma(ds)

    assert "z_rho" in ds_out.coords and "z_w" in ds_out.coords
    assert "s_rho" in ds_out.z_rho.dims and "s_w" not in ds_out.z_rho.dims
    assert "s_w" in ds_out.z_w.dims and "s_rho" not in ds_out.z_w.dims
    assert "z_rho" in ds_out.temp.coords
    np.testing.assert_allclose(ds_out.z_rho.isel(xi_rho=0), 100 * s_rho)
    np.testing.assert_allclose(ds_out.z_w.isel(xi_rho=0), 100 * s_w)


def test_decode_sigma_writing_twice_raises():
    """Without the croco specs, rho and w heights would both be named z"""
    ds = meta.assign_meta_specs(get_staggered_sigma_dataset(), "default")
    with pytest.raises(xoa.exceptions.XoaSigmaError, match="written twice"):
        sigma.decode_sigma(ds)


def test_decode_sigma_ref():
    """Heights relative to the datum or to the free surface"""
    ds = meta.assign_meta_specs(get_staggered_sigma_dataset(zeta=-2.0), "croco")
    s_rho, s_w = ds.s_rho.values, ds.s_w.values

    ds_datum = sigma.decode_sigma(ds)
    np.testing.assert_allclose(ds_datum.z_w.isel(xi_rho=0), -2 + 98 * s_w)
    assert "xoa_vertical_ref" not in ds_datum.z_rho.attrs

    ds_surf = sigma.decode_sigma(ds, ref="surface")
    np.testing.assert_allclose(ds_surf.z_rho.isel(xi_rho=0), 98 * s_rho)
    np.testing.assert_allclose(ds_surf.z_w.isel(xi_rho=0), 98 * s_w)
    assert ds_surf.z_w.isel(s_w=-1).values.tolist() == [0.0] * 3
    for name in ("z_rho", "z_w"):
        assert ds_surf[name].attrs["xoa_vertical_ref"] == "surface"
        assert "standard_name" not in ds_surf[name].attrs
        assert ds_surf[name].dims == ds_datum[name].dims
    assert "z_rho" in ds_surf.temp.coords

    with pytest.raises(xoa.exceptions.XoaSigmaError, match="Invalid ref"):
        sigma.decode_sigma(ds, ref="bottom")
