# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.regrid` module
"""

import numpy as np
import xarray as xr
import pytest

from xoa import regrid
from test_interp import get_grid2locs_coords, vfunc, get_interp1d_data


def test_regrid_regrid1d():

    # Get some data
    xxi, yyi, vari, xxo, yyo = get_interp1d_data()

    # Some inits
    nz1 = xxo.shape[1]
    nx, nz0 = xxi.shape
    nt = 2
    dep0 = xr.DataArray(yyi[0], dims='nz', name='nz')
    dep1 = xr.DataArray(yyo[0], dims='nk', name='nk')
    lon = xr.DataArray(xxi[:, 0], dims='lon')
    time = xr.DataArray(np.arange(nt, dtype='d'), dims='time')

    # data:3d, coord_in:1d, coord_out:1d
    da_in = xr.DataArray(
        np.resize(vari, (nt, nx, nz0)),
        name="banana",
        dims=('time', 'lon', 'nz'),
        coords=(time, lon, dep0),
        attrs={'long_name': 'Big banana'})
    da_out = regrid.regrid1d(da_in, dep1, method="linear")
    assert da_out.dims == ("time", "lon", "nk")
    assert da_out.shape == (nt, nx, nz1)
    assert not np.isnan(da_out).all()
    assert da_out.min() >= da_in.min()
    assert da_out.max() <= da_in.max()

    # data: 3d, coord_in: 2d, coord_out: 2d
    depth_in = xr.DataArray(yyi, dims=("lon", "nz"),)
    depth_out = xr.DataArray(
        yyo, dims=("lon", "nk"),
        attrs={'standard_name': 'ocean_layer_depth'})
    del da_in["nz"]
    da_in = da_in.assign_coords(
        {"time": time, "lon": lon, "depth": depth_in})
    da_out = regrid.regrid1d(da_in, depth_out, method="linear")
    assert da_out.dims == ('time', "lon", "nk")
    assert "depth" in da_out.coords
    assert not np.isnan(da_out).all()

    # same but transposed
    da_in = da_in.transpose("time", "nz", "lon")
    da_out = regrid.regrid1d(da_in, depth_out, method="linear")
    assert da_out.dims == ('time', "nk", "lon")
    assert not np.isnan(da_out).all()


@pytest.mark.parametrize(
    "mode,expected", [
        ["no", [np.nan, 1, np.nan]],
        ["bottom", [1, 1, np.nan]],
        ["below", [1, 1, np.nan]],
        [-1, [1, 1, np.nan]],
        ['top', [np.nan, 1, 1]],
        ['both', [1, 1, 1]],
        ]
    )
def test_regrid_extrap1d(mode, expected):
    nz, ny, nx = 4, 3, 5
    zi = np.linspace(3, 10, nz)
    zi = xr.DataArray(np.arange(nz), dims="z")
    vi = xr.DataArray(np.ones((nz, ny, nx)), dims=('z', 'y', 'x'), coords={"z": zi})
    vi[:, 0] = np.nan
    vi[:, -1] = np.nan
    vi.attrs["long_name"] = "Long name"
    vi.name = "toto"
    vo = regrid.extrap1d(vi, "y", mode)
    np.testing.assert_allclose(vo.values[0, :, 0], expected)
    assert vo.name == vi.name
    assert vo.attrs == vi.attrs
    assert 'z' in vo.coords


def test_regrid_grid2loc():

    np.random.seed(0)

    # Multi-dimensional generic coordinates
    nex = 4
    nexz = 2
    nxi = 7
    nyi = 6
    nzi = 5
    nti = 4
    no = 10
    xxi, yyi, zzi, tti, xo, yo, to, zo = get_grid2locs_coords(
        nex=nex, nexz=nexz, nxi=nxi, nyi=nyi, nzi=nzi, nti=nti, no=no)
    ttidt = tti.astype("m8[us]") + np.datetime64("1950-01-01")
    todt = to.astype("m8[us]") + np.datetime64("1950-01-01")
    todt = xr.DataArray(todt, dims='time')
    xo = xr.DataArray(xo, dims="time")
    yo = xr.DataArray(yo, dims="time")
    zo = xr.DataArray(zo, dims="time")
    loc = xr.Dataset(
        coords={"time": todt, "depth": zo, "lat": yo, "lon": xo})

    # Pure 1D axes
    xi = xr.DataArray(xxi[0, 0, 0, :], dims='lon')
    yi = xr.DataArray(yyi[0, 0, :, 0], dims='lat')
    zi = xr.DataArray(zzi[0, 0, :, 0, 0], dims='depth')
    ti = xr.DataArray(ttidt[:, 0, 0, 0], dims='time')
    mi = xr.DataArray(np.arange(nex), dims='member')
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = xr.DataArray(
        np.resize(vi, (nex, )+vi.shape[1:]),
        dims=('member', 'time', 'depth', 'lat', 'lon'),
        coords={"member": mi, "time": ti, "depth": zi, "lat": yi, "lon": xi},
        attrs={'long_name': "Long name"})
    vo_truth = np.array(vfunc(to, zo.values, yo.values, xo.values))
    vo_interp = regrid.grid2loc(vi, loc)
    assert vo_interp.shape == (nex, no)
    assert vo_interp.dims == ("member", "time")
    assert "time" in vo_interp.coords
    assert "lon" in vo_interp.coords
    assert "member" in vo_interp.coords
    vo_truth[np.isnan(vo_interp[0].values)] = np.nan
    np.testing.assert_almost_equal(vo_interp[0], vo_truth)
    assert "long_name" in vo_interp.attrs
