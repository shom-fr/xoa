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
    xxi, yyi, vari, xxo, yyo, eshapes = get_interp1d_data()

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
        attrs={'long_name': 'Big banana'},
    )
    da_out = regrid.regrid1d(da_in, dep1, method="linear")
    assert da_out.dims == ("time", "lon", "nk")
    assert da_out.shape == (nt, nx, nz1)
    assert not np.isnan(da_out).all()
    assert da_out.min() >= da_in.min()
    assert da_out.max() <= da_in.max()

    # data: 3d, coord_in: 2d, coord_out: 2d
    depth_in = xr.DataArray(
        yyi,
        dims=("lon", "nz"),
    )
    depth_out = xr.DataArray(yyo, dims=("lon", "nk"), attrs={'standard_name': 'ocean_layer_depth'})
    del da_in["nz"]
    da_in = da_in.assign_coords({"time": time, "lon": lon, "depth": depth_in})
    da_out = regrid.regrid1d(da_in, depth_out, method="linear")
    assert da_out.dims == ('time', "lon", "nk")
    assert "depth" in da_out.coords
    assert not np.isnan(da_out).all()

    # same but transposed
    da_in_t = da_in.transpose("time", "nz", "lon")
    da_out_t = regrid.regrid1d(da_in_t, depth_out, method="linear")
    assert da_out_t.dims == ('time', "nk", "lon")
    assert not np.isnan(da_out_t).all()

    # now we remove/add some dims
    depth_in_ed = da_in_t.depth.broadcast_like(da_in_t).isel(lon=0).drop("lon")
    da_in_ed = da_in_t.assign_coords({"depth": depth_in_ed})
    da_out_ed = regrid.regrid1d(da_in_ed, depth_out, method="linear")
    assert da_out_ed.dims == ('time', "nk", "lon")
    np.testing.assert_allclose(da_out_ed.isel(lon=0), da_out_t.isel(lon=0))


def test_regrid_regrid1d_time():
    time_in = xr.DataArray(np.arange("2000-01-01", "2000-01-03", dtype="M8[D]"), dims="time")
    data_in = xr.DataArray(np.arange(time_in.size), coords={"time": time_in})
    time_out = xr.DataArray(np.arange("2000-01-01", "2000-01-03", dtype="M8[h]"), dims="time")
    data_out = regrid.regrid1d(data_in, time_out)
    assert data_out.dtype.char == "d"
    assert data_out.time.dtype.str == '<M8[ns]'
    assert data_out.shape == (48,)
    assert float(data_out.max()) == 1.0


@pytest.mark.parametrize(
    "mode,expected",
    [
        ["no", [np.nan, 1, np.nan]],
        ["bottom", [1, 1, np.nan]],
        ["below", [1, 1, np.nan]],
        [-1, [1, 1, np.nan]],
        ['top', [np.nan, 1, 1]],
        ['both', [1, 1, 1]],
    ],
)
def test_regrid_extrap1d(mode, expected):
    nz, ny, nx = 4, 3, 5
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
        nex=nex, nexz=nexz, nxi=nxi, nyi=nyi, nzi=nzi, nti=nti, no=no
    )
    ttidt = tti.astype("m8[us]") + np.datetime64("1950-01-01")
    todt = to.astype("m8[us]") + np.datetime64("1950-01-01")
    todt = xr.DataArray(todt, dims='time')
    xo = xr.DataArray(xo, dims="time")
    yo = xr.DataArray(yo, dims="time")
    zo = xr.DataArray(zo, dims="time")
    loc = xr.Dataset(coords={"time": todt, "depth": zo, "lat": yo, "lon": xo})

    # Pure 1D axes
    xi = xr.DataArray(xxi[0, 0, 0, :], dims='lon')
    yi = xr.DataArray(yyi[0, 0, :, 0], dims='lat')
    zi = xr.DataArray(zzi[0, 0, :, 0, 0], dims='depth')
    ti = xr.DataArray(ttidt[:, 0, 0, 0], dims='time')
    mi = xr.DataArray(np.arange(nex), dims='member')
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = xr.DataArray(
        np.resize(vi, (nex,) + vi.shape[1:]),
        dims=('member', 'time', 'depth', 'lat', 'lon'),
        coords={"member": mi, "time": ti, "depth": zi, "lat": yi, "lon": xi},
        attrs={'long_name': "Long name"},
    )
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


def test_regrid_isoslice():

    depth = xr.DataArray(np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth"})
    values = xr.DataArray(np.linspace(10, 20.0, 6), dims="z")
    isoval = 15.0

    isodepth = regrid.isoslice(depth, values, isoval, "z")
    assert isodepth == -25.0
    assert isodepth.long_name == "Depth"

    depth = np.resize(depth, (2,) + depth.shape).T
    values = np.resize(values, (2,) + values.shape).T
    depth = xr.DataArray(depth, dims=("z", "x"))
    values = xr.DataArray(values, dims=("z", "x"))
    isoval = xr.DataArray([15.0, 15.0], dims="x")
    isodepth = regrid.isoslice(depth, values, isoval, "z")
    np.testing.assert_allclose(isodepth, [-25.0, -25.0])
    assert isodepth.dims == ("x",)
    assert isodepth.shape == (2,)
