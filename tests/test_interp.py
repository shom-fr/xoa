# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.regrid` module
"""

import numpy as np
import xarray as xr
import pytest

from xoa import interp
from test_core_interp import get_grid2locs_coords, vfunc


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
    vo_interp = interp.grid2loc(vi, loc)
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

    isodepth = interp.isoslice(depth, values, isoval, "z")
    assert isodepth == -25.0
    assert isodepth.long_name == "Depth"

    depth = np.resize(depth, (2,) + depth.shape).T
    values = np.resize(values, (2,) + values.shape).T
    depth = xr.DataArray(depth, dims=("z", "x"))
    values = xr.DataArray(values, dims=("z", "x"))
    isoval = xr.DataArray([15.0, 15.0], dims="x")
    isodepth = interp.isoslice(depth, values, isoval, "z")
    np.testing.assert_allclose(isodepth, [-25.0, -25.0])
    assert isodepth.dims == ("x",)
    assert isodepth.shape == (2,)
