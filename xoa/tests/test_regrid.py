# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.regrid` module
"""

import numpy as np
import xarray as xr

from xoa import regrid


def test_regrid_regrid1d():

    # Get some data
    from test_interp import get_interp1d_data
    xxi, yyi, vari, xxo, yyo = get_interp1d_data(
        yimin=-100., yimax=0., yomin=-90., yomax=10.)

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
