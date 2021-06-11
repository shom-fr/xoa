# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.grid` module
"""
import functools
import warnings
import numpy as np
import xarray as xr
import pytest

import xoa
from xoa import grid


@functools.lru_cache()
def get_da():
    x = xr.DataArray(np.arange(4), dims='x')
    y = xr.DataArray(np.arange(3), dims='y')
    lon = xr.DataArray(np.resize(x*2., (3, 4)), dims=('y', 'x'))
    lat = xr.DataArray(np.resize(y*3., (4, 3)).T, dims=('y', 'x'))
    z = xr.DataArray(np.arange(2), dims='z')
    dep = z*(lat+lon) * 100
    da = xr.DataArray(
        np.resize(lat-lon, (2, 3, 4)),
        dims=('z', 'y', 'x'),
        coords={'dep': dep, 'lat': lat, "z": z, 'lon': lon, 'y': y, 'x': x},
        attrs={"long_name": "SST"},
        name="sst")
    da.encoding.update(cfspecs='croco')
    return da


def test_grid_get_centers():

    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        dac = grid.get_centers(da, dim=("y", "x"))
    assert dac.shape == (da.shape[0], da.shape[1]-1, da.shape[2]-1)
    assert dac.x[0] == 0.5
    assert dac.y[0] == 0.5
    assert dac.lon[0, 0] == 1.
    assert dac.lat[0, 0] == 1.5
    assert dac.dep[-1, 0, 0] == 250
    assert dac[0, 0, 0] == 0.5
    assert dac.name == 'sst'
    assert dac.long_name == "SST"
    assert dac.encoding["cfspecs"] == "croco"


def test_grid_pad():

    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        warnings.filterwarnings("error", ".*ist-or-tuple.*")
        dap = grid.pad(da, {"y": 1, "x": 1}, name_kwargs={'dep': {"mode": 'edge'}})
    assert dap.shape == (da.shape[0], da.shape[1]+2, da.shape[2]+2)
    assert dap.x[0] == -1
    assert dap.x[-1] == da.sizes['x']
    assert dap.y[0] == -1
    assert dap.y[-1] == da.sizes['y']
    assert dap.lon[0, 0] == -2
    assert dap.lat[0, 0] == -3
    assert dap.dep[-1, 0, 0] == da.dep[-1, 0, 0]
    assert dap[-1, 0, 0] == da[-1, 0, 0]
    assert dap.name == 'sst'
    assert dap.long_name == "SST"
    assert dap.encoding["cfspecs"] == "croco"


def test_grid_get_edges():

    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        dae = grid.get_edges(da, "y")
    assert dae.shape == (da.shape[0], da.shape[1]+1, da.shape[2])
    np.testing.assert_allclose(dae.y[:2].values, da.y[:2] - 0.5)
    np.testing.assert_allclose(dae.lat[:2, 0], da.lat[:2, 0] - 1.5)
    assert dae[1, 0, 0] == da[1, 0, 0]


def test_grid_shift():
    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        dae = grid.shift(da, {"x": "left", "y": 1})
    assert dae.shape == da.shape
    np.testing.assert_allclose(dae.x.values, da.x - 0.5)
    np.testing.assert_allclose(dae.y.values, da.y + 0.5)
    assert dae[1, 0, 1] == float(da[1, :2, :2].mean())


@pytest.mark.parametrize(
    "positive, expected, ref, ref_type", [
        ["down", [0, 100., 600., 1600.], None, None],
        ["down", [10, 110., 610., 1610.], 10, "top"],
        ["down", [15, 115., 615., 1615.], 1615, "bottom"],
        ["up", [-1600, -1500, -1000, 0], None, None],
        ["up", [-1610, -1510, -1010, -10], 1610, "bottom"],
        ["up", [-1595, -1495, -995, 5], 5, "top"],
        ["up", [-1595, -1495, -995, 5], xr.DataArray(5, name="ssh"), "top"]
        ])
def test_grid_dz2depth(positive, expected, ref, ref_type):

    dz = xr.DataArray(
        np.resize([100, 500, 1000.], (2, 3)).T,
        dims=("z", "x"),
        coords={"z": ("z", np.arange(3, dtype="d"))}
        )

    depth = grid.dz2depth(dz, positive, ref=ref, ref_type=ref_type)
    np.testing.assert_allclose(depth.isel(x=1), expected)
    assert depth.z[0] == -0.5

    depth = grid.dz2depth(dz, positive, ref=ref, ref_type=ref_type, centered=True)
    assert depth[0, 0] == 0.5 * sum(expected[:2])
    assert depth.z[0] == 0


def test_coords_decode_cf_dz2depth():

    ds = xoa.open_data_sample("hycom.gdp.h.nc")
    ds = ds.rename(h="dz")
    dsd = grid.decode_cf_dz2depth(ds)
    assert "depth" in dsd.coords


def test_grid_torect():

    x = xr.DataArray(np.arange(4), dims='x')
    y = xr.DataArray(np.arange(3), dims='y')
    lon = xr.DataArray(np.ones((3, 4)), dims=('y', 'x'), coords={"y": y, "x": x})
    lat = xr.DataArray(np.ones((3, 4)), dims=('y', 'x'), coords={"y": y, "x": x})
    temp = xr.DataArray(
        np.ones((2, 3, 4)), dims=('time', 'y', 'x'),
        coords={'lon': lon, 'lat': lat, "y": y, "x": x})

    tempr = grid.to_rect(temp)
    assert tempr.dims == ('time', 'lat', 'lon')
    assert tempr.lon.ndim == 1
    np.testing.assert_allclose(tempr.lon.values, temp.lon[0].values)

    ds = xr.Dataset({"temp": temp})
    dsr = grid.to_rect(ds)
    assert dsr.temp.dims == tempr.dims
