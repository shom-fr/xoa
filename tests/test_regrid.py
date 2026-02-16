# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.regrid` module
"""

import numpy as np
import xarray as xr
import pytest

from xoa import regrid
from test_core_regrid import get_regrid1d_data


class TestRegrid1d:

    def test_depth_1d(self):
        """data:3d, coord_in:1d, coord_out:1d"""
        xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()
        nz1 = xxo.shape[1]
        nx, nz0 = xxi.shape
        nt = 2
        dep0 = xr.DataArray(yyi[0], dims='nz', name='nz')
        dep1 = xr.DataArray(yyo[0], dims='nk', name='nk')
        lon = xr.DataArray(xxi[:, 0], dims='lon')
        time = xr.DataArray(np.arange(nt, dtype='d'), dims='time')

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
        assert da_out.name == "banana"
        assert da_out.attrs == {'long_name': 'Big banana'}

    def test_depth_2d(self):
        """data:3d, coord_in:2d, coord_out:2d"""
        xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()
        nx, nz0 = xxi.shape
        nt = 2
        dep0 = xr.DataArray(yyi[0], dims='nz', name='nz')
        lon = xr.DataArray(xxi[:, 0], dims='lon')
        time = xr.DataArray(np.arange(nt, dtype='d'), dims='time')

        da_in = xr.DataArray(
            np.resize(vari, (nt, nx, nz0)),
            name="banana",
            dims=('time', 'lon', 'nz'),
            coords=(time, lon, dep0),
        )
        depth_in = xr.DataArray(yyi, dims=("lon", "nz"))
        depth_out = xr.DataArray(yyo, dims=("lon", "nk"),
                                 attrs={'standard_name': 'ocean_layer_depth'})
        del da_in["nz"]
        da_in = da_in.assign_coords({"time": time, "lon": lon, "depth": depth_in})
        da_out = regrid.regrid1d(da_in, depth_out, method="linear")
        assert da_out.dims == ('time', "lon", "nk")
        assert "depth" in da_out.coords
        assert not np.isnan(da_out).all()

    def test_transposed(self):
        xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()
        nx, nz0 = xxi.shape
        nt = 2
        dep0 = xr.DataArray(yyi[0], dims='nz', name='nz')
        lon = xr.DataArray(xxi[:, 0], dims='lon')
        time = xr.DataArray(np.arange(nt, dtype='d'), dims='time')

        da_in = xr.DataArray(
            np.resize(vari, (nt, nx, nz0)),
            dims=('time', 'lon', 'nz'),
            coords=(time, lon, dep0),
        )
        depth_in = xr.DataArray(yyi, dims=("lon", "nz"))
        depth_out = xr.DataArray(yyo, dims=("lon", "nk"),
                                 attrs={'standard_name': 'ocean_layer_depth'})
        del da_in["nz"]
        da_in = da_in.assign_coords({"time": time, "lon": lon, "depth": depth_in})

        da_in_t = da_in.transpose("time", "nz", "lon")
        da_out_t = regrid.regrid1d(da_in_t, depth_out, method="linear")
        assert da_out_t.dims == ('time', "nk", "lon")
        assert not np.isnan(da_out_t).all()

    def test_extra_dims(self):
        xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()
        nx, nz0 = xxi.shape
        nt = 2
        dep0 = xr.DataArray(yyi[0], dims='nz', name='nz')
        lon = xr.DataArray(xxi[:, 0], dims='lon')
        time = xr.DataArray(np.arange(nt, dtype='d'), dims='time')

        da_in = xr.DataArray(
            np.resize(vari, (nt, nx, nz0)),
            dims=('time', 'lon', 'nz'),
            coords=(time, lon, dep0),
        )
        depth_in = xr.DataArray(yyi, dims=("lon", "nz"))
        depth_out = xr.DataArray(yyo, dims=("lon", "nk"),
                                 attrs={'standard_name': 'ocean_layer_depth'})
        del da_in["nz"]
        da_in = da_in.assign_coords({"time": time, "lon": lon, "depth": depth_in})

        da_in_t = da_in.transpose("time", "nz", "lon")
        da_out_t = regrid.regrid1d(da_in_t, depth_out, method="linear")

        depth_in_ed = da_in_t.depth.broadcast_like(da_in_t).isel(lon=0).drop_vars("lon")
        da_in_ed = da_in_t.assign_coords({"depth": depth_in_ed})
        da_out_ed = regrid.regrid1d(da_in_ed, depth_out, method="linear")
        assert da_out_ed.dims == ('time', "nk", "lon")
        np.testing.assert_allclose(da_out_ed.isel(lon=0), da_out_t.isel(lon=0))

    def test_time(self):
        time_in = xr.DataArray(
            np.arange("2000-01-01", "2000-01-03", dtype="M8[D]").astype("M8[ns]"),
            dims="time")
        data_in = xr.DataArray(np.arange(time_in.size), coords={"time": time_in})
        time_out = xr.DataArray(
            np.arange("2000-01-01", "2000-01-03", dtype="M8[h]").astype("M8[ns]"),
            dims="time")
        data_out = regrid.regrid1d(data_in, time_out)
        assert data_out.dtype.char == "d"
        assert "M8" in data_out.time.dtype.str
        assert data_out.shape == (48,)
        assert float(data_out.max()) == 1.0

    @pytest.mark.parametrize("method", ["linear", "cubic", "hermit", "nearest"])
    @pytest.mark.parametrize(
        "mode,expected",
        [
            ["both", [1, 1, 1]],
            ["top", [np.nan, 1, 1]],
            ["no", [np.nan, 1, np.nan]],
            ["bottom", [1, 1, np.nan]],
        ],
    )
    def test_extrap(self, method, mode, expected):
        nz, ny, nx = 2, 1, 1
        zi = xr.DataArray([1, 2], dims="z")
        zo = xr.DataArray([0, 1.5, 3], dims="z")
        vi = xr.DataArray(np.ones((nz, ny, nx)), dims=('z', 'y', 'x'), coords={"z": zi})
        vo = regrid.regrid1d(vi, zo, dim="z", method=method, extrap=mode)
        np.testing.assert_allclose(vo.values[:, 0, 0], expected)

    def test_cellave(self):
        zi = xr.DataArray([0., 1., 2., 3.], dims="z")
        zo = xr.DataArray([0., 1.5, 3.], dims="z")
        vi = xr.DataArray(
            np.array([1., 2., 3.]).reshape(3, 1),
            dims=('z', 'x'),
            coords={"z": zi[:-1]},
        )
        vo = regrid.regrid1d(vi, zo, dim="z", method="cellave")
        assert vo.dims == ('z', 'x')
        assert vo.shape == (3, 1)
        assert not np.isnan(vo).all()

    def test_coord_in_name(self):
        nz = 4
        zi = xr.DataArray(np.arange(nz, dtype="d"), dims='z')
        zo = xr.DataArray(np.linspace(0, nz - 1, 7), dims='z')
        vi = xr.DataArray(
            np.arange(nz, dtype="d"),
            dims='z',
            coords={"z": zi, "mydepth": ("z", np.arange(nz, dtype="d"))},
        )
        vo = regrid.regrid1d(vi, zo, dim="z", coord_in_name="mydepth")
        assert vo.shape == (7,)
        assert not np.isnan(vo).all()

    def test_drop_na(self):
        zi = xr.DataArray(np.arange(5, dtype="d"), dims='z')
        zo = xr.DataArray(np.linspace(0, 4, 9), dims='z')
        data = np.array([1., np.nan, np.nan, np.nan, 5.])
        vi = xr.DataArray(data, dims='z', coords={"z": zi})
        vo = regrid.regrid1d(vi, zo, dim="z", drop_na=True)
        assert not np.isnan(vo).all()
        assert float(vo[0]) == 1.
        assert float(vo[-1]) == 5.


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
