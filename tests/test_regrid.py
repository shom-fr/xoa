# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.regrid` module
"""

import numpy as np
import xarray as xr
import pytest

from xoa import regrid
from test_core_regrid import get_regrid1d_data
import os
import tempfile
import xoa
from xoa.interp import Interpolator
from xoa.regrid import Regridder
from xoa.core.regrid import XYRegridder
from xoa import interp


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


def test_regrid_regrid1d_dataset():
    """regrid1d on a dataset regrids only variables with the dimension"""
    zi = xr.DataArray(np.array([0.0, 10.0, 20.0, 40.0]), dims="z", attrs={"standard_name": "depth"})
    vi = xr.DataArray(
        np.arange(12.0).reshape(3, 4),
        dims=("x", "z"),
        coords={"z": zi},
        name="temp",
        attrs={"units": "C"},
    )
    ds = xr.Dataset(
        {"temp": vi, "sal": 2 * vi, "bathy": xr.DataArray([1.0, 2.0, 3.0], dims="x")},
        attrs={"title": "test"},
    )
    zo = xr.DataArray(np.array([5.0, 15.0, 30.0]), dims="z", attrs=zi.attrs)

    out = regrid.regrid1d(ds, zo)

    assert isinstance(out, xr.Dataset)
    assert out.attrs == ds.attrs
    assert out.temp.attrs == vi.attrs
    assert out.sizes["z"] == 3
    np.testing.assert_allclose(out.z, zo)
    np.testing.assert_allclose(out.temp, regrid.regrid1d(vi, zo))
    np.testing.assert_allclose(out.sal, 2 * out.temp)
    xr.testing.assert_equal(out.bathy, ds.bathy)


def test_regrid_extrap1d_dataset():
    vi = xr.DataArray(np.array([[np.nan, 1.0, 2.0, np.nan]]), dims=("x", "z"), name="temp")
    ds = xr.Dataset({"temp": vi, "other": xr.DataArray([1.0], dims="x")}, attrs={"title": "test"})
    out = regrid.extrap1d(ds, "z", "both")
    assert isinstance(out, xr.Dataset)
    assert out.attrs == ds.attrs
    np.testing.assert_allclose(out.temp, [[1.0, 1.0, 2.0, 2.0]])
    xr.testing.assert_equal(out.other, ds.other)


class TestRegridder:
    """Test the front-end Regridder class"""

    @staticmethod
    def create_test_dataset(nx, ny, lon_range, lat_range):
        """Create a test dataset with coordinates"""
        # Create grid manually to avoid edges2bounds issue
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        ds_grid = xr.Dataset(
            {
                'lon': (
                    ['y', 'x'],
                    lon_2d,
                    {'standard_name': 'longitude', 'units': 'degrees_east'},
                ),
                'lat': (
                    ['y', 'x'],
                    lat_2d,
                    {'standard_name': 'latitude', 'units': 'degrees_north'},
                ),
                'temperature': (
                    ['y', 'x'],
                    np.random.rand(ny, nx),
                    {'long_name': 'Temperature', 'units': 'degC'},
                ),
            }
        )
        return ds_grid

    def test_regridder_initialization(self):
        """Test Regridder initialization stores the requested method"""
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        for method in ('bilinear', 'bicubic', 'conservative'):
            regridder = Regridder(src_ds, dst_ds, method=method)
            assert regridder.core_regridder.method == method

    def test_regridder_with_bias_tension(self):
        """Test Regridder with bias and tension parameters"""
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        regridder = Regridder(src_ds, dst_ds, method='bicubic', bias=0.5, tension=0.3)

        assert regridder.core_regridder.bias == 0.5
        assert regridder.core_regridder.tension == 0.3

    def test_regrid_linear_field_bilinear(self):
        """Test that bilinear regridding preserves linear fields"""
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        # Linear field
        a, b, c = 2.0, 3.0, 1.0
        src_ds['field'] = a * src_ds['lon'] + b * src_ds['lat'] + c

        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result = regridder.regrid(src_ds, skipna=False)

        assert result['field'].shape == (8, 10)
        assert 'lon' in result.coords
        assert 'lat' in result.coords

        # Expected
        expected = a * dst_ds['lon'] + b * dst_ds['lat'] + c

        # Should be close (within interpolation tolerance)
        np.testing.assert_allclose(result['field'].values, expected.values, rtol=0, atol=1e-11)

    def test_regrid_linear_field_bicubic(self):
        """Test that bicubic regridding preserves linear fields"""
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        # Linear field
        a, b, c = 2.0, 3.0, 1.0
        src_ds['field'] = a * src_ds['lon'] + b * src_ds['lat'] + c

        regridder = Regridder(src_ds, dst_ds, method='bicubic')
        result = regridder.regrid(src_ds, skipna=False)

        # Expected
        expected = a * dst_ds['lon'] + b * dst_ds['lat'] + c

        # Should be close
        np.testing.assert_allclose(result['field'].values, expected.values, rtol=0, atol=1e-9)

    def test_save_load_weights(self):
        """Test saving and loading weights"""
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        with tempfile.TemporaryDirectory() as tmpdir:
            weights_file = os.path.join(tmpdir, 'weights.nc')

            # Create and save weights
            regridder1 = Regridder(src_ds, dst_ds, method='bilinear')
            regridder1.compute_weights()
            regridder1.save_weights(weights_file)

            assert os.path.exists(weights_file)

            # Load weights
            regridder2 = Regridder(src_ds, dst_ds, method='bilinear', weights_file=weights_file)

            # Should be able to regrid
            result = regridder2.regrid(src_ds)
            assert 'temperature' in result

    def test_regrid_with_skipna(self):
        """Test regridding with skipna option"""
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        # Add some NaNs
        src_ds['temperature'].values[5:7, 5:7] = np.nan

        regridder = Regridder(src_ds, dst_ds, method='bilinear')

        # With skipna=False
        result_no_skip = regridder.regrid(src_ds, skipna=False)
        # With skipna=True
        result_skip = regridder.regrid(src_ds, skipna=True)

        # skipna=True should handle NaNs better
        assert np.sum(~np.isnan(result_skip['temperature'].values)) >= np.sum(
            ~np.isnan(result_no_skip['temperature'].values)
        )

    def test_regrid_with_mask(self):
        """Test that masking part of the source changes the regridded result.

        For bilinear/bicubic, src_mask only takes effect when combined with
        skipna=True (see XYInterpolator._apply_bilinear/_bicubic); it is a
        no-op otherwise.
        """
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        src_mask = np.ones((12, 15), dtype=bool)
        src_mask[5:7, 5:7] = False

        regridder_masked = Regridder(src_ds, dst_ds, method='bilinear', src_mask=src_mask)
        result_masked = regridder_masked.regrid(src_ds, skipna=True)

        regridder_full = Regridder(src_ds, dst_ds, method='bilinear')
        result_full = regridder_full.regrid(src_ds, skipna=True)

        assert not np.allclose(
            result_masked['temperature'].values, result_full['temperature'].values, equal_nan=True
        )

    def test_regrid_attributes_preserved(self):
        """Test that variable attributes are preserved during regridding"""
        src_ds = self.create_test_dataset(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self.create_test_dataset(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result = regridder.regrid(src_ds)

        assert result['temperature'].attrs['long_name'] == 'Temperature'
        assert result['temperature'].attrs['units'] == 'degC'


class TestRegridderIntegration:
    """Integration tests for complete regridding workflows"""

    @staticmethod
    def create_simple_grid(nx, ny, lon_range, lat_range):
        """Create a simple grid dataset"""
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        return xr.Dataset(
            {
                'lon': (
                    ['y', 'x'],
                    lon_2d,
                    {'standard_name': 'longitude', 'units': 'degrees_east'},
                ),
                'lat': (
                    ['y', 'x'],
                    lat_2d,
                    {'standard_name': 'latitude', 'units': 'degrees_north'},
                ),
            }
        )

    def test_bilinear_vs_bicubic_smooth_field(self):
        """Compare bilinear and bicubic on smooth field"""
        src_ds = self.create_simple_grid(20, 15, (-10.0, 10.0), (-5.0, 5.0))
        dst_ds = self.create_simple_grid(15, 12, (-8.0, 8.0), (-4.0, 4.0))

        # Smooth field
        src_ds['field'] = np.sin(src_ds['lon'] * np.pi / 10) * np.cos(src_ds['lat'] * np.pi / 5)

        regridder_bilinear = Regridder(src_ds, dst_ds, method='bilinear')
        result_bilinear = regridder_bilinear.regrid(src_ds)

        regridder_bicubic = Regridder(src_ds, dst_ds, method='bicubic')
        result_bicubic = regridder_bicubic.regrid(src_ds)

        # Both should produce valid results
        assert np.sum(~np.isnan(result_bilinear['field'].values)) > 0
        assert np.sum(~np.isnan(result_bicubic['field'].values)) > 0

        # Results should be similar but not identical
        valid_mask = ~np.isnan(result_bilinear['field'].values) & ~np.isnan(
            result_bicubic['field'].values
        )
        corr = np.corrcoef(
            result_bilinear['field'].values[valid_mask], result_bicubic['field'].values[valid_mask]
        )[0, 1]
        assert corr > 0.99  # High correlation for smooth data

    def test_curvilinear_grid_regridding(self):
        """Test regridding with curvilinear grids"""
        # Create rotated (curvilinear) grid manually
        nx, ny = 12, 10
        lon_range, lat_range = (0.0, 10.0), (40.0, 50.0)
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
        lon_2d, lat_2d = np.meshgrid(lon, lat)

        # Add slight rotation
        angle = np.pi / 12
        lon_rot = lon_2d * np.cos(angle) - (lat_2d - 45) * np.sin(angle)
        lat_rot = lon_2d * np.sin(angle) + (lat_2d - 45) * np.cos(angle) + 45

        src_ds = xr.Dataset(
            {
                'lon': (
                    ['y', 'x'],
                    lon_rot,
                    {'standard_name': 'longitude', 'units': 'degrees_east'},
                ),
                'lat': (
                    ['y', 'x'],
                    lat_rot,
                    {'standard_name': 'latitude', 'units': 'degrees_north'},
                ),
            }
        )
        # Ensure coordinates are set for CF conventions
        src_ds = src_ds.set_coords(['lon', 'lat'])

        dst_ds = self.create_simple_grid(10, 8, (-5.0, 5.0), (-4.0, 4.0))

        # Add data
        src_ds['field'] = xr.DataArray(np.random.rand(10, 12), dims=['y', 'x'])

        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result = regridder.regrid(src_ds)

        assert 'field' in result
        assert result['field'].shape == (8, 10)


class TestSaveLoadWeightsNetCDF:
    """NetCDF save/load round-trip tests for all three methods."""

    @staticmethod
    def _make_ds(nx, ny, lon_range, lat_range):
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        return xr.Dataset(
            {
                'lon': (
                    ['y', 'x'],
                    lon_2d,
                    {'standard_name': 'longitude', 'units': 'degrees_east'},
                ),
                'lat': (
                    ['y', 'x'],
                    lat_2d,
                    {'standard_name': 'latitude', 'units': 'degrees_north'},
                ),
                'field': (['y', 'x'], np.random.rand(ny, nx)),
            }
        )

    def _round_trip(self, method):
        from xoa import regrid as xregrid

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        with tempfile.TemporaryDirectory() as tmpdir:
            weights_file = os.path.join(tmpdir, 'weights.nc')
            r1 = Regridder(src_ds, dst_ds, method=method)
            r1.compute_weights()
            r1.save_weights(weights_file)
            assert os.path.exists(weights_file)
            result1 = r1.regrid(src_ds)

            # The weights are read from the file, and not found in the cache
            xregrid.clear_weights_cache()
            r2 = Regridder(src_ds, dst_ds, method=method, weights_file=weights_file)
            assert r2.core_regridder is not r1.core_regridder
            assert r2.core_regridder.has_weights
            result2 = r2.regrid(src_ds)
            np.testing.assert_allclose(
                result1['field'].values, result2['field'].values, equal_nan=True, rtol=1e-12
            )

    def test_save_load_bilinear(self):
        self._round_trip('bilinear')

    def test_save_load_bicubic(self):
        self._round_trip('bicubic')

    def test_save_load_conservative(self):
        self._round_trip('conservative')

    def test_weights_are_saved_in_a_group_named_after_the_grids(self, tmp_path):
        from xoa import weights

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        path = str(tmp_path / 'weights.nc')
        regridder = Regridder(src_ds, dst_ds, method='linear', weights_file=path)
        assert not os.path.exists(path)  # nothing before the weights are needed
        regridder.regrid(src_ds)
        groups = weights.list_groups(path)
        assert groups == [regridder.weights_group]
        assert groups[0].startswith('regrid_bilinear_')
        ds = weights.load_group(path, groups[0])
        assert ds.attrs['fingerprint'] == regridder.fingerprint
        assert ds.attrs['n_dst'] == 80 and ds.attrs['n_src'] == 180
        with xr.open_dataset(path) as root:
            assert root.attrs['xoa_weights_format'] == weights.FORMAT_VERSION

    def test_one_file_for_several_grids_and_methods(self, tmp_path):
        from xoa import regrid as xregrid
        from xoa import weights

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst1 = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        dst2 = self._make_ds(7, 6, (-3.0, 3.0), (-2.0, 2.0))
        path = str(tmp_path / 'weights.nc')
        cases = [(dst1, 'bilinear'), (dst2, 'bilinear'), (dst1, 'bicubic'), (dst2, 'conservative')]
        expected = []
        for dst, method in cases:
            result = Regridder(src_ds, dst, method, weights_file=path).regrid(src_ds)
            expected.append(result['field'].values)
        assert len(weights.list_groups(path)) == 4

        # In another session, all the weights are read from the single file
        xregrid.clear_weights_cache()
        for (dst, method), ref in zip(cases, expected):
            regridder = Regridder(src_ds, dst, method, weights_file=path)
            assert regridder.core_regridder.has_weights
            np.testing.assert_allclose(
                regridder.regrid(src_ds)['field'].values, ref, equal_nan=True, rtol=1e-12
            )
        assert len(weights.list_groups(path)) == 4

    def test_nothing_is_computed_when_the_file_has_the_weights(self, weight_calls, tmp_path):
        from xoa import regrid as xregrid

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        path = str(tmp_path / 'weights.nc')
        for method, counter in (('bilinear', 'frac'), ('conservative', 'conservative')):
            Regridder(src_ds, dst_ds, method, weights_file=path).regrid(src_ds)
            assert weight_calls[counter] == 1
            xregrid.clear_weights_cache()
            Regridder(src_ds, dst_ds, method, weights_file=path).regrid(src_ds)
            assert weight_calls[counter] == 1

    def test_group_is_not_written_twice(self, tmp_path):
        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        path = str(tmp_path / 'weights.nc')
        regridder = Regridder(src_ds, dst_ds, 'bilinear', weights_file=path)
        regridder.regrid(src_ds)
        mtime, size = os.path.getmtime(path), os.path.getsize(path)
        regridder.save_weights(path)
        regridder.regrid(src_ds)
        assert (os.path.getmtime(path), os.path.getsize(path)) == (mtime, size)

    def test_other_grid_with_the_same_sizes_does_not_load_wrong_weights(self, tmp_path):
        """The weights of a grid with the same sizes but other coordinates are not used"""
        from xoa import regrid as xregrid
        from xoa import weights

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        other_dst = self._make_ds(10, 8, (-3.0, 4.0), (-2.0, 3.0))  # same sizes
        path = str(tmp_path / 'weights.nc')
        Regridder(src_ds, dst_ds, 'bilinear', weights_file=path).regrid(src_ds)
        xregrid.clear_weights_cache()
        regridder = Regridder(src_ds, other_dst, 'bilinear', weights_file=path)
        assert not regridder.core_regridder.has_weights
        result = regridder.regrid(src_ds)['field'].values
        xregrid.clear_weights_cache()
        reference = Regridder(src_ds, other_dst, 'bilinear').regrid(src_ds)['field'].values
        np.testing.assert_allclose(result, reference, equal_nan=True)
        assert len(weights.list_groups(path)) == 2

    def test_explicit_load_on_other_grids_raises(self, tmp_path):
        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        path = str(tmp_path / 'weights.nc')
        Regridder(src_ds, dst_ds, 'bicubic', weights_file=path).regrid(src_ds)
        wrong_dst = self._make_ds(5, 4, (-2.0, 2.0), (-2.0, 2.0))
        with pytest.raises(ValueError, match="No weights for these grids"):
            Regridder(src_ds, wrong_dst, 'bicubic').load_weights(path)
        with pytest.raises(ValueError, match="No weights for these grids"):
            Regridder(src_ds, dst_ds, 'bilinear').load_weights(path)  # other method

    def test_wrong_fingerprint_inside_a_group_raises(self, tmp_path):
        from xoa import regrid as xregrid

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        regridder = Regridder(src_ds, dst_ds, 'bilinear')
        regridder.compute_weights()
        # A group that has the right name but the weights of other grids
        path = str(tmp_path / 'weights.nc')
        other = Regridder(src_ds, self._make_ds(10, 8, (-3.0, 4.0), (-2.0, 3.0)), 'bilinear')
        other.compute_weights()
        other.weights_group = regridder.weights_group
        other.save_weights(path)
        xregrid.clear_weights_cache()
        fresh = Regridder(src_ds, dst_ds, 'bilinear')
        with pytest.raises(ValueError, match="fingerprint"):
            fresh.load_weights(path)

    @staticmethod
    def _write_legacy_file(regridder, path):
        cr = regridder.core_regridder
        xr.Dataset(
            {
                'j_base': ('n_dst', cr._j_base),
                'i_base': ('n_dst', cr._i_base),
                'frac_a': ('n_dst', cr._frac_a),
                'frac_b': ('n_dst', cr._frac_b),
                'valid_dst_mask': ('n_dst', cr._valid_dst_mask),
            },
            attrs={
                'method': cr.method,
                'n_dst': cr.dst_grid['lat'].size,
                'n_src': cr.src_grid['lat'].size,
            },
        ).to_netcdf(path)

    def test_legacy_file_is_still_read_with_a_warning(self, tmp_path):
        from xoa import regrid as xregrid
        from xoa import weights

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        reference = Regridder(src_ds, dst_ds, 'bilinear')
        expected = reference.regrid(src_ds)['field'].values
        path = str(tmp_path / 'legacy.nc')
        self._write_legacy_file(reference, path)
        assert weights.is_legacy(path)

        xregrid.clear_weights_cache()
        with pytest.warns(xoa.XoaWarning, match="legacy"):
            regridder = Regridder(src_ds, dst_ds, 'bilinear', weights_file=path)
        assert regridder.core_regridder.has_weights
        np.testing.assert_allclose(
            regridder.regrid(src_ds)['field'].values, expected, equal_nan=True
        )
        # The weights of the grids are added to the file with their fingerprint
        assert weights.list_groups(path) == [regridder.weights_group]

    def test_legacy_file_with_other_sizes_or_method_raises(self, tmp_path):
        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        reference = Regridder(src_ds, dst_ds, 'bilinear')
        reference.compute_weights()
        path = str(tmp_path / 'legacy.nc')
        self._write_legacy_file(reference, path)
        wrong_dst = self._make_ds(5, 4, (-2.0, 2.0), (-2.0, 2.0))
        with pytest.warns(xoa.XoaWarning), pytest.raises(ValueError, match="doesn't match"):
            Regridder(src_ds, wrong_dst, 'bilinear', weights_file=path)
        with pytest.warns(xoa.XoaWarning), pytest.raises(ValueError, match="method"):
            Regridder(src_ds, dst_ds, 'bicubic').load_weights(path)

    def test_interpolators_and_regridders_share_a_file(self, tmp_path):
        from xoa import interp as xinterp
        from xoa import regrid as xregrid
        from xoa import weights

        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        path = str(tmp_path / 'weights.nc')
        lons, lats = np.linspace(-3, 3, 7), np.linspace(-2, 2, 7)
        Regridder(src_ds, dst_ds, 'bilinear', weights_file=path).regrid(src_ds)
        interpolator = Interpolator(src_ds, lons, lats, 'bilinear', weights_file=path)
        expected = interpolator.interp(src_ds)['field'].values
        kinds = sorted(group.split('_')[0] for group in weights.list_groups(path))
        assert kinds == ['interp', 'regrid']

        xregrid.clear_weights_cache()
        xinterp.clear_weights_cache()
        interpolator = Interpolator(src_ds, lons, lats, 'bilinear', weights_file=path)
        assert interpolator.core_interp.has_weights
        np.testing.assert_allclose(interpolator.interp(src_ds)['field'].values, expected)


class TestNDRegridding:
    """Verify that vectorize=False passes full ND arrays through the kernel correctly."""

    @staticmethod
    def _make_ds(nx, ny, lon_range, lat_range):
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        return xr.Dataset(
            {
                'lon': (
                    ['y', 'x'],
                    lon_2d,
                    {'standard_name': 'longitude', 'units': 'degrees_east'},
                ),
                'lat': (
                    ['y', 'x'],
                    lat_2d,
                    {'standard_name': 'latitude', 'units': 'degrees_north'},
                ),
            }
        )

    def test_3d_regrid_shape(self):
        """3D input (extra_dim, lat, lon) should produce (extra_dim, dst_lat, dst_lon)."""
        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        n_depth = 5
        rng = np.random.default_rng(3)
        src_ds['temp'] = xr.DataArray(
            rng.random((n_depth, 12, 15)),
            dims=['depth', 'y', 'x'],
        )
        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result = regridder.regrid(src_ds)
        assert result['temp'].shape == (n_depth, 8, 10)
        assert np.sum(~np.isnan(result['temp'].values)) > 0

    def test_4d_regrid_shape(self):
        """4D input (time, depth, lat, lon) → (time, depth, dst_lat, dst_lon)."""
        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        n_time, n_depth = 3, 4
        rng = np.random.default_rng(5)
        src_ds['salt'] = xr.DataArray(
            rng.random((n_time, n_depth, 12, 15)),
            dims=['time', 'depth', 'y', 'x'],
        )
        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result = regridder.regrid(src_ds)
        assert result['salt'].shape == (n_time, n_depth, 8, 10)
        assert np.sum(~np.isnan(result['salt'].values)) > 0

    def test_3d_linear_field_exact(self):
        """3D case: a field that's linear in lon/lat must be exactly preserved."""
        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        n_depth = 3
        a, b, c = 2.0, 3.0, 1.0
        # Each depth slice is the same linear field
        linear = a * src_ds['lon'] + b * src_ds['lat'] + c
        src_ds['field'] = xr.concat([linear] * n_depth, dim='depth')
        src_ds['field'] = src_ds['field'].transpose('depth', 'y', 'x')

        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result = regridder.regrid(src_ds, skipna=False)

        expected = a * dst_ds['lon'] + b * dst_ds['lat'] + c
        for d in range(n_depth):
            np.testing.assert_allclose(
                result['field'].isel(depth=d).values,
                expected.values,
                rtol=0,
                atol=1e-10,
            )

    def test_nd_slices_independent(self):
        """Each extra-dim slice must equal a 2D-only regridding of that slice."""
        src_ds = self._make_ds(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_ds = self._make_ds(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        n_time = 4
        rng = np.random.default_rng(7)
        data = rng.random((n_time, 12, 15))
        src_ds['temp'] = xr.DataArray(data, dims=['time', 'y', 'x'])

        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result_nd = regridder.regrid(src_ds)

        # Regrid each time slice independently and compare
        for t in range(n_time):
            src_slice = src_ds.isel(time=t).drop_vars('time', errors='ignore')
            result_2d = regridder.regrid(src_slice)
            np.testing.assert_allclose(
                result_nd['temp'].isel(time=t).values,
                result_2d['temp'].values,
                equal_nan=True,
                rtol=1e-12,
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_regrid_scalar_time_coord():
    """A scalar time coordinate must not trigger a temporal interpolation"""
    attrs_lon = {'standard_name': 'longitude', 'units': 'degrees_east'}
    attrs_lat = {'standard_name': 'latitude', 'units': 'degrees_north'}
    src = xr.Dataset(
        coords={
            'lon': ('lon', np.linspace(0, 5, 11), attrs_lon),
            'lat': ('lat', np.linspace(0, 4, 9), attrs_lat),
        }
    )
    dst = xr.Dataset(
        coords={
            'lon': ('lon', np.linspace(1, 4, 4), attrs_lon),
            'lat': ('lat', np.linspace(1, 3, 3), attrs_lat),
        }
    )
    time = xr.DataArray(np.datetime64('2020-01-01', 'ns'), attrs={'standard_name': 'time'})
    dst = dst.assign_coords(time=time)
    da = (src.lon + src.lat).transpose('lat', 'lon').rename('var')
    da = da.assign_coords(time=time)
    out = Regridder(src, dst, 'bilinear').regrid(da)
    assert out.shape == (3, 4)


def test_regrid_methods_choices():
    from xoa.regrid import xy_regrid_methods

    assert str(xy_regrid_methods['linear']) == 'bilinear'
    assert str(xy_regrid_methods['cubic']) == 'bicubic'
    assert str(xy_regrid_methods['conservative']) == 'conservative'
    ds = xr.Dataset(
        coords={
            'lon': ('lon', np.arange(5.0), {'standard_name': 'longitude', 'units': 'degrees_east'}),
            'lat': ('lat', np.arange(4.0), {'standard_name': 'latitude', 'units': 'degrees_north'}),
        }
    )
    assert Regridder(ds, ds, 'linear').core_regridder.method == 'bilinear'
    with pytest.raises(ValueError):
        Regridder(ds, ds, 'nearest')


def _da_and_ds():
    attrs_lon = {'standard_name': 'longitude', 'units': 'degrees_east'}
    attrs_lat = {'standard_name': 'latitude', 'units': 'degrees_north'}
    lon = xr.DataArray(np.linspace(0, 5, 11), dims='lon', attrs=attrs_lon)
    lat = xr.DataArray(np.linspace(0, 4, 9), dims='lat', attrs=attrs_lat)
    time = xr.DataArray(
        np.array(['2020-01-01', '2020-01-02', '2020-01-03'], dtype='datetime64[ns]'),
        dims='time',
        attrs={'standard_name': 'time'},
    )
    temp = (2 * lon + lat + xr.DataArray(np.arange(3.0), dims='time')).transpose(
        'time', 'lat', 'lon'
    )
    temp = temp.rename('temp').assign_coords(lon=lon, lat=lat, time=time)
    temp.attrs['units'] = 'C'
    ds = xr.Dataset({'temp': temp, 'sal': 2 * temp, 'bathy': xr.DataArray([1.0, 2.0], dims='x')})
    return temp, ds


def test_regrid_dataarray_and_dataset_agree():
    temp, ds = _da_and_ds()
    dst = xr.Dataset(
        coords={
            'lon': ('lon', np.linspace(1.2, 3.8, 5), temp.lon.attrs),
            'lat': ('lat', np.linspace(1.2, 2.8, 4), temp.lat.attrs),
        }
    )
    dst_time = temp.time[:2] + np.timedelta64(12, 'h')
    for kw in {}, {'dst_time': dst_time}:
        regridder = Regridder(ds, dst, 'bilinear')
        out_da = regridder.regrid(temp, **kw)
        out_ds = regridder.regrid(ds, **kw)
        assert isinstance(out_da, xr.DataArray)
        assert isinstance(out_ds, xr.Dataset)
        np.testing.assert_allclose(out_ds.temp, out_da)
        np.testing.assert_allclose(out_ds.sal, 2 * out_da)
        assert out_ds.temp.attrs['units'] == 'C'
        xr.testing.assert_equal(out_ds.bathy, ds.bathy)


def test_interp_dataarray_and_dataset_agree():
    temp, ds = _da_and_ds()
    lons, lats = [1.0, 2.5], [1.0, 2.0]
    interpolator = Interpolator(ds, lons, lats)
    out_da = interpolator.interp(temp)
    out_ds = interpolator.interp(ds)
    assert isinstance(out_da, xr.DataArray)
    assert isinstance(out_ds, xr.Dataset)
    np.testing.assert_allclose(out_ds.temp, out_da)

    times = np.array(['2020-01-01T12', '2020-01-02T12'], dtype='datetime64[ns]')
    out_da = interpolator.interp_with_time(temp, times)
    out_ds = interpolator.interp_with_time(ds, times)
    assert isinstance(out_da, xr.DataArray)
    assert isinstance(out_ds, xr.Dataset)
    np.testing.assert_allclose(out_ds.temp, out_da)
    np.testing.assert_allclose(out_da, 2 * np.array(lons) + np.array(lats) + np.array([0.5, 1.5]))


@pytest.fixture(autouse=True)
def clear_caches():
    """Isolate the tests from the weights that are shared in memory"""
    from xoa import interp as xinterp
    from xoa import regrid as xregrid

    xregrid.clear_weights_cache()
    xinterp.clear_weights_cache()
    yield
    xregrid.clear_weights_cache()
    xinterp.clear_weights_cache()


@pytest.fixture
def weight_calls(monkeypatch):
    """Count the calls to the kernels that compute the weights, starting from empty caches"""
    import xoa.core.interp as core_interp
    import xoa.core.regrid as core_regrid
    from xoa import interp as xinterp
    from xoa import regrid as xregrid

    xregrid.clear_weights_cache()
    xinterp.clear_weights_cache()
    calls = {"frac": 0, "conservative": 0}
    frac, conservative = core_interp.compute_frac_indices, core_regrid.compute_conservative_weights

    def count_frac(*args, **kwargs):
        calls["frac"] += 1
        return frac(*args, **kwargs)

    def count_conservative(*args, **kwargs):
        calls["conservative"] += 1
        return conservative(*args, **kwargs)

    monkeypatch.setattr(core_interp, "compute_frac_indices", count_frac)
    monkeypatch.setattr(core_regrid, "compute_conservative_weights", count_conservative)
    yield calls
    xregrid.clear_weights_cache()
    xinterp.clear_weights_cache()


def _multi_var_ds():
    temp, ds = _da_and_ds()
    ds["salt"] = 3 * temp
    ds["other"] = temp + 1
    return temp, ds


def _dst_ds(temp, lon=(1.2, 3.8), lat=(1.2, 2.8)):
    return xr.Dataset(
        coords={
            'lon': ('lon', np.linspace(*lon, 5), temp.lon.attrs),
            'lat': ('lat', np.linspace(*lat, 4), temp.lat.attrs),
        }
    )


class TestWeightsAreComputedOnce:
    @pytest.mark.parametrize(
        "method,counter",
        [("bilinear", "frac"), ("bicubic", "frac"), ("conservative", "conservative")],
    )
    def test_dataset_with_several_variables(self, weight_calls, method, counter):
        temp, ds = _multi_var_ds()
        regridder = Regridder(ds, _dst_ds(temp), method)
        assert weight_calls[counter] == 0  # lazy
        regridder.regrid(ds)
        assert weight_calls[counter] == 1
        regridder.regrid(ds)
        regridder.regrid(ds.temp)
        regridder.regrid(ds, dst_time=temp.time[:2] + np.timedelta64(12, "h"))
        assert weight_calls[counter] == 1

    def test_other_regridders_of_the_same_grids_share_the_weights(self, weight_calls):
        temp, ds = _multi_var_ds()
        Regridder(ds, _dst_ds(temp), "bilinear").regrid(ds.temp)
        # New objects with the same grids, as with the accessors or after a copy
        for var in "temp", "salt", "other":
            regridder = Regridder(
                ds[var].copy(deep=True), _dst_ds(temp).copy(deep=True), "bilinear"
            )
            regridder.regrid(ds[var])
        assert weight_calls["frac"] == 1

    def test_accessors_on_each_variable(self, weight_calls):
        xoa.register_accessors(xoa=True)
        temp, ds = _multi_var_ds()
        dst = _dst_ds(temp)
        outs = [ds[var].xoa.regrid(dst) for var in ("temp", "salt", "other")]
        assert weight_calls["frac"] == 1
        np.testing.assert_allclose(outs[1], 3 * outs[0])
        ds.xoa.regrid(dst)
        lonlat = ([1.0, 2.0], [1.0, 2.0])
        [ds[var].xoa.interp(*lonlat) for var in ("temp", "salt", "other")]
        assert weight_calls["frac"] == 2  # one more for the interpolation

    def test_what_changes_triggers_a_new_computation(self, weight_calls):
        temp, ds = _multi_var_ds()
        dst = _dst_ds(temp)
        Regridder(ds, dst, "bilinear").regrid(ds.temp)
        assert weight_calls["frac"] == 1
        Regridder(ds, dst, "bicubic").regrid(ds.temp)  # method
        assert weight_calls["frac"] == 2
        Regridder(ds, _dst_ds(temp, lon=(1.0, 3.5)), "bilinear").regrid(ds.temp)  # destination
        assert weight_calls["frac"] == 3
        shifted = ds.assign_coords(lon=ds.lon + 0.01)
        Regridder(shifted, dst, "bilinear").regrid(shifted.temp)  # source
        assert weight_calls["frac"] == 4
        mask = np.ones((9, 11), bool)
        mask[0] = False
        Regridder(ds, dst, "bilinear", src_mask=mask).regrid(ds.temp)  # mask
        assert weight_calls["frac"] == 5
        Regridder(ds, dst, "bicubic", bias=0.5).regrid(ds.temp)  # parameters
        assert weight_calls["frac"] == 6
        Regridder(ds, dst, "bilinear").regrid(ds.temp)  # back to a known one
        assert weight_calls["frac"] == 6

    def test_dimension_names_are_not_mixed_up(self, weight_calls):
        temp, ds = _multi_var_ds()
        dst = _dst_ds(temp)
        ds = ds.drop_vars("bathy")
        renamed = ds.rename(lat="y", lon="x")
        renamed["lon"] = renamed.x
        renamed["lat"] = renamed.y
        out_ref = Regridder(ds, dst, "bilinear").regrid(ds.temp)
        out_renamed = Regridder(renamed, dst, "bilinear").regrid(renamed.temp)
        np.testing.assert_allclose(out_ref, out_renamed)
        assert weight_calls["frac"] == 2

    def test_clear_weights_cache(self, weight_calls):
        from xoa import regrid as xregrid

        temp, ds = _multi_var_ds()
        dst = _dst_ds(temp)
        Regridder(ds, dst, "bilinear").regrid(ds.temp)
        xregrid.clear_weights_cache()
        Regridder(ds, dst, "bilinear").regrid(ds.temp)
        assert weight_calls["frac"] == 2

    def test_cache_is_bounded(self, weight_calls):
        from xoa import regrid as xregrid

        temp, ds = _multi_var_ds()
        for i in range(xregrid._WEIGHTS_CACHE.maxsize + 2):
            Regridder(ds, _dst_ds(temp, lon=(1.0 + 0.1 * i, 3.5)), "bilinear")
        assert len(xregrid._WEIGHTS_CACHE) == xregrid._WEIGHTS_CACHE.maxsize

    def test_weights_file_is_not_reloaded_when_known(self, weight_calls, tmp_path):
        temp, ds = _multi_var_ds()
        dst = _dst_ds(temp)
        weights_file = str(tmp_path / "weights.nc")
        Regridder(ds, dst, "bilinear", weights_file=weights_file).regrid(ds.temp)
        assert (tmp_path / "weights.nc").exists()
        first = Regridder(ds, dst, "bilinear")
        # A regridder of the same grids reads the file only if the weights are unknown
        loaded = []
        original = Regridder.load_weights
        Regridder.load_weights = lambda self, path: loaded.append(path) or original(self, path)
        try:
            Regridder(ds, dst, "bilinear", weights_file=weights_file)
            assert loaded == []
            from xoa import regrid as xregrid

            xregrid.clear_weights_cache()
            Regridder(ds, dst, "bilinear", weights_file=weights_file)
            assert loaded == [weights_file]
        finally:
            Regridder.load_weights = original
        assert first.core_regridder.has_weights
        assert weight_calls["frac"] == 1


class TestShallowCopies:
    def test_regridder_shares_the_data_of_the_grids(self):
        temp, ds = _multi_var_ds()
        dst = _dst_ds(temp)
        regridder = Regridder(ds, dst, "bilinear")
        assert np.shares_memory(regridder.ds_src_grid.temp.values, ds.temp.values)
        assert np.shares_memory(regridder.ds_dst_grid.lon.values, dst.lon.values)

    def test_regridder_init_does_not_copy_the_data(self):
        import tracemalloc

        lon = xr.DataArray(
            np.linspace(0, 5, 200),
            dims='lon',
            attrs={'standard_name': 'longitude', 'units': 'degrees_east'},
        )
        lat = xr.DataArray(
            np.linspace(0, 4, 200),
            dims='lat',
            attrs={'standard_name': 'latitude', 'units': 'degrees_north'},
        )
        big = xr.Dataset(
            {'v': (('k', 'lat', 'lon'), np.zeros((50, 200, 200)))}, coords={'lon': lon, 'lat': lat}
        )
        dst = _dst_ds(_da_and_ds()[0])
        tracemalloc.start()
        Regridder(big, dst, 'bilinear')
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < 0.1 * big.v.nbytes


class TestFingerprints:
    @staticmethod
    def get_grids():
        lon = xr.DataArray(
            np.linspace(0, 5, 11),
            dims="lon",
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )
        lat = xr.DataArray(
            np.linspace(0, 4, 9),
            dims="lat",
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )
        src = xr.Dataset(coords={"lon": lon, "lat": lat})
        dst = xr.Dataset(
            coords={
                "lon": ("lon", np.linspace(1, 4, 4), lon.attrs),
                "lat": ("lat", np.linspace(1, 3, 3), lat.attrs),
            }
        )
        return src, dst

    def test_regridder_exposes_the_fingerprints_of_the_grids(self):
        from xoa import grid as xgrid
        from xoa import misc

        src, dst = self.get_grids()
        regridder = Regridder(src, dst, "linear")
        assert regridder.src_fingerprint == xgrid.get_fingerprint(src)
        assert regridder.dst_fingerprint == xgrid.get_fingerprint(dst)
        assert regridder.fingerprint == misc.combine_fingerprints(
            regridder.src_fingerprint, regridder.dst_fingerprint
        )
        assert regridder.weights_group == f"regrid_bilinear_{regridder.fingerprint}"
        # The fingerprint of the couple depends on the order
        assert Regridder(dst, src, "linear").fingerprint != regridder.fingerprint
        assert Regridder(dst, src, "linear").src_fingerprint == regridder.dst_fingerprint

    def test_masks_are_part_of_the_fingerprint(self):
        from xoa import grid as xgrid

        src, dst = self.get_grids()
        mask = np.ones((9, 11), bool)
        mask[0] = False
        regridder = Regridder(src, dst, "linear", src_mask=mask)
        assert regridder.src_fingerprint == xgrid.get_fingerprint(src, mask=mask)
        assert regridder.src_fingerprint != Regridder(src, dst, "linear").src_fingerprint

    def test_groups_of_a_file_are_found_from_the_fingerprint_of_a_grid(self, tmp_path):
        from xoa import grid as xgrid
        from xoa import weights

        src, dst = self.get_grids()
        other = xr.Dataset(
            coords={
                "lon": ("lon", np.linspace(0.5, 3.5, 4), dst.lon.attrs),
                "lat": ("lat", np.linspace(0.5, 3.5, 3), dst.lat.attrs),
            }
        )
        data = xr.DataArray(np.zeros((9, 11)), dims=("lat", "lon"), coords=src.coords)
        path = str(tmp_path / "weights.nc")
        r1 = Regridder(src, dst, "bilinear", weights_file=path)
        r1.regrid(data)
        r2 = Regridder(src, other, "conservative", weights_file=path)
        r2.regrid(data)
        i1 = Interpolator(src, [1.0, 2.0], [1.0, 2.0], weights_file=path)
        i1.interp(data)
        # The source grid is used by all of them, and each destination by one
        found = weights.find_groups(path, xgrid.get_fingerprint(src))
        assert sorted(found) == sorted([r1.weights_group, r2.weights_group, i1.weights_group])
        assert weights.find_groups(path, xgrid.get_fingerprint(dst)) == [r1.weights_group]
        assert weights.find_groups(path, xgrid.get_fingerprint(other)) == [r2.weights_group]
        assert weights.find_groups(path, r1.fingerprint) == [r1.weights_group]
        assert weights.find_groups(path, r2.fingerprint, method="conservative") == [
            r2.weights_group
        ]
        info = {i["group"]: i for i in weights.describe_groups(path)}[r1.weights_group]
        assert info["src_fingerprint"] == r1.src_fingerprint
        assert info["dst_fingerprint"] == r1.dst_fingerprint
        assert info["fingerprint"] == r1.fingerprint
        assert info["kind"] == "regrid" and info["method"] == "bilinear"
        info = {i["group"]: i for i in weights.describe_groups(path)}[i1.weights_group]
        assert info["kind"] == "interp"
        assert info["src_fingerprint"] == i1.src_fingerprint == xgrid.get_fingerprint(src)
        assert info["dst_fingerprint"] == i1.dst_fingerprint
        assert info["fingerprint"] == i1.fingerprint


class TestSameGrid:
    @staticmethod
    def get_src():
        lon = xr.DataArray(
            np.linspace(0, 5, 11),
            dims="lon",
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )
        lat = xr.DataArray(
            np.linspace(0, 4, 9),
            dims="lat",
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )
        temp = (2 * lon + lat).transpose("lat", "lon").rename("temp")
        return temp.assign_coords(lon=lon, lat=lat)

    def test_bilinear_onto_the_same_grid_is_exact_everywhere(self):
        temp = self.get_src()
        out = Regridder(temp, temp.coords.to_dataset(), "bilinear").regrid(temp)
        assert int(out.isnull().sum()) == 0
        np.testing.assert_allclose(out, temp, atol=1e-12)

    def test_bicubic_onto_the_same_grid_needs_one_more_cell(self):
        temp = self.get_src()
        out = Regridder(temp, temp.coords.to_dataset(), "bicubic").regrid(temp)
        assert out.isnull().sum() == 2 * 11 + 2 * 7  # the outer ring
        np.testing.assert_allclose(out[1:-1, 1:-1], temp[1:-1, 1:-1], atol=1e-9)

    def test_conservative_onto_the_same_grid(self):
        temp = self.get_src()
        out = Regridder(temp, temp.coords.to_dataset(), "conservative").regrid(temp)
        assert int(out.isnull().sum()) == 0
        np.testing.assert_allclose(out, temp, atol=1e-9)

    def test_points_outside_the_grid_are_not_extrapolated(self):
        temp = self.get_src()
        dst = xr.Dataset(
            coords={
                "lon": ("lon", [-0.4, 0.0, 2.5, 5.0, 5.4], temp.lon.attrs),
                "lat": ("lat", [2.0], temp.lat.attrs),
            }
        )
        out = Regridder(temp, dst, "bilinear").regrid(temp)
        np.testing.assert_allclose(out.values[0], [np.nan, 2.0, 7.0, 12.0, np.nan])


LON_ATTRS = {"standard_name": "longitude", "units": "degrees_east"}


LAT_ATTRS = {"standard_name": "latitude", "units": "degrees_north"}


def get_series(nt=5):
    lon = xr.DataArray(np.linspace(0, 6, 13), dims="lon", attrs=LON_ATTRS)
    lat = xr.DataArray(np.linspace(0, 4, 9), dims="lat", attrs=LAT_ATTRS)
    time = xr.DataArray(
        np.datetime64("2020-01-01", "ns") + np.arange(nt) * np.timedelta64(1, "D"),
        dims="time",
        attrs={"standard_name": "time"},
    )
    days = xr.DataArray(np.arange(nt, dtype=float), dims="time")
    field = (2.0 * lon - lat + 5.0 * days).transpose("time", "lat", "lon").rename("v")
    return field.assign_coords(lon=lon, lat=lat, time=time)


class TestRegridderTimeAndAgreement:
    @pytest.mark.parametrize("method", ["bilinear", "bicubic", "conservative"])
    def test_regridder_time_interpolation_is_linear(self, method):
        series = get_series()
        dst = xr.Dataset(
            coords={
                "lon": ("lon", np.linspace(1.0, 5.0, 5), LON_ATTRS),
                "lat": ("lat", np.linspace(1.0, 3.0, 3), LAT_ATTRS),
            }
        )
        dst_time = series.time[:4] + np.timedelta64(6, "h")
        out = regrid.Regridder(series, dst, method).regrid(series, dst_time=dst_time)
        assert not out.isnull().any()
        expected_time_part = 5.0 * (np.arange(4) + 0.25)
        spatial = out - expected_time_part[:, None, None]
        # The spatial part does not depend on time, and it is exact for the linear methods
        np.testing.assert_allclose(spatial - spatial.isel(time=0), 0.0, atol=1e-9)
        if method != "conservative":
            expected = 2.0 * out.lon - out.lat
            np.testing.assert_allclose(
                spatial.isel(time=0), expected.transpose("lat", "lon"), atol=1e-9
            )

    @pytest.mark.parametrize("method", ["bilinear", "bicubic"])
    def test_regridder_and_interpolator_agree_with_the_core(self, method):
        series = get_series().isel(time=0)
        dst = xr.Dataset(
            coords={
                "lon": ("lon", np.linspace(1.0, 5.0, 6), LON_ATTRS),
                "lat": ("lat", np.linspace(1.0, 3.0, 4), LAT_ATTRS),
            }
        )
        out = regrid.Regridder(series, dst, method).regrid(series)
        lon, lat = np.meshgrid(series.lon, series.lat)
        dlon, dlat = np.meshgrid(dst.lon, dst.lat)
        core = XYRegridder({"lon": lon, "lat": lat}, {"lon": dlon, "lat": dlat}, method).regrid(
            series.values
        )
        np.testing.assert_array_equal(out.values, core)
        points = interp.Interpolator(series, dlon.ravel(), dlat.ravel(), method).interp(series)
        np.testing.assert_allclose(points.values.reshape(dlon.shape), core, atol=1e-12)


class TestRegridxy:
    """Functional interface of the horizontal regridding"""

    def setup_method(self):
        from test_interp import make_da, make_dst_ds, make_src_ds

        interp.clear_weights_cache()
        self.make_src_ds = make_src_ds
        self.make_da = make_da
        self.src = make_src_ds()
        self.da = make_da(self.src).assign_coords(lon=self.src.lon, lat=self.src.lat)
        self.dst = make_dst_ds()

    @pytest.mark.parametrize("method", ["bilinear", "bicubic", "conservative"])
    def test_same_as_the_class(self, method):
        ref = Regridder(self.da, self.dst, method).regrid(self.da)
        out = regrid.regridxy(self.da, self.dst, method)
        xr.testing.assert_identical(out, ref)

    def test_dataset(self):
        ds = self.src.assign(field=self.da.reset_coords(drop=True))
        out = regrid.regridxy(ds, self.dst)
        np.testing.assert_allclose(out["field"].values, self.dst["lon"] + self.dst["lat"])

    def test_accessor(self):
        xoa.register_accessors(xoa=True)
        out = self.da.xoa.regrid(self.dst)
        xr.testing.assert_identical(out, regrid.regridxy(self.da, self.dst))

    def test_given_regridder_is_used(self, monkeypatch):
        regridder = Regridder(self.da, self.dst, "bilinear")
        calls = []
        monkeypatch.setattr(Regridder, "__init__", lambda *a, **k: calls.append(1))
        out = regrid.regridxy(self.da, regridder=regridder)
        assert not calls
        np.testing.assert_allclose(out.values, self.dst["lon"] + self.dst["lat"])

    def test_given_regridder_must_match_the_grid(self):
        regridder = Regridder(self.da, self.dst, "bilinear")
        with pytest.raises(ValueError, match="does not match"):
            other = self.make_src_ds(nx=20)
            regrid.regridxy(
                self.make_da(other).assign_coords(lon=other.lon, lat=other.lat),
                regridder=regridder,
            )

    def test_weights_are_shared_between_calls(self):
        regrid.regridxy(self.da, self.dst)
        a = Regridder(self.da, self.dst, "bilinear")
        b = Regridder(self.da, self.dst, "bilinear")
        assert a.core_regridder is b.core_regridder
        assert a.core_regridder.has_weights

    def test_weights_file_is_written_once_then_read(self, tmp_path):
        from xoa import weights

        path = str(tmp_path / "weights.nc")
        ref = regrid.regridxy(self.da, self.dst, weights_file=path)
        groups = weights.list_groups(path)
        assert len(groups) == 1
        regrid.regridxy(self.da, self.dst, weights_file=path)
        assert weights.list_groups(path) == groups
        interp.clear_weights_cache()
        regridder = Regridder(self.da, self.dst, "bilinear", weights_file=path)
        assert regridder.core_regridder.has_weights
        np.testing.assert_allclose(regrid.regridxy(self.da, regridder=regridder), ref)

    def test_invalid_method(self):
        with pytest.raises(ValueError):
            regrid.regridxy(self.da, self.dst, method="nope")
