# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.interp` module
"""

import numpy as np
import xarray as xr
import pytest

from xoa import interp
from test_core_interp import (
    get_grid2locs_coords,
    get_sheared_curvilinear_grid,
    relloc_sheared_curvilinear_grid,
    vfunc,
)
import os
import tempfile
import xoa
from xoa.interp import Interpolator
from xoa.regrid import Regridder


class TestGrid2loc:

    def test_full(self):
        """Full 4D interpolation with member, time, depth, lon, lat"""
        np.random.seed(0)

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
        todt = xr.DataArray(todt.astype("datetime64[ns]"), dims='time')
        xo = xr.DataArray(xo, dims="time")
        yo = xr.DataArray(yo, dims="time")
        zo = xr.DataArray(zo, dims="time")
        loc = xr.Dataset(coords={"time": todt, "depth": zo, "lat": yo, "lon": xo})

        xi = xr.DataArray(xxi[0, 0, 0, :], dims='lon')
        yi = xr.DataArray(yyi[0, 0, :, 0], dims='lat')
        zi = xr.DataArray(zzi[0, 0, :, 0, 0], dims='depth')
        ti = xr.DataArray(ttidt[:, 0, 0, 0].astype("datetime64[ns]"), dims='time')
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

    def test_xy_only(self):
        """Horizontal-only interpolation (no time, no depth)"""
        nxi, nyi = 5, 4
        xi = xr.DataArray(np.linspace(0, 10, nxi), dims='lon')
        yi = xr.DataArray(np.linspace(0, 10, nyi), dims='lat')
        data = np.outer(yi.values, xi.values)
        vi = xr.DataArray(
            data,
            dims=('lat', 'lon'),
            coords={"lon": xi, "lat": yi},
        )
        xo = xr.DataArray([2.5, 7.5], dims="npts")
        yo = xr.DataArray([2.5, 7.5], dims="npts")
        loc = xr.Dataset(coords={"lon": xo, "lat": yo})

        vo = interp.grid2loc(vi, loc)
        assert vo.shape == (2,)
        assert "lon" in vo.coords
        assert "lat" in vo.coords
        assert not np.isnan(vo).all()

    def test_xyz(self):
        """3D interpolation with depth (no time)"""
        nxi, nyi, nzi = 5, 4, 3
        xi = xr.DataArray(np.linspace(0, 10, nxi), dims='lon')
        yi = xr.DataArray(np.linspace(0, 10, nyi), dims='lat')
        zi = xr.DataArray(np.linspace(-100, 0, nzi), dims='depth')
        data = np.ones((nzi, nyi, nxi))
        vi = xr.DataArray(
            data,
            dims=('depth', 'lat', 'lon'),
            coords={"lon": xi, "lat": yi, "depth": zi},
        )
        xo = xr.DataArray([2.5, 7.5], dims="npts")
        yo = xr.DataArray([2.5, 7.5], dims="npts")
        zo = xr.DataArray([-50., -25.], dims="npts")
        loc = xr.Dataset(coords={"lon": xo, "lat": yo, "depth": zo})

        vo = interp.grid2loc(vi, loc)
        assert vo.shape == (2,)
        assert not np.isnan(vo).all()
        np.testing.assert_allclose(vo, [1., 1.])

    def test_curvilinear(self):
        """Horizontal-only interpolation on a genuinely curvilinear grid

        Regression test: ``lon``/``lat`` here both depend on the two
        grid dimensions (a sheared grid), unlike a curvilinear-shaped
        grid built from ``np.meshgrid`` of two independent 1D axes. It
        used to return NaN (or crash) because of a data race in the
        underlying ``closest2d`` nearest-point search.
        """
        nxi, nyi = 6, 5
        lon2d, lat2d = get_sheared_curvilinear_grid(nxi=nxi, nyi=nyi)
        ii, jj = np.meshgrid(np.arange(nxi, dtype="d"), np.arange(nyi, dtype="d"))
        data = ii + 10.0 * jj
        vi = xr.DataArray(
            data,
            dims=("y", "x"),
            coords={
                "lon": (("y", "x"), lon2d),
                "lat": (("y", "x"), lat2d),
            },
        )
        vi["lon"].attrs["standard_name"] = "longitude"
        vi["lat"].attrs["standard_name"] = "latitude"

        xo, yo = 2.3, 2.6
        loc = xr.Dataset(coords={"lon": ("npts", [xo]), "lat": ("npts", [yo])})

        vo = interp.grid2loc(vi, loc)

        i_expected, j_expected = relloc_sheared_curvilinear_grid(xo, yo)
        expected = i_expected + 10.0 * j_expected
        np.testing.assert_allclose(vo.values, [expected])


class TestIsoslice:

    def test_1d(self):
        depth = xr.DataArray(
            np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth"}
        )
        values = xr.DataArray(np.linspace(10, 20.0, 6), dims="z")
        isoval = 15.0

        isodepth = interp.isoslice(depth, values, isoval, "z")
        assert isodepth == -25.0
        assert isodepth.long_name == "Depth"

    def test_2d(self):
        depth = xr.DataArray(np.linspace(-50, 0.0, 6), dims="z")
        values = xr.DataArray(np.linspace(10, 20.0, 6), dims="z")

        depth = np.resize(depth, (2,) + depth.shape).T
        values = np.resize(values, (2,) + values.shape).T
        depth = xr.DataArray(depth, dims=("z", "x"))
        values = xr.DataArray(values, dims=("z", "x"))
        isoval = xr.DataArray([15.0, 15.0], dims="x")
        isodepth = interp.isoslice(depth, values, isoval, "z")
        np.testing.assert_allclose(isodepth, [-25.0, -25.0])
        assert isodepth.dims == ("x",)
        assert isodepth.shape == (2,)

    def test_reverse(self):
        depth = xr.DataArray(np.linspace(0.0, -50, 6), dims="z")
        values = xr.DataArray(np.linspace(20, 10.0, 6), dims="z")
        isoval = 15.0

        isodepth = interp.isoslice(depth, values, isoval, "z", reverse=True)
        np.testing.assert_allclose(isodepth, -25.0)


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


def make_src_ds(nx=40, ny=30):
    lon2d, lat2d = np.meshgrid(np.linspace(-10.0, 0.0, nx), np.linspace(40.0, 48.0, ny))
    return xr.Dataset(
        {
            'lon': xr.DataArray(
                lon2d,
                dims=['y', 'x'],
                attrs={'standard_name': 'longitude', 'units': 'degrees_east'},
            ),
            'lat': xr.DataArray(
                lat2d,
                dims=['y', 'x'],
                attrs={'standard_name': 'latitude', 'units': 'degrees_north'},
            ),
        }
    )


def make_da(src_ds, name='field'):
    return xr.DataArray(src_ds['lon'].values + src_ds['lat'].values, dims=['y', 'x'], name=name)


def make_dst_ds(nx=8, ny=6):
    lon2d, lat2d = np.meshgrid(np.linspace(-9.0, -1.0, nx), np.linspace(41.0, 47.0, ny))
    return xr.Dataset(
        {
            'lon': xr.DataArray(
                lon2d,
                dims=['y', 'x'],
                attrs={'standard_name': 'longitude', 'units': 'degrees_east'},
            ),
            'lat': xr.DataArray(
                lat2d,
                dims=['y', 'x'],
                attrs={'standard_name': 'latitude', 'units': 'degrees_north'},
            ),
        }
    )


class TestInterpolatorBasic:
    def test_1d_transect_shape(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = np.linspace(-8.0, -2.0, 20)
        dst_lat = np.linspace(41.0, 45.0, 20)
        interp = Interpolator(src_ds, dst_lon, dst_lat, method='bilinear')
        result = interp.interp(da)
        assert result.shape == (20,)
        assert np.all(np.isfinite(result.values))

    def test_values_lon_plus_lat(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = np.linspace(-8.0, -3.0, 10)
        dst_lat = np.full(10, 43.0)
        interp = Interpolator(src_ds, dst_lon, dst_lat)
        result = interp.interp(da)
        np.testing.assert_allclose(result.values, dst_lon + dst_lat, atol=0.05)

    def test_dataset_input(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = np.linspace(-8.0, -2.0, 10)
        dst_lat = np.full(10, 43.0)
        interp = Interpolator(src_ds, dst_lon, dst_lat)
        result = interp.interp(da.to_dataset())
        assert 'field' in result

    def test_bicubic(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = np.linspace(-8.0, -3.0, 10)
        dst_lat = np.full(10, 43.0)
        interp = Interpolator(src_ds, dst_lon, dst_lat, method='bicubic')
        result = interp.interp(da)
        np.testing.assert_allclose(result.values, dst_lon + dst_lat, atol=0.05)


class TestInterpolatorDims:
    def test_default_1d_dim_name(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = np.linspace(-8.0, -2.0, 10)
        dst_lat = np.full(10, 43.0)
        result = Interpolator(src_ds, dst_lon, dst_lat).interp(da)
        assert 'pts' in result.dims

    def test_default_2d_dim_names(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon, dst_lat = np.meshgrid(np.linspace(-8.0, -2.0, 5), np.linspace(41.0, 44.0, 4))
        result = Interpolator(src_ds, dst_lon, dst_lat).interp(da)
        assert 'pts_y' in result.dims and 'pts_x' in result.dims

    def test_dataarray_dst_dims_preserved(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = xr.DataArray(np.linspace(-8.0, -2.0, 10), dims=['station'])
        dst_lat = xr.DataArray(np.full(10, 43.0), dims=['station'])
        result = Interpolator(src_ds, dst_lon, dst_lat).interp(da)
        assert 'station' in result.dims

    def test_numpy_rectangular_1d_gives_2d_output(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = np.linspace(-8.0, -2.0, 5)  # nx=5
        dst_lat = np.linspace(41.0, 44.0, 4)  # ny=4, different size → rectangular
        result = Interpolator(src_ds, dst_lon, dst_lat).interp(da)
        assert result.dims == ('pts_y', 'pts_x')
        assert result.shape == (4, 5)

    def test_dataarray_rectangular_1d_gives_2d_output(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = xr.DataArray(np.linspace(-8.0, -2.0, 5), dims=['x'])
        dst_lat = xr.DataArray(np.linspace(41.0, 44.0, 4), dims=['y'])
        result = Interpolator(src_ds, dst_lon, dst_lat).interp(da)
        assert result.dims == ('y', 'x')
        assert result.shape == (4, 5)


class TestInterpolatorWeights:
    def test_save_load_netcdf(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_lon = np.linspace(-8.0, -2.0, 15)
        dst_lat = np.full(15, 43.0)

        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as f:
            fname = f.name
        os.unlink(fname)
        try:
            interp1 = Interpolator(src_ds, dst_lon, dst_lat, weights_file=fname)
            result1 = interp1.interp(da)
            assert os.path.exists(fname)

            interp2 = Interpolator(src_ds, dst_lon, dst_lat, weights_file=fname)
            result2 = interp2.interp(da)
            np.testing.assert_array_equal(result1.values, result2.values)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)


def make_timed_da(nx=40, ny=30, nt=8, name='field'):
    """Build f(lon, lat, t) = lon + lat + t: reproduced exactly by bilinear+linear."""
    lon2d, lat2d = np.meshgrid(np.linspace(-10.0, 0.0, nx), np.linspace(40.0, 48.0, ny))
    t0 = np.datetime64('2020-01-01', 'ns')
    dt_ns = int(3600e9)  # 1 h in nanoseconds
    times = np.array([t0 + i * dt_ns for i in range(nt)])
    t_float = np.arange(nt, dtype=float)
    field = lon2d[np.newaxis] + lat2d[np.newaxis] + t_float[:, np.newaxis, np.newaxis]
    return (
        xr.DataArray(
            field,
            dims=['time', 'y', 'x'],
            coords={
                'time': xr.DataArray(times, dims=['time'], attrs={'axis': 'T'}),
                'lon': xr.DataArray(
                    lon2d,
                    dims=['y', 'x'],
                    attrs={'standard_name': 'longitude', 'units': 'degrees_east'},
                ),
                'lat': xr.DataArray(
                    lat2d,
                    dims=['y', 'x'],
                    attrs={'standard_name': 'latitude', 'units': 'degrees_north'},
                ),
            },
            name=name,
        ),
        times,
        dt_ns,
        nt,
    )


def _track_times(times, dt_ns, nt, n_pts):
    t_frac = np.linspace(0.5, nt - 1.5, n_pts)
    return (
        xr.DataArray(
            np.array([times[0] + int(f * dt_ns) for f in t_frac]),
            dims=['loc'],
            name='time',
        ),
        t_frac,
    )


class TestInterpolatorWithTime:
    def test_linear_field_bilinear_exact(self):
        """f = lon + lat + t must be reproduced exactly by bilinear XY + linear T."""
        da, times, dt_ns, nt = make_timed_da()
        n_pts = 15
        dst_lon = xr.DataArray(np.linspace(-8.0, -3.0, n_pts), dims=['loc'], name='lon')
        dst_lat = xr.DataArray(np.full(n_pts, 43.0), dims=['loc'], name='lat')
        dst_times, t_frac = _track_times(times, dt_ns, nt, n_pts)

        result = Interpolator(make_src_ds(), dst_lon, dst_lat).interp_with_time(da, dst_times)

        expected = dst_lon.values + dst_lat.values + t_frac
        np.testing.assert_allclose(result.values, expected, atol=1e-6)

    def test_dataset_input(self):
        """Dataset input returns Dataset with the same variable."""
        da, times, dt_ns, nt = make_timed_da()
        n_pts = 10
        dst_lon = xr.DataArray(np.linspace(-8.0, -3.0, n_pts), dims=['loc'], name='lon')
        dst_lat = xr.DataArray(np.full(n_pts, 43.0), dims=['loc'], name='lat')
        dst_times, _ = _track_times(times, dt_ns, nt, n_pts)

        result = Interpolator(make_src_ds(), dst_lon, dst_lat).interp_with_time(
            da.to_dataset(), dst_times
        )
        assert isinstance(result, xr.Dataset)
        assert 'field' in result

    def test_output_dim_and_coords(self):
        """Output carries the destination dim and lon/lat/time coordinates."""
        da, times, dt_ns, nt = make_timed_da()
        n_pts = 10
        dst_lon = xr.DataArray(np.linspace(-8.0, -3.0, n_pts), dims=['loc'], name='lon')
        dst_lat = xr.DataArray(np.full(n_pts, 43.0), dims=['loc'], name='lat')
        dst_times, _ = _track_times(times, dt_ns, nt, n_pts)

        result = Interpolator(make_src_ds(), dst_lon, dst_lat).interp_with_time(da, dst_times)

        assert 'loc' in result.dims
        assert 'lon' in result.coords
        assert 'lat' in result.coords
        assert 'time' in result.coords

    def test_extra_depth_dim_preserved(self):
        """A depth dimension between time and spatial is preserved in output."""
        da, times, dt_ns, nt = make_timed_da()
        nz = 4
        # broadcast field to (time, depth, y, x)
        da_3d = da.expand_dims({'depth': nz}, axis=1).copy()
        n_pts = 8
        dst_lon = xr.DataArray(np.linspace(-8.0, -3.0, n_pts), dims=['loc'], name='lon')
        dst_lat = xr.DataArray(np.full(n_pts, 43.0), dims=['loc'], name='lat')
        dst_times, _ = _track_times(times, dt_ns, nt, n_pts)

        result = Interpolator(make_src_ds(), dst_lon, dst_lat).interp_with_time(da_3d, dst_times)
        assert result.dims == ('depth', 'loc')
        assert result.shape == (nz, n_pts)


class TestInterpolatorMatchesRegridder:
    def test_bilinear_matches_regridder(self):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        dst_ds = make_dst_ds()
        dst_lon = dst_ds['lon'].values
        dst_lat = dst_ds['lat'].values

        regridder = Regridder(src_ds, dst_ds, method='bilinear')
        result_regrid = regridder.regrid(da.to_dataset())['field'].values

        interp = Interpolator(src_ds, dst_lon, dst_lat, method='bilinear')
        result_interp = interp.interp(da).values

        np.testing.assert_array_equal(result_interp, result_regrid)


def test_interp_methods_choices():
    from xoa.interp import xy_interp_methods

    assert str(xy_interp_methods['linear']) == 'bilinear'
    assert str(xy_interp_methods['cubic']) == 'bicubic'
    assert str(xy_interp_methods[None]) == 'bilinear'
    src_ds = make_src_ds()
    interp = Interpolator(src_ds, [-5.0], [44.0], method='cubic')
    assert interp.core_interp.method == 'bicubic'
    with pytest.raises(ValueError):
        Interpolator(src_ds, [-5.0], [44.0], method='conservative')


def test_interpolator_weights_are_computed_once(monkeypatch):
    import xoa.core.interp as core_interp
    from xoa import interp as xinterp

    xinterp.clear_weights_cache()
    calls = {"n": 0}
    original = core_interp.compute_frac_indices

    def count(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(core_interp, "compute_frac_indices", count)
    src_ds = make_src_ds()
    da = make_da(src_ds)
    ds = xr.Dataset({"a": da, "b": 2 * da, "c": 3 * da})
    lons, lats = np.linspace(-8, -2, 10), np.linspace(41, 45, 10)
    interp = Interpolator(src_ds, lons, lats)
    interp.interp(ds)
    interp.interp(ds.a)
    interp.interp(ds)
    assert calls["n"] == 1
    # A new interpolator of the same grid and points shares the weights
    out = Interpolator(src_ds, lons.copy(), lats.copy()).interp(ds.b)
    assert calls["n"] == 1
    np.testing.assert_allclose(out, 2 * interp.interp(ds.a))
    # Other points need other weights
    Interpolator(src_ds, lons + 0.1, lats).interp(ds.b)
    assert calls["n"] == 2
    xinterp.clear_weights_cache()
    Interpolator(src_ds, lons, lats).interp(ds.b)
    assert calls["n"] == 3


class TestInterpolatorWeightsFile:
    def test_round_trip_and_one_file_for_several_points(self, tmp_path):
        from xoa import interp as xinterp
        from xoa import weights

        src_ds = make_src_ds()
        da = make_da(src_ds)
        path = str(tmp_path / "weights.nc")
        points = [
            (np.linspace(-8, -2, 10), np.linspace(41, 45, 10)),
            (np.linspace(-7, -3, 6), np.linspace(42, 44, 6)),
        ]
        expected = []
        for method, (lons, lats) in zip(("bilinear", "bicubic"), points):
            interp = Interpolator(src_ds, lons, lats, method, weights_file=path)
            assert not interp.core_interp.has_weights
            expected.append(interp.interp(da).values)
        assert len(weights.list_groups(path)) == 2
        assert all(group.startswith("interp_") for group in weights.list_groups(path))

        xinterp.clear_weights_cache()
        for method, (lons, lats), ref in zip(("bilinear", "bicubic"), points, expected):
            interp = Interpolator(src_ds, lons, lats, method, weights_file=path)
            assert interp.core_interp.has_weights  # read from the file
            np.testing.assert_allclose(interp.interp(da).values, ref, equal_nan=True)
        assert len(weights.list_groups(path)) == 2

    def test_other_points_do_not_load_wrong_weights(self, tmp_path):
        from xoa import interp as xinterp

        src_ds = make_src_ds()
        da = make_da(src_ds)
        path = str(tmp_path / "weights.nc")
        lons, lats = np.linspace(-8, -2, 10), np.linspace(41, 45, 10)
        Interpolator(src_ds, lons, lats, weights_file=path).interp(da)
        xinterp.clear_weights_cache()
        other = Interpolator(src_ds, lons + 0.3, lats, weights_file=path)
        assert not other.core_interp.has_weights
        xinterp.clear_weights_cache()
        reference = Interpolator(src_ds, lons + 0.3, lats).interp(da)
        np.testing.assert_allclose(other.interp(da).values, reference.values, equal_nan=True)

    def test_explicit_load_errors(self, tmp_path):
        src_ds = make_src_ds()
        da = make_da(src_ds)
        path = str(tmp_path / "weights.nc")
        lons, lats = np.linspace(-8, -2, 10), np.linspace(41, 45, 10)
        Interpolator(src_ds, lons, lats, weights_file=path).interp(da)
        with pytest.raises(ValueError, match="No weights for this grid"):
            Interpolator(src_ds, lons + 1, lats).load_weights(path)
        with pytest.raises(ValueError, match="No weights for this grid"):
            Interpolator(src_ds, lons, lats, "bicubic").load_weights(path)

    def test_legacy_file(self, tmp_path):
        from xoa import interp as xinterp

        src_ds = make_src_ds()
        da = make_da(src_ds)
        lons, lats = np.linspace(-8, -2, 10), np.linspace(41, 45, 10)
        reference = Interpolator(src_ds, lons, lats)
        expected = reference.interp(da).values
        w = reference.core_interp.get_weights()
        path = str(tmp_path / "legacy.nc")
        xr.Dataset(
            {
                "j_base": ("n_dst", w["j_base"]),
                "i_base": ("n_dst", w["i_base"]),
                "frac_a": ("n_dst", w["frac_a"]),
                "frac_b": ("n_dst", w["frac_b"]),
                "valid_dst_mask": ("n_dst", w["valid_dst_mask"]),
            },
            attrs={"method": "bilinear", "n_dst": 10, "n_src": src_ds.lon.size},
        ).to_netcdf(path)
        xinterp.clear_weights_cache()
        with pytest.warns(xoa.XoaWarning, match="legacy"):
            interp = Interpolator(src_ds, lons, lats, weights_file=path)
        np.testing.assert_allclose(interp.interp(da).values, expected, equal_nan=True)

    def test_wrong_fingerprint_inside_a_group_raises(self, tmp_path):
        from xoa import interp as xinterp

        src_ds = make_src_ds()
        lons, lats = np.linspace(-8, -2, 10), np.linspace(41, 45, 10)
        wanted = Interpolator(src_ds, lons, lats)
        other = Interpolator(src_ds, lons + 0.3, lats)
        other.compute_weights()
        # A group that has the right name but the weights of other points
        other.weights_group = wanted.weights_group
        path = str(tmp_path / "weights.nc")
        other.save_weights(path)
        xinterp.clear_weights_cache()
        with pytest.raises(ValueError, match="fingerprint"):
            Interpolator(src_ds, lons, lats).load_weights(path)

    def test_legacy_file_with_other_method_raises(self, tmp_path):
        src_ds = make_src_ds()
        lons, lats = np.linspace(-8, -2, 10), np.linspace(41, 45, 10)
        reference = Interpolator(src_ds, lons, lats)
        reference.compute_weights()
        w = reference.core_interp.get_weights()
        path = str(tmp_path / "legacy.nc")
        xr.Dataset(
            {
                "j_base": ("n_dst", w["j_base"]),
                "i_base": ("n_dst", w["i_base"]),
                "frac_a": ("n_dst", w["frac_a"]),
                "frac_b": ("n_dst", w["frac_b"]),
                "valid_dst_mask": ("n_dst", w["valid_dst_mask"]),
            },
            attrs={"method": "bilinear", "n_dst": 10, "n_src": src_ds.lon.size},
        ).to_netcdf(path)
        with pytest.warns(xoa.XoaWarning), pytest.raises(ValueError, match="method"):
            Interpolator(src_ds, lons, lats, "bicubic").load_weights(path)


def test_interpolator_fingerprints():
    from xoa import grid as xgrid
    from xoa import misc

    src_ds = make_src_ds()
    lons, lats = np.linspace(-8, -2, 10), np.linspace(41, 45, 10)
    interp = Interpolator(src_ds, lons, lats, "cubic")
    assert interp.src_fingerprint == xgrid.get_fingerprint(src_ds)
    assert interp.dst_fingerprint == misc.get_array_fingerprint(lons, lats)
    assert interp.fingerprint == misc.combine_fingerprints(
        interp.src_fingerprint, interp.dst_fingerprint
    )
    assert interp.weights_group == f"interp_bicubic_{interp.fingerprint}"
    other = Interpolator(src_ds, lons + 1, lats, "cubic")
    assert other.src_fingerprint == interp.src_fingerprint
    assert other.dst_fingerprint != interp.dst_fingerprint
    assert other.fingerprint != interp.fingerprint


def test_interp_with_time_keeps_the_name_of_the_array():
    times = xr.DataArray(
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"),
        dims="time",
        attrs={"standard_name": "time"},
    )
    src_ds = make_src_ds()
    base = make_da(src_ds, name="field")
    series = (base + xr.DataArray([0.0, 1.0], dims="time")).transpose("time", "y", "x")
    series = series.assign_coords(time=times)
    dst_times = np.array(["2020-01-01T12", "2020-01-02"], dtype="datetime64[ns]")
    interp = Interpolator(src_ds, [-5.0, -4.0], [43.0, 44.0])
    assert interp.interp_with_time(series.rename("field"), dst_times).name == "field"
    assert interp.interp_with_time(series.rename(None), dst_times).name is None


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


class TestInterpolatorTime:
    def test_space_time_interpolation_of_a_linear_field_is_exact(self):
        series = get_series()
        rng = np.random.default_rng(3)
        lons, lats = rng.uniform(0.2, 5.8, 50), rng.uniform(0.2, 3.8, 50)
        days = rng.uniform(0.0, 4.0, 50)
        times = np.datetime64("2020-01-01", "ns") + (days * 86400e9).astype("timedelta64[ns]")
        out = interp.Interpolator(series, lons, lats).interp_with_time(series, times)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out, 2.0 * lons - lats + 5.0 * days, atol=1e-9)

    def test_times_outside_the_range_are_nan(self):
        series = get_series()
        times = np.array(["2019-12-31", "2020-01-01", "2020-01-05", "2020-01-06"], "datetime64[ns]")
        out = interp.Interpolator(series, [1.0] * 4, [1.0] * 4).interp_with_time(series, times)
        np.testing.assert_allclose(out.values[1:3], [2.0 - 1.0, 2.0 - 1.0 + 20.0], atol=1e-9)
        assert np.isnan(out.values[[0, 3]]).all()
