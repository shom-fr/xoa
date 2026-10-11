# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.dyn` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import dyn
from xoa import grid as xgrid


def test_get_sea_level():
    """Test get_sea_level function"""
    data = np.random.randn(3, 4)

    # Found by variable name
    ds = xr.Dataset({"ssh": (("lat", "lon"), data)})
    result = dyn.get_sea_level(ds)
    assert result is not None
    assert result.name == "ssh"

    # Found by standard_name attribute
    ds = xr.Dataset({"my_var": (("lat", "lon"), data,
                                {"standard_name": "sea_surface_height"})})
    result = dyn.get_sea_level(ds)
    assert result is not None
    assert result.name == "my_var"

    # Found with variant filter
    ds = xr.Dataset({
        "ssh": (("lat", "lon"), data),
        "sla": (("lat", "lon"), data),
    })
    result = dyn.get_sea_level(ds, variant="sla")
    assert result is not None
    assert result.name == "sla"

    # Not found
    ds = xr.Dataset({"temperature": (("lat", "lon"), data)})
    result = dyn.get_sea_level(ds)
    assert result is None


def test_flow2d():
    """Test flow2d function"""
    lon = xr.DataArray(np.arange(4), dims='lon')
    lat = xr.DataArray(np.arange(3), dims='lat')
    u = xr.DataArray(lat*lon/3, dims=('lat', 'lon'), coords={'lon': lon, 'lat': lat})
    v = xr.DataArray((lat+lon)/2, dims=('lat', 'lon'), coords={'lon': lon, 'lat': lat})

    ff = dyn.flow2d(
        u, v, ([1., 2.], [1., 1.5]), np.timedelta64(3, "h"), np.timedelta64(2, "h"),
        date="2000-01-01")
    assert list(ff.coords) == ['lon', 'lat', 'time']
    assert ff.lon.dims == ('time', 'particles')
    assert ff.lon.shape == (3, 2)
    assert 'lat' in ff.coords
    np.testing.assert_allclose(
        ff.lon.values.ravel(),
        [1., 2., 1.0148674, 2.04497533, 1.02261437, 2.06865695])
    assert ff.time.values[0] == np.datetime64("2000-01-01")


def _velocity_ds(nt=0, ny=20, nx=30, res=5000.0):
    """Solid body rotation on a regular grid with a 5 km step"""
    lat0 = 45.0
    dlat = np.degrees(res / 6371e3)
    lat = xr.DataArray(
        lat0 + dlat * np.arange(ny), dims="lat", attrs={"standard_name": "latitude"}, name="lat"
    )
    dlon = dlat / np.cos(np.radians(lat0 + dlat * ny / 2))
    lon = xr.DataArray(
        dlon * np.arange(nx), dims="lon", attrs={"standard_name": "longitude"}, name="lon"
    )
    return lon, lat


RES = 5000.0


def _rotation(omega=1e-5, nt=0):
    """Solid body rotation on a grid with a constant step in meters"""
    lon, lat = _velocity_ds()
    x = RES * (np.arange(lon.size) - lon.size / 2)
    y = RES * (np.arange(lat.size) - lat.size / 2)
    u = xr.DataArray(-omega * y[:, None] + 0 * x, dims=("lat", "lon"), name="u")
    v = xr.DataArray(omega * x + 0 * y[:, None], dims=("lat", "lon"), name="v")
    u = u.assign_coords(lon=lon, lat=lat)
    v = v.assign_coords(lon=lon, lat=lat)
    if nt:
        u = u.expand_dims(time=nt).transpose("time", "lat", "lon")
        v = v.expand_dims(time=nt).transpose("time", "lat", "lon")
    return u, v


class TestKinematics:
    omega = 1e-5

    def test_solid_body_rotation(self):
        u, v = _rotation(self.omega)
        kw = dict(dx=RES, dy=RES)
        rv = dyn.get_relvort(u, v, **kw)
        assert rv.dims == u.dims
        assert rv.name == "relvort"
        assert rv.attrs["units"] == "s-1"
        np.testing.assert_allclose(rv, 2 * self.omega)
        np.testing.assert_allclose(dyn.get_div(u, v, **kw), 0, atol=1e-12)
        ow = dyn.get_okuboweiss(u, v, **kw)
        assert ow.attrs["units"] == "s-2"
        np.testing.assert_allclose(ow, -4 * self.omega**2)
        sn, ss = dyn.get_strain(u, v, **kw)
        assert sn.name == "strain_normal" and ss.name == "strain_shear"
        np.testing.assert_allclose(sn, 0, atol=1e-12)
        np.testing.assert_allclose(ss, 0, atol=1e-12)

    def test_resolution_from_coordinates(self):
        # The step is only almost constant, since the longitude step is constant
        u, v = _rotation(self.omega)
        np.testing.assert_allclose(dyn.get_relvort(u, v)[2:-2, 2:-2], 2 * self.omega, rtol=0.05)

    def test_explicit_resolution(self):
        u, v = _rotation(self.omega)
        dx, dy = xgrid.get_dx_dy(u)
        np.testing.assert_allclose(dyn.get_relvort(u, v, dx=dx, dy=dy), dyn.get_relvort(u, v))
        # Only one of them
        rv = dyn.get_relvort(u, v, dx=RES)
        np.testing.assert_allclose(rv, dyn.get_relvort(u, v, dx=RES, dy=dy))

    def test_extra_dim_and_transpose(self):
        u, v = _rotation(self.omega, nt=3)
        rv = dyn.get_relvort(u, v, dx=RES, dy=RES)
        assert rv.dims == ("time", "lat", "lon")
        rvt = dyn.get_relvort(
            u.transpose("lon", "lat", "time"), v.transpose("lon", "lat", "time"), dx=RES, dy=RES
        )
        assert rvt.dims == ("lon", "lat", "time")
        np.testing.assert_allclose(rvt.transpose(*rv.dims), rv)

    def test_dask(self):
        pytest.importorskip("dask")
        u, v = _rotation(self.omega, nt=3)
        ref = dyn.get_okuboweiss(u, v)
        res = dyn.get_okuboweiss(u.chunk(time=1), v.chunk(time=1))
        assert res.chunks is not None
        np.testing.assert_allclose(res.compute(), ref)

    def test_nan(self):
        u, v = _rotation(self.omega)
        u[5, 5] = np.nan
        assert np.isnan(dyn.get_relvort(u, v)[5, 5])
        assert not np.isnan(dyn.get_relvort(u, v)[10, 10])


class TestGeostrophy:
    def test_get_coriolis(self):
        lat = xr.DataArray([0.0, 30.0, 90.0], dims="lat")
        f = dyn.get_coriolis(lat)
        assert f.attrs["units"] == "s-1"
        np.testing.assert_allclose(f, [0, 2 * dyn.cdyn.OMEGA * 0.5, 2 * dyn.cdyn.OMEGA])
        assert np.isclose(dyn.get_coriolis(30.0), f[1])

    def test_get_geos(self):
        lon, lat = _velocity_ds()
        ds = xr.Dataset(coords={"lon": lon, "lat": lat})
        _, dy = xgrid.get_dx_dy(ds)
        slope = 1e-6
        ssh = (slope * dy.cumsum("lat")).transpose("lat", "lon").assign_coords(lon=lon, lat=lat)
        ug, vg = dyn.get_geos(ssh)
        assert ug.dims == ssh.dims
        assert ug.attrs["units"] == "m s-1"
        f = dyn.get_coriolis(lat)
        ref = -dyn.cdyn.GRAVITY * slope / f
        np.testing.assert_allclose(ug[:, 10], ref, rtol=1e-2)
        np.testing.assert_allclose(vg, 0, atol=1e-8)


def test_coordinate_attributes_are_kept():
    """The longitudes and latitudes must still be found in the outputs"""
    u, v = _rotation()
    for res in (dyn.get_relvort(u, v), *dyn.get_strain(u, v), *dyn.get_geos(u)):
        assert res.lon.attrs == u.lon.attrs
        assert res.lat.attrs == u.lat.attrs
        assert xgrid.get_dx_dy(res)[0].shape == u.shape
        assert "standard_name" not in res.attrs
