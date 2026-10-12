# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.dyn` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import dyn
from xoa import grid as xgrid
from xoa.core import modes as cmodes
from xoa.exceptions import XoaError


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


class TestCyclogeos:
    @staticmethod
    def get_ssh(amp=0.25, nt=0):
        lon, lat = _velocity_ds(ny=40, nx=40)
        lat = lat + 0  # 45N, where the Coriolis parameter is about 1e-4
        xx, yy = np.meshgrid(np.arange(40) - 20.0, np.arange(40) - 20.0)
        eta = amp * np.exp(-(xx**2 + yy**2) / (2 * 8.0**2))
        ssh = xr.DataArray(eta, dims=("lat", "lon"), coords={"lon": lon, "lat": lat}, name="ssh")
        if nt:
            ssh = (ssh * xr.DataArray(np.linspace(0.5, 1, nt), dims="time")).transpose(
                "time", "lat", "lon"
            )
        return ssh

    def test_values_and_attributes(self):
        ssh = self.get_ssh()
        uc, vc = dyn.get_cyclogeos(ssh, terms="full")
        assert uc.dims == vc.dims == ssh.dims
        assert uc.name == "ucgeos" and vc.name == "vcgeos"
        assert uc.attrs["units"] == "m s-1"
        assert uc.lon.attrs == ssh.lon.attrs
        ug, vg = dyn.get_geos(ssh)
        # An anticyclone is faster than what geostrophy says
        assert float(np.hypot(uc, vc).max()) > 1.02 * float(np.hypot(ug, vg).max())
        # The currents have the same direction
        np.testing.assert_allclose(
            np.sign(uc.where(abs(ug) > 0.1)), np.sign(ug.where(abs(ug) > 0.1)), equal_nan=True
        )
        cyclone = dyn.get_cyclogeos(-ssh, terms="full")
        # With the default terms, the speed changes where the currents are not along an axis
        assert float(np.hypot(*dyn.get_cyclogeos(-ssh)).sum()) < float(np.hypot(ug, vg).sum())
        assert float(np.hypot(*cyclone).max()) < 0.98 * float(np.hypot(ug, vg).max())

    def test_parameters_and_resolution(self):
        ssh = self.get_ssh()
        ref = dyn.get_cyclogeos(ssh)
        dx, dy = xgrid.get_dx_dy(ssh)
        np.testing.assert_allclose(dyn.get_cyclogeos(ssh, dx=dx, dy=dy)[0], ref[0])
        assert dyn.get_cyclogeos(ssh, dx=RES, dy=RES)[0].shape == ssh.shape
        few = dyn.get_cyclogeos(ssh, niter=1, tol=1e-12)
        assert int(np.isnan(few[0]).sum()) > int(np.isnan(ref[0]).sum())
        np.testing.assert_allclose(
            dyn.get_cyclogeos(ssh, relax=1.0, tol=1e-8)[0],
            dyn.get_cyclogeos(ssh, relax=0.5, tol=1e-8, niter=1000)[0],
            atol=1e-5,
        )

    def test_extra_dim_transpose_and_dask(self):
        ssh = self.get_ssh(nt=3)
        uc, vc = dyn.get_cyclogeos(ssh)
        assert uc.dims == ("time", "lat", "lon")
        np.testing.assert_allclose(uc[1], dyn.get_cyclogeos(ssh[1])[0], equal_nan=True)
        uct, _ = dyn.get_cyclogeos(ssh.transpose("lon", "lat", "time"))
        assert uct.dims == ("lon", "lat", "time")
        np.testing.assert_allclose(uct.transpose(*uc.dims), uc, equal_nan=True)
        pytest.importorskip("dask")
        ucd, _ = dyn.get_cyclogeos(ssh.chunk(time=1))
        assert ucd.chunks is not None
        np.testing.assert_allclose(ucd.compute(), uc, equal_nan=True)

    def test_equator(self):
        ssh = self.get_ssh()
        ssh = ssh.assign_coords(lat=ssh.lat - 45.0 + 0.5)
        ssh.lat.attrs["standard_name"] = "latitude"
        uc, _ = dyn.get_cyclogeos(ssh)
        assert np.isnan(uc).all()


def test_get_geos_with_scalar_resolutions():
    lon, lat = _velocity_ds()
    ssh = xr.DataArray(
        1e-6 * RES * np.arange(20.0)[:, None] + np.zeros(30),
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
    )
    ug, vg = dyn.get_geos(ssh, dx=RES, dy=RES)
    assert ug.dims == ssh.dims
    np.testing.assert_allclose(vg, 0, atol=1e-10)
    f = dyn.get_coriolis(lat).values[:, None]
    np.testing.assert_allclose(ug.values, -dyn.cdyn.GRAVITY * 1e-6 / f + 0 * ug.values, rtol=1e-6)


class TestVariableZ:
    omega = 1e-5
    res = 2000.0

    def get_data(self, nt=0):
        """Rotation with a vertical shear on sloping levels, at 45N where f is about 1e-4"""
        nz, ny, nx = 8, 18, 22
        lon, lat = _velocity_ds(ny=ny, nx=nx)
        x = (np.arange(nx) - nx / 2) * self.res
        y = (np.arange(ny) - ny / 2) * self.res
        xx, yy = np.meshgrid(x, y)
        depth = 200.0 + 60.0 * xx / x.max() + 40.0 * yy / y.max()
        z = np.linspace(-1, 0, nz)[:, None, None] * depth
        dims = ("depth", "lat", "lon")
        coords = {"lon": lon, "lat": lat}
        u = xr.DataArray(-self.omega * yy + 3e-4 * z, dims=dims, coords=coords, name="u")
        v = xr.DataArray(self.omega * xx - 1.5e-4 * z, dims=dims, coords=coords, name="v")
        zz = xr.DataArray(z, dims=dims, coords=coords, name="z")
        if nt:
            u = u.expand_dims(time=nt).transpose("time", *dims)
            v = v.expand_dims(time=nt).transpose("time", *dims)
        return u, v, zz

    def test_constant_z_derivatives(self):
        u, v, z = self.get_data()
        kw = {"dx": self.res, "dy": self.res}
        vort = dyn.get_relvort(u, v, z=z, **kw)
        assert vort.dims == u.dims
        np.testing.assert_allclose(vort, 2 * self.omega, rtol=1e-9)
        np.testing.assert_allclose(dyn.get_div(u, v, z=z, **kw), 0, atol=1e-14)
        np.testing.assert_allclose(
            dyn.get_okuboweiss(u, v, z=z, **kw), -4 * self.omega**2, rtol=1e-9
        )
        sn, ss = dyn.get_strain(u, v, z=z, **kw)
        np.testing.assert_allclose(sn, 0, atol=1e-14)
        # Along the levels, it is wrong
        assert float(abs(dyn.get_relvort(u, v, **kw) - 2 * self.omega).max()) > 0.1 * self.omega

    def test_rossby(self):
        u, v, z = self.get_data()
        kw = {"dx": self.res, "dy": self.res}
        rossby = dyn.get_rossby(u, v, z=z, **kw)
        assert rossby.name == "rossby" and rossby.dims == u.dims
        assert rossby.attrs["long_name"] == "Rossby number"
        f = dyn.get_coriolis(u.lat)
        np.testing.assert_allclose(rossby, 2 * self.omega / f.values[:, None] + 0 * rossby, rtol=1e-9)
        assert u.lat.attrs == rossby.lat.attrs
        # Without z, it is the vorticity over the Coriolis parameter, level by level
        np.testing.assert_allclose(
            dyn.get_rossby(u, v, **kw), dyn.get_relvort(u, v, **kw) / f, rtol=1e-12
        )

    def test_extra_dims_transpose_and_dask(self):
        u, v, z = self.get_data(nt=3)
        kw = {"dx": self.res, "dy": self.res}
        vort = dyn.get_relvort(u, v, z=z, **kw)  # z has no time
        assert vort.dims == ("time", "depth", "lat", "lon")
        np.testing.assert_allclose(vort, 2 * self.omega, rtol=1e-9)
        vt = dyn.get_relvort(
            u.transpose("lon", "lat", "depth", "time"),
            v.transpose("lon", "lat", "depth", "time"),
            z=z.transpose("lon", "depth", "lat"),
            **kw,
        )
        assert vt.dims == ("lon", "lat", "depth", "time")
        np.testing.assert_allclose(vt.transpose(*vort.dims), vort, rtol=1e-9)
        pytest.importorskip("dask")
        vd = dyn.get_relvort(u.chunk(time=1), v.chunk(time=1), z=z.chunk(), **kw)
        assert vd.chunks is not None
        np.testing.assert_allclose(vd.compute(), vort)

    def test_resolution_from_the_coordinates(self):
        u, v, z = self.get_data()
        vort = dyn.get_relvort(u, v, z=z)
        # Same shape and about the same values, since the longitude step is constant
        assert vort.shape == u.shape
        assert np.isfinite(vort).all()

    def test_accessors(self):
        u, v, z = self.get_data()
        ds = xr.Dataset({"u": u, "v": v})
        kw = {"dx": self.res, "dy": self.res, "z": z}
        for name in "relvort", "rossby", "div", "okuboweiss":
            res = getattr(ds.xoa, f"get_{name}")(**kw)
            ref = getattr(dyn, f"get_{name}")(u, v, **kw)
            np.testing.assert_allclose(res, ref)

    def test_velocities_must_have_a_vertical_dimension(self):
        u, v, z = self.get_data()
        with pytest.raises(XoaError, match="No dimension found"):
            dyn.get_relvort(u.isel(depth=0), v.isel(depth=0), z=z, dx=self.res, dy=self.res)


class TestVerticalModes:
    nz = 40
    depth = 4000.0

    def get_profile(self, nz=None, positive="up", name="z"):
        """Exponential stratification, on levels that are finer close to the surface"""
        nz = nz or self.nz
        s = (np.arange(nz) + 0.5) / nz
        z = -self.depth * (1 - s) ** 1.5  # From the surface to the bottom would be reversed
        z = np.sort(z)  # From the bottom to the surface
        n2 = 1e-6 + 2e-4 * np.exp(z / 300.0)
        coord = {"standard_name": "altitude" if positive == "up" else "depth", "positive": positive}
        zz = z if positive == "up" else -z
        n2 = xr.DataArray(n2, dims=name, coords={name: (name, zz, coord)}, name="n2")
        return n2

    def get_cube(self, nt=2, ny=3, nx=4, nz=None):
        """Stratification with extra dimensions, whose scale height varies"""
        n2 = self.get_profile(nz)
        z = n2.z
        scale = xr.DataArray(
            np.linspace(200.0, 500.0, nt * ny * nx).reshape(nt, ny, nx), dims=("time", "lat", "lon")
        )
        cube = (1e-6 + 2e-4 * np.exp(z / scale)).transpose("time", "z", "lat", "lon")
        lon = xr.DataArray(np.arange(nx) * 0.1, dims="lon", attrs={"standard_name": "longitude"})
        lat = xr.DataArray(
            45.0 + np.arange(ny) * 0.1, dims="lat", attrs={"standard_name": "latitude"}
        )
        return cube.assign_coords(lon=lon, lat=lat).rename("n2")

    def test_profile(self):
        n2 = self.get_profile()
        vm = dyn.VerticalModes(n2)
        assert vm.nmodes == 5 and vm.bc == "rigid" and vm.zdim == "z"
        assert vm.modes.dims == ("mode", "z") and vm.modes.shape == (5, self.nz)
        assert vm.c.dims == ("mode",) and vm.dz.dims == ("z",)
        assert np.isnan(vm.c[0]) and (vm.c[1:] > 0).all() and (np.diff(vm.c[1:]) < 0).all()
        np.testing.assert_allclose(vm.dz.sum(), self.depth, rtol=0.02)
        np.testing.assert_allclose(vm.modes[0], 1.0)
        assert vm.c.attrs["units"] == "m s-1"
        assert vm.rd is None
        # Same as the numpy routine
        F, c, dz = cmodes.vertical_modes(n2.values, n2.z.values, nmodes=5)
        np.testing.assert_allclose(vm.modes, F)
        np.testing.assert_allclose(vm.c, c, equal_nan=True)
        assert "5 modes" in repr(vm) and "rigid" in repr(vm)

    def test_deformation_radius(self):
        n2 = self.get_profile()
        lat = xr.DataArray(45.0, name="lat", attrs={"standard_name": "latitude"})
        vm = dyn.VerticalModes(n2, lat=lat)
        f = float(dyn.get_coriolis(45.0))
        assert vm.rd.dims == ("mode",) and vm.rd.attrs["units"] == "m"
        assert np.isnan(vm.rd[0])
        np.testing.assert_allclose(vm.rd[1:], vm.c[1:] / f)
        assert 5e3 < float(vm.rd[1]) < 100e3
        # Not defined at the equator
        assert np.isnan(dyn.VerticalModes(n2, lat=0.0).rd).all()
        # The latitude of the coordinates is used when it is not provided
        cube = self.get_cube()
        vm = dyn.VerticalModes(cube)
        assert vm.rd.dims == ("mode", "time", "lat", "lon")

    def test_free_surface(self):
        n2 = self.get_profile()
        vm = dyn.VerticalModes(n2, bc="free")
        np.testing.assert_allclose(vm.c[0], np.sqrt(9.81 * self.depth), rtol=0.05)
        rigid = dyn.VerticalModes(n2)
        np.testing.assert_allclose(vm.c[1:], rigid.c[1:], rtol=0.02)
        with pytest.raises(ValueError, match="bc"):
            dyn.VerticalModes(n2, bc="other")

    def test_extra_dimensions(self):
        cube = self.get_cube()
        vm = dyn.VerticalModes(cube, nmodes=4)
        assert vm.modes.dims == ("mode", "time", "z", "lat", "lon")
        assert vm.c.dims == ("mode", "time", "lat", "lon")
        assert vm.dz.dims == ("time", "z", "lat", "lon")
        column = dyn.VerticalModes(cube.isel(time=1, lat=2, lon=3), nmodes=4)
        np.testing.assert_allclose(vm.modes.isel(time=1, lat=2, lon=3), column.modes)
        np.testing.assert_allclose(vm.c.isel(time=1, lat=2, lon=3), column.c, equal_nan=True)
        # The coordinates and their attributes are kept
        assert vm.c.lat.attrs == cube.lat.attrs and vm.modes.lon.attrs == cube.lon.attrs
        assert (vm.c.mode == np.arange(4)).all()

    def test_variable_z(self):
        """The depths of the levels vary in space, like with sigma coordinates"""
        cube = self.get_cube(nt=1, ny=2, nx=3)
        n2 = cube.isel(time=0)
        h = xr.DataArray([[0.4, 0.7, 1.0], [0.5, 0.8, 0.9]], dims=("lat", "lon"))
        z = (n2.z * h).transpose("z", "lat", "lon").rename("z")
        z.attrs["positive"] = "up"
        vm = dyn.VerticalModes(n2, z=z, nmodes=3)
        assert vm.modes.dims == ("mode", "z", "lat", "lon")
        for j, i in ((0, 0), (1, 2), (0, 2)):
            ref = dyn.VerticalModes(
                n2.isel(lat=j, lon=i).drop_vars(["lat", "lon"]), z=z.isel(lat=j, lon=i), nmodes=3
            )
            np.testing.assert_allclose(vm.modes.isel(lat=j, lon=i), ref.modes)
            np.testing.assert_allclose(vm.c.isel(lat=j, lon=i), ref.c, equal_nan=True)
        # Shallower columns have shallower modes: smaller speeds
        assert float(vm.c.isel(mode=1, lat=0, lon=0)) < float(vm.c.isel(mode=1, lat=0, lon=2))

    def test_missing_levels_below_the_bottom(self):
        cube = self.get_cube(nt=1, ny=1, nx=2)
        n2 = cube.isel(time=0).copy()
        n2[:10, 0, 1] = np.nan
        vm = dyn.VerticalModes(n2, nmodes=3)
        assert np.isnan(vm.modes.isel(z=slice(0, 10), lat=0, lon=1)).all()
        assert np.isfinite(vm.modes.isel(z=slice(10, None), lat=0, lon=1)).all()
        assert np.isfinite(vm.c.isel(mode=1, lat=0, lon=1))
        ref = dyn.VerticalModes(n2.isel(z=slice(10, None), lat=0, lon=1), nmodes=3)
        np.testing.assert_allclose(vm.c.isel(lat=0, lon=1), ref.c, equal_nan=True)

    def test_depth_positive_down(self):
        up = self.get_profile()
        down = self.get_profile(positive="down", name="depth")
        a, b = dyn.VerticalModes(up, nmodes=3), dyn.VerticalModes(down, nmodes=3)
        np.testing.assert_allclose(a.c, b.c, equal_nan=True)
        np.testing.assert_allclose(a.modes, b.modes)
        # From the surface to the bottom
        rev = down.isel(depth=slice(None, None, -1))
        c = dyn.VerticalModes(rev, nmodes=3)
        np.testing.assert_allclose(c.c, a.c, equal_nan=True)
        np.testing.assert_allclose(c.modes.isel(depth=slice(None, None, -1)), a.modes)

    def test_decomp(self):
        n2 = self.get_profile()
        vm = dyn.VerticalModes(n2, nmodes=6)
        field = np.exp(n2.z / 800.0)
        a = vm.decomp(field)
        assert a.dims == ("mode",) and a.shape == (6,)
        np.testing.assert_allclose(a[0], float((field * vm.dz).sum() / vm.dz.sum()))
        assert a.name == "amps" and a.attrs["long_name"].startswith("Amplitude")
        assert vm.decomp(field, nmodes=3).shape == (3,)
        # A mode is projected on itself
        np.testing.assert_allclose(vm.decomp(1.5 * vm.modes.isel(mode=2)), np.eye(6)[2] * 1.5, atol=1e-10)

    def test_decomp_extra_dimensions(self):
        cube = self.get_cube()
        vm = dyn.VerticalModes(cube, nmodes=4)
        # A field with an extra dimension that the stratification does not have
        field = (np.exp(cube.z / 800.0) * xr.DataArray([1.0, 2.0], dims="member")).transpose(
            "member", ...
        )
        field = field.broadcast_like(cube)
        amps = vm.decomp(field)
        # The dimensions of the profiles come after the mode, in their order
        assert amps.dims == ("mode",) + tuple(d for d in field.dims if d != "z")
        assert set(amps.dims) == {"mode", "member", "time", "lat", "lon"}
        np.testing.assert_allclose(amps.isel(member=1), 2 * amps.isel(member=0))
        # A single dimension of the stratification
        amps = vm.decomp(np.exp(cube.z / 800.0))
        assert amps.dims == ("mode", "time", "lat", "lon")
        col = dyn.VerticalModes(cube.isel(time=1, lat=2, lon=3), nmodes=4)
        np.testing.assert_allclose(
            amps.isel(time=1, lat=2, lon=3), col.decomp(np.exp(cube.z / 800.0)), atol=1e-12
        )

    def test_recomp(self):
        n2 = self.get_profile()
        vm = dyn.VerticalModes(n2, nmodes=self.nz)
        field = xr.DataArray(
            np.random.default_rng(0).normal(size=self.nz), dims="z", coords={"z": n2.z}, name="u"
        )
        amps = vm.decomp(field)
        rec = vm.recomp(amps)
        assert rec.dims == ("z",)
        np.testing.assert_allclose(rec, field, atol=1e-9)
        # Truncation: the error decreases when modes are added
        smooth = np.tanh((n2.z + 300.0) / 100.0)
        amps = vm.decomp(smooth)
        err = [float(abs(vm.recomp(amps, modes=n) - smooth).mean()) for n in (1, 3, 8, 20)]
        assert (np.diff(err) < 0).all()
        # A selection of modes
        only1 = vm.recomp(amps, modes=[1])
        np.testing.assert_allclose(only1, amps.isel(mode=1) * vm.modes.isel(mode=1), atol=1e-12)
        both = vm.recomp(amps, modes=[0, 1])
        np.testing.assert_allclose(both, only1 + amps.isel(mode=0) * vm.modes.isel(mode=0))
        # Fewer amplitudes than modes
        np.testing.assert_allclose(vm.recomp(vm.decomp(smooth, nmodes=3)), vm.recomp(amps, modes=3))

    def test_recomp_extra_dimensions_and_order(self):
        cube = self.get_cube()
        vm = dyn.VerticalModes(cube, nmodes=cube.sizes["z"])
        field = (np.exp(cube.z / 500.0) * (1 + cube.lat / 90.0)).broadcast_like(cube)
        field = field.transpose("time", "z", "lat", "lon")
        rec = vm.recomp(vm.decomp(field))
        assert rec.dims == ("time", "z", "lat", "lon")
        np.testing.assert_allclose(rec, field, atol=1e-9)

    def test_vertical_zoom(self):
        n2 = self.get_profile()
        zoom = dyn.VerticalModes(n2, nmodes=4, zrange=(-1000.0, -100.0))
        inside = (n2.z >= -1000.0) & (n2.z <= -100.0)
        assert zoom.zrange == (-1000.0, -100.0)
        assert np.isnan(zoom.modes.isel(z=np.where(~inside.values)[0])).all()
        assert np.isfinite(zoom.modes.isel(z=np.where(inside.values)[0])).all()
        ref = dyn.VerticalModes(n2.where(inside, drop=True), nmodes=4)
        np.testing.assert_allclose(zoom.c, ref.c, equal_nan=True)
        np.testing.assert_allclose(zoom.modes.where(inside, drop=True), ref.modes)
        # Only the layer is decomposed and recomposed
        field = np.exp(n2.z / 800.0)
        amps = zoom.decomp(field)
        np.testing.assert_allclose(amps, ref.decomp(field.where(inside, drop=True)), atol=1e-12)
        full = dyn.VerticalModes(n2, nmodes=self.nz)
        zoomed = dyn.VerticalModes(n2, nmodes=int(inside.sum()), zrange=(-1000.0, -100.0))
        rec = zoomed.recomp(zoomed.decomp(field))
        np.testing.assert_allclose(rec.where(inside), field.where(inside), atol=1e-9)
        assert np.isnan(rec.where(~inside)).all() and full.zrange is None

    def test_depths_for_the_zoom_follow_the_convention_of_z(self):
        up = self.get_profile()
        down = self.get_profile(positive="down", name="depth")
        a = dyn.VerticalModes(up, nmodes=3, zrange=(-1000.0, -100.0))
        b = dyn.VerticalModes(down, nmodes=3, zrange=(100.0, 1000.0))
        np.testing.assert_allclose(a.c, b.c, equal_nan=True)

    def test_dask(self):
        pytest.importorskip("dask")
        cube = self.get_cube()
        vm = dyn.VerticalModes(cube, nmodes=4)
        vmd = dyn.VerticalModes(cube.chunk({"time": 1, "lat": 2}), nmodes=4)
        assert vmd.modes.chunks is not None
        np.testing.assert_allclose(vmd.modes.compute(), vm.modes)
        np.testing.assert_allclose(vmd.c.compute(), vm.c, equal_nan=True)
        field = np.exp(cube.z / 800.0).broadcast_like(cube)
        amps = vmd.decomp(field.chunk({"time": 1, "lat": 2}))
        assert amps.chunks is not None
        np.testing.assert_allclose(amps.compute(), vm.decomp(field))
        rec = vmd.recomp(amps)
        assert rec.chunks is not None
        np.testing.assert_allclose(rec.compute(), vm.recomp(vm.decomp(field)))

    def test_errors(self):
        n2 = self.get_profile()
        with pytest.raises(ValueError, match="nmodes"):
            dyn.VerticalModes(n2, nmodes=0)
        with pytest.raises(XoaError):
            dyn.VerticalModes(n2.isel(z=0))
        vm = dyn.VerticalModes(n2, nmodes=3)
        with pytest.raises(ValueError, match="mode"):
            vm.recomp(vm.decomp(np.exp(n2.z / 800.0)), modes=[7])
