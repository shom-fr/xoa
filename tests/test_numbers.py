"""
Test the :mod:`xoa.numbers` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import dyn
from xoa import numbers as xnum

RES = 5000.0


def get_uv(ny=20, nx=30):
    """Solid body rotation at 45N"""
    omega = 1e-5
    dlat = np.degrees(RES / 6371e3)
    lat = xr.DataArray(
        45.0 + dlat * np.arange(ny), dims="lat", attrs={"standard_name": "latitude"}, name="lat"
    )
    lon = xr.DataArray(
        dlat * np.arange(nx), dims="lon", attrs={"standard_name": "longitude"}, name="lon"
    )
    x = (np.arange(nx) - nx / 2) * RES
    y = (np.arange(ny) - ny / 2) * RES
    coords = {"lon": lon, "lat": lat}
    u = xr.DataArray(-omega * y[:, None] + 0 * x, dims=("lat", "lon"), coords=coords, name="u")
    v = xr.DataArray(omega * x + 0 * y[:, None], dims=("lat", "lon"), coords=coords, name="v")
    return u, v, omega


def test_get_rossby():
    u, v, omega = get_uv()
    kw = {"dx": RES, "dy": RES}
    rossby = xnum.get_rossby(u, v, **kw)
    assert rossby.name == "rossby" and rossby.dims == u.dims
    assert rossby.attrs == {"long_name": "Rossby number", "units": "1"}
    assert u.lat.attrs == rossby.lat.attrs
    f = dyn.get_coriolis(u.lat)
    np.testing.assert_allclose(rossby, dyn.get_relvort(u, v, **kw) / f, rtol=1e-12)
    np.testing.assert_allclose(rossby, 2 * omega / f.values[:, None] + 0 * rossby, rtol=1e-9)
    assert not hasattr(dyn, "get_rossby")


def test_get_rossby_variable_z():
    from test_dyn import TestVariableZ

    t = TestVariableZ()
    u, v, z = t.get_data()
    kw = {"dx": t.res, "dy": t.res}
    rossby = xnum.get_rossby(u, v, z=z, **kw)
    assert rossby.dims == u.dims
    f = dyn.get_coriolis(u.lat)
    np.testing.assert_allclose(rossby, 2 * t.omega / f.values[:, None] + 0 * rossby, rtol=1e-9)
    ds = xr.Dataset({"u": u, "v": v})
    np.testing.assert_allclose(ds.xoa.get_rossby(z=z, **kw), rossby)


def test_get_rossby_accessor():
    u, v, _ = get_uv()
    ds = xr.Dataset({"u": u, "v": v})
    np.testing.assert_allclose(
        ds.xoa.get_rossby(dx=RES, dy=RES), xnum.get_rossby(u, v, dx=RES, dy=RES)
    )


class TestBurger:
    lat = xr.DataArray([30.0, 45.0], dims="lat", attrs={"standard_name": "latitude"}, name="lat")

    def test_rd(self):
        bu = xnum.get_burger(1e5, rd=5e4)
        assert float(bu) == pytest.approx(0.25)
        assert bu.name == "burger" and bu.attrs["units"] == "1"

    def test_n2_h(self):
        n2 = xr.DataArray([1e-5, 1e-5], dims="lat", coords={"lat": self.lat}, name="n2")
        bu = xnum.get_burger(1e5, n2=n2, h=1000.0)
        f = abs(dyn.get_coriolis(self.lat)).values
        np.testing.assert_allclose(bu, (np.sqrt(1e-5) * 1000.0 / f / 1e5) ** 2)
        assert bu.lat.attrs == self.lat.attrs
        # Same result with rd and an explicit latitude
        np.testing.assert_allclose(
            bu, xnum.get_burger(1e5, rd=np.sqrt(1e-5) * 1000.0 / f), rtol=1e-12
        )
        np.testing.assert_allclose(
            xnum.get_burger(1e5, n2=n2.values[0], h=1000.0, lat=self.lat), bu
        )

    def test_errors(self):
        with pytest.raises(ValueError):
            xnum.get_burger(1e5)
        with pytest.raises(ValueError):
            xnum.get_burger(1e5, n2=1e-5, h=1000.0)

    def test_equator(self):
        n2 = xr.DataArray(1e-5)
        assert np.isnan(xnum.get_burger(1e5, n2=n2, h=1000.0, lat=0.0))


def test_get_froude():
    n2 = xr.DataArray([1e-4, -1e-4, 1e-4], dims="z", coords={"z": [-30.0, -20.0, -10.0]})
    speed = xr.DataArray([0.1, 0.1, 0.5], dims="z", coords=n2.coords)
    fr = xnum.get_froude(speed, n2, 10.0)
    np.testing.assert_allclose(fr.values, [1.0, np.nan, 5.0])
    assert fr.name == "froude" and fr.attrs["units"] == "1"


def test_get_ekman():
    lat = xr.DataArray([0.0, 45.0], dims="lat", attrs={"standard_name": "latitude"}, name="lat")
    ek = xnum.get_ekman(1e-2, 100.0, lat)
    f45 = float(abs(dyn.get_coriolis(45.0)))
    assert np.isnan(ek[0]) and float(ek[1]) == pytest.approx(1e-2 / (f45 * 1e4))
    assert ek.name == "ekman" and ek.attrs["long_name"] == "Ekman number"
    assert float(xnum.get_ekman(1e-2, 100.0, 45.0)) == pytest.approx(float(ek[1]))
