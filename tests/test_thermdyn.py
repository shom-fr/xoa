# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.thermdyn` module
"""

import numpy as np
import xarray as xr
import pytest

from xoa import thermdyn


class TestMixedLayerDepth:
    """Test mixed layer depth functions"""

    def test_deltatemp(self):
        """Test MLD with deltatemp method"""
        depth = xr.DataArray(
            np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth", "positive": "up"}
        )
        temp = xr.DataArray(np.linspace(10, 20.0, 6), dims="z", coords={"depth": depth}, name="temp")
        mld = thermdyn.mixed_layer_depth(temp, method="deltatemp")
        np.testing.assert_allclose(mld, 1)
        assert mld.long_name == "Mixed layer depth"
        assert mld.dims == ()

    @pytest.mark.parametrize("positive", ["up", "down"])
    def test_z_or_depth_coord(self, positive):
        """Test that the level order follows the native vertical coordinate"""
        z = np.linspace(-50, 0.0, 6)
        if positive == "down":
            z = -z[::-1]
        vert = xr.DataArray(z, dims="z", attrs={"positive": positive})
        temp = np.linspace(10, 20.0, 6)
        temp = temp if positive == "up" else temp[::-1]
        name = "z" if positive == "up" else "depth"
        temp = xr.DataArray(temp, dims="z", coords={name: vert}, name="temp")
        mld = thermdyn.mixed_layer_depth(temp, method="deltatemp")
        np.testing.assert_allclose(mld, 1)

    def test_implicit(self):
        """Test MLD method inference from temperature data"""
        depth = xr.DataArray(
            np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth", "positive": "up"}
        )
        temp = xr.DataArray(np.linspace(10, 20.0, 6), dims="z", coords={"depth": depth}, name="temp")
        mld = thermdyn.mixed_layer_depth(temp)
        np.testing.assert_allclose(mld, 1)

    def test_kzmax(self):
        """Test MLD with kzmax method"""
        depth = xr.DataArray(
            np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth", "positive": "up"}
        )
        kz = xr.DataArray(np.linspace(0.0005 * 10, 0, 6), dims="z", coords={"depth": depth})
        mld = thermdyn.mixed_layer_depth(kz, method="kzmax")
        np.testing.assert_allclose(mld, 5)

    def test_deltadens(self):
        """Test MLD with deltadens method"""
        depth = xr.DataArray(
            np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth", "positive": "up"}
        )
        dens = xr.DataArray(
            np.linspace(1030, 1025.0, 6), dims="z",
            coords={"depth": depth}, name="dens",
        )
        mld = thermdyn.mixed_layer_depth(dens, method="deltadens", deltadens=1.0)
        assert mld.dims == ()
        np.testing.assert_allclose(mld, 10)

    def test_2d(self):
        """Test MLD with 2D data"""
        depth = xr.DataArray(
            np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth", "positive": "up"}
        )
        temp = xr.DataArray(np.linspace(10, 20.0, 6), dims="z", coords={"depth": depth}, name="temp")
        temp2d = temp.expand_dims("lon").transpose(..., "lon")
        depth2d = depth.expand_dims("lon").transpose(..., "lon")
        temp2d.coords["depth"] = depth2d
        mld2d = thermdyn.mixed_layer_depth(temp2d, method="deltatemp")
        assert mld2d.dims == ("lon",)
        assert mld2d.shape == (1,)
        np.testing.assert_allclose(mld2d, 1)


class TestTemperature:
    """Test temperature identification functions"""

    def test_is_temp(self):
        """Test temperature identification"""
        temp = xr.DataArray([15, 16, 17], dims="z", name="temp",
                           attrs={"standard_name": "sea_water_temperature"})
        assert thermdyn.is_temp(temp)

        not_temp = xr.DataArray([35, 35.5], dims="z", name="sal")
        assert not thermdyn.is_temp(not_temp)

    def test_get_temp(self):
        """Test getting temperature from dataset"""
        ds = xr.Dataset({
            "temp": xr.DataArray([15, 16, 17], dims="z"),
            "sal": xr.DataArray([35, 35.5, 36], dims="z")
        })
        temp = thermdyn.get_temp(ds, errors="ignore")
        assert temp is not None
        assert temp.name == "temp"

    def test_get_temp_not_found(self):
        """Test get_temp returns None when no temperature in dataset"""
        ds = xr.Dataset({"sal": xr.DataArray([35, 35.5, 36], dims="z")})
        temp = thermdyn.get_temp(ds, errors="ignore")
        assert temp is None


class TestDensity:
    """Test density identification functions"""

    def test_is_dens(self):
        """Test density identification"""
        dens = xr.DataArray([1025, 1026, 1027], dims="z", name="dens",
                           attrs={"standard_name": "sea_water_density"})
        assert thermdyn.is_dens(dens)

        not_dens = xr.DataArray([15, 16], dims="z", name="temp")
        assert not thermdyn.is_dens(not_dens)

    def test_get_dens(self):
        """Test getting density from dataset"""
        ds = xr.Dataset({
            "dens": xr.DataArray([1025, 1026, 1027], dims="z"),
            "temp": xr.DataArray([15, 16, 17], dims="z")
        })
        dens = thermdyn.get_dens(ds, errors="ignore")
        assert dens is not None
        assert dens.name == "dens"

    def test_get_dens_not_found(self):
        """Test get_dens returns None when no density in dataset"""
        ds = xr.Dataset({"temp": xr.DataArray([15, 16, 17], dims="z")})
        dens = thermdyn.get_dens(ds, errors="ignore")
        assert dens is None


class TestSalinity:
    """Test salinity identification functions"""

    def test_is_sal(self):
        """Test salinity identification"""
        sal = xr.DataArray([35, 35.5, 36], dims="z", name="sal",
                          attrs={"standard_name": "sea_water_salinity"})
        assert thermdyn.is_sal(sal)

        not_sal = xr.DataArray([15, 16], dims="z", name="temp")
        assert not thermdyn.is_sal(not_sal)

    def test_get_sal(self):
        """Test getting salinity from dataset"""
        ds = xr.Dataset({
            "sal": xr.DataArray([35, 35.5, 36], dims="z"),
            "temp": xr.DataArray([15, 16, 17], dims="z")
        })
        sal = thermdyn.get_sal(ds, errors="ignore")
        assert sal is not None
        assert sal.name == "sal"

    def test_get_sal_not_found(self):
        """Test get_sal returns None when no salinity in dataset"""
        ds = xr.Dataset({"temp": xr.DataArray([15, 16, 17], dims="z")})
        sal = thermdyn.get_sal(ds, errors="ignore")
        assert sal is None


class TestN2:
    """Test the squared Brunt-Vaisala frequency"""

    @staticmethod
    def get_profile(positive="up"):
        z = np.array([-1000.0, -500.0, -300.0, -200.0, -100.0, -50.0, -10.0, 0.0])
        zz = z if positive == "up" else -z
        coords = {"z": ("z", zz, {"positive": positive, "standard_name": "altitude"})}
        temp = xr.DataArray(
            4.0 + 16.0 * np.exp(z / 200.0), dims="z", coords=coords, name="temp",
            attrs={"standard_name": "sea_water_potential_temperature"},
        )
        sal = xr.DataArray(35.0 - 0.5 * np.exp(z / 300.0), dims="z", coords=coords, name="sal")
        return temp, sal

    def test_values(self):
        import gsw

        temp, sal = self.get_profile()
        n2 = thermdyn.get_n2(temp, sal, lat=45.0)
        assert n2.dims == ("z",) and n2.name == "n2"
        assert n2.attrs["units"] == "s-2" and "Brunt" in n2.attrs["long_name"]
        assert (n2 > 0).all()
        # Orders of magnitude: weak at depth and strong in the thermocline
        assert 1e-8 < float(n2.isel(z=0)) < 1e-5 and 1e-5 < float(n2.max()) < 1e-3
        # Same as gsw at the middle of the levels
        z = temp.z.values
        p = gsw.p_from_z(z, 45.0)
        sa = gsw.SA_from_SP(sal.values, p, 0.0, 45.0)
        ct = gsw.CT_from_pt(sa, temp.values)
        mid, _ = gsw.Nsquared(sa, ct, p, 45.0)
        np.testing.assert_allclose(n2.values[1:-1], 0.5 * (mid[:-1] + mid[1:]), rtol=1e-12)
        np.testing.assert_allclose(n2.values[[0, -1]], mid[[0, -1]], rtol=1e-12)

    def test_depth_positive_down_and_level_order(self):
        up = thermdyn.get_n2(*self.get_profile(), lat=45.0)
        temp, sal = self.get_profile(positive="down")
        down = thermdyn.get_n2(temp, sal, lat=45.0)
        np.testing.assert_allclose(down, up)
        rev = thermdyn.get_n2(temp.isel(z=slice(None, None, -1)), sal.isel(z=slice(None, None, -1)), lat=45.0)
        np.testing.assert_allclose(rev.isel(z=slice(None, None, -1)), up)

    def test_extra_dimensions_latitudes_and_z(self):
        temp, sal = self.get_profile()
        lat = xr.DataArray([30.0, 45.0, 60.0], dims="lat", attrs={"standard_name": "latitude"})
        lon = xr.DataArray([0.0, 5.0], dims="lon", attrs={"standard_name": "longitude"})
        cube_t = temp.expand_dims(lat=lat, lon=lon).transpose("lat", "z", "lon").assign_coords(lat=lat, lon=lon)
        cube_s = sal.expand_dims(lat=lat, lon=lon).transpose("lat", "z", "lon").assign_coords(lat=lat, lon=lon)
        n2 = thermdyn.get_n2(cube_t, cube_s)  # lat and lon are found
        assert n2.dims == ("lat", "z", "lon")
        np.testing.assert_allclose(
            n2.isel(lat=1, lon=0), thermdyn.get_n2(temp, sal, lat=45.0, lon=0.0), rtol=1e-10
        )
        assert not np.allclose(n2.isel(lat=0), n2.isel(lat=2))
        # The depths of the levels vary in space
        z = (temp.z * xr.DataArray([1.0, 0.5], dims="lon")).transpose("z", "lon")
        z.attrs["positive"] = "up"
        n2z = thermdyn.get_n2(cube_t, cube_s, z=z)
        assert n2z.dims == ("lat", "z", "lon")
        assert float(n2z.isel(lat=1, lon=1, z=2)) != float(n2.isel(lat=1, lon=1, z=2))

    def test_in_situ_temperature(self):
        temp, sal = self.get_profile()
        pt = thermdyn.get_n2(temp, sal, lat=45.0)
        insitu = thermdyn.get_n2(temp, sal, lat=45.0, ptemp=False)
        assert not np.allclose(pt, insitu)
        np.testing.assert_allclose(pt.isel(z=slice(4, None)), insitu.isel(z=slice(4, None)), rtol=0.05)

    def test_missing_values_and_dask(self):
        temp, sal = self.get_profile()
        temp = temp.copy()
        temp[0] = np.nan
        n2 = thermdyn.get_n2(temp, sal, lat=45.0)
        assert np.isnan(n2[0]) and np.isfinite(n2[2:]).all()
        pytest.importorskip("dask")
        t2 = temp.expand_dims(x=3).transpose("z", "x")
        s2 = sal.expand_dims(x=3).transpose("z", "x")
        lazy = thermdyn.get_n2(t2.chunk({"x": 1}), s2.chunk({"x": 1}), lat=45.0)
        assert lazy.chunks is not None
        np.testing.assert_allclose(lazy.compute().isel(x=1), n2, equal_nan=True)

    def test_latitude_is_needed(self):
        temp, sal = self.get_profile()
        with pytest.raises(Exception, match="lat"):
            thermdyn.get_n2(temp, sal)
