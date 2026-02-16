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
