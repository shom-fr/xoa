"""
Test the :mod:`xoa.tide.filter` module
"""

import numpy as np
import pytest
import xarray as xr

import xoa.tide as xtide


class TestTidalFilter:
    def test_demerliac(self):
        times = np.arange("2000-01-01", "2000-01-10", dtype="M8[h]").astype("M8[ns]")
        nt = len(times)
        data = np.cos(np.arange(nt) * np.pi / 12.2)
        data = np.resize(data, (3, 4, nt)).T
        da_in = xr.DataArray(data, dims=('time', 'ny', 'nx'), coords={"time": times})

        da_out = xtide.demerliac(da_in, na_thres=0)
        assert da_out.dims == da_in.dims
        assert da_out.shape == da_in.shape
        assert da_out.min() > -1e-4
        assert da_out.max() < 1e-4

        da_out = xtide.demerliac(da_in, na_thres=1)
        assert abs(da_out.mean()) < 0.1

    @pytest.mark.parametrize("filter_name", ["demerliac", "godin", "mean"])
    def test_filter_names(self, filter_name):
        times = np.arange("2000-01-01", "2000-01-10", dtype="M8[h]").astype("M8[ns]")
        nt = len(times)
        data = np.cos(np.arange(nt) * np.pi / 12.5)
        data = np.resize(data, (3, 4, nt)).T
        da_in = xr.DataArray(data, dims=('time', 'ny', 'nx'), coords={"time": times})

        da_out = xtide.tidal_filter(da_in, filter_name, na_thres=0)
        assert da_out.dims == da_in.dims
        assert da_out.shape == da_in.shape
        assert da_out.min() > -1e-4
        assert da_out.max() < 1e-4

    @pytest.mark.parametrize("minutes", [30, 20])
    def test_subhourly(self, minutes):
        times = np.arange("2000-01-01", "2000-01-10", minutes, dtype="M8[m]").astype("M8[ns]")
        hours = np.arange(len(times)) * minutes / 60
        da_in = xr.DataArray(np.cos(2 * np.pi * hours / 12.42), dims="time", coords={"time": times})

        da_out = xtide.tidal_filter(da_in, "demerliac")
        assert da_out.shape == da_in.shape
        assert da_out.notnull().any()
        assert float(abs(da_out).max()) < 1e-2

    def test_na_thres(self):
        times = np.arange("2000-01-01", "2000-01-10", dtype="M8[h]").astype("M8[ns]")
        nt = len(times)
        data = np.cos(np.arange(nt) * np.pi / 12.5)
        data = np.resize(data, (3, 4, nt)).T
        da_in = xr.DataArray(data, dims=('time', 'ny', 'nx'), coords={"time": times})

        da_out = xtide.tidal_filter(da_in, "demerliac", na_thres=1)
        assert abs(da_out.mean()) < 0.1

    def test_invalid_filter_name(self):
        times = np.arange("2000-01-01", "2000-01-03", dtype="M8[h]").astype("M8[ns]")
        da_in = xr.DataArray(np.ones(len(times)), dims='time', coords={"time": times})
        with pytest.raises(Exception):
            xtide.tidal_filter(da_in, "invalid_filter")
