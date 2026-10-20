# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.tide.diags` module
"""

import numpy as np
import pytest
import xarray as xr

import xoa.tide as xtide


def semidiurnal_series(nt=24 * 30):
    time = np.datetime64("2024-01-01", "s") + np.arange(nt) * np.timedelta64(3600, "s")
    hours = np.arange(nt)
    h = np.cos(2 * np.pi * hours / 12.42) + 0.3 * np.cos(2 * np.pi * hours / 354.4)
    return xr.DataArray(h, dims="time", coords={"time": time})


def test_high_waters_and_range():
    h = semidiurnal_series()
    hw = xtide.high_waters(h)
    assert hw.dtype == bool and hw.dims == h.dims
    assert abs(int(hw.sum()) - 30 * 24 / 12.42) <= 2
    rng = xtide.tidal_range(h)
    assert float(rng.max()) <= 2 * 1.3 + 1e-6
    assert xtide.high_waters(h.expand_dims(x=2).transpose("time", "x")).shape == (len(h), 2)
    with pytest.raises(Exception):
        xtide.high_waters(h.expand_dims(x=2), time_dim="time")


def test_skew_surge_and_spring():
    pred = semidiurnal_series()
    obs = pred + 0.2
    hw = xtide.high_waters(pred)
    surge = xtide.skew_surge(obs, pred)
    np.testing.assert_allclose(surge.where(hw).dropna("time"), 0.2, atol=1e-8)
    assert int(surge.notnull().sum()) <= int(hw.sum())
    spring = xtide.spring_high_waters(pred)
    assert 0 < int(spring.sum()) < int(hw.sum())
    assert not (spring & ~hw).any()
    atlas = xtide.tidal_atlas(pred, spring.values)
    assert atlas.dims == ("offset",) and atlas.sizes["offset"] == 13


def test_form_number():
    amp = xr.DataArray(
        [1.0, 0.5, 0.2, 0.1], dims="constituent", coords={"constituent": ["M2", "S2", "K1", "O1"]}
    )
    assert float(xtide.form_number(xr.Dataset({"amp": amp}))) == pytest.approx(0.3 / 1.5)
