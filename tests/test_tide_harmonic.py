# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.tide.harmonic` module
"""

import numpy as np
import pytest
import xarray as xr

import xoa.tide as xtide
from xoa.exceptions import XoaError
from conftest import TIDE_NAMES, TIDE_NX, TIDE_NY, TIDE_TIME


@pytest.mark.parametrize("greenwich", [False, True])
def test_analysis_prediction_roundtrip_3d(greenwich, tide_ssh):
    da, h0, amp, phase = tide_ssh
    coef = xtide.harmonic_analysis(da, names=TIDE_NAMES, nodal=True, greenwich=greenwich)
    assert coef["amp"].dims == ("constituent", "y", "x")
    assert coef["h0"].dims == ("y", "x")
    assert coef.attrs["phase_convention"] == ("greenwich" if greenwich else "local")
    np.testing.assert_allclose(coef["h0"], h0, atol=1e-8)
    pred = xtide.harmonic_predict(coef, da.time.values)
    assert pred.dims == da.dims
    np.testing.assert_allclose(pred, da, atol=1e-8)
    # prediction on another period
    later = TIDE_TIME[-1] + np.arange(1, 49) * np.timedelta64(3600, "s")
    assert xtide.harmonic_predict(coef, later).shape == (48, TIDE_NY, TIDE_NX)


def test_analysis_without_nodal_recovers_coefficients(tide_ssh):
    da, h0, amp, phase = tide_ssh
    coef = xtide.harmonic_analysis(da, names=TIDE_NAMES, nodal=False, uncertainty=False)
    np.testing.assert_allclose(coef["amp"], amp, atol=1e-8)
    np.testing.assert_allclose(np.cos(coef["phase"]), np.cos(phase), atol=1e-8)
    assert "amp_std" not in coef


def test_analysis_uncertainty_and_ensemble(tide_ssh):
    da, *_ = tide_ssh
    da = da + np.random.default_rng(3).normal(scale=0.05, size=da.shape)
    th = xtide.TidalHarmonic(da.time.values, TIDE_NAMES)
    coef = th.analyse(da)
    assert (coef["amp_std"] > 0).all()
    ens = th.ensemble(coef, n_members=5, seed=0)
    assert ens["amp"].dims[:2] == ("constituent", "member")
    pred = th.predict(ens, da.time.values)
    assert pred.dims[:2] == ("time", "member")
    with pytest.raises(Exception):
        th.ensemble(th.analyse(da, uncertainty=False))


def test_analysis_1d_and_nan_points(tide_ssh):
    da, *_ = tide_ssh
    da[:, 0, 0] = np.nan
    coef = xtide.harmonic_analysis(da, names=TIDE_NAMES)
    assert np.isnan(coef["amp"].isel(y=0, x=0)).all()
    assert coef["amp"].isel(y=1, x=1).notnull().all()
    coef1d = xtide.harmonic_analysis(da.isel(y=1, x=1), names=TIDE_NAMES)
    assert coef1d["amp"].dims == ("constituent",)
    assert xtide.harmonic_predict(coef1d, da.time.values).dims == ("time",)


def get_ssh_with_lonlat(tide_ssh):
    da, *_ = tide_ssh
    return da.assign_coords(lon=("x", np.arange(TIDE_NX) * 1.0), lat=("y", np.arange(TIDE_NY) * 1.0))


def test_analysis_keeps_spatial_coords(tide_ssh):
    da = get_ssh_with_lonlat(tide_ssh)
    coef = xtide.harmonic_analysis(da, names=TIDE_NAMES, uncertainty=False)
    assert "lon" in coef.coords and "lat" in coef.coords
    pred = xtide.harmonic_predict(coef, da.time.values[:5])
    assert pred.dims == ("time", "y", "x") and "lon" in pred.coords and "lat" in pred.coords


def test_analysis_dask(tide_ssh):
    pytest.importorskip("dask")
    da = get_ssh_with_lonlat(tide_ssh)
    coef = xtide.harmonic_analysis(da.chunk({"y": 1}), names=TIDE_NAMES, uncertainty=False)
    ref = xtide.harmonic_analysis(da, names=TIDE_NAMES, uncertainty=False)
    np.testing.assert_allclose(coef["amp"], ref["amp"])
    assert "lon" in coef.coords


def test_auto_names(tide_ssh):
    da, *_ = tide_ssh
    th = xtide.TidalHarmonic(da.time.values)
    assert th.names == xtide.auto_constituents(29.96, 1.0)


def test_currents_roundtrip(tide_uv):
    da_u, da_v, smaj, smin = tide_uv
    coef = xtide.harmonic_analysis_uv(da_u, da_v, names=TIDE_NAMES, nodal=False)
    np.testing.assert_allclose(coef["semi_major"], smaj, atol=1e-8)
    np.testing.assert_allclose(coef["semi_minor"], smin, atol=1e-8)
    u, v = xtide.harmonic_predict_uv(coef, da_u.time.values)
    np.testing.assert_allclose(u, da_u, atol=1e-8)
    np.testing.assert_allclose(v, da_v, atol=1e-8)


def test_time_dim_and_variables_are_searched(tide_ssh, tide_uv):
    da, *_ = tide_ssh
    ref = xtide.harmonic_analysis(da, names=TIDE_NAMES)
    renamed = da.rename(time="valid_time")
    coef = xtide.harmonic_analysis(renamed.to_dataset(name="ssh"), names=TIDE_NAMES)
    np.testing.assert_allclose(coef["amp"], ref["amp"])
    assert xtide.harmonic_predict(coef, renamed.valid_time).dims == renamed.dims
    da_u, da_v, *_ = tide_uv
    coef = xtide.harmonic_analysis_uv(
        xr.Dataset({"u": da_u, "v": da_v}), names=TIDE_NAMES, nodal=False
    )
    assert "semi_major" in coef
    with pytest.raises(XoaError):
        xtide.harmonic_analysis(xr.Dataset({"foo": da}))
