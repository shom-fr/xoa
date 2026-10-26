import numpy as np
import pytest
import xarray as xr

import xoa.tide as xtide
from xoa.core import tide as ctide

collect_ignore = ["test_cf.py"]

TIDE_NAMES = ["M2", "S2", "K1", "O1"]
TIDE_NT, TIDE_NY, TIDE_NX = 720, 3, 4  # 30 days of hourly data
TIDE_TIME = np.datetime64("2024-01-01", "s") + np.arange(TIDE_NT) * np.timedelta64(3600, "s")


def make_tide_ssh(seed=0):
    """Synthetic sea level (time, y, x) and its true mean, amplitudes and phases"""
    rng = np.random.default_rng(seed)
    nf, ny, nx = len(TIDE_NAMES), TIDE_NY, TIDE_NX
    omega = xtide.get_frequencies(TIDE_NAMES)
    h0 = rng.uniform(0.1, 0.5, (ny, nx))
    amp = rng.uniform(0.1, 2.0, (nf, ny, nx))
    phase = rng.uniform(0, 2 * np.pi, (nf, ny, nx))
    h = ctide.predict_ssh(
        np.arange(TIDE_NT) * 3600.0, omega, h0.ravel(), amp.reshape(nf, -1), phase.reshape(nf, -1)
    )
    da = xr.DataArray(
        h.reshape(TIDE_NT, ny, nx), dims=("time", "y", "x"), coords={"time": TIDE_TIME}
    )
    return da, h0, amp, phase


def make_tide_uv(seed=1):
    """Synthetic currents (time, y, x) and their true semi-major and semi-minor axes"""
    rng = np.random.default_rng(seed)
    nf, ny, nx = len(TIDE_NAMES), TIDE_NY, TIDE_NX
    ns = ny * nx
    omega = xtide.get_frequencies(TIDE_NAMES)
    a = rng.uniform(0.1, 0.5, (nf, ns))
    b = a * rng.uniform(0.0, 1.0, (nf, ns))
    smaj, smin = a + b, a - b
    incl = rng.uniform(-np.pi / 2, np.pi / 2, (nf, ns))
    phase = rng.uniform(0, 2 * np.pi, (nf, ns))
    u0, v0 = rng.uniform(-0.05, 0.05, (2, ns))
    u, v = ctide.predict_currents(
        np.arange(TIDE_NT) * 3600.0, omega, u0, v0, smaj, smin, incl, phase
    )
    dims = ("time", "y", "x")
    da_u = xr.DataArray(u.reshape(TIDE_NT, ny, nx), dims=dims, coords={"time": TIDE_TIME})
    da_v = xr.DataArray(v.reshape(TIDE_NT, ny, nx), dims=dims, coords={"time": TIDE_TIME})
    return da_u, da_v, smaj.reshape(nf, ny, nx), smin.reshape(nf, ny, nx)


@pytest.fixture
def tide_ssh():
    return make_tide_ssh()


@pytest.fixture
def tide_uv():
    return make_tide_uv()


@pytest.fixture
def tide_coef_uv(tide_uv):
    """Coefficients of the analysis of the synthetic currents, with lon/lat coordinates"""
    da_u, da_v, *_ = tide_uv
    coords = dict(
        lon=("x", np.linspace(-6.0, -4.0, TIDE_NX)), lat=("y", np.linspace(47.0, 48.0, TIDE_NY))
    )
    return xtide.harmonic_analysis_uv(
        da_u.assign_coords(coords), da_v.assign_coords(coords), names=TIDE_NAMES, nodal=False
    )
