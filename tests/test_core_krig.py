"""
Test the :mod:`xoa.core.krig` module
"""

import numpy as np
import pytest

from xoa.core import krig as ckrig
from xoa.core.geo import EARTH_RADIUS, haversine


def formula(mtype, h, n, s, r):
    u = h / r
    if mtype == ckrig.LINEAR:
        return n + (s - n) * np.minimum(u, 1)
    if mtype == ckrig.EXPONENTIAL:
        return n + (s - n) * (1 - np.exp(-3 * u))
    if mtype == ckrig.GAUSSIAN:
        return n + (s - n) * (1 - np.exp(-3 * u**2))
    return n + (s - n) * np.where(u <= 1, 1.5 * u - 0.5 * u**3, 1)


@pytest.mark.parametrize(
    "mtype", [ckrig.LINEAR, ckrig.EXPONENTIAL, ckrig.GAUSSIAN, ckrig.SPHERICAL]
)
def test_core_krig_variogram(mtype):
    h = np.linspace(0, 30, 50)
    np.testing.assert_allclose(
        ckrig.variogram(h, mtype, 0.1, 1.0, 10.0), formula(mtype, h, 0.1, 1.0, 10.0)
    )


def test_core_krig_variogram_zero_range():
    np.testing.assert_allclose(
        ckrig.variogram(np.array([0.0, 1.0]), ckrig.SPHERICAL, 0.1, 1, 0), [0.1, 1]
    )


def get_points(n=40):
    rng = np.random.default_rng(1)
    lon, lat = rng.uniform(-5, -4, n), rng.uniform(47, 48, n)
    return lon, lat, np.sin(5 * (lon + 5))[None]


def get_neighbors(lon, lat, lono, lato):
    ineigh = np.tile(np.arange(lon.size), (lono.size, 1))
    dneigh = EARTH_RADIUS * haversine(lono[:, None], lato[:, None], lon[None], lat[None])
    return ineigh, dneigh


@pytest.mark.parametrize("ordinary", [True, False])
def test_core_krig_local_is_exact_at_observations(ordinary):
    lon, lat, values = get_points()
    ineigh, dneigh = get_neighbors(lon, lat, lon, lat)
    zo, vo = ckrig.krig_local(
        lon, lat, values, lon, lat, ineigh, dneigh, ckrig.EXPONENTIAL, 0.0, 1.0, 5e4, 3,
        ordinary, False, 0.0, EARTH_RADIUS,
    )  # fmt: skip
    np.testing.assert_allclose(zo, values, atol=1e-6)
    np.testing.assert_allclose(vo, 0, atol=1e-6)


def test_core_krig_local_missing_neighbors():
    lon, lat, values = get_points()
    ineigh, dneigh = get_neighbors(lon, lat, lon[:2], lat[:2])
    ineigh[0, 2:] = -1
    ineigh[1, 3:] = -1
    zo, vo = ckrig.krig_local(
        lon, lat, values, lon[:2], lat[:2], ineigh, dneigh, ckrig.EXPONENTIAL, 0.0, 1.0, 5e4, 3,
        True, False, 0.0, EARTH_RADIUS,
    )  # fmt: skip
    assert np.isnan(zo[0, 0]) and np.isnan(vo[0])
    assert np.isfinite(zo[0, 1]) and np.isfinite(vo[1])
