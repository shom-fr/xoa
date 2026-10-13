"""
Test the :mod:`xoa.krig` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import krig as xkrig


def get_obs(n=30):
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(-5, -4, n), rng.uniform(47, 48, n)
    values = np.sin(5 * (lon + 5)) + np.cos(5 * (lat - 47))
    return xr.DataArray(
        values, dims="npts", coords={"lon": ("npts", lon), "lat": ("npts", lat)}, name="temp"
    )


@pytest.mark.parametrize("dist_units", ["m", "km"])
def test_krig_kriger_is_exact_at_observations(dist_units):
    """Same distance units for the input and output points, whatever they are"""
    obs = get_obs()
    scale = 1.0 if dist_units == "km" else 1e3
    func = lambda d: 1.0 - np.exp(-3 * d / (50 * scale))  # noqa: E731
    kriger = xkrig.Kriger(obs, "ordinary", func, dist_units=dist_units)
    target = xr.Dataset(coords={"lon": obs.lon, "lat": obs.lat})
    out = kriger.interp(target)
    np.testing.assert_allclose(out.temp.values, obs.values, atol=1e-6)


def test_krig_empirical_variogram_and_fit():
    obs = get_obs(60)
    ev = xkrig.empirical_variogram(obs, nbin=10, dist_units="km")
    assert ev.name == "semivariogram"
    assert ev.sizes["dist"] > 1
    model = xkrig.VariogramModel("spherical", dist_units="km")
    model.fit(ev)
    assert model["s"] > 0
    assert model["r"] > 0
