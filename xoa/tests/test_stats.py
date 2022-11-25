"""
Test the :mod:`xoa.stats` module
"""

import numpy as np
import xarray as xr

import xoa.stats as xstats


def test_stats_stataccum():
    np.random.seed(0)

    da0 = xr.DataArray(np.random.normal(size=(4, 5)), dims=("time", "lon"))
    da1 = xr.DataArray(np.random.normal(size=(4, 5)), dims=("time", "lon"))

    sa0 = xstats.StatAccum(temporal=True, spatial=True)
    sa0 += da0, da1
    stats0 = sa0.get_stats()

    # Content
    for dim, pre in [("time", "t"), ("lon", "s")]:
        xr.testing.assert_allclose(stats0[pre + "mean0"], da0.mean(dim))
        xr.testing.assert_allclose(stats0[pre + "std0"], da0.std(dim))
        xr.testing.assert_allclose(stats0[pre + "var0"], da0.var(dim))
        xr.testing.assert_allclose(stats0[pre + "min0"], da0.min(dim))
        xr.testing.assert_allclose(stats0[pre + "max0"], da0.max(dim))
        xr.testing.assert_allclose(stats0[pre + "count"], da0.count(dim))
        xr.testing.assert_allclose(stats0[pre + "cov"], xr.cov(da0, da1, dim, ddof=0))
        xr.testing.assert_allclose(stats0[pre + "corr"], xr.corr(da0, da1, dim))

    # Split
    sa1 = xstats.StatAccum(temporal=True, spatial=True)
    sa1 += da0[:2], da1[:2]
    sa1 += da0[2:], da1[2:]
    stats1 = sa1.get_stats()
    for name in stats0:
        xr.testing.assert_allclose(stats0[name], stats1[name])
