# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.stats` module
"""

import numpy as np

from xoa.core import stats as cstats


def test_taylor_stats_against_numpy():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=100)
    model = np.array([0.8 * ref + 0.3 * rng.normal(size=100), -ref, 2 * ref])
    std, std_ref, corr, crmsd = cstats.taylor_stats(model, ref)
    for i in range(3):
        assert np.isclose(std[i], model[i].std())
        assert np.isclose(std_ref[i], ref.std())
        assert np.isclose(corr[i], np.corrcoef(model[i], ref)[0, 1])
        assert np.isclose(crmsd[i], (model[i] - model[i].mean() - ref + ref.mean()).std())
    assert np.allclose(crmsd**2, std**2 + std_ref**2 - 2 * std * std_ref * corr)
    assert corr[1] < 0


def test_taylor_stats_nans_and_1d():
    ref = np.array([1.0, 2, 3, 4, np.nan, 6])
    model = np.array([1.0, 2, np.nan, 4, 5, 7])
    std, std_ref, corr, crmsd = cstats.taylor_stats(model, ref)
    ok = ~np.isnan(ref) & ~np.isnan(model)
    assert std.shape == (1,)
    assert np.isclose(std[0], model[ok].std())
    assert np.isclose(corr[0], np.corrcoef(model[ok], ref[ok])[0, 1])


def test_taylor_stats_degenerate():
    std, _, corr, _ = cstats.taylor_stats(np.ones((2, 4)), np.arange(4.0))
    assert np.allclose(std, 0)
    assert np.isnan(corr).all()
    std, _, corr, _ = cstats.taylor_stats(np.full((1, 3), np.nan), np.arange(3.0))
    assert np.isnan(std).all() and np.isnan(corr).all()


def test_moments_against_numpy():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(4, 60))
    y = 0.5 * x + rng.normal(size=(4, 60))
    x[0, 5] = np.nan
    y[1, 7] = np.nan
    stats = cstats.moments2stats(cstats.get_moments(x, y), dual=True)
    for i in range(4):
        ok = np.isfinite(x[i]) & np.isfinite(y[i])
        xi, yi = x[i, ok], y[i, ok]
        assert stats["count"][i] == ok.sum()
        assert np.isclose(stats["mean0"][i], xi.mean())
        assert np.isclose(stats["std1"][i], yi.std())
        assert np.isclose(stats["var0"][i], xi.var())
        assert np.isclose(stats["min0"][i], xi.min())
        assert np.isclose(stats["max1"][i], yi.max())
        assert np.isclose(stats["bias"][i], (yi - xi).mean())
        assert np.isclose(stats["rms"][i], np.sqrt(((yi - xi) ** 2).mean()))
        assert np.isclose(stats["crms"][i], (yi - xi).std())
        assert np.isclose(stats["corr"][i], np.corrcoef(xi, yi)[0, 1])
        assert np.isclose(stats["cov"][i], np.cov(xi, yi, ddof=0)[0, 1])
    stats = cstats.moments2stats(cstats.get_moments(x), ddof=1)
    assert np.isclose(stats["std"][2], x[2].std(ddof=1))


def test_moments_weighted_analytic():
    # Linear field on layers whose thickness is proportional to the depth h
    nz = 10
    h = np.array([2.0, 5.0, 11.0])
    z = -h[:, None] * (np.arange(nz) + 0.5) / nz
    dz = np.broadcast_to(h[:, None] / nz, z.shape)
    stats = cstats.moments2stats(cstats.get_moments(2 * z, w=dz))
    assert np.allclose(stats["mean"], -h)
    assert np.allclose(stats["var"], 4 * h**2 * (nz**2 - 1) / (12 * nz**2))
    # All columns together, which needs the weights
    stats = cstats.moments2stats(cstats.get_moments(z.reshape(1, -1), w=dz.reshape(1, -1)))
    assert np.isclose(stats["mean"][0], -(h**2).sum() / (2 * h.sum()))
    stats = cstats.moments2stats(cstats.get_moments(z.reshape(1, -1)))
    assert not np.isclose(stats["mean"][0], -(h**2).sum() / (2 * h.sum()))


def test_moments_empty_and_merge():
    x = np.array([[np.nan, np.nan, np.nan], [1.0, 2.0, 3.0]])
    mom = cstats.get_moments(x)
    stats = cstats.moments2stats(mom)
    assert stats["count"][0] == 0
    assert np.isnan(stats["mean"][0]) and np.isnan(stats["min"][0]) and np.isnan(stats["std"][0])
    y = 2 * x + 1
    mom = cstats.get_moments(x, y)
    merged = cstats.merge_moments(
        cstats.get_moments(x[:, :1], y[:, :1]), cstats.get_moments(x[:, 1:], y[:, 1:])
    )
    assert np.allclose(merged, mom)
    # Invalid weights are ignored
    w = np.array([[1.0, 1, 1], [0.0, np.nan, 2.0]])
    assert cstats.moments2stats(cstats.get_moments(x, w=w))["mean"][1] == 3.0


def test_moments_stable_with_large_offset():
    # Raw sums of squares lose all digits of such a small variance
    rng = np.random.default_rng(2)
    x = 1e9 + rng.normal(size=(1, 1000))
    mom = cstats.merge_moments(cstats.get_moments(x[:, :300]), cstats.get_moments(x[:, 300:]))
    stats = cstats.moments2stats(mom)
    assert np.isclose(stats["var"][0], x.var(), rtol=1e-6)
    assert np.isclose(stats["mean"][0], x.mean(), rtol=1e-14)
