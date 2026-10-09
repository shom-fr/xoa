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
