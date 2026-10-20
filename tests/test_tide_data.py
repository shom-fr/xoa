# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.tide.data` module
"""

import numpy as np
import pytest
from validate import VdtValueError

import xoa.tide as xtide
from xoa.tide.data import is_satellites


def test_is_satellites():
    assert is_satellites("") == []
    assert is_satellites("0.1 1 0 0 0 ; -0.2 0 2 0 0") == [(0.1, 1, 0, 0, 0), (-0.2, 0, 2, 0, 0)]
    with pytest.raises(VdtValueError):
        is_satellites("0.1 1 0")


def test_constituent_db():
    assert {"M2", "S2", "K1", "O1", "M4"} <= set(xtide.CONSTITUENT_NAMES)
    assert xtide.CONSTITUENT_DB["M4"]["compound_of"] == [("M2", 2)]
    assert xtide.CONSTITUENT_DB["M2"]["compound_of"] is None
    assert len(xtide.CONSTITUENT_DB["K1"]["satellites"]) == 4


def test_frequencies():
    omega = xtide.get_frequencies(["M2", "K1"])
    np.testing.assert_allclose(2 * np.pi / omega / 3600, [12.4206, 23.9345], atol=1e-3)


def test_auto_constituents_and_beats():
    short, long = xtide.auto_constituents(15), xtide.auto_constituents(365)
    assert set(short) < set(long)
    assert "M4" not in long
    assert xtide.beat_period("S2", "K2") == pytest.approx(182.6, abs=0.5)
    assert ("S2", "K2") in xtide.unresolved_pairs(["S2", "K2", "M2"], 100)
    assert ("S2", "K2") not in xtide.unresolved_pairs(["S2", "K2", "M2"], 400)


def test_nodal_corrections():
    t_ref = float(np.datetime64("2024-01-01", "s").astype(float))
    f, u = xtide.nodal_corrections(t_ref, ["M2", "S2", "M4"])
    assert 0.9 < f[0] < 1.05 and f[1] == 1 and u[1] == 0
    np.testing.assert_allclose([f[2], u[2]], [f[0] ** 2, 2 * u[0]])
    assert xtide.equilibrium_argument(["M2"], t_ref).shape == (1,)
