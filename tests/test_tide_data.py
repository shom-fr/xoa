"""
Test the :mod:`xoa.tide.data` module
"""

import numpy as np
import pytest
import xarray as xr
from validate import VdtValueError

import xoa.tide as xtide
from xoa.tide.data import is_satellites


def test_is_satellites():
    assert is_satellites("") == []
    assert is_satellites("0.1 1 0 0 0 ; -0.2 0 2 0 0") == [(0.1, 1, 0, 0, 0), (-0.2, 0, 2, 0, 0)]
    with pytest.raises(VdtValueError):
        is_satellites("0.1 1 0")


@pytest.fixture(autouse=True)
def reset_constituents():
    """Do not leak the constituents that a test sets"""
    yield
    xtide.reset_constituents()


MSF_CFG = """
[MSf]
doodson = 073.555
species = 0

[2MS6]
doodson = 673.555
species = 6
[[compound_of]]
[[[M2]]]
multiplier = 2
[[[S2]]]
multiplier = 1
"""


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "tide.cfg"
    path.write_text(MSF_CFG)
    return path


def test_constituent_db():
    db = xtide.get_constituents()
    assert isinstance(db, xtide.ConstituentDB)
    assert {"M2", "S2", "K1", "O1", "M4"} <= set(xtide.get_constituent_names())
    assert db.names == xtide.get_constituent_names()
    assert db["M4"]["compound_of"] == [("M2", 2)]
    assert db["M2"]["compound_of"] is None
    assert len(db["K1"]["satellites"]) == 4


def test_set_constituents_extend(cfg_file):
    default_names = xtide.get_constituent_names()
    db = xtide.set_constituents(cfg_file).db
    assert xtide.get_constituents() is db
    assert xtide.get_constituent_names() == default_names + ["MSf", "2MS6"]
    assert db["2MS6"]["compound_of"] == [("M2", 2), ("S2", 1)]
    period = 2 * np.pi / xtide.get_frequencies(["MSf"]) / 86400
    np.testing.assert_allclose(period, 14.765, atol=1e-3)
    assert xtide.get_default_constituents().names == default_names  # unchanged
    xtide.reset_constituents()
    assert xtide.get_constituent_names() == default_names


def test_set_constituents_context(cfg_file):
    default_names = xtide.get_constituent_names()
    with xtide.set_constituents(cfg_file, replace=True) as db:
        assert db.names == ["MSf", "2MS6"]
        assert xtide.get_constituent_names() == ["MSf", "2MS6"]
        assert xtide.auto_constituents(400) == ["MSf"]
    assert xtide.get_constituent_names() == default_names
    with xtide.set_constituents(cfg_file):
        with xtide.set_constituents({"X1": {"doodson": "145.555", "species": 1}}):
            # Sources complete the default constituents, not the current ones
            assert "X1" in xtide.get_constituent_names()
            assert "MSf" not in xtide.get_constituent_names()
        assert "X1" not in xtide.get_constituent_names()
        assert "MSf" in xtide.get_constituent_names()


def test_set_constituents_override(tmp_path):
    path = tmp_path / "tide.cfg"
    path.write_text("[M2]\ndoodson = 255.555\nspecies = 2\n")
    with xtide.set_constituents(path):
        assert xtide.get_constituents()["M2"]["satellites"] == []  # replaced as a whole
    assert len(xtide.get_constituents()["M2"]["satellites"]) > 0


def test_set_constituents_error(tmp_path):
    path = tmp_path / "bad.cfg"
    path.write_text("[X1]\ndoodson = 145.555\nspecies = 1\nsatellites = 0.1 1 0\n")
    before = xtide.get_constituents()
    with pytest.raises(Exception):
        xtide.set_constituents(path)
    assert xtide.get_constituents() is before


def test_user_constituents_file(cfg_file, monkeypatch):
    monkeypatch.setattr(xtide.data, "get_user_cfg_file", lambda: str(cfg_file))
    xtide.reset_constituents()
    assert "MSf" in xtide.get_constituent_names()


def test_analysis_with_set_constituents(cfg_file):
    time = np.datetime64("2024-01-01") + np.arange(60 * 24) * np.timedelta64(1, "h")
    hours = np.arange(time.size)
    with xtide.set_constituents(cfg_file):
        omega = xtide.get_frequencies(["MSf"])[0] * 3600
        da = xr.DataArray(0.3 * np.cos(omega * hours), dims="time", coords={"time": time})
        out = xtide.harmonic_analysis(da, names=["M2", "S2", "MSf"], uncertainty=False)
    assert float(out.amp.sel(constituent="MSf")) == pytest.approx(0.3, abs=0.05)


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


def test_nodal_corrections_m2_and_diurnal_phases():
    t_ref = float(np.datetime64("2024-01-01", "s").astype(float))
    N = np.radians(20.875)  # mean longitude of the ascending node
    f, u = xtide.nodal_corrections(t_ref, ["M2", "N2"])
    np.testing.assert_allclose(f, 1 - 0.0373 * np.cos(N), atol=2e-3)
    np.testing.assert_allclose(u, -np.radians(2.14) * np.sin(N), atol=2e-3)
    # Greenwich equilibrium arguments of the diurnal constituents: t + h +/- 90, with t = 0 h
    h = np.radians(280.158)
    v0 = xtide.equilibrium_argument(["K1", "P1"], t_ref)
    np.testing.assert_allclose(np.cos(v0), np.cos([h + np.pi / 2, -h - np.pi / 2]), atol=1e-3)
