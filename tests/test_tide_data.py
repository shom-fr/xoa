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


MSM_CFG = """
[MSm]
doodson = 063.655
species = 0

[2MK6]
doodson = 675.555
species = 6
[[compound_of]]
[[[M2]]]
multiplier = 2
[[[K2]]]
multiplier = 1
"""


@pytest.fixture
def cfg_file(tmp_path):
    path = tmp_path / "tide.cfg"
    path.write_text(MSM_CFG)
    return path


def test_constituent_db():
    db = xtide.get_constituents()
    assert isinstance(db, xtide.ConstituentDB)
    assert {"M2", "S2", "K1", "O1", "M4"} <= set(xtide.get_constituent_names())
    assert db.names == xtide.get_constituent_names()
    assert db["M4"]["compound_of"] == [("M2", 2)]
    assert db["M2"]["compound_of"] is None
    assert len(db["K1"]["satellites"]) == 5


def test_set_constituents_extend(cfg_file):
    default_names = xtide.get_constituent_names()
    db = xtide.set_constituents(cfg_file).db
    assert xtide.get_constituents() is db
    assert xtide.get_constituent_names() == default_names + ["MSm", "2MK6"]
    assert db["2MK6"]["compound_of"] == [("M2", 2), ("K2", 1)]
    period = 2 * np.pi / xtide.get_frequencies(["MSm"]) / 86400
    np.testing.assert_allclose(period, 31.812, atol=1e-3)
    assert xtide.get_default_constituents().names == default_names  # unchanged
    xtide.reset_constituents()
    assert xtide.get_constituent_names() == default_names


def test_set_constituents_context(cfg_file):
    default_names = xtide.get_constituent_names()
    with xtide.set_constituents(cfg_file, replace=True) as db:
        assert db.names == ["MSm", "2MK6"]
        assert xtide.get_constituent_names() == ["MSm", "2MK6"]
        assert xtide.auto_constituents(400) == ["MSm"]
    assert xtide.get_constituent_names() == default_names
    with xtide.set_constituents(cfg_file):
        with xtide.set_constituents({"X1": {"doodson": "145.555", "species": 1}}):
            # Sources complete the default constituents, not the current ones
            assert "X1" in xtide.get_constituent_names()
            assert "MSm" not in xtide.get_constituent_names()
        assert "X1" not in xtide.get_constituent_names()
        assert "MSm" in xtide.get_constituent_names()


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
    assert "MSm" in xtide.get_constituent_names()


def test_analysis_with_set_constituents(cfg_file):
    time = np.datetime64("2024-01-01") + np.arange(60 * 24) * np.timedelta64(1, "h")
    hours = np.arange(time.size)
    with xtide.set_constituents(cfg_file):
        omega = xtide.get_frequencies(["MSm"])[0] * 3600
        da = xr.DataArray(0.3 * np.cos(omega * hours), dims="time", coords={"time": time})
        out = xtide.harmonic_analysis(da, names=["M2", "S2", "MSm"], uncertainty=False)
    assert float(out.amp.sel(constituent="MSm")) == pytest.approx(0.3, abs=0.05)


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
    f, u = xtide.nodal_corrections(t_ref, ["M2", "T2", "M4"])
    assert 0.9 < f[0] < 1.05 and f[1] == 1 and u[1] == 0
    np.testing.assert_allclose([f[2], u[2]], [f[0] ** 2, 2 * u[0]])
    assert xtide.equilibrium_argument(["M2"], t_ref).shape == (1,)


def test_nodal_corrections_m2_and_diurnal_phases():
    t_ref = float(np.datetime64("2024-01-01", "s").astype(float))
    N = np.radians(20.875)  # mean longitude of the ascending node
    f, u = xtide.nodal_corrections(t_ref, ["M2", "N2"])
    np.testing.assert_allclose(f, 1 - 0.0373 * np.cos(N), atol=2e-3)
    np.testing.assert_allclose(u, -np.radians(2.14) * np.sin(N), atol=6e-3)
    # Greenwich equilibrium arguments of the diurnal constituents: t + h +/- 90, with t = 0 h
    h = np.radians(280.158)
    v0 = xtide.equilibrium_argument(["K1", "P1"], t_ref)
    np.testing.assert_allclose(np.cos(v0), np.cos([h + np.pi / 2, -h - np.pi / 2]), atol=1e-3)


def test_nodal_corrections_vectorized():
    names = ["M2", "K1", "M4"]
    t = float(np.datetime64("2024-01-01", "s").astype(float)) + np.array([0, 1e7, 1e8])
    f, u = xtide.nodal_corrections(t, names)
    assert f.shape == u.shape == (3, 3)
    for i, ti in enumerate(t):
        fi, ui = xtide.nodal_corrections(ti, names)
        np.testing.assert_allclose(f[:, i], fi)
        np.testing.assert_allclose(u[:, i], ui)
    assert xtide.equilibrium_argument(names, t).shape == (3, 3)


def test_nodal_corrections_negative_multiplier(tmp_path):
    path = tmp_path / "tide.cfg"
    path.write_text(
        "[MSf2]\ndoodson = 073.555\nspecies = 0\n"
        "[[compound_of]]\n[[[S2]]]\nmultiplier = 1\n[[[M2]]]\nmultiplier = -1\n"
    )
    with xtide.set_constituents(path) as db:
        assert db["MSf2"]["compound_of"] == [("S2", 1), ("M2", -1)]
        t_ref = float(np.datetime64("2024-01-01", "s").astype(float))
        f, u = xtide.nodal_corrections(t_ref, ["MSf2", "M2", "S2"])
    np.testing.assert_allclose([f[0], u[0]], [f[1] * f[2], u[2] - u[1]])


@pytest.mark.parametrize("date", ["2011-06-15", "2015-01-01", "2020-01-01", "2024-07-01"])
def test_nodal_corrections_vs_schureman(date):
    """Nodal corrections close to the series of Schureman (1958), table 14"""
    from xoa.core.tide import astronomical_args

    t = float(np.datetime64(date, "s").astype(float))
    N = np.deg2rad(astronomical_args(t)[3])
    c1, c2, c3 = np.cos(N), np.cos(2 * N), np.cos(3 * N)
    s1, s2, s3 = np.sin(N), np.sin(2 * N), np.sin(3 * N)
    ref = {  # f, u in degrees
        "M2": (1.0004 - 0.0373 * c1 + 0.0002 * c2, -2.14 * s1),
        "K2": (
            1.0241 + 0.2863 * c1 + 0.0083 * c2 - 0.0015 * c3,
            -17.74 * s1 + 0.68 * s2 - 0.04 * s3,
        ),
        "K1": (
            1.0060 + 0.1150 * c1 - 0.0088 * c2 + 0.0006 * c3,
            -8.86 * s1 + 0.68 * s2 - 0.07 * s3,
        ),
        "O1": (
            1.0089 + 0.1871 * c1 - 0.0147 * c2 + 0.0014 * c3,
            10.80 * s1 - 1.34 * s2 + 0.19 * s3,
        ),
        "Mf": (1.043 + 0.414 * c1, -23.74 * s1 + 2.68 * s2 - 0.38 * s3),
        "Mm": (1.000 - 0.130 * c1, 0.0),
    }
    f, u = xtide.nodal_corrections(t, list(ref))
    np.testing.assert_allclose(f, [v[0] for v in ref.values()], atol=0.01)
    np.testing.assert_allclose(np.rad2deg(u), [v[1] for v in ref.values()], atol=0.5)
