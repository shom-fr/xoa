# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.tide.plot` module
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pytest

import xoa.tide as xtide
from xoa.tide import plot as tplot

pytest.importorskip("cartopy")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def coef_ssh(tide_ssh):
    da = tide_ssh[0].assign_coords(
        lon=("x", np.linspace(-6.0, -4.0, tide_ssh[0].sizes["x"])),
        lat=("y", np.linspace(47.0, 48.0, tide_ssh[0].sizes["y"])),
    )
    return xtide.harmonic_analysis(da, names=["M2", "S2", "K1", "O1"], nodal=False)


def test_plot_tidal_ellipse(tide_coef_uv):
    point = tide_coef_uv.isel(x=1, y=1)
    ax = tplot.plot_tidal_ellipse(point, "M2")
    assert ax.get_title() == "M2 tidal current ellipse"
    assert tplot.plot_tidal_ellipse(point, "S2", annotate=False, color="k") is not None
    with pytest.raises(Exception):
        tplot.plot_tidal_ellipse(point)


@pytest.mark.parametrize("color_by", ["rotation", "phase", "ellipticity", "semi_major", "green"])
def test_plot_tidal_ellipses(tide_coef_uv, color_by):
    lc = tplot.plot_tidal_ellipses(tide_coef_uv, "M2", color_by=color_by, ref=0.5)
    assert isinstance(lc, mcollections.LineCollection)
    assert len(lc.get_segments()) == tide_coef_uv.semi_major.isel(constituent=0).size
    assert lc.axes.get_title() == "M2 tidal current ellipses"


def test_plot_tidal_ellipses_options(tide_coef_uv):
    smaj = tide_coef_uv.semi_major.sel(constituent="M2")
    lc = tplot.plot_tidal_ellipses(
        tide_coef_uv, "M2", step=2, color_by=smaj, phase_ticks=False, background=smaj, title="T"
    )
    assert len(lc.get_segments()) == 4  # 2 x 2 points
    assert lc.axes.get_title() == "T"
    land = tide_coef_uv.where(tide_coef_uv.lon > 100)
    with pytest.raises(Exception):
        tplot.plot_tidal_ellipses(land, "M2")


def test_plot_tidal_constituents(tide_coef_uv):
    fig, axes = tplot.plot_tidal_constituents(
        tide_coef_uv, ["M2", "S2", "K1"], ncols=2, color_by="phase", scale={"M2": 0.1}
    )
    assert len(axes) == 3 and len(fig.axes) == 4  # one shared colorbar
    fig, axes = tplot.plot_tidal_constituents(tide_coef_uv, ["M2", "S2"])
    assert len(axes) == 2


def test_plot_cotidal(coef_ssh):
    ax = tplot.plot_cotidal(coef_ssh, "M2")
    assert ax.get_title() == "M2 co-tidal chart"
    with pytest.raises(Exception):
        tplot.plot_cotidal(coef_ssh.isel(x=0), "M2")


@pytest.mark.parametrize("var", ["ssh", "uv"])
def test_plot_tidal_spectrum(var, coef_ssh, tide_coef_uv):
    coef = coef_ssh if var == "ssh" else tide_coef_uv
    ax = tplot.plot_tidal_spectrum(coef, record_days=30, reduce="max", label_min=0.1)
    assert ax.get_xlabel().startswith("Frequency")
    assert ax.get_ylabel().startswith("Amplitude" if var == "ssh" else "Semi-major")
