"""
Test the :mod:`xoa.core.tiles` module
"""

import numpy as np
import pytest

from xoa import exceptions
from xoa.core import tiles


def test_get_grid_indices():
    ind, err = tiles.get_grid_indices(np.array([10.0, 10.5, 11.01]), 10.0, 0.5)
    np.testing.assert_array_equal(ind, [0, 1, 2])
    assert err < 0.03
    assert tiles.get_grid_indices(np.array([10.25]), 10.0, 0.5)[1] == 0.5


def _paste_(mode, first, second):
    out = np.full((1, 3, 3), np.nan)
    count = np.zeros((1, 3, 3), "i") if mode == "mean" else np.zeros((1, 1, 1), "i")
    mode = tiles.OVERLAPS[mode]
    tiles.paste(out, first, 0, 0, mode, count)
    nconf = tiles.paste(out, second, 1, 1, mode, count)
    return out, count, nconf


def test_paste_modes():
    first = np.ones((1, 2, 2))
    second = np.full((1, 2, 2), 3.0)
    second[0, 1, 1] = np.nan
    out, _, _ = _paste_("last", first, second)
    assert out[0, 1, 1] == 3 and out[0, 0, 0] == 1 and np.isnan(out[0, 2, 2])
    out, _, _ = _paste_("first", first, second)
    assert out[0, 1, 1] == 1 and out[0, 1, 2] == 3
    out, count, _ = _paste_("mean", first, second)
    tiles.finalize_mean(out, count)
    assert out[0, 1, 1] == 2 and out[0, 1, 2] == 3
    _, _, nconf = _paste_("raise", first, second)
    assert nconf == 1
    _, _, nconf = _paste_("raise", first, first)
    assert nconf == 0


def test_fit_regular_axis():
    # Single precision coordinates on a large grid must not drift
    full = (-170.0 + 0.003379 * np.arange(100000)).astype("f")
    axes = [full[:40000], full[39000:70000], full[70000:]]
    step, origin, size = tiles.fit_regular_axis(axes)
    assert size == full.size
    assert origin == full[0]
    np.testing.assert_allclose(step, 0.003379, rtol=1e-5)
    assert tiles.fit_regular_axis(axes, step=0.003379, origin=-171.0)[2] > size
    assert tiles.fit_regular_axis([np.array([1.0, 3.0]), np.array([5.0])], size=9)[2] == 9


def test_fit_regular_axis_errors():
    with pytest.raises(exceptions.XoaError, match="Cannot infer the lat step"):
        tiles.fit_regular_axis([np.array([1.0])], name="lat")
    with pytest.raises(exceptions.XoaError, match="before"):
        tiles.fit_regular_axis([np.array([1.0, 2.0])], step=1.0, origin=2.5)


def test_wrap_lon():
    np.testing.assert_allclose(tiles.wrap_lon([-10.0, 10.0, 350.0, 370.0], 0.0), [350, 10, 350, 10])
    np.testing.assert_allclose(tiles.wrap_lon([170.0, 190.0], -180.0), [170, -170])


def test_get_lon_window_start():
    # The window starts at the tile after the largest gap, whatever the convention
    axes = [np.arange(340.0, 360), np.arange(0.0, 20)]
    assert tiles.get_lon_window_start(axes) == pytest.approx(340, abs=0.01)
    axes = [np.arange(-20.0, 0), np.arange(0.0, 20), np.arange(340.0, 360)]
    assert tiles.get_lon_window_start(axes) == pytest.approx(340, abs=0.01)
    axes = [np.arange(0.0, 10), np.arange(100.0, 110)]
    assert tiles.get_lon_window_start(axes) == pytest.approx(0, abs=0.01)
    # Full circle
    axes = [np.arange(0.0, 180), np.arange(180.0, 360)]
    assert tiles.get_lon_window_start(axes, default=-180.0) == -180.0
