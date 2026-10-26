"""
Test the :mod:`xoa.core.tiles` module
"""

import numpy as np

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
