"""Unit tests for xoa.core.signal."""

import numpy as np
import pytest
from scipy.signal import find_peaks

from xoa.core.signal import (
    peak_mask,
    quantile_mask,
    window_max_at,
    window_min_at,
)


def _cases():
    rng = np.random.default_rng(3)
    t = np.arange(400.0)
    tide = np.cos(2 * np.pi * t / 12.4) + 0.3 * np.cos(2 * np.pi * t / 354.0)
    gaps = tide + 0.1 * rng.standard_normal(t.size)
    gaps[50:58] = np.nan
    gaps[300] = np.nan
    return {
        "tide": tide,
        "noisy": tide + 0.3 * rng.standard_normal(t.size),
        "white": rng.standard_normal(t.size),
        "flat_tops": np.round(tide, 1),  # plateaus and ties
        "gaps": gaps,
        "monotonic": t,
        "short": tide[:5],
        "constant": np.ones(30),
    }


def _ref_peaks(x, distance):
    mask = np.zeros(x.shape, dtype=bool)
    mask[find_peaks(x, distance=distance)[0]] = True
    return mask


class TestPeakMask:
    @pytest.mark.parametrize("name", list(_cases()))
    @pytest.mark.parametrize("distance", [1, 5, 9, 15])
    def test_matches_scipy_find_peaks(self, name, distance):
        x = _cases()[name]
        got, ref = peak_mask(x, distance), _ref_peaks(x, distance)
        if name == "flat_tops" and distance == 15:
            # equal heights: scipy's unstable sort picks another, equally valid, set
            assert got.sum() == ref.sum() and np.diff(np.flatnonzero(got)).min() >= distance
        else:
            assert np.array_equal(got, ref)

    def test_flat_top_counts_once_at_its_middle(self):
        x = np.array([0.0, 1, 2, 2, 2, 1, 0, 1, 0])
        assert np.flatnonzero(peak_mask(x)).tolist() == [3, 7]

    def test_points_are_independent_in_nd_fields(self):
        rng = np.random.default_rng(5)
        field = rng.standard_normal((120, 4, 5))
        got = peak_mask(field, 6)
        assert got.shape == field.shape
        for j in range(4):
            for i in range(5):
                assert np.array_equal(got[:, j, i], _ref_peaks(field[:, j, i], 6))

    def test_non_contiguous_and_float32_inputs(self):
        x = _cases()["noisy"]
        field = np.stack([x, x[::-1]], axis=1)[:, ::-1]  # non-contiguous view
        assert np.array_equal(peak_mask(field)[:, 0], _ref_peaks(field[:, 0], 1))
        f32 = field.astype(np.float32)
        assert np.array_equal(peak_mask(f32), peak_mask(f32.astype(float)))


class TestWindows:
    def test_window_max_and_min_match_numpy_with_nans(self):
        x = _cases()["gaps"]
        mask = np.zeros(x.size, bool)
        mask[[0, 2, 3, 53, 54, 120, 299, 300, 397, 399]] = True
        hmax, hmin = window_max_at(x, mask, 3), window_min_at(x, mask, 3)
        for t in range(x.size):
            window = x[t - 3 : t + 4] if mask[t] and 3 <= t < x.size - 3 else np.array([np.nan])
            if np.isnan(window).all():
                assert np.isnan(hmax[t]) and np.isnan(hmin[t])
            else:
                assert hmax[t] == np.nanmax(window) and hmin[t] == np.nanmin(window)

    def test_ref_gives_the_excess_beyond_the_reference(self):
        rng = np.random.default_rng(4)
        x, ref = rng.standard_normal((2, 60, 3))
        mask = np.zeros(x.shape, bool)
        mask[10:50:5] = True
        up, down = window_max_at(x, mask, 2, ref=ref), window_min_at(x, mask, 2, ref=ref)
        np.testing.assert_allclose(up, window_max_at(x, mask, 2) - ref)
        np.testing.assert_allclose(down, ref - window_min_at(x, mask, 2))
        assert (up[mask] >= x[mask] - ref[mask]).all() and (down[mask] >= ref[mask] - x[mask]).all()

    def test_unflagged_samples_stay_nan(self):
        x = np.arange(20.0)
        out = window_max_at(x, np.zeros(20, bool), 2)
        assert np.isnan(out).all()


class TestQuantileMask:
    @pytest.mark.parametrize("q", [0.0, 0.5, 0.75, 1.0])
    def test_matches_nanquantile_per_point(self, q):
        rng = np.random.default_rng(7)
        field = rng.standard_normal((80, 3, 4))
        field[::7, 1, 2] = np.nan
        got = quantile_mask(field, q)
        for j in range(3):
            for i in range(4):
                col = field[:, j, i]
                assert np.array_equal(got[:, j, i], col >= np.nanquantile(col, q))

    def test_all_nan_series_selects_nothing(self):
        assert not quantile_mask(np.full((10, 2), np.nan), 0.5).any()


# %% TestDetectWiggles1d
