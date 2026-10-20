"""
Signal processing kernels: peaks, windows and quantiles of series.

The numba kernels (peaks, windows, quantiles, rolling range) work on regularly
sampled time series and take arrays with time as **first** axis, ``(nt, ...)``:
the other axes (points, grid, depth) are flattened without a copy to
``(nt, npoints)`` and the points are processed independently and in parallel.
A 1-D series ``(nt,)`` is a field with one point. NaNs are skipped.
"""

# Copyright 2020-2026 Shom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numba as nb
import numpy as np

# 8 float64 share a cache line: points are processed by blocks of BLOCK so that
# the strided copy of the time series of a block reads the memory once.
BLOCK = 8

# %% peaks


@nb.njit(cache=True, nogil=True)
def _find_peaks(x, distance):
    # Indices of the local maxima of x, a flat top counting once (middle
    # sample), then thinned so that two kept maxima are at least ``distance``
    # samples apart, the highest ones having priority. Same semantics as
    # scipy.signal.find_peaks(x, distance=distance); edges are never maxima.
    n = x.size
    peaks = np.empty(n // 2 + 1, np.int64)
    m = 0
    i = 1
    while i < n - 1:
        if x[i - 1] < x[i]:
            ahead = i + 1
            while ahead < n - 1 and x[ahead] == x[i]:
                ahead += 1
            if x[ahead] < x[i]:
                peaks[m] = (i + ahead - 1) // 2
                m += 1
                i = ahead
        i += 1
    peaks = peaks[:m]
    if m < 2 or distance < 2:
        return peaks
    order = np.argsort(x[peaks], kind="mergesort")
    keep = np.ones(m, np.bool_)
    for r in range(m - 1, -1, -1):
        j = order[r]
        if not keep[j]:
            continue
        k = j - 1
        while k >= 0 and peaks[j] - peaks[k] < distance:
            keep[k] = False
            k -= 1
        k = j + 1
        while k < m and peaks[k] - peaks[j] < distance:
            keep[k] = False
            k += 1
    return peaks[keep]


@nb.njit(parallel=True, cache=True)
def _peak_mask_kernel(x, distance):
    nt, ns = x.shape
    out = np.zeros((nt, ns), np.bool_)
    for b in nb.prange((ns + BLOCK - 1) // BLOCK):
        for p in range(b * BLOCK, min((b + 1) * BLOCK, ns)):
            peaks = _find_peaks(x[:, p].copy(), distance)
            for q in range(peaks.size):
                out[peaks[q], p] = True
    return out


# %% windows


@nb.njit(parallel=True, cache=True)
def _window_extreme_kernel(x, mask, ref, use_ref, half_window, sign):
    # sign=+1: window maximum, sign=-1: window minimum (NaNs skipped). With
    # use_ref, returns the excess of the extreme beyond ref: max - ref, ref - min.
    nt, ns = x.shape
    out = np.full((nt, ns), np.nan)
    for b in nb.prange((ns + BLOCK - 1) // BLOCK):
        for p in range(b * BLOCK, min((b + 1) * BLOCK, ns)):
            xp = x[:, p].copy()
            mp = mask[:, p].copy()
            for t in range(half_window, nt - half_window):
                if not mp[t]:
                    continue
                best = np.nan
                found = False
                for k in range(t - half_window, t + half_window + 1):
                    v = sign * xp[k]
                    if v == v and (not found or v > best):
                        best = v
                        found = True
                if found:
                    if use_ref:
                        out[t, p] = sign * (sign * best - ref[t, p])
                    else:
                        out[t, p] = sign * best
    return out


@nb.njit(parallel=True, cache=True)
def _rolling_range_kernel(x, window):
    nt, ns = x.shape
    out = np.full((nt, ns), np.nan)
    left = window // 2
    for b in nb.prange((ns + BLOCK - 1) // BLOCK):
        for p in range(b * BLOCK, min((b + 1) * BLOCK, ns)):
            xp = x[:, p].copy()
            for t in range(left, nt - (window - 1 - left)):
                lo = xp[t - left]
                hi = lo
                for k in range(t - left, t - left + window):
                    v = xp[k]
                    if v < lo:
                        lo = v
                    if v > hi:
                        hi = v
                    if v != v:
                        lo = np.nan
                        break
                if lo == lo:
                    out[t, p] = hi - lo
    return out


# %% quantiles


@nb.njit(parallel=True, cache=True)
def _quantile_mask_kernel(x, quantile):
    nt, ns = x.shape
    out = np.zeros((nt, ns), np.bool_)
    for b in nb.prange((ns + BLOCK - 1) // BLOCK):
        for p in range(b * BLOCK, min((b + 1) * BLOCK, ns)):
            xp = x[:, p].copy()
            vals = np.empty(nt)
            n = 0
            for t in range(nt):
                if xp[t] == xp[t]:
                    vals[n] = xp[t]
                    n += 1
            if n == 0:
                continue
            thr = np.quantile(vals[:n], quantile)
            for t in range(nt):
                if xp[t] >= thr:
                    out[t, p] = True
    return out


# %% public functions


def _flat(a, dtype):
    # (nt, ...) -> contiguous (nt, npoints) view or copy
    a = np.ascontiguousarray(a, dtype=dtype)
    return a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(-1, 1)


def peak_mask(x, distance=1):
    """Boolean mask of the local maxima of series.

    A flat top counts once, at its middle sample; the first and last samples
    are never maxima. Same result as ``scipy.signal.find_peaks(x,
    distance=distance)``, but for many series in parallel (equal peak heights
    may be tied differently).

    Parameters
    ----------
    x : ndarray (nt, ...)
        **Time first**.
    distance : int
        Minimal number of samples between two maxima; the highest ones are
        kept.

    Returns
    -------
    ndarray of bool, same shape as ``x``
    """
    x = np.asarray(x)
    return _peak_mask_kernel(_flat(x, float), int(distance)).reshape(x.shape)


def _window_extreme(x, mask, half_window, ref, sign):
    x = np.asarray(x)
    use_ref = ref is not None
    ref = _flat(ref, float) if use_ref else np.zeros((1, 1))
    out = _window_extreme_kernel(
        _flat(x, float), _flat(mask, bool), ref, use_ref, int(half_window), sign
    )
    return out.reshape(x.shape)


def window_max_at(x, mask, half_window, ref=None):
    """Maximum of ``x`` in a window centred on the samples flagged by ``mask``.

    Parameters
    ----------
    x : ndarray (nt, ...)
        **Time first**; NaNs are skipped.
    mask : ndarray of bool, same shape
        Centres of the windows.
    half_window : int
        The window spans ``2 * half_window + 1`` samples.
    ref : ndarray, same shape, optional
        If given, the maximum minus ``ref`` is returned (the excess of the
        maximum above ``ref``).

    Returns
    -------
    ndarray, same shape as ``x``
        The result at the flagged samples whose window is complete and holds
        some data, NaN elsewhere.
    """
    return _window_extreme(x, mask, half_window, ref, 1.0)


def window_min_at(x, mask, half_window, ref=None):
    """Minimum of ``x`` in a window centred on the flagged samples.

    Same as :func:`window_max_at`, but with ``ref`` given the result is ``ref``
    minus the minimum (the excess of ``ref`` above the minimum, positive when
    the minimum lies below ``ref``).
    """
    return _window_extreme(x, mask, half_window, ref, -1.0)


def rolling_range(x, window):
    """Maximum minus minimum on a centred sliding window.

    Parameters
    ----------
    x : ndarray (nt, ...)
        **Time first**.
    window : int
        Number of samples.

    Returns
    -------
    ndarray, same shape as ``x``
        NaN where the window is incomplete (edges) or contains a NaN.
    """
    x = np.asarray(x)
    return _rolling_range_kernel(_flat(x, float), int(window)).reshape(x.shape)


def quantile_mask(x, quantile):
    """Mask of the samples at or above a quantile of their own series.

    Parameters
    ----------
    x : ndarray (nt, ...)
        **Time first**; NaNs are ignored (and never selected).
    quantile : float
        Between 0 and 1, computed along time for each point.

    Returns
    -------
    ndarray of bool, same shape as ``x``
    """
    x = np.asarray(x)
    return _quantile_mask_kernel(_flat(x, float), float(quantile)).reshape(x.shape)
