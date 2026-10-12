"""
Low level statistics

The numerical inputs and outputs of all these routines are of scalar
or numpy.ndarray type.

The accumulation of statistics relies on additive "moments", which are stored
in arrays of shape ``(NMOM, ...)`` whose first axis is indexed by the
``N``, ``SW``, ... constants. They are centered (weighted means, and sums of
squared and cross deviations), and they are merged with the pairwise update of
Chan et al., which is free of the cancellation errors of raw sums of squares. See :func:`get_moments`, :func:`merge_moments`
and :func:`moments2stats`.
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
import os

import numba
import numpy as np

NOT_CI = os.environ.get("CI", "false") == "false"


@numba.njit(cache=NOT_CI, parallel=True)
def _taylor_stats_(model, ref):
    n, m = model.shape
    std = np.full(n, np.nan)
    std_ref = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    crmsd = np.full(n, np.nan)
    for i in numba.prange(n):
        # Only the points that are valid in both arrays are used
        cnt = 0
        sm = 0.0
        sr = 0.0
        for j in range(m):
            if not (np.isnan(model[i, j]) or np.isnan(ref[i, j])):
                cnt += 1
                sm += model[i, j]
                sr += ref[i, j]
        if cnt == 0:
            continue
        mm = sm / cnt
        mr = sr / cnt
        vm = 0.0
        vr = 0.0
        cov = 0.0
        vd = 0.0
        for j in range(m):
            if not (np.isnan(model[i, j]) or np.isnan(ref[i, j])):
                dm = model[i, j] - mm
                dr = ref[i, j] - mr
                vm += dm * dm
                vr += dr * dr
                cov += dm * dr
                vd += (dm - dr) * (dm - dr)
        std[i] = np.sqrt(vm / cnt)
        std_ref[i] = np.sqrt(vr / cnt)
        crmsd[i] = np.sqrt(vd / cnt)
        if vm > 0 and vr > 0:
            corr[i] = cov / np.sqrt(vm * vr)
    return std, std_ref, corr, crmsd


def taylor_stats(model, ref):
    """Statistics that are displayed in a Taylor diagram

    Only the samples that are valid in both arrays are used.
    The standard deviations are the biased ones (``ddof=0``), so that the
    centered root mean square difference ``crmsd`` verifies
    ``crmsd**2 = std**2 + std_ref**2 - 2 * std * std_ref * corr``.

    Parameters
    ----------
    model: array_like(n, m)
        ``n`` series of ``m`` samples. A 1D array is a single series.
    ref: array_like(m) or array_like(n, m)
        Reference series, that are shared by the ``n`` series of the model
        when it is 1D.

    Return
    ------
    numpy.ndarray(n)
        Standard deviation of the model
    numpy.ndarray(n)
        Standard deviation of the reference
    numpy.ndarray(n)
        Correlation, which is nan when a series is constant
    numpy.ndarray(n)
        Centered root mean square difference
    """
    model = np.atleast_2d(np.asarray(model, dtype="d"))
    ref = np.asarray(ref, dtype="d")
    ref = np.ascontiguousarray(np.broadcast_to(ref, model.shape))
    return _taylor_stats_(np.ascontiguousarray(model), ref)


#: Indices of the moments along the first axis of moment arrays
N, SW, MX, MY, M2X, M2Y, CXY, M2D, XMIN, XMAX, YMIN, YMAX = range(12)

#: Number of moments
NMOM = 12

#: Names of the moments, in the order of their indices
MOMENT_NAMES = ("n", "sw", "mx", "my", "m2x", "m2y", "cxy", "m2d", "xmin", "xmax", "ymin", "ymax")


@numba.njit(cache=NOT_CI, parallel=True)
def _get_moments_(x, y, w, dual, weighted):
    n, m = x.shape
    out = np.zeros((NMOM, n))
    for i in numba.prange(n):
        xmin = ymin = np.inf
        xmax = ymax = -np.inf
        cnt = 0
        sw = sx = sy = 0.0
        for j in range(m):
            xv = x[i, j]
            if not np.isfinite(xv):
                continue
            yv = 0.0
            if dual:
                yv = y[i, j]
                if not np.isfinite(yv):
                    continue
            wv = 1.0
            if weighted:
                wv = w[i, j]
                # Also discards nans
                if not (wv > 0 and np.isfinite(wv)):
                    continue
            cnt += 1
            sw += wv
            sx += wv * xv
            xmin = min(xmin, xv)
            xmax = max(xmax, xv)
            if dual:
                sy += wv * yv
                ymin = min(ymin, yv)
                ymax = max(ymax, yv)
        mx = sx / sw if cnt else 0.0
        my = sy / sw if cnt else 0.0
        # Second pass on deviations
        m2x = m2y = cxy = m2d = 0.0
        for j in range(m):
            xv = x[i, j]
            if not np.isfinite(xv):
                continue
            yv = 0.0
            if dual:
                yv = y[i, j]
                if not np.isfinite(yv):
                    continue
            wv = 1.0
            if weighted:
                wv = w[i, j]
                if not (wv > 0 and np.isfinite(wv)):
                    continue
            dx = xv - mx
            m2x += wv * dx * dx
            if dual:
                dy = yv - my
                m2y += wv * dy * dy
                cxy += wv * dx * dy
                dd = dy - dx
                m2d += wv * dd * dd
        out[N, i] = cnt
        out[SW, i] = sw
        out[MX, i] = mx
        out[MY, i] = my
        out[M2X, i] = m2x
        out[M2Y, i] = m2y
        out[CXY, i] = cxy
        out[M2D, i] = m2d
        out[XMIN, i] = xmin
        out[XMAX, i] = xmax
        out[YMIN, i] = ymin
        out[YMAX, i] = ymax
    return out


def get_moments(x, y=None, w=None):
    """Get the moments of series, along the last axis

    A sample is used only if it is finite in `x` and `y`, and if its weight
    is finite and strictly positive.

    Parameters
    ----------
    x: array_like(n, m)
        ``n`` series of ``m`` samples.
    y: None, array_like(n, m)
        Second variable, for comparative statistics.
    w: None, array_like(n, m)
        Weights, which default to 1.

    Return
    ------
    numpy.ndarray(NMOM, n)
        Count, sum of weights, weighted means of ``x`` and ``y``, weighted sums of
        squared deviations of ``x``, ``y`` and ``y - x`` and of the product ``x * y``, then minima and
        maxima of ``x`` and ``y``.
        The minima and maxima of empty series are respectively ``inf`` and ``-inf``.
    """
    x = np.ascontiguousarray(x, dtype="d")
    dummy = np.empty((0, 0))
    dual = y is not None
    weighted = w is not None
    y = np.ascontiguousarray(y, dtype="d") if dual else dummy
    w = np.ascontiguousarray(w, dtype="d") if weighted else dummy
    return _get_moments_(x, y, w, dual, weighted)


def merge_moments(mom0, mom1):
    """Merge two arrays of moments of the same shape, which are obtained
    from independent samples, with the pairwise update of Chan et al."""
    out = np.empty_like(mom0)
    out[N] = mom0[N] + mom1[N]
    sw0, sw1 = mom0[SW], mom1[SW]
    sw = sw0 + sw1
    out[SW] = sw
    ratio = np.divide(sw1, sw, out=np.zeros(np.shape(sw)), where=sw > 0)
    dx = mom1[MX] - mom0[MX]
    dy = mom1[MY] - mom0[MY]
    cross = sw0 * ratio
    out[MX] = mom0[MX] + dx * ratio
    out[MY] = mom0[MY] + dy * ratio
    out[M2X] = mom0[M2X] + mom1[M2X] + dx * dx * cross
    out[M2Y] = mom0[M2Y] + mom1[M2Y] + dy * dy * cross
    out[CXY] = mom0[CXY] + mom1[CXY] + dx * dy * cross
    dd = dy - dx
    out[M2D] = mom0[M2D] + mom1[M2D] + dd * dd * cross
    for i, func in ((XMIN, np.minimum), (XMAX, np.maximum), (YMIN, np.minimum), (YMAX, np.maximum)):
        out[i] = func(mom0[i], mom1[i])
    return out


def _div_(a, b):
    out = np.full(np.shape(a), np.nan)
    return np.divide(a, b, out=out, where=np.asarray(b) > 0)


def moments2stats(mom, dual=False, ddof=0):
    """Compute weighted statistics from moments

    Parameters
    ----------
    mom: numpy.ndarray(NMOM, ...)
        Moments as returned by :func:`get_moments` and :func:`merge_moments`.
    dual: bool
        Compute comparative statistics.
    ddof: int
        Delta degrees of freedom of variances and covariances, which are
        divided by ``sw - ddof``.

    Return
    ------
    dict
        Statistics ``count``, ``mean``, ``var``, ``std``, ``min`` and ``max``,
        suffixed with ``0`` and ``1`` when `dual` is True, and in this case
        ``bias`` (second minus first), ``rms``, ``crms``, ``cov`` and ``corr``.
    """
    mom = np.asarray(mom, dtype="d")
    sw = mom[SW]
    valid = mom[N] > 0
    dsw = sw - ddof
    stats = {"count": mom[N]}
    parts = [("0", MX, M2X, XMIN, XMAX)]
    if dual:
        parts.append(("1", MY, M2Y, YMIN, YMAX))
    for suf, imean, im2, imin, imax in parts:
        if not dual:
            suf = ""
        var = _div_(mom[im2], dsw)
        stats["mean" + suf] = np.where(valid, mom[imean], np.nan)
        stats["var" + suf] = var
        stats["std" + suf] = np.sqrt(var)
        stats["min" + suf] = np.where(valid, mom[imin], np.nan)
        stats["max" + suf] = np.where(valid, mom[imax], np.nan)
    if dual:
        bias = stats["mean1"] - stats["mean0"]
        stats["bias"] = bias
        stats["crms"] = np.sqrt(_div_(mom[M2D], dsw))
        stats["rms"] = np.sqrt(_div_(mom[M2D], sw) + bias**2)
        stats["cov"] = _div_(mom[CXY], dsw)
        stats["corr"] = _div_(mom[CXY], np.sqrt(mom[M2X] * mom[M2Y]))
    return stats
