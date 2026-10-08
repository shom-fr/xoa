"""
Bilinear and bicubic (Hermite) interpolation kernels

Contains the Hermite/cubic basis functions, the fractional cell indices
computation for regular, rectangular and curvilinear grids, and the
bilinear/bicubic kernels used by :class:`xoa.core.interp.XYInterpolator`.
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

from .geo import closest_point_fast, diff_lon, relative_cell_coords
from .num import EPSILON

NOT_CI = os.environ.get("CI", "false") == "false"

#: Tolerance of the fractional indices, in cells, to include the points on the edges of cells
FRAC_TOL = 1e-9


@numba.njit(fastmath=True)
def hermite_basis(mu):
    """
    Compute the four Hermite basis polynomials at fractional position ``mu`` in [0, 1].

    Returns ``[h00, h10, h01, h11]`` such that the Hermite cubic through
    points p1, p2 with tangents m0, m1 evaluates to::

        h00*p1 + h10*m0 + h01*p2 + h11*m1

    Used by :func:`interpolate_cubic`, :func:`compute_cubic_weights_1d`,
    and directly in :func:`bicubic_frac`.
    """
    mu2 = mu * mu
    mu3 = mu2 * mu
    h00 = 2 * mu3 - 3 * mu2 + 1
    h10 = mu3 - 2 * mu2 + mu
    h01 = -2 * mu3 + 3 * mu2
    h11 = mu3 - mu2
    return np.array([h00, h10, h01, h11])


@numba.njit(fastmath=True)
def compute_tangent(p0, p1, p2, bias, tension):
    """
    Compute the Kochanek–Bartels tangent at p1 given neighbours p0 and p2.

    The tangent controls the slope of the Hermite cubic at p1::

        m = (1-tension)*(1+bias)/2 * (p1-p0)
          + (1-tension)*(1-bias)/2 * (p2-p1)

    ``bias > 0`` weights toward the incoming interval (p0→p1);
    ``bias < 0`` weights toward the outgoing interval (p1→p2).
    ``tension = 1`` collapses the tangent to zero (linear interpolation).

    Used directly by :func:`interpolate_cubic` and, via unit-basis evaluation,
    by :func:`compute_cubic_weights_1d`.
    """
    a = (1 - tension) * (1 + bias) * 0.5 * (p1 - p0)
    b = (1 - tension) * (1 - bias) * 0.5 * (p2 - p1)
    return a + b


@numba.njit(fastmath=True)
def compute_cubic_weights_1d(hh, bias, tension):
    """
    Decompose 1-D Hermite cubic interpolation into per-point weights.

    The Hermite formula is::

        result = h00*p1 + h10*m0 + h01*p2 + h11*m1

    where ``m0 = compute_tangent(p0, p1, p2, ...)`` and
    ``m1 = compute_tangent(p1, p2, p3, ...)``.

    Because ``compute_tangent`` is linear in its point arguments, evaluating
    it on the three unit-basis vectors ``(1,0,0)``, ``(0,1,0)``, ``(0,0,1)``
    gives the scalar coefficient that each point contributes to a tangent::

        m0 = c0*p0 + c1*p1 + c2*p2
        m1 = c0*p1 + c1*p2 + c2*p3

    Substituting back yields weights ``w`` such that
    ``result = w[0]*p0 + w[1]*p1 + w[2]*p2 + w[3]*p3``.

    Parameters
    ----------
    hh : array
        Hermite basis values ``[h00, h10, h01, h11]``.
    bias : float
        Bias parameter.
    tension : float
        Tension parameter.

    Return
    ------
    array
        Weights ``[w0, w1, w2, w3]`` for the 4 points.
    """
    # Per-point coefficients of any compute_tangent call
    c0 = compute_tangent(1.0, 0.0, 0.0, bias, tension)  # first-point coeff
    c1 = compute_tangent(0.0, 1.0, 0.0, bias, tension)  # middle-point coeff
    c2 = compute_tangent(0.0, 0.0, 1.0, bias, tension)  # last-point coeff

    # result = hh[0]*p1 + hh[1]*m0 + hh[2]*p2 + hh[3]*m1
    weights = np.zeros(4)
    weights[0] = hh[1] * c0
    weights[1] = hh[0] + hh[1] * c1 + hh[3] * c0
    weights[2] = hh[2] + hh[1] * c2 + hh[3] * c1
    weights[3] = hh[3] * c2
    return weights


@numba.njit(fastmath=True)
def interpolate_cubic(hh, p0, p1, p2, p3, bias, tension):
    """
    Evaluate the Hermite cubic between p1 and p2 at fractional position encoded by ``hh``.

    Parameters
    ----------
    hh : array
        Hermite basis values from :func:`hermite_basis`.
    p0, p1, p2, p3 : float
        Four consecutive sample values; the curve is evaluated between p1 and p2.
    bias, tension : float
        Kochanek–Bartels parameters passed to :func:`compute_tangent`.

    Notes
    -----
    Used by :func:`interpolate_bicubic` (scalar 2-D reference).
    For vectorised horizontal regridding use :func:`compute_cubic_weights_1d`
    with :func:`bicubic_frac` instead.
    """
    m0 = compute_tangent(p0, p1, p2, bias, tension)
    m1 = compute_tangent(p1, p2, p3, bias, tension)
    return hh[0] * p1 + hh[1] * m0 + hh[2] * p2 + hh[3] * m1


@numba.njit(fastmath=True)
def interpolate_bicubic(xhh, yhh, pp, bias, tension):
    """
    Evaluate the separable Hermite bicubic on a 4×4 stencil at a single point.

    Applies :func:`interpolate_cubic` first along the x-axis (for each of the
    4 rows), then along the y-axis on the 4 resulting values.  This is the
    scalar, readable reference implementation; the vectorised equivalent for
    horizontal regridding is :func:`bicubic_frac`, which reaches the same
    result via the outer product of two :func:`compute_cubic_weights_1d`
    weight vectors.

    Parameters
    ----------
    xhh : array
        Hermite basis values along x from :func:`hermite_basis`.
    yhh : array
        Hermite basis values along y from :func:`hermite_basis`.
    pp : ndarray (4, 4)
        Stencil values in row-major order (row = y, col = x).
    bias, tension : float
        Kochanek–Bartels parameters.

    Return
    ------
    float
        Interpolated value.
    """
    # Interpolate along x for each row
    col_results = np.empty(4)
    for row in range(4):
        col = pp[row, :]
        col_results[row] = interpolate_cubic(xhh, col[0], col[1], col[2], col[3], bias, tension)

    # Interpolate along y
    return interpolate_cubic(
        yhh, col_results[0], col_results[1], col_results[2], col_results[3], bias, tension
    )


# %% Point interpolation kernels


@numba.njit(cache=NOT_CI, parallel=True)
def compute_frac_indices(
    src_centers_lon,
    src_centers_lat,
    dst_centers_lon,
    dst_centers_lat,
    grid_type,
    stencil_margin=0,
):
    """
    For each destination point find the containing source cell and return fractional coordinates.

    Parameters
    ----------
    src_centers_lon, src_centers_lat : (ny_src, nx_src)
    dst_centers_lon, dst_centers_lat : (ny_dst, nx_dst)
    grid_type : int  0=regular, 1=rectangular, 2=curvilinear
    stencil_margin : int
        0 for bilinear (2x2 stencil), 1 for bicubic (4x4 stencil)

    Return
    ------
    j_base : (n_dst,) int64  row of SW corner; -1 = invalid (outside domain)
    i_base : (n_dst,) int64  col of SW corner
    frac_a : (n_dst,) float64  fractional x within cell [0, 1]
    frac_b : (n_dst,) float64  fractional y within cell [0, 1]
    """
    src_ny, src_nx = src_centers_lat.shape
    dst_ny, dst_nx = dst_centers_lat.shape
    n_dst = dst_ny * dst_nx
    m = stencil_margin

    j_base = np.full(n_dst, -1, dtype=np.int64)
    i_base = np.full(n_dst, -1, dtype=np.int64)
    frac_a = np.zeros(n_dst, dtype=np.float64)
    frac_b = np.zeros(n_dst, dtype=np.float64)

    src_dlon = src_dlat = src_lon_start = src_lat_start = 0.0
    if grid_type == 0:
        src_dlon = diff_lon(src_centers_lon[0, 0], src_centers_lon[0, 1])
        src_dlat = src_centers_lat[1, 0] - src_centers_lat[0, 0]
        src_lon_start = src_centers_lon[0, 0]
        src_lat_start = src_centers_lat[0, 0]

    for dst_idx in numba.prange(n_dst):
        dst_j = dst_idx // dst_nx
        dst_i = dst_idx % dst_nx
        dst_lon = dst_centers_lon[dst_j, dst_i]
        dst_lat = dst_centers_lat[dst_j, dst_i]

        if not (-180.0 <= dst_lon <= 180.0 and -90.0 <= dst_lat <= 90.0):
            continue

        found = False
        si = sj = -1
        a = b = 0.0

        if grid_type == 0:
            fi = diff_lon(src_lon_start, dst_lon) / src_dlon
            fj = (dst_lat - src_lat_start) / src_dlat
            # Cells are closed: points on their edges are inside, like the ones of the first
            # and last lines of the grid. Truncating toward zero would let in the points
            # that are up to one cell before the first line.
            si = int(np.floor(fi + FRAC_TOL))
            sj = int(np.floor(fj + FRAC_TOL))
            max_si = src_nx - 2 - m
            max_sj = src_ny - 2 - m
            if si == max_si + 1 and fi <= max_si + 1 + FRAC_TOL:
                si = max_si
            if sj == max_sj + 1 and fj <= max_sj + 1 + FRAC_TOL:
                sj = max_sj
            if m <= si <= max_si and m <= sj <= max_sj:
                a = fi - si
                b = fj - sj
                found = True

        elif grid_type == 1:
            si = sj = -1
            # Only the cells that have their stencil are searched, so that a point on a line
            # belongs to the first valid cell, like on regular grids.
            # The position in a cell is the ratio to its step, whatever the orientation of
            # the axis, and it is between 0 and 1 inside the cell.
            for ii in range(m, src_nx - 1 - m):
                step = diff_lon(src_centers_lon[0, ii], src_centers_lon[0, ii + 1])
                if step != 0.0:
                    ratio = diff_lon(src_centers_lon[0, ii], dst_lon) / step
                    if -FRAC_TOL <= ratio <= 1.0 + FRAC_TOL:
                        si = ii
                        break
            for jj in range(m, src_ny - 1 - m):
                step = src_centers_lat[jj + 1, 0] - src_centers_lat[jj, 0]
                if step != 0.0:
                    ratio = (dst_lat - src_centers_lat[jj, 0]) / step
                    if -FRAC_TOL <= ratio <= 1.0 + FRAC_TOL:
                        sj = jj
                        break
            if m <= si < src_nx - 1 - m and m <= sj < src_ny - 1 - m:
                a = diff_lon(src_centers_lon[0, si], dst_lon) / diff_lon(
                    src_centers_lon[0, si], src_centers_lon[0, si + 1]
                )
                b = (dst_lat - src_centers_lat[sj, 0]) / (
                    src_centers_lat[sj + 1, 0] - src_centers_lat[sj, 0]
                )
                found = True

        else:  # curvilinear
            ic, jc = closest_point_fast(src_centers_lon, src_centers_lat, dst_lon, dst_lat)
            for src_j in range(max(jc - 1, m), min(jc + 2, src_ny - 1 - m)):
                for src_i in range(max(ic - 1, m), min(ic + 2, src_nx - 1 - m)):
                    # Longitudes relative to the target, which do not have the jump of the
                    # dateline and are better conditioned
                    a2, b2 = relative_cell_coords(
                        diff_lon(dst_lon, src_centers_lon[src_j, src_i]),
                        diff_lon(dst_lon, src_centers_lon[src_j, src_i + 1]),
                        diff_lon(dst_lon, src_centers_lon[src_j + 1, src_i + 1]),
                        diff_lon(dst_lon, src_centers_lon[src_j + 1, src_i]),
                        src_centers_lat[src_j, src_i],
                        src_centers_lat[src_j, src_i + 1],
                        src_centers_lat[src_j + 1, src_i + 1],
                        src_centers_lat[src_j + 1, src_i],
                        0.0,
                        dst_lat,
                    )
                    if -FRAC_TOL <= a2 <= 1.0 + FRAC_TOL and -FRAC_TOL <= b2 <= 1.0 + FRAC_TOL:
                        si = src_i
                        sj = src_j
                        a = a2
                        b = b2
                        found = True
                        break
                if found:
                    break

        if found:
            a = max(0.0, min(1.0, a))
            b = max(0.0, min(1.0, b))
            j_base[dst_idx] = sj
            i_base[dst_idx] = si
            frac_a[dst_idx] = a
            frac_b[dst_idx] = b

    return j_base, i_base, frac_a, frac_b


@numba.njit(cache=NOT_CI, parallel=True)
def bilinear_frac(j_base, i_base, frac_a, frac_b, X, nx_src, out, skipna, na_thres):
    """
    Bilinear interpolation over K columns using fractional cell indices.

    Parameters
    ----------
    j_base, i_base : (n_dst,) int64  SW corner; j_base=-1 → NaN output
    frac_a, frac_b : (n_dst,) float64
    X              : (n_src, K) float64
    nx_src         : int
    out            : (n_dst, K) float64  filled in-place
    skipna         : bool
    na_thres       : float
    """
    nan = np.nan
    na_threshold = max(EPSILON, 1.0 - na_thres)
    n_dst = j_base.shape[0]
    K = X.shape[1]

    for dst_idx in numba.prange(n_dst):
        j0 = j_base[dst_idx]
        if j0 < 0:
            for k in range(K):
                out[dst_idx, k] = nan
            continue

        i0 = i_base[dst_idx]
        a = frac_a[dst_idx]
        b = frac_b[dst_idx]

        w00 = (1.0 - a) * (1.0 - b)  # SW
        w10 = a * (1.0 - b)  # SE
        w01 = (1.0 - a) * b  # NW
        w11 = a * b  # NE

        c00 = j0 * nx_src + i0
        c10 = j0 * nx_src + i0 + 1
        c01 = (j0 + 1) * nx_src + i0
        c11 = (j0 + 1) * nx_src + i0 + 1

        for k in range(K):
            v00 = X[c00, k]
            v10 = X[c10, k]
            v01 = X[c01, k]
            v11 = X[c11, k]

            if skipna:
                s = wsum = 0.0
                if not np.isnan(v00):
                    s += w00 * v00
                    wsum += w00
                if not np.isnan(v10):
                    s += w10 * v10
                    wsum += w10
                if not np.isnan(v01):
                    s += w01 * v01
                    wsum += w01
                if not np.isnan(v11):
                    s += w11 * v11
                    wsum += w11
                out[dst_idx, k] = s / wsum if wsum >= na_threshold else nan
            else:
                out[dst_idx, k] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11


@numba.njit(cache=NOT_CI, parallel=True)
def bicubic_frac(
    j_base, i_base, frac_a, frac_b, X, ny_src, nx_src, out, skipna, na_thres, bias=0.0, tension=0.0
):
    """
    Bicubic (Hermite) interpolation over K columns using fractional cell indices.

    When any of the 16 stencil points has NaN the kernel falls back to bilinear
    interpolation on the 4 center points.  When skipna=True, the bilinear
    fallback further renormalises over the valid subset.

    Parameters
    ----------
    j_base, i_base : (n_dst,) int64  cell anchoring the 4x4 stencil;
                      j_base=-1 → NaN output
    frac_a, frac_b : (n_dst,) float64
    X              : (n_src, K) float64
    ny_src, nx_src : int
    out            : (n_dst, K) float64  filled in-place
    skipna         : bool
    na_thres       : float
    bias, tension  : float  Hermite parameters (default 0)
    """
    nan = np.nan
    na_threshold = max(EPSILON, 1.0 - na_thres)
    n_dst = j_base.shape[0]
    K = X.shape[1]

    for dst_idx in numba.prange(n_dst):
        j0 = j_base[dst_idx]
        if j0 < 0:
            for k in range(K):
                out[dst_idx, k] = nan
            continue

        i0 = i_base[dst_idx]
        a = frac_a[dst_idx]
        b = frac_b[dst_idx]

        xhh = hermite_basis(a)
        yhh = hermite_basis(b)
        x_w = compute_cubic_weights_1d(xhh, bias, tension)
        y_w = compute_cubic_weights_1d(yhh, bias, tension)

        stencil_flat = np.empty(16, dtype=np.int64)
        w_bic = np.empty(16, dtype=np.float64)

        for ri in range(4):
            for ci in range(4):
                sj = j0 + ri - 1
                si = i0 + ci - 1
                p = ri * 4 + ci
                stencil_flat[p] = sj * nx_src + si
                w_bic[p] = y_w[ri] * x_w[ci]

        # Bilinear fallback weights from same (a, b) — center 4 of the 4x4 stencil
        # positions: (ri=1,ci=1)=5, (ri=1,ci=2)=6, (ri=2,ci=1)=9, (ri=2,ci=2)=10
        w_bil = np.empty(4, dtype=np.float64)
        c_bil = np.empty(4, dtype=np.int64)
        w_bil[0] = (1.0 - a) * (1.0 - b)
        c_bil[0] = stencil_flat[5]  # SW
        w_bil[1] = a * (1.0 - b)
        c_bil[1] = stencil_flat[6]  # SE
        w_bil[2] = (1.0 - a) * b
        c_bil[2] = stencil_flat[9]  # NW
        w_bil[3] = a * b
        c_bil[3] = stencil_flat[10]  # NE

        for k in range(K):
            has_nan = False
            for p in range(16):
                if np.isnan(X[stencil_flat[p], k]):
                    has_nan = True
                    break

            if not has_nan:
                s = 0.0
                for p in range(16):
                    s += w_bic[p] * X[stencil_flat[p], k]
                out[dst_idx, k] = s
            else:
                s = wsum = 0.0
                for p in range(4):
                    v = X[c_bil[p], k]
                    if not np.isnan(v):
                        s += w_bil[p] * v
                        wsum += w_bil[p]
                out[dst_idx, k] = s / wsum if wsum >= na_threshold else nan
