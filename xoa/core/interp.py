"""
Low level interpolation routines accelerated with numba

The numerical inputs and outputs of all these routines are of scalar
or numpy.ndarray type.
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

import numpy as np
import numba

from ..exceptions import xoa_warn
from .geo import closest_point_fast, relative_cell_coords
from .grid import check_grid_type
from .num import EPSILON
from .spline import (
    bicubic_frac,
    bilinear_frac,
    compute_cubic_weights_1d,
    compute_frac_indices,
    hermite_basis,
)
from .time import compute_time_frac_indices

NOT_CI = os.environ.get("CI", "false") == "false"


# %% 2D routines


def closest2d(xxi, yyi, xo, yo):
    """Find indices of closest point on 2D lon/lat grid

    .. deprecated::
        Use :func:`xoa.core.geo.closest_point_fast` instead.

    Parameters
    ----------
    xxi: array_like(nyi, nxi)
        Grid longitudes in degrees
    yyi: array_like(nyi, nxi)
        Grid latitudes in degrees
    xo:
        Point longtitude
    yo:
        Point latitude

    Return
    ------
    int: i
        index along second dim
    int: j
        Index along first dim
    """
    xoa_warn("closest2d is deprecated. Use xoa.core.geo.closest_point_fast instead.", "deprecation")
    return closest_point_fast(xxi, yyi, xo, yo)


def cell2relloc(x1, x2, x3, x4, y1, y2, y3, y4, x, y):
    """Compute coordinates of point relative to a curvilinear cell

    .. deprecated::
        Use :func:`xoa.core.geo.relative_cell_coords` instead, which takes
        the vertices in counter-clockwise order (1, 2, 3, 4 = 1, 4, 3, 2 here).

    Cell shape::

      2 - 3
      |   |
      1 - 4

    Example
    -------
    >>> cell2relloc(0., -2., 0., 2., 0., 1., 1., 0., 0., 0.5)
    (0.5, 0.5)
    """
    xoa_warn(
        "cell2relloc is deprecated. Use xoa.core.geo.relative_cell_coords instead.", "deprecation"
    )
    return relative_cell_coords(x1, x4, x3, x2, y1, y4, y3, y2, x, y)


@numba.njit(cache=NOT_CI)
def grid2relloc(xxi, yyi, xo, yo):
    """Compute coordinates of point relative to a curvilinear grid

    Parameters
    ----------
    xxi: array_like(nyi, nxi)
        Grid longitudes in degrees
    yyi: array_like(nyi, nxi)
        Grid latitudes in degrees
    xo: float
        Point longitude
    yo: float
        Point latitude

    Return
    ------
    float:
        The integer part gives the grid cell index along the second dim,
        and the fractional part gives the coordinate relative this cell.
        A value of -1 means outside the grid.
    float:
        The integer part gives the grid cell index along the first dim,
        and the fractional part gives the coordinate relative this cell.
        A value of -1 means outside the grid.
    """

    p = -1.0
    q = -1.0
    small = np.finfo(np.float64).eps
    nyi, nxi = xxi.shape

    # Find the closest corner
    ic, jc = closest_point_fast(xxi, yyi, xo, yo)

    # Curvilinear to rectangular with a loop on four candidate cells
    for j in range(max(jc - 1, 0), min(jc + 1, nyi - 1)):
        for i in range(max(ic - 1, 0), min(ic + 1, nxi - 1)):
            # Get relative position
            a, b = relative_cell_coords(
                xxi[j, i],
                xxi[j, i + 1],
                xxi[j + 1, i + 1],
                xxi[j + 1, i],
                yyi[j, i],
                yyi[j, i + 1],
                yyi[j + 1, i + 1],
                yyi[j + 1, i],
                xo,
                yo,
            )

            # Store absolute indices
            if (
                (a >= 0.0 - small)
                and (a <= 1.0 + small)
                and (b >= 0.0 - small)
                and (b <= 1.0 + small)
            ):
                p = np.float64(i) + a
                q = np.float64(j) + b
                break
        else:
            continue
        break
    return p, q


@numba.njit(fastmath=True, cache=NOT_CI)
def _grid2rellocs_(xxi, yyi, xo, yo):
    """Compute coordinates of points relative to a curvilinear grid

    Parameters
    ----------
    xxi: array_like(nyi, nxi)
        Grid longitudes in degrees
    yyi: array_like(nyi, nxi)
        Grid latitudes in degrees
    xo: array_like(no)
        Point longitude
    yo: array_like(no)
        Point latitude

    Return
    ------
    array_like(no):
        The integer part gives the grid cell index along the second dim,
        and the fractional part gives the coordinate relative this cell.
        A value of -1 means outside the grid.
    array_like(no):
        The integer part gives the grid cell index along the first dim,
        and the fractional part gives the coordinate relative this cell.
        A value of -1 means outside the grid.
    """
    no = xo.size
    pp = np.zeros(no)
    qq = np.zeros(no)
    for i in range(xo.size):
        pp[i], qq[i] = grid2relloc(xxi, yyi, xo[i], yo[i])
    return pp, qq


def grid2rellocs(xxi, yyi, xo, yo):
    """Compute coordinates of points relative to a curvilinear grid

    .. deprecated::
        Use :func:`xoa.core.spline.compute_frac_indices` or
        :class:`xoa.core.interp.XYInterpolator` instead.

    See :func:`grid2relloc` for the parameters and the return values.
    """
    xoa_warn(
        "grid2rellocs is deprecated. Use xoa.core.spline.compute_frac_indices "
        "or xoa.core.interp.XYInterpolator instead.",
        "deprecation",
    )
    return _grid2rellocs_(xxi, yyi, xo, yo)


@numba.njit(parallel=True, cache=NOT_CI)
def grid2locs(xxi, yyi, zzi, ti, vi, xo, yo, zo, to):
    """Linear interpolation of gridded data to random positions

    Parameters
    ----------
    xxi: array_like(nyi, nxi)
        Input grid longitudes in degrees, with `nyi==1` for 1D coordinates.
    yyi: array_like(nyi, nxi)
        Input grid latitudes in degrees, with `nxi==1` for 1D coordinates.
    zzi: array_like(nexz, nti, nzi, nyi, nxi)
        Input grid depths, positive up. Non effective dimensions
        must be set to 1. `nexz` may be equal or a multiple of `nex`.
    ti:  array_like(nti)
        Input times
    vi: array_like(nexz, ntiz, nzi, nyiz, nxiz)
        Input values.
    xo: array_like(no)
        Points longitude
    yo: array_like(no)
        Points latitude
    zo: array_like(no)
        Points depth, positive up.
    to: array_like(no)
        Points time.

    Return
    ------
    array_like(nex, no)
        Points value.
    """
    # Dimensions
    nyix, nxi = xxi.shape
    nyi, nxiy = yyi.shape
    nexz, ntiz, nzi, nyiz, nxiz = zzi.shape
    nexv, nti, nzi, nyi, nxi = vi.shape
    no = xo.shape[0]

    # Initalisations
    nex = max(nexv, nexz)
    vo = np.full((nex, no), np.nan, dtype=vi.dtype)
    bmask = np.isnan(vi)
    masko = np.isnan(xo) | np.isnan(yo) | np.isnan(zo) | np.isnan(to)
    ximin = xxi.min()
    ximax = xxi.max()
    yimin = yyi.min()
    yimax = yyi.max()
    zimin = zzi.min()
    zimax = zzi.max()
    timin = ti.min()
    timax = ti.max()
    curved = nyix != 1

    # Verifications
    assert not curved or (nxi == nxiy and nyi == nyix), "linear4dto1: Invalid curved dimensions"
    assert nxiz == 1 or nxiz == nxi, "grid2locs: Invalid nxiz dimension"
    assert nyiz == 1 or nyiz == nyi, "grid2locs: Invalid nyiz dimension"
    assert ntiz == 1 or ntiz == nti, "grid2locs: Invalid ntiz dimension"

    # Loop on ouput points
    for io in numba.prange(no):
        # for io in range(no):

        if masko[io]:
            continue

        if (
            (nxi != 1 and (xo[io] < ximin or xo[io] > ximax))
            or (nyi != 1 and (yo[io] < yimin or yo[io] > yimax))
            or (nzi != 1 and (zo[io] < zimin or zo[io] > zimax))
            or (nti != 1 and (to[io] < timin or to[io] > timax))
        ):
            continue

        # Weights
        if curved:
            p, q = grid2relloc(xxi, yyi, xo[io], yo[io])
            if p < 0 or p > nxi - 1 or q < 0 or q > nyi - 1:
                continue  # outside the grid
            i = int(p)
            j = int(q)
            a = p - i
            b = q - j
            npi = 2
            npj = 2

        else:
            # - X
            if nxi == 1:
                i = 0
                a = 0.0
                npi = 1
            elif xxi[0, nxi - 1] == xo[io]:
                i = nxi - 1
                a = 0.0
                npi = 1
            else:
                # i = minloc(xxi[1, :], dim=1, mask=xxi(1,:)>xo[io])-1
                i = np.searchsorted(xxi[0, :], xo[io], "right") - 1
                a = xo[io] - xxi[0, i]
                if abs(a) > 180.0:
                    a -= 180.0  # FIXME: grid2locs: abs(a)>180.
                a = a / (xxi[0, i + 1] - xxi[0, i])
                npi = 2

            # - Y
            if nyi == 1:
                j = 0
                b = 0.0
                npj = 1
            elif yyi[nyi - 1, 0] == yo[io]:
                j = nyi - 1
                b = 0.0
                npj = 1
            else:
                # j = minloc(yyi[:,1], dim=1, mask=yyi[:,1]>yo[io]) - 1
                j = np.searchsorted(yyi[:, 0], yo[io], "right") - 1
                b = (yo[io] - yyi[j, 0]) / (yyi[j + 1, 0] - yyi[j, 0])
                npj = 2

        # - T
        if nti == 1:
            l = 0
            d = 0.0
            npl = 1
        elif ti[nti - 1] == to[io]:
            l = nti - 1
            d = 0.0
            npl = 1
        else:
            l = np.searchsorted(ti, to[io], "right") - 1
            # l = minloc(ti, dim=1, mask=ti>to[io])-1
            if ti[l + 1] == ti[l]:
                d = 0.0
                npl = 1
            else:
                d = (to[io] - ti[l]) / (ti[l + 1] - ti[l])
                npl = 2

        # - Z
        c = np.zeros(nexz)
        k = np.zeros(nexz, "l")
        npk = np.zeros(nexz, "l")
        if nzi == 1:
            k[:] = 0
            c[:] = 0.0
            npk[:] = 1
        else:
            # Local zi

            if nxiz == 1:
                npiz = 1
                az = 0.0
                iz = 0
            else:
                npiz = npi
                az = a
                iz = i

            if nyiz == 1:
                npjz = 1
                bz = 0.0
                jz = 0
            else:
                npjz = npj
                bz = b
                jz = j

            if ntiz == 1:
                nplz = 1
                dz = 0.0
                lz = 0
            else:
                nplz = npl
                dz = d
                lz = l

            zi = np.zeros((nexz, nzi))
            for ie in range(nexz):
                for ll in range(nplz):
                    for kk in range(nzi):
                        for jj in range(0, npjz):
                            for ii in range(0, npiz):
                                zi[ie, kk] += (
                                    zzi[ie, lz + ll, kk, jz + jj, iz + ii]
                                    * ((1 - az) * (1 - ii) + az * ii)
                                    * ((1 - bz) * (1 - jj) + bz * jj)
                                    * ((1 - dz) * (1 - ll) + dz * ll)
                                )

            # Normal stuff (c(nexz),zi[nexz,nzi),k(nexz)
            for ie in range(nexz):  # extra dim
                if zi[ie, nzi - 1] == zo[io]:
                    k[ie] = nzi - 1
                    c[ie] = 0.0
                    npk[ie] = 1
                else:
                    k[ie] = np.searchsorted(zi[ie], zo[io], "right") - 1
                    if zi[ie, k[ie] + 1] == zi[ie, k[ie]]:
                        c[ie] = 0.0
                        npk[ie] = 1
                    else:
                        c[ie] = (zo[io] - zi[ie, k[ie]]) / (zi[ie, k[ie] + 1] - zi[ie, k[ie]])
                        npk[ie] = 2

        # Interpolate
        for ie in range(nex):
            if not bmask[
                ie % nexv,
                l : l + npl,
                k[ie % nexz] : k[ie % nexz] + npk[ie % nexz],
                j : j + npj,
                i : i + npi,
            ].any():
                vo[ie % nex, io] = 0.0
                for ll in range(npl):
                    for kk in range(npk[ie % nexz]):
                        for jj in range(npj):
                            for ii in range(npi):
                                vo[ie % nex, io] = vo[ie % nex, io] + vi[
                                    ie % nex,
                                    l + ll,
                                    k[ie % nexz] + kk,
                                    j + jj,
                                    i + ii,
                                ] * ((1 - a) * (1 - ii) + a * ii) * (
                                    (1 - b) * (1 - jj) + b * jj
                                ) * (
                                    (1 - c[ie % nexz]) * (1 - kk) + c[ie % nexz] * kk
                                ) * (
                                    (1 - d) * (1 - ll) + d * ll
                                )

    return vo


@numba.guvectorize(
    [
        (
            numba.float64[:],
            numba.float64[:],
            numba.float64[:],
            numba.boolean,
            numba.float64[:],
        )
    ],
    "(nz),(nz),(),()->()",
)
def isoslice(var, values, isoval, reverse, isovar):
    """Extract data from var where values==isoval

    Parameters
    ----------
    var: array_like
        array from which the data are extracted
    values: array_like
        array on which a research of isoval is made
    isoval: float
        value of interest on which we perform research in values array
    reverse: bool
        search from the last index instead of the first one

    Return
    ------
    isovar : array_like
           Sliced array based on var where values==isoval

    Example
    -------

    Let's define depth and temperature variables both in 3 dimensions (i,j,k)
    where i and j are horizontal dimension and k the vertical one::

        dep_at_t20 = isoslice(dep, temp, 20)   # depth at temperature=20°C
        temp_at_z15 = isoslice(temp, dep, -15) # temperature at depth=-15m


    """
    nz = var.shape[-1]
    isovar[0] = np.nan
    if reverse:
        istart, istop, istep = nz - 1, 1, -1
    else:
        istart, istop, istep = 0, nz - 2, 1

    # From the top
    for i in numba.prange(istart, istop + istep, istep):
        if (values[i] >= isoval[0] and values[i + istep] <= isoval[0]) or (
            values[i] <= isoval[0] and values[i + istep] >= isoval[0]
        ):
            if values[i + istep] == values[i]:
                isovar[0] = values[i]
            else:
                isovar[0] = var[i + istep] + (isoval[0] - values[i + istep]) * (
                    var[i] - var[i + istep]
                ) / (values[i] - values[i + istep])


# %% XYT point interpolation


@numba.njit(cache=NOT_CI, parallel=True)
def interp_transect(
    data,
    j_base,
    i_base,
    frac_a,
    frac_b,
    it_base,
    frac_t,
    nx_src,
    ny_src,
    skipna,
    na_thres,
    method,
    time_method,
    bias,
    tension,
):
    """
    Trilinear interpolation at scattered (x, y, t) locations.

    For each output point, the two bracketing source time steps are accessed
    and a spatial interpolation (bilinear or bicubic) is applied at each step,
    followed by linear time interpolation.  No large intermediate array is built;
    the working memory per thread is O(K) scalars.

    Parameters
    ----------
    data : (nt_src, K, n_src) float64
        Source data; K = product of extra dims (e.g. depth levels collapsed),
        n_src = ny_src * nx_src (spatial grid flattened in C order).
    j_base : (n_dst,) int64
        Row index of the SW corner; -1 marks an invalid (out-of-domain) point.
    i_base : (n_dst,) int64
        Column index of the SW corner.
    frac_a : (n_dst,) float64
        Fractional x position in [0, 1].
    frac_b : (n_dst,) float64
        Fractional y position in [0, 1].
    it_base : (n_dst,) int64
        Index of the lower bracketing source time step; -1 if out-of-range.
    frac_t : (n_dst,) float64
        Fractional time position in [0, 1] toward it_base+1.
    nx_src : int
        Number of source grid columns.
    ny_src : int
        Number of source grid rows (used only for bicubic bounds checks).
    skipna : bool
        When True, renormalise bilinear weights over valid (non-NaN) neighbours.
    na_thres : float
        Minimum valid weight fraction; output is NaN when below this threshold.
    method : int
        Spatial interpolation: 1=bilinear, 2=bicubic.
    time_method : int
        Temporal interpolation: 1=linear.
    bias : float
        Kochanek-Bartels bias for bicubic (0.0 = Catmull-Rom).
    tension : float
        Kochanek-Bartels tension for bicubic (0.0 = standard).

    Returns
    -------
    (K, n_dst) float64
    """
    n_dst = j_base.shape[0]
    K = data.shape[1]
    nan = np.nan
    na_threshold = max(EPSILON, 1.0 - na_thres)

    out = np.full((K, n_dst), nan)

    for dst_idx in numba.prange(n_dst):
        j0 = j_base[dst_idx]
        if j0 < 0:
            continue
        it = it_base[dst_idx]
        if it < 0:
            continue

        i0 = i_base[dst_idx]
        a = frac_a[dst_idx]
        b = frac_b[dst_idx]
        wt = frac_t[dst_idx]

        if method == 1:
            # Bilinear: 4-point stencil, computed once per output point
            w00 = (1.0 - a) * (1.0 - b)
            w10 = a * (1.0 - b)
            w01 = (1.0 - a) * b
            w11 = a * b
            c00 = j0 * nx_src + i0
            c10 = j0 * nx_src + i0 + 1
            c01 = (j0 + 1) * nx_src + i0
            c11 = (j0 + 1) * nx_src + i0 + 1

            for k in range(K):
                v00_0 = data[it, k, c00]
                v10_0 = data[it, k, c10]
                v01_0 = data[it, k, c01]
                v11_0 = data[it, k, c11]
                v00_1 = data[it + 1, k, c00]
                v10_1 = data[it + 1, k, c10]
                v01_1 = data[it + 1, k, c01]
                v11_1 = data[it + 1, k, c11]

                if skipna:
                    s0 = ws0 = s1 = ws1 = 0.0
                    if not np.isnan(v00_0):
                        s0 += w00 * v00_0
                        ws0 += w00
                    if not np.isnan(v10_0):
                        s0 += w10 * v10_0
                        ws0 += w10
                    if not np.isnan(v01_0):
                        s0 += w01 * v01_0
                        ws0 += w01
                    if not np.isnan(v11_0):
                        s0 += w11 * v11_0
                        ws0 += w11
                    if not np.isnan(v00_1):
                        s1 += w00 * v00_1
                        ws1 += w00
                    if not np.isnan(v10_1):
                        s1 += w10 * v10_1
                        ws1 += w10
                    if not np.isnan(v01_1):
                        s1 += w01 * v01_1
                        ws1 += w01
                    if not np.isnan(v11_1):
                        s1 += w11 * v11_1
                        ws1 += w11
                    if ws0 < na_threshold or ws1 < na_threshold:
                        continue
                    val0 = s0 / ws0
                    val1 = s1 / ws1
                else:
                    val0 = w00 * v00_0 + w10 * v10_0 + w01 * v01_0 + w11 * v11_0
                    val1 = w00 * v00_1 + w10 * v10_1 + w01 * v01_1 + w11 * v11_1
                    if np.isnan(val0) or np.isnan(val1):
                        continue

                out[k, dst_idx] = (1.0 - wt) * val0 + wt * val1

        else:
            # Bicubic: 4x4 stencil, computed once per output point
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

            # Bilinear fallback on the inner 2x2 (indices 5,6,9,10 in the 4x4)
            w_bil = np.empty(4, dtype=np.float64)
            c_bil = np.empty(4, dtype=np.int64)
            w_bil[0] = (1.0 - a) * (1.0 - b)
            c_bil[0] = stencil_flat[5]
            w_bil[1] = a * (1.0 - b)
            c_bil[1] = stencil_flat[6]
            w_bil[2] = (1.0 - a) * b
            c_bil[2] = stencil_flat[9]
            w_bil[3] = a * b
            c_bil[3] = stencil_flat[10]

            for k in range(K):
                has_nan_0 = False
                has_nan_1 = False
                for p in range(16):
                    if np.isnan(data[it, k, stencil_flat[p]]):
                        has_nan_0 = True
                        break
                for p in range(16):
                    if np.isnan(data[it + 1, k, stencil_flat[p]]):
                        has_nan_1 = True
                        break

                if not has_nan_0:
                    s0 = 0.0
                    for p in range(16):
                        s0 += w_bic[p] * data[it, k, stencil_flat[p]]
                    val0 = s0
                else:
                    s0 = ws0 = 0.0
                    for p in range(4):
                        v = data[it, k, c_bil[p]]
                        if not np.isnan(v):
                            s0 += w_bil[p] * v
                            ws0 += w_bil[p]
                    val0 = s0 / ws0 if ws0 >= na_threshold else nan

                if not has_nan_1:
                    s1 = 0.0
                    for p in range(16):
                        s1 += w_bic[p] * data[it + 1, k, stencil_flat[p]]
                    val1 = s1
                else:
                    s1 = ws1 = 0.0
                    for p in range(4):
                        v = data[it + 1, k, c_bil[p]]
                        if not np.isnan(v):
                            s1 += w_bil[p] * v
                            ws1 += w_bil[p]
                    val1 = s1 / ws1 if ws1 >= na_threshold else nan

                if np.isnan(val0) or np.isnan(val1):
                    continue

                out[k, dst_idx] = (1.0 - wt) * val0 + wt * val1

    return out


# %% XYInterpolator


def _get_source_matrix_(data, mask=None):
    """Get the ``(n_src, K)`` matrix of the kernels from ``(..., ny, nx)`` data

    The matrix is C-contiguous and masked points are set to nan.
    The input is never modified, and it is copied only when needed: not at all for a
    single contiguous field without mask, and once otherwise.
    """
    ny, nx = data.shape[-2:]
    nex = int(np.prod(data.shape[:-2])) if data.ndim > 2 else 1
    matrix = np.ascontiguousarray(data.reshape(nex, ny * nx).T)
    if mask is not None:
        if np.shares_memory(matrix, data):
            matrix = matrix.copy()
        matrix[~mask.ravel()] = np.nan
    return matrix


class XYInterpolator:
    """
    Numba-accelerated interpolator from a 2D source grid to arbitrary destination points.

    Supports bilinear and bicubic methods. The destination can be any shape:
    a single point, a 1D transect, a 2D grid, etc.

    The source must be a 2D curvilinear, rectangular, or regular grid.
    The destination is any array of (lon, lat) pairs — no grid structure required.
    """

    valid_methods = ['bilinear', 'bicubic']
    valid_grid_types = ['regular', 'rectangular', 'curvilinear']

    def __init__(self, src_grid, dst_lon, dst_lat, method='bilinear', bias=0.0, tension=0.0):
        """
        Parameters
        ----------
        src_grid : dict
            Keys: 'lon' (2D array), 'lat' (2D array), optional 'mask' (2D bool), 'type' (str).
        dst_lon, dst_lat : array-like
            Destination coordinates, any shape. Scalars are treated as shape (1,).
            Output shape matches this shape.
        method : str
            'bilinear' or 'bicubic'
        bias, tension : float
            Kochanek-Bartels parameters for bicubic interpolation (default 0).
        """
        if method not in self.valid_methods:
            raise ValueError(
                f"Method '{method}' not supported. Valid methods: {self.valid_methods}"
            )

        self.src_grid = src_grid.copy()
        self.method = method
        self.bias = bias
        self.tension = tension

        dst_lon = np.asarray(dst_lon, dtype=np.float64)
        dst_lat = np.asarray(dst_lat, dtype=np.float64)
        if dst_lon.ndim == 0:
            dst_lon = dst_lon.reshape(1)
            dst_lat = dst_lat.reshape(1)
        if dst_lon.shape != dst_lat.shape:
            raise ValueError('dst_lon and dst_lat must have the same shape')
        self._dst_shape = dst_lon.shape
        self._dst_lon_flat = np.ascontiguousarray(dst_lon.ravel())
        self._dst_lat_flat = np.ascontiguousarray(dst_lat.ravel())

        # Weight state
        self._j_base = None
        self._i_base = None
        self._frac_a = None
        self._frac_b = None
        self._valid_dst_mask = None

        self._validate_src()
        self._prepare_src()

    @property
    def has_weights(self):
        """True if :meth:`compute_weights` has been called successfully."""
        return self._j_base is not None

    @property
    def dst_shape(self):
        """Shape of the destination point array."""
        return self._dst_shape

    def get_weights(self):
        """
        Return the computed fractional-index weights as a dict.

        Keys: ``j_base``, ``i_base``, ``frac_a``, ``frac_b``, ``valid_dst_mask``.
        Raises ``ValueError`` if weights have not been computed yet.
        """
        if not self.has_weights:
            raise ValueError('No weights computed yet. Call compute_weights() first.')
        return {
            'j_base': self._j_base,
            'i_base': self._i_base,
            'frac_a': self._frac_a,
            'frac_b': self._frac_b,
            'valid_dst_mask': self._valid_dst_mask,
        }

    def set_weights(self, weights):
        """
        Restore fractional-index weights from a dict (as returned by :meth:`get_weights`).

        Parameters
        ----------
        weights : dict
            Must contain ``j_base``, ``i_base``, ``frac_a``, ``frac_b``, ``valid_dst_mask``.
        """
        self._j_base = weights['j_base']
        self._i_base = weights['i_base']
        self._frac_a = weights['frac_a']
        self._frac_b = weights['frac_b']
        self._valid_dst_mask = weights['valid_dst_mask']

    def _validate_src(self):
        grid = self.src_grid
        for key in ['lon', 'lat']:
            if key not in grid:
                raise ValueError(f"source grid missing required key: '{key}'")

        if not isinstance(grid['lon'], np.ndarray) or not isinstance(grid['lat'], np.ndarray):
            raise ValueError("source grid 'lon' and 'lat' must be numpy arrays")

        if grid['lon'].shape != grid['lat'].shape:
            raise ValueError("source grid 'lon' and 'lat' must have the same shape")

        if len(grid['lon'].shape) != 2:
            raise ValueError("source grid 'lon' and 'lat' must be 2D arrays")

        if np.any(grid['lon'] < -180) or np.any(grid['lon'] > 180):
            raise ValueError('source grid longitudes must be in range [-180, 180]')

        if np.any(grid['lat'] < -90) or np.any(grid['lat'] > 90):
            raise ValueError('source grid latitudes must be in range [-90, 90]')

        if 'mask' in grid and grid['mask'] is not None:
            if grid['mask'].shape != grid['lon'].shape:
                raise ValueError('source grid mask must have same shape as coordinates')
            if grid['mask'].dtype != bool:
                xoa_warn('Converting source grid mask to boolean')
                grid['mask'] = grid['mask'].astype(bool)

        if 'type' not in grid or grid['type'] is None:
            check_grid_type(grid)
        else:
            if grid['type'] not in self.valid_grid_types:
                raise ValueError(f"Unsupported grid type '{grid['type']}'")

    def _prepare_src(self):
        for key in self.src_grid:
            if isinstance(self.src_grid[key], np.ndarray) and key != 'mask':
                self.src_grid[key] = np.ascontiguousarray(self.src_grid[key], dtype=np.float64)

    def compute_weights(self, skipna=False):
        """
        Compute fractional cell indices for each destination point.

        Parameters
        ----------
        skipna : bool
            Kept for API compatibility; does not affect which weights are computed.
        """
        grid_type_idx = self.valid_grid_types.index(self.src_grid['type'])
        stencil_margin = 1 if self.method == 'bicubic' else 0
        n = len(self._dst_lon_flat)
        dst_lon_2d = self._dst_lon_flat.reshape(n, 1)
        dst_lat_2d = self._dst_lat_flat.reshape(n, 1)
        self._j_base, self._i_base, self._frac_a, self._frac_b = compute_frac_indices(
            self.src_grid['lon'],
            self.src_grid['lat'],
            dst_lon_2d,
            dst_lat_2d,
            grid_type_idx,
            stencil_margin=stencil_margin,
        )
        self._valid_dst_mask = self._j_base >= 0

    def interp(self, data, skipna=False, na_thres=1.0):
        """
        Interpolate source data to destination points.

        Parameters
        ----------
        data : np.ndarray, shape (..., ny_src, nx_src)
        skipna : bool
        na_thres : float

        Returns
        -------
        np.ndarray, shape (..., *dst_shape)
        """
        if self._j_base is None:
            self.compute_weights()

        data = np.asarray(data, dtype=np.float64)
        ny_src, nx_src = self.src_grid['lon'].shape
        if data.shape[-2:] != (ny_src, nx_src):
            raise ValueError(
                f"Data shape {data.shape} doesn't match source grid ({ny_src}, {nx_src})"
            )

        n_dst = len(self._dst_lon_flat)
        extra = data.shape[:-2]
        K = int(np.prod(extra)) if extra else 1

        mask = self.src_grid.get('mask') if skipna else None
        X = _get_source_matrix_(data, mask)  # (n_src, K)
        out = np.empty((n_dst, K), dtype=np.float64)

        if self.method == 'bilinear':
            bilinear_frac(
                self._j_base,
                self._i_base,
                self._frac_a,
                self._frac_b,
                X,
                nx_src,
                out,
                skipna,
                na_thres,
            )
        else:  # bicubic
            bicubic_frac(
                self._j_base,
                self._i_base,
                self._frac_a,
                self._frac_b,
                X,
                ny_src,
                nx_src,
                out,
                skipna,
                na_thres,
                self.bias,
                self.tension,
            )

        return out.T.reshape(extra + self._dst_shape)

    def interp_with_time(
        self, data, src_times, dst_times, skipna=False, na_thres=1.0, time_method=1
    ):
        """
        Interpolate to scattered (x, y, t) locations.

        Applies the precomputed XY fractional weights at the two bracketing
        source time steps for each output point, then interpolates linearly in
        time.  No intermediate (nt, n_dst) array is constructed.

        Parameters
        ----------
        data : (nt, ..., ny_src, nx_src) float64
            Source data with time as the leading axis.
        src_times : (nt,) float64
            Source time coordinates, monotonically increasing (same units as
            dst_times; use :func:`xoa.core.num.as_float_array` to
            convert datetime arrays).
        dst_times : (n_dst,) float64
            Destination time coordinate, same shape as dst_lon/lat.
        skipna : bool, default False
            When True, renormalise bilinear weights over valid neighbours.
        na_thres : float, default 1.0
            Minimum valid-weight fraction; output is NaN below this threshold.
        time_method : int, default 1
            Temporal interpolation method: 1 = linear.

        Returns
        -------
        ndarray (..., *dst_shape)
            Extra dimensions (e.g. depth levels) are preserved; the time and
            spatial dimensions are replaced by the destination point shape.
        """
        if self._j_base is None:
            self.compute_weights()

        data = np.asarray(data, dtype=np.float64)
        ny_src, nx_src = self.src_grid['lon'].shape
        src_times = np.ascontiguousarray(src_times, dtype=np.float64)
        nt = len(src_times)

        if data.shape[0] != nt:
            raise ValueError(f'data.shape[0]={data.shape[0]} != len(src_times)={nt}')
        if data.shape[-2:] != (ny_src, nx_src):
            raise ValueError(
                f'data shape {data.shape} incompatible with source grid ({ny_src}, {nx_src})'
            )

        dst_times_flat = np.ascontiguousarray(np.asarray(dst_times, dtype=np.float64).ravel())
        n_dst = len(self._dst_lon_flat)
        if len(dst_times_flat) != n_dst:
            raise ValueError(f'dst_times size {len(dst_times_flat)} != n_dst {n_dst}')

        extra = data.shape[1:-2]
        K = int(np.prod(extra)) if extra else 1
        n_src = ny_src * nx_src

        has_mask = 'mask' in self.src_grid and self.src_grid['mask'] is not None
        if skipna and has_mask:
            data = data.copy()
            data[..., ~self.src_grid['mask']] = np.nan

        data_r = np.ascontiguousarray(data.reshape(nt, K, n_src))

        it_base, frac_t = compute_time_frac_indices(src_times, dst_times_flat)

        method_int = 1 if self.method == 'bilinear' else 2

        out = interp_transect(
            data_r,
            self._j_base,
            self._i_base,
            self._frac_a,
            self._frac_b,
            it_base,
            frac_t,
            nx_src,
            ny_src,
            bool(skipna),
            float(na_thres),
            method_int,
            time_method,
            float(self.bias),
            float(self.tension),
        )
        # out: (K, n_dst) → reshape to (...extra, *dst_shape)
        return out.reshape(extra + self._dst_shape)

    def save_weights(self, filename):
        """Save computed weights to a npz file"""
        if self._j_base is None:
            raise ValueError('No weights computed yet')
        with open(filename, "wb") as f:
            np.savez(
                f,
                method=self.method,
                src_shape=self.src_grid['lon'].shape,
                dst_shape=self._dst_shape,
                valid_dst_mask=self._valid_dst_mask,
                j_base=self._j_base,
                i_base=self._i_base,
                frac_a=self._frac_a,
                frac_b=self._frac_b,
            )

    def load_weights(self, filename):
        """Load precomputed weights from a npz file"""
        with np.load(filename) as p:
            p = {key: p[key] for key in p.files}
        if self.src_grid['lon'].shape != tuple(p['src_shape']):
            raise ValueError(
                f"Source grid shape {self.src_grid['lon'].shape} "
                f"doesn't match saved weights shape {tuple(p['src_shape'])}"
            )
        if self._dst_shape != tuple(p['dst_shape']):
            raise ValueError(
                f"Destination shape {self._dst_shape} doesn't match saved weights shape "
                f"{tuple(p['dst_shape'])}"
            )
        if str(p['method']) != self.method:
            xoa_warn(f"Saved method '{p['method']}' differs from '{self.method}'")
        self._valid_dst_mask = p['valid_dst_mask']
        self._j_base = p['j_base']
        self._i_base = p['i_base']
        self._frac_a = p['frac_a']
        self._frac_b = p['frac_b']
