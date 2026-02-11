"""
Low level regridding routines accelerated with numba

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

from .num import ravel_index, unravel_index, get_iminmax


NOT_CI = os.environ.get("CI", "false") == "false"


@numba.njit(parallel=True, cache=NOT_CI)
def nearest1d(vari, yi, yo, eshapes, extrap="no", drop_na=False, maxgap=0):
    """Nearest interpolation of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0.0:  # above
                    break

                # Below
                if dy0 < 0.0:
                    iyominv = iyo + 1

                # Interpolations
                elif dy0 <= dy1:
                    varo[ix, iyo] = vari[ixi, iyi0]
                else:
                    varo[ix, iyo] = vari[ixi, iyi1]

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] >= yi[ixiy, iyimin]:
                    varo[ix, :iyo] = vari[ixi, iyimin]
                    break
        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo:] = vari[ixi, iyimax]
                    break

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def linear1d(vari, yi, yo, eshapes, extrap="no", drop_na=False, maxgap=0):
    """Linear interpolation of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0.0:  # above
                    break

                # Below
                if dy0 < 0.0:
                    iyominv = iyo + 1

                # Interpolation
                elif dy0 > 0.0 or dy1 > 0.0:
                    varo[ix, iyo] = (vari[ixi, iyi0] * dy1 + vari[ixi, iyi1] * dy0) / (dy0 + dy1)

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] < yi[ixiy, iyimin]:
                    varo[ix, iyo] = vari[ixi, iyimin]
                else:
                    break

        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomax, iyomin - 1, -1):  # Loop backwards
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo] = vari[ixi, iyimax]
                else:
                    break

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def cubic1d(vari, yi, yo, eshapes, extrap="no", drop_na=False, maxgap=0):
    """Cubic interpolation of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1
            if iyi0 == iyimin or np.isnan(vari[ixi, iyi0 - 1]):
                iyim1 = iyi0
            else:
                iyim1 = iyi0 - 1
            if iyi1 == iyimax or np.isnan(vari[ixi, iyi1 + 1]):
                iyi2 = iyi1
            else:
                iyi2 = iyi1 + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0.0:  # above
                    break

                # Below
                if dy0 < 0.0:
                    iyominv = iyo + 1

                # Inside
                if dy0 >= 0.0 and dy1 >= 0.0:
                    iyominv = iyo
                    mu = dy0 / (dy0 + dy1)

                    # Extrapolations
                    if iyi0 == iyimin:  # y0
                        vc0 = 2 * vari[ixi, iyi0] - vari[ixi, iyi1]
                    else:
                        vc0 = vari[ixi, iyim1]
                    if iyi1 == iyimax:  # y3
                        vc1 = 2 * vari[ixi, iyi1] - vari[ixi, iyi0]
                    else:
                        vc1 = vari[ixi, iyi2]

                    # Interpolation
                    varo[ix, iyo] = vc1 - vari[ixi, iyi1] - vc0 + vari[ixi, iyi0]
                    varo[ix, iyo] = mu**3 * varo[ix, iyo] + mu**2 * (
                        vc0 - vari[ixi, iyi0] - varo[ix, iyo]
                    )
                    varo[ix, iyo] += mu * (vari[ixi, iyi1] - vc0)
                    varo[ix, iyo] += vari[ixi, iyi0]

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] >= yi[ixiy, iyimin]:
                    varo[ix, :iyo] = vari[ixi, iyimin]
                    break
        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo:] = vari[ixi, iyimax]
                    break

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def hermit1d(
    vari,
    yi,
    yo,
    eshapes,
    extrap="no",
    bias=0.0,
    tension=0.0,
    drop_na=False,
    maxgap=0,
):
    """Hermitian interp. of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)
    bias: float
    tension: float

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1
            if iyi0 == iyimin or np.isnan(vari[ixi, iyi0 - 1]):
                iyim1 = iyi0
            else:
                iyim1 = iyi0 - 1
            if iyi1 == iyimax or np.isnan(vari[ixi, iyi1 + 1]):
                iyi2 = iyi1
            else:
                iyi2 = iyi1 + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0.0:  # above
                    break

                # Below
                if dy0 < 0.0:
                    iyominv = iyo + 1

                # Inside
                if dy0 >= 0.0 and dy1 >= 0.0:
                    iyominv = iyo
                    mu = dy0 / (dy0 + dy1)

                    # Extrapolations
                    if iyi0 == iyimin:  # y0
                        vc0 = 2 * vari[ixi, iyi0] - vari[ixi, iyi1]
                    else:
                        vc0 = vari[ixi, iyim1]
                    if iyi1 == iyimax:  # y3
                        vc1 = 2 * vari[ixi, iyi1] - vari[ixi, iyi0]
                    else:
                        vc1 = vari[ixi, iyi2]

                    # Interpolation
                    mu = dy0 / (dy0 + dy1)
                    a0 = 2 * mu**3 - 3 * mu**2 + 1
                    a1 = mu**3 - 2 * mu**2 + mu
                    a2 = mu**3 - mu**2
                    a3 = -2 * mu**3 + 3 * mu**2
                    varo[ix, iyo] = a0 * vari[ixi, iyi0]
                    varo[ix, iyo] += a1 * (
                        (vari[ixi, iyi0] - vc0) * (1 + bias) * (1 - tension) / 2
                        + (vari[ixi, iyi1] - vari[ixi, iyi0]) * (1 - bias) * (1 - tension) / 2
                    )
                    varo[ix, iyo] += a2 * (
                        (vari[ixi, iyi1] - vari[ixi, iyi0]) * (1 + bias) * (1 - tension) / 2
                        + (vc1 - vari[ixi, iyi1]) * (1 - bias) * (1 - tension) / 2
                    )
                    varo[ix, iyo] += a3 * vari[ixi, iyi1]

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] >= yi[ixiy, iyimin]:
                    varo[ix, :iyo] = vari[ixi, iyimin]
                    break
        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo:] = vari[ixi, iyimax]
                    break

    return varo


@numba.njit(parallel=True)
def extrap1d(vari, mode):
    """Extrapolate valid data to the top and/or bottom

    Parameters
    ----------
    vari: array_like(nx, ny)
    mode: {"top", "bottom", "both", "no"}
        Extrapolation mode

    Return
    ------
    array_like(nx, ny): varo
    """
    varo = vari.copy()
    if mode == "no":
        return varo
    nx, ny = vari.shape

    # Loop on varying dim
    for ix in numba.prange(0, nx):
        iybot, iytop = get_iminmax(vari[ix])
        if iybot == -1:
            continue
        if mode == "both" or mode == "bottom":
            varo[ix, :iybot] = varo[ix, iybot]
        if mode == "both" or mode == "top":
            varo[ix, iytop + 1 :] = varo[ix, iytop]

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def cellave1d(
    vari,
    yib,
    yob,
    eshapes,
    extrap="no",
    conserv=False,
    drop_na=False,
    maxgap=0,
):
    """Cell average regrid. of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yib: array_like(nxiy, nyi+1)
    yob: array_like(nxo, nyo+1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    # nxi, nyib = vari.shape
    # nxiy, nyi = yib.shape
    # nxi, nyi = vari.shape
    # nxo, nyob = yob.shape
    # nx = max(nxi, nxo)
    # nyo = nyob - 1

    nyi = vari.shape[1]
    nyo = yob.shape[1] - 1
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.zeros((nx, nyo), dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for coordinate arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on output cells to be filled
        iyi0 = 0
        for iyo in range(nyo):
            if yob[ixoy, iyo] == yob[ixoy, iyo + 1]:
                continue

            # Loop on input cells
            wo = 0.0
            for iyi in range(iyi0, nyi):
                # Current input bounds
                yib0 = yib[ixiy, iyi]
                yib1 = yib[ixiy, iyi + 1]

                # Extrapolation
                if (extrap == "bellow" or extrap == "both") and iyi == 0 and yib0 > yob[ixoy, iyo]:
                    yib0 = yob[ixoy, iyo]
                if (
                    (extrap == "above" or extrap == "both")
                    and iyi == nyi - 1
                    and yib1 < yob[ixoy, iyo + 1]
                ):
                    yib1 = yob[ixoy, iyo + 1]

                # No intersection
                if yib0 > yob[ixoy, iyo + 1]:
                    break
                if yib1 < yob[ixoy, iyo]:
                    iyi0 = iyi + 1
                    continue

                # Contribution of intersection
                dyio = min(yib1, yob[ixoy, iyo + 1]) - max(yib0, yob[ixoy, iyo])
                if conserv and yib0 != yib1:
                    dyio = dyio / (yob[ixoy, iyo + 1] - yob[ixoy, iyo])
                if not np.isnan(vari[ixi, iyi]):
                    wo = wo + dyio
                    varo[ixi, iyo] += vari[ixi, iyi] * dyio

                # Next input cell?
                if yib1 >= yob[ixoy, iyo + 1]:
                    break

            # Normalize
            if not conserv:
                if wo != 0:
                    varo[ix, iyo] /= wo
                else:
                    varo[ix, iyo] = np.nan

    return varo
