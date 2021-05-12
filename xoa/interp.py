"""
Low level interpolation routines accelerated with numba

The numerical inputs and outputs of all these routines are of scalar
or numpy.ndarray type.
"""
# Copyright 2020-2021 Shom
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
import math

import numpy as np
import numba

from .geo import _haversine_

NOT_CI = os.environ.get("CI", "false") == "false"


# %% 1D routines

@numba.njit(cache=NOT_CI)
def get_iminmax(data1d):
    """The first and last non nan values for a 1d array

    Parameters
    ----------
    data1d: array_like(n)

    Return
    ------
    int
        Index of the first valid value
    int
        Index of the last valid value
    """
    imin = -1
    imax = -1
    n = len(data1d)
    for i in range(n):
        if imin < 0 and not np.isnan(data1d[i]):
            imin = i
        if imax < 0 and not np.isnan(data1d[n-1-i]):
            imax = n-1-i
        if imax > 0 and imin > 0:
            break
    return imin, imax


@numba.njit(parallel=False, cache=NOT_CI)
def nearest1d(vari, yi, yo, extrap="no"):
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

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nxi, nyi = vari.shape
    nxiy = yi.shape[0]
    nxi, nyi = vari.shape
    nxo, nyo = yo.shape
    nx = max(nxi, nxo)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):

        # Index along x for coordinate arrays
        ixi = min(nxi-1, ix % nxi)
        ixiy = min(nxiy-1, ix % nxiy)
        ixoy = min(nxo-1, ix % nxo)

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        for iyi in range(iyimin, iyimax):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Loop on output grid
            for iyo in range(iyomin, iyomax+1):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Below
                if dy0 < 0:
                    iyomin = iyo + 1

                # Interpolations
                elif dy0 <= dy1:
                    varo[ix, iyo] = vari[ixi, iyi]
                else:
                    varo[ix, iyo] = vari[ixi, iyi+1]

    # Extrapolation
    if extrap != "no":
        varo = extrap1d(varo, extrap)

    return varo


@numba.njit(parallel=False, cache=NOT_CI)
def linear1d(vari, yi, yo, extrap="no"):
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

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nxi, nyi = vari.shape
    nxiy = yi.shape[0]
    nxi, nyi = vari.shape
    nxo, nyo = yo.shape
    nx = max(nxi, nxo)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):

        # Index along x for coordinate arrays
        ixi = min(nxi-1, ix % nxi)
        ixiy = min(nxiy-1, ix % nxiy)
        ixoy = min(nxo-1, ix % nxo)

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        for iyi in range(iyimin, iyimax):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Loop on output grid
            for iyo in range(iyomin, iyomax+1):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Below
                if dy0 < 0:
                    iyomin = iyo + 1

                # Interpolation
                elif dy0 > 0 or dy1 > 0:

                    varo[ix, iyo] = (
                        (vari[ixi, iyi]*dy1 + vari[ixi, iyi+1]*dy0) /
                        (dy0+dy1))

    # Extrapolation
    if extrap != "no":
        varo = extrap1d(varo, extrap)

    return varo


@numba.njit(parallel=False, cache=NOT_CI)
def cubic1d(vari, yi, yo, extrap="no"):
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

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nxi, nyi = vari.shape
    nxiy = yi.shape[0]
    nxi, nyi = vari.shape
    nxo, nyo = yo.shape
    nx = max(nxi, nxo)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):

        # Index along x for coordinate arrays
        ixi = min(nxi-1, ix % nxi)
        ixiy = min(nxiy-1, ix % nxiy)
        ixoy = min(nxo-1, ix % nxo)

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        for iyi in range(iyimin, iyimax):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Loop on output grid
            for iyo in range(iyomin, nyo):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Inside
                if dy0 >= 0 and dy1 >= 0:

                    iyomin = iyo
                    mu = dy0 / (dy0+dy1)

                    # Extrapolations
                    if iyi == iyimin:  # y0
                        vc0 = 2*vari[ix, iyi] - vari[ix, iyi+1]
                    else:
                        vc0 = vari[ixi, iyi-1]
                    if iyi == iyimax-1:  # y3
                        vc1 = 2*vari[ixi, iyi+1] - vari[ixi, iyi]
                    else:
                        vc1 = vari[ixi, iyi+2]

                    # Interpolation
                    varo[ix, iyo] = (vc1 - vari[ix, iyi+1]
                                     - vc0 + vari[ix, iyi])
                    varo[ix, iyo] = (
                        mu**3*varo[ix, iyo] +
                        mu**2*(vc0 - vari[ix, iyi] - varo[ix, iyo]))
                    varo[ix, iyo] += mu*(vari[ix, iyi+1] - vc0)
                    varo[ix, iyo] += vari[ix, iyi]

    # Extrapolation
    if extrap != "no":
        varo = extrap1d(varo, extrap)

    return varo


@numba.njit(parallel=False, cache=NOT_CI)
def hermit1d(vari, yi, yo, extrap="no", bias=0., tension=0.):
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
    bias: float
    tension: float

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nxi, nyi = vari.shape
    nxiy = yi.shape[0]
    nxi, nyi = vari.shape
    nxo, nyo = yo.shape
    nx = max(nxi, nxo)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):

        # Index along x for coordinate arrays
        ixi = min(nxi-1, ix % nxi)
        ixiy = min(nxiy-1, ix % nxiy)
        ixoy = min(nxo-1, ix % nxo)

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        for iyi in range(iyimin, iyimax):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Loop on output grid
            for iyo in range(iyomin, nyo):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Inside
                if dy0 >= 0 and dy1 >= 0:

                    iyomin = iyo
                    mu = dy0 / (dy0+dy1)

                    # Extrapolations
                    if iyi == iyimin:  # y0
                        vc0 = 2*vari[ix, iyi] - vari[ix, iyi+1]
                    else:
                        vc0 = vari[ixi, iyi-1]
                    if iyi == iyimax-1:  # y3
                        vc1 = 2*vari[ixi, iyi+1] - vari[ixi, iyi]
                    else:
                        vc1 = vari[ixi, iyi+2]

                    # Interpolation
                    mu = dy0 / (dy0+dy1)
                    a0 = 2*mu**3 - 3*mu**2 + 1
                    a1 = mu**3 - 2*mu**2 + mu
                    a2 = mu**3 - mu**2
                    a3 = -2*mu**3 + 3*mu**2
                    varo[ix, iyo] = a0*vari[ix, iyi]
                    varo[ix, iyo] += a1*(
                        (vari[ix, iyi]-vc0) *
                        (1+bias)*(1-tension)/2 +
                        (vari[ix, iyi+1]-vari[ix, iyi]) *
                        (1-bias)*(1-tension)/2)
                    varo[ix, iyo] += a2*(
                        (vari[ix, iyi+1]-vari[ix, iyi]) *
                        (1+bias)*(1-tension)/2 +
                        (vc1-vari[ix, iyi+1]) *
                        (1-bias)*(1-tension)/2)
                    varo[ix, iyo] += a3*vari[ix, iyi+1]

    if extrap != "no":
        varo = extrap1d(varo, extrap)

    return varo


@numba.njit(parallel=False)
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
            varo[ix, iytop+1:] = varo[ix, iytop]

    return varo


@numba.njit(parallel=False, cache=NOT_CI)
def cellave1d(vari, yib, yob, extrap="no", conserv=False):
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
    nxi, nyib = vari.shape
    nxiy, nyi = yib.shape
    nxi, nyi = vari.shape
    nxo, nyob = yob.shape
    nx = max(nxi, nxo)
    nyo = nyob - 1

    # Init output
    varo = np.zeros((nx, nyo), dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):

        # Index along x for coordinate arrays
        ixi = min(nxi-1, ix % nxi)
        ixiy = min(nxiy-1, ix % nxiy)
        ixoy = min(nxo-1, ix % nxo)

        # Loop on output cells to be filled
        iyi0 = 0
        for iyo in range(nyo):

            if yob[ixoy, iyo] == yob[ixoy, iyo+1]:
                continue

            # Loop on input cells
            wo = 0.
            for iyi in range(iyi0, nyi):

                # Current input bounds
                yib0 = yib[ixiy, iyi]
                yib1 = yib[ixiy, iyi+1]

                # Extrapolation
                if((extrap == "bellow" or extrap == "both") and
                   iyi == 0 and yib0 > yob[ixoy, iyo]):
                    yib0 = yob[ixoy, iyo]
                if((extrap == "above" or extrap == "both") and
                   iyi == nyi-1 and yib1 < yob[ixoy, iyo+1]):
                    yib1 = yob[ixoy, iyo+1]

                # No intersection
                if yib0 > yob[ixoy, iyo+1]:
                    break
                if yib1 < yob[ixoy, iyo]:
                    iyi0 = iyi + 1
                    continue

                # Contribution of intersection
                dyio = min(yib1, yob[ixoy, iyo+1]) - max(yib0, yob[ixoy, iyo])
                if conserv and yib0 != yib1:
                    dyio = dyio / (yob[ixoy, iyo+1] - yob[ixoy, iyo])
                if not np.isnan(vari[ixi, iyi]):
                    wo = wo + dyio
                    varo[ixi, iyo] += vari[ix, iyi] * dyio

                # Next input cell?
                if yib1 >= yob[ixoy, iyo+1]:
                    break

            # Normalize
            if not conserv:
                if wo != 0:
                    varo[ix, iyo] /= wo
                else:
                    varo[ix, iyo] = np.nan

    return varo


# %% 2D routines

@numba.njit(fastmath=True)
def closest2d(xxi, yyi, xo, yo):
    """Find indices of closest point on 2D lon/lat grid

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
    nyi, nxi = xxi.shape
    mindist = np.pi
    i = 0
    j = 0
    for jt in range(0, nyi):
        for it in range(0, nxi):
            dist = _haversine_(xo, yo, xxi[jt, it], yyi[jt, it])
            if dist <= mindist:
                i = it
                j = jt
                mindist = dist
    return i, j


@numba.njit(fastmath=True)
def cell2relloc(x1, x2, x3, x4, y1, y2, y3, y4, x, y):
    """Compute coordinates of point relative to a curvilinear cell

    Cell shape::

      2 - 3
      |   |
      1 - 4

    Example
    -------
    >>> cell2relloc(0., -2., 0., 2., 0., 1., 1., 0., 0., 0.5)
    (0.5, 0.5)


    See also
    --------
    http://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/19890018062_1989018062.pdf
    """
    small = np.finfo(np.float64).eps * 2

    # Coefs
    a = x4 - x1
    b = x2 - x1
    c = x3 - x4 - x2 + x1
    d = y4 - y1
    e = y2 - y1
    f = y3 - y4 - y2 + y1

    # Solve A*p**2 + B*p + C = 0
    yy = y - y1
    xx = x - x1
    AA = c*d - a*f
    BB = -c*yy + b*d + xx*f - a*e
    CC = -yy*b + e*xx
    if abs(AA) < small:
        p1 = -CC / BB
        p2 = p1
    else:
        DD = BB**2 - 4*AA*CC
        sDD = math.sqrt(DD)
        p1 = (-BB-sDD) / (2*AA)
        p2 = (-BB+sDD) / (2*AA)

    # Get q from p
    if abs(b+c*p1) > small:
        q1 = (xx-a*p1)/(b+c*p1)
    else:
        q1 = (yy-d*p1)/(e+f*p1)

    # Select point closest to center
    if p1 < 0. or p1 > 1. or q1 < 0. or q1 > 1.:
        if abs(b+c*p2) > small:
            q2 = (xx-a*p2) / (b+c*p2)
        else:
            q2 = (yy-d*p2) / (e+f*p2)
        p = p2
        q = q2
    else:
        p = p1
        q = q1

    if p < -small or q < -small:
        p = -1.
        q = -1.

    return p, q


@numba.njit(fastmath=True, cache=NOT_CI)
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

    p = -1.
    q = -1.
    small = np.finfo(np.float64).eps
    nyi, nxi = xxi.shape

    # Find the closest corner
    ic, jc = closest2d(xxi, yyi, xo, yo)

    # Curvilinear to rectangular with a loop on four candidate cells
    for j in range(max(jc-1, 0), min(jc+1, nyi-1)):
        for i in range(max(ic-1, 0), min(ic+1, nxi-1)):

            # Get relative position
            a, b = cell2relloc(
                xxi[j, i], xxi[j+1, i], xxi[j+1, i+1], xxi[j, i+1],
                yyi[j, i], yyi[j+1, i], yyi[j+1, i+1], yyi[j, i+1],
                xo, yo)

            # Store absolute indices
            if ((a >= 0.-small) and (a <= 1.+small) and
                    (b >= 0.-small) and (b <= 1.+small)):
                p = np.float64(i) + a
                q = np.float64(j) + b
                return p, q
    return p, q


@numba.njit(fastmath=True, cache=NOT_CI)
def grid2rellocs(xxi, yyi, xo, yo):
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


@numba.njit(parallel=False, cache=NOT_CI)
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
    assert not curved or (nxi == nxiy and nyi == nyix), (
        "linear4dto1: Invalid curved dimensions")
    assert nxiz == 1 or nxiz == nxi, "grid2locs: Invalid nxiz dimension"
    assert nyiz == 1 or nyiz == nyi, "grid2locs: Invalid nyiz dimension"
    assert ntiz == 1 or ntiz == nti, "grid2locs: Invalid ntiz dimension"

    # Loop on ouput points
    for io in numba.prange(no):
    # for io in range(no):

        if masko[io]:
            continue

        if ((nxi != 1 and (xo[io] < ximin or xo[io] > ximax)) or
                (nyi != 1 and (yo[io] < yimin or yo[io] > yimax)) or
                (nzi != 1 and (zo[io] < zimin or zo[io] > zimax)) or
                (nti != 1 and (to[io] < timin or to[io] > timax))):
            continue

        # Weights
        if curved:

            p, q = grid2relloc(xxi, yyi, xo[io], yo[io])
            if p < 0 or p > nxi-1 or q < 0 or q > nyi-1:
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
                a = 0.
                npi = 1
            elif xxi[0, nxi-1] == xo[io]:
                i = nxi - 1
                a = 0.
                npi = 1
            else:
                # i = minloc(xxi[1, :], dim=1, mask=xxi(1,:)>xo[io])-1
                i = np.searchsorted(xxi[0, :], xo[io], "right") - 1
                a = xo[io] - xxi[0, i]
                if abs(a) > 180.:
                    a -= 180.  # FIXME: grid2locs: abs(a)>180.
                a = a / (xxi[0, i+1] - xxi[0, i])
                npi = 2

            # - Y
            if nyi == 1:
                j = 0
                b = 0.
                npj = 1
            elif yyi[nyi-1, 0] == yo[io]:
                j = nyi - 1
                b = 0.
                npj = 1
            else:
                # j = minloc(yyi[:,1], dim=1, mask=yyi[:,1]>yo[io]) - 1
                j = np.searchsorted(yyi[:, 0], yo[io], "right") - 1
                b = (yo[io]-yyi[j, 0]) / (yyi[j+1, 0]-yyi[j, 0])
                npj = 2

        # - T
        if nti == 1:
            l = 0
            d = 0.
            npl = 1
        elif ti[nti-1] == to[io]:
            l = nti - 1
            d = 0.
            npl = 1
        else:
            l = np.searchsorted(ti, to[io], "right") - 1
            # l = minloc(ti, dim=1, mask=ti>to[io])-1
            if ti[l+1] == ti[l]:
                d = 0.
                npl = 1
            else:
                d = (to[io]-ti[l]) / (ti[l+1]-ti[l])
                npl = 2

        # - Z
        c = np.zeros(nexz)
        k = np.zeros(nexz, 'l')
        npk = np.zeros(nexz, 'l')
        if nzi == 1:
            k[:] = 0
            c[:] = 0.
            npk[:] = 1
        else:

            # Local zi

            if nxiz == 1:
                npiz = 1
                az = 0.
                iz = 0
            else:
                npiz = npi
                az = a
                iz = i

            if nyiz == 1:
                npjz = 1
                bz = 0.
                jz = 0
            else:
                npjz = npj
                bz = b
                jz = j

            if ntiz == 1:
                nplz = 1
                dz = 0.
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
                                    zzi[ie, lz+ll, kk, jz+jj, iz+ii] *
                                    ((1-az) * (1-ii) + az * ii) *
                                    ((1-bz) * (1-jj) + bz * jj) *
                                    ((1-dz) * (1-ll) + dz * ll))

            # Normal stuff (c(nexz),zi[nexz,nzi),k(nexz)
            for ie in range(nexz):  # extra dim
                if zi[ie, nzi-1] == zo[io]:
                    k[ie] = nzi - 1
                    c[ie] = 0.
                    npk[ie] = 1
                else:
                    k[ie] = np.searchsorted(zi[ie], zo[io], "right") - 1
                    if zi[ie, k[ie]+1] == zi[ie, k[ie]]:
                        c[ie] = 0.
                        npk[ie] = 1
                    else:
                        c[ie] = ((zo[io]-zi[ie, k[ie]]) /
                                  (zi[ie, k[ie]+1] - zi[ie, k[ie]]))
                        npk[ie] = 2

        # Interpolate
        for ie in range(nex):
            if not bmask[
                    ie % nexv, l:l+npl,
                    k[ie % nexz]:k[ie % nexz]+npk[ie % nexz],
                    j:j+npj, i:i+npi].any():
                vo[ie % nex, io] = 0.
                for ll in range(npl):
                    for kk in range(npk[ie % nexz]):
                        for jj in range(npj):
                            for ii in range(npi):
                                vo[ie % nex, io] = (
                                    vo[ie % nex, io] +
                                    vi[ie % nex,
                                        l+ll, k[ie % nexz]+kk, j+jj, i+ii] *
                                    ((1-a) * (1-ii) + a * ii) *
                                    ((1-b) * (1-jj) + b * jj) *
                                    ((1-c[ie % nexz]) *
                                      (1-kk) + c[ie % nexz] * kk) *
                                    ((1-d) * (1-ll) + d * ll))

    return vo
