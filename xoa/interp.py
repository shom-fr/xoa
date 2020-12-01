"""
Low level interpolation routines accelerated with numba
"""
# Copyright or © or Copr. Shom, 2020
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import math
import numpy as np
import numba


# %% 1D routines

@numba.njit(parallel=True, cache=True)
def nearest1d(vari, yi, yo):
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
        iyo0 = 0
        for iyi in range(0, nyi-1):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, 0]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, -1]:
                break

            # Loop on output grid
            for iyo in range(iyo0, nyo):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Below
                if dy0 < 0:
                    iyo0 = iyo + 1

                # Interpolations
                elif dy0 <= dy1:
                    varo[ix, iyo] = vari[ixi, iyi]
                else:
                    varo[ix, iyo] = vari[ixi, iyi+1]

    return varo


@numba.njit(parallel=True, cache=True)
def linear1d(vari, yi, yo, bias=0., tension=0.):
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
        iyo0 = 0
        for iyi in range(0, nyi-1):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, 0]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, -1]:
                break

            # Loop on output grid
            for iyo in range(iyo0, nyo):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Below
                if dy0 < 0:
                    iyo0 = iyo + 1

                # Interpolation
                elif dy0 >= 0 and dy1 >= 0:

                    varo[ix, iyo] = (
                        (vari[ixi, iyi]*dy1 + vari[ixi, iyi+1]*dy0) /
                        (dy0+dy1))

        # # Extrapolation with nearest
        # if extrap != "no":
        #     for iyo in range(0, nyo):
        #         if extrap == "both" or extrap == "bottom":
        #             if yo[ixoy, iyo] < yi[ixiy, 0]:
        #                 varo[ix, iyo] = vari[ixi, 0]
        #         if extrap == "both" or extrap == "top":
        #             if yo[ixoy, iyo] > yi[ixiy, -1]:
        #                 varo[ix, iyo] = vari[ixi, -1]

    return varo


@numba.njit(parallel=True, cache=True)
def cubic1d(vari, yi, yo):
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
        iyo0 = 0
        for iyi in range(0, nyi-1):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, 0]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, -1]:
                break

            # Loop on output grid
            for iyo in range(iyo0, nyo):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Inside
                if dy0 >= 0 and dy1 >= 0:

                    iyo0 = iyo
                    mu = dy0 / (dy0+dy1)

                    # Extrapolations
                    if iyi == 0:  # y0
                        vc0 = 2*vari[ix, iyi] - vari[ix, iyi+1]
                    else:
                        vc0 = vari[ixi, iyi-1]
                    if iyi == nyi-2:  # y3
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

    return varo


@numba.njit(parallel=True, cache=True)
def hermit1d(vari, yi, yo, bias=0., tension=0.):
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
        iyo0 = 0
        for iyi in range(0, nyi-1):

            # Out of bounds
            if yi[ixiy, iyi+1] < yo[ixoy, 0]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, -1]:
                break

            # Loop on output grid
            for iyo in range(iyo0, nyo):

                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi]
                dy1 = yi[ixiy, iyi+1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Inside
                if dy0 >= 0 and dy1 >= 0:

                    iyo0 = iyo
                    mu = dy0 / (dy0+dy1)

                    # Extrapolations
                    if iyi == 0:  # y0
                        vc0 = 2*vari[ix, iyi] - vari[ix, iyi+1]
                    else:
                        vc0 = vari[ixi, iyi-1]
                    if iyi == nyi-2:  # y3
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

    return varo


@numba.njit(parallel=True, fastmath=True)
def extrap1d(vari, extrap):
    """Extrapolate valid data to the top and/or bottom

    Parameters
    ----------
    vari: array_like(nx, ny)
    extrap: {"top", "bottom", "both", "no"}
        Extrapolation mode

    Return
    ------
    array_like(nx, ny): varo
    """
    varo = vari.copy()
    if extrap == "no":
        return varo
    nx, ny = vari.shape

    # Loop on varying dim
    for ix in numba.prange(0, nx):

        iybot = -1
        iytop = -1
        for iy in range(ny):
            if not np.isnan(vari[ix, iy]):
                if iybot == -1:
                    iybot = iy
                iytop = iy
        if iybot == -1:
            continue

        if extrap == "both" or extrap == "bottom":
            varo[ix, :iybot] = varo[ix, iybot]
        if extrap == "both" or extrap == "top":
            varo[ix, iytop+1:] = varo[ix, iytop]

    return varo


@numba.njit(parallel=False, cache=True)
def cellave1d(vari, yib, yob, conserv=False, extrap="no"):
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

@numba.vectorize
def haversine(lon0, lat0, lon1, lat1):
    """Haversine distance between two points on a unit sphere

    Parameters
    ----------
    lon0: float
        Longitude of the first point
    lat0: float
        Latitude of the first point
    lon1: float
        Longitude of the second point
    lat1: float
        Latitude of the second point

    Return
    ------
    float
        Distance
    """
    deg2rad = np.pi / 180.
    dist = math.sin(deg2rad*(lat0-lat1)*0.5)**2
    dist += (math.cos(deg2rad*lat0) * math.cos(deg2rad*lat1) *
             math.sin(deg2rad*(lon0-lon1)*0.5)**2)
    dist = 2. * math.asin(math.sqrt(dist))
    return dist


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
    for it in range(0, nxi):
        for jt in range(0, nyi):
            dist = haversine(xo, yo, xxi[jt, it], yyi[jt, it])
            if dist < mindist:
                i = it
                j = jt
                mindist = dist
    return i, j


@numba.njit(fastmath=True)
def curvcell2relpt(x1, x2, x3, x4, y1, y2, y3, y4, x, y):
    """Compute coordinates of point relative to a curvilinear cell

    Cell shape::

      2 - 3
      |   |
      1 - 4

    Example
    -------
    >>> curv2rect(0., -2., 0., 2., 0., 1., 1., 0., 0., 0.5)
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

    return p, q


@numba.njit(fastmath=True)
def curvgrid2relpt(xxi, yyi, xo, yo):
    """Compute coordinates of point relative to a curvilinear grid

    Parameters
    ----------
    xxi: array_like(nyi, nxi)
        Grid longitudes in degrees
    yyi: array_like(nyi, nxi)
        Grid latitudes in degrees
    xo:
        Point longitude
    yo:
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
    for i in range(max(ic-1, 0), min(ic, nxi-2)):
        for j in range(max(jc-1, 0), min(jc, nyi-2)):

            # Get relative position
            a, b = curvcell2relpt(
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


def linear4dto1dxx(xxi, yyi, zzi, ti, vi, xo, yo, zo, to, vo):
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
    vi: array_like(nex, nti, nzi, nyi, nxi)
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
    nex, nti, nzi, nyi, nxi = vi.shape
    no = xo.shape[0]

    # Initalisations
    vo = np.full((nex, no), np.nan, dtype=vi.dtype)
    bmask = np.isnan(vi)
    ximin = vi.min()
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
    assert nxiz == 1 or nxiz == nxi, "linear4dto1: Invalid nxiz dimension"
    assert nyiz == 1 or nyiz == nyi, "linear4dto1: Invalid nyiz dimension"
    assert ntiz == 1 or ntiz == nti, "linear4dto1: Invalid ntiz dimension"

    # Loop on ouput points
    for io in numba.prange(no):
        if ((nxi == 1 or (xo[io] >= ximin and xo[io] <= ximax)) and
                (nyi == 1 or (yo[io] >= yimin and yo[io] <= yimax)) and
                (nzi == 1 or (zo[io] >= zimin and zo[io] <= zimax)) and
                (nti == 1 or (to[io] >= timin and to[io] <= timax))):
            continue

        # Weights
        if curved:

            p, q = curvgrid2relpt(xxi, yyi, xo[io], yo[io])
            if p < 1 or p > nxi or q < 1 or q > nyi:
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
                npi = 0
                a = 0.
            else:
                # i = minloc(xxi[1, :], dim=1, mask=xxi(1,:)>xo[io])-1
                i = np.searchsorted(xxi[0, :], xo[io], "right") - 1
                npi = 2
                a = xo[io] - xxi[0, i]
                if abs(a) > 180.:
                    a -= 180.  # FIXME: linear4dto1dxx: abs(a)>180.
                a = a / (xxi[0, i+1] - xxi[0, i])

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
        if nzi == 1:
            k = 0
            c = 0.
            npk = 1
        else:

            # Local zi

            if nxiz == 1:
                npiz = 1
                az = 0
                iz = 0
            else:
                npiz = npi
                az = a
                iz = i

            if nyiz == 1:
                npjz = 1
                bz = 0
                jz = 0
            else:
                npjz = npj
                bz = b
                jz = j

            if ntiz == 1:
                nplz = 1
                dz = 0
                lz = 0
            else:
                nplz = npl
                dz = d
                lz = l

            zi = 0.
            for ie in range(nex):
                for ll in range(0, nplz):
                    for kk in range(nzi):
                        for jj in range(0, npjz):
                            for ii in range(0, npiz):
                                zi = (zi + zzi[ie, lz+ll, kk, jz+jj, iz+ii] *
                                      ((1-az) * (1-ii) + az * ii) *
                                      ((1-bz) * (1-jj) + bz * jj) *
                                      ((1-dz) * (1-ll) + dz * ll))

            # Normal stuff (c(nexz),zi[nexz,nzi),k(nexz)
            for ie in range(nex):  # extra dim
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
        for ieb in range(0, nex//nexz):

            ie0 = (ieb-1)*nexz - 1

            for iez in range(nexz):
                if not bmask[ie0:ie0+iez, l:l+npl,
                             k[iez+1]:k[iez+1]+npk[iez+1],
                             j:j+npj,
                             i:i+npi].any():
                    vo[ie0+iez, io] = 0.
                    for ll in range(npl):
                        for kk in range(npk[iez+1]):
                            for jj in range(npj):
                                for ii in range(npi):
                                    vo[ie0+iez, io] = (
                                        vo[ie0+iez, io] +
                                        vi[ie0+iez,
                                           l+ll, k[iez]+kk, j+jj, i+ii] *
                                        ((1-a) * (1-ii) + a * ii) *
                                        ((1-b) * (1-jj) + b * jj) *
                                        ((1-c[iez]) *
                                         (1-kk) + c[iez] * kk) *
                                        ((1-d) * (1-ll) + d * ll))
