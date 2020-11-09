"""
Low level interpolation routines accelerated with numba

"""
import math
import numpy as np
import numba

# %% 1D routines

@numba.njit(parallel=True, fastmath=True)
def interp1d(vari, yi, yo, method="linear", extrap="no", bias=0., tension=0.):
    """Interpolation of nD data along an axis with varying coordinates

    Parameters
    ----------
    vari: array_like(nx, nb, nyi)
    yi: array_like(nx, nyi)
    yo: array_like(nx, nyo)
    method: {"nearest", "linear", "cubic", "hermit"}
        Interpolation method
    bias: float
        For the hermit method
    tension: float
        for the hermit method
    extrap: {"top", "bottom", "both", "no"}

    Return
    ------
    array_like(nx, nb, nyo): varo
    """
    # Shapes
    if vari.ndim == 2:
        vari = vari.reshape(vari.shape[0], 1, vari.shape[1])
    nx, nb, nyi = vari.shape
    yi = np.atleast_2d(yi)
    yo = np.atleast_2d(yo)
    nyo = yo.shape[1]

    # Init output
    varo = np.full((nx, nb, nyo), np.nan, dtype=vari.dtype)

    # Loop on varying dimension
    for ix in numba.prange(nx):

        # Index along x for coordinate arrays
        ixi = int(1 if yi.shape[0] == 1 else ix)
        ixo = int(1 if yo.shape[0] == 1 else ix)

        # Loop on input grid
        for iyi in range(0, nyi-1):

            # Loop on output grid
            for iyo in range(0, nyo):

                dy0 = yo[ixo, iyo] - yi[ixi, iyi]
                dy1 = yi[ixi, iyi+1] - yo[ixo, iyo]

                # Above
                if dy1 < 0:  # above
                    break

                # Interpolations
                if dy0 >= 0 and dy1 >= 0:
                    mu = dy0 / (dy0+dy1)
                    for ib in range(nb):  # loop on extra dimension

                        if method == "nearest":

                            if dy0 <= dy1:
                                varo[ix, ib, iyo] = vari[ix, ib, iyi]
                            else:
                                varo[ix, ib, iyo] = vari[ix, ib, iyi+1]

                        elif method == "linear":

                            varo[ix, ib, iyo] = (
                                (vari[ix, ib, iyi]*dy1 +
                                 vari[ix, ib, iyi+1]*dy0) /
                                (dy0+dy1))

                        else:

                            # Extrapolations
                            if iyi == 0:  # y0
                                vc0 = (2*vari[ix, ib, iyi]
                                       - vari[ix, ib, iyi+1])
                            else:
                                vc0 = vari[ix, ib, iyi-1]
                            if iyi == nyi-2:  # y3
                                vc1 = (2*vari[ix, ib, iyi+1]
                                       - vari[ix, ib, iyi])
                            else:
                                vc1 = vari[ix, ib, iyi+2]

                            if method == "cubic":

                                # Cubic
                                varo[ix, ib, iyo] = (vc1 - vari[ix, ib, iyi+1]
                                                     - vc0 + vari[ix, ib, iyi])
                                varo[ix, ib, iyo] = (
                                    mu**3*varo[ix, ib, iyo] +
                                    mu**2*(vc0 - vari[ix, ib, iyi] -
                                           varo[ix, ib, iyo]))
                                varo[ix, ib, iyo] += mu*(
                                    vari[ix, ib, iyi+1] - vc0)
                                varo[ix, ib, iyo] += vari[ix, ib, iyi]

                            else:

                                # Hermit
                                a0 = 2*mu**3 - 3*mu**2 + 1
                                a1 = mu**3 - 2*mu**2 + mu
                                a2 = mu**3 - mu**2
                                a3 = -2*mu**3 + 3*mu**2
                                varo[ix, ib, iyo] = a0*vari[ix, ib, iyi]
                                varo[ix, ib, iyo] += a1*(
                                    (vari[ix, ib, iyi]-vc0) *
                                    (1+bias)*(1-tension)/2 +
                                    (vari[ix, ib, iyi+1]-vari[ix, ib, iyi]) *
                                    (1-bias)*(1-tension)/2)
                                varo[ix, ib, iyo] += a2*(
                                    (vari[ix, ib, iyi+1]-vari[ix, ib, iyi]) *
                                    (1+bias)*(1-tension)/2 +
                                    (vc1-vari[ix, ib, iyi+1]) *
                                    (1-bias)*(1-tension)/2)
                                varo[ix, ib, iyo] += a3*vari[ix, ib, iyi+1]

        # Extrapolation with nearest
        if extrap != "no":
            for iyo in range(0, nyo):
                if extrap == "both" or extrap == "bottom":
                    if yo[ixo, iyo] < yi[ixi, 0]:
                        varo[ix, ib, iyo] = vari[ix, ib, 0]
                if extrap == "both" or extrap == "top":
                    if yo[ixo, iyo] > yi[ixi, -1]:
                        varo[ix, ib, iyo] = vari[ix, ib, -1]

    return varo


@numba.njit(parallel=True, fastmath=True)
def extrap1d(vari, varo, extrap):
    """Extrapolate valid data to the top and/or bottom

    Parameters
    ----------
    vari: array_like(nx, ny)
    extrap: {"top", "bottom", "both", "no"}

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


# %% 2D routines

#@numba.njit(fastmath=True)
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
                xxi(j, i), xxi(j+1, i), xxi(j+1, i+1), xxi(j, i+1),
                yyi(j, i), yyi(j+1, i), yyi(j+1, i+1), yyi(j, i+1),
                xo, yo)

            # Store absolute indices
            if ((a >= 0.-small) and (a <= 1.+small) and
                    (b >= 0.-small) and (b <= 1.+small)):
                p = np.float64(i) + a
                q = np.float64(j) + b
                return p, q

    return p, q
