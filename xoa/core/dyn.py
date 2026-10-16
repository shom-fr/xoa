"""
Ocean dynamics numeric kernels

Kinematic quantities computed on collocated grids from arrays whose last two
axes are Y and X. The resolutions ``dx`` and ``dy`` are in meters and can be
scalars or arrays broadcastable to the last two axes. All the routines are
numba accelerated and propagate the missing values.

Sign conventions are the usual ones: the relative vorticity is positive for a
cyclonic rotation in the northern hemisphere, i.e. counter-clockwise, and the
Okubo-Weiss parameter is negative where rotation dominates strain.
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

#: Gravity acceleration in m s-2
GRAVITY = 9.81

#: Earth angular velocity in rad s-1
OMEGA = 2 * np.pi / 86400


@numba.njit(cache=NOT_CI, nogil=True)
def _diff_(a, j, i, axis):
    """Centered difference, one-sided on the edges, like :func:`numpy.gradient`

    It is the difference of two grid points, to be divided by the local resolution.
    """
    n = a.shape[axis]
    k = j if axis == 0 else i
    if n < 2:
        return np.nan
    if k == 0:
        k0, k1, h = 0, 1, 1.0
    elif k == n - 1:
        k0, k1, h = n - 2, n - 1, 1.0
    else:
        k0, k1, h = k - 1, k + 1, 0.5
    if axis == 0:
        return h * (a[k1, i] - a[k0, i])
    return h * (a[j, k1] - a[j, k0])


@numba.guvectorize(
    [
        (
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
        )
    ],
    "(ny,nx),(ny,nx),(ny,nx),(ny,nx)->(ny,nx),(ny,nx),(ny,nx),(ny,nx)",
    nopython=True,
    cache=NOT_CI,
)
def _kinematics_(u, v, dx, dy, div, sn, ss, vort):
    ny, nx = u.shape
    for j in range(ny):
        for i in range(nx):
            if np.isnan(u[j, i]) or np.isnan(v[j, i]):
                div[j, i] = sn[j, i] = ss[j, i] = vort[j, i] = np.nan
                continue
            ux = _diff_(u, j, i, 1) / dx[j, i]
            uy = _diff_(u, j, i, 0) / dy[j, i]
            vx = _diff_(v, j, i, 1) / dx[j, i]
            vy = _diff_(v, j, i, 0) / dy[j, i]
            div[j, i] = ux + vy
            sn[j, i] = ux - vy
            ss[j, i] = vx + uy
            vort[j, i] = vx - uy


@numba.njit(cache=NOT_CI, nogil=True)
def _diff_z_(a, z, k, j, i, axis):
    """Derivative along the vertical, in the same unit as `a` over the one of `z`

    It is centered, and one-sided at the first and last levels.
    """
    nz = a.shape[0]
    if nz < 2:
        return np.nan
    k0 = max(k - 1, 0)
    k1 = min(k + 1, nz - 1)
    return (a[k1, j, i] - a[k0, j, i]) / (z[k1, j, i] - z[k0, j, i])


@numba.njit(cache=NOT_CI, nogil=True)
def _gradient_z_(a, z, dx, dy, k, j, i, az):
    """Horizontal derivatives at constant z of the level k, given the vertical one

    The derivatives are first computed along the levels, and then corrected with the slope
    of the levels, since :math:`\\partial_x a|_z = \\partial_x a|_s - \\partial_x z|_s \\partial_z a`.
    """
    ax = _diff_(a[k], j, i, 1) / dx[j, i] - _diff_(z[k], j, i, 1) / dx[j, i] * az
    ay = _diff_(a[k], j, i, 0) / dy[j, i] - _diff_(z[k], j, i, 0) / dy[j, i] * az
    return ax, ay


@numba.guvectorize(
    [
        (
            numba.float64[:, :, :],
            numba.float64[:, :, :],
            numba.float64[:, :, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :, :],
            numba.float64[:, :, :],
            numba.float64[:, :, :],
            numba.float64[:, :, :],
        )
    ],
    "(nz,ny,nx),(nz,ny,nx),(nz,ny,nx),(ny,nx),(ny,nx)->(nz,ny,nx),(nz,ny,nx),(nz,ny,nx),(nz,ny,nx)",
    nopython=True,
    cache=NOT_CI,
)
def _kinematics_z_(u, v, z, dx, dy, div, sn, ss, vort):
    nz, ny, nx = u.shape
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                if np.isnan(u[k, j, i]) or np.isnan(v[k, j, i]) or np.isnan(z[k, j, i]):
                    div[k, j, i] = sn[k, j, i] = ss[k, j, i] = vort[k, j, i] = np.nan
                    continue
                uz = _diff_z_(u, z, k, j, i, 0)
                vz = _diff_z_(v, z, k, j, i, 0)
                ux, uy = _gradient_z_(u, z, dx, dy, k, j, i, uz)
                vx, vy = _gradient_z_(v, z, dx, dy, k, j, i, vz)
                div[k, j, i] = ux + vy
                sn[k, j, i] = ux - vy
                ss[k, j, i] = vx + uy
                vort[k, j, i] = vx - uy


def _prepare_(u, v, dx, dy):
    u = np.asarray(u, dtype="d")
    v = np.asarray(v, dtype="d")
    shape = u.shape[-2:]
    dx = np.broadcast_to(np.asarray(dx, dtype="d"), shape)
    dy = np.broadcast_to(np.asarray(dy, dtype="d"), shape)
    return u, v, dx, dy


def kinematics(u, v, dx, dy, z=None):
    """Divergence, strain and relative vorticity, computed in a single pass

    Parameters
    ----------
    u, v: array_like(..., ny, nx), array_like(..., nz, ny, nx)
        Velocity components in m s-1
    dx, dy: float, array_like(ny, nx)
        Resolutions along X and Y in meters
    z: None, array_like(..., nz, ny, nx)
        Vertical coordinate of the levels, that varies in space, like with sigma
        coordinates, and whose sign convention does not matter.
        The velocities are then 3D, and the horizontal derivatives are computed
        at constant z instead of along the levels, with :math:`\\partial_x u|_z =
        \\partial_x u|_s - \\partial_x z|_s \\partial_z u`. They are different
        where the levels are sloping and the currents are sheared.
        The vertical derivatives are one-sided at the first and last levels.

    Return
    ------
    array(..., ny, nx)
        Horizontal divergence in s-1
    array(..., ny, nx)
        Normal strain in s-1
    array(..., ny, nx)
        Shear strain in s-1
    array(..., ny, nx)
        Relative vorticity in s-1
    """
    if z is None:
        return _kinematics_(*_prepare_(u, v, dx, dy))
    u, v, dx, dy = _prepare_(u, v, dx, dy)
    z = np.asarray(z, dtype="d")
    if u.ndim < 3 or z.shape[-3:] != u.shape[-3:]:
        raise ValueError("u, v and z must have the same last three dimensions: z, y and x")
    return _kinematics_z_(u, v, z, dx[..., :, :], dy[..., :, :])


def div(u, v, dx, dy, z=None):
    """Horizontal divergence in s-1, see :func:`kinematics`"""
    return kinematics(u, v, dx, dy, z)[0]


def strain(u, v, dx, dy, z=None):
    """Normal and shear strain in s-1, see :func:`kinematics`"""
    return kinematics(u, v, dx, dy, z)[1:3]


def relvort(u, v, dx, dy, z=None):
    """Relative vorticity in s-1, see :func:`kinematics`"""
    return kinematics(u, v, dx, dy, z)[3]


def okuboweiss(u, v, dx, dy, z=None):
    """Okubo-Weiss parameter in s-2, see :func:`kinematics`"""
    _, sn, ss, vort = kinematics(u, v, dx, dy, z)
    return sn**2 + ss**2 - vort**2


def coriolis(lat):
    """Coriolis parameter in s-1, from latitudes in degrees"""
    return 2 * OMEGA * np.sin(np.radians(lat))


@numba.guvectorize(
    [
        (
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
        )
    ],
    "(ny,nx),(ny,nx),(ny,nx),(ny,nx)->(ny,nx),(ny,nx)",
    nopython=True,
    cache=NOT_CI,
)
def _geos_(ssh, dx, dy, corio, ug, vg):
    ny, nx = ssh.shape
    for j in range(ny):
        for i in range(nx):
            if np.isnan(ssh[j, i]) or corio[j, i] == 0.0:
                ug[j, i] = vg[j, i] = np.nan
                continue
            fg = GRAVITY / corio[j, i]
            ug[j, i] = -fg * _diff_(ssh, j, i, 0) / dy[j, i]
            vg[j, i] = fg * _diff_(ssh, j, i, 1) / dx[j, i]


def geos(ssh, dx, dy, corio):
    """Geostrophic currents in m s-1 from the sea surface height in meters

    They balance the Coriolis force with the pressure gradient, with centered differences:

    .. math::

        u_g = -\\frac{g}{f} \\partial_y \\eta, \\quad
        v_g = \\frac{g}{f} \\partial_x \\eta

    where :math:`\\eta` is the sea surface height, :math:`f` the Coriolis parameter
    and :math:`g = 9.81` m s-2 the gravity.

    Parameters
    ----------
    ssh: array_like(..., ny, nx)
        Sea surface height in meters
    dx, dy: float, array_like(ny, nx)
        Resolutions along X and Y in meters
    corio: float, array_like(ny), array_like(ny, nx)
        Coriolis parameter in s-1, that is constant, or varies along Y or along X and Y.
        The currents are missing where it is null.

    Return
    ------
    array(..., ny, nx)
        Zonal current
    array(..., ny, nx)
        Meridional current
    """
    ssh = np.asarray(ssh, dtype="d")
    shape = ssh.shape[-2:]
    corio = np.asarray(corio, dtype="d")
    if corio.ndim == 1:
        corio = corio[:, None]
    dx = np.broadcast_to(np.asarray(dx, dtype="d"), shape)
    dy = np.broadcast_to(np.asarray(dy, dtype="d"), shape)
    corio = np.broadcast_to(corio, shape)
    return _geos_(ssh, dx, dy, corio)

#: Advection terms of the iterative cyclogeostrophic method
CYCLOGEOS_TERMS = ("full", "penven")


@numba.njit(cache=NOT_CI, nogil=True)
def _smooth_(a, valid):
    """Smooth in place with a 1-2-1 filter along X and Y, using only the valid points"""
    ny, nx = a.shape
    tmp = a.copy()
    for j in range(ny):
        for i in range(nx):
            if valid[j, i]:
                w = 2.0
                tot = 2.0 * a[j, i]
                if i > 0 and valid[j, i - 1]:
                    w += 1.0
                    tot += a[j, i - 1]
                if i < nx - 1 and valid[j, i + 1]:
                    w += 1.0
                    tot += a[j, i + 1]
                tmp[j, i] = tot / w
    for j in range(ny):
        for i in range(nx):
            if valid[j, i]:
                w = 2.0
                tot = 2.0 * tmp[j, i]
                if j > 0 and valid[j - 1, i]:
                    w += 1.0
                    tot += tmp[j - 1, i]
                if j < ny - 1 and valid[j + 1, i]:
                    w += 1.0
                    tot += tmp[j + 1, i]
                a[j, i] = tot / w


@numba.guvectorize(
    [
        (
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.float64[:, :],
            numba.int64,
            numba.float64,
            numba.float64,
            numba.float64,
            numba.float64,
            numba.int64,
            numba.int64,
            numba.float64[:, :],
            numba.float64[:, :],
        )
    ],
    "(ny,nx),(ny,nx),(ny,nx),(ny,nx),(ny,nx),(),(),(),(),(),(),()->(ny,nx),(ny,nx)",
    nopython=True,
    cache=NOT_CI,
)
def _cyclogeos_(ug, vg, dx, dy, corio, niter, tol, relax, fmin, divmax, full, smooth, uc, vc):
    ny, nx = ug.shape
    u = ug.copy()
    v = vg.copy()
    un = ug.copy()
    vn = vg.copy()

    # Points that are updated: the ones whose geostrophic currents and neighbors are known
    active = np.zeros((ny, nx), dtype=np.bool_)
    vmax = 0.0
    for j in range(ny):
        for i in range(nx):
            if abs(corio[j, i]) >= fmin and not (np.isnan(ug[j, i]) or np.isnan(vg[j, i])):
                vmax = max(vmax, np.hypot(ug[j, i], vg[j, i]))
                ux = _diff_(ug, j, i, 1)
                uy = _diff_(ug, j, i, 0)
                vx = _diff_(vg, j, i, 1)
                vy = _diff_(vg, j, i, 0)
                active[j, i] = not np.isnan(ux + uy + vx + vy)
    vlim = divmax * vmax

    # Iterations: u = ug + k x (u . grad) u / f, where k x (a, b) = (-b, a)
    bad = np.zeros((ny, nx), dtype=np.bool_)
    change = np.zeros((ny, nx))
    adx = np.zeros((ny, nx))
    ady = np.zeros((ny, nx))
    for _ in range(niter):
        # Advection terms
        for j in range(ny):
            for i in range(nx):
                if not active[j, i]:
                    continue
                ux = _diff_(u, j, i, 1) / dx[j, i]
                uy = _diff_(u, j, i, 0) / dy[j, i]
                vx = _diff_(v, j, i, 1) / dx[j, i]
                vy = _diff_(v, j, i, 0) / dy[j, i]
                if full:
                    adx[j, i] = u[j, i] * ux + v[j, i] * uy
                    ady[j, i] = u[j, i] * vx + v[j, i] * vy
                else:
                    adx[j, i] = u[j, i] * ux
                    ady[j, i] = v[j, i] * vy

        # Smoothing that damps the grid scale modes, which are amplified by the iterations
        for _ in range(smooth):
            _smooth_(adx, active)
            _smooth_(ady, active)

        # New currents
        for j in range(ny):
            for i in range(nx):
                if not active[j, i] or bad[j, i]:
                    continue
                un[j, i] = u[j, i] + relax * (ug[j, i] - ady[j, i] / corio[j, i] - u[j, i])
                vn[j, i] = v[j, i] + relax * (vg[j, i] + adx[j, i] / corio[j, i] - v[j, i])
        maxchange = 0.0
        for j in range(ny):
            for i in range(nx):
                if not active[j, i] or bad[j, i]:
                    continue
                speed = np.hypot(un[j, i], vn[j, i])
                if not np.isfinite(speed) or speed > vlim:
                    # Diverging points are frozen to the geostrophic currents
                    bad[j, i] = True
                    u[j, i] = ug[j, i]
                    v[j, i] = vg[j, i]
                    un[j, i] = ug[j, i]
                    vn[j, i] = vg[j, i]
                    continue
                change[j, i] = np.hypot(un[j, i] - u[j, i], vn[j, i] - v[j, i])
                maxchange = max(maxchange, change[j, i])
                u[j, i] = un[j, i]
                v[j, i] = vn[j, i]
        if maxchange < tol:
            break

    for j in range(ny):
        for i in range(nx):
            if active[j, i] and not bad[j, i] and change[j, i] < tol:
                uc[j, i] = u[j, i]
                vc[j, i] = v[j, i]
            else:
                uc[j, i] = np.nan
                vc[j, i] = np.nan


def cyclogeos(
    ug,
    vg,
    dx,
    dy,
    corio,
    terms="full",
    smooth=None,
    niter=300,
    tol=1e-4,
    relax=0.5,
    fmin=1e-5,
    divmax=2.0,
):
    """Cyclogeostrophic currents from the geostrophic ones, with an iterative method

    The cyclogeostrophic currents :math:`\\mathbf{u}` balance the pressure gradient
    with the Coriolis and the advection terms, which are significant where the
    currents are curved, like in eddies.
    They are found with an iterative method, like Penven et al. (2014), which starts from
    the geostrophic currents and corrects them with the advection terms that are computed
    from the currents of the previous iteration, until they no longer change:

    .. math::

        u_{n+1} = u_g - \\frac{u_n \\partial_x v_n + v_n \\partial_y v_n}{f}, \\quad
        v_{n+1} = v_g + \\frac{u_n \\partial_x u_n + v_n \\partial_y u_n}{f}

    These are the four terms of the complete advection, which reproduces the gradient wind
    balance of an axisymmetric vortex. With ``terms="penven"``, only the two terms
    :math:`v \\partial_y v` and :math:`u \\partial_x u` are kept, which makes a smaller correction:

    .. math::

        u_{n+1} = u_g - \\frac{v_n \\partial_y v_n}{f}, \\quad
        v_{n+1} = v_g + \\frac{u_n \\partial_x u_n}{f}

    The strong anticyclones that have no solution make the iteration diverge.

    Parameters
    ----------
    ug, vg: array_like(..., ny, nx)
        Geostrophic currents in m s-1, see :func:`geos`
    dx, dy: float, array_like(ny, nx)
        Resolutions along X and Y in meters
    corio: float, array_like(ny), array_like(ny, nx)
        Coriolis parameter in s-1
    terms: {"full", "penven"}
        Advection terms of the correction: the four terms of the complete advection,
        which reproduces the gradient wind balance of an axisymmetric vortex, or only two of
        them, which is the formulation that is taught as the one of Penven et al. (2014).
    smooth: None, int
        Number of times the advection terms are smoothed at each iteration, with a 1-2-1
        filter along X and Y. The iterations amplify the shortest scales when the currents
        are strong compared to the Coriolis parameter times the resolution, which makes
        them diverge, even where a solution exists, and the smoothing damps these scales.
        It defaults to 1 with the ``"full"`` terms, that diverge otherwise, and to 0 with the
        ``"penven"`` terms, that converge.
    niter: int
        Maximal number of iterations
    tol: float
        The iterations stop when the currents change by less than this value, in m s-1
    relax: float
        Relaxation factor between 0 and 1, which slows down the convergence of the
        iterations that oscillate
    fmin: float
        Minimal absolute value of the Coriolis parameter, in s-1, below which the currents
        are missing, which is close to the equator
    divmax: float
        Points whose speed exceeds this multiple of the maximal geostrophic speed
        are considered as diverging

    Return
    ------
    array(..., ny, nx)
        Zonal cyclogeostrophic current
    array(..., ny, nx)
        Meridional cyclogeostrophic current

    The currents are missing where the geostrophic ones or one of their neighbors are,
    where the Coriolis parameter is too small, and where the iterations did not converge.

    References
    ----------
    Penven, P., Halo, I., Pous, S., Marié, L. (2014), Cyclogeostrophic balance in the
    Mozambique Channel, J. Geophys. Res. Oceans, 119, 1054-1067, doi:10.1002/2013JC009528.

    See also
    --------
    geos
    """
    if terms not in CYCLOGEOS_TERMS:
        raise ValueError(f"terms must be one of {', '.join(CYCLOGEOS_TERMS)}, not {terms!r}")
    if smooth is None:
        smooth = int(terms == "full")
    ug = np.asarray(ug, dtype="d")
    shape = ug.shape[-2:]
    corio = np.asarray(corio, dtype="d")
    if corio.ndim == 1:
        corio = corio[:, None]
    return _cyclogeos_(
        ug,
        np.asarray(vg, dtype="d"),
        np.broadcast_to(np.asarray(dx, dtype="d"), shape),
        np.broadcast_to(np.asarray(dy, dtype="d"), shape),
        np.broadcast_to(corio, shape),
        int(niter),
        float(tol),
        float(relax),
        float(fmin),
        float(divmax),
        int(terms == "full"),
        int(smooth),
    )
