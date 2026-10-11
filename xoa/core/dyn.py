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


def _prepare_(u, v, dx, dy):
    u = np.asarray(u, dtype="d")
    v = np.asarray(v, dtype="d")
    shape = u.shape[-2:]
    dx = np.broadcast_to(np.asarray(dx, dtype="d"), shape)
    dy = np.broadcast_to(np.asarray(dy, dtype="d"), shape)
    return u, v, dx, dy


def kinematics(u, v, dx, dy):
    """Divergence, strain and relative vorticity, computed in a single pass

    Parameters
    ----------
    u, v: array_like(..., ny, nx)
        Velocity components in m s-1
    dx, dy: float, array_like(ny, nx)
        Resolutions along X and Y in meters

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
    return _kinematics_(*_prepare_(u, v, dx, dy))


def div(u, v, dx, dy):
    """Horizontal divergence in s-1"""
    return kinematics(u, v, dx, dy)[0]


def strain(u, v, dx, dy):
    """Normal and shear strain in s-1"""
    return kinematics(u, v, dx, dy)[1:3]


def relvort(u, v, dx, dy):
    """Relative vorticity in s-1"""
    return kinematics(u, v, dx, dy)[3]


def okuboweiss(u, v, dx, dy):
    """Okubo-Weiss parameter in s-2"""
    _, sn, ss, vort = kinematics(u, v, dx, dy)
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
