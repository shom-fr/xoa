"""
Vertical modes of the ocean

Numba routines that compute the quasi-geostrophic vertical modes of water columns from
their stratification, and that project profiles on them. The last axis of all the arrays is
the vertical one, and the other ones are loop dimensions, that are broadcast.

The vertical coordinate ``z`` is positive up, and the levels may be in any order.
The modes :math:`F_n` are the solutions of the Sturm-Liouville problem

.. math::

    -\\frac{d}{dz} \\left( \\frac{1}{N^2} \\frac{dF}{dz} \\right) = \\frac{1}{c^2} F

with :math:`dF/dz = 0` at the bottom, and either :math:`dF/dz = 0` at the surface (rigid lid)
or :math:`g \\, dF/dz + N^2 F = 0` (free surface). They are orthonormal for the mean
over the water column: :math:`\\frac{1}{H} \\int F_m F_n dz = \\delta_{mn}`.
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

from .num import trideigh

NOT_CI = os.environ.get("CI", "false") == "false"

#: Gravity acceleration in m s-2
GRAVITY = 9.81

#: The modes are found with the tridiagonal solver when there are few of them compared to
#: the number of levels, and with the dense one otherwise
TRIDIAGONAL_MAX_MODES = 20
DENSE_RATIO = 4


@numba.guvectorize(
    [
        (
            numba.float64[:],
            numba.float64[:],
            numba.int64[:],
            numba.int64,
            numba.float64,
            numba.float64,
            numba.float64,
            numba.float64,
            numba.float64[:, :],
            numba.float64[:],
            numba.float64[:],
        )
    ],
    "(nz),(nz),(nm),(),(),(),(),()->(nm,nz),(nm),(nz)",
    nopython=True,
    cache=NOT_CI,
    target="parallel",
)
def _vertical_modes_(n2, z, nmodes, free_surface, zmin, zmax, n2min, gravity, modes, c, dz):
    nz = z.size
    nm = nmodes.size
    modes[:, :] = np.nan
    c[:] = np.nan
    dz[:] = np.nan

    # Valid levels, that are in the layer, sorted from the bottom to the surface
    count = 0
    for k in range(nz):
        if np.isfinite(z[k]) and np.isfinite(n2[k]) and z[k] >= zmin and z[k] <= zmax:
            count += 1
    if count < 2:
        return
    index = np.empty(count, dtype=np.int64)
    zv = np.empty(count)
    count = 0
    for k in range(nz):
        if np.isfinite(z[k]) and np.isfinite(n2[k]) and z[k] >= zmin and z[k] <= zmax:
            index[count] = k
            zv[count] = z[k]
            count += 1
    order = np.argsort(zv)
    m = count
    zs = zv[order]
    n2s = np.empty(m)
    for k in range(m):
        n2s[k] = max(n2[index[order[k]]], n2min)
    for k in range(m - 1):
        if not zs[k + 1] > zs[k]:
            return

    # Thickness of the layers, with half layers at the ends
    thick = np.empty(m)
    below = zs[0] - 0.5 * (zs[1] - zs[0])
    for k in range(m):
        if k < m - 1:
            above = 0.5 * (zs[k] + zs[k + 1])
        else:
            above = zs[m - 1] + 0.5 * (zs[m - 1] - zs[m - 2])
        thick[k] = above - below
        below = above
    depth = thick.sum()

    # Symmetric problem: K F = lambda W F, where W is the thickness, which is tridiagonal
    diag = np.zeros(m)
    off = np.empty(m - 1)
    for k in range(m - 1):
        coupling = 1.0 / (0.5 * (n2s[k] + n2s[k + 1]) * (zs[k + 1] - zs[k]))
        diag[k] += coupling
        diag[k + 1] += coupling
        off[k] = -coupling
    if free_surface:
        diag[m - 1] += 1.0 / gravity
    for k in range(m):
        diag[k] /= thick[k]
    for k in range(m - 1):
        off[k] /= np.sqrt(thick[k] * thick[k + 1])
    nmax = min(nm, m)
    if nmax <= TRIDIAGONAL_MAX_MODES and DENSE_RATIO * nmax <= m:
        values, vectors = trideigh(diag, off, nmax)
    else:
        a = np.zeros((m, m))
        for k in range(m):
            a[k, k] = diag[k]
        for k in range(m - 1):
            a[k, k + 1] = off[k]
            a[k + 1, k] = off[k]
        values, vectors = np.linalg.eigh(a)

    # Normalized modes, that are positive at the surface, and speeds
    for n in range(nmax):
        vmax = 0.0
        for k in range(m):
            vmax = max(vmax, abs(vectors[k, n]))
        sign = 1.0
        for k in range(m - 1, -1, -1):
            if abs(vectors[k, n]) > 1e-12 * vmax:
                sign = 1.0 if vectors[k, n] > 0 else -1.0
                break
        for k in range(m):
            modes[n, index[order[k]]] = sign * vectors[k, n] / np.sqrt(thick[k]) * np.sqrt(depth)
        if (n > 0 or free_surface) and values[n] > 0:
            c[n] = 1.0 / np.sqrt(values[n])
    for k in range(m):
        dz[index[order[k]]] = thick[k]


def vertical_modes(
    n2, z, nmodes=5, free_surface=False, zmin=-np.inf, zmax=np.inf, n2min=1e-9, gravity=GRAVITY
):
    """Vertical modes of water columns

    The problem is discretized with the levels where the stratification is given.
    The levels where the stratification or the depth is missing, like the ones that are
    below the bottom, are not used, which allows columns of different depths.

    Parameters
    ----------
    n2: array_like(..., nz)
        Squared Brunt-Vaisala frequency in s-2, at the levels
    z: array_like(..., nz)
        Vertical coordinate of the levels in meters, that is positive up. It is broadcast
        with `n2`, and the levels may be in any order.
    nmodes: int
        Number of modes, the first one being the barotropic mode.
        The modes that cannot be computed, because there are not enough levels,
        are missing.
    free_surface: bool
        Use the free surface boundary condition instead of the rigid lid.
        The speed of the barotropic mode is only defined with the free surface.
    zmin, zmax: float
        Only the levels that are in this layer are used, and the modes then
        have no flux through its limits, which focuses on a part of the water column.
    n2min: float
        Minimal stratification in s-2, which is used where it is smaller, like in unstable
        or mixed layers.
    gravity: float
        Gravity acceleration, which is used by the free surface

    Return
    ------
    array(..., nmodes, nz)
        The modes, normalized so that the mean of their squares over the water column
        is one, and positive at the surface. They are missing where `dz` is.
    array(..., nmodes)
        The speeds of the modes in m s-1, which are missing for the barotropic mode of the
        rigid lid, that is infinitely fast.
    array(..., nz)
        Thickness of the layers around the levels, with half layers at the ends of the water
        column, which are the weights of the integrals.

    See also
    --------
    project
    reconstruct
    """
    nmodes = int(nmodes)
    if nmodes < 1:
        raise ValueError("nmodes must be a positive integer")
    n2 = np.asarray(n2, dtype="d")
    z = np.asarray(z, dtype="d")
    if n2.ndim == 0 or z.ndim == 0 or n2.shape[-1] != z.shape[-1]:
        raise ValueError(
            "n2 and z must have the same last dimension, which is the vertical one: shape mismatch"
        )
    return _vertical_modes_(
        n2,
        z,
        np.arange(nmodes),
        int(free_surface),
        float(zmin),
        float(zmax),
        float(n2min),
        float(gravity),
    )


@numba.guvectorize(
    [(numba.float64[:], numba.float64[:, :], numba.float64[:], numba.float64[:])],
    "(nz),(nm,nz),(nz)->(nm)",
    nopython=True,
    cache=NOT_CI,
)
def _project_(field, modes, dz, amps):
    nm, nz = modes.shape
    amps[:] = np.nan
    total = 0.0
    for k in range(nz):
        if np.isfinite(dz[k]):
            if not np.isfinite(field[k]):
                return
            total += dz[k]
    if total == 0.0:
        return
    for n in range(nm):
        value = 0.0
        ok = False
        for k in range(nz):
            if np.isfinite(dz[k]) and np.isfinite(modes[n, k]):
                value += field[k] * modes[n, k] * dz[k]
                ok = True
        if ok:
            amps[n] = value / total


def project(field, modes, dz):
    """Amplitudes of the vertical modes of a profile

    They are the weighted means of the profile times the modes, so that the amplitude of
    the barotropic mode is the vertical mean of the profile.

    Parameters
    ----------
    field: array_like(..., nz)
        Profile. The levels where `dz` is missing are not used, and all the amplitudes
        are missing if the profile is missing at one of the others.
    modes: array_like(..., nmodes, nz)
        Modes as returned by :func:`vertical_modes`
    dz: array_like(..., nz)
        Weights as returned by :func:`vertical_modes`

    Return
    ------
    array(..., nmodes)
        The amplitudes, that are missing for the modes that were not computed.

    See also
    --------
    vertical_modes
    reconstruct
    """
    return _project_(
        np.asarray(field, dtype="d"), np.asarray(modes, dtype="d"), np.asarray(dz, dtype="d")
    )


@numba.guvectorize(
    [(numba.float64[:], numba.float64[:, :], numba.float64[:])],
    "(nm),(nm,nz)->(nz)",
    nopython=True,
    cache=NOT_CI,
)
def _reconstruct_(amps, modes, out):
    nm, nz = modes.shape
    for k in range(nz):
        value = 0.0
        ok = False
        for n in range(nm):
            if np.isfinite(modes[n, k]):
                value += amps[n] * modes[n, k]
                ok = True
        out[k] = value if ok else np.nan


def reconstruct(amps, modes):
    """Profile from the amplitudes of its vertical modes

    Parameters
    ----------
    amps: array_like(..., nmodes)
        Amplitudes of the modes, that are as many as the modes are.
        Select the first modes of both to truncate the sum, or set some amplitudes to zero.
    modes: array_like(..., nmodes, nz)
        Modes as returned by :func:`vertical_modes`

    Return
    ------
    array(..., nz)
        The profile, which is missing where the modes are.

    See also
    --------
    vertical_modes
    project
    """
    return _reconstruct_(np.asarray(amps, dtype="d"), np.asarray(modes, dtype="d"))
