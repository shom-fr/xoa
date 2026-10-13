"""
Low level kriging

The numerical inputs and outputs of all these routines are of scalar
or numpy.ndarray type.

Distances are great circle distances on a sphere of a given radius,
which sets their units, and positions are longitudes and latitudes in degrees.
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

from .geo import haversine

NOT_CI = os.environ.get("CI", "false") == "false"

#: Variogram model codes
LINEAR = 0
EXPONENTIAL = 1
GAUSSIAN = 2
SPHERICAL = 3


def _variogram_(h, mtype, n, s, r):
    if r <= 0.0:
        return n + (s - n) * (1.0 if h > 0.0 else 0.0)
    u = h / r
    if mtype == LINEAR:
        return n + (s - n) * (u if u <= 1.0 else 1.0)
    if mtype == EXPONENTIAL:
        return n + (s - n) * (1.0 - np.exp(-3.0 * u))
    if mtype == GAUSSIAN:
        return n + (s - n) * (1.0 - np.exp(-3.0 * u * u))
    if u <= 1.0:  # spherical
        return n + (s - n) * (1.5 * u - 0.5 * u * u * u)
    return s


_variogram_scalar_ = numba.njit(cache=NOT_CI)(_variogram_)

#: Variogram model as a function of the distance(s) `h`, with the model code `mtype`,
#: the nugget `n`, the sill `s` and the range `r`
variogram = numba.vectorize(cache=NOT_CI)(_variogram_)


@numba.njit(cache=NOT_CI)
def _krig_point_(
    io,
    lons,
    lats,
    values,
    ineigh,
    dneigh,
    mtype,
    n,
    s,
    r,
    nmin,
    ordinary,
    exact,
    mean,
    radius,
    zo,
    vo,
):
    """Krig a single output point and store the results in `zo` and `vo`"""
    nz = values.shape[0]
    k = ineigh.shape[1]
    plus1 = 1 if ordinary else 0
    eps = 1e-9 * radius

    # Valid neighbors
    m = 0
    for j in range(k):
        if ineigh[io, j] >= 0:
            m += 1
    if m < nmin or m == 0:
        return
    idx = np.empty(m, np.int64)
    dist = np.empty(m)
    m = 0
    for j in range(k):
        if ineigh[io, j] >= 0:
            idx[m] = ineigh[io, j]
            dist[m] = dneigh[io, j]
            m += 1

    # System: variogram for ordinary kriging, covariance for simple kriging
    size = m + plus1
    A = np.zeros((size, size))
    b = np.zeros(size)
    for i in range(m):
        for j in range(i + 1):
            if i == j:
                h = 0.0
            else:
                h = radius * haversine(lons[idx[i]], lats[idx[i]], lons[idx[j]], lats[idx[j]])
            g = _variogram_scalar_(h, mtype, n, s, r)
            if exact and h <= eps:
                g = 0.0
            A[i, j] = g if ordinary else s - g
            A[j, i] = A[i, j]
        g = _variogram_scalar_(dist[i], mtype, n, s, r)
        if exact and dist[i] <= eps:
            g = 0.0
        b[i] = g if ordinary else s - g
    if ordinary:
        for i in range(m):
            A[i, m] = 1.0
            A[m, i] = 1.0
        b[m] = 1.0

    # Weights, with a small regularization if singular (duplicate points)
    try:
        w = np.linalg.solve(A, b)
    except Exception:
        for i in range(m):
            A[i, i] += 1e-8 * s
        try:
            w = np.linalg.solve(A, b)
        except Exception:
            return

    # Estimation and variance
    for iz in range(nz):
        z = 0.0
        for i in range(m):
            z += w[i] * (values[iz, idx[i]] - (0.0 if ordinary else mean))
        zo[iz, io] = z + (0.0 if ordinary else mean)
    v = 0.0
    if ordinary:
        for i in range(size):
            v += w[i] * b[i]
    else:
        v = s
        for i in range(m):
            v -= w[i] * b[i]
    vo[io] = v


@numba.njit(cache=NOT_CI, parallel=True)
def krig_local(
    lons,
    lats,
    values,
    lono,
    lato,
    ineigh,
    dneigh,
    mtype,
    n,
    s,
    r,
    nmin,
    ordinary,
    exact,
    mean,
    radius,
):
    """Krig at output positions using only their nearest input points

    Parameters
    ----------
    lons: array(np)
        Longitudes of input points
    lats: array(np)
        Latitudes of input points
    values: array(nz, np)
        Values at input points, without missing values
    lono: array(no)
        Longitudes of output points
    lato: array(no)
        Latitudes of output points
    ineigh: array(no, k)
        Indices of the neighbors in the input points, -1 for a missing neighbor
    dneigh: array(no, k)
        Distances to the neighbors
    mtype: int
        Variogram model code: `LINEAR`, `EXPONENTIAL`, `GAUSSIAN` or `SPHERICAL`
    n: float
        Nugget
    s: float
        Sill
    r: float
        Range
    nmin: int
        Minimal number of neighbors, below which the result is nan
    ordinary: bool
        Ordinary kriging, else simple kriging
    exact: bool
        The variogram is exactly zero at a zero distance
    mean: float
        Mean for the simple kriging
    radius: float
        Radius of the sphere, which sets the units of `dneigh`

    Return
    ------
    array(nz, no)
        Estimations, nan where there are not enough neighbors
    array(no)
        Kriging variances
    """
    nz = values.shape[0]
    no = ineigh.shape[0]
    zo = np.full((nz, no), np.nan)
    vo = np.full(no, np.nan)
    for io in numba.prange(no):
        _krig_point_(
            io,
            lons,
            lats,
            values,
            ineigh,
            dneigh,
            mtype,
            n,
            s,
            r,
            nmin,
            ordinary,
            exact,
            mean,
            radius,
            zo,
            vo,
        )
    return zo, vo
