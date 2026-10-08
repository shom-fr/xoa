#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geographic utilities
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

import math
import os

import numba
import numpy as np

from .num import EPSILON

NOT_CI = os.environ.get("CI", "false") == "false"

# Constants
EARTH_RADIUS = 6371.0e3  # Earth radius in meters
DEG2RAD = np.pi / 180.0


@numba.vectorize(cache=True)
def haversine(lon0, lat0, lon1, lat1):
    """Haversine distance between two points on a **unit sphere**

    Parameters
    ----------
    lon0: float, array_like
        Longitude of the first point(s)
    lat0: float, array_like
        Latitude of the first point(s)
    lon1: float, array_like
        Longitude of the second point(s)
    lat1: float, array_like
        Latitude of the second point(s)

    Return
    ------
    float, array_like
        Distance(s)
    """
    deg2rad = math.pi / 180.0
    dist = math.sin(deg2rad * (lat0 - lat1) * 0.5) ** 2
    dist += (
        math.cos(deg2rad * lat0)
        * math.cos(deg2rad * lat1)
        * math.sin(deg2rad * (lon0 - lon1) * 0.5) ** 2
    )
    dist = 2.0 * math.asin(math.sqrt(dist))
    return dist


@numba.vectorize
def bearing(lon0, lat0, lon1, lat1):
    """Compute the bearing angle (forward azimuth)

    Parameters
    ----------
    lon0: float, array_like
        Longitude of the first point(s)
    lat0: float, array_like
        Latitude of the first point(s)
    lon1: float, array_like
        Longitude of the second point(s)
    lat1: float, array_like
        Latitude of the second point(s)

    Return
    ------
    float, array_like
        Angle(s)
    """
    deg2rad = math.pi / 180.0
    a = math.atan2(
        math.cos(deg2rad * lat0) * math.sin(deg2rad * lat1)
        - math.sin(deg2rad * lat0) * math.cos(deg2rad * lat1) * math.cos(deg2rad * (lon1 - lon0)),
        math.sin(deg2rad * (lon1 - lon0)) * math.cos(deg2rad * lat1),
    )
    return a * 180 / math.pi


@numba.njit(cache=NOT_CI, nogil=True)
def normalize_longitude(lon):
    """Normalize longitude to [-180, 180] range."""
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


@numba.vectorize(
    [
        numba.int32(numba.int32, numba.int32),
        numba.int64(numba.int64, numba.int64),
        numba.float32(numba.float32, numba.float32),
        numba.float64(numba.float64, numba.float64),
    ],
    cache=NOT_CI,
)
def diff_lon(lon0, lon1):
    dlon = lon1 - lon0
    if dlon > 0:
        while dlon > 180:
            dlon -= 360
    else:
        while dlon < -180:
            dlon += 360
    return dlon


@numba.njit(cache=NOT_CI, nogil=True)
def closest_point_range(lons, lats, target_lon, target_lat, imin, ni, istep, jmin, nj, jstep):
    """Find indices of closest point on 2D lon/lat grid

    Parameters
    ----------
    lons: array_like(nyi, nxi)
        Grid longitudes in degrees
    lats: array_like(nyi, nxi)
        Grid latitudes in degrees
    target_lon:
        Point longtitude
    target_lat:
        Point latitude

    Return
    ------
    int: i
        index along second dim
    int: j
        Index along first dim
    """
    ny, nx = lons.shape
    mindist = np.inf
    i = -1
    j = -1
    for jt in range(jmin, ny, jstep):
        for it in range(imin, nx, istep):
            dist = haversine(target_lon, target_lat, lons[jt, it], lats[jt, it])
            if dist <= mindist:
                i = it
                j = jt
                mindist = dist
    return i, j, mindist


@numba.njit(cache=NOT_CI, nogil=True)
def closest_point_fast(lons, lats, target_lon, target_lat):
    """
    Find the closest point of a 2D grid, without scanning large grids

    Small grids are fully scanned. For larger ones, the closest point of a subsampled grid
    is refined by walking downhill from it, to the closest point of its neighbours, until
    no neighbour is closer. Unlike a search in a window of fixed size, this is not limited
    by the distance between the subsampled point and the true closest point, which is
    large when cells are not square.

    Returns
    -------
    int
        I index
    int
        J index
    """
    ny, nx = lons.shape

    # For very large grids, use hierarchical search
    if ny > 50 or nx > 50:
        # Coarse search with subsampling
        step_y = max(1, ny // 20)
        step_x = max(1, nx // 20)

        best_i, best_j, mindist = closest_point_range(
            lons, lats, target_lon, target_lat, 0, nx, step_x, 0, ny, step_y
        )
        if best_i < 0:  # no valid point in the subsampled grid
            best_i, best_j, mindist = closest_point_range(
                lons, lats, target_lon, target_lat, 0, nx, 1, 0, ny, 1
            )
            return best_i, best_j

        # Walk downhill to the closest point
        invalid = False
        for _ in range(ny + nx):
            next_i = best_i
            next_j = best_j
            for j in range(max(best_j - 1, 0), min(best_j + 2, ny)):
                for i in range(max(best_i - 1, 0), min(best_i + 2, nx)):
                    dist = haversine(target_lon, target_lat, lons[j, i], lats[j, i])
                    if np.isnan(dist):
                        invalid = True
                    elif dist < mindist:
                        mindist = dist
                        next_i = i
                        next_j = j
            if next_i == best_i and next_j == best_j:
                break
            best_i = next_i
            best_j = next_j

        # Invalid points, like masked ones, make local minima: scan everything
        if invalid:
            best_i, best_j, mindist = closest_point_range(
                lons, lats, target_lon, target_lat, 0, nx, 1, 0, ny, 1
            )

        return best_i, best_j
    else:
        # For small grids, do full search
        best_i, best_j, mindist = closest_point_range(
            lons, lats, target_lon, target_lat, 0, nx, 1, 0, ny, 1
        )

        return best_i, best_j


@numba.njit(cache=NOT_CI, nogil=True)
def relative_cell_coords(x1, x2, x3, x4, y1, y2, y3, y4, x, y):
    """
    Compute relative cell coordinates for counter-clockwise vertex ordering.

    This function computes the relative coordinates (p, q) of a point (x, y) within
    a quadrilateral cell defined by four vertices in counter-clockwise order.

    Parameters
    ----------
    x1, y1 : float
        Coordinates of the first vertex (bottom-left)
    x2, y2 : float
        Coordinates of the second vertex (bottom-right)
    x3, y3 : float
        Coordinates of the third vertex (top-right)
    x4, y4 : float
        Coordinates of the fourth vertex (top-left)
    x, y : float
        Coordinates of the point to locate within the cell

    Returns
    -------
    p : float
        Relative coordinate in the first parametric direction [0, 1]
        p=0 corresponds to the left edge, p=1 to the right edge
    q : float
        Relative coordinate in the second parametric direction [0, 1]
        q=0 corresponds to the bottom edge, q=1 to the top edge

    Notes
    -----
    The vertices must be ordered counter-clockwise:

    4 --- 3
    |     |
    1 --- 2

    If the point is outside the cell, returns (-1.0, -1.0).

    The bilinear transformation is:
    x(p,q) = (1-p)(1-q)*x1 + p(1-q)*x2 + p*q*x3 + (1-p)*q*x4
    y(p,q) = (1-p)(1-q)*y1 + p(1-q)*y2 + p*q*y3 + (1-p)*q*y4

    Examples
    --------
    >>> # Unit square with counter-clockwise vertices
    >>> p, q = bilinear_cell_coords(0.0, 1.0, 1.0, 0.0,  # x coords
    ...                            0.0, 0.0, 1.0, 1.0,  # y coords
    ...                            0.5, 0.5)             # target point
    >>> print(f"p={p:.3f}, q={q:.3f}")
    p=0.500, q=0.500
    """
    # FIXME: wrap lon
    # Remapped coordinates for the algorithm
    xx1, yy1 = x1, y1  # bottom-left (same)
    xx2, yy2 = x4, y4  # top-left (was input 4)
    xx3, yy3 = x3, y3  # top-right (same)
    xx4, yy4 = x2, y2  # bottom-right (was input 2)

    # Coefficients for the bilinear transformation
    a = xx4 - xx1
    b = xx2 - xx1
    c = xx3 - xx4 - xx2 + xx1
    d = yy4 - yy1
    e = yy2 - yy1
    f = yy3 - yy4 - yy2 + yy1

    # Solve quadratic equation A*p**2 + B*p + C = 0
    yy = y - yy1
    xx = x - xx1
    AA = c * d - a * f
    BB = -c * yy + b * d + xx * f - a * e
    CC = -yy * b + e * xx

    if abs(AA) < EPSILON:
        # Linear case
        if abs(BB) < EPSILON:
            # Degenerate case
            p1 = 0.5
            p2 = 0.5
        else:
            p1 = -CC / BB
            p2 = p1
    else:
        # Quadratic case
        DD = BB * BB - 4.0 * AA * CC
        if DD < 0.0:
            # No real solution - point is outside
            return -1.0, -1.0
        sDD = math.sqrt(DD)
        p1 = (-BB - sDD) / (2.0 * AA)
        p2 = (-BB + sDD) / (2.0 * AA)

    # Get q from p for first solution
    if abs(b + c * p1) > EPSILON:
        q1 = (xx - a * p1) / (b + c * p1)
    elif abs(e + f * p1) > EPSILON:
        q1 = (yy - d * p1) / (e + f * p1)
    else:
        q1 = 0.5  # fallback

    # Check if first solution is valid (inside unit square)
    if 0.0 <= p1 <= 1.0 and 0.0 <= q1 <= 1.0:
        p, q = p1, q1
    else:
        # Try second solution
        if abs(b + c * p2) > EPSILON:
            q2 = (xx - a * p2) / (b + c * p2)
        elif abs(e + f * p2) > EPSILON:
            q2 = (yy - d * p2) / (e + f * p2)
        else:
            q2 = 0.5  # fallback

        if 0.0 <= p2 <= 1.0 and 0.0 <= q2 <= 1.0:
            p, q = p2, q2
        else:
            # Neither solution is valid - check with tolerance
            if -EPSILON <= p1 <= 1.0 + EPSILON and -EPSILON <= q1 <= 1.0 + EPSILON:
                p, q = p1, q1
            elif -EPSILON <= p2 <= 1.0 + EPSILON and -EPSILON <= q2 <= 1.0 + EPSILON:
                p, q = p2, q2
            else:
                # Point is outside the cell
                return -1.0, -1.0

    # Final check with tolerance
    if p < -EPSILON or q < -EPSILON or p > 1.0 + EPSILON or q > 1.0 + EPSILON:
        return -1.0, -1.0

    return p, q
