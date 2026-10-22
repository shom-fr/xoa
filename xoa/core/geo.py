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
from scipy.spatial import cKDTree

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

    The angle is in degrees, in ]-180, 180], counter-clockwise from the east like the angle
    of a vector ``(u, v)``: east is 0, north is 90, west is 180 and south is -90.

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


@numba.njit(cache=NOT_CI)
def lonlat2xyz(lon, lat):
    """Convert longitudes and latitudes to cartesian coordinates on the unit sphere

    Parameters
    ----------
    lon: array_like(n)
        Longitudes in degrees
    lat: array_like(n)
        Latitudes in degrees

    Return
    ------
    array_like(n, 3)
    """
    n = lon.size
    xyz = np.empty((n, 3))
    for i in range(n):
        lo = DEG2RAD * lon[i]
        la = DEG2RAD * lat[i]
        coslat = math.cos(la)
        xyz[i, 0] = coslat * math.cos(lo)
        xyz[i, 1] = coslat * math.sin(lo)
        xyz[i, 2] = math.sin(la)
    return xyz


@numba.vectorize(cache=NOT_CI)
def dist2chord(dist, radius):
    """Convert great circle distances to straight line distances through the sphere"""
    return 2.0 * math.sin(0.5 * min(dist / radius, math.pi))


@numba.vectorize(cache=NOT_CI)
def chord2dist(chord, radius):
    """Convert straight line distances through the sphere to great circle distances"""
    if math.isinf(chord):
        return chord
    return 2.0 * radius * math.asin(min(0.5 * chord, 1.0))


class SphericalKDTree:
    """A KD-tree of points on a sphere, defined by their longitudes and latitudes

    The points are converted to cartesian coordinates, which is exact since the
    straight line distance through the sphere is an increasing function of the
    great circle distance. The neighbors are thus the same, and the great circle
    distances are recovered from the straight line ones.
    There is no problem at the poles or across the dateline.

    Parameters
    ----------
    lon: array_like
        Longitudes in degrees. Points with a nan longitude or latitude are ignored.
    lat: array_like
        Latitudes in degrees
    radius: float
        Radius of the sphere, which also sets the unit of distances.
        It defaults to the earth radius in meters.
    leafsize: int
        Number of points at which the tree switches to a brute force search

    Note
    ----
    The indices refer to the flattened input arrays, and are -1 for missing neighbors.
    """

    def __init__(self, lon, lat, radius=EARTH_RADIUS, leafsize=16):
        lon = np.asarray(lon, dtype="d").ravel()
        lat = np.asarray(lat, dtype="d").ravel()
        valid = np.flatnonzero(np.isfinite(lon) & np.isfinite(lat))
        if valid.size == 0:
            raise ValueError("No valid point to build the tree")
        self.radius = radius
        self.size = lon.size
        self._lon = lon[valid]
        self._lat = lat[valid]
        # With a trailing -1 to translate missing neighbors
        self._index = np.append(valid, -1)
        self.tree = cKDTree(lonlat2xyz(self._lon, self._lat), leafsize=leafsize)

    def _get_targets_(self, lon, lat):
        lon = np.asarray(lon, dtype="d").ravel()
        lat = np.asarray(lat, dtype="d").ravel()
        valid = np.isfinite(lon) & np.isfinite(lat)
        return lonlat2xyz(np.where(valid, lon, 0.0), np.where(valid, lat, 0.0)), valid

    def query(self, lon, lat, k=1, max_dist=None, workers=1):
        """Find the nearest neighbors of points

        Parameters
        ----------
        lon: array_like(m)
        lat: array_like(m)
        k: int
            Number of neighbors
        max_dist: float, None
            Maximal distance
        workers: int
            Number of threads, -1 for all

        Return
        ------
        array_like(m) or array_like(m, k)
            Great circle distances, which are inf for missing neighbors
        array_like(m) or array_like(m, k)
            Indices of neighbors
        """
        xyz, valid = self._get_targets_(lon, lat)
        bound = np.inf if max_dist is None else float(dist2chord(max_dist, self.radius))
        chord, idx = self.tree.query(xyz, k=k, distance_upper_bound=bound, workers=workers)
        chord[~valid] = np.inf
        idx[~valid] = len(self._index) - 1
        return chord2dist(chord, self.radius), self._index[idx]

    def query_ball_point(self, lon, lat, r, return_dist=False, workers=1):
        """Find the neighbors of points within a distance

        Parameters
        ----------
        lon: array_like(m)
        lat: array_like(m)
        r: float
            Maximal distance
        return_dist: bool
            Also return the distances
        workers: int
            Number of threads, -1 for all

        Return
        ------
        list
            One array of indices per point, sorted by increasing indices
        list
            One array of great circle distances per point, if `return_dist`
        """
        xyz, valid = self._get_targets_(lon, lat)
        found = self.tree.query_ball_point(
            xyz, float(dist2chord(r, self.radius)), workers=workers, return_sorted=True
        )
        empty = np.array([], dtype="i8")
        found = [f if ok else empty for f, ok in zip(found, valid)]
        if not return_dist:
            return [self._index[f] for f in found]
        lon, lat = np.ravel(lon), np.ravel(lat)
        dists = [
            haversine(lon[i], lat[i], self._lon[f], self._lat[f]) * self.radius
            for i, f in enumerate(found)
        ]
        return [self._index[f] for f in found], dists


# %% Geographic selection


@numba.vectorize(cache=NOT_CI)
def lon_in_range(lon, lonmin, lonmax):
    """Tell if longitudes are in a range, which may cross the dateline

    The range goes eastward from `lonmin` to `lonmax`, like ``170, 190``, and covers
    all longitudes when it spans 360° or more. Longitudes may be in any 360° range.
    """
    width = lonmax - lonmin
    if width >= 360.0:
        return lon == lon  # not nan
    return (lon - lonmin) % 360.0 <= width


@numba.njit(cache=NOT_CI, nogil=True)
def bbox_window(lon, lat, lonmin, lonmax, latmin, latmax):
    """Bounding window of the points of a 2D grid that are in a lon/lat box

    Return
    ------
    int, int, int, int
        ``j0, j1, i0, i1`` such that the window is ``[j0:j1, i0:i1]``, empty if ``j1 <= j0``
    """
    ny, nx = lon.shape
    j0 = ny
    j1 = -1
    i0 = nx
    i1 = -1
    for j in range(ny):
        for i in range(nx):
            if latmin <= lat[j, i] <= latmax and lon_in_range(lon[j, i], lonmin, lonmax):
                j0 = min(j0, j)
                j1 = max(j1, j)
                i0 = min(i0, i)
                i1 = max(i1, i)
    return j0, j1 + 1, i0, i1 + 1


@numba.njit(cache=NOT_CI, nogil=True)
def max_true_rectangle(mask):
    """Largest rectangle of a 2D boolean array that contains only true values

    Return
    ------
    int, int, int, int
        ``j0, j1, i0, i1`` such that the rectangle is ``[j0:j1, i0:i1]``,
        all zeros if there is no true value
    """
    ny, nx = mask.shape
    heights = np.zeros(nx, dtype=np.int64)
    stack = np.empty(nx + 1, dtype=np.int64)
    best = 0
    j0 = j1 = i0 = i1 = 0
    for j in range(ny):
        for i in range(nx):
            heights[i] = heights[i] + 1 if mask[j, i] else 0
        # Largest rectangle under the histogram of heights
        top = 0
        for i in range(nx + 1):
            h = heights[i] if i < nx else 0
            while top > 0 and heights[stack[top - 1]] >= h:
                hh = heights[stack[top - 1]]
                top -= 1
                left = stack[top - 1] + 1 if top > 0 else 0
                if hh * (i - left) > best:
                    best = hh * (i - left)
                    j0, j1, i0, i1 = j - hh + 1, j + 1, left, i
            stack[top] = i
            top += 1
    return j0, j1, i0, i1


@numba.njit(cache=NOT_CI, nogil=True)
def _argnearest_(values, targets, circular):
    """Index of the closest value of each target, -1 for nan targets"""
    out = np.full(targets.size, -1, dtype=np.int64)
    for k in range(targets.size):
        best = np.inf
        for i in range(values.size):
            d = abs(diff_lon(targets[k], values[i])) if circular else abs(targets[k] - values[i])
            if d < best:
                best = d
                out[k] = i
    return out


@numba.njit(cache=NOT_CI, nogil=True)
def cell_half_extents(lon, lat, elon, elat):
    """Half extents in longitude and latitude of the cells of a 2D grid

    The extent of a cell is that of the envelope of its corners, that are given
    by the edges as returned by :func:`xoa.core.grid.edges`. It is twice the distance
    from the center to the envelope, which is large for rotated cells.

    Parameters
    ----------
    lon, lat: array_like(ny, nx)
        Cell centers
    elon, elat: array_like(ny+1, nx+1)
        Cell edges, with longitudes made continuous across the dateline

    Return
    ------
    array_like(ny, nx)
        Half extent in longitude
    array_like(ny, nx)
        Half extent in latitude
    """
    ny, nx = lon.shape
    hlon = np.zeros((ny, nx))
    hlat = np.zeros((ny, nx))
    for j in range(ny):
        for i in range(nx):
            for dj in range(2):
                for di in range(2):
                    dx = abs(diff_lon(lon[j, i], elon[j + dj, i + di]))
                    dy = abs(elat[j + dj, i + di] - lat[j, i])
                    if dx > hlon[j, i]:
                        hlon[j, i] = dx
                    if dy > hlat[j, i]:
                        hlat[j, i] = dy
    return hlon, hlat


@numba.njit(cache=NOT_CI, nogil=True)
def polygon_distance(poly, lon, lat):
    """Approximate distance in radians of points to the boundary of a polygon

    The distance is computed in a plane tangent to each point, with the longitudes scaled
    by the cosine of the latitude of the point.

    Parameters
    ----------
    poly: array_like(n, 2)
        Vertices as (lon, lat) pairs in degrees
    lon, lat: array_like(m)
        Points
    """
    n = poly.shape[0]
    out = np.empty(lon.size)
    for k in range(lon.size):
        c = math.cos(DEG2RAD * lat[k])
        best = np.inf
        for s in range(n):
            ax = diff_lon(lon[k], poly[s, 0]) * c
            ay = poly[s, 1] - lat[k]
            bx = diff_lon(lon[k], poly[(s + 1) % n, 0]) * c
            by = poly[(s + 1) % n, 1] - lat[k]
            dx = bx - ax
            dy = by - ay
            dd = dx * dx + dy * dy
            t = 0.0 if dd == 0.0 else min(max(-(ax * dx + ay * dy) / dd, 0.0), 1.0)
            best = min(best, math.hypot(ax + t * dx, ay + t * dy))
        out[k] = best * DEG2RAD
    return out


def _margin_lonlat_(margin, lat):
    """Margins along lon and lat in degrees, from a margin that may be an arc along lat"""
    mlon, mlat = margin
    if mlon is None:  # an arc
        mlon = mlat / np.maximum(np.cos(DEG2RAD * np.asarray(lat)), 0.01)
    return mlon, mlat


def _margin_arc_(margin, lat):
    """Isotropic margin in radians, from a margin that may be along lon and lat"""
    mlon, mlat = margin
    if mlon is None:  # an arc
        return DEG2RAD * mlat
    return DEG2RAD * np.maximum(mlon * np.cos(DEG2RAD * np.asarray(lat)), mlat)


class Box:
    """A box of longitudes and latitudes, where the longitude range may cross the dateline

    Parameters
    ----------
    lon: (float, float), None
        Eastward range of longitudes, like ``(170, -170)`` across the dateline,
        which is global with ``None``
    lat: (float, float), None
        Range of latitudes, which is global with ``None``
    """

    #: A box is exactly represented by lon and lat ranges on a rectangular grid
    exact_on_rect = True

    def __init__(self, lon=None, lat=None):
        self.lon = (0.0, 360.0) if lon is None else tuple(float(v) for v in lon)
        if self.lon[1] < self.lon[0]:  # across the dateline
            self.lon = (self.lon[0], self.lon[1] + 360.0)
        self.lat = (-90.0, 90.0) if lat is None else tuple(float(v) for v in lat)

    def bounds(self):
        """Get ``lonmin, lonmax, latmin, latmax``"""
        return (*self.lon, *self.lat)

    def contains(self, lon, lat, margin=None):
        """Boolean array of the points that are in the box, grown by an optional margin"""
        mlon, mlat = (0.0, 0.0) if margin is None else _margin_lonlat_(margin, lat)
        return (
            lon_in_range(lon, self.lon[0] - mlon, self.lon[1] + mlon)
            & (lat >= self.lat[0] - mlat)
            & (lat <= self.lat[1] + mlat)
        )


class Circle:
    """A circle of great circle distance around a point

    Parameters
    ----------
    lon, lat: float
        Center
    dist: float
        Distance, in the units of `radius`
    radius: float
        Radius of the sphere, which defaults to the earth radius in meters
    """

    exact_on_rect = False
    isotropic = True  # the margin is an angular distance

    def __init__(self, lon, lat, dist, radius=EARTH_RADIUS):
        self.lon = float(lon)
        self.lat = float(lat)
        self.angle = float(dist) / radius  # angular distance in radians

    def bounds(self, grow=0.0):
        """Get ``lonmin, lonmax, latmin, latmax``, for a radius grown by an angle in radians"""
        angle = self.angle + grow
        dlat = angle / DEG2RAD
        latmin, latmax = self.lat - dlat, self.lat + dlat
        if latmin <= -90.0 or latmax >= 90.0:  # includes a pole
            return 0.0, 360.0, max(latmin, -90.0), min(latmax, 90.0)
        dlon = math.asin(min(math.sin(angle) / math.cos(DEG2RAD * self.lat), 1.0)) / DEG2RAD
        return self.lon - dlon, self.lon + dlon, latmin, latmax

    def contains(self, lon, lat, margin=None):
        """Boolean array of the points that are in the circle, grown by an optional margin"""
        angle = self.angle if margin is None else self.angle + _margin_arc_(margin, lat)
        return haversine(self.lon, self.lat, lon, lat) <= angle


class Polygon:
    """A polygon, which may be concave and cross the dateline

    Parameters
    ----------
    poly: array_like(n, 2)
        Vertices as (lon, lat) pairs in degrees. It must not span more than 180° of longitude.
    """

    exact_on_rect = False
    isotropic = False

    def __init__(self, poly):
        from .poly import unwrap_longitudes

        self.poly = np.array(poly, dtype="d")
        self.poly[:, 0] = unwrap_longitudes(self.poly[:, 0], self.poly[0, 0])

    def bounds(self):
        """Get ``lonmin, lonmax, latmin, latmax``"""
        lon, lat = self.poly[:, 0], self.poly[:, 1]
        return lon.min(), lon.max(), lat.min(), lat.max()

    def contains(self, lon, lat, margin=None):
        """Boolean array of the points that are in the polygon, grown by an optional margin"""
        from .mask import polygon_mask

        lon, lat = np.broadcast_arrays(lon, lat)
        inside = polygon_mask(self.poly, lon.reshape(1, -1), lat.reshape(1, -1)).reshape(lon.shape)
        if margin is not None:
            arc = np.broadcast_to(_margin_arc_(margin, lat), lon.shape)
            out = ~inside
            near = polygon_distance(self.poly, lon[out], lat[out]) <= arc[out]
            inside[out] = near
        return inside


def _take_margin_(margin, index_lon, index_lat):
    """Margin on the points of a window, whose components may be arrays, scalars or None"""
    if margin is None:
        return None
    return tuple(
        m if m is None or np.ndim(m) == 0 else m[i] for m, i in zip(margin, (index_lon, index_lat))
    )


def _as_indexer_(idx):
    """Convert increasing contiguous indices to a slice"""
    if idx.size and np.all(np.diff(idx) == 1):
        return slice(int(idx[0]), int(idx[-1]) + 1)
    return idx


def _true_extent_(mask):
    """Slices of the bounding window of the true values of a 2D mask"""
    out = []
    for axis in (1, 0):
        idx = np.flatnonzero(mask.any(axis=axis))
        out.append(slice(int(idx[0]), int(idx[-1]) + 1))
    return out


class GeoSelector:
    """Find the indices of the points of a grid or cloud that are near a point or in a shape

    The result is made of indexers along the axes of the coordinates, that are slices
    where possible, and an optional mask.

    Parameters
    ----------
    lon: array_like
        Longitudes in degrees, 1D or 2D
    lat: array_like
        Latitudes in degrees, 1D or 2D
    kind: {"rect", "curv", "cloud"}, None
        Kind of horizontal coordinates:

        - ``"rect"``: rectangular grid, with 1D axes ``lon`` and ``lat`` that have
          different dimensions. The indexers are along ``(lon, lat)``.
        - ``"curv"``: curvilinear grid, with 2D ``lon`` and ``lat``.
          The indexers are along the two axes.
        - ``"cloud"``: points with 1D ``lon`` and ``lat`` that have the same dimension.
          There is a single indexer.

        It is inferred from the shapes when not provided, two 1D arrays of the same
        size being a cloud.
    radius: float
        Radius of the sphere, which also sets the unit of distances.
    leafsize: int
        Leaf size of the KD-tree, that is built when needed only.
    """

    def __init__(self, lon, lat, kind=None, radius=EARTH_RADIUS, leafsize=16):
        self.lon = np.ascontiguousarray(lon, dtype="d")
        self.lat = np.ascontiguousarray(lat, dtype="d")
        if kind is None:
            if self.lon.ndim == 2:
                kind = "curv"
            else:
                kind = "cloud" if self.lon.size == self.lat.size else "rect"
        ndim, same = {"rect": (1, False), "curv": (2, True), "cloud": (1, True)}.get(
            kind, (None, None)
        )
        if ndim is None:
            raise ValueError(f"Invalid kind: {kind!r}. Use 'rect', 'curv' or 'cloud'")
        if (
            self.lon.ndim != ndim
            or self.lat.ndim != ndim
            or (same and self.lon.shape != self.lat.shape)
        ):
            raise ValueError(f"Invalid shapes of lon and lat for a {kind!r} kind")
        self.kind = kind
        self.radius = radius
        self.leafsize = leafsize
        self._tree = None
        self._half_extents = None

    @property
    def half_extents(self):
        """Half extents of the cells along lon and lat, that are local to each point

        The cells are delimited by the edges that :func:`xoa.core.grid.edges` computes
        from the centers, and their extent is that of the envelope of their corners.
        The arrays are 1D along the axes of a rectangular grid,
        and 2D for a curvilinear grid.
        """
        if self.kind == "cloud":
            raise ValueError("Grid points are needed to get cell extents, not a cloud of points")
        if self._half_extents is None:
            from .grid import edges

            if self.kind == "rect":
                hlon, hlat = (
                    np.maximum(abs(diff_lon(c, e[:-1])), abs(diff_lon(c, e[1:])))
                    if lonmode
                    else np.maximum(abs(c - e[:-1]), abs(c - e[1:]))
                    for c, e, lonmode in (
                        (self.lon, edges(self.lon, lon=True), True),
                        (self.lat, edges(self.lat), False),
                    )
                )
            else:
                elon = edges(edges(self.lon, 1, lon=True), 0, lon=True)
                elat = edges(edges(self.lat, 1), 0)
                hlon, hlat = cell_half_extents(self.lon, self.lat, elon, elat)
            self._half_extents = hlon, hlat
        return self._half_extents

    @property
    def tree(self):
        """The :class:`SphericalKDTree` of the points of a curvilinear grid or cloud"""
        if self._tree is None:
            self._tree = SphericalKDTree(self.lon, self.lat, self.radius, self.leafsize)
        return self._tree

    def nearest(self, lon, lat, max_dist=None, workers=1):
        """Find the indices of the points that are the closest to target points

        Parameters
        ----------
        lon, lat: array_like
            Target longitudes and latitudes, scalar or 1D
        max_dist: float, None
            Maximal distance, in the units of the radius
        workers: int
            Number of threads for the KD-tree, -1 for all

        Return
        ------
        tuple
            Arrays of indices along each axis, with a size equal to the number of targets,
            and -1 for targets without neighbor
        array_like
            Great circle distances, inf for targets without neighbor
        """
        lon = np.atleast_1d(np.asarray(lon, dtype="d"))
        lat = np.atleast_1d(np.asarray(lat, dtype="d"))
        if self.kind == "rect":
            idx = (_argnearest_(self.lon, lon, True), _argnearest_(self.lat, lat, False))
            found = (idx[0] >= 0) & (idx[1] >= 0)
            dist = np.full(lon.shape, np.inf)
            dist[found] = (
                haversine(lon[found], lat[found], self.lon[idx[0][found]], self.lat[idx[1][found]])
                * self.radius
            )
        elif self.kind == "curv" and lon.size <= 3 and self._tree is None:
            idx = np.full((2, lon.size), -1)
            dist = np.full(lon.size, np.inf)
            for k in range(lon.size):
                if np.isfinite(lon[k]) and np.isfinite(lat[k]):
                    i, j = closest_point_fast(self.lon, self.lat, lon[k], lat[k])
                    idx[:, k] = j, i
                    dist[k] = self.radius * haversine(
                        lon[k], lat[k], self.lon[j, i], self.lat[j, i]
                    )
            idx = tuple(idx)
        else:
            dist, flat = self.tree.query(lon, lat, max_dist=max_dist, workers=workers)
            miss = flat < 0
            idx = np.unravel_index(np.where(miss, 0, flat), self.lon.shape)
            idx = tuple(np.where(miss, -1, ii) for ii in idx)
        if max_dist is not None:
            far = ~(dist <= max_dist)
            dist = np.where(far, np.inf, dist)
            idx = tuple(np.where(far, -1, ii) for ii in idx)
        return idx, dist

    def _get_margin_(self, pad, pad_units, shape):
        """Margin along lon and lat in degrees, as a pair of scalars or arrays, and its maxima

        An arc margin has a null longitude margin, that depends on the latitude.
        """
        if isinstance(pad, str):
            if pad != "edges":
                raise ValueError(f"Invalid pad: {pad!r}. Use a number or 'edges'")
            pad, pad_units = 0.5, "points"
        if not pad:
            return None, None
        if pad_units == "points":
            hlon, hlat = self.half_extents
            margin = 2 * pad * hlon, 2 * pad * hlat
            return margin, (np.nanmax(margin[0]), np.nanmax(margin[1]))
        if pad_units not in ("dist", "deg"):
            raise ValueError(f"Invalid pad_units: {pad_units!r}. Use 'points', 'dist' or 'deg'")
        mlat = pad if pad_units == "deg" else pad / self.radius / DEG2RAD
        latmax = min(max(abs(shape.bounds()[2]), abs(shape.bounds()[3])) + mlat, 89.0)
        return (None, mlat), (mlat / math.cos(DEG2RAD * latmax), mlat)

    def select(self, shape, fit="outer", pad=0, pad_units="points"):
        """Find the indices of the points that are in a shape

        Parameters
        ----------
        shape: Box, Circle, Polygon
            Any object with a ``bounds`` and a ``contains`` method
        fit: {"outer", "inner"}
            Either the smallest window that contains all the points inside the shape,
            with a mask when some points of the window are outside,
            or the largest window with only points inside, without mask.
        pad: float, "edges"
            Margin around the shape, in `pad_units`. With ``"edges"``, which is ``0.5``
            grid point, points are selected when their cell touches the shape.
        pad_units: {"points", "dist", "deg"}
            Either grid points, with a local estimate of the step given by the cell extent
            (:attr:`half_extents`), a distance in the units of the radius,
            or an arc in degrees. Grid points are not valid for a cloud of points.

        Return
        ------
        tuple
            Indexers along each axis, as slices or arrays of indices
        array_like, None
            Boolean mask of the points of the window that are inside, with one dimension per
            indexer, or None if all of them are inside or if there is no window.
        """
        if fit not in ("outer", "inner"):
            raise ValueError(f"Invalid fit: {fit!r}. Use 'outer' or 'inner'")
        margin, mmax = self._get_margin_(pad, pad_units, shape)
        kind = self.kind
        if mmax is None:
            lonmin, lonmax, latmin, latmax = shape.bounds()
        elif getattr(shape, "isotropic", False):
            lonmin, lonmax, latmin, latmax = shape.bounds(DEG2RAD * max(mmax))
        else:
            lonmin, lonmax, latmin, latmax = shape.bounds()
            lonmin, lonmax = lonmin - mmax[0], lonmax + mmax[0]
            latmin, latmax = latmin - mmax[1], latmax + mmax[1]
        empty = (slice(0, 0), slice(0, 0)), None

        if kind == "cloud":
            idx = np.flatnonzero(shape.contains(self.lon, self.lat, margin))
            return (_as_indexer_(idx),), None

        if kind == "rect":
            if shape.exact_on_rect and margin is not None:  # local margins along each axis
                mlon = margin[0] if margin[0] is not None else mmax[0]
                mlat = margin[1]
                blonmin, blonmax, blatmin, blatmax = shape.bounds()
                ix = np.flatnonzero(lon_in_range(self.lon, blonmin - mlon, blonmax + mlon))
                iy = np.flatnonzero((self.lat >= blatmin - mlat) & (self.lat <= blatmax + mlat))
            else:
                ix = np.flatnonzero(lon_in_range(self.lon, lonmin, lonmax))
                iy = np.flatnonzero((self.lat >= latmin) & (self.lat <= latmax))
            if ix.size and not np.all(np.diff(ix) == 1):  # across the cut, from west to east
                ix = ix[np.argsort((self.lon[ix] - lonmin) % 360.0, kind="stable")]
                if self.lon.size > 1 and diff_lon(self.lon[0], self.lon[1]) < 0:
                    ix = ix[::-1]
            if ix.size == 0 or iy.size == 0:
                return empty
            if shape.exact_on_rect:
                return (_as_indexer_(ix), _as_indexer_(iy)), None
            ix, iy = ix[:, None], iy[None, :]
            mask = shape.contains(self.lon[ix], self.lat[iy], _take_margin_(margin, ix, iy))
            ix, iy = ix[:, 0], iy[0]
        else:
            j0, j1, i0, i1 = bbox_window(self.lon, self.lat, lonmin, lonmax, latmin, latmax)
            if j1 <= j0:
                return empty
            win = slice(j0, j1), slice(i0, i1)
            mask = shape.contains(self.lon[win], self.lat[win], _take_margin_(margin, win, win))

        # Window with a mask
        if not mask.any():
            return empty
        if fit == "inner":
            a0, a1, b0, b1 = max_true_rectangle(mask)
            sa, sb = slice(a0, a1), slice(b0, b1)
            if a1 <= a0:
                return empty
            mask = None
        else:
            sa, sb = _true_extent_(mask)
            mask = mask[sa, sb]
            if mask.all():
                mask = None
        if kind == "rect":
            return (_as_indexer_(ix[sa]), _as_indexer_(iy[sb])), mask
        return (
            slice(j0 + sa.start, j0 + sa.stop),
            slice(i0 + sb.start, i0 + sb.stop),
        ), mask
