"""
Pure numeric grid utilities
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

from .geo import EARTH_RADIUS, diff_lon, haversine

NOT_CI = os.environ.get("CI", "false") == "false"

#: Size of the blocks of the trailing dimensions that are shared among threads
_BLOCK = 512


@numba.njit(cache=NOT_CI, parallel=True)
def _centers3d_(a, o):
    no, n, ni = a.shape
    nb = (ni + _BLOCK - 1) // _BLOCK
    for t in numba.prange(no * nb):
        i = t // nb
        k0 = (t % nb) * _BLOCK
        k1 = min(ni, k0 + _BLOCK)
        for j in range(n - 1):
            for k in range(k0, k1):
                o[i, j, k] = a[i, j, k] + 0.5 * (a[i, j + 1, k] - a[i, j, k])


@numba.njit(cache=NOT_CI, parallel=True)
def _edges3d_(a, o):
    no, n, ni = a.shape
    nb = (ni + _BLOCK - 1) // _BLOCK
    for t in numba.prange(no * nb):
        i = t // nb
        k0 = (t % nb) * _BLOCK
        k1 = min(ni, k0 + _BLOCK)
        for k in range(k0, k1):
            o[i, 0, k] = a[i, 0, k] + 0.5 * (a[i, 0, k] - a[i, 1, k])
            o[i, n, k] = a[i, n - 1, k] + 0.5 * (a[i, n - 1, k] - a[i, n - 2, k])
        for j in range(1, n):
            for k in range(k0, k1):
                o[i, j, k] = a[i, j - 1, k] + 0.5 * (a[i, j, k] - a[i, j - 1, k])


@numba.njit(cache=NOT_CI, parallel=True)
def _diff3d_(a, o):
    no, n, ni = a.shape
    nb = (ni + _BLOCK - 1) // _BLOCK
    for t in numba.prange(no * nb):
        i = t // nb
        k0 = (t % nb) * _BLOCK
        k1 = min(ni, k0 + _BLOCK)
        for j in range(n - 1):
            for k in range(k0, k1):
                o[i, j, k] = a[i, j + 1, k] - a[i, j, k]


@numba.njit(cache=NOT_CI, parallel=True)
def _pad3d_(a, o, p0, p1, mode, cval):
    # mode: 0 edge, 1 linear_extrap, 2 constant, 3 wrap
    no, n, ni = a.shape
    nb = (ni + _BLOCK - 1) // _BLOCK
    for t in numba.prange(no * nb):
        i = t // nb
        k0 = (t % nb) * _BLOCK
        k1 = min(ni, k0 + _BLOCK)
        for k in range(k0, k1):
            a0 = a[i, 0, k]
            a1 = a[i, n - 1, k]
            if mode == 3:
                for j in range(p0):
                    o[i, j, k] = a[i, (j - p0) % n, k]
                for j in range(p1):
                    o[i, p0 + n + j, k] = a[i, j % n, k]
            elif mode == 1:
                d0 = a0 - a[i, 1, k]
                d1 = a1 - a[i, n - 2, k]
                for j in range(p0):
                    o[i, j, k] = a0 + d0 * (p0 - j)
                for j in range(p1):
                    o[i, p0 + n + j, k] = a1 + d1 * (j + 1)
            else:
                v0 = cval if mode == 2 else a0
                v1 = cval if mode == 2 else a1
                for j in range(p0):
                    o[i, j, k] = v0
                for j in range(p1):
                    o[i, p0 + n + j, k] = v1
        for j in range(n):
            for k in range(k0, k1):
                o[i, p0 + j, k] = a[i, j, k]


@numba.njit(cache=NOT_CI, parallel=True)
def _shift3d_(a, o, direction, linear):
    # Centers of the points and their neighbour up (direction > 0) or down,
    # which is replicated or extrapolated at the end
    no, n, ni = a.shape
    nb = (ni + _BLOCK - 1) // _BLOCK
    up = direction > 0
    s = 0 if up else 1
    for t in numba.prange(no * nb):
        i = t // nb
        k0 = (t % nb) * _BLOCK
        k1 = min(ni, k0 + _BLOCK)
        for j in range(n - 1):
            for k in range(k0, k1):
                o[i, j + s, k] = a[i, j, k] + 0.5 * (a[i, j + 1, k] - a[i, j, k])
        for k in range(k0, k1):
            if up:
                e = a[i, n - 1, k]
                o[i, n - 1, k] = e + 0.5 * (e - a[i, n - 2, k]) if linear else e
            else:
                e = a[i, 0, k]
                o[i, 0, k] = e + 0.5 * (e - a[i, 1, k]) if linear else e


@numba.njit(cache=NOT_CI, parallel=True)
def _gradient3d_(a, o, x):
    no, n, ni = a.shape
    nb = (ni + _BLOCK - 1) // _BLOCK
    for t in numba.prange(no * nb):
        i = t // nb
        k0 = (t % nb) * _BLOCK
        k1 = min(ni, k0 + _BLOCK)
        ix = i if x.shape[0] > 1 else 0
        for j in range(n - 1):
            for k in range(k0, k1):
                kx = k if x.shape[2] > 1 else 0
                o[i, j, k] = (a[i, j + 1, k] - a[i, j, k]) / (x[ix, j + 1, kx] - x[ix, j, kx])


def _as3d_(a, axis):
    """Contiguous float array seen as ``(before, n, after)`` around `axis`"""
    a = np.asarray(a)
    if a.dtype not in (np.float32, np.float64):
        a = a.astype("d")
    shape = a.shape
    return np.ascontiguousarray(a).reshape(
        math.prod(shape[:axis]), shape[axis], math.prod(shape[axis + 1 :])
    )


def _apply_along_axis_(func, a, axis, size, *args):
    """Run a 3D kernel along an axis, with a size-`size` output along it"""
    axis %= np.ndim(a)
    a3 = _as3d_(a, axis)
    shape = np.shape(a)
    o = np.empty((a3.shape[0], size, a3.shape[2]), a3.dtype)
    func(a3, o, *args)
    return o.reshape(shape[:axis] + (size,) + shape[axis + 1 :])


def centers(a, axis=-1):
    """Average consecutive points along an axis: size ``n`` becomes ``n-1``"""
    return _apply_along_axis_(_centers3d_, a, axis, np.shape(a)[axis] - 1)


def edges(a, axis=-1):
    """Cell edges from centers along an axis, extrapolated at the ends: ``n`` becomes ``n+1``"""
    n = np.shape(a)[axis]
    if n < 2:
        raise ValueError("At least two points are needed to extrapolate")
    return _apply_along_axis_(_edges3d_, a, axis, n + 1)


def diff(a, axis=-1):
    """Difference of consecutive points along an axis: size ``n`` becomes ``n-1``"""
    return _apply_along_axis_(_diff3d_, a, axis, np.shape(a)[axis] - 1)


_PAD_MODES = {"edge": 0, "linear_extrap": 1, "constant": 2, "wrap": 3}


def pad(a, axis=-1, width=1, mode="edge", constant_values=np.nan):
    """Pad an array along an axis

    Parameters
    ----------
    a: array_like
    axis: int
    width: int, (int, int)
        Number of points added before and after
    mode: {"edge", "linear_extrap", "constant", "wrap"}
        Replicate the end values, extrapolate them linearly, fill with
        `constant_values`, or repeat the array periodically
    constant_values: float
        Value used with the ``"constant"`` mode
    """
    if mode not in _PAD_MODES:
        raise ValueError(f"Invalid pad mode: {mode}")
    p0, p1 = (width, width) if isinstance(width, int) else (width[0], width[-1])
    n = np.shape(a)[axis]
    if n < 2 and mode == "linear_extrap":
        raise ValueError("At least two points are needed to extrapolate")
    if n < 1 and mode == "wrap":
        raise ValueError("At least one point is needed to wrap")
    return _apply_along_axis_(
        _pad3d_, a, axis, n + p0 + p1, p0, p1, _PAD_MODES[mode], float(constant_values)
    )


def shift(a, axis=-1, direction=1, mode="edge"):
    """Move an array by half a grid cell along an axis, keeping its size

    Parameters
    ----------
    a: array_like
    axis: int
    direction: int
        Positive to move up, and negative to move down
    mode: {"edge", "linear_extrap"}
        How the end value is obtained
    """
    if mode not in ("edge", "linear_extrap"):
        raise ValueError(f"Invalid shift mode: {mode}")
    n = np.shape(a)[axis]
    if n < 2:
        raise ValueError("At least two points are needed to shift")
    return _apply_along_axis_(_shift3d_, a, axis, n, direction, mode == "linear_extrap")


def gradient(a, x, axis=-1):
    """Derivative of an array with respect to a coordinate, at the centers: ``n`` becomes ``n-1``

    Parameters
    ----------
    a: array_like
    x: array_like
        Coordinate, 1D along the axis or of the same shape as `a`
    axis: int
    """
    axis %= np.ndim(a)
    x = np.asarray(x, dtype="d")
    if x.ndim == 1:
        x3 = x.reshape(1, -1, 1)
    elif x.shape == np.shape(a):
        x3 = _as3d_(x, axis)
    else:
        raise ValueError("x must be 1D or have the shape of a")
    if x3.shape[1] != np.shape(a)[axis]:
        raise ValueError("x must have the size of a along the axis")
    return _apply_along_axis_(_gradient3d_, a, axis, np.shape(a)[axis] - 1, x3)


def centers2edges(zz, axis=None):
    """Convert from centers to edges

    Parameters
    ----------
    zz: array(NX) or (NY,NX)
    axis: int, None
        Single axis, or all axes by default

    Return
    ------
    array(NX+1) or (NY+1, NX+1)
    """
    zz = np.asarray(zz)
    for ax in range(zz.ndim) if axis is None else [axis]:
        zz = edges(zz, ax)
    return zz


def _edges2bounds1d_(edges):
    nxp1 = edges.shape[0]
    bounds = np.empty((nxp1 - 1, 2), dtype=np.float64)
    for i in range(nxp1 - 1):
        bounds[i] = edges[i : i + 2]
    return bounds


def edges2bounds(edges, axis=None):
    """Convert edges (nx+1)|(ny+1,nx+1) to bounds (nx,2)|(ny,nx,4)"""
    if edges.ndim == 2 and axis is None:
        nyp1, nxp1 = edges.shape
        bounds = np.zeros((nyp1 - 1, nxp1 - 1, 4), dtype=np.float64)
        for j in range(nyp1 - 1):
            for i in range(nxp1 - 1):
                bounds[j, i, :2] = edges[j, i : i + 2]
                bounds[j, i, 2:] = edges[j + 1, i + 1 : (i - 1) if i else None : -1]
        return bounds
    if axis is None:
        axis = 0
    return np.apply_along_axis(_edges2bounds1d_, axis, edges)


def check_grid_type(grid, tol=1e-8):
    """Infer if grid is regular, rectangular or curvilinear"""
    if 'type' in grid and grid['type'] is not None:
        return grid['type']
    for axis, coord in ((0, 'lat'), (1, 'lon')):
        coord = grid[coord]
        if not np.allclose(coord.min(1 - axis), coord.max(1 - axis), atol=tol, equal_nan=True):
            grid['type'] = 'curvilinear'
            return 'curvilinear'
        diff = np.diff(coord, axis=axis)
        if diff.size and not np.allclose(diff, diff.mean(), atol=tol, equal_nan=True):
            grid['type'] = 'rectangular'
            return 'rectangular'
    grid['type'] = 'regular'
    return 'regular'


def create_regular_grid(nx, ny, lon_range, lat_range, mask=None):
    """Create a regular latitude-longitude grid

    Parameters:
    -----------
    nx, ny
        Grid dimensions
    lon_range, lat_range : tuple(float, float)
        Longitude and latitude ranges in degrees

    Return
    ------
    dict
        With the following keys: lon, lat, lon_bounds, lat_bounds, lon_edges, lat_edges

    """
    # Create 1D coordinates for cell centers
    lons_1d = np.linspace(lon_range[0], lon_range[1], nx)
    lats_1d = np.linspace(lat_range[0], lat_range[1], ny)

    # Create 2D meshgrids for cell centers
    lons, lats = np.meshgrid(lons_1d, lats_1d)

    # Create cell boundaries (edges between cells)
    dlon = (lon_range[1] - lon_range[0]) / (nx - 1)
    dlat = (lat_range[1] - lat_range[0]) / (ny - 1)

    # Edges
    lon_edges_1d = np.linspace(lon_range[0] - dlon / 2, lon_range[1] + dlon / 2, nx + 1)
    lat_edges_1d = np.linspace(lat_range[0] - dlat / 2, lat_range[1] + dlat / 2, ny + 1)
    lon_edges, lat_edges = np.meshgrid(lon_edges_1d, lat_edges_1d)

    # Corners/bounds (ny, nx, 4)
    lon_bounds = edges2bounds(lon_edges)
    lat_bounds = edges2bounds(lat_edges)

    # Return new copies of arrays to ensure no sharing
    grid = {
        'lon': lons,
        'lat': lats,
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'lon_edges': lon_edges,
        'lat_edges': lat_edges,
        'type': 'rectangular',
        'mask': mask,
    }
    return grid


def create_rotated_grid(nx, ny, pole_lon, pole_lat, rotation, lon_span, lat_span, mask=None):
    """
    Create a rotated grid (simplified rotated pole).

    Parameters:
    -----------
    nx, ny
        Grid dimensions
    pole_lon, pole_lat
        Location of the rotated pole in degrees
    rotation
        Additional rotation angle in degrees
    lon_span, lat_span
        Span of the grid in rotated coordinates

    Return
    ------
    dict
        With the following keys: lon, lat, lon_bounds, lat_bounds, lon_edges, lat_edges
    """
    lon_range = (-lon_span / 2, lon_span / 2)
    lat_range = (-lat_span / 2, lat_span / 2)
    rgrid = create_regular_grid(nx, ny, lon_range, lat_range)

    # Simple rotation (not full rotated pole transformation)
    rot_rad = np.radians(rotation)

    # Apply rotation
    lons = rgrid['lon'] * np.cos(rot_rad) - rgrid['lat'] * np.sin(rot_rad) + pole_lon
    lats = rgrid['lon'] * np.sin(rot_rad) + rgrid['lat'] * np.cos(rot_rad) + pole_lat

    lon_edges = (
        rgrid['lon_edges'] * np.cos(rot_rad) - rgrid['lat_edges'] * np.sin(rot_rad) + pole_lon
    )
    lat_edges = (
        rgrid['lon_edges'] * np.sin(rot_rad) + rgrid['lat_edges'] * np.cos(rot_rad) + pole_lat
    )

    # Corners/bounds (ny, nx, 4)
    lon_bounds = edges2bounds(lon_edges)
    lat_bounds = edges2bounds(lat_edges)

    # Return new copies of all arrays to ensure no sharing
    grid = {
        'lon': lons,
        'lat': lats,
        'lon_bounds': lon_bounds,
        'lat_bounds': lat_bounds,
        'lon_edges': lon_edges,
        'lat_edges': lat_edges,
        'type': 'curvilinear',
        'mask': mask,
    }
    return grid


def compute_resolution(lon, lat, radius=EARTH_RADIUS):
    """Compute the grid resolution between adjacent points along x and y

    Parameters
    ----------
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes in degrees
    radius: float
        Radius of the sphere in meters, which defaults to the earth radius

    Return
    ------
    array(ny, nx-1)
        Distance in meters between adjacent points along x
    array(ny-1, nx)
        Distance in meters between adjacent points along y
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    dx = haversine(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])
    dy = haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return dx * radius, dy * radius


def compute_centered_resolution(lon, lat, radius=EARTH_RADIUS):
    """Compute the grid resolution at the grid points along x and y

    It is the mean of the resolutions between the adjacent points, and the
    resolution of the only adjacent point on the edges.
    See :func:`compute_resolution` for the resolutions between adjacent points.

    Parameters
    ----------
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes in degrees
    radius: float
        Radius of the sphere in meters, which defaults to the earth radius

    Return
    ------
    array(ny, nx)
        Resolution in meters along x
    array(ny, nx)
        Resolution in meters along y
    """
    dxe, dye = compute_resolution(lon, lat, radius=radius)
    dx = np.empty(np.shape(lon))
    dx[:, 1:-1] = 0.5 * (dxe[:, :-1] + dxe[:, 1:])
    dx[:, 0] = dxe[:, 0]
    dx[:, -1] = dxe[:, -1]
    dy = np.empty(np.shape(lon))
    dy[1:-1] = 0.5 * (dye[:-1] + dye[1:])
    dy[0] = dye[0]
    dy[-1] = dye[-1]
    return dx, dy


def median_resolution_deg(lon, lat):
    """Compute the median grid resolution in degrees

    It uses the coordinate differences, so it handles rotated and curvilinear
    grids: the result is the extent in degrees spanned by a grid cell
    along the longitude and latitude axes.

    Parameters
    ----------
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes in degrees

    Return
    ------
    float
        Maximum of the median extents along longitude and latitude,
        considering both grid directions.
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    res_lon = max(
        float(np.median(np.abs(np.diff(lon, axis=1)))),
        float(np.median(np.abs(np.diff(lon, axis=0)))),
    )
    res_lat = max(
        float(np.median(np.abs(np.diff(lat, axis=1)))),
        float(np.median(np.abs(np.diff(lat, axis=0)))),
    )
    return max(res_lon, res_lat)


def compute_center_resolution(lon, lat, radius=EARTH_RADIUS):
    """Compute the grid resolution at the centers

    It is the geometric mean of the resolutions along x and y, where the resolution
    at a point is the mean of the distances to its neighbours along each direction.

    Parameters
    ----------
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes in degrees
    radius: float
        Radius of the sphere in meters, which defaults to the earth radius

    Return
    ------
    array(ny, nx)
        Resolution in meters

    See also
    --------
    compute_resolution
    """
    dx, dy = compute_resolution(lon, lat, radius=radius)
    ny, nx = np.shape(lon)
    cdx = np.empty((ny, nx))
    cdx[:, 0] = dx[:, 0]
    cdx[:, -1] = dx[:, -1]
    cdx[:, 1:-1] = 0.5 * (dx[:, :-1] + dx[:, 1:])
    cdy = np.empty((ny, nx))
    cdy[0] = dy[0]
    cdy[-1] = dy[-1]
    cdy[1:-1] = 0.5 * (dy[:-1] + dy[1:])
    return np.sqrt(cdx * cdy)


def unwrap_grid_longitudes(lon):
    """Make the longitudes of a 2D grid continuous across the dateline

    The first point is kept, and each other one is shifted by a multiple of 360°
    to be next to its neighbour along the lines, or along the first column.
    The result may be out of [-180, 180].

    Parameters
    ----------
    lon: array_like
        2D longitudes

    Return
    ------
    numpy.ndarray
    """
    lon = np.asarray(lon, dtype="d")
    out = np.empty_like(lon)
    out[:, 0] = lon[0, 0] + np.concatenate([[0.0], np.cumsum(diff_lon(lon[:-1, 0], lon[1:, 0]))])
    out[:, 1:] = out[:, :1] + np.cumsum(diff_lon(lon[:, :-1], lon[:, 1:]), axis=1)
    return out
