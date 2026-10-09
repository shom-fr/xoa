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

import numpy as np

from .geo import EARTH_RADIUS, diff_lon, haversine


def _centers2edges1d_(zi):
    zo = np.empty(zi.size + 1)
    zo[0] = zi[0] + 0.5 * (zi[0] - zi[1])
    zo[-1] = zi[-1] + 0.5 * (zi[-1] - zi[-2])
    zo[1:-1] = 0.5 * (zi[:-1] + zi[1:])
    return zo


def centers2edges(zz, axis=None):
    """Convert from centers to edges

    Parameters
    ----------
    zz: array(NX) or (NY,NX)

    Return
    ------
    array(NX+1) or (NY+1, NX+1)
    """
    if axis is not None:
        return np.apply_along_axis(_centers2edges1d_, axis, zz)

    # 1D
    if zz.ndim == 1:
        return _centers2edges1d_(zz)

    # 2D
    for axis in [0, 1]:
        zz = np.apply_along_axis(_centers2edges1d_, axis, zz)
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
