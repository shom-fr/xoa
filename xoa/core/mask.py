"""
Masks of grid cells

Numba routines that tell which cells of a grid, given by their centers and corners,
are inside polygons. All the inputs are numpy arrays in degrees.
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

import numba as nb
import numpy as np

from .num import EPSILON
from .poly import clip_polygon, point_in_polygon, unwrap_longitudes

NOT_CI = os.environ.get("CI", "false") == "false"


@nb.njit(cache=NOT_CI, nogil=True)
def _signed_area_(poly):
    """Planar signed area, positive for a counter-clockwise polygon"""
    n = poly.shape[0]
    area = 0.0
    for k in range(n):
        k1 = (k + 1) % n
        area += poly[k, 0] * poly[k1, 1] - poly[k1, 0] * poly[k, 1]
    return 0.5 * area


@nb.njit(cache=NOT_CI, nogil=True, parallel=True)
def _polygon_cell_fraction_(poly, lon, lat, lon_bounds, lat_bounds, centers_only):
    ny, nx = lon.shape
    frac = np.zeros((ny, nx))
    ref = poly[:, 0].mean()
    pxmin, pxmax = poly[:, 0].min(), poly[:, 0].max()
    pymin, pymax = poly[:, 1].min(), poly[:, 1].max()
    for j in nb.prange(ny):
        for i in range(nx):
            if np.isnan(lon[j, i]) or np.isnan(lat[j, i]):
                continue

            # Same 360° range as the polygon
            shift = 360.0 * np.round((lon[j, i] - ref) / 360.0)
            if centers_only:
                x = lon[j, i] - shift
                if pxmin <= x <= pxmax and pymin <= lat[j, i] <= pymax:
                    if point_in_polygon(x, lat[j, i], poly):
                        frac[j, i] = 1.0
                continue

            # Cell as a counter-clockwise polygon, continuous in longitude
            cell = np.empty((4, 2))
            cell[:, 0] = unwrap_longitudes(lon_bounds[j, i], lon[j, i]) - shift
            cell[:, 1] = lat_bounds[j, i]
            cxmin, cxmax = cell[:, 0].min(), cell[:, 0].max()
            cymin, cymax = cell[:, 1].min(), cell[:, 1].max()
            if cxmax < pxmin or cxmin > pxmax or cymax < pymin or cymin > pymax:
                continue
            carea = _signed_area_(cell)
            if carea < 0:
                cell = cell[::-1].copy()
                carea = -carea

            # The cell is entirely inside when its corners are and no vertex is in it
            nin = 0
            for k in range(4):
                if point_in_polygon(cell[k, 0], cell[k, 1], poly):
                    nin += 1
            vertex_in = False
            for k in range(poly.shape[0]):
                if cxmin <= poly[k, 0] <= cxmax and cymin <= poly[k, 1] <= cymax:
                    vertex_in = True
                    break
            if nin == 4 and not vertex_in:
                frac[j, i] = 1.0
                continue

            # Part of the cell that is inside
            clipped = clip_polygon(poly, cell)
            if clipped.shape[0] < 3:
                continue
            if carea > 0:
                frac[j, i] = min(abs(_signed_area_(clipped)) / carea, 1.0)
    return frac


def polygon_cell_fraction(poly, lon, lat, lon_bounds=None, lat_bounds=None):
    """Fraction of grid cells that is covered by a polygon

    The polygon is in geographical coordinates and may be concave and cross the dateline,
    but it must not span more than 180° of longitude.
    The fraction is the ratio of the areas computed in longitude and latitude coordinates.

    Parameters
    ----------
    poly: array_like(n, 2)
        Vertices of the polygon as (lon, lat) pairs in degrees
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes of the cell centers
    lon_bounds, lat_bounds: array_like(ny, nx, 4), None
        Longitudes and latitudes of the cell corners, as returned by
        :func:`xoa.core.grid.edges2bounds`.
        When not provided, the fraction is 1 where the cell center is inside the polygon,
        and 0 elsewhere.

    Return
    ------
    array(ny, nx)
        Fractions between 0 and 1, or 0 where the center is missing

    See also
    --------
    polygon_mask
    """
    poly = np.array(poly, dtype=np.float64)
    poly[:, 0] = unwrap_longitudes(poly[:, 0], poly[0, 0])
    lon = np.ascontiguousarray(lon, dtype=np.float64)
    lat = np.ascontiguousarray(lat, dtype=np.float64)
    centers_only = lon_bounds is None or lat_bounds is None
    if centers_only:
        lon_bounds = lat_bounds = np.empty((*lon.shape, 4))
    return _polygon_cell_fraction_(
        poly,
        lon,
        lat,
        np.ascontiguousarray(lon_bounds, dtype=np.float64),
        np.ascontiguousarray(lat_bounds, dtype=np.float64),
        centers_only,
    )


def polygon_mask(poly, lon, lat, lon_bounds=None, lat_bounds=None, threshold=0.5):
    """Mask of the grid cells that are inside a polygon

    A cell is inside when the fraction of its area covered by the polygon,
    as computed by :func:`polygon_cell_fraction`, is at least `threshold`.

    Parameters
    ----------
    poly: array_like(n, 2)
        Vertices of the polygon as (lon, lat) pairs in degrees
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes of the cell centers
    lon_bounds, lat_bounds: array_like(ny, nx, 4), None
        Longitudes and latitudes of the cell corners.
        When not provided, a cell is inside when its center is.
    threshold: float
        Minimal covered fraction between 0 and 1, which is ignored without bounds.
        A null threshold selects all the cells that touch the polygon.

    Return
    ------
    array(bool)
        Shape of `lon`
    """
    frac = polygon_cell_fraction(poly, lon, lat, lon_bounds, lat_bounds)
    if lon_bounds is None or lat_bounds is None:
        return frac > 0.5
    if threshold <= 0:
        return frac > 0
    return frac >= threshold - EPSILON
