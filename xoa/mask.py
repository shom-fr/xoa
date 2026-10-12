"""
Masks of grid cells

Tell which cells of a grid are inside polygons, see :func:`get_polygon_mask`,
or where several arrays are all valid, see :func:`get_common_mask`.
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
import xarray as xr

from . import coords as xcoords
from . import exceptions
from . import grid as xgrid
from .core import grid as cgrid
from .core import mask as cmask


def _as_polygons_(polygon):
    """Convert polygons to a list of (lon, lat) arrays of shape (n, 2)"""
    if hasattr(polygon, "geoms"):  # Multi-geometry
        return [p for g in polygon.geoms for p in _as_polygons_(g)]
    if hasattr(polygon, "exterior"):  # Polygon, whose holes are ignored
        return [np.asarray(polygon.exterior.coords, dtype="d")[:, :2]]
    if isinstance(polygon, (list, tuple)) and len(polygon) and not np.ndim(polygon[0]) == 1:
        return [p for g in polygon for p in _as_polygons_(g)]
    poly = np.asarray(polygon, dtype="d")
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3:
        raise exceptions.XoaError("A polygon must be an array of at least 3 (lon, lat) pairs")
    return [poly]


def get_polygon_mask(obj, polygon, threshold=0.5):
    """Get a mask of the grid cells that are inside polygons

    A cell is inside a polygon when the fraction of its area covered by the polygon
    is at least `threshold`. The cells are delimited by edges that are
    half way between the grid points.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D
    polygon: array_like(n, 2), list, shapely.Polygon
        Polygon vertices as (lon, lat) pairs in degrees, or a list of them,
        or a shapely polygon or multi-polygon whose holes are ignored.
        The mask is the union of all the polygons.
        A polygon may cross the dateline but must not span more than 180° of longitude.
    threshold: float, None
        Minimal covered fraction of a cell, between 0 and 1.
        A null threshold selects all the cells that touch the polygon,
        and a threshold of 1 only those that are entirely inside.
        With ``None``, a cell is inside if its center is, which is faster.

    Return
    ------
    xarray.DataArray
        Boolean array with the horizontal dimensions of the longitudes and latitudes

    See also
    --------
    xoa.core.mask.polygon_mask
    """
    lon, lat = xgrid._get_lonlat_yx_(obj)
    lonv, latv = lon.values, lat.values
    kwargs = {}
    if threshold is not None:
        kwargs = dict(
            lon_bounds=cgrid.edges2bounds(cgrid.centers2edges(lonv)),
            lat_bounds=cgrid.edges2bounds(cgrid.centers2edges(latv)),
            threshold=threshold,
        )
    mask = np.zeros(lonv.shape, dtype=bool)
    for poly in _as_polygons_(polygon):
        mask |= cmask.polygon_mask(poly, lonv, latv, **kwargs)
    return xr.DataArray(
        mask,
        dims=lon.dims,
        coords=lon.coords,
        name="mask",
        attrs={"long_name": "Inside polygons"},
    )


def get_common_mask(*das, edge=0):
    """Get a mask of the points where all the arrays are finite

    Parameters
    ----------
    das: xarray.DataArray
        Arrays that are broadcast together
    edge: int
        Number of cells to discard along the borders of the horizontal dimensions,
        where derivatives are not reliable.

    Return
    ------
    xarray.DataArray
        Boolean array

    Example
    -------
    .. code-block:: python

        valid = get_common_mask(rossby_ref, rossby_s, rossby_z, edge=3)
        rossby_z.where(valid)
    """
    if not das:
        raise exceptions.XoaError("At least one array is needed")
    mask = np.isfinite(das[0])
    for da in das[1:]:
        mask = mask & np.isfinite(da)
    if edge:
        for dim in (
            xcoords.get_ydim(mask, errors="ignore"),
            xcoords.get_xdim(mask, errors="ignore"),
        ):
            if dim is not None:
                index = xr.DataArray(np.arange(mask.sizes[dim]), dims=dim)
                mask = mask & (index >= edge) & (index < mask.sizes[dim] - edge)
    return mask.rename("mask").assign_attrs(long_name="Valid points")
