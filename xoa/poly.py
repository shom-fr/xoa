"""
Polygons from xarray objects

See :func:`get_contour_polygons` to extract the contours of a field.
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

from . import exceptions
from . import grid as xgrid
from .core import poly as cpoly


def get_contour_polygons(field, level, containing=None):
    """Get the closed contours of a 2D field as polygons

    Parameters
    ----------
    field: xarray.DataArray
        Field with horizontal dimensions and longitudes and latitudes that may be 1D or 2D.
        Missing values are ignored.
    level: float
        Value of the contour
    containing: None, tuple(float), {"min", "max"}
        Only get the polygon that surrounds this ``(lon, lat)`` point, or the point
        where the field is the lowest or the highest. In this case, the polygon is returned
        alone and an error is raised if there is none.

    Return
    ------
    list(numpy.ndarray(n, 2)), numpy.ndarray(n, 2)
        Polygons as arrays of ``(lon, lat)`` vertices, whose last one repeats the first one.
        They can be passed to :func:`xoa.mask.get_polygon_mask`.

    Example
    -------
    .. code-block:: python

        polygon = get_contour_polygons(okuboweiss, -0.2, containing="min")
        mask = xoa.mask.get_polygon_mask(okuboweiss, polygon)
    """
    from contourpy import contour_generator

    if field.ndim != 2:
        raise exceptions.XoaError("The field must be two-dimensional")
    lon, lat = xgrid._get_lonlat_yx_(field)
    lonv, latv = lon.values, lat.values
    values = np.ma.masked_invalid(field.transpose(*lon.dims).values)
    generator = contour_generator(lonv, latv, values)
    polygons = [p for p in generator.lines(level) if np.allclose(p[0], p[-1])]
    if containing is None:
        return polygons
    if isinstance(containing, str):
        if containing not in ("min", "max"):
            raise exceptions.XoaError("containing must be a point, 'min' or 'max'")
        find = np.nanargmin if containing == "min" else np.nanargmax
        index = find(values.filled(np.nan))
        containing = lonv.flat[index], latv.flat[index]
    for polygon in polygons:
        if cpoly.point_in_polygon(float(containing[0]), float(containing[1]), polygon):
            return polygon
    raise exceptions.XoaError("No contour surrounds the point")
