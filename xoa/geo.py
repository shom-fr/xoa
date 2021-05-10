#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geographic utilities
"""
# Copyright 2020-2021 Shom
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
import numpy as np
import numba

from . import coords as xcoords


#: Earth radius in meters
EARTH_RADIUS = 6371e3


@numba.vectorize
def _haversine_(lon0, lat0, lon1, lat1):
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
    deg2rad = math.pi / 180.
    dist = math.sin(deg2rad*(lat0-lat1)*0.5)**2
    dist += (math.cos(deg2rad*lat0) * math.cos(deg2rad*lat1) *
             math.sin(deg2rad*(lon0-lon1)*0.5)**2)
    dist = 2. * math.asin(math.sqrt(dist))
    return dist


def haversine(lon0, lat0, lon1, lat1, radius=EARTH_RADIUS):
    """Haversine distance between two points

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
    radius: float
        Radius of the sphere which defaults to the earth radius

    Return
    ------
    float, array_like
        Distance(s)
    """
    return _haversine_(
        np.double(lon0), np.double(lat0),
        np.double(lon1), np.double(lat1)) * radius


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
    return _bearing_(
        np.double(lon0), np.double(lat0), np.double(lon1), np.double(lat1))


@numba.vectorize
def _bearing_(lon0, lat0, lon1, lat1):
    deg2rad = math.pi / 180.
    a = math.arctan2(
        math.cos(deg2rad*lat0)*math.sin(deg2rad*lat1) -
        math.sin(deg2rad*lat0)*math.cos(deg2rad*lat1)*math.cos(deg2rad*(lon1-lon0)),
        math.sin(deg2rad*(lon1-lon0))*math.cos(deg2rad*lat1))
    return a * 180 / math.pi


def get_extent(extent, margin=0, square=False):
    """Compute the geographic extent in degrees

    Parameters
    ----------
    extent: xarray.DataArray, xarray.Dataset, dict, tuple, list
        Either:

        - An array or dataset with longitude and latitude coordinates.
        - A dict with ``lon`` and ``lat`` keys: ``dict(lon=..., lat=...)``
        - A two-element tuple of longitudes and latitudes: ``(lon, lat)``
        - A extent list: ``[xmin, xmax, ymin, ymax]``.
    margin: float
        A relative fraction of the width and height that is used to set margins.
        For instance, a value of ``-0.1`` shrinks the box of 10% on each side.
    square: bool
        Force the box to be square in degrees.

    Return
    ------
    list
        ``[xmin, xmax, ymin, ymax]``

    Example
    -------

    .. ipython:: python

        @suppress
        from xoa.geo import get_extent
        @suppress
        import numpy as np
        get_extent([10., 20., 10., 20.], margin=0.1)
        get_extent({"lon": np.linspace(10, 20, 5), "lat": np.linspace(10, 20, 5)}, square=True)
        get_extent((np.linspace(10, 20, 5), np.linspace(10, 20, 5)), margin=-.1, square=True)

    """
    # Get min and max
    if hasattr(extent, "coords"):
        lon = xcoords.get_lon(extent)
        lat = xcoords.get_lat(extent)
        extent = (lon, lat)
    elif isinstance(extent, dict):
        extent = extent['lon'], extent['lat']
    if isinstance(extent, (list, np.ndarray)):
        xmin, xmax, ymin, ymax = extent
    else:  # tuple
        lon, lat = extent
        xmin = float(np.min(lon))
        xmax = float(np.max(lon))
        ymin = float(np.min(lat))
        ymax = float(np.max(lat))

    # Scale
    if square or margin:
        dx = xmax - xmin
        dy = ymax - ymin
        x0 = 0.5 * (xmin+xmax)
        y0 = 0.5 * (ymin+ymax)
        if square:
            aspect = dx * math.cos(y0*math.pi/180) / dy
            if aspect > 1:
                dy *= aspect
            else:
                dx /= aspect
        xmargin = margin * dx
        ymargin = margin * dy
        xmin = x0 - 0.5 * dx - xmargin
        xmax = x0 + 0.5 * dx + xmargin
        ymin = y0 - 0.5 * dy - ymargin
        ymax = y0 + 0.5 * dy + ymargin

    return [xmin, xmax, ymin, ymax]
