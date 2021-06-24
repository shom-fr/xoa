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


@numba.vectorize(cache=True)
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
    deg2rad = math.pi / 180.0
    dist = math.sin(deg2rad * (lat0 - lat1) * 0.5) ** 2
    dist += (
        math.cos(deg2rad * lat0)
        * math.cos(deg2rad * lat1)
        * math.sin(deg2rad * (lon0 - lon1) * 0.5) ** 2
    )
    dist = 2.0 * math.asin(math.sqrt(dist))
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
    return _haversine_(np.double(lon0), np.double(lat0), np.double(lon1), np.double(lat1)) * radius


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
    return _bearing_(np.double(lon0), np.double(lat0), np.double(lon1), np.double(lat1))


@numba.vectorize
def _bearing_(lon0, lat0, lon1, lat1):
    deg2rad = math.pi / 180.0
    a = math.arctan2(
        math.cos(deg2rad * lat0) * math.sin(deg2rad * lat1)
        - math.sin(deg2rad * lat0) * math.cos(deg2rad * lat1) * math.cos(deg2rad * (lon1 - lon0)),
        math.sin(deg2rad * (lon1 - lon0)) * math.cos(deg2rad * lat1),
    )
    return a * 180 / math.pi


def cdist(XA, XB, radius=EARTH_RADIUS):
    """Compute the haversine distances between positions of two datasets

    Parameters
    ----------
    XA: numpy.array
        An m by 2 array of coordinates in a geographical space.
    XB: numpy.array
        An m by 2 array of coordinates in a geographical space.
    radius: float
        Radius of the sphere which defaults to the earth radius

    Returns
    -------
    numpy.array
        2D array of distances

    See also
    --------
    haversine
    pdist
    scipy.sparial.distances.cdist
    scipy.sparial.distances.pdist
    """
    lons0 = XA[:, 0]
    lats0 = XA[:, 1]
    lons1 = XB[:, 0]
    lats1 = XB[:, 1]
    xx = np.meshgrid(lons1, lons0)
    yy = np.meshgrid(lats1, lats0)
    return haversine(xx[0], yy[0], xx[1], yy[1], radius=radius)


def pdist(X, compact=False, radius=EARTH_RADIUS):
    """Compute the pairwise haversine distances between positions of a single dataset

    Parameters
    ----------
    X: numpy.array
        An m by 2 array of coordinates in a geographical space.
    compact: bool
        Compact the distance matrix to remove duplicate and zeros.
        It is the strict upper triangle of the distance matrix.
    radius: float
        Radius of the sphere which defaults to the earth radius

    Returns
    -------
    numpy.array
        Either 2D (square form) or 1D (compact form) the distance matrix

    See also
    --------
    haversine
    cdist
    numpy.triu
    scipy.sparial.distances.pdist
    scipy.sparial.distances.cdist
    scipy.sparial.distances.squareform
    """
    dd = cdist(X, X, radius=radius)
    if compact:
        return dd[np.triu_indices(dd.shape[0], 1)]
    return dd


def _adapative_cdist_(XA, XB, method="haversine", **kwargs):
    if method == "haversine":
        return cdist(XA, XB, **kwargs)
    import scipy.spatial.distance

    return scipy.spatial.distance.cdist(XA, XB, method=method, **kwargs)


def _adapative_pdist_(X, method="haversine", **kwargs):
    if method == "haversine":
        return pdist(X, **kwargs)
    import scipy.spatial.distance

    return scipy.spatial.distance.pdist(X, method=method, **kwargs)


class ScipyDistContext(object):
    """Context to switch the :func:`scipy.spatial.distance.cdist` fonction to :func:`cdist`

    Parameters
    ----------
    force: bool
        If true, this function will be used whatever the distance method asked for is.
    """

    def __init__(self, cdist=True, pdist=True, force=False):
        self.switch_cdist = cdist
        self.switch_pdist = pdist
        import scipy.spatial.distance

        self.distmod = scipy.spatial.distance
        if force:
            self.cdist = cdist
            self.pdist = pdist
        else:
            self.cdist = _adapative_cdist_
            self.pdist = _adapative_pdist_

    def __enter__(self):
        if self.switch_cdist:
            self._old_cdist = getattr(self.distmod, "cdist")
            setattr(self.distmod, "cdist", self.cdist)
        if self.switch_pdist:
            self._old_pdist = getattr(self.distmod, "pdist")
            setattr(self.distmod, "pdist", self.pdist)
        return self

    def __exit__(self, type, value, traceback):
        if self.switch_cdist:
            setattr(self.distmod, "cdist", self._old_cdist)
        if self.switch_pdist:
            setattr(self.distmod, "pdist", self._old_pdist)


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
        x0 = 0.5 * (xmin + xmax)
        y0 = 0.5 * (ymin + ymax)
        if square:
            aspect = dx * math.cos(y0 * math.pi / 180) / dy
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
