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
import xarray as xr

from . import misc
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


class distance_units(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported units of distance"""

    #: Meters (default)
    m = 0
    #: Meters (default)
    meters = 0
    #: Meters (default)
    meter = 0
    #: Kilometers
    km = 1
    #: Kilometers
    kilometers = 1
    #: Kilometers
    kilometer = 1


def get_distances(da0, da1=None, radius=EARTH_RADIUS, units="m"):
    """Compute the haversine distance between two datasets/data arrays

    Parameters
    ----------
    da0: xarray.Dataset, xarray.DataArray
    da1: xarray.Dataset, xarray.DataArray
    radius: float
        Radius of the sphere which defaults to the earth radius
    units: int, str, distance_units
        Distance units as one of: {distance_units.rst_with_links}

    Return
    ------
    xarray.DataArray
        An array with dims `(npts0, npts1)`
    See also
    --------
    haversine
    cdist
    pdist
    """
    ds0 = xcoords.geo_stack(da0, rename=True, drop=True, reset_index=True)
    XY0 = np.dstack([ds0.lon.values, ds0.lat.values])[0]
    units = distance_units[units]
    du = str(units)
    import xarray as xr

    if da1 is None:
        dd = pdist(XY0, radius=radius)
        if units == distance_units.km:
            dd *= 1e-3
        dd = xr.DataArray(
            dd,
            dims=("npts", "npts"),
            attrs={"units": du},
            name="distance",
            coords={"lon": ds0.lon, "lat": ds0.lat},
        )
    else:
        ds0 = ds0.rename(npts="npts0")
        ds1 = xcoords.geo_stack(da1, "npts1", rename=True, drop=True, reset_index=True)
        XY1 = np.dstack([ds1.lon.values, ds1.lat.values])[0]
        dd = cdist(XY0, XY1, radius=radius)
        if units == distance_units.km:
            dd *= 1e-3
        dd = xr.DataArray(
            dd,
            dims=("npts0", "npts1"),
            attrs={"units": du},
            name="distance",
            coords={"lon_0": ds0.lon, "lat_0": ds0.lat, "lon_1": ds1.lon, "lat_1": ds1.lat},
        )
    return dd


get_distances.__doc__ = get_distances.__doc__.format(**locals())


def cdist(XA, XB, radius=EARTH_RADIUS, **kwargs):
    """A scipy-distance like cdist function for the haversine method

    Parameters
    ----------
    XA: numpy.array
        An ma by 2 array of coordinates of the first dataset in a geographical space.
    XB: numpy.array
        An mb by 2 array of coordinates of the second dataset in a geographical space.
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


def pdist(X, compact=False, radius=EARTH_RADIUS, **kwargs):
    """A scipy-distance like pdist function for the haversine method

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


def get_extent(extent, margin=0, square=False, min_extent=None):
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
    min_extent: None, float, (float, float)
        Minimal extent along x and y: ``(dx, dy)``.
        If a single floating value is provided, it is valid for both x and y.

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
    if square or margin or min_extent is not None:
        dx = xmax - xmin
        dy = ymax - ymin
        x0 = 0.5 * (xmin + xmax)
        y0 = 0.5 * (ymin + ymax)
        if min_extent is not None:
            if isinstance(min_extent, (float, int)):
                min_extent = (min_extent, min_extent)
            dx_min, dy_min = min_extent
            dx = max(dx, dx_min)
            dy = max(dy, dy_min)
        if square and dy != 0:
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


def clusterize(obj, npmax, split=False):
    """Split data into clouds of points of max size `npmax`

    Input object must have valid geographic coordinates.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        If a data array, it must have valid longitude and latitude coordinates.
    npmax: int
        Maximal number of point per cluster
    split:
        Return one dataset per cluster

    Return
    ------
    xarray.Dataset, list of xarray.Dataset
        A dataset has its longitude and latitude coordinates renamed `lon` and `lat`,
        and its stacked dimension renamed `npts`.
        It contains only arrays that contains the `npts` dimension.
        If a clustering was needed, the dataset contains the following arrays:

        iclust:
            Index of the cluster points belongs to.
        indices:
            Indices in the original dataset.
        centroid:
            Coordinate of the centroid(s)
        distorsion:
            Distances to the centroid that the points belongs to.

        If the input is a dataset, the global attribute :attr:`var_names`
        is set to the list of data var names.


    """
    # Positions
    obj = xcoords.geo_stack(obj, "npts", rename=True, drop=True)
    x = obj.lon.values
    y = obj.lat.values

    from scipy.cluster.vq import kmeans, vq

    # Nothing to do
    csize = len(x)
    if not isinstance(obj, xr.Dataset):
        obj = obj.to_dataset(name=obj.name or "data")
    obj.encoding["clust_var_names"] = list(obj)
    if npmax < 2 or csize <= npmax:
        return [obj] if split else obj

    # Loop on the number of clusters
    nclust = 2
    points = np.dstack((x, y))[0]
    ii = np.arange(csize)
    while csize > npmax:
        centroids, global_distorsion = kmeans(points, nclust)
        indices, distorsions = vq(points, centroids)
        sindices = [ii[indices == nc] for nc in range(nclust)]
        csizes = [sii.shape[0] for sii in sindices]
        order = np.argsort(csizes)[::-1]
        csize = csizes[order[0]]
        sdistorsions = [distorsions[sii] for sii in sindices]
        nclust += 1
    nclust = len(centroids)

    if split:

        #  Reorder
        sindices = [sindices[i] for i in order]
        sdistorsions = [sdistorsions[i] for i in order]
        centroids = centroids[order]

        # Split
        out = []
        for ic in range(nclust):
            obji = obj.isel(npts=sindices[ic])
            obji["iclust"] = xr.DataArray(ic)
            obji["indices"] = xr.DataArray(sindices[ic], dims="npts")
            obji["centroid"] = xr.DataArray(centroids[ic], dims="xy")
            obji["distorsion"] = xr.DataArray(sdistorsions[ic], dims="npts")
            obji.attrs["global_distorsion"] = global_distorsion
            obji.attrs["npmax"] = npmax
            out.append(obji)
        return out

    obj["iclust"] = xr.DataArray(indices, dims="npts")
    obj["indices"] = xr.DataArray(ii, dims="npts")
    obj["centroid"] = xr.DataArray(centroids, dims=("nclust", "xy"))
    obj["distorsion"] = xr.DataArray(distorsions, dims="npts")
    obj.attrs["global_distorsion"] = global_distorsion
    obj.attrs["npmax"] = npmax
    return obj


def deg2m(deg, lat=None, radius=EARTH_RADIUS):
    """Convert to meters a zonal or meridional distance in degrees

    Parameters
    ----------
    deg: float
        Longitude step
    lat: float
        Latitude for a zonal distance

    Return
    ------
    float
    """
    dd = deg * np.pi * radius / 180.0
    if lat:
        dd *= np.cos(np.radians(lat))
    return dd


def m2deg(met, lat=None, radius=EARTH_RADIUS):
    """Convert to degrees a zonal or meridional distance in meters

    Parameters
    ----------
    met: float
        Longitude step
    lat: float
        Latitude for a zonal distance

    Return
    ------
    float
    """
    dd = met * 180 / (np.pi * radius)
    if lat:
        dd /= np.cos(np.radians(lat))
    return dd
