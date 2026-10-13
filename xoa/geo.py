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
import numpy as np
import xarray as xr

from . import exceptions
from . import misc
from . import coords as xcoords
from .core import geo as _geo

#: Earth radius in meters
EARTH_RADIUS = _geo.EARTH_RADIUS


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
    return (
        _geo.haversine(np.double(lon0), np.double(lat0), np.double(lon1), np.double(lat1)) * radius
    )


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
    return _geo.bearing(np.double(lon0), np.double(lat0), np.double(lon1), np.double(lat1))


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
    scipy.spatial.distance.cdist
    scipy.spatial.distance.pdist
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
    scipy.spatial.distance.pdist
    scipy.spatial.distance.cdist
    scipy.spatial.distance.squareform
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
    """Context to switch the :func:`scipy.spatial.distance.cdist` function to :func:`cdist`

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
    if lat is not None:
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
    if lat is not None:
        dd /= np.cos(np.radians(lat))
    return dd


def _get_lonlat_(obj):
    """Get broadcasted longitudes and latitudes as a pair of data arrays"""
    if isinstance(obj, (tuple, list)):
        lon, lat = (np.asarray(o, dtype="d") for o in obj)
        if lon.ndim > 1:
            raise exceptions.XoaError("Longitudes and latitudes must be scalars or 1D arrays")
        dims = () if lon.ndim == 0 else ("target",)
        lon = xr.DataArray(lon, dims=dims)
        lat = xr.DataArray(lat, dims=dims)
        return lon, lat
    lon = xcoords.get_lon(obj)
    lat = xcoords.get_lat(obj)
    lat, lon = xr.broadcast(lat, lon)
    return lon, lat


def _check_output_(output):
    if output not in ("index", "dist", "mask"):
        raise exceptions.XoaError(f"Invalid output: {output!r}. Use 'index', 'dist' or 'mask'")


def _get_scale_(units):
    """Factor to convert meters to `units`"""
    return 1e-3 if distance_units[units] == distance_units.km else 1.0


def _scale_units_(scale):
    return "km" if scale == 1e-3 else "m"


class GeoKDTree:
    """A KD-tree of the points of a geographic object, to make several queries

    The tree is built once, from the longitudes and latitudes of `obj`, and used by
    :meth:`query` and :meth:`query_radius`. Distances are great circle distances.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D,
        defining a grid or a set of unstructured points.
    radius: float
        Radius of the sphere in meters, which defaults to the earth radius
    leafsize: int
        Number of points at which the tree switches to a brute force search

    See also
    --------
    get_neighbors
    xoa.core.geo.SphericalKDTree
    """

    def __init__(self, obj, radius=EARTH_RADIUS, leafsize=16):
        self.lon, self.lat = _get_lonlat_(obj)
        self.tree = _geo.SphericalKDTree(
            self.lon.values, self.lat.values, radius=radius, leafsize=leafsize
        )

    def _as_indices_(self, idx, dims, coords):
        """Convert flat indices to a dict of data arrays, with -1 for missing ones"""
        missing = idx < 0
        out = {}
        for dim, ii in zip(
            self.lon.dims, np.unravel_index(np.where(missing, 0, idx), self.lon.shape)
        ):
            out[dim] = xr.DataArray(np.where(missing, -1, ii), dims=dims, coords=coords)
        return out

    def _as_grid_(self, idx, dist, output, scale):
        """Get the distance to the closest target, or a mask, on the points of the tree"""
        idx = np.ravel(idx)
        dist = np.ravel(dist)
        valid = idx >= 0
        field = np.full(self.lon.size, np.inf)
        np.minimum.at(field, idx[valid], dist[valid])
        field[np.isinf(field)] = np.nan
        field = field.reshape(self.lon.shape)
        kwargs = dict(dims=self.lon.dims, coords=self.lon.coords)
        if output == "mask":
            return xr.DataArray(
                ~np.isnan(field),
                name="mask",
                attrs={"long_name": "Neighbors of targets"},
                **kwargs,
            )
        return xr.DataArray(
            field * scale,
            name="distance",
            attrs={"long_name": "Distance to the closest target", "units": _scale_units_(scale)},
            **kwargs,
        )

    def query(self, target, k=1, max_dist=None, units="m", output="index", workers=1):
        """Find the nearest neighbors of target points

        Parameters
        ----------
        target: xarray.DataArray, xarray.Dataset, tuple
            Object with longitude and latitude coordinates, or a (lon, lat) pair
            of scalars or 1D arrays
        k: int
            Number of neighbors
        max_dist: float, None
            Maximal distance
        units: int, str, distance_units
            Units of `max_dist` and of the returned distances, as one of:
            {distance_units.rst_with_links}
        output: {{"index", "dist", "mask"}}
            Return the distances and indices of the neighbors of each target,
            or a distance or mask array on the points of `obj`.
        workers: int
            Number of threads, -1 for all

        Return
        ------
        xarray.DataArray, dict
            With ``output="index"``, the distances, with the dimensions of the target and
            a trailing ``neighbor`` dimension if `k` is greater than 1, which are inf for
            missing neighbors. Then, indices as data arrays with the same dimensions,
            for each dimension of the longitudes and latitudes of `obj`,
            which are -1 for missing neighbors.
            They are suited for pointwise selection with :meth:`xarray.DataArray.isel`.
        xarray.DataArray
            With ``output="dist"``, the distance to the closest target on the points of `obj`
            that are neighbors of at least one target, and nan elsewhere.
            With ``output="mask"``, the boolean array of these points.
        """
        _check_output_(output)
        lon, lat = _get_lonlat_(target)
        scale = _get_scale_(units)
        if max_dist is not None:
            max_dist = max_dist / scale
        dist, idx = self.tree.query(lon.values, lat.values, k=k, max_dist=max_dist, workers=workers)
        if output != "index":
            return self._as_grid_(idx, dist, output, scale)
        dims = lon.dims
        if k > 1:
            dims = dims + ("neighbor",)
        shape = lon.shape + ((k,) if k > 1 else ())
        dist = xr.DataArray(
            dist.reshape(shape) * scale,
            dims=dims,
            coords=lon.coords,
            name="distance",
            attrs={"long_name": "Distance to neighbors", "units": _scale_units_(scale)},
        )
        return dist, self._as_indices_(idx.reshape(shape), dims, lon.coords)

    def query_radius(self, target, radius, units="m", output="index", workers=1):
        """Find the neighbors of target points within a distance

        Parameters
        ----------
        target: xarray.DataArray, xarray.Dataset, tuple
            Object with longitude and latitude coordinates, or a (lon, lat) pair
            of scalars or 1D arrays
        radius: float
            Maximal distance
        units: int, str, distance_units
            Units of `radius` and of the returned distances, as one of:
            {distance_units.rst_with_links}
        output: {{"index", "dist", "mask"}}
            Return the indices of the neighbors of each target,
            or a distance or mask array on the points of `obj`.
        workers: int
            Number of threads, -1 for all

        Return
        ------
        list
            With ``output="index"``, a list with one item per target point, in the
            flattened order, which is a dict of indices along a ``neighbor`` dimension
            for each dimension of the longitudes and latitudes of `obj`.
            They are suited for pointwise selection with :meth:`xarray.DataArray.isel`.
        xarray.DataArray
            With ``output="dist"``, the distance to the closest target on the points of `obj`
            that are within `radius` of at least one target, and nan elsewhere.
            With ``output="mask"``, the boolean array of these points.
        """
        _check_output_(output)
        lon, lat = _get_lonlat_(target)
        scale = _get_scale_(units)
        found = self.tree.query_ball_point(lon.values, lat.values, radius / scale, workers=workers)
        if output == "index":
            return [self._as_indices_(ff, "neighbor", None) for ff in found]
        idx = np.concatenate(found) if found else np.array([], dtype="i8")
        counts = [len(ff) for ff in found]
        dist = _geo.haversine(
            np.repeat(lon.values.ravel(), counts),
            np.repeat(lat.values.ravel(), counts),
            self.lon.values.ravel()[idx],
            self.lat.values.ravel()[idx],
        )
        return self._as_grid_(idx, dist * self.tree.radius, output, scale)


GeoKDTree.query.__doc__ = GeoKDTree.query.__doc__.format(**locals())
GeoKDTree.query_radius.__doc__ = GeoKDTree.query_radius.__doc__.format(**locals())


def _pop_tree_kwargs_(kwargs):
    """Pop the :class:`GeoKDTree` parameters from query ones, `radius` being `sphere_radius`"""
    tkw = {name: kwargs.pop(name) for name in ("leafsize",) if name in kwargs}
    if "sphere_radius" in kwargs:
        tkw["radius"] = kwargs.pop("sphere_radius")
    return tkw


def get_neighbors(obj, target, k=None, radius=None, units="m", output="index", **kwargs):
    """Find the nearest neighbors of target points, or those within a distance

    This builds a :class:`GeoKDTree` for a single query. Build it once instead
    to make several queries on the same points.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset, GeoKDTree
        Object with longitude and latitude coordinates, that may be 1D or 2D,
        or an already built tree
    target: xarray.DataArray, xarray.Dataset, tuple
        Object with longitude and latitude coordinates, or a (lon, lat) pair
    k: int, None
        Number of neighbors, see :meth:`GeoKDTree.query`
    radius: float, None
        Maximal distance, see :meth:`GeoKDTree.query_radius`.
        With `k`, it is the maximal distance of the neighbors.
    units: int, str, distance_units
        Distance units
    output: {"index", "dist", "mask"}
        Return the neighbors of each target, or a distance or mask array on the points of `obj`
    kwargs:
        ``sphere_radius`` and ``leafsize`` are passed to :class:`GeoKDTree`,
        and ``workers`` to the query.

    Return
    ------
    See :meth:`GeoKDTree.query` with `k`, :meth:`GeoKDTree.query_radius` otherwise.
    """
    if k is None and radius is None:
        raise exceptions.XoaError("Either k or radius must be given")
    tree = obj if isinstance(obj, GeoKDTree) else GeoKDTree(obj, **_pop_tree_kwargs_(kwargs))
    if k is not None:
        return tree.query(target, k=k, max_dist=radius, units=units, output=output, **kwargs)
    return tree.query_radius(target, radius, units=units, output=output, **kwargs)
