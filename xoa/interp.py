#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
High level interpolation routines.

Provides :func:`grid2loc` for interpolating gridded data to random
locations and :func:`isoslice` for extracting iso-surfaces.

.. note::
    This module also provides backward compatibility by re-importing
    core routines from the :mod:`xoa.core.interp` and
    :mod:`xoa.core.regrid` modules.
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
# import warnings

import numpy as np
import xarray as xr

from . import coords as xcoords
from . import grid as xgrid
from . import misc
from . import weights
from .core import num
from .core.interp import XYInterpolator as CoreXYInterpolator

# Backward compat
from .core.interp import (  # noqa
    closest2d,
    cell2relloc,
    grid2relloc,
    grid2rellocs,
    grid2locs,
    isoslice as core_isoslice,
)
from .core.regrid import (  # noqa
    nearest1d,
    linear1d,
    cubic1d,
    hermit1d,
    extrap1d,
    cellave1d,
)

# warnings.warn("The 'xoa.interp' module is deprecated in favour of the 'xoa.core.interp' module")


def grid2loc(da, loc, compat="warn"):
    """Interpolate a gridded data array to random locations

    ``da`` and ``loc`` must comply with CF conventions.

    Parameters
    ----------
    da: xarray.DataArray
        A data array with at least an horizontal rectilinear or
        a curvilinear grid.
    loc: xarray.Dataset, xarray.DataArray, pandas.DataFrame
        A dataset or data array with coordinates as 1d arrays
        that share the same dimension.
        For example, such dataset may be initialized as follows::

            loc = xr.Dataset(coords={
                'lon': ('npts', [5, 6]),
                'lat': ('npts', [4, 5]),
                'depth': ('npts',  [-10, -20])
                })

    compat: {"ignore", "warn"}
        In case a requested coordinate is not found in the input dataset.

    Return
    ------
    xarray.DataArray
        The interpolated data array, with the location dimension(s)
        replacing the grid dimensions.

    Example
    -------
    .. code-block:: python

        >>> vi = xr.DataArray(
        ...     np.ones((4, 5)), dims=("lat", "lon"),
        ...     coords=dict(lon=np.arange(5.), lat=np.arange(4.)))
        >>> loc = xr.Dataset(coords=dict(
        ...     lon=("npts", [1.5, 3.5]),
        ...     lat=("npts", [1.5, 2.5])))
        >>> grid2loc(vi, loc)

    See Also
    --------
    xoa.core.interp.grid2locs
    xoa.core.interp.grid2relloc
    xoa.core.interp.grid2rellocs
    xoa.core.interp.cell2relloc
    """

    # Ensure nanosecond precision for datetime coordinates
    da = xcoords.ensure_ns_datetime(da)
    if hasattr(loc, "to_xarray"):
        loc = loc.to_xarray()
    loc = xcoords.ensure_ns_datetime(loc)
    # - horizontal
    order = "yx"
    lons = xcoords.get_lon(loc)
    lats = xcoords.get_lat(loc)
    xo = np.atleast_1d(lons.values)
    yo = np.atleast_1d(lats.values)
    # - vertical
    deps = xcoords.get_vertical(loc, errors="ignore")
    if deps is not None:
        gdep = xcoords.get_vertical(da, errors=compat)
        if gdep is not None:
            order = "z" + order
    # - temporal
    times = xcoords.get_time(loc, errors="ignore")
    if times is not None:
        gtime = xcoords.get_time(da, errors=compat)
        if gtime is not None:
            order = "t" + order

    # Transpose following the tzyx order
    glon = xcoords.get_lon(da)  # before to_rect
    glat = xcoords.get_lat(da)  # before to_rect
    dims_in = set(glon.dims).union(glat.dims)
    da_tmp = xgrid.to_rect(da, errors="ignore")
    da_tmp = xcoords.reorder(da_tmp, order)

    # To numpy with singletons
    # - data
    vi = da_tmp.values
    for axis_type, axis in (("z", -3), ("t", -4)):
        if axis_type not in order:
            vi = np.expand_dims(vi, axis)
    vi = vi.reshape((-1,) + vi.shape[-4:])
    # - xy
    glon = xcoords.get_lon(da_tmp)  # after to_rect
    glat = xcoords.get_lat(da_tmp)  # after to_rect
    xi = glon.values
    yi = glat.values
    coords_out = [lons, lats]
    if xi.ndim == 1:
        xi = xi.reshape(1, -1)
    if yi.ndim == 1:
        yi = yi.reshape(-1, 1)
    # - z
    if "z" in order:
        gdep_order = xcoords.get_order(da_tmp[gdep.name])
        dims_in.update(gdep.dims)
        zi = da_tmp[gdep.name].values
        for axis_type, axis in (("x", -1), ("y", -2), ("t", -4)):
            if axis_type not in gdep_order:
                zi = np.expand_dims(zi, axis)
        zo = deps.data
        coords_out.append(deps)
    else:
        zi = np.zeros((1, 1, 1, 1))
        zo = np.zeros_like(xo)
    zi = zi.reshape((-1,) + zi.shape[-4:])
    # - t
    if "t" in order:
        # numeric times
        ti = num.as_float_array(gtime.values)
        to = num.as_float_array(times.values)
        to = np.atleast_1d(to)
        dims_in.update(gtime.dims)
        coords_out.append(times)
    else:
        ti = np.zeros(1)
        to = np.zeros(xo.shape)

    # Interpolate
    vo = grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)

    # As data array
    dims_out = [dim for dim in da.dims if dim not in dims_in]
    sizes_out = [size for dim, size in da.sizes.items() if dim in dims_out]
    dims_out.extend(loc.dims)
    sizes_out.extend(lons.shape)
    coords_out = coords_out + xcoords.get_coords_compat_with_dims(da, exclude_dims=dims_in)
    da_out = xr.DataArray(
        vo.reshape(sizes_out),
        dims=dims_out,
        coords=dict((coord.name, coord) for coord in coords_out),
        attrs=da.attrs,
        name=da.name,
    )

    # Transpose
    da_out = xcoords.transpose(da_out, da.dims, mode="compat")

    return da_out


# %% 2D


def isoslice(da, values, isoval, dim, reverse=False, dask='parallelized', **kwargs):
    """Extract a slice of ``da`` where ``values`` equals ``isoval``

    Interpolates ``da`` along ``dim`` at the position where ``values``
    crosses ``isoval``.

    Parameters
    -----------
    da: xarray.DataArray
        Array from which the data are extracted.
    values: xarray.DataArray
        Array in which ``isoval`` is searched for.
    isoval: float, xarray.DataArray
        Target value to locate in ``values``.
    dim: str
        Dimension shared by ``da`` and ``values`` along which the
        slice is performed.
    reverse: bool
        If True, search from the end of the ``dim`` axis instead
        of from the beginning.
    dask: str
        See :func:`xarray.apply_ufunc`.
    kwargs: dict
        Extra keyword arguments passed to :func:`xarray.apply_ufunc`.

    Return
    ------
    xarray.DataArray
        Sliced array with ``dim`` removed.

    Example
    -------
    Extract depth at a given temperature and temperature at a given depth::

        dep_at_t20 = isoslice(dep, temp, 20, "z")   # depth at temperature=20
        temp_at_z15 = isoslice(temp, dep, -15, "z")  # temperature at depth=-15m

    See Also
    --------
    xoa.core.interp.isoslice
    xarray.apply_ufunc
    """

    assert dim in da.dims
    assert dim in values.dims

    da_out = xr.apply_ufunc(
        core_isoslice,
        da,
        values,
        isoval,
        reverse,
        join="override",
        input_core_dims=[[dim], [dim], [], []],
        exclude_dims={dim},
        dask=dask,
        **kwargs,
    )
    da_out.attrs.update(da.attrs)
    da_out.encoding.update(da.encoding)
    return da_out


class xy_interp_methods(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :class:`Interpolator` methods"""

    #: Bilinear interpolation (default)
    bilinear = 1
    #: Bilinear interpolation (default)
    linear = 1
    #: Bicubic interpolation
    bicubic = 2
    #: Bicubic interpolation
    cubic = 2


#: Cache of the core interpolators, which hold the weights
_WEIGHTS_CACHE = misc.SmallCache(maxsize=8)


def clear_weights_cache():
    """Forget the weights that are shared by the :class:`Interpolator` of the same grids"""
    _WEIGHTS_CACHE.clear()


class Interpolator:
    """Interpolate from a source grid to arbitrary destination points

    The destination can be any shape: a single point, a transect, a 2D grid
    of scattered points, etc. Unlike :class:`~xoa.regrid.Regridder`,
    the conservative method is not available.

    Parameters
    ----------
    ds_src_grid: xarray.Dataset, xarray.DataArray
        Source grid with 1D or 2D longitude and latitude coordinates
    dst_lon, dst_lat: array_like, xarray.DataArray
        Destination coordinates, any shape, merged with :func:`xoa.coords.geo_merge`.
        The dimension names of data arrays are preserved on the output.
    method: str, int
        Interpolation method among {xy_interp_methods.rst_with_links}
    weights_file: str, None
        Path to a netcdf file that may hold the weights of many grids, each one in a group
        that is named after the method and a fingerprint of the grid and points.
        The weights are loaded from it if the group exists, and saved to it otherwise,
        after the first interpolation. See :mod:`xoa.weights`.
    src_mask: str, xarray.DataArray, array_like, None
        Valid points of the source grid
    bias, tension: float
        Kochanek-Bartels parameters for the bicubic method

    Attributes
    ----------
    src_fingerprint: str
        Fingerprint of the source grid, as given by :func:`xoa.grid.get_fingerprint`
    dst_fingerprint: str
        Fingerprint of the destination points
    fingerprint: str
        Fingerprint of the grid and the points
    weights_group: str
        Name of the group of the weights in a weights file. It is made of the method
        and the fingerprint, and :func:`xoa.weights.find_groups` finds it from the
        fingerprint of a grid.

    Notes
    -----
    The weights are computed once, when needed, and shared by all the interpolators that
    have the same source grid, the same destination points, the same method and the same
    parameters, as long as they are among the most recently used ones.
    :func:`clear_weights_cache` frees them.

    See also
    --------
    xoa.core.interp.XYInterpolator
    """

    def __init__(
        self,
        ds_src_grid,
        dst_lon,
        dst_lat,
        method="bilinear",
        weights_file=None,
        src_mask=None,
        bias=0.0,
        tension=0.0,
    ):
        try:
            method = str(xy_interp_methods[method])
        except (KeyError, ValueError):
            raise ValueError(f"Invalid method {method!r}. Choose among: {xy_interp_methods.rst}")

        self.ds_src_grid = ds_src_grid
        src_grid = xgrid.ds2grid_dict(ds_src_grid, src_mask)

        self._dst_lon_da, self._dst_lat_da = xcoords.geo_merge(dst_lon, dst_lat)
        self._dst_dims = self._dst_lon_da.dims

        # The core interpolator holds the weights, which are shared by the interpolators
        # of the same grids and points
        self.src_fingerprint = xgrid.get_fingerprint(src_grid)
        self.dst_fingerprint = misc.get_array_fingerprint(
            self._dst_lon_da.values, self._dst_lat_da.values
        )
        self.fingerprint = misc.combine_fingerprints(self.src_fingerprint, self.dst_fingerprint)
        self.weights_group = weights.get_group_name("interp", method, self.fingerprint)
        key = (method, float(bias), float(tension), src_grid["dims"], self.fingerprint)
        self.core_interp = _WEIGHTS_CACHE.get_or_create(
            key,
            lambda: CoreXYInterpolator(
                src_grid,
                self._dst_lon_da.values,
                self._dst_lat_da.values,
                method,
                bias=bias,
                tension=tension,
            ),
        )
        self.src_grid = self.core_interp.src_grid

        self.weights_file = weights_file
        if (
            weights_file
            and not self.core_interp.has_weights
            and (
                weights.has_group(weights_file, self.weights_group)
                or weights.is_legacy(weights_file)
            )
        ):
            self.load_weights(weights_file)

    def compute_weights(self, skipna=False):
        """Compute the fractional cell indices"""
        self.core_interp.compute_weights(skipna)

    def save_weights(self, weights_file):
        """Save the weights to a group of a netcdf file

        The group is named after the method and a fingerprint of the grid and points,
        and it is added to the file, which may hold the weights of other grids.
        Nothing is written if the group already exists.
        """
        ci = self.core_interp
        w = ci.get_weights()
        attrs = {
            "kind": "interp",
            "method": ci.method,
            "fingerprint": self.fingerprint,
            "src_fingerprint": self.src_fingerprint,
            "dst_fingerprint": self.dst_fingerprint,
            "n_dst": int(np.prod(ci.dst_shape)),
            "n_src": ci.src_grid["lat"].size,
        }
        variables = {
            "j_base": ("n_dst", w["j_base"]),
            "i_base": ("n_dst", w["i_base"]),
            "frac_a": ("n_dst", w["frac_a"]),
            "frac_b": ("n_dst", w["frac_b"]),
            "valid_dst_mask": ("n_dst", w["valid_dst_mask"]),
        }
        weights.save_group(weights_file, self.weights_group, variables, attrs)

    def load_weights(self, weights_file):
        """Load the weights of this grid and these points from a netcdf file

        The group that matches the fingerprint of the grids and the method is searched.
        A file in the legacy format, that holds the weights of a single grid without
        fingerprint, is still read, but only the sizes of the grids can be checked.

        Raises
        ------
        ValueError
            When the file has no weights for this grid, these points and this method
        """
        ci = self.core_interp
        n_dst = int(np.prod(ci.dst_shape))
        n_src = ci.src_grid["lat"].size
        ds = weights.select_group(
            weights_file,
            self.weights_group,
            self.fingerprint,
            ci.method,
            n_dst,
            n_src,
            "this grid, these points",
        )
        ci.set_weights(
            {
                "j_base": ds["j_base"].values.astype(np.int64, copy=False),
                "i_base": ds["i_base"].values.astype(np.int64, copy=False),
                "frac_a": ds["frac_a"].values.astype(np.float64, copy=False),
                "frac_b": ds["frac_b"].values.astype(np.float64, copy=False),
                "valid_dst_mask": ds["valid_dst_mask"].values,
            }
        )

    def _assign_dst_coords_(self, obj, extra=None):
        coords = {
            self._dst_lon_da.name: self._dst_lon_da,
            self._dst_lat_da.name: self._dst_lat_da,
        }
        if extra:
            coords.update(extra)
        return obj.assign_coords(coords)

    def interp(self, src_ds, skipna=False, na_thres=1.0):
        """Interpolate a source dataset or data array to the destination points

        Parameters
        ----------
        src_ds: xarray.Dataset, xarray.DataArray
            Source data with horizontal dimensions matching the source grid
        skipna: bool
        na_thres: float

        Return
        ------
        xarray.Dataset, xarray.DataArray
            The horizontal dimensions are replaced by the destination ones.
            Other dimensions are preserved.
        """
        if not self.core_interp.has_weights:
            self.compute_weights()

        output_sizes = dict(zip(self._dst_dims, self.core_interp.dst_shape))
        result = xr.apply_ufunc(
            lambda arr, **kw: self.core_interp.interp(np.asarray(arr), **kw),
            src_ds,
            input_core_dims=[list(self.src_grid["dims"])],
            output_core_dims=[list(self._dst_dims)],
            exclude_dims=set(self.src_grid["dims"]),
            kwargs={"skipna": skipna, "na_thres": na_thres},
            vectorize=False,
            dask="allowed",
            keep_attrs=False,
            on_missing_core_dim="copy",
            output_dtypes=[np.float64],
            output_sizes=output_sizes,
        )

        if self.weights_file and not weights.has_group(self.weights_file, self.weights_group):
            self.save_weights(self.weights_file)

        return self._assign_dst_coords_(result)

    def interp_with_time(self, src_ds, dst_times, skipna=False, na_thres=1.0, time_method=1):
        """Interpolate to scattered (lon, lat, time) locations

        Parameters
        ----------
        src_ds: xarray.Dataset, xarray.DataArray
            Source data with a time dimension and horizontal dimensions matching
            the source grid
        dst_times: array_like, xarray.DataArray
            Destination times, with the same shape as the destination coordinates
        skipna: bool
        na_thres: float
        time_method: int
            Temporal interpolation: 1 for linear

        Return
        ------
        xarray.Dataset, xarray.DataArray
            The time and horizontal dimensions are consumed.
            Other dimensions and destination coordinates are preserved.
        """
        if not self.core_interp.has_weights:
            self.compute_weights()

        return_da = isinstance(src_ds, xr.DataArray)
        if return_da:
            orig_name = src_ds.name
            da_name = orig_name or "_data"
            src_ds = src_ds.to_dataset(name=da_name)

        src_time = xcoords.get_time(src_ds)
        src_times_f64 = num.as_float_array(src_time.values)
        time_dim = src_time.dims[0]
        dst_times_f64 = num.as_float_array(dst_times).ravel()

        ci = self.core_interp
        src_sdims = list(ci.src_grid["dims"])

        result_vars = {}
        for name, da in src_ds.data_vars.items():
            if time_dim not in da.dims or not set(src_sdims).issubset(da.dims):
                continue
            extra_dims = [d for d in da.dims if d != time_dim and d not in src_sdims]
            ordered = [time_dim] + extra_dims + src_sdims
            data = np.ascontiguousarray(da.transpose(*ordered).values, dtype=np.float64)
            result = ci.interp_with_time(
                data,
                src_times_f64,
                dst_times_f64,
                skipna=skipna,
                na_thres=na_thres,
                time_method=time_method,
            )
            result_vars[name] = xr.DataArray(
                result, dims=extra_dims + list(self._dst_dims), attrs=da.attrs
            )

        extra = None
        if isinstance(dst_times, xr.DataArray) and dst_times.name:
            extra = {dst_times.name: dst_times}
        result_ds = self._assign_dst_coords_(xr.Dataset(result_vars), extra)
        if return_da:
            result = result_ds[da_name]
            result.name = orig_name
            return result
        return result_ds


Interpolator.__doc__ = Interpolator.__doc__.format(**locals())


def interpxy(
    da,
    dst_lon=None,
    dst_lat=None,
    method="bilinear",
    skipna=False,
    na_thres=1.0,
    interpolator=None,
    **kwargs,
):
    """Interpolate horizontally to arbitrary destination points

    It is a shortcut to :class:`Interpolator`.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        Source data with longitude and latitude coordinates
    dst_lon, dst_lat: array_like, xarray.DataArray, None
        Destination coordinates, any shape. See :func:`xoa.coords.geo_merge`.
        They are not needed when an ``interpolator`` is given.
    method: str, int
        Interpolation method, see :class:`xy_interp_methods`
    skipna: bool
        Skip NaN values
    na_thres: float
        Threshold for NaN handling
    interpolator: Interpolator, None
        Existing interpolator to use, to avoid initializing it again in a loop.
        Its source grid must be the one of ``da``. ``dst_lon``, ``dst_lat``,
        ``method`` and ``kwargs`` are then ignored, but ``src_mask`` must be provided again
        if it was used to create it.
    kwargs:
        Extra parameters are passed to :class:`Interpolator`, like ``weights_file``,
        ``src_mask``, ``bias`` or ``tension``.
        When ``weights_file`` is provided, the weights are loaded from it if they
        are in, and saved to it otherwise.

    Return
    ------
    xarray.DataArray, xarray.Dataset

    Notes
    -----
    The weights are shared by the calls that use the same grid, points, method and
    parameters, but the grids must be fingerprinted at each call. Create an
    :class:`Interpolator` once and pass it as ``interpolator`` when interpolating
    many variables or time steps.

    See also
    --------
    Interpolator
    xoa.regrid.regridxy
    """
    if interpolator is None:
        if dst_lon is None or dst_lat is None:
            raise ValueError("Destination points or an interpolator are needed")
        interpolator = Interpolator(da, dst_lon, dst_lat, method=method, **kwargs)
    else:
        src_grid = xgrid.ds2grid_dict(da, kwargs.get("src_mask"))
        fingerprint = xgrid.get_fingerprint(src_grid)
        if fingerprint != interpolator.src_fingerprint:
            raise ValueError(
                "The source grid of the data does not match the one of the interpolator: "
                f"{fingerprint} instead of {interpolator.src_fingerprint}."
            )
    return interpolator.interp(da, skipna=skipna, na_thres=na_thres)
