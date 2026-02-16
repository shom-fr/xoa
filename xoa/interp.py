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
from .core import num

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
