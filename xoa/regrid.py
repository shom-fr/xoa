#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regridding utilities
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

import numpy as np
import xarray as xr

from .__init__ import XoaError
from . import misc
from . import cf as xcf
from . import coords as xcoords
from . import grid as xgrid
from . import interp


class XoaRegridError(XoaError):
    pass


# %% 1D


class regrid1d_methods(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :func:`regrid1d` methods"""

    #: Linear interpolation (default)
    linear = 1
    #: Linear iterpolation (default)
    interp = 1  # compat
    #: Nearest interpolation
    nearest = 0
    #: Cubic interpolation
    cubic = 2
    #: Hermitian interpolation
    hermit = 3
    #: Hermitian iterpolation
    hermitian = 3
    #: Cell-averaging or conservative regridding
    cellave = -1
    #: Cell-averaging or conservative regridding
    cellerr = -2


class extrap_modes(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported extrapolation modes"""

    #: No extrapolation (default)
    no = 0
    #: No extrapolation (default)
    none = 0
    #: No extrapolation (default)
    false = 0
    #: Toward the top (after)
    top = 1
    #: Toward the top (after)
    above = 1
    #: Toward the top (after)
    after = 1
    #: Toward the bottom (before)
    bottom = -1
    #: Toward the bottom (before)
    below = -1
    #: Both below and above
    both = 2
    #: Both below and above
    all = 2
    #: Both below and above
    yes = 2
    #: Both below and above
    true = 2


def _asfloat_(arr):
    """We need arrays that are at least 1D and that are not dates, booleans or integers"""
    arr = np.asarray(arr)
    arr = np.atleast_1d(arr)
    if arr.dtype.type is np.datetime64:
        arr = (arr - np.datetime64("1950-01-01", "us")) / np.timedelta64(1, "us")
    elif arr.dtype.char in 'il?':
        arr = arr.astype("d")
    return arr


def _wrapper1d_(vari, *args, func_name, **kwargs):
    """To make sure arrays have a 2D shape

    Output array is reshaped back accordindly
    """

    # The function to call
    func = getattr(interp, func_name)

    # To 2D
    args = [vari] + list(args)
    eshapes = []
    for arr in args:
        eshape = list(arr.shape[:-1])
        if len(eshapes) and len(eshape) < len(eshapes[-1]):
            eshape = [1] * (len(eshapes[-1]) - len(eshape)) + eshape
        eshapes.append(eshape)
    eshapes = np.array(eshapes, dtype='l')
    args = [_asfloat_(arr).reshape(-1, arr.shape[-1]) for arr in args]
    func_code = getattr(func, "func_code", getattr(func, "__code__"))
    if "eshapes" in func_code.co_varnames[: func_code.co_argcount]:
        args = args + [eshapes]

    # Call
    varo = func(*args, **kwargs)

    # From 2D
    return varo.reshape(tuple(eshapes.max(axis=0)) + varo.shape[-1:])


def regrid1d(
    da,
    coord,
    method=None,
    dim=None,
    coord_in_name=None,
    edges=None,
    conserv=False,
    extrap="no",
    bias=0.0,
    tension=0.0,
    dask='parallelized',
):
    """Regrid along a single dimension

    The input and output coordinates may vary along other dimensions,
    which useful for instance for vertical interpolation in coastal
    ocean models.
    Since it uses func:`xarray.apply_ufunc`, it support dask features.
    The core computation is performed by the numba-accelerated routines
    of :mod:`xoa.interp`.

    Parameters
    ----------
    da: xarray.DataArray
        Array to interpolate
    coord: xarray.DataArray
        Output coordinate
    method: str, int
        Regridding method as one of the following:
        {regrid1d_methods.rst_with_links}
    dim:  str, tuple(str), None
        Dimension on which to operate. If a string, it is expected to
        be the same dimension for both input and output coordinates.
        Else, provide a two-element tuple: ``(dim_in, dim_out)``.
        It is inferred by default from output coordinate et input data array.
    coord_in_name: str, None
        Name of the input coordinate array, which must be known of ``da``.
        It is inferred from the input dara array and dimension name
        by default.
    edges: dict, None
        Grid edge coordinates along the interpolation dimension,
        for the conservative regridding.
        When not provided, edges are computed with :func:`xoa.grid.get_edges`.
        Keys are `"in"` and/or `"out"` and values are arrays with the same shape as
        coordinates except along the interpolation dimension on which 1 is added.
    conserv: bool
        Use conservative regridding when using ``cellave`` method.
    extrap: str, int
        Extrapolation mode as one of the following:
        {extrap_modes.rst_with_links}
    dask: str
        See :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.DataArray
        Regridded array with ``coord`` as new coordinate array.

    See Also
    --------
    xoa.interp.nearest1d
    xoa.interp.linear2d
    xoa.interp.cubic2d
    xoa.interp.hermit1d
    xoa.interp.cellave1d
    xoa.interp.extrap1d
    xarray.apply_ufunc
    """
    # Get the working dimensions
    if not isinstance(dim, (tuple, list)):
        dim = (dim, dim)
    dim_in, dim_out = dim
    cfspecs_in = xcf.get_cf_specs(da)
    cfspecs_out = xcf.get_cf_specs(coord)
    # - dim out
    if dim_out is None:  # get dim_out from coord_out
        dim_dict = cfspecs_out.search_dim(coord, errors="raise")
        dim_out = dim_dict["dim"]
        dim_type = dim_dict["type"]
    else:  # dim_out is provided
        dim_type = cfspecs_out.coords.get_dim_type(dim_out, coord)
    # - dim in
    if dim_in is None:
        if dim_type:
            dim_in = cfspecs_in.coords.search_dim(da, dim_type, errors="raise")
        else:
            dim_in = dim_out  # be cafeful, dim1 must be in input!
    assert dim_in in da.dims
    assert dim_out in coord.dims

    # Input coordinate
    if coord_in_name:
        assert coord_in_name in da.coords, 'Invalid coordinate'
        coord_in = da.coords[coord_in_name]
    else:
        coord_in = cfspecs_in.search_coord_from_dim(da, dim_in, errors="raise")
        coord_in_name = coord_in.name

    # Coordinate arguments
    output_sizes = {dim_out: coord.sizes[dim_out]}
    input_core_dims = [[dim_in]]
    method = regrid1d_methods[method]
    coord_out = coord
    exclude_dims = {dim_in, dim_out}
    if int(method) < 0:
        idimin = coord_in.get_axis_num(dim_in)
        idimout = coord.get_axis_num(dim_out)
        if edges and "in" in edges:
            coord_in = edges["in"]
        else:
            coord_in = xgrid.get_edges(coord_in, dim_in)
        if edges and "out" in edges:
            coord = edges["out"]
        else:
            coord = xgrid.get_edges(coord, dim_out)
        namein = coord_in.dims[idimin]
        nameout = coord.dims[idimout]
        input_core_dims.extend([[namein], [nameout]])
        exclude_dims = {dim_in, dim_out, namein, nameout}
    else:
        exclude_dims = {dim_in, dim_out}
        input_core_dims.extend([[dim_in], [dim_out]])
    output_core_dims = [[dim_out]]
    for cname in coord.coords:
        if cname != coord.name:
            coord = coord.drop(cname)

    # Interpolation function name and arguments
    func_name = str(method) + "1d"
    if method == regrid1d_methods.cellerr and not (coord_in.ndim == coord.ndim == 1):
        raise XoaRegridError(
            "cellerr regrid method works only with 1D input and output coordinates"
        )
    # func = getattr(interp, func_name)
    extrap = str(extrap_modes[extrap])
    func_kwargs = {"func_name": func_name, "extrap": extrap}
    if method == "hermit":
        func_kwargs.update(bias=bias, tension=tension)

    # Apply
    da_out = xr.apply_ufunc(
        _wrapper1d_,
        da,
        coord_in,
        coord,
        join="override",
        kwargs=func_kwargs,
        input_core_dims=input_core_dims,
        output_core_dims=output_core_dims,
        exclude_dims=exclude_dims,
        dask_gufunc_kwargs={"output_sizes": output_sizes},
        dask=dask,
    )

    # Transpose
    dims = list(da.dims)
    dims[dims.index(dim_in)] = dim_out
    da_out = da_out.transpose(..., *dims, missing_dims="ignore")

    # Add output coordinates
    coord_out_name = coord_out.name if coord_out.name else coord_in.name
    for cname in coord_out.coords:
        if cname != coord_out.name:
            coord_out = coord_out.drop(cname)
    da_out = da_out.assign_coords({coord_out_name: coord_out})
    da_out.name = da.name
    da_out.attrs = da.attrs

    return da_out


regrid1d.__doc__ = regrid1d.__doc__.format(**locals())


def extrap1d(da, dim, mode, dask='parallelized'):
    """Extrapolate along a single dimension


    Parameters
    ----------
    da: xarray.DataArray
        Array to interpolate
    dim:  str
        Dimension on which to operate.
    mode: str, int
        Extrapolation mode as one of the following:
        {extrap_modes.rst_with_links}
    dask: str
        See :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.DataArray
        Extrapolated array.

    See also
    --------
    xoa.interp.extrap1d
    xarray.apply_ufunc
    """
    da_out = xr.apply_ufunc(
        _wrapper1d_,
        da,
        join="override",
        kwargs={"func_name": "extrap1d", "mode": str(extrap_modes[mode])},
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        exclude_dims={dim},
        dask=dask,
        dask_gufunc_kwargs={"output_sizes": da.sizes},
    )
    da_out = da_out.transpose(*da.dims)
    da_out = da_out.assign_coords(da.coords)
    da_out.attrs.update(da.attrs)
    da_out.encoding.update(da.encoding)
    return da_out


extrap1d.__doc__ = extrap1d.__doc__.format(**locals())


def isoslice(da, values, isoval, dim, reverse=False, dask='parallelized', **kwargs):
    """Extract data from var where values==isoval

    Parameters
    -----------
    da: xarray.DataArray
          array from which the data are extracted
    values: array_like
          array on which a research of isoval is made
    isoval: float
          value of interest on which we perform research in values array
    dim: str
          dimension shared by da and values on which the slice is made
    dask: str
        See :func:`xarray.apply_ufunc`.

    Return
    ------
    isovar : array_like
            Sliced array based on data where values==isoval

    Example
    -------

    Let's define depth and temperature variables both in 3 dimensions (i,j,k)
    where i and j are horizontal dimension and k the vertical one::

        dep_at_t20 = isoslice(dep, temp, 20, "z")   # depth at temperature=20°C
        temp_at_z15 = isoslice(temp, dep, -15, "z") # temperature at depth=-15m

    See Also
    --------
    xoa.interp.isoslice
    xarray.apply_ufunc
    """

    assert dim in da.dims
    assert dim in values.dims

    da_out = xr.apply_ufunc(
        interp.isoslice,
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


# %% 2D


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
    xarray.dataArray
        The interpolated data array.

    See Also
    --------
    xoa.interp.grid2locs
    xoa.interp.grid2relloc
    xoa.interp.grid2rellocs
    xoa.interp.cell2relloc
    """

    # Get coordinates
    if hasattr(loc, "to_xarray"):
        loc = loc.to_xarray()
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
    da_tmp = xgrid.to_rect(da)
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
        ti = _asfloat_(gtime.values)
        to = _asfloat_(times.values)
        to = np.atleast_1d(to)
        dims_in.update(gtime.dims)
        coords_out.append(times)
    else:
        ti = np.zeros(1)
        to = np.zeros(xo.shape)

    # Interpolate
    vo = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)

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
