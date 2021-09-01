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


class regrid1d_methods(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :func:`regrid1d` methods"""
    #: Linear iterpolation (default)
    linear = 1
    interp = 1  # compat
    #: Nearest iterpolation
    nearest = 0
    #: Cubic iterpolation
    cubic = 2
    #: Hermitian iterpolation
    hermit = 3
    hermitian = 3
    #: Cell-averaging or conservative regridding
    cellave = -1
    cellerr = -2


class extrap_modes(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported extrapolation modes"""
    #: No extrapolation (default)
    no = 0
    none = 0
    false = 0
    #: Toward the top (after)
    top = 1
    above = 1
    after = 1
    #: Toward the bottom (before)
    bottom = -1
    below = -1
    #: Both below and above
    both = 2
    all = 2
    yes = 2
    true = 2


def _wrapper1d_(vari, *args, func_name, **kwargs):
    """To make sure arrays have a 2D shape

    Output array is reshaped back accordindly
    """
    # To 2D
    eshape = vari.shape[:-1]
    args = [np.reshape(arr, (-1, arr.shape[-1])) for arr in (vari, ) + args]
    args = [np.asarray(arr) for arr in args]

    # Call
    func = getattr(interp, func_name)
    varo = func(*args, **kwargs)

    # From 2D
    return varo.reshape(eshape+varo.shape[-1:])


def regrid1d(
        da, coord, method=None, dim=None, coord_in_name=None,
        conserv=False, extrap="no", bias=0., tension=0.,
        dask='allowed'):
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
    conserv: bool
        Use conservative regridding when using ``cellave`` method.
    extrap: str, int
        Extrapolation mode as one of the following:
        {extrap_modes.rst_with_links}
    dask: str

    Returns
    -------
    xarray.DataArray
        Regridded array with ``coord`` as new coordinate array.

    See also
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
        coord_in = da.coords['coord_in_name']
    else:
        coord_in = cfspecs_in.search_coord_from_dim(da, dim_in, errors="raise")
        coord_in_name = coord_in.name

    # Coordinate arguments
    output_sizes = {dim_out: coord.sizes[dim_out]}
    input_core_dims = [[dim_in]]
    method = regrid1d_methods[method]
    if int(method) < 0:
        coord_in = xgrid.get_edges_1d(coord_in, axis=dim_in)
        coord = xgrid.get_edges_1d(coord, axis=dim_out)
        input_core_dims.extend([[dim_in+"_edges"], [dim_out+"_edges"]])
    else:
        input_core_dims.extend([[dim_in], [dim_out]])
    output_core_dims = [[dim_out]]

    # Fortran function name and arguments
    func_name = str(method) + "1d"
    if (not (coord_in.sizes[dim_in] == coord.sizes[dim_out] == 1) and
            method == regrid1d_methods.cellerr):
        raise XoaRegridError("cellerr regrid method is works only "
                             "with 1D input and output cordinates")
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
        exclude_dims={dim_in, dim_out},
        dask_gufunc_kwargs={"output_sizes": output_sizes},
        dask=dask
        )

    # Transpose
    dims = list(da.dims)
    dims[dims.index(dim_in)] = dim_out
    da_out = da_out.transpose(..., *dims, missing_dims="ignore")

    # Add output coordinates
    coord_out_name = coord.name if coord.name else coord_in.name
    da_out = da_out.assign_coords({coord_out_name: coord})

    return da_out


regrid1d.__doc__ = regrid1d.__doc__.format(**locals())


def extrap1d(da, dim, mode, **kwargs):
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
    kwargs: dict
        Extra arguments are passed to :func:`xarray.apply_ufunc`

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
        dask_gufunc_kwargs={"output_sizes": da.sizes},
        **kwargs
        )
    da_out = da_out.transpose(*da.dims)
    da_out = da_out.assign_coords(da.coords)
    da_out.attrs.update(da.attrs)
    da_out.encoding.update(da.encoding)
    return da_out


extrap1d.__doc__ = extrap1d.__doc__.format(**locals())


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

    See also
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
    xo = lons.values
    yo = lats.values
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
    glat = xcoords.get_lat(da) # before to_rect
    dims_in = set(glon.dims).union(glat.dims)
    da_tmp = xgrid.to_rect(da)
    da_tmp = xcoords.reorder(da_tmp, order)

    # To numpy with singletons
    # - data
    vi = da_tmp.data
    for axis_type, axis in (("z", -3), ("t", -4)):
        if axis_type not in order:
            vi = np.expand_dims(vi, axis)
    vi = vi.reshape((-1,)+vi.shape[-4:])
    # - xy
    glon = xcoords.get_lon(da_tmp)  # after to_rect
    glat = xcoords.get_lat(da_tmp)  # after to_rect
    xi = glon.values
    yi = glat.values
    coords_out = [lons, lats]
    if xi.ndim == 1:
        xi = xi.reshape(1, - 1)
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
        zo = np.zeros_like(lons.values)
    zi = zi.reshape((-1,)+zi.shape[-4:])
    # - t
    if "t" in order:
        # numeric times
        ti = (gtime.values - np.datetime64("1950-01-01", "us")) / np.timedelta64(1, "us")
        to = (times.values - np.datetime64("1950-01-01", "us")) / np.timedelta64(1, "us")
        dims_in.update(gtime.dims)
        coords_out.append(times)
    else:
        ti = np.zeros(1)
        to = np.zeros(lons.shape)

    # Interpolate
    vo = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)

    # As data array
    dims_out = [dim for dim in da.dims if dim not in dims_in]
    sizes_out = [size for dim, size in da.sizes.items() if dim in dims_out]
    dims_out.extend(loc.dims)
    sizes_out.append(lons.shape[-1])
    coords_out = coords_out + xcoords.get_coords_compat_with_dims(da, exclude_dims=dims_in)
    da_out = xr.DataArray(
        vo.reshape(sizes_out),
        dims=dims_out,
        coords=dict((coord.name, coord) for coord in coords_out),
        attrs=da.attrs,
        name=da.name
    )

    # Transpose
    da_out = xcoords.transpose(da_out, da.dims, mode="compat")

    return da_out
