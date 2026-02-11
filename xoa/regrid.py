#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regridding utilities
"""
# Copyright 2020-2024 Shom
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

from . import exceptions
from . import misc
from . import meta
from . import grid as xgrid
from .core import num
from .core import regrid

# Backward compat
from .interp import grid2loc, isoslice  # noqa

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


def _wrapper1d_(vari, *args, func_name, **kwargs):
    """To make sure arrays have a 2D shape

    Output array is reshaped back accordindly
    """

    # The function to call
    func = getattr(regrid, func_name)

    # To 2D
    args = [vari] + list(args)
    eshapes = []
    for arr in args:
        eshape = list(arr.shape[:-1])
        if len(eshapes) and len(eshape) < len(eshapes[-1]):
            eshape = [1] * (len(eshapes[-1]) - len(eshape)) + eshape
        eshapes.append(eshape)
    eshapes = np.array(eshapes, dtype='l')
    args = [num.as_float_array(arr).reshape(-1, arr.shape[-1]) for arr in args]
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
    drop_na=False,
    maxgap=0,
    dask='parallelized',
):
    """Regrid along a single dimension

    The input and output coordinates may vary along other dimensions,
    which useful for instance for vertical interpolation in coastal
    ocean models.
    Since it uses func:`xarray.apply_ufunc`, it support dask features.
    The core computation is performed by the numba-accelerated routines
    of :mod:`xoa.core.regrid`.

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
    drop_na: bool
        Drop input inner NaNs during interpolation. Note that outer NaNs are
        always ignored. ``cellave`` and ``cellerr`` methods don't support the parameter.
    maxgap: int
        Max size for a gap to be interpolated when `drop_name is True.
        Size is not checked when `maxgap` is zero.
    dask: str
        See :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.DataArray
        Regridded array with ``coord`` as new coordinate array.

    See Also
    --------
    xoa.core.regrid.nearest1d
    xoa.core.regrid.linear2d
    xoa.core.regrid.cubic2d
    xoa.core.regrid.hermit1d
    xoa.core.regrid.cellave1d
    xoa.core.regrid.extrap1d
    xarray.apply_ufunc
    """
    # Get the working dimensions
    if not isinstance(dim, (tuple, list)):
        dim = (dim, dim)
    dim_in, dim_out = dim
    cfspecs_in = meta.get_meta_specs(da)
    cfspecs_out = meta.get_meta_specs(coord)
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
        raise exceptions.XoaRegridError(
            "cellerr regrid method works only with 1D input and output coordinates"
        )
    # func = getattr(interp, func_name)
    extrap = str(extrap_modes[extrap])
    func_kwargs = {"func_name": func_name, "extrap": extrap}
    if method == "hermit":
        func_kwargs.update(bias=bias, tension=tension)
    if drop_na:
        if method == "cellave" or method == "cellerr":
            raise exceptions.XoaRegridError(
                "cellerr and cellave regrid method still not support the drop_na paramater"
            )
        func_kwargs.update(drop_na=drop_na, maxgap=maxgap)

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
    xoa.regrid.extrap1d
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
