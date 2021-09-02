# -*- coding: utf-8 -*-
"""
Coordinates and dimensions utilities
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

from collections.abc import Mapping

import xarray as xr

from .__init__ import XoaError, xoa_warn
from . import misc
from . import cf as xcf


@misc.ERRORS.format_function_docstring
def get_lon(da, errors="raise"):
    """Get the longitude coordinate

    Parameters
    ----------
    da: xarray.DataArray
    {errors}

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lat
    get_depth
    get_altitude
    get_level
    get_vertical
    get_time
    xoa.cf.CFSpecs.search_coord
    """
    return xcf.get_cf_specs(da).search(da, 'lon', errors=errors)


def is_lon(da, loc="any"):
    """Tell if a data array is identified as longitudes

    Parameters    is_vertical

    da: xarray.DataArray

    Return
    ------
    bool

    See also
    --------
    is_lat
    is_depth
    is_altitude
    is_level
    is_time
    xoa.cf.CFCoordSpecs.match
    """
    return xcf.get_cf_specs(da).coords.match(da, "lon", loc=loc)


@misc.ERRORS.format_function_docstring
def get_lat(da, errors="raise"):
    """Get the latitude coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lon
    get_depth
    get_altitude
    get_level
    get_vertical
    get_time
    xoa.cf.CFSpecs.search_coord
    """
    return xcf.get_cf_specs(da).search(da, 'lat', errors=errors)


def is_lat(da, loc="any"):
    """Tell if a data array is identified as latitudes

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    bool

    See also
    --------
    is_lon
    is_depth
    is_altitude
    is_level
    is_time
    xoa.cf.CFCoordSpecs.match
    """
    return xcf.get_cf_specs(da).coords.match(da, "lat", loc=loc)


@misc.ERRORS.format_function_docstring
def get_depth(da, errors="raise"):
    """Get or compute the depth coordinate

    If a depth variable cannot be found, it tries to compute either
    from sigma-like coordinates or from layer thinknesses.

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lon
    get_lat
    get_time
    get_altitude
    get_level
    get_vertical
    xoa.cf.CFSpecs.search_coord
    xoa.sigma.decode_cf_sigma
    xoa.grid.decode_cf_dz2depth
    """
    cfspecs = xcf.get_cf_specs(da)
    errors = misc.ERRORS[errors]
    ztype = cfspecs["vertical"]["type"]

    # From variable
    depth = cfspecs.search(da, 'depth', errors="ignore")
    if depth is not None:
        return depth
    if ztype == "z" or not hasattr(da, "data_vars"):  # explicitly
        msg = "No depth coordinate found"
        if errors == "raise":
            raise XoaError(msg)
        xoa_warn(msg)
        return

    # Decode the dataset
    if ztype == "sigma" or ztype is None:
        err = "ignore" if ztype is None else errors
        from .sigma import decode_cf_sigma
        da = decode_cf_sigma(da, errors=err)
        if "depth" in da:
            return da.depth
    if ztype == "dz2depth" or ztype is None:
        err = "ignore" if ztype is None else errors
        from .grid import decode_cf_dz2depth
        da = decode_cf_dz2depth(da, errors=err)
        if "depth" in da:
            return da.depth
    msg = "Can't infer depth coordinate from dataset"
    if errors == "raise":
        raise XoaError(msg)
    xoa_warn(msg)


def is_depth(da, loc="any"):
    """Tell if a data array is identified as depths

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    bool

    See also
    --------
    is_lon
    is_lat
    is_altitude
    is_level
    is_time
    xoa.cf.CFCoordSpecs.match
    """
    return xcf.get_cf_specs(da).coords.match(da, "depth", loc=loc)


@misc.ERRORS.format_function_docstring
def get_altitude(da, errors="raise"):
    """Get the altitude coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lon
    get_lat
    get_depth
    get_level
    get_vertical
    get_time
    xoa.cf.CFSpecs.search_coord
    """
    return xcf.get_cf_specs(da).search(da, 'altitude', errors=errors)


def is_altitude(da, loc="any"):
    """Tell if a data array is identified as altitudes

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    bool

    See also
    --------
    is_lon
    is_lat
    is_depth
    is_level
    is_time
    xoa.cf.CFCoordSpecs.match
    """
    return xcf.get_cf_specs(da).coords.match(da, "altitude", loc=loc)


@misc.ERRORS.format_function_docstring
def get_level(da, errors="raise"):
    """Get the level coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lon
    get_lat
    get_depth
    get_altitude
    get_time
    xoa.cf.CFSpecs.search_coord
    """
    return xcf.get_cf_specs(da).coords.search(da, 'level', errors=errors)


def is_level(da, loc="any"):
    """Tell if a data array is identified as levels

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    bool

    See also
    --------
    is_lon
    is_lat
    is_depth
    is_altitude
    is_time
    xoa.cf.CFCoordSpecs.match
    """
    return xcf.get_cf_specs(da).coords.match(da, "levels", loc=loc)


@misc.ERRORS.format_function_docstring
def get_vertical(da, errors="raise"):
    """Get either depth or altitude

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lon
    get_lat
    get_depth
    get_altitude
    get_level
    get_time
    xoa.cf.CFSpecs.search_coord
    """
    cfspecs = xcf.get_cf_specs()
    height = cfspecs.search(da, 'depth', errors="ignore")
    if height is None:
        height = cfspecs.search(da, 'altitude', errors="ignore")
    if height is None:
        errors = misc.ERRORS[errors]
        msg = "No vertical coordinate found"
        if errors == "raise":
            raise xcf.XoaCFError(msg)
        elif errors == "warn":
            xoa_warn(msg)
    else:
        return height


@misc.ERRORS.format_function_docstring
def get_time(da, errors="raise"):
    """Get the time coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lon
    get_lat
    get_depth
    get_altitude
    get_level
    get_vertical
    xoa.cf.CFSpecs.search_coord
    """
    return xcf.get_cf_specs(da).coords.search(da, 'time', errors=errors)


def is_time(da):
    """Tell if a data array is identified as time

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    bool

    See also
    --------
    is_lon
    is_lat
    is_depth
    is_altitude
    is_level
    xoa.cf.CFCoordSpecs.match
    """
    return xcf.get_cf_specs(da).match(da, "time")


@misc.ERRORS.format_function_docstring
def get_cf_coords(da, coord_names, errors="raise"):
    """Get several coordinates

    Parameters
    ----------
    {errors}

    Return
    ------
    list(xarray.DataArray)

    See also
    --------
    xoa.cf.CFSpecs.search_coord
    """
    cfspecs = xcf.get_cf_specs(da)
    return [cfspecs.search_coord(da, coord_name, errors=errors)
            for coord_name in coord_names]


@misc.ERRORS.format_function_docstring
def get_cf_dims(da, cf_args, allow_positional=False, positions='tzyx', errors="warn"):
    """Get the data array dimensions names from their type

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    cf_args: str, list
        Letters among "x", "y", "z", "t" and "f",
        or generic CF names.
    allow_positional: bool
        Fall back to positional dimension of types if unkown.
    positions: str
        Default position per type starting from the end.
    {errors}

    Return
    ------
    tuple
        Tuple of dimension name or None when the dimension if not found

    See also
    --------
    xoa.cf.CFSpecs.get_dims
    """
    return xcf.get_cf_specs(da).get_dims(
        da, cf_args, allow_positional=allow_positional,
        positions=positions, errors=errors)


@misc.ERRORS.format_function_docstring
def get_xdim(da, errors="warn", **kwargs):
    """Get the dimension of X type

    It is a simple call to :func:`get_dims` with ``dim_types="x"``

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    positions: str
        Default position per type starting from the end.
    {errors}
    kwargs: dict
        Extra parameters are passed to :func:`get_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_dims
    """
    dims = get_cf_dims(da, "x", errors=errors)
    if dims:
        return dims[0]


@misc.ERRORS.format_function_docstring
def get_ydim(da, errors="warn", **kwargs):
    """Get the dimension of Y type

    It is a simple call to :func:`get_dims` with ``dim_types="y"``

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    positions: str
        Default position per type starting from the end.
    {errors}
    kwargs: dict
        Extra parameters are passed to :func:`get_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_dims
    """
    dims = get_cf_dims(da, "y", errors=errors)
    if dims:
        return dims[0]


@misc.ERRORS.format_function_docstring
def get_zdim(da, errors="warn", **kwargs):
    """Get the dimension of Z type

    It is a simple call to :func:`get_dims` with ``dim_types="z"``

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    positions: str
        Default position per type starting from the end.
    {errors}
    kwargs: dict
        Extra parameters are passed to :func:`get_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_dims
    """
    dims = get_cf_dims(da, "z", errors=errors)
    if dims:
        return dims[0]


@misc.ERRORS.format_function_docstring
def get_tdim(da, errors="warn", **kwargs):
    """Get the dimension of T type

    It is a simple call to :func:`get_dims` with ``dim_types="t"``

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    positions: str
        Default position per type starting from the end.
    {errors}
    kwargs: dict
        Extra parameters are passed to :func:`get_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_dims
    """
    dims = get_cf_dims(da, "t", errors=errors)
    if dims:
        return dims[0]


@misc.ERRORS.format_function_docstring
def get_fdim(da, errors="warn", **kwargs):
    """Get the dimension of F type

    It is a simple call to :func:`get_dims` with ``dim_types="f"``

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    positions: str
        Default position per type starting from the end.
    {errors}
    kwargs: dict
        Extra parameters are passed to :func:`get_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_dims
    """
    dims = get_cf_dims(da, "f", errors=errors)
    if dims:
        return dims[0]


class transpose_modes(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :func:`transpose` modes"""
    #: Basic xarray transpose with :meth:`xarray.DataArray.transpose`
    classic = 0
    basic = 0
    xarray = 0
    #: Transpose skipping incompatible dimensions
    compat = -1
    #: Transpose adding missing dimensions with a size of 1
    insert = 1
    #: Transpose resizing to missing dimensions.
    #: Note that dims must be an array or a dict of sizes
    #: otherwise new dimensions will have a size of 1.
    resize = 2


def transpose(da, dims, mode='compat'):
    """Transpose an array

    Parameters
    ----------
    da: xarray.DataArray
        Array to tranpose
    dims: tuple(str), xarray.DataArray, dict
        Target dimensions or array with dimensions
    mode: str, int
        Transpose mode as one of the following:
        {transpose_modes.rst_with_links}

    Return
    ------
    xarray.DataArray
        Transposed array

    Example
    -------
    .. ipython:: python

        @suppress
        import xarray as xr, numpy as np
        @suppress
        from xoa.coords import transpose
        a = xr.DataArray(np.ones((2, 3, 4)), dims=('y', 'x', 't'))
        b = xr.DataArray(np.ones((10, 3, 2)), dims=('m', 'y', 'x'))
        # classic
        transpose(a, (Ellipsis, 'y', 'x'), mode='classic')
        # insert
        transpose(a, ('m', 'y', 'x', 'z'), mode='insert')
        transpose(a, b, mode='insert')
        # resize
        transpose(a, b, mode='resize')
        transpose(a, b.sizes, mode='resize') # with dict
        # compat mode
        transpose(a, ('y', 'x'), mode='compat').dims
        transpose(a, b.dims, mode='compat').dims
        transpose(a, b, mode='compat').dims  # same as with b.dims

    See also
    --------
    xarray.DataArray.transpose
    """
    # Inits
    if hasattr(dims, 'dims'):
        sizes = dims.sizes
        dims = dims.dims
    elif isinstance(dims, Mapping):
        sizes = dims
        dims = list(dims.keys())
    else:
        sizes = None
    mode = str(transpose_modes[mode])
    kw = dict(transpose_coords=True)

    # Classic
    if mode == "classic":
        return da.transpose(*dims, **kw)

    # Get specs
    odims = ()
    expand_dims = {}
    with_ell = False
    for dim in dims:
        if dim is Ellipsis:
            with_ell = True
            odims += dim,
        elif dim in da.dims:
            odims += dim,
        elif mode == "insert":
            expand_dims[dim] = 1
            odims += dim,
        elif mode == "resize":
            if sizes is None or dim not in sizes:
                xoa_warn(f"new dim '{dim}' in transposition is set to one"
                         " since no size is provided to it")
                size = 1
            else:
                size = sizes[dim]
            expand_dims[dim] = size
            odims += dim,

    # Expand
    if expand_dims:
        da = da.expand_dims(expand_dims)

    # Input dimensions that were not specified in transposition
    # are flushed to the left
    if not with_ell and set(odims) < set(da.dims):
        odims = (...,) + odims

    # Transpose
    return da.transpose(*odims, **kw)


transpose.__doc__ = transpose.__doc__.format(**locals())


def get_dim_types(da, unknown=None, asdict=False):
    """Get dimension types

    Parameters
    ----------
    da: xarray.DataArray or tuple(str)
        Data array or tuple of dimensions
    unknown:
        Value to assign to unknown types
    asdict: bool
        Get the result as dictionary

    Return
    ------
    tuple
    """
    return xcf.get_cf_specs(da).coords.get_dim_types(
        da, unknown=unknown, asdict=asdict)


def get_order(da):
    """Like :func:`get_dim_types` but returning a string"""
    return "".join(get_dim_types(da, unknown="-", asdict=False))


def reorder(da, order):
    """Transpose an array to match a given order

    Parameters
    ----------
    da: xarray.DataArray
        Data array to transpose
    order: str
        A combination of x, y, z, t, f and - symbols and
        their upper case value.
        Letters refer to the dimension type.
        When the value is -, it may match any dimension type.

    Return
    ------
    xarray.DataArray
    """
    # Convert from dim_types
    if isinstance(order, dict):
        order = tuple(order.values())
    if isinstance(order, tuple):
        order = ''.join([
            ('-' if o not in "ftzyx" else o) for o in order])

    # From order to dims
    to_dims = ()
    dim_types = get_dim_types(da, asdict=True)
    ndim = len(dim_types)
    for i, o in enumerate(order[::-1]):
        if i+1 == ndim:
            break
        for dim in da.dims:
            if o == dim_types[dim]:
                to_dims = (dim, ) + to_dims
                break
        else:
            raise XoaError(
                f"Coordinate type not found: {o}. Dims are: {da.dims}")

    # Final transpose
    return transpose(da, to_dims)


def get_coords_compat_with_dims(da, include_dims=None, exclude_dims=None):
    """Return the coordinates that are compatible with dims

    Parameters
    ----------
    da: xarray.DataArray
        Data array
    include_dims: set(str)
        If provided, the coordinates must have at least one of these
        dimensions
    exclude_dims: set(str)
        If provided, the coordinates must not have one of these dimnesions

    Return
    ------
    list(str)
        List of coordinates
    """
    if isinstance(include_dims, str):
        include_dims = {include_dims}
    if isinstance(exclude_dims, str):
        exclude_dims = {exclude_dims}
    coords = []
    for coord in da.coords.values():
        dims = set(coord.dims)
        if include_dims and not include_dims.intersection(dims):
            continue
        if exclude_dims and exclude_dims.intersection(dims):
            continue
        coords.append(coord)
    return coords


def change_index(da, dim, values):
    """Change the values of a dataset or data array index

    Parameters
    ----------
    da: xarray.Dataset, xarray.DataArray
    dim: str
    values: array_like

    Return
    ------
    xarray.Dataset, xarray.DataArray

    See also
    --------
    xarray.DataArray.reset_index
    xarray.DataArray.assign_coords
    """
    attrs = da.coords[dim].attrs
    if hasattr(values, "attrs"):
        attrs.update(attrs)
    if dim in da.indexes:
        da = da.reset_index([dim], drop=True)
    coord = xr.DataArray(values, dims=dim, attrs=attrs)
    return da.assign_coords({dim: coord})


def drop_dim_coords(da, dim):
    """Drop coords that have a particular dim"""
    return da.drop([c.name for c in da.coords.values() if dim in c.dims])


class positive_attr(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Allowed value for the positive attribute argument"""
    #: Infer it from the axis coordinate
    infer = 0
    guess = 0
    #: Coordinates are increasing up
    up = 1
    #: Coordinates are increasing down
    down = -1


def get_positive_attr(da, zdim=None):
    """Get the positive attribute of a dataset

    Parameters
    ----------
    da: xarray.Dataset, xarray.DataArray
    zdim: None, str
        The index coordinate name that is supposed to have this attribute,
        which is usually the vertical dimension

    Return
    ------
    None, "up" or "down"
    """
    # Targets
    if zdim is None:
        zdim = get_cf_dims(da, "z", errors="ignore")
        if zdim:
            zdim = zdim[0]
    if zdim and zdim in da.coords:
        targets = [da.coords[zdim]]
    else:
        targets = list(da.coords.values())
        if isinstance(da, xr.Dataset):
            targets.extend(da.data_vars.values())

    # Loop on targets
    for target in targets:
        if "positive" in target.attrs:
            positive = da.coords[zdim].attrs["positive"]
            return positive_attr[positive].name

    # Fall back to current CFSpecs
    cfspecs = xcf.get_cf_specs(da)
    return cfspecs["vertical"]["positive"]


def get_binding_data_vars(ds, coord, as_names=False):
    """Get the data_vars that have this coordinate

    Parameters
    ----------
    ds: xarray.Dataset
    coord_name: str

    Return
    ------
    list
        List of data_var names
    """
    if not isinstance(coord, str):
        coord = coord.name
    out = [da for da in ds if coord in da.coords]
    if as_names:
        out = [da.name for da in out]
    return out
