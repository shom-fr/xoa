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
from . import cf


@misc.ERRORS.format_function_docstring
def get_lon(da, errors="raise"):
    """Get the longitude coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None
    """
    return cf.get_cf_specs(da).search(da, 'lon', errors=errors)


@misc.ERRORS.format_function_docstring
def get_lat(da, errors="raise"):
    """Get the latitude coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None
    """
    return cf.get_cf_specs(da).search(da, 'lat', errors=errors)


@misc.ERRORS.format_function_docstring
def get_depth(da, errors="raise"):
    """Get the depth coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None
    """
    return cf.get_cf_specs(da).search(da, 'depth', errors=errors)


@misc.ERRORS.format_function_docstring
def get_altitude(da, errors="raise"):
    """Get the altitude coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None
    """
    return cf.get_cf_specs(da).search(da, 'altitude', errors=errors)


@misc.ERRORS.format_function_docstring
def get_level(da, errors="raise"):
    """Get the level coordinate

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None
    """
    return cf.get_cf_specs(da).coords.search(da, 'level', errors=errors)


@misc.ERRORS.format_function_docstring
def get_vertical(da, errors="raise"):
    """Get either depth or altitude

    Parameters
    ----------
    {errors}

    Return
    ------
    xarray.DataArray or None
    """
    cfspecs = cf.get_cf_specs()
    height = cfspecs.search(da, 'depth', errors="ignore")
    if height is None:
        height = cfspecs.search(da, 'altitude', errors="ignore")
    if height is None:
        errors = misc.ERRORS[errors]
        msg = "No vertical coordinate found"
        if errors == "raise":
            raise cf.XoaCFError(msg)
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
    """
    return cf.get_cf_specs(da).coords.search(da, 'time', errors=errors)


@misc.ERRORS.format_function_docstring
def get_cf_coords(da, coord_names, errors="raise"):
    """Get several coordinates

    Parameters
    ----------
    {errors}

    Return
    ------
    list(xarray.DataArray)
    """
    cfspecs = cf.get_cf_specs(da)
    return [cfspecs.search_coord(da, coord_name, errors=errors)
            for coord_name in coord_names]


@misc.ERRORS.format_function_docstring
def get_dims(da, dim_types, allow_positional=False, positions='tzyx',
             errors="warn"):
    """Get the data array dimensions names from their type

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    dim_types: str, list
        Letters among "x", "y", "z", "t" and "f".
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
    return cf.get_cf_specs(da).get_dims(
        da, dim_types, allow_positional=allow_positional,
        positions=positions, errors=errors)


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


class DimFlusher1D(object):

    def __init__(self, da_in, coord_out, dim=None, coord_in_name=None):
        """Right-flush the working dimension

        Parameters
        ----------
        da_in: xarray.DataArray
            Input data array
        coord_out: xarray.DataArray
            Output coordinate array
        dim: str, tuple, None
            Working dimension
        coord_in_name: str, None
            Input coordinate name. If not provided, it is infered.
        """
        # Get the working dimensions
        if not isinstance(dim, (tuple, list)):
            dim = (dim, dim)
        dim0, dim1 = dim
        if None in dim or coord_in_name is None:
            cfspecs = cf.get_cf_specs(da_in)
        # - dim1 (out)
        if dim1 is None:  # get dim1 from coord_out
            dim1 = cfspecs.search_dim(coord_out)
            if dim1 is None:
                raise cf.XoaCFError("No CF dimension found for output coord. "
                                    "Please specifiy the working dimension.")
            dim1, dim_type = dim1
        else:  # dim1 is provided
            dim_type = cfspecs.coords.get_dim_type(dim1, coord_out)
        # - dim0 (in)
        if dim0 is None:
            if dim_type:
                for c0 in da_in.coords.values():
                    dim0 = cfspecs.coords.search_dim(c0, dim_type)
                    if dim0:
                        if dim_type is None:
                            dim0 = dim0[0]
                        break
                else:
                    raise cf.XoaCFError(
                        "No CF {dim_type }dimension found for datarray. "
                        "Please specifiy the working dimension.")
            else:
                dim0 = dim1  # be cafeful, dim1 must be in input!
        assert dim0 in da_in.dims
        assert dim1 in coord_out.dims

        # Input coordinate
        if coord_in_name:
            assert coord_in_name in da_in.coords, 'Invalid coordinate'
        else:
            coord_in = cfspecs.search_coord_from_dim(da_in, dim0)
            if coord_in is None:
                raise cf.XoaCFError(
                    f"No coordinate found matching dimension '{dim0}'")
            coord_in_name = coord_in.name

        # Check dims
        # - non-common other dimensions
        odims0 = set(da_in.dims).difference({dim0})
        odims1 = set(coord_out.dims).difference({dim1})
        if odims0.difference(odims1) and odims1.difference(odims0):
            raise XoaError("Conflicting non working dimensions")
        # - common dims, with size checking
        cdims = odims0.intersection(odims1).difference({dim0})
        for cdim in cdims:
            assert da_in.sizes[cdim] == coord_out.sizes[cdim]
        # - common dims in the order of da_in
        cdims0 = []
        for cdim0 in da_in.dims:
            if cdim0 in cdims:
                cdims0.append(cdim0)
        # - input dims with output dim
        dims01 = list(da_in.dims)
        if dim0 != dim1:
            dims01[dims01.index(dim0)] = dim1
        dims01 = tuple(dims01)

        # Store
        self._dim0, self._dim1 = dim0, dim1
        self._da_in = da_in
        self.coord_out = transpose(coord_out, (Ellipsis,) + dims01, "compat")
        self.coord_out_name = self.coord_out.name or coord_in.name
        # self._odims0 = odims0
        # self._odims1 = odims1
        # self._cdims0 = cdims0
        self.da_in = da_in

        # Transpose to push work dim right
        da_in = da_in.transpose(
            Ellipsis, *(cdims0 + [self._dim0]), transpose_coords=True)
        coord_out = coord_out.transpose(
            Ellipsis, *(cdims0 + [self._dim1]))

        # Broadcast data array
        # - data var
        if set(da_in.dims[:-1]) < set(coord_out.dims[:-1]):
            da_in = da_in.broadcast_like(coord_out,
                                         exclude=(self._dim0, self._dim1))
        # - input coordinate
        # if (set(coord_out.dims[:-1])set(da_in.coords[coord_in_name].dims[:-1])
        #         < set(coord_out.dims[:-1])):
        if (coord_out.ndim > 1 and set(coord_out.dims[:-1]) not in
                set(da_in.coords[coord_in_name].dims[:-1])):

        #set(coord_out.dims[:-1]) > set(da_in.coords[coord_in_name].dims[:-1]):
            if da_in.coords[coord_in_name].ndim == 1:
                coord_in_name, old_coord_in_name = (
                    coord_in_name + '_dimflush1d', coord_in_name)
            else:
                old_coord_in_name = coord_in_name
            da_in.coords[coord_in_name] = (
                da_in.coords[old_coord_in_name].broadcast_like(
                    coord_out, exclude=(self._dim0, self._dim1)))
        coord_in = da_in.coords[coord_in_name]
        # - output coordinate
        if (coord_out.ndim > 1 and
                set(coord_in.dims[:-1]) > set(coord_out.dims[:-1])):
            coord_out = coord_out.broadcast_like(
                coord_in, exclude=(self._dim0, self._dim1))

        # Info reverse transfoms
        # - input coords that doesn't have dim0 inside and must copied
        self.extra_coords = dict([(name, coord) for name, coord
                                  in da_in.coords.items()
                                  if dim0 not in coord.dims])
        # - da shape after reshape + broadcast
        self.work_shape = da_in.shape[:-1] + (coord_out.sizes[dim1], )
        self.work_dims = da_in.dims[:-1] + (dim1, )
        self.final_dims = list(self._da_in.dims)
        idim0 = self.final_dims.index(dim0)
        self.final_dims[idim0] = dim1
        self.final_dims = tuple(self.final_dims)
        # self.final_shape = list(self._da_in.shape)
        # self.final_shape[idim0] = coord_out.sizes[dim1]

        # Convert to numpy 2D
        self.da_in_data = da_in.data.reshape(-1, da_in.shape[-1])
        self.coord_in_data = coord_in.data.reshape(-1, coord_in.shape[-1])
        self.coord_out_data = coord_out.data.reshape(-1, coord_out.shape[-1])

    def get_back(self, data_out):

        data_out = data_out.reshape(self.work_shape)
        da_out = xr.DataArray(data_out, dims=self.work_dims)
        da_out = da_out.transpose(Ellipsis, *self.final_dims)
        da_out[self.coord_out_name] = self.coord_out
        da_out.attrs.update(self.da_in.attrs)
        da_out.coords.update(self.extra_coords)
        da_out.name = self.da_in.name

        return da_out


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
    return cf.get_cf_specs(da).coords.get_dim_types(
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
        zdim = get_dims(da, "z", errors="ignore")
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
    cfspecs = cf.get_cf_specs(da)
    return cfspecs["vertical"]["positive"]
