# -*- coding: utf-8 -*-
"""
Utilities for working with coordinates and dimensions.

This module provides functions to identify, retrieve and manipulate
coordinates (longitude, latitude, depth, time, etc.) and dimensions
from :mod:`xarray` data arrays and datasets, based on the
:mod:`xoa.meta` metadata specifications.
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

from collections.abc import Mapping

import numpy as np
import xarray as xr

from . import exceptions
from . import misc
from . import meta


def ensure_ns_datetime(da):
    """Convert datetime coordinates to nanosecond precision

    This avoids warnings from xarray when datetime coordinates
    use a different resolution (e.g. microseconds or hours).

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset

    Return
    ------
    xarray.DataArray or xarray.Dataset
        Copy with datetime coordinates converted to ``datetime64[ns]``.

    Example
    -------
    .. code-block:: python

        >>> times = xr.DataArray(np.array(["2000-01-01"], dtype="M8[us]"), dims="time")
        >>> da = xr.DataArray([1.0], coords={"time": times})
        >>> da = ensure_ns_datetime(da)
        >>> da.time.dtype
        dtype('<M8[ns]')
    """
    for name, coord in da.coords.items():
        if coord.dtype.kind == "M" and coord.dtype != np.dtype("datetime64[ns]"):
            da = da.assign_coords({name: coord.values.astype("datetime64[ns]")})
    return da


@misc.ERRORS.format_function_docstring
def get_lon(da, errors="raise", **kwargs):
    """Get the longitude coordinate

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

    Return
    ------
    xarray.DataArray or None

    Example
    -------
    .. code-block:: python

        >>> da = xr.DataArray([1, 2], dims="lon",
        ...     coords=dict(lon=[10., 20.]))
        >>> get_lon(da)
        <xarray.DataArray 'lon' (lon: 2)> ...

    See also
    --------
    get_lat
    get_depth
    get_altitude
    get_level
    get_vertical
    get_time
    xoa.meta.MetaSpecs.search
    """
    return meta.get_meta_specs(da).search(da, 'lon', errors=errors, **kwargs)


def is_lon(da, loc="any"):
    """Tell if a data array is identified as longitudes

    Parameters
    ----------
    da: xarray.DataArray
    loc: str
        Staggered grid location

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
    xoa.meta.MetaCoordSpecs.match
    """
    return meta.get_meta_specs(da).coords.match(da, "lon", loc=loc)


@misc.ERRORS.format_function_docstring
def get_lat(da, errors="raise", **kwargs):
    """Get the latitude coordinate

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

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
    xoa.meta.MetaSpecs.search
    """
    return meta.get_meta_specs(da).search(da, 'lat', errors=errors, **kwargs)


def is_lat(da, loc="any"):
    """Tell if a data array is identified as latitudes

    Parameters
    ----------
    da: xarray.DataArray
    loc: str
        Staggered grid location

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
    xoa.meta.MetaCoordSpecs.match
    """
    return meta.get_meta_specs(da).coords.match(da, "lat", loc=loc)


def _search_z_depth_(da, name, **kwargs):
    """Search for `name` ("z" or "depth"), else for the other one, converted"""
    metaspecs = meta.get_meta_specs(da)
    other = "depth" if name == "z" else "z"
    found = metaspecs.search(da, name, errors="ignore", **kwargs)
    if found is not None:
        return found
    found = metaspecs.search(da, other, errors="ignore", **kwargs)
    if found is not None:
        return to_z(found) if name == "z" else to_depth(found)


def _get_z_depth_(da, name, errors, **kwargs):
    """Common part of :func:`get_z` and :func:`get_depth`"""
    errors = misc.ERRORS[errors]
    ztype = meta.get_meta_specs(da)["vertical"]["type"]

    # From variable, converted if needed
    found = _search_z_depth_(da, name, **kwargs)
    if found is not None:
        return found

    # Decode the dataset
    if ztype != "z" and hasattr(da, "data_vars"):
        decoders = []
        if ztype in ("sigma", None):
            from .sigma import decode_sigma

            decoders.append(decode_sigma)
        if ztype in ("dz", "layer", None):
            from .grid import decode_dz2depth

            decoders.append(decode_dz2depth)
        for decoder in decoders:
            decoded = decoder(da, errors="ignore" if ztype is None else errors)
            found = _search_z_depth_(decoded, name, **kwargs)
            if found is not None:
                return found
        msg = "Can't infer {} coordinate from dataset".format(name)
    else:
        msg = "No {} coordinate found".format(name)
    if errors == "raise":
        raise exceptions.XoaCoordsError(msg)
    if errors == "warn":
        exceptions.xoa_warn(msg)


@misc.ERRORS.format_function_docstring
def get_z(da, errors="raise", **kwargs):
    """Get or compute the z coordinate, which is positive up

    The z coordinate is negative in the ocean and increases from the bottom
    to the surface, as opposed to the depth, which is positive in the ocean.
    If a z variable cannot be found, it uses the depth, if any, with
    :func:`to_z`, or tries to compute it either from sigma-like coordinates
    or from layer thicknesses.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_depth
    get_vertical
    to_z
    to_depth
    xoa.meta.MetaSpecs.search
    xoa.sigma.decode_sigma
    xoa.grid.decode_dz2depth
    """
    return _get_z_depth_(da, "z", errors, **kwargs)


def is_z(da, loc="any"):
    """Tell if a data array is identified as z, i.e. positive up

    Parameters
    ----------
    da: xarray.DataArray
    loc: str
        Staggered grid location

    Return
    ------
    bool

    See also
    --------
    is_depth
    is_altitude
    is_level
    xoa.meta.MetaCoordSpecs.match
    """
    return meta.get_meta_specs(da).coords.match(da, "z", loc=loc)


@misc.ERRORS.format_function_docstring
def get_depth(da, errors="raise", **kwargs):
    """Get or compute the depth coordinate, which is positive down

    If a depth variable cannot be found, it uses the z coordinate, if any,
    with :func:`to_depth`, or tries to compute it either
    from sigma-like coordinates or from layer thicknesses.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_z
    get_lon
    get_lat
    get_time
    get_altitude
    get_level
    get_vertical
    to_depth
    to_z
    xoa.meta.MetaSpecs.search
    xoa.sigma.decode_sigma
    xoa.grid.decode_dz2depth
    """
    return _get_z_depth_(da, "depth", errors, **kwargs)


def is_depth(da, loc="any"):
    """Tell if a data array is identified as depths

    Parameters
    ----------
    da: xarray.DataArray
    loc: str
        Staggered grid location

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
    xoa.meta.MetaCoordSpecs.match
    """
    return meta.get_meta_specs(da).coords.match(da, "depth", loc=loc)


@misc.ERRORS.format_function_docstring
def get_altitude(da, errors="raise", **kwargs):
    """Get the altitude coordinate

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

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
    xoa.meta.MetaSpecs.search
    """
    da = meta.infer_coords(da)
    return meta.get_meta_specs(da).search(da, 'altitude', errors=errors, **kwargs)


def is_altitude(da, loc="any"):
    """Tell if a data array is identified as altitudes

    Parameters
    ----------
    da: xarray.DataArray
    loc: str
        Staggered grid location

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
    xoa.meta.MetaCoordSpecs.match
    """
    da = meta.infer_coords(da)
    return meta.get_meta_specs(da).coords.match(da, "altitude", loc=loc)


@misc.ERRORS.format_function_docstring
def get_level(da, errors="raise", *kwargs):
    """Get the level coordinate

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

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
    xoa.meta.MetaSpecs.search
    """
    return meta.get_meta_specs(da).coords.search(da, 'level', errors=errors, **kwargs)


def is_level(da, loc="any"):
    """Tell if a data array is identified as levels

    Parameters
    ----------
    da: xarray.DataArray
    loc: str
        Staggered grid location

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
    xoa.meta.MetaCoordSpecs.match
    """
    return meta.get_meta_specs(da).coords.match(da, "levels", loc=loc)


@misc.ERRORS.format_function_docstring
def get_vertical(da, errors="raise", **kwargs):
    """Get either z, depth or altitude

    Tries to find a depth coordinate first, then a z and finally an altitude.
    No conversion is performed: use :func:`get_z` or :func:`get_depth` for that.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    get_lon
    get_lat
    get_z
    get_depth
    get_altitude
    get_level
    get_time
    xoa.meta.MetaSpecs.search
    """
    metaspecs = meta.get_meta_specs(da)
    for meta_name in ("depth", "z", "altitude"):
        height = metaspecs.search(da, meta_name, errors="ignore", **kwargs)
        if height is not None:
            break
    if height is None:
        errors = misc.ERRORS[errors]
        msg = "No vertical coordinate found"
        if errors == "raise":
            raise exceptions.XoaCoordsError(msg)
        elif errors == "warn":
            exceptions.xoa_warn(msg)
    else:
        return height


def reverse_dim(da, dim=None):
    """Reverse a data array along a dimension

    This is a view for numpy arrays and a lazy operation for dask arrays.

    Parameters
    ----------
    da: xarray.DataArray
    dim: None, str
        Dimension name, which defaults to the first one.

    Return
    ------
    xarray.DataArray

    See also
    --------
    to_z
    to_depth
    """
    dim = dim or da.dims[0]
    return da.isel({dim: slice(None, None, -1)})


def _flip_vertical_(da, name, positive, reverse):
    """Common part of :func:`to_z` and :func:`to_depth`"""
    out = 0 - da  # avoids negative zeros
    if da.name in out.coords:  # stale copy of itself
        out = out.drop_vars(da.name)
    out.attrs = {k: v for k, v in da.attrs.items() if k == "units"}
    out.attrs["positive"] = positive
    out = out.rename(name)
    return reverse_dim(out, None if reverse is True else reverse) if reverse else out


def to_z(da, name="z", reverse=False):
    """Convert a depth (positive down) to z (positive up) by changing its sign

    Only the ``units`` attribute is kept, and the ``positive`` attribute
    is set to ``"up"``. The array is lazily negated if it is a dask array.

    Parameters
    ----------
    da: xarray.DataArray
        Depths, i.e. positive in the ocean.
    name: str
        Name of the output array.
    reverse: bool, str
        Also reverse the array along its first dimension, or along the given
        dimension, to switch from an order going from the surface to the bottom
        to the opposite.

    Return
    ------
    xarray.DataArray

    See also
    --------
    to_depth
    reverse_dim
    get_z

    Example
    -------
    .. code-block:: python

        >>> depth = xr.DataArray([0., 10, 50], dims="k", attrs={"units": "m"})
        >>> to_z(depth, reverse=True).values
        array([-50., -10.,   0.])
    """
    return _flip_vertical_(da, name, "up", reverse)


def to_depth(da, name="depth", reverse=False):
    """Convert a z (positive up) to depth (positive down) by changing its sign

    This is the mirror of :func:`to_z`.

    Parameters
    ----------
    da: xarray.DataArray
        Heights, i.e. negative in the ocean.
    name: str
        Name of the output array.
    reverse: bool, str
        Also reverse the array along its first dimension, or along the given
        dimension, to switch from an order going from the bottom to the surface
        to the opposite.

    Return
    ------
    xarray.DataArray

    See also
    --------
    to_z
    reverse_dim
    get_depth
    """
    return _flip_vertical_(da, name, "down", reverse)


@misc.ERRORS.format_function_docstring
def get_time(da, errors="raise", **kwargs):
    """Get the time coordinate

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

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
    xoa.meta.MetaSpecs.search
    """
    return meta.get_meta_specs(da).coords.search(da, 'time', errors=errors, **kwargs)


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
    xoa.meta.MetaCoordSpecs.match
    """
    da = meta.infer_coords(da)
    return meta.get_meta_specs(da).match(da, "time")


@misc.ERRORS.format_function_docstring
def get_meta_coords(da, coord_names, errors="raise", **kwargs):
    """Get several coordinates at once

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    coord_names: list(str)
        List of coordinate names to search for (e.g. ``["lon", "lat"]``).
    {errors}
    kwargs:
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.search`

    Return
    ------
    list(xarray.DataArray)

    Example
    -------
    .. code-block:: python

        >>> lon, lat = get_meta_coords(ds, ["lon", "lat"])

    See also
    --------
    xoa.meta.MetaSpecs.search_coord
    """
    da = meta.infer_coords(da)
    metaspecs = meta.get_meta_specs(da)
    return [
        metaspecs.search_coord(da, coord_name, errors=errors, **kwargs)
        for coord_name in coord_names
    ]


def get_cf_coords(*args, **kwargs):
    exceptions.exceptions.xoa_warn(
        "get_meta_coords is deprecated. Please use get_meta_coords instead.", "deprecation"
    )
    return get_meta_coords(*args, **kwargs)


@misc.ERRORS.format_function_docstring
def get_meta_dims(da, meta_args, allow_positional=False, positions='tzyx', errors="warn", **kwargs):
    """Get the data array dimensions names from their type

    Parameters
    ----------
    da: xarray.DataArray
        Array to scan
    meta_args: str, list
        Letters among "x", "y", "z", "t" and "f",
        or generic CF names.
    allow_positional: bool
        Fall back to positional dimension of types if unkown.
    positions: str
        Default position per type starting from the end.
    {errors}
    kwargs: dict
        Extra parameters are passed to :meth:`xoa.meta.MetaSpecs.get_dims` in addition to
        the above parameters.

    Return
    ------
    tuple(list) or list
        Tuple of dimension name or None when the dimension if not found

    See also
    --------
    xoa.meta.MetaSpecs.get_dims
    """
    da = meta.infer_coords(da)
    return meta.get_meta_specs(da).get_dims(
        da,
        meta_args,
        allow_positional=allow_positional,
        positions=positions,
        errors=errors,
        **kwargs,
    )


def get_cf_dims(*args, **kwargs):
    exceptions.exceptions.xoa_warn(
        "get_meta_meta is deprecated. Please use get_meta_dims instead.", "deprecation"
    )
    return get_meta_dims(*args, **kwargs)


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
        Extra parameters are passed to :func:`get_meta_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_dims
    """
    return get_meta_dims(da, "x", errors=errors, **kwargs)


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
        Extra parameters are passed to :func:`get_meta_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_dims
    """
    return get_meta_dims(da, "y", errors=errors, **kwargs)


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
        Extra parameters are passed to :func:`get_meta_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_meta_dims
    """
    return get_meta_dims(da, "z", errors=errors, **kwargs)


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
        Extra parameters are passed to :func:`get_meta_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_meta_dims
    """
    return get_meta_dims(da, "t", errors=errors, **kwargs)


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
        Extra parameters are passed to :func:`get_meta_dims`

    Return
    ------
    str or None
        The dimension name or None

    See also
    --------
    get_meta_dims
    """
    return get_meta_dims(da, "f", errors=errors, **kwargs)


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
            odims += (dim,)
        elif dim in da.dims:
            odims += (dim,)
        elif mode == "insert":
            expand_dims[dim] = 1
            odims += (dim,)
        elif mode == "resize":
            if sizes is None or dim not in sizes:
                exceptions.xoa_warn(
                    f"new dim '{dim}' in transposition is set to one"
                    " since no size is provided to it"
                )
                size = 1
            else:
                size = sizes[dim]
            expand_dims[dim] = size
            odims += (dim,)

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
        Get the result as a dictionary mapping dimension names to types

    Return
    ------
    tuple or dict
        Dimension types as single-letter strings ("x", "y", "z", "t", "f")
        or ``unknown`` for unrecognized dimensions.

    Example
    -------
    .. code-block:: python

        >>> da = xr.DataArray(np.ones((3, 4)), dims=("lat", "lon"))
        >>> get_dim_types(da, unknown="-")
        ('y', 'x')
        >>> get_dim_types(da, asdict=True)
        {'lat': 'y', 'lon': 'x'}
    """
    return meta.get_meta_specs(da).coords.get_dim_types(da, unknown=unknown, asdict=asdict)


def get_order(da):
    """Get the dimension order as a string like ``"tzy-x"``

    Unknown dimensions are represented by ``"-"``.

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    str

    Example
    -------
    .. code-block:: python

        >>> da = xr.DataArray(np.ones((2, 3, 4)), dims=("time", "lat", "lon"))
        >>> get_order(da)
        'tyx'

    See also
    --------
    get_dim_types
    """
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
        order = ''.join([('-' if o not in "ftzyx" else o) for o in order])

    # From order to dims
    to_dims = ()
    dim_types = get_dim_types(da, asdict=True)
    ndim = len(dim_types)
    for i, o in enumerate(order[::-1]):
        if i + 1 == ndim:
            break
        for dim in da.dims:
            if o == dim_types[dim]:
                to_dims = (dim,) + to_dims
                break
        else:
            raise exceptions.XoaCoordsError(f"Coordinate type not found: {o}. Dims are: {da.dims}")

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
        If provided, the coordinates must not have one of these dimensions

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
    """Drop all coordinates that depend on a given dimension

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    dim: str
        Dimension name

    Return
    ------
    xarray.DataArray or xarray.Dataset
    """
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
    """Get the positive attribute of a dataset or data array

    Searches for the ``positive`` attribute in the vertical coordinate
    or falls back to the current meta specs default.

    Parameters
    ----------
    da: xarray.Dataset, xarray.DataArray
    zdim: None, str
        The index coordinate name that is supposed to have this attribute,
        which is usually the vertical dimension.

    Return
    ------
    None, "up" or "down"

    Example
    -------
    .. code-block:: python

        >>> depth = xr.DataArray([-100., -50., 0.], dims="z",
        ...     attrs={"positive": "up"})
        >>> da = xr.DataArray([1, 2, 3], dims="z",
        ...     coords={"depth": depth})
        >>> get_positive_attr(da)
        'up'
    """
    # Targets
    if zdim is None:
        zdim = get_meta_dims(da, "z", errors="ignore")
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
            positive = target.attrs["positive"]
            return positive_attr[positive].name

    # Fall back to current MetaSpecs
    metaspecs = meta.get_meta_specs(da)
    return metaspecs["vertical"]["positive"]


def get_binding_data_vars(ds, coord, as_names=False):
    """Get the data_vars that have a given coordinate

    Parameters
    ----------
    ds: xarray.Dataset
    coord: str, xarray.DataArray
        Coordinate name or data array (its ``name`` attribute is used).
    as_names: bool
        If True, return variable names instead of data arrays.

    Return
    ------
    list
        List of :class:`xarray.DataArray` or of names if ``as_names`` is True.
    """
    if not isinstance(coord, str):
        coord = coord.name
    out = [da for da in ds if coord in da.coords]
    if as_names:
        out = [da.name for da in out]
    return out


def geo_stack(
    obj, stack_dim="npts", rename=False, drop=False, reset_index=False, create_index=False
):
    """Stack the dimensions of longitude and latitude coordinates

    .. note:: If already stacked or similar, a simple copy is returned,
        except if `rename` is True.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with valid longitude and latitude coordinates
    stack_dim: str
        Name of the new stack dimension
    rename: False
        Rename longitude to `lon` and latitude to `lat` for convenience.
        If no need to stack, rename the single dimension to `stack_dim`.
    drop: bool
        Drop all variables and coordinates that does not contain final stack dimension
    reset_index: bool
        Remove MultiIndexes that contains lon and lat dimensions before any operations
    create_index: bool
        Create a MultiIndex when calling :meth:`~xarray.DataArray.stack`

    Return
    ------
    xarray.DataArray, xarray.Dataset
        Array or dataset with lon and lat stacked.
        Its "geo_stack" :attr:`~xarray.DataArray.encoding` attribute value is set
        to `(stack_dim, lon_dim, lat_dim)`.

    See also
    --------
    xarray.DataArray.stack
    """
    lon = get_lon(obj)
    lat = get_lat(obj)

    # Singleton
    singleton = lon.ndim == 0 and lat.ndim == 0
    dims = list(set(lon.dims).union(lat.dims))

    # Rename to lon and lat?
    if rename:
        if (lon.name in obj.coords and lat.name in obj.coords) or reset_index:
            if not singleton and lon.name not in obj.coords:
                obj = obj.reset_index(dims[0])
            obj = obj.rename({lon.name: "lon", lat.name: "lat"})
            lon = obj.lon
            lat = obj.lat

        else:
            exceptions.xoa_warn(
                "Cannot rename lon and lat coordinates since "
                "they are component of a MultiIndex. "
                "Pass reset_index=True to allow it."
            )

    # Stack or not stack?
    if singleton:
        stack_dim = None
    elif lon.ndim == 1 and lat.ndim == 1 and lon.dims == lat.dims:
        if rename:
            obj = obj.rename({dims[0]: stack_dim})
        else:
            stack_dim = dims[0]
            obj = obj.copy()
    else:
        obj = obj.stack({stack_dim: dims}, create_index=create_index)

    # No multiindex?
    if not singleton and reset_index and lon.name not in obj.coords:
        obj = obj.reset_index(stack_dim)

    # Drop variables with no stacked coordinates
    if drop and hasattr(obj, "data_vars"):
        names = list(obj)
        for name in list(obj):
            if (stack_dim in obj[name].dims) or (
                singleton and lon.name in obj[name].coords and lat.name in obj[name].coords
            ):
                names.remove(name)
        obj = obj.drop_vars(names)

    # Keep trace
    obj.encoding["geo_stack"] = (stack_dim, lon.name, lat.name)
    return obj


def geo_merge(lon, lat, stack_dim="pts"):
    """Merge lon/lat coordinates into data arrays with coherent dimensions

    Four cases are handled:

    - **DataArray, same dims**: already a point cloud or an aligned nD array.
      Dimension and variable names are preserved.
    - **DataArray, different dims**: rectangular (``lon(x)`` + ``lat(y)``) or
      partially shared grid. Both arrays are broadcast to the union of
      dimensions with latitude-only dimensions first (y before x).
    - **Numpy, same shape**: point cloud or nD grid, with auto-generated dimension
      names (``stack_dim``, ``stack_dim_y``/``stack_dim_x``, ...).
    - **Numpy 1D, different sizes**: rectangular grid converted with
      :func:`numpy.meshgrid` to shape ``(ny, nx)``.

    Parameters
    ----------
    lon, lat: array_like, xarray.DataArray
        Longitude and latitude arrays, any shape.
        Names of data arrays are preserved, and others are named ``"lon"``
        and ``"lat"``.
    stack_dim: str
        Base name used when auto-generating dimension names.

    Return
    ------
    xarray.DataArray
        Longitudes with coherent dimensions.
    xarray.DataArray
        Latitudes with the same dimensions.

    See also
    --------
    geo_stack
    """
    lon_is_da = isinstance(lon, xr.DataArray)
    lat_is_da = isinstance(lat, xr.DataArray)
    lon_name = (lon.name if lon_is_da and lon.name else None) or "lon"
    lat_name = (lat.name if lat_is_da and lat.name else None) or "lat"

    if lon_is_da or lat_is_da:
        if not lon_is_da:
            lon = xr.DataArray(
                np.asarray(lon, dtype=np.float64), dims=[f"{stack_dim}_x"], name=lon_name
            )
        if not lat_is_da:
            lat = xr.DataArray(
                np.asarray(lat, dtype=np.float64), dims=[f"{stack_dim}_y"], name=lat_name
            )
        if lon.dims == lat.dims:
            return (
                lon.astype(np.float64, copy=False).rename(lon_name),
                lat.astype(np.float64, copy=False).rename(lat_name),
            )

        # Different dims: broadcast with lat-only dims first
        lat_only = [d for d in lat.dims if d not in lon.dims]
        lon_only = [d for d in lon.dims if d not in lat.dims]
        shared = [d for d in lat.dims if d in lon.dims]
        dims = tuple(lat_only + shared + lon_only)
        lon_bc, lat_bc = xr.broadcast(lon, lat)
        return (
            lon_bc.transpose(*dims).astype(np.float64, copy=False).rename(lon_name),
            lat_bc.transpose(*dims).astype(np.float64, copy=False).rename(lat_name),
        )

    lon_np = np.asarray(lon, dtype=np.float64)
    lat_np = np.asarray(lat, dtype=np.float64)

    if lon_np.shape == lat_np.shape:
        lon_np = lon_np.reshape(1) if lon_np.ndim == 0 else lon_np
        lat_np = lat_np.reshape(1) if lat_np.ndim == 0 else lat_np
        if lon_np.ndim == 1:
            dims = (stack_dim,)
        elif lon_np.ndim == 2:
            dims = (f"{stack_dim}_y", f"{stack_dim}_x")
        else:
            dims = tuple(f"{stack_dim}_{i}" for i in range(lon_np.ndim))
        return (
            xr.DataArray(lon_np, dims=dims, name=lon_name),
            xr.DataArray(lat_np, dims=dims, name=lat_name),
        )

    if lon_np.ndim == 1 and lat_np.ndim == 1:
        lon_np, lat_np = np.meshgrid(lon_np, lat_np)  # (ny, nx)
        dims = (f"{stack_dim}_y", f"{stack_dim}_x")
        return (
            xr.DataArray(lon_np, dims=dims, name=lon_name),
            xr.DataArray(lat_np, dims=dims, name=lat_name),
        )

    raise exceptions.XoaError(
        f"lon shape {np.shape(lon)} and lat shape {np.shape(lat)} are incompatible: "
        "only 1D arrays of different sizes are accepted as a rectangular grid; "
        "otherwise both arrays must have the same shape."
    )
