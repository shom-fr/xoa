"""
This module provides 1d to nD grid utilities to get information
or perform operations on a grid.

For operations between different grids, please see :mod:`xoa.regrid`.
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

import numpy as np
import xarray as xr

from . import coords as xcoords
from . import exceptions, meta, misc
from . import geo as xgeo
from .core import grid as cgrid
from .core import num as cnum


def apply_along_dim(
    ds,
    dim,
    func,
    coord_func=None,
    data_kwargs=None,
    coord_kwargs=None,
    name_kwargs=None,
    **kwargs,
):
    """Apply an operator on data array or dataset dimensions

    The operator may potentially change size of the array.
    It is applied on the data array with the data_kwargs
    arguments and the coordinate arrays with the coord_kwargs arguments.

    Parameters
    ----------
    ds: xarray.DataArray, xarray.Dataset
    dim: str, tuple(str)
    func: callable
        Operator function that works on a specific dimension.
        It is applied to both data and coordinates, unless
        ``coord_func`` is provided.
    coord_func: callable, None
        Function to apply to coordinates specifically, which defaults
        to ``func``
    data_kwargs: None, dict
        Parameters passed to func for the data array
    coord_kwargs: None, dict
        Parameters passed to func for the coordinates
    name_kwargs: dict(dict)
        A dict of whose keys are coordinate name and whose values
        are passed to func only for these coordinates.
    kwargs: dict
        Extra keywords are passed to the ``func`` function

    Return
    ------
    xarray.DataArray, xarray.Dataset

    See also
    --------
    get_centers
    get_edges
    pad
    """
    # Always return a copy
    dso = ds.copy()

    # Loop on dims
    if coord_func is None:
        coord_func = func
    dim = meta.get_meta_specs(ds).parse_dims(dim, ds)
    dims = (dim,) if isinstance(dim, str) else dim
    for dim in dims:
        if dim not in dso.dims:
            continue

        # Data array or dataset
        old_coords = dso.coords
        if isinstance(dso, xr.Dataset):
            das = dso.data_vars.values()
            dso = xr.Dataset()
        else:
            das = [dso]
        daos = {}
        kwd = kwargs.copy()
        if data_kwargs:
            kwd.update(data_kwargs)
        for da in das:
            if dim not in da.dims:
                dao = da
            else:
                kw = kwd.copy()
                if name_kwargs and da.name in name_kwargs:
                    kw.update(name_kwargs.get(da.name))
                dao = func(xr.DataArray(da.data, dims=da.dims), dim, **kw)
                dao.name = da.name
                dao.encoding = da.encoding
                dao.attrs = da.attrs
            daos[dao.name] = dao
        if isinstance(dso, xr.Dataset):
            dso = dso.update(daos)
            dso.attrs = ds.attrs
            dso.encoding = ds.encoding
        else:
            dso = list(daos.values())[0]
        da_names = [name for name in daos.keys() if name]

        # Coordinates
        coords = {}
        if name_kwargs is None:
            name_kwargs = {}
        for coord_name, old_coord in old_coords.items():
            if coord_name in da_names:
                continue
            if dim in old_coord.dims:
                kw = kwargs.copy()
                for dd in (coord_kwargs, name_kwargs.get(coord_name)):
                    if dd:
                        kw.update(dd)
                coord = coord_func(
                    xr.DataArray(old_coord.data, dims=old_coord.dims),
                    dim,
                    **kw,
                )
                coord.attrs = old_coord.attrs
                coord.encoding = old_coord.encoding
            else:
                coord = old_coord
            coords[coord_name] = coord
        dso = dso.assign_coords(coords)

    meta.assign_meta_specs(dso, ds)

    return dso


def map_along_dim(func, da, dim, *coords, out_dim=None, out_size=None, dtype=None, **kwargs):
    """Apply a function along a dimension that is kept in place

    Contrary to :func:`xarray.apply_ufunc`, the working dimension is not moved
    to the end, so the arrays are not transposed and copied: the function can view
    them as ``(nouter, n, ninner)`` with :func:`xoa.core.num.broadcast_along_axis`.

    The coordinates, like the input and output coordinates of an interpolation,
    may have any subset of the dimensions of the data array, plus other ones,
    which are put first in the output.
    Dask arrays are processed with :func:`dask.array.map_blocks`: only the working
    dimension must be a single chunk, and the other chunks are kept.

    Parameters
    ----------
    func: callable
        Function called as ``func(data, *coords, axis=axis, **kwargs)`` with numpy
        arrays that have all the dimensions of the output, in the same order,
        with a size of 1 for those they don't have.
        It returns an array of the shape of the output.
    da: xarray.DataArray
        Data array
    dim: str
        Working dimension
    coords: xarray.DataArray
        Other arrays that have `dim` or `out_dim`
    out_dim: str, None
        Output dimension, which defaults to `dim`
    out_size: int, None
        Output size along `out_dim`, which defaults to the size of the coordinates
        along `out_dim` if any, else to the size of `da` along `dim`
    dtype: None, numpy.dtype
        Output data type, which defaults to the type of `da` promoted to float
    kwargs: dict
        Extra keywords are passed to `func`

    Return
    ------
    xarray.DataArray
        Array with the dimensions of `da`, `dim` being replaced by `out_dim`,
        and the other dimensions of the coordinates first.
        Its coordinates are those of `da` that don't have `dim`, and its name and
        attributes are those of `da`.

    Example
    -------
    .. ipython:: python

        @suppress
        import numpy as np, xarray as xr
        @suppress
        from xoa.grid import map_along_dim
        def double(data, axis):
            return np.repeat(data, 2, axis=axis)

        da = xr.DataArray(np.ones((3, 4, 5)), dims=("time", "z", "x"))
        map_along_dim(double, da, "z", out_size=8).dims

    See also
    --------
    xoa.core.num.broadcast_along_axis
    xoa.regrid.regrid1d
    """
    out_dim = out_dim or dim
    if out_size is None:
        out_size = next((c.sizes[out_dim] for c in coords if out_dim in c.dims), da.sizes[dim])
    if dtype is None:
        dtype = np.result_type(da.dtype, float)

    # Dimensions of the output, with dim as working dimension
    renamed = []
    for coord in coords:
        if out_dim != dim and out_dim in coord.dims:
            if dim in coord.dims:
                raise exceptions.XoaError(f"An array cannot have both '{dim}' and '{out_dim}'")
            coord = coord.rename({out_dim: dim})
        if dim not in coord.dims:
            raise exceptions.XoaError(f"Arrays must have the '{dim}' or '{out_dim}' dimension")
        renamed.append(coord)
    extra = []
    for coord in renamed:
        extra.extend(d for d in coord.dims if d not in da.dims and d not in extra)
    dims = tuple(extra) + da.dims
    axis = dims.index(dim)

    # Arrays with all the dimensions, the missing ones being of size 1
    arrays = []
    for arr in [da] + renamed:
        arr = arr.transpose(*[d for d in dims if d in arr.dims])
        arrays.append(arr.data.reshape([arr.sizes.get(d, 1) for d in dims]))

    # Apply
    if not any(hasattr(arr, "dask") for arr in arrays):
        out = func(*arrays, axis=axis, **kwargs)
    else:
        import dask.array

        # The chunks of the output come from the first dask array that has the dimension
        ref_chunks = []
        for i in range(len(dims)):
            size = max(arr.shape[i] for arr in arrays)
            chunks = next(
                (
                    arr.chunks[i]
                    for arr in arrays
                    if hasattr(arr, "dask") and arr.shape[i] == size and size > 1
                ),
                (size,),
            )
            ref_chunks.append(chunks)
        arrays = [
            dask.array.asarray(arr).rechunk(
                tuple(
                    -1 if (i == axis or n == 1) else ref_chunks[i] for i, n in enumerate(arr.shape)
                )
            )
            for arr in arrays
        ]
        ref_chunks[axis] = (out_size,)
        out = dask.array.map_blocks(
            func,
            *arrays,
            axis=axis,
            chunks=tuple(ref_chunks),
            dtype=dtype,
            meta=np.array((), dtype=dtype),
            **kwargs,
        )

    out_dims = tuple(out_dim if d == dim else d for d in dims)
    coords = {name: coord for name, coord in da.coords.items() if dim not in coord.dims}
    return xr.DataArray(out, dims=out_dims, coords=coords, name=da.name, attrs=da.attrs)


def _lon_names_(da, name_kwargs=None):
    """Flag the longitude coordinates in the per-coordinate keywords"""
    name_kwargs = {name: dict(kw) for name, kw in (name_kwargs or {}).items()}
    for name, coord in da.coords.items():
        if xcoords.is_lon(coord):
            name_kwargs.setdefault(name, {})["lon"] = True
    return name_kwargs


def _is_lon_data_(da):
    return isinstance(da, xr.DataArray) and xcoords.is_lon(da)


def _fast_pad_(da, mode, kwargs, lon):
    if not (cnum.is_float(da) or lon):
        return False
    if mode == "constant":
        return set(kwargs) <= {"constant_values"} and np.ndim(kwargs.get("constant_values", 0)) == 0
    return mode in ("edge", "linear_extrap", "wrap") and not kwargs


def _pad_(da, dim, pad_width, mode, lon=False, **kwargs):
    pad_width = pad_width.get(dim, 0)
    if not pad_width:
        return da.copy()

    if isinstance(pad_width, int):
        pad_width = (pad_width,)
    pad_width0 = pad_width[0]
    pad_width1 = pad_width[-1]

    if _fast_pad_(da, mode, kwargs, lon):
        data = cgrid.pad(
            da.values, da.get_axis_num(dim), (pad_width0, pad_width1), mode, lon=lon, **kwargs
        )
        return xr.DataArray(data, dims=da.dims)

    if mode != "linear_extrap":
        return da.pad({dim: (pad_width0, pad_width1)}, mode=mode, **kwargs)

    to_concat = []
    if pad_width0:
        ramp0 = xr.DataArray(np.arange(pad_width0, 0, -1, dtype=da.dtype), dims=dim)
        da0 = da[{dim: 0}] + (da[{dim: 0}] - da[{dim: 1}]) * ramp0
        to_concat.append(da0.transpose(*da.dims))
    to_concat.append(da)
    if pad_width1:
        ramp1 = xr.DataArray(np.arange(1, pad_width1 + 1, dtype=da.dtype), dims=dim)
        da1 = da[{dim: -1}] + (da[{dim: -1}] - da[{dim: -2}]) * ramp1
        to_concat.append(da1.transpose(*da.dims))

    return xr.concat(to_concat, dim=dim)


def pad(
    da,
    pad_width,
    mode="edge",
    coord_mode="linear_extrap",
    name_kwargs=None,
    **kwargs,
):
    """Pad data and coordinates along dimensions

    This function adds the ``"linear_extrap"`` mode support to the builtin
    :meth:`xarray.DataArray.pad` methods, and runs numba kernels for the
    ``"edge"``, ``"linear_extrap"``, ``"constant"`` and ``"wrap"`` modes.
    Longitudes are kept continuous across the dateline.

    Parameters
    ----------
    da: xarray.DataArray
    pad_width: dict
        Pad widths. Keys are dimensions and values are int or tuple of ints.
    mode: str
        Extrapolation mode for the data array, like ``"edge"``, ``"linear_extrap"``,
        ``"constant"`` or ``"wrap"``
    coord_mode: str
        Extrapolation mode for the coordinates
    name_kwargs: dict(dict)
        Keys are coordinates names and values are parameters to pass
        to :func:`xarray.pad` for this coordinate array
    kwargs:
        Extra arguments are passed to :func:`xarray.pad`

    Return
    ------
    xarray.DataArray

    See also
    --------
    get_centers
    get_edges
    apply_along_dim
    xarray.pad
    """
    pad_width = meta.get_meta_specs(da).parse_dims(pad_width, da)
    return apply_along_dim(
        da,
        list(pad_width.keys()),
        _pad_,
        data_kwargs={"mode": mode, "lon": _is_lon_data_(da), **kwargs},
        coord_kwargs={"mode": coord_mode},
        name_kwargs=_lon_names_(da, name_kwargs),
        pad_width=pad_width,
    )


def _get_centers_(da, dim, lon=False):
    if cnum.is_float(da) or lon:
        return xr.DataArray(cgrid.centers(da.values, da.get_axis_num(dim), lon), dims=da.dims)
    dao = da.isel({dim: slice(None, -1)})
    dao = dao + 0.5 * da.diff(dim).data
    return dao


def get_centers(da, dim):
    """Interpolate the data array at mid grid points along the `dim` dimension(s)

    .. note:: Coordinates are also centered

    Parameters
    ----------
    da: xarray.DataArray
    dim: str, tuple
        Single or tuple of data-array or generic dimension names.

    Return
    ------
    xarray.DataArray

    See also
    --------
    pad
    get_edges
    apply_along_dim
    """
    dim = meta.get_meta_specs(da).parse_dims(dim, da)
    return apply_along_dim(
        da,
        dim,
        _get_centers_,
        data_kwargs={"lon": _is_lon_data_(da)},
        name_kwargs=_lon_names_(da),
    )


def _get_edges_(da, dim, mode, lon=False, **kwargs):
    if mode == "linear_extrap" and (cnum.is_float(da) or lon):
        return xr.DataArray(cgrid.edges(da.values, da.get_axis_num(dim), lon), dims=da.dims)
    da = _pad_(da, dim, {dim: 1}, mode, lon, **kwargs)
    return _get_centers_(da, dim, lon)


def get_edges(da, dim, mode="linear_extrap", **kwargs):
    """Interpolate and extrapolate a data array at grid edges along the `dim` dimension(s)

    Inner edges are the middle of consecutive centers, and the outer edges are
    obtained by extrapolating the centers by default.

    .. note:: Coordinates are linearly extrapolated

    Parameters
    ----------
    da: xarray.DataArray
    dim: str, tuple
        Single or tuple of data-array or generic dimension names.
    mode: str
        Extrapolation mode at grid edges, which can be ``"linear_extrap"``
        or any mode of :func:`pad`, like ``"edge"`` to replicate the end values
    kwargs:
        Extra arguments are passed to :func:`pad`

    Return
    ------
    xarray.DataArray

    See also
    --------
    pad
    get_centers
    apply_along_dim
    """
    dim = meta.get_meta_specs(da).parse_dims(dim, da)
    return apply_along_dim(
        da,
        dim,
        _get_edges_,
        data_kwargs={"mode": mode, "lon": _is_lon_data_(da), **kwargs},
        coord_kwargs={"mode": "linear_extrap"},
        name_kwargs=_lon_names_(da),
    )


class shift_directions(misc.IntEnumChoices, metaclass=misc.XEnumMeta):
    """Shift directions for :func:`shift`"""

    #: To the left/bottom/west/south/low
    left = -1
    #: To the left/bottom/west/south/low
    bottom = -1
    #: To the left/bottom/west/south/low
    south = -1
    #: To the left/bottom/west/south/low
    low = -1
    #: To the left/bottom/west/south/low
    west = -1
    #: To the right/top/east/north/high
    right = 1
    #: To the right/top/east/north/high
    top = 1
    #: To the right/top/east/north/high
    north = 1
    #: To the right/top/east/north/high
    high = 1
    #: To the right/top/east/north/high
    east = 1


def _shift_(da, dim, direction, mode, lon=False, **kwargs):
    if mode in ("edge", "linear_extrap") and (cnum.is_float(da) or lon) and not kwargs:
        data = cgrid.shift(da.values, da.get_axis_num(dim), direction, mode, lon)
        return xr.DataArray(data, dims=da.dims)
    pad_width = (1, 0) if direction < 0 else (0, 1)
    return _get_centers_(_pad_(da, dim, {dim: pad_width}, mode, lon, **kwargs), dim, lon)


def shift(da, shift_dirs, mode="edge", **kwargs):
    """Shift the grid by an half grid cell along specified dimensions and directions

    This is typically useful with Arakawa grids.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    shift_dirs: dict
        Keys are dimension names and values are directions:
        {shift_directions.rst_with_links}
    mode: str
        Extrapolation mode at grid edges
    kwargs:
        Extra arguments are passed to :func:`pad`

    Return
    ------
    xarray.DataArray, xarray.Dataset

    See also
    --------
    pad
    get_edges
    get_centers
    """
    shift_dirs = meta.get_meta_specs(da).parse_dims(shift_dirs, da)
    name_kwargs = _lon_names_(da)
    data_lon = _is_lon_data_(da)
    for dim, shift_dir in shift_dirs.items():
        da = apply_along_dim(
            da,
            dim,
            _shift_,
            data_kwargs={"mode": mode, "lon": data_lon, **kwargs},
            coord_kwargs={"mode": "linear_extrap"},
            name_kwargs=name_kwargs,
            direction=shift_directions[shift_dir],
        )
    return da


shift.__doc__ = shift.__doc__.format(**locals())


def _diff_(da, dim, lon=False):
    if cnum.is_float(da) or lon:
        return xr.DataArray(cgrid.diff(da.values, da.get_axis_num(dim), lon), dims=da.dims)
    return da.diff(dim)


def diff(da, dim):
    """Compute the difference between consecutive grid points

    .. note:: Coordinates are centered between grid point with :func:`get_centers`

    Parameters
    ----------
    da: xarray.DataArray
    dim: str, tuple

    Return
    ------
    xarray.DataArray

    See also
    --------
    pad
    get_edges
    get_centers
    apply_along_dim
    """
    return apply_along_dim(
        da,
        dim,
        _diff_,
        coord_func=_get_centers_,
        data_kwargs={"lon": _is_lon_data_(da)},
        name_kwargs=_lon_names_(da),
    )


def _gradient_(da, dim, coord):
    if coord.dims != (dim,):
        coord = coord.broadcast_like(da).transpose(*da.dims)
    data = cgrid.gradient(da.values, coord.values, da.get_axis_num(dim))
    return xr.DataArray(data, dims=da.dims)


def gradient(da, dim, coord=None):
    """Derivative along a dimension with respect to a coordinate, at the grid centers

    The result is ``diff(da, dim) / diff(coord, dim)``, so its size is reduced by one
    along `dim` and its coordinates are centered like with :func:`diff`.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
    dim: str
    coord: xarray.DataArray, str, None
        Numeric coordinate, or its name, that is either 1D along `dim` or has the
        dimensions of `da`, like depths on sigma levels.
        It defaults to the coordinate of `dim`.

    Return
    ------
    xarray.DataArray, xarray.Dataset

    See also
    --------
    diff
    get_centers
    """
    dim = meta.get_meta_specs(da).parse_dims(dim, da)
    if coord is None:
        coord = da[dim]
    elif isinstance(coord, str):
        coord = da.coords[coord]
    coord = xr.DataArray(coord.data, dims=coord.dims)
    return apply_along_dim(
        da,
        dim,
        _gradient_,
        coord_func=_get_centers_,
        data_kwargs={"coord": coord},
        name_kwargs=_lon_names_(da),
    )


class dz2depth_ref_types(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Integration ref types for :func:`dz2depth`"""

    #: Infer it (default)
    infer = 0
    #: Up (SSH)
    top = 1
    #: Up (SSH)
    ssh = 1
    #: Bottom (bathy)
    bottom = -1
    #: Bottom (bathy)
    bathy = -1


def dz2z(dz, positive=None, zdim=None, top=None, bottom=None, centered=False):
    """Integrate layer thicknesses to compute z, which is positive up

    The output z are the coordinates at the interfaces of the layers
    and the top is at zero by default. Thus, the output array has
    one more level than the input array, along the vertical dimension, unless `centered`
    is True.

    Parameters
    ----------
    dz: xarray.DataArray
        Layer thicknesses
    positive: str, int, None
        Direction over which the levels of `dz` are increasing:
        {xcoords.positive_attr.rst_with_links}
        When "up", the first level is supposed to be the bottom.
        When "down", the first level is supposed to be the top.
        When "guess", the dz array must have an axis coordinate
        of the same name as the z dimension, and this coordinate must have
        a valid positive attribute.
    zdim: str
        Name of the vertical dimension.
        If not set, it is inferred with :func:`~xoa.coords.get_meta_dims`.
    top: None, float, xarray.DataArray
        The z of the top interface, i.e. the sea surface height. It defaults to zero
        when `bottom` is not provided.
    bottom: None, float, xarray.DataArray
        The z of the bottom interface, i.e. minus the bathymetry.
        It cannot be provided with `top`, since the layer thicknesses already
        give the total thickness.
    centered: bool
        Get z at the middle of layers instead of at their edge

    Return
    ------
    xarray.DataArray
        Output z, which are negative in the ocean.

    See also
    --------
    z2dz
    dz2depth
    get_dz

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.grid import dz2z
        @suppress
        import xarray as xr
        dz = xr.DataArray([1., 3., 4.], dims="nz")

        # First level at the top
        print(dz2z(dz, "down"))

        # First level at the bottom, with a sea surface height
        print(dz2z(dz, "up", top=0.5))
    """
    # Vertical dimension
    if zdim is None:
        zdim = xcoords.get_zdim(dz, errors="raise")
    positive = xcoords.get_positive_attr(dz, zdim, positive, errors="raise")
    if top is not None and bottom is not None:
        raise exceptions.XoaGridError(
            "top and bottom cannot be both provided, since the total thickness is known"
        )

    # Integrate
    cum = dz.cumsum(dim=zdim)
    cum = pad(cum, {zdim: (1, 0)}, mode="constant", constant_values=0)
    total = cum.isel({zdim: -1})
    if positive == "up":
        if bottom is None:
            bottom = (0 if top is None else top) - total
        z = cum + bottom
    else:
        if top is None:
            top = 0 if bottom is None else bottom + total
        z = top - cum

    # Fix index
    if zdim in z.indexes:
        dnz = z[zdim].diff(zdim).pad({zdim: (0, 1)}, mode="edge")
        z = xcoords.change_index(z, zdim, z[zdim] + 0.5 * dnz.data)

    # Centered
    if centered:
        z = get_centers(z, zdim)
        if zdim in z.indexes:
            z = z.assign_coords({zdim: dz[zdim]})

    # Finalize
    z.attrs["positive"] = "up"
    return meta.get_meta_specs(dz).format_coord(
        z, "z", rename=True, format_coords=False, rename_dims=False
    )


dz2z.__doc__ = dz2z.__doc__.format(**locals())


def dz2depth(dz, positive=None, zdim=None, ref=None, ref_type="infer", centered=False):
    """Integrate layer thicknesses to compute depths

    The output depths are the depths at the bottom of the layers and the top
    is at a depth of zero. Thus, the output array has the same dimensions
    as the input array of layer thicknesses.

    Parameters
    ----------
    dz: xarray.DataArray
        Layer thicknesses
    positive: str, int, None
        Direction over which coordinates are increasing:
        {xcoords.positive_attr.rst_with_links}
        When "up", the first level is supposed to be the bottom
        and the output coordinates are negative, as with :func:`dz2z`.
        When "down", first level is supposed to be the top
        and the output coordinates are positive.
        When "guess", the dz array must have an axis coordinate
        of the same name as the z dimension, and this coordinate must have
        a valid positive attribute.
    zdim: str
        Name of the vertical dimension.
        If not set, it is inferred with :func:`~xoa.coords.get_meta_dims`.
    ref: xarray.DataArray
        Reference array converting layer thicknesses to depth.
        When it is not provided, the top is at zero.
        The type of reference is given by `ref_type`, and, if it is not set,
        it is inferred from the meta specs of `ref` or, by default,
        it is a **bathymetry** if **positive up** and a **top** if **positive down**.
        A bathymetry is always positive, and a top is the sea surface height if positive up
        and its depth if positive down.
    ref_type: str, int
        Type of `ref`:
        {dz2depth_ref_types.rst_with_links}
    centered: bool
        Get depth at the middle of layers instead of at their edge

    Return
    ------
    xarray.DataArray
        Output depths with the same dimensions as input array.

    See also
    --------
    dz2z

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.grid import dz2depth
        @suppress
        import xarray as xr
        dz = xr.DataArray([1., 3., 4.], dims="nz")

        # Positive down
        print(dz2depth(dz, "down"))

        # Positive up
        print(dz2depth(dz, "up"))
    """
    if zdim is None:
        zdim = xcoords.get_zdim(dz, errors="raise")
    positive = xcoords.get_positive_attr(dz, zdim, positive, errors="raise")

    # Reference as top and bottom
    top = bottom = None
    if ref is not None:
        ref_type = dz2depth_ref_types[ref_type].name
        if ref_type == "infer":
            specs = meta.get_meta_specs(dz).data_vars
            if isinstance(ref, xr.DataArray) and specs.match(ref, "bathy"):
                ref_type = "bottom"
            elif isinstance(ref, xr.DataArray) and specs.match(ref, "ssh"):
                ref_type = "top"
            else:
                ref_type = "bottom" if positive == "up" else "top"
        if ref_type == "bottom":
            bottom = -ref
        else:
            top = ref if positive == "up" else -ref

    z = dz2z(dz, positive, zdim, top=top, bottom=bottom, centered=centered)
    if positive == "up":
        return z
    depth = 0 - z
    depth.attrs["positive"] = "down"
    return meta.get_meta_specs(dz).format_coord(
        depth, "depth", rename=True, format_coords=False, rename_dims=False
    )


dz2depth.__doc__ = dz2depth.__doc__.format(**locals())


@misc.ERRORS.format_function_docstring
def get_dz_terms(ds, errors="raise"):
    """Search a dataset for the layer thicknesses and what is needed to integrate them

    Parameters
    ----------
    ds: xarray.Dataset
    {errors}

    Return
    ------
    tuple or None
        The thicknesses, the vertical dimension, the positive attribute,
        the sea surface height and the bathymetry, the last two being None when not found

    See also
    --------
    decode_dz2z
    decode_dz2depth
    xoa.sigma.get_sigma_terms
    """
    errors = misc.ERRORS[errors]
    meta_specs = meta.get_meta_specs(ds)
    dz = meta_specs.search(ds, "dz", errors=errors)
    if dz is None:
        return
    zdim = xcoords.get_meta_dims(dz, "z", errors=errors)
    if zdim is None:
        return
    positive = xcoords.get_positive_attr(
        ds, zdim, meta_specs["vertical"]["positive"], errors=errors
    )
    if positive is None:
        return
    ssh = meta_specs.search(ds, "ssh", errors="ignore")
    bathy = meta_specs.search(ds, "bathy", errors="ignore")
    return dz, zdim, positive, ssh, bathy


@misc.ERRORS.format_function_docstring
def decode_dz2z(ds, errors="raise", **kwargs):
    """Compute z from layer thickness in a dataset

    This makes use of the :meth:`~xoa.meta.MetaSpecs` instance that is retrieved
    with :func:`xoa.meta.get_meta_specs` with ds as an argument in order to
    find needed variables.
    The integration starts from the bathymetry if the first level is the bottom,
    else from the sea surface height, with a fallback on the other one.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset that contains everything
    {errors}
    kwargs: dict
        Extra keywords are passed to :func:`dz2z`

    Return
    ------
    xarray.Dataset
        A new dataset with a z coordinate

    See also
    --------
    dz2z
    decode_dz2depth
    xoa.meta.get_meta_specs
    """
    ds = ds.copy()
    errors = misc.ERRORS[errors]
    found = get_dz_terms(ds, errors)
    if found is None:
        return ds
    dz, zdim, positive, ssh, bathy = found
    top = bottom = None
    for ref, ref_type in [(bathy, "bottom"), (ssh, "top")][:: 1 if positive == "up" else -1]:
        if ref is not None:
            top, bottom = (ref, None) if ref_type == "top" else (None, -ref)
            break
    z = dz2z(dz, positive=positive, zdim=zdim, top=top, bottom=bottom, centered=True, **kwargs)
    if z.name in ds.dims:
        msg = f"Can't assign the {z.name} coordinate since a dimension has the same name"
        if errors == "raise":
            raise exceptions.XoaGridError(msg)
        if errors == "warn":
            exceptions.xoa_warn(msg)
        return ds
    return ds.assign_coords({z.name: z})


@misc.ERRORS.format_function_docstring
def decode_dz2depth(ds, errors="raise", **kwargs):
    """Compute depth from layer thickness in a dataset

    This makes use of the :meth:`~xoa.meta.MetaSpecs` instance that is retrieved
    with :func:`xoa.meta.get_meta_specs` with ds as an argument in order to
    find needed variables.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset that contains everything
    {errors}
    kwargs: dict
        Extra keywords are passed to :func:`dz2depth`

    Return
    ------
    xarray.Dataset
        A new dataset with a depth coordinate if positive down, else a z coordinate

    See also
    --------
    dz2depth
    decode_dz2z
    xoa.meta.get_meta_specs
    """
    ds = ds.copy()
    errors = misc.ERRORS[errors]
    found = get_dz_terms(ds, errors)
    if found is None:
        return ds
    dz, zdim, positive, ssh, bathy = found

    # Make choices
    if ssh is None and bathy is None:
        ref, ref_type = None, "infer"
    else:
        for ref, ref_type in [(bathy, "bathy"), (ssh, "ssh")][:: 1 if positive == "up" else -1]:
            if ref is not None:
                break

    # Compute depth
    depth = dz2depth(
        dz,
        positive=positive,
        zdim=zdim,
        ref=ref,
        ref_type=ref_type,
        centered=True,
        **kwargs,
    )
    if depth.name in ds.dims:
        msg = f"Can't assign the {depth.name} coordinate since a dimension has the same name"
        if errors == "raise":
            raise exceptions.XoaGridError(msg)
        if errors == "warn":
            exceptions.xoa_warn(msg)
        return ds
    return ds.assign_coords({depth.name: depth})


def decode_cf_dz2depth(*args, **kwargs):
    exceptions.xoa_warn(
        "decode_cf_dz2depth is deprecated. Please use decode_dz2depth instead", "deprecation"
    )
    return decode_dz2depth(*args, **kwargs)


def z2dz(z, zdim=None, top=None, bottom=None):
    """Compute layer thicknesses from the z or depth of the layers

    The interfaces are the middle of consecutive levels, and the outer ones are
    extrapolated with :func:`get_edges`, unless `top` and `bottom` are provided.
    The coordinates may vary along the other dimensions, like with sigma coordinates.

    .. note:: This is exact for evenly spaced levels, like uniform sigma layers. Otherwise
        the inner interfaces are approximate, and the total thickness is exact
        only if both `top` and `bottom` are provided.

    Parameters
    ----------
    z: xarray.DataArray
        Coordinates at the center of the layers, either positive up or down
        as given by the ``positive`` attribute, or guessed from the values like in
        :func:`~xoa.coords.as_z`.
    zdim: str
        Name of the vertical dimension.
        If not set, it is inferred with :func:`~xoa.coords.get_meta_dims`.
    top: None, float, xarray.DataArray
        The z of the top interface, i.e. the sea surface height,
        instead of the extrapolated one.
    bottom: None, float, xarray.DataArray
        The z of the bottom interface, i.e. minus the bathymetry,
        instead of the extrapolated one.

    Return
    ------
    xarray.DataArray
        Positive layer thicknesses, with the dimensions of `z`

    See also
    --------
    dz2z
    get_dz
    get_edges
    """
    z = xcoords.as_z(z)
    if zdim is None:
        zdim = xcoords.get_zdim(z, errors="raise")
    edges = get_edges(z, zdim)
    if top is not None or bottom is not None:
        # The orientation of the axis tells where the top is
        up = float((z.isel({zdim: -1}) - z.isel({zdim: 0})).mean()) > 0
        edges = edges.copy()
        for value, last in ((top, up), (bottom, not up)):
            if value is not None:
                edges[{zdim: -1 if last else 0}] = value
    dz = abs(edges.isel({zdim: slice(1, None)}).data - edges.isel({zdim: slice(None, -1)}).data)
    attrs = {"long_name": "Layer thickness"}
    if "units" in z.attrs:
        attrs["units"] = z.attrs["units"]
    return z.copy(data=dz).rename("dz").assign_attrs(attrs)


@misc.ERRORS.format_function_docstring
def get_dz(obj, errors="raise", **kwargs):
    """Get or compute the layer thicknesses

    A thickness variable is searched first. Else it is computed from the z or depth
    coordinate that is returned by :func:`~xoa.coords.get_z` with :func:`z2dz`.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
    {errors}
    kwargs:
        Extra parameters are passed to :func:`z2dz`

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    z2dz
    dz2z
    xoa.coords.get_z
    """
    errors = misc.ERRORS[errors]
    dz = meta.get_meta_specs(obj).search(obj, "dz", errors="ignore")
    if dz is not None:
        return dz
    z = xcoords.get_z(obj, errors=errors)
    if z is not None:
        return z2dz(z, **kwargs)


def _get_vweights_(da, zmin, zmax, dz, zdim):
    """Get the layer thicknesses of the layers to integrate and the vertical dimension"""
    if zdim is None:
        zdim = xcoords.get_zdim(da, errors="raise")
    if dz is None:
        dz = get_dz(da, zdim=zdim)
    if zmin is not None or zmax is not None:
        # Part of each layer, centered on its z, that lies within the bounds
        z = xcoords.as_z(xcoords.get_z(da))
        top = z + dz / 2 if zmax is None else np.minimum(z + dz / 2, zmax)
        bottom = z - dz / 2 if zmin is None else np.maximum(z - dz / 2, zmin)
        dz = (top - bottom).where(top > bottom).where(dz.notnull())
    return dz, zdim


def vintegrate(da, zmin=None, zmax=None, dz=None, zdim=None):
    """Integrate along the vertical, accounting for the layer thicknesses

    This is the sum of the values times the thicknesses of the layers, where the
    values are valid. Contrary to a simple sum or mean along the vertical dimension, it is
    right with variable thicknesses, which is the rule in the ocean.

    Parameters
    ----------
    da: xarray.DataArray
        Array with a vertical dimension, and either a z or depth coordinate
        or a thickness coordinate when `dz` is not provided
    zmin: None, float
        Lower bound of the layers to integrate, in z, i.e. negative in the ocean.
        The layers that cross it are partly accounted for, assuming that their
        z is at their center.
    zmax: None, float
        Upper bound, in z, treated like `zmin`.
    dz: None, xarray.DataArray
        Layer thicknesses, which are searched or computed with :func:`get_dz` by default
    zdim: None, str
        Name of the vertical dimension, which is inferred by default

    Return
    ------
    xarray.DataArray
        Integral, with the vertical dimension removed. It is masked
        where there is no valid value.

    See also
    --------
    vmean
    get_dz
    """
    dz, zdim = _get_vweights_(da, zmin, zmax, dz, zdim)
    out = (da * dz).sum(zdim, min_count=1)
    attrs = {"long_name": f"Vertical integral of {da.attrs.get('long_name', da.name)}"}
    if "units" in da.attrs:
        attrs["units"] = f"{da.attrs['units']} m"
    return out.rename(da.name).assign_attrs(attrs)


def vmean(da, zmin=None, zmax=None, dz=None, zdim=None):
    """Average along the vertical, weighted by the layer thicknesses

    The layers where the values are not valid are ignored.
    Contrary to ``da.mean(zdim)``, it is right with variable thicknesses.

    Parameters
    ----------
    da: xarray.DataArray
        Array with a vertical dimension, and either a z or depth coordinate
        or a thickness coordinate when `dz` is not provided
    zmin: None, float
        Lower bound of the layers to average, in z, i.e. negative in the ocean.
        The layers that cross it are partly accounted for, assuming that their
        z is at their center.
    zmax: None, float
        Upper bound, in z, treated like `zmin`.
    dz: None, xarray.DataArray
        Layer thicknesses, which are searched or computed with :func:`get_dz` by default
    zdim: None, str
        Name of the vertical dimension, which is inferred by default

    Return
    ------
    xarray.DataArray
        Mean, with the vertical dimension removed. It is masked
        where there is no valid value.

    See also
    --------
    vintegrate
    get_dz
    """
    dz, zdim = _get_vweights_(da, zmin, zmax, dz, zdim)
    dz = dz.where(da.notnull())
    out = (da * dz).sum(zdim, min_count=1) / dz.sum(zdim, min_count=1)
    return out.rename(da.name).assign_attrs(
        {k: v for k, v in da.attrs.items() if k in ("long_name", "units")}
    )


@misc.ERRORS.format_function_docstring
def to_rect(da, tol=1e-5, errors="warn"):
    """Convert the curvilinear coordinates of array/dataset to rectangular axis coordinates

    It checks if the coordinates may be converted to 1D  axis without loss of information.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        In case of a dataset, it must contain longitudes and latitudes.
    tol: float
        Absolute tolerance of the variability of a coordinate along its constant dimension
        to consider it as a 1D axis coordinate.
    {errors}

    Return
    ------
    xarray.DataArray, xarray.Dataset
    """
    new_coords = {}
    rename_args = {}
    da = meta.infer_coords(da)
    errors = misc.ERRORS[errors]
    coords2d = {name: coord for name, coord in da.coords.items() if coord.ndim == 2}
    for lon_name, lon in coords2d.items():
        if not xcoords.is_lon(lon):
            continue
        # The latitude that shares the dimensions of this longitude
        lat_name = None
        for name, coord in coords2d.items():
            if xcoords.is_lat(coord) and set(coord.dims) == set(lon.dims):
                lat_name, lat = name, coord
                break
        if lat_name is None:
            continue

        # Check with the core function after ordering dimensions as (y, x)
        ydim = xcoords.get_ydim(lon, errors="ignore")
        xdim = xcoords.get_xdim(lon, errors="ignore")
        if ydim is None or xdim is None:
            ydim, xdim = lon.dims
        grid_type = cgrid.check_grid_type(
            {"lon": lon.transpose(ydim, xdim).values, "lat": lat.transpose(ydim, xdim).values},
            tol=tol,
        )
        if grid_type == "curvilinear":
            msg = (
                "Cannot convert curvilinear to rectangular grid since coordinates "
                f"'{lon_name}' and '{lat_name}' are not constant along one of their dimensions"
            )
            if errors == "raise":
                raise exceptions.XoaError(msg)
            elif errors == "warn":
                exceptions.xoa_warn(msg)
            continue

        # Axis coordinates
        for name, coord, odim, dim in (
            (lon_name, lon, ydim, xdim),
            (lat_name, lat, xdim, ydim),
        ):
            new_coords[name] = xr.DataArray(
                coord.isel({odim: 0}).data, dims=name, attrs=coord.attrs
            )
            new_coords[name].encoding.update(coord.encoding)
            rename_args[dim] = name
    if new_coords:
        return (
            da.reset_coords(list(new_coords), drop=True)
            .rename(rename_args)
            .assign_coords(new_coords)
        )
    return da


def ds2grid_dict(obj, mask=None, lon_name=None, lat_name=None, time_name=None, bounds=False):
    """Convert a data array or dataset to a horizontal grid dictionary

    The dictionary is suitable for the core interpolation and regridding
    classes :class:`xoa.core.interp.XYInterpolator` and
    :class:`xoa.core.regrid.XYRegridder`.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D
    mask: str, xarray.DataArray, array_like, None
        Name of a variable or array of valid points (True)
    lon_name, lat_name, time_name: str, None
        Names of the longitude, latitude and time coordinates.
        They are searched with :mod:`xoa.coords` when not provided.
    bounds: bool
        Compute the edges and bounds of the cells, which are large arrays and
        that the core classes compute by themselves when they need them.

    Return
    ------
    dict
        With the following keys: ``lon``, ``lat``, ``shape``, ``dims``, ``sizes``,
        ``lon_name``, ``lat_name``, ``coords``, ``type``, ``lon_edges``,
        ``lat_edges``, ``lon_bounds``, ``lat_bounds``, and optionally ``mask`` and
        ``time_name``. Edges and bounds are only present if ``bounds`` is True.
    """

    def _get(name, getter):
        if name is not None:
            return obj[name] if name in obj else obj.coords[name]
        return getter(obj)

    lon = _get(lon_name, xcoords.get_lon)
    lat = _get(lat_name, xcoords.get_lat)

    # Broadcast to ensure same shape
    latb, lonb = xr.broadcast(lat, lon)

    grid = {
        "lon": lonb.values,
        "lat": latb.values,
        "shape": lonb.shape,
        "lon_name": lon.name,
        "lat_name": lat.name,
        "dims": lonb.dims,
        "sizes": lonb.sizes,
        "coords": {lon.name: lon, lat.name: lat},
    }
    grid["type"] = cgrid.check_grid_type(grid)

    # Time
    time = _get(time_name, lambda o: xcoords.get_time(o, errors="ignore"))
    if time is not None:
        grid["time_name"] = time.name
        grid["coords"][time.name] = time

    # Bounds and edges
    if bounds:
        grid["lon_edges"] = cgrid.centers2edges(grid["lon"])
        grid["lon_bounds"] = cgrid.edges2bounds(grid["lon_edges"])
        grid["lat_edges"] = cgrid.centers2edges(grid["lat"])
        grid["lat_bounds"] = cgrid.edges2bounds(grid["lat_edges"])

    # Mask
    if isinstance(mask, str):
        grid["mask"] = obj[mask].values
    elif mask is not None:
        grid["mask"] = mask.values if hasattr(mask, "values") else mask

    return grid


def get_lonlat_2d(obj):
    """Get 2D longitudes and latitudes with (y, x) dimensions

    1D longitudes and latitudes of a rectilinear grid are broadcasted
    against each other, and 2D ones of a curvilinear grid are returned
    as is. In both cases, the arrays are transposed to (y, x).

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D

    Return
    ------
    xarray.DataArray
        2D longitudes with (y, x) dimensions
    xarray.DataArray
        2D latitudes with (y, x) dimensions

    Raises
    ------
    xoa.exceptions.XoaError
        When the broadcasted longitudes and latitudes are not 2D

    See also
    --------
    xoa.coords.get_lon
    xoa.coords.get_lat
    """
    lon = xcoords.get_lon(obj)
    lat = xcoords.get_lat(obj)
    lat, lon = xr.broadcast(lat, lon)
    if lon.ndim != 2:
        raise exceptions.XoaError(
            f"Longitudes and latitudes must be 2D, but got {lon.ndim} dimensions"
        )
    ydim = xcoords.get_ydim(lon, errors="ignore")
    xdim = xcoords.get_xdim(lon, errors="ignore")
    if ydim is None or xdim is None:
        ydim, xdim = lon.dims
    return lon.transpose(ydim, xdim), lat.transpose(ydim, xdim)


def get_resolution(obj, radius=xgeo.EARTH_RADIUS):
    """Compute the horizontal grid resolution along x and y

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D
    radius: float
        Radius of the sphere in meters, which defaults to the earth radius

    Return
    ------
    xarray.DataArray
        Distance in meters between adjacent points along x, with one point
        less than the grid along its last dimension.
    xarray.DataArray
        Distance in meters between adjacent points along y, with one point
        less than the grid along its first dimension.

    See also
    --------
    get_median_resolution
    xoa.core.grid.compute_resolution
    """
    lon, lat = get_lonlat_2d(obj)
    dx, dy = cgrid.compute_resolution(lon.values, lat.values, radius=radius)
    attrs = {"units": "m"}
    return (
        xr.DataArray(
            dx, dims=lon.dims, name="dx", attrs={"long_name": "Resolution along x", **attrs}
        ),
        xr.DataArray(
            dy, dims=lon.dims, name="dy", attrs={"long_name": "Resolution along y", **attrs}
        ),
    )


def get_dx_dy(obj, radius=xgeo.EARTH_RADIUS):
    """Compute the horizontal grid resolution at the grid points

    Contrary to :func:`get_resolution`, the resolutions have the shape
    of the grid since they are centered on the grid points. They are suited
    for centered finite differences, like in :mod:`xoa.dyn`.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D
    radius: float
        Radius of the sphere in meters, which defaults to the earth radius

    Return
    ------
    xarray.DataArray
        Resolution in meters along x
    xarray.DataArray
        Resolution in meters along y

    See also
    --------
    get_resolution
    xoa.core.grid.compute_centered_resolution
    """
    lon, lat = get_lonlat_2d(obj)
    dx, dy = cgrid.compute_centered_resolution(lon.values, lat.values, radius=radius)
    attrs = {"units": "m"}
    return (
        xr.DataArray(
            dx,
            dims=lon.dims,
            coords=lon.coords,
            name="dx",
            attrs={"long_name": "Resolution along x", **attrs},
        ),
        xr.DataArray(
            dy,
            dims=lon.dims,
            coords=lon.coords,
            name="dy",
            attrs={"long_name": "Resolution along y", **attrs},
        ),
    )


def get_median_resolution(obj):
    """Compute the median horizontal grid resolution in degrees

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D

    Return
    ------
    float

    See also
    --------
    get_resolution
    xoa.core.grid.median_resolution_deg
    """
    lon, lat = get_lonlat_2d(obj)
    return cgrid.median_resolution_deg(lon.values, lat.values)


def get_edge_extents(obj, edges="all", n_cells=1):
    """Get the geographic extent of the strips along the edges of a grid

    The strips are made of the first or last cells of the grid, so that they follow
    the grid when it is rotated or curvilinear.
    North and east are the last indices along the y and x dimensions,
    and south and west are the first ones.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D
    edges: str, list(str)
        Edge names among ``"north"``, ``"south"``, ``"east"``, ``"west"``
        and ``"all"``
    n_cells: int
        Number of grid cells in each strip, counted from the edge

    Return
    ------
    dict
        Keys are edge names and values are extents
        ``[xmin, xmax, ymin, ymax]``, as returned by :func:`xoa.geo.get_extent`

    See also
    --------
    xoa.geo.get_extent
    """
    lon, lat = get_lonlat_2d(obj)
    ydim, xdim = lon.dims
    slices = {
        "north": {ydim: slice(-n_cells, None)},
        "south": {ydim: slice(None, n_cells)},
        "east": {xdim: slice(-n_cells, None)},
        "west": {xdim: slice(None, n_cells)},
    }
    if isinstance(edges, str):
        edges = [edges]
    names = []
    for edge in edges:
        for name in slices if edge == "all" else [edge]:
            if name not in slices:
                raise exceptions.XoaError(
                    f"Invalid edge '{edge}'. Choose among: all, {', '.join(slices)}"
                )
            if name not in names:
                names.append(name)
    return {
        name: xgeo.get_extent((lon.isel(slices[name]).values, lat.isel(slices[name]).values))
        for name in names
    }


def get_fingerprint(obj, mask=None):
    """Get the fingerprint of a horizontal grid

    The fingerprint depends on the longitudes, latitudes and mask of the grid, whatever
    their dimensions are. Equal grids have the same fingerprint, so it is the way
    to recognise a grid, for instance in a weights file, where the weights
    of a regridding or an interpolation are stored in a group that is named after
    the fingerprints of the grids. See :mod:`xoa.weights`.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset, dict
        Object with longitude and latitude coordinates, that may be 1D or 2D,
        or a grid dictionary like the one of :func:`ds2grid_dict`
    mask: str, xarray.DataArray, array_like, None
        Valid points of the grid, which is ignored for a dictionary

    Return
    ------
    str

    Example
    -------
    .. code-block:: python

        fingerprint = xoa.grid.get_fingerprint(ds)
        xoa.weights.find_groups("weights.nc", fingerprint)

    See also
    --------
    ds2grid_dict
    xoa.misc.get_array_fingerprint
    xoa.weights.find_groups
    """
    grid = obj if isinstance(obj, dict) else ds2grid_dict(obj, mask)
    return misc.get_array_fingerprint(grid["lon"], grid["lat"], grid.get("mask"))
