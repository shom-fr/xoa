# -*- coding: utf-8 -*-
"""
This module provides 1d to nD grid utilities to get information
or perform operation on a grid.

For operations between different grids, please see :mod:`xoa.regrid`.
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

from .__init__ import XoaError, xoa_warn
from . import misc
from . import cf
from . import coords as xcoords


def apply_along_dim(
        ds, dim, func, coord_func=None,
        data_kwargs=None, coord_kwargs=None, name_kwargs=None,
        **kwargs):
    """Apply an operator on data array or dataset dimensions

    The operator may potentially change size of the array.
    It is is applied on the data array with the data_kwargs
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
        Fonction to apply to coordinates specifically, which defaults
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
    dim = cf.get_cf_specs(ds).parse_dims(dim, ds)
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
                kw = kwargs or {}
                for dd in (coord_kwargs, name_kwargs.get(coord_name)):
                    if dd:
                        kw.update(dd)
                coord = coord_func(xr.DataArray(old_coord.data, dims=old_coord.dims), dim, **kw)
                coord.attrs = old_coord.attrs
                coord.encoding = old_coord.encoding
            else:
                coord = old_coord
            coords[coord_name] = coord
        dso = dso.assign_coords(coords)

    cf.assign_cf_specs(dso, ds)

    return dso


def _pad_(da, dim, pad_width, mode, **kwargs):

    pad_width = pad_width.get(dim, 0)
    if not pad_width:
        return da.copy()

    if mode != "linear_extrap":
        return da.pad({dim: pad_width}, mode=mode, **kwargs)

    to_concat = []
    if isinstance(pad_width, int):
        pad_width = pad_width,
    pad_width0 = pad_width[0]
    pad_width1 = pad_width[-1]
    if not pad_width0 and not pad_width1:
        return da
    if pad_width0:
        ramp0 = xr.DataArray(np.arange(pad_width0, 0, -1, dtype=da.dtype), dims=dim)
        da0 = da[{dim: 0}] + (da[{dim: 0}] - da[{dim: 1}]) * ramp0
        to_concat.append(da0.transpose(*da.dims))
    to_concat.append(da)
    if pad_width1:
        ramp1 = xr.DataArray(np.arange(1, pad_width1+1, dtype=da.dtype), dims=dim)
        da1 = da[{dim: -1}] + (da[{dim: -1}] - da[{dim: -2}]) * ramp1
        to_concat.append(da1.transpose(*da.dims))

    return xr.concat(to_concat, dim=dim)


def pad(da, pad_width, mode="edge", coord_mode="linear_extrap", name_kwargs=None, **kwargs):
    """Pad data and coordinates along dimensions

    This function adds the ``"linear_extrap"`` mode support to the builtin
    :meth:`xarray.DataArray.pad` methods.

    Parameters
    ----------
    da: xarray.DataArray
    pad_width: dict
        Pad widths. Keys are dimensions and values are int or tuple of ints.
    mode: str
        Extrapolation mode for the data array
    coord_mode: str
        Extrapolation mode for the coordinates
    name_kwargs: dict(dict)
        Keys are coordinates names and valkues are parameters to pass
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
    pad_width = cf.get_cf_specs(da).parse_dims(pad_width, da)
    return apply_along_dim(
        da, list(pad_width.keys()), _pad_,
        data_kwargs={"mode": mode, **kwargs},
        coord_kwargs={"mode": coord_mode},
        name_kwargs=name_kwargs,
        pad_width=pad_width)


def _get_centers_(da, dim):
    dao = da.isel({dim: slice(None, -1)})
    dao = dao + 0.5 * da.diff(dim).data
    return dao


def get_centers(da, dim):
    """Interpolate the data array at mid grid points

    .. note:: Coordinates are also centered

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
    apply_along_dim
    """
    dim = cf.get_cf_specs(da).parse_dims(dim, da)
    return apply_along_dim(da, dim, _get_centers_)


def get_edges(da, dim, mode="edge", **kwargs):
    """Interpolate and extrapolate a data array at grid edges

    .. note:: Coordinates are linearly extrapolated

    Parameters
    ----------
    da: xarray.DataArray
    dim: str, tuple
    mode: str
        Extrapolation mode at grid edges
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
    # Extrapolate
    dim = cf.get_cf_specs(da).parse_dims(dim, da)
    dims = (dim,) if isinstance(dim, str) else dim
    pad_width = dict((dim, 1) for dim in dims)
    da = pad(da, pad_width=pad_width, mode=mode, **kwargs)

    # Inner edges
    return get_centers(da, dim)


class shift_directions(misc.IntEnumChoices, metaclass=misc.XEnumMeta):
    """Shift directions for :func:`shift``"""
    #: To the left/bottom/west
    left = -1
    bottom = -1
    south = -1
    low = -1
    #: To the right/top/north
    right = 1
    top = 1
    north = 1
    high = 1


def shift(da, shift_dirs, mode="edge", **kwargs):
    f"""Shift the grid by an half grid cell along specified dimensions and directions

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
    shift_dirs = cf.get_cf_specs(da).parse_dims(shift_dirs, da)

    # Extrapolate
    pad_width = {}
    for dim, shift_dir in shift_dirs.items():
        pad_width[dim] = (1, 0) if shift_directions[shift_dir] < 0 else (0, 1)
    da = pad(da, pad_width=pad_width, mode=mode, **kwargs)

    # Inner edges
    return get_centers(da, list(shift_dirs.keys()))


def _diff_(da, dim):
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
    return apply_along_dim(da, dim, _diff_, coord_func=_get_centers_)


class dz2depth_ref_types(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Integration ref types for :func:`dz2depth`"""
    #: Infer it (default)
    infer = 0
    #: Up (SSH)
    top = 1
    ssh = 1
    #: Bottom (bathy)
    bottom = -1
    bathy = -1


def dz2depth(
        dz, positive=None, zdim=None, ref=None, ref_type='infer',
        centered=False):
    f"""Integrate layer thicknesses to compute depths

    The output depths are the depths at the bottom of the layers and the top
    is at a depth of zero. Thus, the output array has the same dimensions
    as the input array of layer thinknesses.

    Parameters
    ----------
    dz: xarray.DataArray
        Layer thinknesses
    positive: str, int, None
        Direction over wich coordinates are increasing:
        {xcoords.positive_attr.rst_with_links}
        When "up", the first level is supposed to be the bottom
        and the output coordinates are negative.
        When "down", first level is supposed to be the top
        and the output coordinates are positive.
        When "guess", the dz array must have an axis coordinate
        of the same name as the z dimension, and this coordinate must have
        a valid positive attribute.
    zdim: str
        Name of the vertical dimension.
        If note set, it is infered with :func:`~xoa.coords.get_cf_dims`.
    ref: xarray.DataArray
        Reference array converting layer thicknesses to depth:

        - If **positive up", it is expected to be the **SSH** (sea surface heigth)
          by default
        - If **positive down", it is expected to be by default the depth of ground
          also known as **bathymetry**, which should be positive.

    ref_type: str, int
        Type of ``ref``:
        {dz2depth_ref_types.rst_with_links}
    centered: bool
        Get depth a the middle of layers instead at their edge

    Return
    ------
    xr.DataArray
        Output depths with the same dimensions as input array.

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
    # Vertical dimension
    if zdim is None:
        zdim = xcoords.get_zdim(dz, errors="raise")

    # Positive attribute
    positive = xcoords.positive_attr[positive].name
    if positive == "infer":
        positive = xcoords.get_positive_attr(dz, zdim)
        if positive is None:
            raise XoaError("Can't infer positive attribute from data array/dataset")

    # Integrate
    depth = dz.cumsum(dim=zdim)
    depth = pad(depth, {zdim: (1, 0)}, mode="constant", constant_values=0)
    ref_type = dz2depth_ref_types[ref_type].name
    cfspecs = cf.get_cf_specs(dz)
    if ref is None and ref_type == "infer":
        if cfspecs.data_vars.match(ref, "bathy"):
            ref_type = "bottom"
        elif cfspecs.data_vars.match(ref, "ssh"):
            ref_type = "top"
        else:
            ref_type = "top" if positive == "down" else "bottom"
    if positive == "up":
        if ref is None:
            ref = depth[-1]
        elif ref is not None and ref_type == "top":
            ref = depth[-1] - ref
        depth[:] -= ref
    else:
        if ref is not None:
            if ref_type == "bottom":
                depth[:] -= depth[-1]
            depth[:] += ref

    # Fix index
    if zdim in depth.indexes:
        dnz = depth[zdim].diff(zdim).pad({zdim: (0, 1)}, mode="edge")
        depth = xcoords.change_index(depth, zdim, depth[zdim]+0.5*dnz.data)

    # Centered
    if centered:
        depth = get_centers(depth, zdim)
        if zdim in depth.indexes:
            depth = depth.assign_coords({zdim: dz[zdim]})

    # Finalize
    depth.attrs["positive"] = positive
    depth = cfspecs.format_coord(depth, "depth", rename=False, format_coords=False)

    return depth


@misc.ERRORS.format_function_docstring
def decode_cf_dz2depth(ds, errors="raise", **kwargs):
    """Compute depth from layer thickness in a dataset

    This makes use of the :meth:`~xoa.cf.CFSpecs` instance that is retreived
    with :func:`xoa.cf.get_cf_specs` with ds as an argument in order to
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
        A new dataset with a depth coordinate

    See also
    --------
    dz2depth
    xoa.cf.get_cf_specs
    """
    ds = ds.copy()
    errors = misc.ERRORS[errors]

    # Find needed stuff
    cfspecs = cf.get_cf_specs(ds)
    dz = cfspecs.search(ds, 'dz', errors=errors)
    if dz is None:
        return ds
    zdims = xcoords.get_cf_dims(dz, "z", errors=errors)
    if zdims is None:
        return ds
    zdim = zdims[0]
    positive = cfspecs["vertical"]["positive"]
    if positive is None:
        positive = xcoords.get_positive_attr(ds, zdim)
    if positive is None:
        msg = "Can't infer positive attribute from data dataset"
        if errors == "raise":
            raise XoaError(msg)
        xoa_warn(msg)
        return ds
    ssh = cfspecs.search(ds, 'ssh', errors="ignore")
    bathy = cfspecs.search(ds, 'bathy', errors="ignore")

    # Make choices
    if ssh is None and bathy is None:
        ref, ref_type = None, "infer"
    else:
        for ref, ref_type in [(bathy, "bathy"), (ssh, "ssh")][::int(positive)]:
            if ref is not None:
                break

    # Compute depth
    depth = dz2depth(dz, positive=positive, zdim=zdim, ref=ref, ref_type=ref_type, centered=True)

    # Assign to dataset
    return ds.assign_coords(depth=depth)


def to_rect(da, tol=1e-5):
    """Convert a curvilinear coordinate array to a rectangular 1d coordinate array

    It checks if the coordinates may be converted to 1D without loss of information.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        In case of a dataset, it must contain longitudes and latitudes.

    Return
    ------
    xarray.DataArray, xarray.Dataset
    """
    da = da.copy()
    new_coords = {}
    rename_args = {}
    for name, coord in da.coords.items():
        if coord.ndim != 2:
            continue
        if xcoords.is_lon(coord):
            odim = xcoords.get_ydim(coord, errors="ignore")
        elif xcoords.is_lat(coord):
            odim = xcoords.get_xdim(coord, errors="ignore")
        else:
            continue
        dims = [odim] if odim else coord.dims
        for odim in dims:
            if np.allclose(coord.min(odim), coord.max(odim), atol=tol):
                new_coords[name] = xr.DataArray(coord.isel(
                    {odim: 0}).values, dims=name, attrs=coord.attrs)
                new_coords[name].encoding.update(coord.encoding)
                dim = coord.dims[0] if coord.dims[1] == odim else coord.dims[1]
                rename_args[dim] = name
                break
    if new_coords:
        return da.reset_coords(list(new_coords), drop=True).rename(
            rename_args).assign_coords(new_coords)
    return da
