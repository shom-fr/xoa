# -*- coding: utf-8 -*-
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

from . import exceptions
from . import misc
from . import meta
from . import coords as xcoords
from . import geo as xgeo
from .core import grid as cgrid

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
                kw = kwargs or {}
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


def _pad_(da, dim, pad_width, mode, **kwargs):
    pad_width = pad_width.get(dim, 0)
    if not pad_width:
        return da.copy()

    if mode != "linear_extrap":
        return da.pad({dim: pad_width}, mode=mode, **kwargs)

    to_concat = []
    if isinstance(pad_width, int):
        pad_width = (pad_width,)
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
        data_kwargs={"mode": mode, **kwargs},
        coord_kwargs={"mode": coord_mode},
        name_kwargs=name_kwargs,
        pad_width=pad_width,
    )


def _get_centers_(da, dim):
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
    return apply_along_dim(da, dim, _get_centers_)


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
    # Extrapolate
    dim = meta.get_meta_specs(da).parse_dims(dim, da)
    dims = (dim,) if isinstance(dim, str) else dim
    pad_width = dict((dim, 1) for dim in dims)
    da = pad(da, pad_width=pad_width, mode=mode, **kwargs)

    # Inner edges
    return get_centers(da, dim)


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

    # Extrapolate
    pad_width = {}
    for dim, shift_dir in shift_dirs.items():
        pad_width[dim] = (1, 0) if shift_directions[shift_dir] < 0 else (0, 1)
    da = pad(da, pad_width=pad_width, mode=mode, **kwargs)

    # Inner edges
    return get_centers(da, list(shift_dirs.keys()))


shift.__doc__ = shift.__doc__.format(**locals())


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
    #: Up (SSH)
    ssh = 1
    #: Bottom (bathy)
    bottom = -1
    #: Bottom (bathy)
    bathy = -1


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
        and the output coordinates are negative.
        When "down", first level is supposed to be the top
        and the output coordinates are positive.
        When "guess", the dz array must have an axis coordinate
        of the same name as the z dimension, and this coordinate must have
        a valid positive attribute.
    zdim: str
        Name of the vertical dimension.
        If not set, it is inferred with :func:`~xoa.coords.get_meta_dims`.
    ref: xarray.DataArray
        Reference array converting layer thicknesses to depth:

        - If **positive up**, it is expected to be the **SSH** (sea surface height)
          by default
        - If **positive down**, it is expected to be by default the depth of ground
          also known as **bathymetry**, which should be positive.

    ref_type: str, int
        Type of `ref`:
        {dz2depth_ref_types.rst_with_links}
    centered: bool
        Get depth at the middle of layers instead of at their edge

    Return
    ------
    xarray.DataArray
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
            raise exceptions.XoaGridError("Can't infer positive attribute from data array/dataset")

    # Integrate
    depth = dz.cumsum(dim=zdim)
    depth = pad(depth, {zdim: (1, 0)}, mode="constant", constant_values=0)
    ref_type = dz2depth_ref_types[ref_type].name
    meta_specs = meta.get_meta_specs(dz)
    if ref is None and ref_type == "infer":
        if meta_specs.data_vars.match(ref, "bathy"):
            ref_type = "bottom"
        elif meta_specs.data_vars.match(ref, "ssh"):
            ref_type = "top"
        else:
            ref_type = "top" if positive == "down" else "bottom"
    if positive == "up":
        if ref is None:
            ref = depth.isel({zdim: -1})
        elif ref is not None and ref_type == "top":
            ref = depth.isel({zdim: -1}) - ref
        depth[:] -= ref
    else:
        if ref is not None:
            if ref_type == "bottom":
                depth[:] -= depth.isel({zdim: -1})
            depth[:] += ref

    # Fix index
    if zdim in depth.indexes:
        dnz = depth[zdim].diff(zdim).pad({zdim: (0, 1)}, mode="edge")
        depth = xcoords.change_index(depth, zdim, depth[zdim] + 0.5 * dnz.data)

    # Centered
    if centered:
        depth = get_centers(depth, zdim)
        if zdim in depth.indexes:
            depth = depth.assign_coords({zdim: dz[zdim]})

    # Finalize
    depth.attrs["positive"] = positive
    depth = meta_specs.format_coord(
        depth, "z" if positive == "up" else "depth", rename=True, format_coords=False,
        rename_dims=False
    )

    return depth


dz2depth.__doc__ = dz2depth.__doc__.format(**locals())


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
    xoa.meta.get_meta_specs
    """
    ds = ds.copy()
    errors = misc.ERRORS[errors]

    # Find needed stuff
    meta_specs = meta.get_meta_specs(ds)
    dz = meta_specs.search(ds, "dz", errors=errors)
    if dz is None:
        return ds
    zdim = xcoords.get_meta_dims(dz, "z", errors=errors)
    if zdim is None:
        return ds
    positive = meta_specs["vertical"]["positive"]
    if positive is None:
        positive = xcoords.get_positive_attr(ds, zdim)
    if positive is None:
        msg = "Can't infer positive attribute from data dataset"
        if errors == "raise":
            raise exceptions.XoaGridError(msg)
        exceptions.xoa_warn(msg)
        return ds
    ssh = meta_specs.search(ds, "ssh", errors="ignore")
    bathy = meta_specs.search(ds, "bathy", errors="ignore")

    # Make choices
    if ssh is None and bathy is None:
        ref, ref_type = None, "infer"
    else:
        for ref, ref_type in [(bathy, "bathy"), (ssh, "ssh")][:: int(positive)]:
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
    )

    # Assign to dataset
    if depth.name in ds.dims:
        msg = "Can't assign the {} coordinate since a dimension has the same name".format(
            depth.name
        )
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


def _get_lonlat_yx_(obj):
    """Get 2D longitudes and latitudes with (y, x) dimensions"""
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
    lon, lat = _get_lonlat_yx_(obj)
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
    lon, lat = _get_lonlat_yx_(obj)
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
    lon, lat = _get_lonlat_yx_(obj)
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
