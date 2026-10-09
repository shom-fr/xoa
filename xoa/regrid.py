#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regridding utilities.

Provides :func:`regrid1d` for 1D regridding along a single dimension
(e.g. vertical interpolation) and :func:`extrap1d` for 1D extrapolation.
The core computation is performed by numba-accelerated routines
from :mod:`xoa.core.regrid`.
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
from . import grid as xgrid
from . import weights
from .core import num
from .core import regrid
from .core.num import CsrMatrix
from .core.regrid import XYRegridder as CoreXYRegridder

# Backward compat
from .interp import grid2loc, isoslice  # noqa

# %% 1D


class regrid1d_methods(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :func:`regrid1d` methods"""

    #: Linear interpolation (default)
    linear = 1
    #: Linear interpolation (default)
    interp = 1  # compat
    #: Nearest interpolation
    nearest = 0
    #: Cubic interpolation
    cubic = 2
    #: Hermitian interpolation
    hermit = 3
    #: Hermitian interpolation
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

    Output array is reshaped back accordingly.
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


def _get_dims_(da, coord, dim):
    """Get the input and output dimensions of :func:`regrid1d`"""
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
    return dim_in, dim_out


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
    Since it uses :func:`xarray.apply_ufunc`, it supports dask arrays.
    The core computation is performed by the numba-accelerated routines
    of :mod:`xoa.core.regrid`.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        Array or dataset to regrid. In a dataset, only the variables that have the
        input dimension are regridded, and others are left unchanged.
    coord: xarray.DataArray
        Output coordinate.
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
        It is inferred from the input data array and dimension name
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
        Max size for a gap to be interpolated when ``drop_na`` is True.
        Size is not checked when ``maxgap`` is zero.
    dask: str
        See :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.DataArray, xarray.Dataset
        Regridded array or dataset with ``coord`` as new coordinate array.
        Name and attributes are preserved from ``da``.

    Example
    -------
    Linear interpolation from 4 depth levels to 7::

        zi = xr.DataArray(np.arange(4.), dims="z")
        vi = xr.DataArray(np.arange(4.), dims="z", coords=dict(z=zi))
        zo = xr.DataArray(np.linspace(0, 3, 7), dims="z")
        vo = regrid1d(vi, zo, method="linear")

    See Also
    --------
    extrap1d
    xoa.core.regrid.nearest1d
    xoa.core.regrid.linear1d
    xoa.core.regrid.cubic1d
    xoa.core.regrid.hermit1d
    xoa.core.regrid.cellave1d
    xoa.core.regrid.extrap1d
    xarray.apply_ufunc
    """
    # Ensure nanosecond precision for datetime coordinates
    da = xcoords.ensure_ns_datetime(da)
    coord = xcoords.ensure_ns_datetime(coord)

    # Get the working dimensions
    dim_in, dim_out = _get_dims_(da, coord, dim)
    cfspecs_in = meta.get_meta_specs(da)

    # Dataset: apply to the variables that have the input dimension
    if isinstance(da, xr.Dataset):
        kwargs = dict(
            method=method,
            dim=(dim_in, dim_out),
            coord_in_name=coord_in_name,
            edges=edges,
            conserv=conserv,
            extrap=extrap,
            bias=bias,
            tension=tension,
            drop_na=drop_na,
            maxgap=maxgap,
            dask=dask,
        )
        out = xr.Dataset(
            {
                name: regrid1d(var, coord, **kwargs) if dim_in in var.dims else var
                for name, var in da.data_vars.items()
            },
            attrs=da.attrs,
        )
        coords = {
            name: c
            for name, c in da.coords.items()
            if dim_in not in c.dims and name not in out.coords
        }
        return out.assign_coords(coords)

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
            coord = coord.drop_vars(cname)

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
            coord_out = coord_out.drop_vars(cname)
    da_out = da_out.assign_coords({coord_out_name: coord_out})
    da_out.name = da.name
    da_out.attrs = da.attrs

    return da_out


regrid1d.__doc__ = regrid1d.__doc__.format(**locals())


def extrap1d(da, dim, mode, dask='parallelized'):
    """Extrapolate along a single dimension

    Fills NaN values at the edges of ``dim`` by nearest-neighbor
    extrapolation. Name, attributes and coordinates are preserved.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        Array or dataset to extrapolate. In a dataset, only the variables that have
        ``dim`` are extrapolated.
    dim: str
        Dimension along which to extrapolate.
    mode: str, int
        Extrapolation mode as one of the following:
        {extrap_modes.rst_with_links}
    dask: str
        See :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.DataArray, xarray.Dataset
        Extrapolated array or dataset.

    Example
    -------
    Fill NaN values at both ends along the ``"y"`` dimension::

        vo = extrap1d(vi, "y", mode="both")

    See also
    --------
    regrid1d
    xoa.core.regrid.extrap1d
    xarray.apply_ufunc
    """
    if isinstance(da, xr.Dataset):
        return da.map(
            lambda var: extrap1d(var, dim, mode, dask) if dim in var.dims else var,
            keep_attrs=True,
        )
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


# %% Horizontal


class xy_regrid_methods(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :class:`Regridder` methods"""

    #: Bilinear interpolation (default)
    bilinear = 1
    #: Bilinear interpolation (default)
    linear = 1
    #: Bicubic interpolation
    bicubic = 2
    #: Bicubic interpolation
    cubic = 2
    #: Conservative regridding
    conservative = 3


#: Cache of the core regridders, which hold the weights
_WEIGHTS_CACHE = misc.SmallCache(maxsize=8)


def clear_weights_cache():
    """Forget the weights that are shared by the :class:`Regridder` of the same grids"""
    _WEIGHTS_CACHE.clear()


class Regridder:
    """Horizontal and temporal regridder

    Parameters
    ----------
    ds_src_grid, ds_dst_grid: xarray.Dataset, xarray.DataArray
        Source and destination grids with 1D or 2D longitude and latitude
        coordinates
    method: str, int
        Regridding method among {xy_regrid_methods.rst_with_links}
    weights_file: str, None
        Path to a netcdf file that may hold the weights of many grids, each one in a group
        that is named after the method and a fingerprint of the grids.
        The weights are loaded from it if the group exists, and saved to it otherwise,
        after the first regridding. See :mod:`xoa.weights`.
    src_mask, dst_mask: str, xarray.DataArray, array_like, None
        Valid points of the source and destination grids
    bias, tension: float
        Kochanek-Bartels parameters for the bicubic method

    Attributes
    ----------
    src_fingerprint, dst_fingerprint: str
        Fingerprints of the source and destination grids, as given by
        :func:`xoa.grid.get_fingerprint`
    fingerprint: str
        Fingerprint of the couple of grids
    weights_group: str
        Name of the group of the weights in a weights file. It is made of the method
        and the fingerprint, and :func:`xoa.weights.find_groups` finds it from the
        fingerprint of a grid.

    Notes
    -----
    The weights are computed once, when needed, and shared by all the regridders that
    have the same source and destination grids, the same method and the same
    parameters, as long as they are among the most recently used ones.
    :func:`clear_weights_cache` frees them.

    See also
    --------
    xoa.core.regrid.XYRegridder
    xoa.interp.Interpolator
    """

    def __init__(
        self,
        ds_src_grid,
        ds_dst_grid,
        method,
        weights_file=None,
        src_mask=None,
        dst_mask=None,
        bias=0.0,
        tension=0.0,
    ):
        self.ds_src_grid = ds_src_grid.copy(deep=False)
        self.ds_dst_grid = ds_dst_grid.copy(deep=False)
        src_grid = xgrid.ds2grid_dict(ds_src_grid, src_mask)
        dst_grid = xgrid.ds2grid_dict(ds_dst_grid, dst_mask)

        try:
            method = str(xy_regrid_methods[method])
        except (KeyError, ValueError):
            raise ValueError(f"Invalid method {method!r}. Choose among: {xy_regrid_methods.rst}")
        # The core regridder holds the weights, which are shared by the regridders of the same grids
        self.src_fingerprint = xgrid.get_fingerprint(src_grid)
        self.dst_fingerprint = xgrid.get_fingerprint(dst_grid)
        self.fingerprint = misc.combine_fingerprints(self.src_fingerprint, self.dst_fingerprint)
        self.weights_group = weights.get_group_name("regrid", method, self.fingerprint)
        key = (
            method,
            float(bias),
            float(tension),
            src_grid["dims"],
            dst_grid["dims"],
            self.fingerprint,
        )
        self.core_regridder = _WEIGHTS_CACHE.get_or_create(
            key, lambda: CoreXYRegridder(src_grid, dst_grid, method, bias=bias, tension=tension)
        )
        self.src_grid = self.core_regridder.src_grid
        self.dst_grid = self.core_regridder.dst_grid

        self.weights_file = weights_file
        if (
            weights_file
            and not self.core_regridder.has_weights
            and (
                weights.has_group(weights_file, self.weights_group)
                or weights.is_legacy(weights_file)
            )
        ):
            self.load_weights(weights_file)

    def compute_weights(self, skipna=False):
        """Compute the weights"""
        self.core_regridder.compute_weights(skipna)

    def save_weights(self, weights_file):
        """Save the weights to a group of a netcdf file

        The group is named after the method and a fingerprint of the grids, and it is added
        to the file, which may hold the weights of other grids.
        Nothing is written if the group already exists.
        Bilinear and bicubic methods store fractional indices, and the
        conservative one stores the arrays of a sparse matrix.
        """
        cr = self.core_regridder
        if not cr.has_weights:
            raise ValueError("No weights computed yet")
        attrs = {
            "kind": "regrid",
            "method": cr.method,
            "fingerprint": self.fingerprint,
            "src_fingerprint": self.src_fingerprint,
            "dst_fingerprint": self.dst_fingerprint,
            "n_dst": cr.dst_grid["lat"].size,
            "n_src": cr.src_grid["lat"].size,
        }
        if cr.method in ("bilinear", "bicubic"):
            variables = {
                "j_base": ("n_dst", cr._j_base),
                "i_base": ("n_dst", cr._i_base),
                "frac_a": ("n_dst", cr._frac_a),
                "frac_b": ("n_dst", cr._frac_b),
                "valid_dst_mask": ("n_dst", cr._valid_dst_mask),
            }
        else:
            variables = {
                "indptr": ("n_dst_p1", cr._nb_indptr),
                "indices": ("nnz", cr._nb_indices),
                "wdata": ("nnz", cr._nb_wdata),
                "valid_dst_mask": ("n_dst", cr._valid_dst_mask),
            }
        weights.save_group(weights_file, self.weights_group, variables, attrs)

    def load_weights(self, weights_file):
        """Load the weights of these grids from a netcdf file

        The group that matches the fingerprint of the grids and the method is searched.
        A file in the legacy format, that holds the weights of a single grid without
        fingerprint, is still read, but only the sizes of the grids can be checked.

        Raises
        ------
        ValueError
            When the file has no weights for these grids and this method
        """
        cr = self.core_regridder
        n_dst = self.dst_grid["lon"].size
        n_src = self.src_grid["lon"].size
        ds = weights.select_group(
            weights_file,
            self.weights_group,
            self.fingerprint,
            cr.method,
            n_dst,
            n_src,
            "these grids",
        )
        cr._valid_dst_mask = ds["valid_dst_mask"].values
        if cr.method in ("bilinear", "bicubic"):
            cr._j_base = ds["j_base"].values.astype(np.int64, copy=False)
            cr._i_base = ds["i_base"].values.astype(np.int64, copy=False)
            cr._frac_a = ds["frac_a"].values.astype(np.float64, copy=False)
            cr._frac_b = ds["frac_b"].values.astype(np.float64, copy=False)
        else:
            cr._nb_indptr = ds["indptr"].values.astype(np.int64, copy=False)
            cr._nb_indices = ds["indices"].values.astype(np.int64, copy=False)
            cr._nb_wdata = ds["wdata"].values.astype(np.float64, copy=False)
            cr.weights = CsrMatrix(cr._nb_indptr, cr._nb_indices, cr._nb_wdata, (n_dst, n_src))

    def regrid(self, src_ds, dst_time=None, skipna=False, na_thres=1.0):
        """Regrid a dataset or data array horizontally, and optionally in time

        Parameters
        ----------
        src_ds: xarray.Dataset, xarray.DataArray
            Source data
        dst_time: array_like, xarray.DataArray, None
            Target times. When not provided, the time of the destination grid
            is used if any. No temporal interpolation is performed if there is
            no target time or no time in the source.
        skipna: bool
            Skip NaN values
        na_thres: float
            Threshold for NaN handling

        Return
        ------
        xarray.Dataset, xarray.DataArray
            Regridded data. Dimensions other than the horizontal ones are preserved.
        """
        # Horizontal
        ds_xy = xr.apply_ufunc(
            lambda arr, **kw: self.core_regridder.regrid(np.asarray(arr), **kw),
            src_ds,
            input_core_dims=[self.src_grid["dims"]],
            output_core_dims=[self.dst_grid["dims"]],
            exclude_dims=set(self.src_grid["dims"]),
            kwargs={"skipna": skipna, "na_thres": na_thres},
            vectorize=False,
            dask="allowed",
            keep_attrs=False,
            on_missing_core_dim="copy",
            output_dtypes=[np.float64],
        )
        for getter in xcoords.get_lat, xcoords.get_lon:
            coord = getter(self.ds_dst_grid)
            ds_xy.coords[coord.name] = coord
        if isinstance(src_ds, xr.Dataset):
            for name in ds_xy.data_vars:
                if set(self.dst_grid["dims"]).intersection(ds_xy[name].dims):
                    ds_xy[name].attrs.update(src_ds[name].attrs)
        elif set(self.dst_grid["dims"]).intersection(ds_xy.dims):
            ds_xy.attrs.update(src_ds.attrs)
        # Restore attributes of surviving coordinates dropped by keep_attrs=False
        for coord_name in ds_xy.coords:
            if coord_name in src_ds.coords and not ds_xy.coords[coord_name].attrs:
                ds_xy.coords[coord_name].attrs.update(src_ds.coords[coord_name].attrs)

        if self.weights_file and not weights.has_group(self.weights_file, self.weights_group):
            self.save_weights(self.weights_file)

        # Time
        src_time = xcoords.get_time(src_ds, errors="ignore")
        if src_time is None or src_time.name not in ds_xy.dims:
            return ds_xy
        if dst_time is None:
            dst_time = xcoords.get_time(self.ds_dst_grid, errors="ignore")
        if dst_time is None:
            return ds_xy
        if not isinstance(dst_time, xr.DataArray):
            dst_time = xr.DataArray(dst_time, name=src_time.name, dims=src_time.name)
        return ds_xy.interp({src_time.name: dst_time})


Regridder.__doc__ = Regridder.__doc__.format(**locals())


def regridxy(
    da,
    dst=None,
    method="bilinear",
    dst_time=None,
    skipna=False,
    na_thres=1.0,
    regridder=None,
    **kwargs,
):
    """Regrid horizontally, and optionally in time, to a destination grid

    This is the functional counterpart of :func:`regrid1d` for the horizontal
    dimensions. It is a shortcut to :class:`Regridder`.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        Source data with longitude and latitude coordinates
    dst: xarray.DataArray, xarray.Dataset, None
        Destination grid with longitude and latitude coordinates.
        It is not needed when a ``regridder`` is given.
    method: str, int
        Regridding method, see :class:`xy_regrid_methods`
    dst_time: array_like, xarray.DataArray, None
        Target times
    skipna: bool
        Skip NaN values
    na_thres: float
        Threshold for NaN handling
    regridder: Regridder, None
        Existing regridder to use, to avoid initializing it again in a loop.
        Its source grid must be the one of ``da``. ``dst``, ``method`` and ``kwargs``
        are then ignored, but ``src_mask`` must be provided again if it was used to create it.
    kwargs:
        Extra parameters are passed to :class:`Regridder`, like ``weights_file``,
        ``src_mask``, ``dst_mask``, ``bias`` or ``tension``.
        When ``weights_file`` is provided, the weights are loaded from it if they
        are in, and saved to it otherwise.

    Return
    ------
    xarray.DataArray, xarray.Dataset

    Notes
    -----
    The weights are shared by the calls that use the same grids, method and parameters,
    but the grids must be fingerprinted at each call. Create a :class:`Regridder` once
    and pass it as ``regridder`` when regridding many variables or time steps.

    See also
    --------
    Regridder
    xoa.interp.interpxy
    """
    if regridder is None:
        if dst is None:
            raise ValueError("A destination grid or a regridder is needed")
        regridder = Regridder(da, dst, method, **kwargs)
    else:
        src_grid = xgrid.ds2grid_dict(da, kwargs.get("src_mask"))
        fingerprint = xgrid.get_fingerprint(src_grid)
        if fingerprint != regridder.src_fingerprint:
            raise ValueError(
                "The source grid of the data does not match the one of the regridder: "
                f"{fingerprint} instead of {regridder.src_fingerprint}."
            )
    return regridder.regrid(da, dst_time=dst_time, skipna=skipna, na_thres=na_thres)
