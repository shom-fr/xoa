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


from .__init__ import XoaError
from . import misc
from . import coords
from . import grid
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
    # cellerr = -2


class extrap_modes(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported extrapolation modes"""
    #: No extrapolation (default)
    no = 0
    none = 0
    #: Above (after)
    above = 1
    #: Below (before)
    below = -1
    #: Both below and above
    both = 2
    all = 2
    yes = 2


def regrid1d(da, coord, method=None, dim=None, coord_in_name=None,
             conserv=False, extrap=0, bias=0., tension=0.):
    """Regrid along a single dimension

    The input and output coordinates may vary along other dimensions,
    which useful for instance for vertical interpolation in coastal
    ocean models.

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
        Else, provide a two-element tuple: ``(dim_coord_in, dim_coord_out)``.
    coord_in_name: str, None
        Name of the input coordinate array, which must be known of ``da``
    conserv: bool
        Use conservative regridding when using ``cellave`` method.
    extrap: str, int
        Extrapolation mode as one of the following:
        {extrap_modes.rst_with_links}

    Returns
    -------
    xarray.DataArray
        Regridded array with ``coord`` as new coordinate array.
    """

    # Array manager
    dfl = coords.DimFlusher1D(da, coord, dim=dim,
                              coord_in_name=coord_in_name)

    # Method
    method = regrid1d_methods[method]

    # Fortran function name and arguments
    func_name = str(method) + "1d"
    yi = dfl.coord_in_data
    yo = dfl.coord_out_data
    func_kwargs = {"vari": dfl.da_in_data}
    # if not (dfl.coord_in_data.shape[0] == dfl.coord_out_data.shape[0] == 1):  # 1d
        # if method == regrid1d_methods.cellerr:
        #     raise XoaRegridError("cellerr regrid method is works only "
        #                          "with 1D input and output cordinates")
    if int(method) < 0:
        func_kwargs.update(yib=grid.get_edges_1d(yi, axis=-1),
                           yob=grid.get_edges_1d(yo, axis=-1))
    else:
        func_kwargs.update(yi=yi, yo=yo)
    func = getattr(interp, func_name)
    if method == "hermit":
        func_kwargs.update(bias=bias, tension=tension)

    # Regrid
    varo = func(**func_kwargs)

    # Extrap
    extrap = extrap_modes[extrap]
    if extrap != extrap_modes.no:
        varo = interp.extrap1d(varo, extrap)

    # Reform back
    return dfl.get_back(varo)


regrid1d.__doc__ = regrid1d.__doc__.format(**locals())


def grid2loc(da, loc, compat="warn"):
    """Interpolate a gridded data array to random locations
    
    ``da`` and ``loc`` must comply with CF conventions.

    Parameters
    ----------
    da: xarray.DataArray
        A data array with at least an horizontal grid.
    loc: xarray.Dataset, xarray.DataArray
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
    """

    # Get coordinates
    # - horizontal
    # if set(glon.dims).isdisjoint(glat.dims):
    #     gdims = glat.dims + glon.dims
    # else:
    #     gdims = glon.dims
    order = "yx"
    lons = coords.get_lon(loc)
    lats = coords.get_lat(loc)
    # - vertical
    deps = coords.get_vertical(loc, errors="ignore")
    if deps is not None:
        gdep = coords.get_vertical(da, errors=compat)
        if gdep is not None:
            order = "z" + order
    # - temporal
    times = coords.get_time(loc, errors="ignore")
    if times is not None:
        gtime = coords.get_time(da, errors=compat)
        if gtime is not None:
            order = "t" + order

    # Transpose following the tzyx order
    da_tmp = coords.reorder(da, order)
    # TODO: reshape with singleton insertions, also for cordinates
    
    # To numpy with singletons
    # - data
    da_num = da.values
    for axis_type, axis in (("z", -3), ("t", -4)):
        if axis_type not in order:
            da_num = np.expand_dims(axis_type, axis)
    # - xy
    glon_num = coords.get_lon(da_tmp).values
    glat_num = coords.get_lat(da_tmp).values
    # - z
    if "z" in order:
        gdep_order = coords.get_order(da_tmp[gdep.name])
        gdep_num = da_tmp[gdep.name].values
        for axis_type, axis in (("x", -1), ("y", -2), ("t", -4)):
            if axis_type not in gdep_order:
                da_num = np.expand_dims(axis_type, axis)
    else:
        gdep_num = np.zeros((1, 1, 1, 1))
    if gdep_num.ndim == 4:
        gdep_num = gdep_num.reshape((1,)+gdep_num.shape))
    # - t
    if "t" in order:
        # numeric times
        gtime_num = (
            (gtime.values - np.datetime64("1950-01-01")) /
            np.timedelta64(1, "D"))
        times_num = (
            (times.values - np.datetime64("1950-01-01")) /
            np.timedelta64(1, "D"))
    else:
        gtime_num = np.zeros(1)
    
