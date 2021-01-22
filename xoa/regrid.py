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
