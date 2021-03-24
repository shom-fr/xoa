# -*- coding: utf-8 -*-
"""
1d to nD grid utilities
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

from .__init__ import XoaError
from . import misc
from . import cf
from . import coords


def get_edges_1d(da, axis=-1, name_suffix='_edges'):
    """Get edges of coordinates from centers along a single axis

    Parameters
    ----------
    da: array
    axis: int, str
        Axis index to work on.
        May be a dimension name if ``da`` is a :class:`~xarray.DataArray`

    Example
    -------
    .. ipython:: python

        @suppress
        import numpy as np, xarray as xr
        @suppress
        from xoa.grid import get_edges_1d

        # 1D
        x = np.arange(3) * 2.
        print(get_edges_1d(x))

        # 2D
        xx = x[np.newaxis, :, np.newaxis]
        print(xx.shape)
        xxe = get_edges_1d(xx, axis=1)
        print(xxe.shape)
        print(xxe)

        # Xarrays
        x_ = xr.DataArray(x, dims='x', name='x')
        print(get_edges_1d(x_))
        xx_ = xr.DataArray(xx, dims=('y', 'x', 't'), name='lon')
        print(get_edges_1d(xx_, axis='x'))  # with dim name
        print(get_edges_1d(xx_, axis=1))

    """
    # Init
    if isinstance(axis, str):
        if not isinstance(da, xr.DataArray):
            raise XoaError('da must be a DataArray is axis is a str')
        axis = da.dims.index(axis)
    ss = misc.get_axis_slices(da, axis)
    shape = list(da.shape)
    shape[axis] += 1
    edges = np.empty(shape, dtype=da.dtype)
    data = da.data if isinstance(da, xr.DataArray) else da

    # Compute
    edges[ss["first"]] = (data[ss["first"]] -
                          (data[ss["firstp1"]] - data[ss["first"]]) * .5)
    edges[ss["mid"]] = 0.5 * (data[ss["firsts"]] + data[ss["lasts"]])
    edges[ss["last"]] = (data[ss["last"]] -
                         (data[ss["lastm1"]] - data[ss["last"]]) * .5)

    # Finalize
    if isinstance(da, xr.DataArray):
        dims = list(da.dims)
        dims[axis] += name_suffix
        name = (da.name+name_suffix) if da.name else da.name
        edges = xr.DataArray(edges, dims=dims, name=name)
    return edges


def get_edges_2d(da, name_suffix='_edges'):
    """Get edges of a 2D coordinates array

    Parameters
    ----------
    da: array(ny, nx)
    """

    # Init
    if da.ndim != 2:
        raise ValueError(f'Input must be a 2d array, but got {da.ndim}d.')
    ny, nx = da.shape
    edges = np.empty((ny + 1, nx + 1), da.dtype)
    data = da.data if isinstance(da, xr.DataArray) else da

    # Inner
    edges[1:-1, 1:-1] = 0.25 * (
        data[1:, 1:] + data[:-1, 1:] + data[1:, :-1] + data[:-1, :-1]

    )

    # Lower and upper
    edges[0] += get_edges_1d(1.5 * data[0] - 0.5 * data[1])
    edges[-1] += get_edges_1d(1.5 * data[-1] - 0.5 * data[-2])

    # Left and right
    edges[:, 0] += get_edges_1d(1.5 * data[:, 0] - 0.5 * data[:, 1])
    edges[:, -1] += get_edges_1d(1.5 * data[:, -1] - 0.5 * data[:, -2])

    # Corners
    edges[[0, 0, -1, -1], [0, -1, -1, 0]] *= 0.5

    # Finalize
    dims = list(da.dims)
    dims[0] += name_suffix
    dims[1] += name_suffix
    name = (da.name+name_suffix) if da.name else da.name
    if isinstance(da, xr.DataArray):
        edges = xr.DataArray(edges, dims=dims, name=name)
    return edges


class positive_attr(misc.IntEnumChoices):
    """Allowed value for the positive attribute"""
    #: Guessed from the axis coordinate
    guess = 0
    #: Coordinates are increasing up
    up = 1
    #: Coordinates are increasing down
    down = -1


def dz2depth(dz, positive="guessed", zdim=None, base=None, cfname="depth"):
    """Integrate layer thicknesses to compute depths

    The output depths are the depths at the bottom of the layers and the top
    is at a depth of zero. Thus, the output array has the same dimensions
    as the input array of layer thinknesses.

    Parameters
    ----------
    dz: xarray.DataArray
        Layer thinknesses
    positive: str, int, None
        Direction over wich coordinates are increasing:
        {positive_attr.rst_with_links}
        When "up", the first level is supposed to be the bottom
        and the output coordinates are negative.
        When "down", first level is supposed to be the top
        and the output coordinates are positive.
        When "guess", the dz array must have an axis coordinate
        of the same name as the z dimension, and this coordinate must have
        a valid positive attribute.
    zdim: str
        Name of the vertical dimension.
        If note set, it is infered with :func:`~xoa.coords.get_dims`.
    base: xarray.DataArray
        Base array from which to integrate:

        - If **positive up", it is expected to be the **SSH** (sea surface heigth)
        - If **positive down", it is expected to be the depth of ground,
          also known as **bathymetry**, which should be positive.

    cfname: str
        CF name used to format the output depth variable.

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
        zdim = coords.get_dims(dz, "z", errors="raise")[0]

    # Positive attribute
    positive = positive_attr[positive].name
    if positive == "guess":
        if zdim not in dz.coords and "positive" not in dz.coords[zdim].attrs:
            raise XoaError("Can't guess positive attribute from data array")
        positive = positive_attr[dz.coords["zdim"].attrs["positive"]].name

    # Positive down as cumsum
    depth = dz.cumsum(dim=zdim)

    # Positive up: integrate from the ground
    if positive == "up":

        depth = depth.roll({zdim: 1}, roll_coords=False)
        if base is None:
            depth[{zdim: 0}] *= -1
        else:
            depth[{zdim: 0}] = -base  # bath is bottom depth
        depth[{zdim: slice(1, None)}] += depth.isel({zdim: 0}).values

    elif base is not None:  # base is ssh
        depth[:] -= base

    # Finalize
    depth.attrs["positive"] = positive
    return cf.get_cf_specs().format_coord(depth, cfname)
