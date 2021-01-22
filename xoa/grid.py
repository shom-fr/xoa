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
