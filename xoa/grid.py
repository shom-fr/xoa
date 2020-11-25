# -*- coding: utf-8 -*-
"""
1d to nD grid utilities
"""
# Copyright or © or Copr. Shom, 2020
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

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
