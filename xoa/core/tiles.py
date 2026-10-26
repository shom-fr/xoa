"""
Low level utilities to merge tiles of a regular grid

The numerical inputs and outputs of all these routines are of scalar
or numpy.ndarray type.
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

import numba
import numpy as np

from .num import NOT_CI

#: Overlap modes of :func:`paste`
OVERLAPS = {"last": 0, "first": 1, "mean": 2, "raise": 3}


@numba.njit(cache=NOT_CI)
def get_grid_indices(coord, origin, step):
    """Indices of 1D coordinates on a regular axis

    Parameters
    ----------
    coord: array_like(n)
    origin: float
        Coordinate of the index 0
    step: float

    Return
    ------
    array_like(n)
        Nearest integer indices
    float
        Largest distance to the axis, in fraction of `step`
    """
    n = coord.shape[0]
    indices = np.empty(n, np.int64)
    err = 0.0
    for i in range(n):
        pos = (coord[i] - origin) / step
        indices[i] = np.round(pos)
        err = max(err, abs(pos - indices[i]))
    return indices, err


@numba.njit(parallel=True, cache=NOT_CI)
def paste(out, tile, iy0, ix0, mode, count):
    """Paste a tile into the output array, ignoring its nans

    Parameters
    ----------
    out: array_like(nouter, ny, nx)
        Output array, filled with nans where nothing is yet pasted
    tile: array_like(nouter, nty, ntx)
    iy0: int
        Index of the first row of the tile in `out`
    ix0: int
        Index of the first column of the tile in `out`
    mode: int
        How to deal with the cells that are already filled, from :data:`OVERLAPS`:
        0 to overwrite, 1 to keep, 2 to sum them and 3 to count the differences
    count: array_like(nouter, ny, nx)
        Number of pasted values, only used with the mode 2

    Return
    ------
    int
        Number of cells that differ from the existing ones, only with the mode 3
    """
    nouter, nty, ntx = tile.shape
    nconflicts = 0
    for io in numba.prange(nouter):
        for iy in range(nty):
            for ix in range(ntx):
                val = tile[io, iy, ix]
                if np.isnan(val):
                    continue
                cur = out[io, iy0 + iy, ix0 + ix]
                if np.isnan(cur):
                    out[io, iy0 + iy, ix0 + ix] = val
                    if mode == 2:
                        count[io, iy0 + iy, ix0 + ix] = 1
                elif mode == 0:
                    out[io, iy0 + iy, ix0 + ix] = val
                elif mode == 2:
                    out[io, iy0 + iy, ix0 + ix] = cur + val
                    count[io, iy0 + iy, ix0 + ix] += 1
                elif mode == 3 and cur != val:
                    nconflicts += 1
    return nconflicts


def finalize_mean(out, count):
    """Convert the sums of :func:`paste` with the mode 2 to means, in place

    Parameters
    ----------
    out: array_like
    count: array_like
        Same shape as `out`
    """
    np.divide(out, count, out=out, where=count > 1)
