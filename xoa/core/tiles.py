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

from .. import exceptions
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


def get_lon_window_start(axes, default=-180.0, margin=1e-3):
    """Start of the 360 degrees window that holds longitude axes with the smallest extent

    The window starts after the largest gap that is left on the circle by the axes,
    a gap being larger than the step between contiguous axes.

    Parameters
    ----------
    axes: list(array_like)
        Increasing and continuous 1D longitudes, like those of tiles
    default: float
        Start used when the axes cover the whole circle
    margin: float
        Tolerance in degrees to be left before the first coordinate

    Return
    ------
    float
        Longitude of the start of the window

    See also
    --------
    wrap_lon
    """
    start = np.array([ax[0] for ax in axes]) % 360
    length = np.array([ax[-1] - ax[0] for ax in axes])
    order = np.argsort(start)
    start, length = start[order], length[order]
    end = np.maximum.accumulate(start + length)
    steps = [(ax[-1] - ax[0]) / (ax.size - 1) for ax in axes if len(ax) > 1]
    step = np.median(steps) if steps else 0.0
    # Gap before each axis, including the one that wraps around the first axis
    gaps = np.r_[start[0] + 360 - end[-1], start[1:] - end[:-1]] - step
    i = np.argmax(gaps)
    return default if gaps[i] <= 0.5 * step + margin else start[i] - margin


def wrap_lon(lon, start):
    """Shift longitudes by multiples of 360 degrees to be in ``[start, start + 360[``

    Parameters
    ----------
    lon: array_like
    start: float

    Return
    ------
    numpy.ndarray
    """
    lon = np.asarray(lon, "d")
    return lon - 360 * np.floor((lon - start) / 360)


def fit_regular_axis(axes, step=None, origin=None, size=None, name="axis"):
    """Get the regular axis that holds increasing 1D axes

    What is not given is inferred. The step is first estimated from the axes, then refined
    with a least squares fit on all their points, which avoids a drift over large grids
    with single precision coordinates, and rounded to 8 significant digits. The origin is the lowest coordinate and the size
    covers the last coordinate.

    Parameters
    ----------
    axes: list(array_like)
        Increasing 1D coordinates, like those of tiles
    step: None, float
    origin: None, float
        Coordinate of the index 0
    size: None, int
    name: str
        Name of the axis for error messages

    Return
    ------
    float
        Step
    float
        Origin
    int
        Size

    Raises
    ------
    xoa.exceptions.XoaError
        When the step cannot be inferred, or when an axis starts before the origin
    """
    axes = [np.asarray(ax, "d") for ax in axes]
    if step is None:
        steps = [(ax[-1] - ax[0]) / (ax.size - 1) for ax in axes if ax.size > 1]
        if not steps:
            raise exceptions.XoaError(f"Cannot infer the {name} step: provide it")
        step = np.median(steps)
        if origin is None:
            origin = min(ax[0] for ax in axes)
        pos = np.concatenate([np.round((ax - origin) / step) for ax in axes])
        # The fit is rounded to 8 digits to get round steps and coordinates
        step = float(f"{np.polyfit(pos, np.concatenate(axes), 1)[0]:.8g}")
    if origin is None:
        origin = min(ax[0] for ax in axes)
    ends = [get_grid_indices(np.array([ax[0], ax[-1]]), origin, step) for ax in axes]
    if any(ind[0] < 0 for ind, _ in ends):
        raise exceptions.XoaError(f"Some tiles start before the {name} origin {origin}")
    if size is None:
        size = int(max(ind[1] for ind, _ in ends)) + 1
    return step, origin, size


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
