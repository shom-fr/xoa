"""
Low level regridding routines accelerated with numba

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
import os

import numba
import numpy as np

from ..exceptions import xoa_warn
from .conserv import compute_conservative_weights
from .geo import normalize_longitudes
from .grid import centers2edges, check_grid_type, edges2bounds, unwrap_grid_longitudes
from .interp import XYInterpolator, get_source_matrix
from .num import (
    EMPTY_OFFSETS,
    CsrMatrix,
    as_float_array,
    broadcast_along_axis,
    coo_to_csr,
    csr_spmm_skipna_unified,
    get_iminmax,
    view_along_axis,
)

NOT_CI = os.environ.get("CI", "false") == "false"


# %% 1D regridding
#
# The kernels work along an axis that is kept in place: the arrays are viewed as
# (nouter, n, ninner) with broadcast_along_axis, and the columns are processed in
# parallel. Each method has a function that works on a single column.


#: Integer codes of the extrapolation modes in the kernels
_EXTRAP_CODES_ = {"no": 0, "bottom": -1, "top": 1, "both": 2}


@numba.njit(nogil=True, cache=NOT_CI)
def _iminmax_valid_(y, v):
    """First and last indices where both y and v are valid"""
    imin = -1
    imax = -1
    n = len(y)
    for i in range(n):
        if imin < 0 and not np.isnan(y[i] * v[i]):
            imin = i
        if imax < 0 and not np.isnan(y[n - 1 - i] * v[n - 1 - i]):
            imax = n - 1 - i
        if imax > 0 and imin > 0:
            break
    return imin, imax


@numba.njit(nogil=True, cache=NOT_CI)
def _nearest1d_col_(vi, yi, yo, vo, extrap, drop_na, maxgap):
    iyimin, iyimax = _iminmax_valid_(yi, vi)
    iyomin, iyomax = get_iminmax(yo)
    iyominv = iyomin
    gap = 0
    for iyi in range(iyimin, iyimax):
        # Out of bounds
        if yi[iyi + 1] < yo[iyomin]:
            continue
        if yi[iyi] > yo[iyomax]:
            break

        # Gap check
        if (
            drop_na
            and (np.isnan(yi[iyi + 1]) or np.isnan(vi[iyi + 1]))
            and (maxgap == 0 or gap < maxgap)
        ):
            gap += 1
            continue

        iyi0 = iyi - gap
        iyi1 = iyi + 1

        # Loop on output grid
        for iyo in range(iyominv, iyomax + 1):
            dy0 = yo[iyo] - yi[iyi0]
            dy1 = yi[iyi1] - yo[iyo]

            # Above
            if dy1 < 0.0:  # above
                break

            # Below
            if dy0 < 0.0:
                iyominv = iyo + 1

            # Interpolations
            elif dy0 <= dy1:
                vo[iyo] = vi[iyi0]
            else:
                vo[iyo] = vi[iyi1]

        gap = 0

    # Extrapolation with nearest
    if (extrap == -1 or extrap == 2) and yo[iyomin] < yi[iyimin]:
        for iyo in range(iyomin, iyomax + 1):
            if yo[iyo] >= yi[iyimin]:
                vo[:iyo] = vi[iyimin]
                break
    if (extrap == 1 or extrap == 2) and yo[iyomax] > yi[iyimax]:
        for iyo in range(iyomin, iyomax + 1):
            if yo[iyo] > yi[iyimax]:
                vo[iyo:] = vi[iyimax]
                break


@numba.njit(nogil=True, cache=NOT_CI)
def _linear1d_col_(vi, yi, yo, vo, extrap, drop_na, maxgap):
    iyimin, iyimax = _iminmax_valid_(yi, vi)
    iyomin, iyomax = get_iminmax(yo)
    iyominv = iyomin
    gap = 0
    for iyi in range(iyimin, iyimax):
        # Out of bounds
        if yi[iyi + 1] < yo[iyomin]:
            continue
        if yi[iyi] > yo[iyomax]:
            break

        # Gap check
        if (
            drop_na
            and (np.isnan(yi[iyi + 1]) or np.isnan(vi[iyi + 1]))
            and (maxgap == 0 or gap < maxgap)
        ):
            gap += 1
            continue

        iyi0 = iyi - gap
        iyi1 = iyi + 1

        # Loop on output grid
        for iyo in range(iyominv, iyomax + 1):
            dy0 = yo[iyo] - yi[iyi0]
            dy1 = yi[iyi1] - yo[iyo]

            # Above
            if dy1 < 0.0:  # above
                break

            # Below
            if dy0 < 0.0:
                iyominv = iyo + 1

            # Interpolation
            elif dy0 > 0.0 or dy1 > 0.0:
                vo[iyo] = (vi[iyi0] * dy1 + vi[iyi1] * dy0) / (dy0 + dy1)

        gap = 0

    # Extrapolation with nearest
    if (extrap == -1 or extrap == 2) and yo[iyomin] < yi[iyimin]:
        for iyo in range(iyomin, iyomax + 1):
            if yo[iyo] < yi[iyimin]:
                vo[iyo] = vi[iyimin]
            else:
                break

    if (extrap == 1 or extrap == 2) and yo[iyomax] > yi[iyimax]:
        for iyo in range(iyomax, iyomin - 1, -1):  # Loop backwards
            if yo[iyo] > yi[iyimax]:
                vo[iyo] = vi[iyimax]
            else:
                break


@numba.njit(nogil=True, cache=NOT_CI)
def _cubic1d_col_(vi, yi, yo, vo, extrap, drop_na, maxgap):
    iyimin, iyimax = _iminmax_valid_(yi, vi)
    iyomin, iyomax = get_iminmax(yo)
    iyominv = iyomin
    gap = 0
    for iyi in range(iyimin, iyimax):
        # Out of bounds
        if yi[iyi + 1] < yo[iyomin]:
            continue
        if yi[iyi] > yo[iyomax]:
            break

        # Gap check
        if (
            drop_na
            and (np.isnan(yi[iyi + 1]) or np.isnan(vi[iyi + 1]))
            and (maxgap == 0 or gap < maxgap)
        ):
            gap += 1
            continue

        iyi0 = iyi - gap
        iyi1 = iyi + 1
        if iyi0 == iyimin or np.isnan(vi[iyi0 - 1]):
            iyim1 = iyi0
        else:
            iyim1 = iyi0 - 1
        if iyi1 == iyimax or np.isnan(vi[iyi1 + 1]):
            iyi2 = iyi1
        else:
            iyi2 = iyi1 + 1

        # Loop on output grid
        for iyo in range(iyominv, iyomax + 1):
            dy0 = yo[iyo] - yi[iyi0]
            dy1 = yi[iyi1] - yo[iyo]

            # Above
            if dy1 < 0.0:  # above
                break

            # Below
            if dy0 < 0.0:
                iyominv = iyo + 1

            # Inside
            if dy0 >= 0.0 and dy1 >= 0.0:
                iyominv = iyo
                mu = dy0 / (dy0 + dy1)

                # Extrapolations
                if iyi0 == iyimin:  # y0
                    vc0 = 2 * vi[iyi0] - vi[iyi1]
                else:
                    vc0 = vi[iyim1]
                if iyi1 == iyimax:  # y3
                    vc1 = 2 * vi[iyi1] - vi[iyi0]
                else:
                    vc1 = vi[iyi2]

                # Interpolation
                vo[iyo] = vc1 - vi[iyi1] - vc0 + vi[iyi0]
                vo[iyo] = mu**3 * vo[iyo] + mu**2 * (vc0 - vi[iyi0] - vo[iyo])
                vo[iyo] += mu * (vi[iyi1] - vc0)
                vo[iyo] += vi[iyi0]

        gap = 0

    # Extrapolation with nearest
    if (extrap == -1 or extrap == 2) and yo[iyomin] < yi[iyimin]:
        for iyo in range(iyomin, iyomax + 1):
            if yo[iyo] >= yi[iyimin]:
                vo[:iyo] = vi[iyimin]
                break
    if (extrap == 1 or extrap == 2) and yo[iyomax] > yi[iyimax]:
        for iyo in range(iyomin, iyomax + 1):
            if yo[iyo] > yi[iyimax]:
                vo[iyo:] = vi[iyimax]
                break


@numba.njit(nogil=True, cache=NOT_CI)
def _hermit1d_col_(vi, yi, yo, vo, extrap, bias, tension, drop_na, maxgap):
    iyimin, iyimax = _iminmax_valid_(yi, vi)
    iyomin, iyomax = get_iminmax(yo)
    iyominv = iyomin
    gap = 0
    for iyi in range(iyimin, iyimax):
        # Out of bounds
        if yi[iyi + 1] < yo[iyomin]:
            continue
        if yi[iyi] > yo[iyomax]:
            break

        # Gap check
        if (
            drop_na
            and (np.isnan(yi[iyi + 1]) or np.isnan(vi[iyi + 1]))
            and (maxgap == 0 or gap < maxgap)
        ):
            gap += 1
            continue

        iyi0 = iyi - gap
        iyi1 = iyi + 1
        if iyi0 == iyimin or np.isnan(vi[iyi0 - 1]):
            iyim1 = iyi0
        else:
            iyim1 = iyi0 - 1
        if iyi1 == iyimax or np.isnan(vi[iyi1 + 1]):
            iyi2 = iyi1
        else:
            iyi2 = iyi1 + 1

        # Loop on output grid
        for iyo in range(iyominv, iyomax + 1):
            dy0 = yo[iyo] - yi[iyi0]
            dy1 = yi[iyi1] - yo[iyo]

            # Above
            if dy1 < 0.0:  # above
                break

            # Below
            if dy0 < 0.0:
                iyominv = iyo + 1

            # Inside
            if dy0 >= 0.0 and dy1 >= 0.0:
                iyominv = iyo
                mu = dy0 / (dy0 + dy1)

                # Extrapolations
                if iyi0 == iyimin:  # y0
                    vc0 = 2 * vi[iyi0] - vi[iyi1]
                else:
                    vc0 = vi[iyim1]
                if iyi1 == iyimax:  # y3
                    vc1 = 2 * vi[iyi1] - vi[iyi0]
                else:
                    vc1 = vi[iyi2]

                # Interpolation
                mu = dy0 / (dy0 + dy1)
                a0 = 2 * mu**3 - 3 * mu**2 + 1
                a1 = mu**3 - 2 * mu**2 + mu
                a2 = mu**3 - mu**2
                a3 = -2 * mu**3 + 3 * mu**2
                vo[iyo] = a0 * vi[iyi0]
                vo[iyo] += a1 * (
                    (vi[iyi0] - vc0) * (1 + bias) * (1 - tension) / 2
                    + (vi[iyi1] - vi[iyi0]) * (1 - bias) * (1 - tension) / 2
                )
                vo[iyo] += a2 * (
                    (vi[iyi1] - vi[iyi0]) * (1 + bias) * (1 - tension) / 2
                    + (vc1 - vi[iyi1]) * (1 - bias) * (1 - tension) / 2
                )
                vo[iyo] += a3 * vi[iyi1]

        gap = 0

    # Extrapolation with nearest
    if (extrap == -1 or extrap == 2) and yo[iyomin] < yi[iyimin]:
        for iyo in range(iyomin, iyomax + 1):
            if yo[iyo] >= yi[iyimin]:
                vo[:iyo] = vi[iyimin]
                break
    if (extrap == 1 or extrap == 2) and yo[iyomax] > yi[iyimax]:
        for iyo in range(iyomin, iyomax + 1):
            if yo[iyo] > yi[iyimax]:
                vo[iyo:] = vi[iyimax]
                break


@numba.njit(nogil=True, cache=NOT_CI)
def _cellave1d_col_(vi, yib, yob, vo, extrap, conserv, na_thres):
    nyi = vi.size
    nyo = vo.size

    # Loop on output cells to be filled
    iyi0 = 0
    for iyo in range(nyo):
        if yob[iyo] == yob[iyo + 1]:
            continue

        # Loop on input cells
        wo = 0.0
        for iyi in range(iyi0, nyi):
            # Current input bounds
            yib0 = yib[iyi]
            yib1 = yib[iyi + 1]

            # Extrapolation
            if (extrap == -1 or extrap == 2) and iyi == 0 and yib0 > yob[iyo]:
                yib0 = yob[iyo]
            if (extrap == 1 or extrap == 2) and iyi == nyi - 1 and yib1 < yob[iyo + 1]:
                yib1 = yob[iyo + 1]

            # No intersection
            if yib0 > yob[iyo + 1]:
                break
            if yib1 < yob[iyo]:
                iyi0 = iyi + 1
                continue

            # Contribution of intersection
            dyio = min(yib1, yob[iyo + 1]) - max(yib0, yob[iyo])
            if conserv and yib0 != yib1:
                dyio = dyio / (yob[iyo + 1] - yob[iyo])
            if not np.isnan(vi[iyi]):
                wo = wo + dyio
                vo[iyo] += vi[iyi] * dyio

            # Next input cell?
            if yib1 >= yob[iyo + 1]:
                break

        # Normalize, or nan where the valid input covers too little of the output cell
        coverage = wo if conserv else wo / (yob[iyo + 1] - yob[iyo])
        if wo == 0 or coverage < na_thres:
            vo[iyo] = np.nan
        elif not conserv:
            vo[iyo] /= wo


@numba.njit(parallel=True, cache=NOT_CI)
def _regrid1d_kernel_(
    vi, yi, yo, method, extrap, bias, tension, drop_na, maxgap, conserv, na_thres
):
    """Loop on the columns of (m, n, m) arrays, the outer and inner m being 1 or the size

    Options are numbers rather than strings, whose reference counting
    would slow down the parallel loop.
    """
    no = max(vi.shape[0], yi.shape[0], yo.shape[0])
    ni = max(vi.shape[2], yi.shape[2], yo.shape[2])
    if method < 0:  # cell methods: yo are edges
        varo = np.zeros((no, yo.shape[1] - 1, ni), dtype=vi.dtype)
    else:
        varo = np.full((no, yo.shape[1], ni), np.nan, dtype=vi.dtype)
    for ic in numba.prange(no * ni):
        io = ic // ni
        p = ic % ni
        v = vi[io % vi.shape[0], :, p % vi.shape[2]]
        y = yi[io % yi.shape[0], :, p % yi.shape[2]]
        z = yo[io % yo.shape[0], :, p % yo.shape[2]]
        o = varo[io, :, p]
        if method == 0:
            _nearest1d_col_(v, y, z, o, extrap, drop_na, maxgap)
        elif method == 1:
            _linear1d_col_(v, y, z, o, extrap, drop_na, maxgap)
        elif method == 2:
            _cubic1d_col_(v, y, z, o, extrap, drop_na, maxgap)
        elif method == 3:
            _hermit1d_col_(v, y, z, o, extrap, bias, tension, drop_na, maxgap)
        else:
            _cellave1d_col_(v, y, z, o, extrap, conserv, na_thres)
    return varo


def _regrid1d_(
    vari,
    yi,
    yo,
    axis,
    method,
    extrap="no",
    bias=0.0,
    tension=0.0,
    drop_na=False,
    maxgap=0,
    conserv=False,
    na_thres=0.0,
):
    """Broadcast the arrays along an axis, call the kernel and reshape the output"""
    (vari, yi, yo), outer, inner = broadcast_along_axis(
        as_float_array(vari), as_float_array(yi), as_float_array(yo), axis=axis
    )
    varo = _regrid1d_kernel_(
        vari,
        yi,
        yo,
        method,
        _EXTRAP_CODES_[str(extrap)],
        float(bias),
        float(tension),
        bool(drop_na),
        int(maxgap),
        bool(conserv),
        float(na_thres),
    )
    return varo.reshape(outer + varo.shape[1:2] + inner)


_DOC_1D_ = """
    The arrays are broadcasted against each other except along `axis`, like with numpy,
    so the coordinates may vary along the other axes, or not, without a copy.
    See :func:`xoa.core.num.broadcast_along_axis`.

    Parameters
    ----------
    vari: array_like
        Input data
    {coords}
    axis: int
        Working axis, which is kept in place, relative to the array of greatest
        number of dimensions
    extrap: {{"no", "bottom", "top", "both"}}
        Extrapolation mode{params}

    Return
    ------
    numpy.ndarray
        Output data with the broadcasted shape of the arrays,
        and the size of `yo` along `axis`
"""
_DOC_INTERP_COORDS_ = """yi: array_like
        Input coordinates, increasing along `axis`
    yo: array_like
        Output coordinates, increasing along `axis`"""
_DOC_DROP_NA_ = """
    drop_na: bool
        Interpolate across the inner NaNs of the input data
    maxgap: int
        Maximal number of consecutive NaNs to interpolate across when `drop_na`
        is True, which is not checked when it is zero"""


def nearest1d(vari, yi, yo, axis=-1, extrap="no", drop_na=False, maxgap=0):
    return _regrid1d_(vari, yi, yo, axis, 0, extrap, drop_na=drop_na, maxgap=maxgap)


def linear1d(vari, yi, yo, axis=-1, extrap="no", drop_na=False, maxgap=0):
    return _regrid1d_(vari, yi, yo, axis, 1, extrap, drop_na=drop_na, maxgap=maxgap)


def cubic1d(vari, yi, yo, axis=-1, extrap="no", drop_na=False, maxgap=0):
    return _regrid1d_(vari, yi, yo, axis, 2, extrap, drop_na=drop_na, maxgap=maxgap)


def hermit1d(vari, yi, yo, axis=-1, extrap="no", bias=0.0, tension=0.0, drop_na=False, maxgap=0):
    return _regrid1d_(vari, yi, yo, axis, 3, extrap, bias, tension, drop_na, maxgap)


def cellave1d(vari, yib, yob, axis=-1, extrap="no", conserv=False, na_thres=0.0):
    return _regrid1d_(vari, yib, yob, axis, -1, extrap, conserv=conserv, na_thres=na_thres)


for _func, _title, _coords, _params in [
    (nearest1d, "Nearest interpolation", _DOC_INTERP_COORDS_, _DOC_DROP_NA_),
    (linear1d, "Linear interpolation", _DOC_INTERP_COORDS_, _DOC_DROP_NA_),
    (cubic1d, "Cubic interpolation", _DOC_INTERP_COORDS_, _DOC_DROP_NA_),
    (
        hermit1d,
        "Hermitian interpolation",
        _DOC_INTERP_COORDS_,
        """
    bias: float
        Bias of the Kochanek-Bartels tangents
    tension: float
        Tension of the Kochanek-Bartels tangents"""
        + _DOC_DROP_NA_,
    ),
    (
        cellave1d,
        "Cell averaging or conservative regridding",
        """yib: array_like
        Input cell edges, increasing along `axis`, with one more element than `vari`
    yob: array_like
        Output cell edges, increasing along `axis`""",
        """
    conserv: bool
        Conservative regridding: the integral is preserved instead of the average
    na_thres: float
        Minimal fraction, from 0 to 1, of an output cell that must be covered
        by valid input cells, else the output is NaN.
        Zero means that any valid input is enough.""",
    ),
]:
    _func.__doc__ = (
        f"{_title} of nD data along an axis with varying coordinates\n"
        + _DOC_1D_.format(coords=_coords, params=_params)
    )


@numba.njit(parallel=True, cache=NOT_CI)
def _extrap1d_kernel_(vari, mode):
    varo = vari.copy()
    if mode == 0:
        return varo
    no, _, ni = vari.shape
    for ic in numba.prange(no * ni):
        vo = varo[ic // ni, :, ic % ni]
        iybot, iytop = get_iminmax(vo)
        if iybot == -1:
            continue
        if mode == 2 or mode == -1:
            vo[:iybot] = vo[iybot]
        if mode == 2 or mode == 1:
            vo[iytop + 1 :] = vo[iytop]
    return varo


def extrap1d(vari, mode, axis=-1):
    """Extrapolate valid data to the top and/or bottom along an axis

    Parameters
    ----------
    vari: array_like
    mode: {"top", "bottom", "both", "no"}
        Extrapolation mode
    axis: int
        Working axis, which is kept in place

    Return
    ------
    numpy.ndarray
        Array of the same shape as `vari`
    """
    vari = as_float_array(vari)
    mode = _EXTRAP_CODES_[str(mode)]
    return _extrap1d_kernel_(view_along_axis(vari, axis), mode).reshape(vari.shape)


# %% Horizontal regridding


class XYRegridder(XYInterpolator):
    """
    Numba-accelerated horizontal regridder for curvilinear and rectangular horizontal grids.

    Inherits bilinear and bicubic interpolation from XYInterpolator.
    Adds conservative remapping (requires destination to be a structured 2D grid).
    """

    valid_methods = ['bilinear', 'bicubic', 'conservative']

    def __init__(self, src_grid, dst_grid, method='bilinear', num_threads=0, bias=0.0, tension=0.0):
        if method not in self.valid_methods:
            raise ValueError(
                f"Method '{method}' not supported. Valid methods: {self.valid_methods}"
            )

        self.dst_grid = dst_grid.copy()
        self.num_threads = num_threads

        # Conservative CSR state (None for bilinear/bicubic)
        self.weights = None
        self._nb_indptr = None
        self._nb_indices = None
        self._nb_wdata = None

        if method == 'conservative':
            self.src_grid = src_grid.copy()
            self.method = method
            self.bias = bias
            self.tension = tension
            self._j_base = None
            self._i_base = None
            self._frac_a = None
            self._frac_b = None
            self._valid_dst_mask = None
            self._dst_shape = self.dst_grid['lon'].shape
            self._validate_conservative_grids()
            self._prepare_grids()
        else:
            super().__init__(
                src_grid, dst_grid['lon'], dst_grid['lat'], method, bias=bias, tension=tension
            )

    @property
    def has_weights(self):
        """True if the weights have been computed or loaded."""
        if self.method == 'conservative':
            return self._nb_indptr is not None
        return super().has_weights

    def _validate_conservative_grids(self):
        for grid_name, grid in [('source', self.src_grid), ('destination', self.dst_grid)]:
            for key in ['lon', 'lat']:
                if key not in grid:
                    raise ValueError(f"{grid_name} grid missing required key: '{key}'")

            if not isinstance(grid['lon'], np.ndarray) or not isinstance(grid['lat'], np.ndarray):
                raise ValueError(f"{grid_name} grid 'lon' and 'lat' must be numpy arrays")

            if grid['lon'].shape != grid['lat'].shape:
                raise ValueError(f"{grid_name} grid 'lon' and 'lat' must have the same shape")

            if len(grid['lon'].shape) != 2:
                raise ValueError(f"{grid_name} grid 'lon' and 'lat' must be 2D arrays")

            ny, nx = grid['lon'].shape
            if 'lon_bounds' not in grid or 'lat_bounds' not in grid:
                if 'lon_edges' not in grid:
                    grid['lon_edges'] = centers2edges(unwrap_grid_longitudes(grid['lon']))
                    grid['lat_edges'] = centers2edges(grid['lat'])
                grid['lon_bounds'] = edges2bounds(grid['lon_edges'])
                grid['lat_bounds'] = edges2bounds(grid['lat_edges'])
            elif grid['lon_bounds'].shape != (ny, nx, 4) or grid['lat_bounds'].shape != (ny, nx, 4):
                raise ValueError(f'{grid_name} grid corner arrays must have shape ({ny}, {nx}, 4)')

            grid['lon'] = normalize_longitudes(grid['lon'])

            if np.any(grid['lat'] < -90) or np.any(grid['lat'] > 90):
                raise ValueError(f'{grid_name} grid latitudes must be in range [-90, 90]')

            if 'mask' in grid and grid['mask'] is not None:
                if grid['mask'].shape != grid['lon'].shape:
                    raise ValueError(f'{grid_name} grid mask must have same shape as coordinates')
                if grid['mask'].dtype != bool:
                    xoa_warn(f'Converting {grid_name} grid mask to boolean')
                    grid['mask'] = grid['mask'].astype(bool)

            if 'type' not in grid or grid['type'] is None:
                check_grid_type(grid)
            else:
                if grid['type'] not in self.valid_grid_types:
                    raise ValueError(f'Unsupported grid type {grid["type"]}')

    def _prepare_grids(self):
        for grid in [self.src_grid, self.dst_grid]:
            for key in grid:
                if isinstance(grid[key], np.ndarray) and key != 'mask':
                    grid[key] = np.ascontiguousarray(grid[key], dtype=np.float64)

    def compute_weights(self, skipna=False):
        """Compute interpolation weights.

        For bilinear/bicubic: computes fractional cell indices (cheap, no CSR).
        For conservative: computes a precomputed CSR weight matrix.
        The skipna parameter is kept for API compatibility.
        """
        if self.method != 'conservative':
            super().compute_weights(skipna)
            return None

        n_dst = self.dst_grid['lat'].size
        n_src = self.src_grid['lat'].size

        kw = {}
        if 'mask' in self.dst_grid and self.dst_grid['mask'] is not None:
            kw['dst_mask'] = self.dst_grid['mask'].ravel()
        row_indices, col_indices, weights_data, valid_dst = compute_conservative_weights(
            self.src_grid['lat_bounds'].reshape(-1, 4),
            self.src_grid['lon_bounds'].reshape(-1, 4),
            self.dst_grid['lat_bounds'].reshape(-1, 4),
            self.dst_grid['lon_bounds'].reshape(-1, 4),
            **kw,
        )
        self.weights = coo_to_csr(row_indices, col_indices, weights_data, n_dst, n_src)
        self._nb_indptr = self.weights.indptr
        self._nb_indices = self.weights.indices
        self._nb_wdata = self.weights.data
        self._valid_dst_mask = valid_dst
        return self.weights

    def _apply_conservative(self, data, skipna, na_thres):
        n_dst = self.dst_grid['lat'].size
        extra = data.shape[:-2]
        K = int(np.prod(extra)) if extra else 1

        X = get_source_matrix(data, self.src_grid.get('mask'))  # (n_src, K)
        out = np.empty((n_dst, K), dtype=np.float64)

        csr_spmm_skipna_unified(
            self._nb_indptr,
            self._nb_indices,
            self._nb_wdata,
            self._nb_wdata,
            EMPTY_OFFSETS,
            X,
            out,
            na_thres,
        )

        dst_shape = self.dst_grid['lon'].shape
        return out.T.reshape(extra + dst_shape)

    def regrid(self, data, skipna=False, na_thres=1.0):
        """
        Regrid data from source to destination grid.

        Parameters
        ----------
        data : np.ndarray, shape (..., ny_src, nx_src)
        skipna : bool
        na_thres : float

        Returns
        -------
        np.ndarray, shape (..., ny_dst, nx_dst)
        """
        if self.method == 'conservative':
            if self._nb_indptr is None:
                self.compute_weights()
            data = np.asarray(data, dtype=np.float64)
            if data.shape[-2:] != self.src_grid['lon'].shape:
                raise ValueError(
                    f"Data shape {data.shape} doesn't match source grid {self.src_grid['lon'].shape}"
                )
            return self._apply_conservative(data, skipna, na_thres)
        return self.interp(data, skipna, na_thres)

    def save_weights(self, filename):
        """Save computed weights to a npz file"""
        if self.method != 'conservative':
            super().save_weights(filename)
            return

        if self._nb_indptr is None:
            raise ValueError('No weights computed yet')

        with open(filename, "wb") as f:
            np.savez(
                f,
                method=self.method,
                src_shape=self.src_grid['lon'].shape,
                dst_shape=self.dst_grid['lon'].shape,
                valid_dst_mask=self._valid_dst_mask,
                nb_indptr=self._nb_indptr,
                nb_indices=self._nb_indices,
                nb_wdata=self._nb_wdata,
            )

    def load_weights(self, filename):
        """Load precomputed weights from a npz file"""
        if self.method != 'conservative':
            super().load_weights(filename)
            return

        with np.load(filename) as p:
            p = {key: p[key] for key in p.files}

        if self.src_grid['lon'].shape != tuple(p['src_shape']):
            raise ValueError(
                f"Source grid shape {self.src_grid['lon'].shape} "
                f"doesn't match saved weights shape {tuple(p['src_shape'])}"
            )
        if self.dst_grid['lon'].shape != tuple(p['dst_shape']):
            raise ValueError(
                f"Destination grid shape {self.dst_grid['lon'].shape} "
                f"doesn't match saved weights shape {tuple(p['dst_shape'])}"
            )
        if str(p['method']) != self.method:
            xoa_warn(f"Saved method '{p['method']}' differs from '{self.method}'")

        self._valid_dst_mask = p['valid_dst_mask']
        self._nb_indptr = p.get('nb_indptr')
        self._nb_indices = p.get('nb_indices')
        self._nb_wdata = p.get('nb_wdata')
        if self._nb_indptr is not None:
            n_dst = self.dst_grid['lat'].size
            n_src = self.src_grid['lat'].size
            self.weights = CsrMatrix(
                self._nb_indptr, self._nb_indices, self._nb_wdata, (n_dst, n_src)
            )
