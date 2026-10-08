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

import numpy as np
import numba

from ..exceptions import xoa_warn
from .conserv import compute_conservative_weights
from .grid import centers2edges, check_grid_type, edges2bounds, unwrap_grid_longitudes
from .interp import XYInterpolator, _get_source_matrix_
from .num import (
    EMPTY_OFFSETS,
    CsrMatrix,
    coo_to_csr,
    csr_spmm_skipna_unified,
    get_iminmax,
    ravel_index,
    unravel_index,
)

NOT_CI = os.environ.get("CI", "false") == "false"


@numba.njit(parallel=True, cache=NOT_CI)
def nearest1d(vari, yi, yo, eshapes, extrap="no", drop_na=False, maxgap=0):
    """Nearest interpolation of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0.0:  # above
                    break

                # Below
                if dy0 < 0.0:
                    iyominv = iyo + 1

                # Interpolations
                elif dy0 <= dy1:
                    varo[ix, iyo] = vari[ixi, iyi0]
                else:
                    varo[ix, iyo] = vari[ixi, iyi1]

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] >= yi[ixiy, iyimin]:
                    varo[ix, :iyo] = vari[ixi, iyimin]
                    break
        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo:] = vari[ixi, iyimax]
                    break

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def linear1d(vari, yi, yo, eshapes, extrap="no", drop_na=False, maxgap=0):
    """Linear interpolation of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

                # Above
                if dy1 < 0.0:  # above
                    break

                # Below
                if dy0 < 0.0:
                    iyominv = iyo + 1

                # Interpolation
                elif dy0 > 0.0 or dy1 > 0.0:
                    varo[ix, iyo] = (vari[ixi, iyi0] * dy1 + vari[ixi, iyi1] * dy0) / (dy0 + dy1)

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] < yi[ixiy, iyimin]:
                    varo[ix, iyo] = vari[ixi, iyimin]
                else:
                    break

        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomax, iyomin - 1, -1):  # Loop backwards
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo] = vari[ixi, iyimax]
                else:
                    break

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def cubic1d(vari, yi, yo, eshapes, extrap="no", drop_na=False, maxgap=0):
    """Cubic interpolation of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1
            if iyi0 == iyimin or np.isnan(vari[ixi, iyi0 - 1]):
                iyim1 = iyi0
            else:
                iyim1 = iyi0 - 1
            if iyi1 == iyimax or np.isnan(vari[ixi, iyi1 + 1]):
                iyi2 = iyi1
            else:
                iyi2 = iyi1 + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

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
                        vc0 = 2 * vari[ixi, iyi0] - vari[ixi, iyi1]
                    else:
                        vc0 = vari[ixi, iyim1]
                    if iyi1 == iyimax:  # y3
                        vc1 = 2 * vari[ixi, iyi1] - vari[ixi, iyi0]
                    else:
                        vc1 = vari[ixi, iyi2]

                    # Interpolation
                    varo[ix, iyo] = vc1 - vari[ixi, iyi1] - vc0 + vari[ixi, iyi0]
                    varo[ix, iyo] = mu**3 * varo[ix, iyo] + mu**2 * (
                        vc0 - vari[ixi, iyi0] - varo[ix, iyo]
                    )
                    varo[ix, iyo] += mu * (vari[ixi, iyi1] - vc0)
                    varo[ix, iyo] += vari[ixi, iyi0]

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] >= yi[ixiy, iyimin]:
                    varo[ix, :iyo] = vari[ixi, iyimin]
                    break
        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo:] = vari[ixi, iyimax]
                    break

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def hermit1d(
    vari,
    yi,
    yo,
    eshapes,
    extrap="no",
    bias=0.0,
    tension=0.0,
    drop_na=False,
    maxgap=0,
):
    """Hermitian interp. of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yi: array_like(nxiy, nyi)
    yo: array_like(nxo, nyo)
    eshapes: array_like(3, ndim-1)
    bias: float
    tension: float

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    nyo = yo.shape[1]
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.full((nx, nyo), np.nan, dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for all arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on input grid
        iyimin, iyimax = get_iminmax(yi[ixiy] * vari[ixi])
        iyomin, iyomax = get_iminmax(yo[ixoy])
        iyominv = iyomin
        gap = 0
        for iyi in range(iyimin, iyimax):
            # Out of bounds
            if yi[ixiy, iyi + 1] < yo[ixoy, iyomin]:
                continue
            if yi[ixiy, iyi] > yo[ixoy, iyomax]:
                break

            # Gap check
            if (
                drop_na
                and (np.isnan(yi[ixiy, iyi + 1]) or np.isnan(vari[ixi, iyi + 1]))
                and (maxgap == 0 or gap < maxgap)
            ):
                gap += 1
                continue

            iyi0 = iyi - gap
            iyi1 = iyi + 1
            if iyi0 == iyimin or np.isnan(vari[ixi, iyi0 - 1]):
                iyim1 = iyi0
            else:
                iyim1 = iyi0 - 1
            if iyi1 == iyimax or np.isnan(vari[ixi, iyi1 + 1]):
                iyi2 = iyi1
            else:
                iyi2 = iyi1 + 1

            # Loop on output grid
            for iyo in range(iyominv, iyomax + 1):
                dy0 = yo[ixoy, iyo] - yi[ixiy, iyi0]
                dy1 = yi[ixiy, iyi1] - yo[ixoy, iyo]

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
                        vc0 = 2 * vari[ixi, iyi0] - vari[ixi, iyi1]
                    else:
                        vc0 = vari[ixi, iyim1]
                    if iyi1 == iyimax:  # y3
                        vc1 = 2 * vari[ixi, iyi1] - vari[ixi, iyi0]
                    else:
                        vc1 = vari[ixi, iyi2]

                    # Interpolation
                    mu = dy0 / (dy0 + dy1)
                    a0 = 2 * mu**3 - 3 * mu**2 + 1
                    a1 = mu**3 - 2 * mu**2 + mu
                    a2 = mu**3 - mu**2
                    a3 = -2 * mu**3 + 3 * mu**2
                    varo[ix, iyo] = a0 * vari[ixi, iyi0]
                    varo[ix, iyo] += a1 * (
                        (vari[ixi, iyi0] - vc0) * (1 + bias) * (1 - tension) / 2
                        + (vari[ixi, iyi1] - vari[ixi, iyi0]) * (1 - bias) * (1 - tension) / 2
                    )
                    varo[ix, iyo] += a2 * (
                        (vari[ixi, iyi1] - vari[ixi, iyi0]) * (1 + bias) * (1 - tension) / 2
                        + (vc1 - vari[ixi, iyi1]) * (1 - bias) * (1 - tension) / 2
                    )
                    varo[ix, iyo] += a3 * vari[ixi, iyi1]

            gap = 0

        # Extrapolation with nearest
        if extrap in ("bottom", "both") and yo[ixoy, iyomin] < yi[ixiy, iyimin]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] >= yi[ixiy, iyimin]:
                    varo[ix, :iyo] = vari[ixi, iyimin]
                    break
        if extrap in ("top", "both") and yo[ixoy, iyomax] > yi[ixiy, iyimax]:
            for iyo in range(iyomin, iyomax + 1):
                if yo[ixoy, iyo] > yi[ixiy, iyimax]:
                    varo[ix, iyo:] = vari[ixi, iyimax]
                    break

    return varo


@numba.njit(parallel=True)
def extrap1d(vari, mode):
    """Extrapolate valid data to the top and/or bottom

    Parameters
    ----------
    vari: array_like(nx, ny)
    mode: {"top", "bottom", "both", "no"}
        Extrapolation mode

    Return
    ------
    array_like(nx, ny): varo
    """
    varo = vari.copy()
    if mode == "no":
        return varo
    nx, ny = vari.shape

    # Loop on varying dim
    for ix in numba.prange(0, nx):
        iybot, iytop = get_iminmax(vari[ix])
        if iybot == -1:
            continue
        if mode == "both" or mode == "bottom":
            varo[ix, :iybot] = varo[ix, iybot]
        if mode == "both" or mode == "top":
            varo[ix, iytop + 1 :] = varo[ix, iytop]

    return varo


@numba.njit(parallel=True, cache=NOT_CI)
def cellave1d(
    vari,
    yib,
    yob,
    eshapes,
    extrap="no",
    conserv=False,
    drop_na=False,
    maxgap=0,
):
    """Cell average regrid. of nD data along an axis with varying coordinates

    Warning
    -------
    `nxi` must be either a multiple or a divisor of `nxo`,
    and multiple of `nxiy`.

    Parameters
    ----------
    vari: array_like(nxi, nyi)
    yib: array_like(nxiy, nyi+1)
    yob: array_like(nxo, nyo+1)

    Return
    ------
    array_like(nx, nyo): varo
        With `nx=max(nxi, nxo)`
    """
    # Shapes
    # nxi, nyib = vari.shape
    # nxiy, nyi = yib.shape
    # nxi, nyi = vari.shape
    # nxo, nyob = yob.shape
    # nx = max(nxi, nxo)
    # nyo = nyob - 1

    nyi = vari.shape[1]
    nyo = yob.shape[1] - 1
    eshape = np.empty(eshapes.shape[1], eshapes.dtype)
    for i in range(eshape.size):
        eshape[i] = eshapes[:, i].max()
    nx = np.prod(eshape)

    # Init output
    varo = np.zeros((nx, nyo), dtype=vari.dtype)

    # Loop on the varying dimension
    for ix in numba.prange(nx):
        # Index along x for coordinate arrays
        ii = unravel_index(ix, eshape)
        ixi = ravel_index(np.minimum(ii, eshapes[0] - 1), eshapes[0])
        ixiy = ravel_index(np.minimum(ii, eshapes[1] - 1), eshapes[1])
        ixoy = ravel_index(np.minimum(ii, eshapes[2] - 1), eshapes[2])

        # Loop on output cells to be filled
        iyi0 = 0
        for iyo in range(nyo):
            if yob[ixoy, iyo] == yob[ixoy, iyo + 1]:
                continue

            # Loop on input cells
            wo = 0.0
            for iyi in range(iyi0, nyi):
                # Current input bounds
                yib0 = yib[ixiy, iyi]
                yib1 = yib[ixiy, iyi + 1]

                # Extrapolation
                if (extrap == "bellow" or extrap == "both") and iyi == 0 and yib0 > yob[ixoy, iyo]:
                    yib0 = yob[ixoy, iyo]
                if (
                    (extrap == "above" or extrap == "both")
                    and iyi == nyi - 1
                    and yib1 < yob[ixoy, iyo + 1]
                ):
                    yib1 = yob[ixoy, iyo + 1]

                # No intersection
                if yib0 > yob[ixoy, iyo + 1]:
                    break
                if yib1 < yob[ixoy, iyo]:
                    iyi0 = iyi + 1
                    continue

                # Contribution of intersection
                dyio = min(yib1, yob[ixoy, iyo + 1]) - max(yib0, yob[ixoy, iyo])
                if conserv and yib0 != yib1:
                    dyio = dyio / (yob[ixoy, iyo + 1] - yob[ixoy, iyo])
                if not np.isnan(vari[ixi, iyi]):
                    wo = wo + dyio
                    varo[ixi, iyo] += vari[ixi, iyi] * dyio

                # Next input cell?
                if yib1 >= yob[ixoy, iyo + 1]:
                    break

            # Normalize
            if not conserv:
                if wo != 0:
                    varo[ix, iyo] /= wo
                else:
                    varo[ix, iyo] = np.nan

    return varo


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

            if np.any(grid['lon'] < -180) or np.any(grid['lon'] > 180):
                raise ValueError(f'{grid_name} grid longitudes must be in range [-180, 180]')

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

        X = _get_source_matrix_(data, self.src_grid.get('mask'))  # (n_src, K)
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
