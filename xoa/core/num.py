"""
Low level numeric utilities

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

NOT_CI = os.environ.get("CI", "false") == "false"
EPSILON = 1e-12


@numba.njit(cache=NOT_CI)
def get_iminmax(data1d):
    """The first and last non nan values for a 1d array

    Parameters
    ----------
    data1d: array_like(n)

    Return
    ------
    int
        Index of the first valid value
    int
        Index of the last valid value
    """
    imin = -1
    imax = -1
    n = len(data1d)
    for i in range(n):
        if imin < 0 and not np.isnan(data1d[i]):
            imin = i
        if imax < 0 and not np.isnan(data1d[n - 1 - i]):
            imax = n - 1 - i
        if imax > 0 and imin > 0:
            break
    return imin, imax


@numba.njit(numba.int64[:](numba.int64, numba.int64[:]), cache=NOT_CI)
def unravel_index(i, shape):

    ir = i
    ndim = len(shape)
    ii = np.zeros(ndim, np.int64)
    for o in range(ndim):
        if o != ndim - 1:
            base = np.prod(shape[o + 1 :])
        else:
            base = 1
        ii[o] = ir // base
        ir -= ii[o] * base
        # print(o, base, ir)
    return ii


@numba.njit(numba.int64(numba.int64[:], numba.int64[:]), cache=NOT_CI)
def ravel_index(ii, shape):
    ir = 0
    ndim = len(shape)
    for o in range(ndim):
        if o != ndim - 1:
            base = np.prod(shape[o + 1 :])
        else:
            base = 1
        # print(ii[o], base)
        ir += ii[o] * base
    return ir


def as_float_array(arr):
    """Convert input to at least 1D float array, useful for numba accelerated functions

    Parameter
    ---------
    arr: boolean, int, float, datetime64, dask.array

    Returns
    -------
    array
        Array of floats
    """
    arr = np.asarray(arr)
    arr = np.atleast_1d(arr)
    if arr.dtype.type is np.datetime64:
        arr = (arr - np.datetime64("1950-01-01", "us")) / np.timedelta64(1, "us")
    elif arr.dtype.char in 'il?':
        arr = arr.astype("d")
    return arr


EMPTY_OFFSETS = np.empty(0, dtype=np.int64)


class CsrMatrix:
    """Minimal CSR sparse matrix — replaces scipy.sparse.csr_matrix for the regridder."""

    __slots__ = ('indptr', 'indices', 'data', 'shape', 'nnz')

    def __init__(self, indptr, indices, data, shape):
        self.indptr = np.asarray(indptr, dtype=np.int64)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.data = np.asarray(data, dtype=np.float64)
        self.shape = tuple(shape)
        self.nnz = int(len(self.data))


def coo_to_csr(row_indices, col_indices, weights_data, n_rows, n_cols):
    """Convert COO sparse triplets to a CsrMatrix.

    Parameters
    ----------
    row_indices, col_indices : array-like of int
        Row and column indices of non-zero entries (COO format).
    weights_data : array-like of float
        Non-zero values.
    n_rows, n_cols : int
        Matrix dimensions.

    Returns
    -------
    CsrMatrix
    """
    row = np.asarray(row_indices, dtype=np.int64)
    col = np.asarray(col_indices, dtype=np.int64)
    data = np.asarray(weights_data, dtype=np.float64)
    order = np.argsort(row, kind='stable')
    col_sorted = col[order]
    data_sorted = data[order]
    counts = np.bincount(row[order], minlength=n_rows)
    indptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return CsrMatrix(indptr, col_sorted, data_sorted, (n_rows, n_cols))


@numba.njit(cache=NOT_CI, parallel=True)
def csr_spmm_skipna_unified(
    indptr, indices, full_wdata, core_wdata, core_offsets, X, out, na_thres=1.0
):
    """
    Unified skipna SpMM: out[i,k] = weighted sum of X[stencil, k] with NaN renorm.

    For each output cell (i, k):
      1. Scan the full stencil for NaN.
      2. No NaN  → apply full_wdata (clean interpolation).
      3. NaN found, core_offsets empty → renormalize over all entries (bilinear/conservative).
      4. NaN found, core_offsets set  → renormalize over core entries only (bicubic fallback).

    Parameters
    ----------
    indptr      : int64[n_dst+1]
    indices     : int64[nnz]
    full_wdata  : float64[nnz]   weights for the no-NaN path
    core_wdata  : float64[nnz]   weights for the NaN fallback (same as full_wdata for non-bicubic)
    core_offsets: int64[n_core]  intra-row offsets; EMPTY_OFFSETS → use all entries
    X           : float64[n_src, K]  C-contiguous
    out         : float64[n_dst, K]  pre-allocated, overwritten in place
    na_thres    : float
    """
    n_dst = indptr.shape[0] - 1
    K = X.shape[1]
    thres = min(max(1.0 - na_thres, EPSILON), 1.0 - EPSILON)
    n_core = core_offsets.shape[0]
    use_all_fallback = n_core == 0

    for i in numba.prange(n_dst):
        p0 = indptr[i]
        p1 = indptr[i + 1]

        if p0 == p1:
            for k in range(K):
                out[i, k] = np.nan
            continue

        for k in range(K):
            has_nan = False
            for p in range(p0, p1):
                if np.isnan(X[indices[p], k]):
                    has_nan = True
                    break

            if not has_nan:
                s = 0.0
                for p in range(p0, p1):
                    s += full_wdata[p] * X[indices[p], k]
                out[i, k] = s

            elif use_all_fallback:
                s = 0.0
                w_valid = 0.0
                for p in range(p0, p1):
                    v = X[indices[p], k]
                    if not np.isnan(v):
                        w = core_wdata[p]
                        s += w * v
                        w_valid += w
                out[i, k] = np.nan if w_valid < thres else s / w_valid

            else:
                s = 0.0
                w_valid = 0.0
                for c in range(n_core):
                    p = p0 + core_offsets[c]
                    v = X[indices[p], k]
                    if not np.isnan(v):
                        w = core_wdata[p]
                        s += w * v
                        w_valid += w
                out[i, k] = np.nan if w_valid < thres else s / w_valid
