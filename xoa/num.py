"""
Low level numeric utilities

The numerical inputs and outputs of all these routines are of scalar
or numpy.ndarray type.
"""
# Copyright 2020-2022 Shom
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
