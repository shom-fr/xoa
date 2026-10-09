"""
Low level statistics

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

NOT_CI = os.environ.get("CI", "false") == "false"


@numba.njit(cache=NOT_CI, parallel=True)
def _taylor_stats_(model, ref):
    n, m = model.shape
    std = np.full(n, np.nan)
    std_ref = np.full(n, np.nan)
    corr = np.full(n, np.nan)
    crmsd = np.full(n, np.nan)
    for i in numba.prange(n):
        # Only the points that are valid in both arrays are used
        cnt = 0
        sm = 0.0
        sr = 0.0
        for j in range(m):
            if not (np.isnan(model[i, j]) or np.isnan(ref[i, j])):
                cnt += 1
                sm += model[i, j]
                sr += ref[i, j]
        if cnt == 0:
            continue
        mm = sm / cnt
        mr = sr / cnt
        vm = 0.0
        vr = 0.0
        cov = 0.0
        vd = 0.0
        for j in range(m):
            if not (np.isnan(model[i, j]) or np.isnan(ref[i, j])):
                dm = model[i, j] - mm
                dr = ref[i, j] - mr
                vm += dm * dm
                vr += dr * dr
                cov += dm * dr
                vd += (dm - dr) * (dm - dr)
        std[i] = np.sqrt(vm / cnt)
        std_ref[i] = np.sqrt(vr / cnt)
        crmsd[i] = np.sqrt(vd / cnt)
        if vm > 0 and vr > 0:
            corr[i] = cov / np.sqrt(vm * vr)
    return std, std_ref, corr, crmsd


def taylor_stats(model, ref):
    """Statistics that are displayed in a Taylor diagram

    Only the samples that are valid in both arrays are used.
    The standard deviations are the biased ones (``ddof=0``), so that the
    centered root mean square difference ``crmsd`` verifies
    ``crmsd**2 = std**2 + std_ref**2 - 2 * std * std_ref * corr``.

    Parameters
    ----------
    model: array_like(n, m)
        ``n`` series of ``m`` samples. A 1D array is a single series.
    ref: array_like(m) or array_like(n, m)
        Reference series, that are shared by the ``n`` series of the model
        when it is 1D.

    Return
    ------
    numpy.ndarray(n)
        Standard deviation of the model
    numpy.ndarray(n)
        Standard deviation of the reference
    numpy.ndarray(n)
        Correlation, which is nan when a series is constant
    numpy.ndarray(n)
        Centered root mean square difference
    """
    model = np.atleast_2d(np.asarray(model, dtype="d"))
    ref = np.asarray(ref, dtype="d")
    ref = np.ascontiguousarray(np.broadcast_to(ref, model.shape))
    return _taylor_stats_(np.ascontiguousarray(model), ref)
