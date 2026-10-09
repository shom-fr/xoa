"""
Time utilities for core interpolation
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

import numba as nb
import numpy as np

NOT_CI = os.environ.get("CI", "false") == "false"

# %% Fractional time indices


@nb.njit(cache=NOT_CI, parallel=True)
def compute_time_frac_indices(src_times, dst_times):
    """
    For each destination time find the bracketing source interval.

    Parameters
    ----------
    src_times : (nt_src,) float64
        Source time coordinates, monotonically increasing.
    dst_times : (n_dst,) float64
        Destination time coordinates, same units as src_times.

    Return
    ------
    it_base : (n_dst,) int64
        Index of the lower bracketing source step; -1 if out-of-range or NaN.
    frac_t : (n_dst,) float64
        Weight toward it_base+1 in [0, 1].
    """
    nt = src_times.shape[0]
    n_dst = dst_times.shape[0]
    it_base = np.full(n_dst, -1, dtype=np.int64)
    frac_t = np.zeros(n_dst, dtype=np.float64)

    t_min = src_times[0]
    t_max = src_times[nt - 1]

    for k in nb.prange(n_dst):
        t = dst_times[k]
        if np.isnan(t) or t < t_min or t > t_max:
            continue
        for i in range(nt - 1):
            if src_times[i] <= t <= src_times[i + 1]:
                dt = src_times[i + 1] - src_times[i]
                it_base[k] = i
                frac_t[k] = 0.0 if dt == 0.0 else (t - src_times[i]) / dt
                break

    return it_base, frac_t
