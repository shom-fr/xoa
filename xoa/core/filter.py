"""
Filtering numeric kernels

Numba accelerated routines for the decimation of scattered points.
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

from . import geo as core_geo


@numba.njit(cache=True)
def get_decimate_arg(lons, lats, radius):
    """Get which point to keep to make sure they are enough distant of one another

    A loop is made on all points: a point that is at a distance less that `radius`
    from a point that was previously marked as True is marked as False.

    Parameters
    ----------
    lons: numpy.ndarray
        1D array of longitudes
    lats: numpy.ndarray
        1D array of latitudes
    radius: float
        Radius relative to the sphere radius

    Returns
    -------
    numpy.array
        1D array of boolean with True set for the points to keep

    See also
    --------
    xoa.core.geo.haversine
    xoa.filter.decimate
    """
    npts = lons.size
    keep = np.ones(npts, dtype="?")
    for i in range(1, npts):
        for j in range(i):
            if keep[j]:
                dist = core_geo.haversine(lons[i], lats[i], lons[j], lats[j])
                if dist < radius:
                    keep[i] = False
                    break
    return keep


@numba.njit(cache=True)
def decimate_by_average(lons, lats, radius, keep, data):
    """Average data at kept points within a radius"""
    npts = lons.size
    nkept = keep.sum()
    cdata = np.zeros(data.shape[:-1] + (nkept,), data.dtype)
    ccount = np.zeros(nkept, data.dtype)
    k = 0
    for i in numba.prange(npts):
        if keep[i]:
            for j in range(npts):
                dist = core_geo.haversine(lons[i], lats[i], lons[j], lats[j])
                if dist <= radius:
                    cdata[..., k] += data[..., j]
                    ccount[k] += 1.0
            k += 1
    for i in numba.prange(nkept):
        cdata[..., i] = cdata[..., i] / ccount[i]
    return cdata
