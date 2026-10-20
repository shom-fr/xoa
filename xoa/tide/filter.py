"""
Tidal filters
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

import numpy as np

from .. import coords as xcoords
from .. import exceptions
from .. import filter as xfilter

# %% tidal filters

#: Weights of the filters for hourly data
HOURLY_TIDAL_FILTERS_WEIGHTS = {
    "demerliac": [
        1,
        3,
        8,
        15,
        21,
        32,
        45,
        55,
        72,
        91,
        105,
        128,
        153,
        171,
        200,
        231,
        253,
        288,
        325,
        351,
        392,
        435,
        465,
        512,
        558,
        586,
        624,
        658,
        678,
        704,
        726,
        738,
        752,
        762,
        766,
        768,
        766,
        762,
        752,
        738,
        726,
        704,
        678,
        658,
        624,
        586,
        558,
        512,
        465,
        435,
        392,
        351,
        325,
        288,
        253,
        231,
        200,
        171,
        153,
        128,
        105,
        91,
        72,
        55,
        45,
        32,
        21,
        15,
        8,
        3,
        1,
    ],
    "godin": [
        1,
        3,
        6,
        10,
        15,
        21,
        28,
        36,
        45,
        55,
        66,
        78,
        91,
        105,
        120,
        136,
        153,
        171,
        190,
        210,
        231,
        253,
        276,
        300,
        323,
        344,
        363,
        380,
        395,
        408,
        419,
        428,
        435,
        440,
        443,
        444,
        443,
        440,
        435,
        428,
        419,
        408,
        395,
        380,
        363,
        344,
        323,
        300,
        276,
        253,
        231,
        210,
        190,
        171,
        153,
        136,
        120,
        105,
        91,
        78,
        66,
        55,
        45,
        36,
        28,
        21,
        15,
        10,
        6,
        3,
        1,
    ],
    "mean": [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ],
}


def demerliac(da, na_thres=0, dt_tol=0.01):
    """Apply a Demerliac filter on a data array

    This is a shortcut to ``tidal_filter(da, "demerliac", ...)``.

    Parameters
    ----------
    da: xarray.DataArray
    dt_tol: float
        Relative tolerance for the time step variability
    na_thres: float
        A float between 0 and 1 that defines the allowed level a NaN contamination.
        See :func:`xoa.filter.convolve`.

    Return
    ------
    xarray.DataArray
    """
    return tidal_filter(da, "demerliac", na_thres=na_thres, dt_tol=dt_tol)


def tidal_filter(da, filter_name, na_thres=0, dt_tol=0.01):
    """Apply a tidal filter on a data array

    The data array must have a valid time dimension.
    When the time step is less than an hour, the filter weights are interpolated
    since they are designed for hourly time series.
    An error is raised when the time step is too variable or above one hour.

    Parameters
    ----------
    da: xarray.DataArray
        Data array with a time dimension.
    filter_name: str
        Type of tidal filter: ``"demerliac"``, ``"godin"`` or ``"mean"``.
    dt_tol: float
        Relative tolerance for the time step variability.
    na_thres: float
        A float between 0 and 1 that defines the allowed level of NaN contamination.
        See :func:`xoa.filter.convolve`.

    Return
    ------
    xarray.DataArray
        Filtered data array with the tidal signal removed.

    See also
    --------
    demerliac
    xoa.filter.convolve
    """
    # Ensure nanosecond precision for datetime coordinates
    da = xcoords.ensure_ns_datetime(da)

    # Get time dimension
    tdim = xcoords.get_tdim(da, errors="ignore")
    if tdim is None:
        exceptions.xoa_warn("Cannot apply the Demerliac filter since no time dimension found")
        return da.copy()

    # Weights
    if filter_name not in HOURLY_TIDAL_FILTERS_WEIGHTS:
        raise exceptions.XoaError(
            f"Invalid filter name '{filter_name}'. Please use one of: "
            + ', '.join(HOURLY_TIDAL_FILTERS_WEIGHTS)
        )
    weights = np.array(HOURLY_TIDAL_FILTERS_WEIGHTS[filter_name], "d")
    if tdim not in da.indexes:
        exceptions.xoa_warn("Not time coordinate found so we assume hourly data")
    else:
        dt = np.diff(da[tdim].values) / np.timedelta64(3600, "s")
        ddt = float(np.ptp(dt))
        mdt = float(dt.mean())
        if ddt > dt_tol * mdt:
            raise exceptions.XoaError(
                "The variability of your time steps is above the allowed level "
                f" to apply a {filter_name} filter"
            )
        if mdt > 1 + dt_tol:
            raise exceptions.XoaError(
                f"You should not apply a {filter_name} filter to data that are less "
                f"than hourly sampled. Current mean time step: {mdt:1.2f} h"
            )
        elif mdt < 1 - dt_tol:
            nw = len(weights)
            from scipy.interpolate import interp1d

            weights = interp1d(np.linspace(0, 1, nw), weights, "cubic")(
                np.linspace(0, 1, int(round((nw - 1) / mdt)) + 1)
            )

    # Apply
    return xfilter.convolve(da, {tdim: weights}, normalize=True, na_thres=na_thres)
