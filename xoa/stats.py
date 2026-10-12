"""
Statistics of xarray objects

See :func:`rms` for root mean square errors.
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
import xarray as xr

from . import exceptions


def rms(da, ref=None, mask=None, dim=None, by=None):
    """Compute a root mean square, or a root mean square error

    Parameters
    ----------
    da: xarray.DataArray
        Estimate, or the error when `ref` is not provided.
    ref: None, xarray.DataArray
        Reference, which is subtracted from `da`.
    mask: None, xarray.DataArray
        Boolean array that tells where to compute the statistics.
        Missing values are always discarded.
    dim: None, str, list(str)
        Dimensions to reduce, which default to all of them.
    by: None, dict
        Regions as a dict that maps names to boolean arrays. The root mean square is then
        computed in each of them, in addition to `mask`, and the result has
        an extra ``region`` dimension.

    Return
    ------
    xarray.DataArray

    Example
    -------
    .. code-block:: python

        rms(rossby_z, ref=rossby_ref, mask=valid)
        rms(speed_error, by={"Everywhere": valid, "Cores": valid & core})
    """
    sq = (da if ref is None else da - ref) ** 2
    if mask is not None:
        sq = sq.where(mask)
    if by is None:
        return np.sqrt(sq.mean(dim)).rename("rms")
    if not len(by):
        raise exceptions.XoaError("by must not be empty")
    res = [np.sqrt(sq.where(region).mean(dim)) for region in by.values()]
    region = xr.DataArray(list(by), dims="region", name="region")
    return xr.concat(res, dim=region).rename("rms")
