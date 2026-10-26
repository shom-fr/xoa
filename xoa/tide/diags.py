"""
Tidal diagnostics: form number, high waters, skew surge, tidal range and atlas
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

from .. import exceptions
from ..core import signal as signal_core
from ..core import tide as tide_core
from .data import get_time_dim

# %% diagnostics


def _apply_kernel_(func, dtype, template, *others, time_dim=None, **kwargs):
    # Apply a core kernel working on time-first (nt, ...) arrays to data arrays.
    # The axes are never moved, so time_dim must be the first dimension of all
    # inputs. Dask arrays are processed block by block (time is one chunk).
    time_dim = get_time_dim(template, time_dim)
    for x in (template, *others):
        if x.dims[0] != time_dim:
            raise exceptions.XoaError(f"'{time_dim}' must be the first dimension, got {x.dims}")
    arrays = [template] + [x.broadcast_like(template).transpose(*template.dims) for x in others]
    if any(x.chunks is not None for x in arrays):
        import dask.array

        # one chunk along time, the same spatial chunks for all the inputs
        data = [x.data for x in xr.unify_chunks(*(x.chunk({time_dim: -1}) for x in arrays))]
        out = dask.array.map_blocks(lambda *b: func(*b, **kwargs), *data, dtype=dtype)
    else:
        out = func(*(x.values for x in arrays), **kwargs)
    return xr.DataArray(out, dims=template.dims, coords=template.coords)


def form_number(coef):
    """Tidal form number ``F = (K1 + O1) / (M2 + S2)``

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :meth:`TidalHarmonic.analyse` that contains K1, O1, M2 and S2.

    Return
    ------
    xarray.DataArray
        F < 0.25 semi-diurnal, 0.25-1.5 mixed mainly semi-diurnal,
        1.5-3 mixed mainly diurnal, > 3 diurnal.
    """
    amp = coef["amp"]
    k1, o1, m2, s2 = (amp.sel(constituent=n, drop=True) for n in ("K1", "O1", "M2", "S2"))
    return ((k1 + o1) / (m2 + s2)).rename("form_number")


def high_waters(h, min_separation=9, time_dim=None):
    """High waters of a regularly sampled series, at every point

    Parameters
    ----------
    h: xarray.DataArray
        Sea level, typically the predicted tide, with ``time_dim`` first.
    min_separation: int
        Minimal number of samples between two high waters (9 for hourly
        data and a semi-diurnal tide).
    time_dim: str, None
        Searched with :func:`xoa.coords.get_tdim` when None.

    Return
    ------
    xarray.DataArray
        Boolean array with the dims of ``h``, True at the high waters.
        For a single series, ``h[time_dim][mask]`` gives their times.
    """
    mask = _apply_kernel_(
        tide_core.high_water_mask, bool, h, time_dim=time_dim, min_separation=min_separation
    )
    return mask.rename("high_water")


def skew_surge(obs, pred, hw=None, half_window=3, time_dim=None):
    """Skew surge at each predicted high water, at every point

    It is the difference between the highest observed level around a predicted
    high water and the predicted high water level, whatever the timing offset.

    Parameters
    ----------
    obs, pred: xarray.DataArray
        Observed and predicted sea level on the same regular time axis.
        ``obs`` may contain NaNs.
    hw: xarray.DataArray, None
        High water mask, see :func:`high_waters`. It defaults to that of ``pred``.
    half_window: int
        Half width in samples of the window searched for the observed maximum.
    time_dim: str, None
        Searched with :func:`xoa.coords.get_tdim` when None.

    Return
    ------
    xarray.DataArray
        Skew surge at the predicted high waters whose window is complete and
        holds data, NaN elsewhere.
    """
    if hw is None:
        hw = high_waters(pred, 2 * half_window + 3, time_dim=time_dim)
    surge = _apply_kernel_(
        tide_core.skew_surge_series,
        float,
        obs,
        pred,
        hw,
        time_dim=time_dim,
        half_window=half_window,
    )
    return surge.rename("skew_surge")


def spring_high_waters(h, hw=None, quantile=0.75, half_window=6, time_dim=None):
    """High waters with the largest tidal range, i.e. spring tides

    The range of a high water is its level minus the lowest level within
    ``half_window`` samples on each side.

    Parameters
    ----------
    h: xarray.DataArray
        Sea level, typically the predicted tide.
    hw: xarray.DataArray, None
        High water mask, see :func:`high_waters`. It defaults to that of ``h``.
    quantile: float
        Keep the high waters whose range is above this quantile.
    half_window: int
    time_dim: str, None
        Searched with :func:`xoa.coords.get_tdim` when None.

    Return
    ------
    xarray.DataArray
        Boolean array with the dims of ``h``. High waters without a full window are dropped.
    """
    if hw is None:
        hw = high_waters(h, time_dim=time_dim)
    mask = _apply_kernel_(
        tide_core.spring_mask,
        bool,
        h,
        hw,
        time_dim=time_dim,
        quantile=quantile,
        half_window=half_window,
    )
    return mask.rename("spring_high_water")


def tidal_range(h, window=13, time_dim=None):
    """Tidal range as the max-min of a centred rolling window

    Parameters
    ----------
    h: xarray.DataArray
        Array with ``time_dim`` first.
    window: int
        Window in samples, a bit more than one semi-diurnal cycle.
    time_dim: str, None
        Searched with :func:`xoa.coords.get_tdim` when None.

    Return
    ------
    xarray.DataArray
    """
    return _apply_kernel_(
        signal_core.rolling_range, float, h, time_dim=time_dim, window=window
    ).rename("tidal_range")


def tidal_atlas(da, hw, offsets=range(-6, 7), time_dim=None):
    """Composite maps at fixed offsets from high water

    Parameters
    ----------
    da: xarray.DataArray
        Field to composite, like a current component, with a time dimension.
    hw: xarray.DataArray
        Boolean mask of the selected high waters of a reference series,
        see :func:`spring_high_waters`.
    offsets: iterable of int
        Offsets in samples with respect to the high water.
    time_dim: str, None
        Searched with :func:`xoa.coords.get_tdim` when None.

    Return
    ------
    xarray.DataArray
        Mean of ``da`` over the selected high waters, for each offset,
        along the new ``offset`` dimension.
    """
    hw = np.flatnonzero(np.asarray(hw))
    offsets = np.asarray(list(offsets))
    time_dim = get_time_dim(da, time_dim)
    maps = [da.isel({time_dim: hw + k}).mean(time_dim) for k in offsets]
    return xr.concat(maps, dim="offset").assign_coords(offset=offsets)
