"""
Routines related to the ocean dynamics.
"""
# Copyright 2020-2021 Shom
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
import pandas as pd
import xarray as xr

from .__init__ import XoaError
from . import coords as xcoords
from . import grid as xgrid
from xoa.interp import grid2locs
from xoa.geo import EARTH_RADIUS


def _get_uv2d_(t, txy, gx, gy, gz, gt, guv):
    """Interpolate gridded u and v on track positions"""
    # Shape
    npts = int(txy.size / 2)
    tx = txy[:npts]
    ty = txy[npts:]
    tt = np.full(npts, t)
    tz = np.zeros(npts)

    # Interpolate
    tuv = grid2locs(gx, gy, gz, gt, guv, tx, ty, tz, tt)

    # Scale the speed for degrees
    tuv[0] *= 180. / (np.pi * EARTH_RADIUS)
    tuv[1] *= 180. / (np.pi * EARTH_RADIUS)
    tuv[1] *= np.cos(ty*np.pi/180)

    # Pack velocity
    return tuv.ravel()


def _rk4_(xy, f, t, dt, **kwargs):
    """Integrate one time step with RK4"""
    k1 = dt * f(t, xy, **kwargs)
    k2 = dt * f(t + 0.5*dt, xy + 0.5*k1, **kwargs)
    k3 = dt * f(t + 0.5*dt, xy + 0.5*k2, **kwargs)
    k4 = dt * f(t + 0.5*dt, xy + 0.5*k3, **kwargs)
    return xy + (k1 + k2 + k3 + k4)/6.


def _integrate_(xy, f, t0, t1, dt, **kwargs):
    """Low level integration with packed, pure-numeric data"""

    # Fit the time steps to the time range
    dt *= 1.
    if t1 < 0:
        dts = [dt] * int(-t1)
    else:
        nt = int((t1-t0)//dt)
        dts = [dt] * nt
        dtf = (t1-t0) % dt
        if dtf:
            dts.append(dtf)

    # Interative integration
    t = t0
    tt = [t0]
    xxyy = [xy]
    for dt in dts:
        xy2 = _rk4_(xy, f, t, dt, **kwargs)
        t2 = t + dt
        tt.append(t2)
        xxyy.append(xy2)
        t = t2
        xy = xy2
    return np.array(tt), np.asarray(xxyy)


def flow2d(u, v, xy0, duration, step, date=None):
    """Integrate gridded 2D velocities from random positions

    Parameters
    ----------
    u: xarray.DataArray
        Gridded zonal velocity
    v: xarray.DataArray
        Gridded meridional velocity
    duration: int, numpy.timedelta64
        Total integration time in seconds
    step: int, numpy.timedelta64
        Integratiin step in seconds
    xy0: int, xarray.Dataset, tuple
        Either a number of particles or a dataset of initial positions
        with longitude and latitude coordinates
    date: None, numpy.datetime64
        A reference date for the time integration

    Return
    ------
    xarray.Dataset
        Output positions with ``lon`` and ``lat`` coordinates that vary with time.

    """
    # Gridded field
    time0 = xcoords.get_time(u, errors="ignore")
    u = u.squeeze(drop=True)
    v = v.squeeze(drop=True)
    if u.ndim != 2 or v.ndim != 2:
        raise XoaError("The velocity field must 2D")
    u = xgrid.to_rect(u)
    gx = xcoords.get_lon(u).values
    gy = xcoords.get_lat(u).values
    if gx.ndim == 1:
        gx = gx.reshape(1, -1)
    if gy.ndim == 1:
        gy = gy.reshape(-1, 1)
    gz = np.zeros((1,)*5)
    gt = np.zeros(1)
    gu = u.values.reshape((1,)*2+u.shape[-2:])
    gv = v.values.reshape(gu.shape)
    guv = np.array([gu, gv])

    # Initial positions
    tt0 = None
    if isinstance(xy0, xr.Dataset):
        tx0 = xcoords.get_lon(xy0).values.ravel()
        ty0 = xcoords.get_lon(xy0).values.ravel()
        tt0 = xcoords.get_time(xy0, errors='ignore')
    elif isinstance(xy0, int):
        tx0 = np.random.uniform(gx.min(), gx.max(), xy0)
        ty0 = np.random.uniform(gy.min(), gy.max(), xy0)
    else:
        tx0, ty0 = xy0
        tx0 = np.asarray(tx0)
        ty0 = np.asarray(ty0)
    txy0 = np.concatenate((tx0, ty0))

    # Integration
    t0 = 0
    if isinstance(duration, np.timedelta64):
        duration /= np.timedelta64(1, "s")
    if isinstance(step, np.timedelta64):
        step /= np.timedelta64(1, "s")
    t1 = duration
    tt, txy = _integrate_(
        txy0,
        _get_uv2d_,
        t0, t1, step,
        gx=gx, gy=gy, gz=gz, gt=gt,
        guv=guv)
    tx, ty = np.split(txy, 2, axis=1)

    # As a dataset
    if date is not None:
        date = pd.to_datetime(date).to_datetime64()
    elif tt0 is not None and len(tt0) <= 1:
        date = tt0.values
    elif time0 is not None:
        date = time0.values
    else:
        date = pd.Timestamp.now().to_datetime64()
    time = xr.DataArray(tt*np.timedelta64(1, 's')+date, dims='time')

    return xr.Dataset(
        coords={'lon': (('time', 'particles'), tx),
                'lat': (('time', 'particles'), ty),
                'time': time})
