# -*- coding: utf-8 -*-
"""
Plotting utilities
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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections

from . import geo as xgeo
from . import dyn


def plot_flow(
        u, v, duration=None, step=None, particles=2000, axes=None,
        alpha=(.2, 1), linewidth=.3, color="k", autolim=None, **kwargs):
    """Plot currents as a windy-like plot with random little lagrangian tracks

    Parameters
    ----------
    u: xarray.DataArray
        Gridded zonal velocity
    v: xarray.DataArray
        Gridded meridional velocity
    duration: int, numpy.timedelta64
        Total integration time in seconds
    step: int, numpy.timedelta64
        Integration step in seconds
    particles: int, xarray.Dataset, tuple
        Either a number of particles or a dataset of initial positions
        with longitude and latitude coordinates
    axes: matplotlib.axes.Axes
        The axes instance
    alpha: float, tuple
        Alpha transparency. If a tuple, apply a linear alpha ramp to the track
        from its start to its end.
    linewidth: float, tuple
        Linewidth of the track. If a tuple, apply a linear linewidth ramp to the track
        from its start to its end.
    color:
        Single color for the track.
    autolim: None, bool
        Wether to auto-update the data limits.
        A value of None sets autolim to True if "axes" is not provided,
        else to None. See :meth:`matplotlib.axes.Axes.add_collection`.

    Return
    ------
    matplotlib.collections.LineCollection
        The duration and step are set as attributes with values in seconds.

    Example
    ------

    .. ipython:: python

        @suppress
        import numpy as np, xarray as xr
        @suppress
        from xoa.plot import plot_flow

        # Setup data
        x = np.linspace(0, 2*np.pi, 20)
        y = np.linspace(0, 2*np.pi, 20)
        X, Y = np.meshgrid(x,y)
        U = np.sin(X) * np.cos(Y)
        V = -np.cos(X) * np.sin(Y)

        # As a dataset
        ds = xr.Dataset(
            {"u": (('lat', 'lon'), U), "v": (('lat', 'lon'), V)},
            coords={"lat": ("lat", y), "lon": ("lon", x)})

        # Plot
        lc = plot_flow(ds["u"], ds["v"])
        @savefig api.plot.plot_flow.png
        lc.duration, lc.step

    See also
    --------
    xoa.dyn.flow2d
    matplotlib.axes.Axes.add_collection
    """
    # Infer parameters
    if duration is None or step is None:

        # Default duration based on a the fraction of the area crossing time
        if duration is None:
            xmin, xmax, ymin, ymax = xgeo.get_extent(u)
            dist = xgeo.haversine(xmin, ymin, xmax, ymax)
            speed = np.sqrt(np.abs(u.values).mean()**2+np.abs(v.values).mean()**2)
            duration = dist/speed/30
        elif isinstance(duration, np.timedelta64):
            duration /= np.timedelta64(1, "s")

        # Step
        if step is None:
            step = duration / 10
    if isinstance(duration, np.timedelta64):
        duration /= np.timedelta64(1, "s")
    if isinstance(step, np.timedelta64):
        step /= np.timedelta64(1, "s")

    # Compute the flow
    flow = dyn.flow2d(u, v, particles, duration, step)
    tx = flow.lon.values
    ty = flow.lat.values

    # Get the distance from start for each track
    ramp = not np.isscalar(alpha) or not np.isscalar(linewidth)
    if ramp:
        dists = xgeo.haversine(tx[:-1], ty[:-1], tx[1:], ty[1:])
        cdists = np.cumsum(dists, axis=0)
        cdists /= np.nanmax(cdists)
        cdists[np.isnan(cdists)] = 0

    # Plot specs
    color = mcolors.to_rgb(color)
    segments = []
    linewidths = []
    colors = []
    for j in range(tx.shape[0]-1):
        for i in range(tx.shape[1]):
            segments.append(((tx[j, i], ty[j, i]), (tx[j+1, i], ty[j+1, i])))
            if not np.isscalar(alpha):
                colors.append(color+(alpha[0]+cdists[j, i]*(alpha[-1]-alpha[0]),))
            else:
                colors.append(color+(alpha,))
            if not np.isscalar(linewidth):
                linewidths.append(linewidth[0]+cdists[j, i]*(linewidth[-1]-linewidth[0]),)
            else:
                linewidths.append(linewidth)

    # Plot
    kwargs.setdefault("colors", colors)
    kwargs.setdefault("linewidths", linewidths)
    lc = mcollections.LineCollection(segments, **kwargs)
    lc.step = step
    lc.duration = duration
    newaxes = axes is None
    if newaxes:
        axes = plt.gca()
    if autolim is None:
        autolim = newaxes
    axes.add_collection(lc, autolim=autolim)

    return lc



