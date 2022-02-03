# -*- coding: utf-8 -*-
"""
Plotting utilities
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

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import xarray as xr

from . import misc as xmisc
from . import geo as xgeo
from . import cf as xcf
from . import coords as xcoords
from . import dyn


def plot_flow(
    u,
    v,
    duration=None,
    step=None,
    particles=2000,
    axes=None,
    alpha=(0.2, 1),
    linewidth=0.3,
    color="k",
    autolim=None,
    **kwargs,
):
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
    Return
    ------
    dict
        With the following keys:
        `axes`, `linecollection`, `step`, `duration`.
        `linecollection` refere to the :class:`matplotlib.collections.LineCollection` instance.
        The duration and step are also stored in the output.

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
        @savefig api.plot.plot_flow.png
        plot_flow(ds["u"], ds["v"])

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
            speed = np.sqrt(np.abs(u.values).mean() ** 2 + np.abs(v.values).mean() ** 2)
            duration = dist / speed / 30
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
    for j in range(tx.shape[0] - 1):
        for i in range(tx.shape[1]):
            segments.append(((tx[j, i], ty[j, i]), (tx[j + 1, i], ty[j + 1, i])))
            if not np.isscalar(alpha):
                colors.append(color + (alpha[0] + cdists[j, i] * (alpha[-1] - alpha[0]),))
            else:
                colors.append(color + (alpha,))
            if not np.isscalar(linewidth):
                linewidths.append(
                    linewidth[0] + cdists[j, i] * (linewidth[-1] - linewidth[0]),
                )
            else:
                linewidths.append(linewidth)

    # Plot
    kwargs.setdefault("colors", colors)
    kwargs.setdefault("linewidths", linewidths)
    lc = mcollections.LineCollection(segments, **kwargs)
    newaxes = axes is None
    if newaxes:
        axes = plt.gca()
    if autolim is None:
        autolim = newaxes
    axes.add_collection(lc, autolim=autolim)

    return {"axes": axes, "step": step, "duration": duration, "linecollection": lc}


def plot_ts(
    temp,
    sal,
    dens=True,
    potential=None,
    axes=None,
    scatter_kwargs=None,
    contour_kwargs=None,
    clabel=True,
    clabel_kwargs=None,
    colorbar=None,
    colorbar_kwargs=None,
    **kwargs,
):
    """Plot a TS diagram

    A TS diagram is a scatter plot with absolute salinity as X axis
    and potential temperature as Y axis.
    The density is generally added as background contours.

    Parameters
    ----------
    temp: xarray.DataArray
        Temperature. If not potential temperature, please set `potential=True`.
    sal: xarray.DataArray
        Salinity
    dens: bool
        Add contours of density.
        The density is computed with function :func:`gsw.density.sigma0`.
    potential: bool, None
        Is the temperature potential? If None, infer from attributes.
    clabel: bool
        Add labels to density contours
    clabel_kwargs: dict, None
        Parameters that are passed to :func:`~matplotlib.pyplot.clabel`.
    colorbar: bool, None
        Should we add the colorbar? If None, check if scatter plot color is a data array.
    colorbar_kwargs: dict, None
        Parameters that are passed to :func:`~matplotlib.pyplot.colorbar`.
    contour_kwargs: dict, None
        Parameters that are passed to :func:`~matplotlib.pyplot.contour`.
    axes: None
        Matplotlib axes instance
    kwargs: dict
        Extra parameters are filtered by :func:`xoa.misc.dict_filter`
        and passed to the plot functions.

    See also
    --------
    :mod:`gsw.density`
    :mod:`gsw.conversions`

    Return
    ------
    dict
        With the following keys, depending on what is plotted:
        `axes`, `scatter`, `colorbar`, `contour`, `clabel`.


    Example
    -------
    .. ipython:: python

        @suppress
        import numpy as np, xarray as xr, xoa, xoa.coords, cmocean
        @suppress
        from xoa.plot import plot_ts

        # Register the main xoa accessor
        xoa.register_accessors()

        # Load the CROCO meridional section
        ds = xoa.open_data_sample("croco.south-africa.meridional.nc")
        ds = ds.isel(eta_rho=slice(40))
        temp = ds.xoa.get('temp')     # requests are based...
        sal = ds.xoa.get('sal')       # ...on the generic name
        depth = ds.xoa.get_depth(ds)  # or xoa.coords.get_depth(ds)

        # Plot
        @savefig api.plot.plot_ts.png
        plot_ts(temp, sal, potential=True, scatter_c=depth, contour_linewidths=0.2, clabel_fontsize=8)

    """

    # Potential temperature
    cfspecs = xcf.get_cf_specs(temp)
    # potential = POTENTIAL[potential]
    if potential is None:
        potential = cfspecs.match_data_var(temp, "ptemp")
    if not potential:
        import gsw

        lat = xcoords.get_lat(temp)
        depth = xcoords.get_depth(temp)
        lat, depth = xr.broadcast(lat, depth)
        pres = gsw.p_from_z(depth, lat)
        temp = gsw.pt_from_t(sal, temp, pres)

    # Init plot
    if axes is None:
        axes = plt.gca()
    out = {"axes": axes}

    # Scatter plot
    scatter_kwargs = xmisc.dict_filter(
        kwargs, "scatter_", defaults={"s": 10}, **(scatter_kwargs or {})
    )
    out["scatter"] = axes.scatter(sal.values, temp.values, **scatter_kwargs)

    # Colorbar
    if colorbar is None:
        colorbar = "c" in scatter_kwargs and hasattr(scatter_kwargs["c"], "data")
    if colorbar:
        colorbar_kwargs = xmisc.dict_filter(
            kwargs, "colorbar_", defaults={}, **(colorbar_kwargs or {})
        )
        c = scatter_kwargs["c"]
        if "label" not in colorbar_kwargs and hasattr(c, "attrs"):
            label = c.attrs.get("long_name") or c.name
            if label:
                label = label.title()
                units = c.attrs.get("units")
                if units:
                    label = f"{label} [{units}]"
                colorbar_kwargs["label"] = label
        out["colorbar"] = plt.colorbar(out["scatter"], ax=axes, **colorbar_kwargs)

    # Labels
    axes.set_xlabel(sal.attrs.get("long_name", "Salinity").title())
    tlabel = temp.attrs.get("long_name", "Temperature").title()
    tunits = temp.attrs.get("units", "°C")
    axes.set_ylabel(f"{tlabel} [{tunits}]")

    # Density contours
    if dens is not False:

        import gsw

        # Density as sigma0
        (smin, tmin), (smax, tmax) = axes.viewLim.get_points()
        tt = np.linspace(tmin, tmax, 100)
        ss = np.linspace(smin, smax, 100)
        ss, tt = np.meshgrid(ss, tt)
        dd = gsw.sigma0(ss, tt)

        # Contours
        contour_kwargs = xmisc.dict_filter(
            kwargs, "contour_", defaults={"colors": ".3"}, **(contour_kwargs or {})
        )
        out["contour"] = axes.contour(ss, tt, dd, **contour_kwargs)

        # Contour labels
        clabel_kwargs = xmisc.dict_filter(kwargs, "clabel_", defaults={}, **(clabel_kwargs or {}))
        out["clabel"] = axes.clabel(out["contour"], **clabel_kwargs)

    return out
