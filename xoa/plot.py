# -*- coding: utf-8 -*-
"""
Plotting utilities

Filters are adapted from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html?highlight=agg%20filter
and http://vacumm.github.io/vacumm/library/misc.core_plot.html

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
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.collections as mcollections
import matplotlib.artist as martist
import matplotlib.text as mtext
import matplotlib.patheffects as mpatheffects
import xarray as xr

from . import exceptions
from . import misc as xmisc
from . import geo as xgeo
from . import meta as xmeta
from . import coords as xcoords
from . import dyn
from .core import grid as cgrid
from .core import plot as cplot
from .core import stats as cstats
from .core.plot import (  # noqa
    add_land,
    create_base_map,
    setup_map_axes,
    _import_cartopy_,
)

_AX_SETUP_KEYS = cplot.AX_SETUP_KEYS

# %% Special functions


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
        Whether to auto-update the data limits.
        A value of None sets autolim to True if "axes" is not provided,
        else to None. See :meth:`matplotlib.axes.Axes.add_collection`.

    Return
    ------
    dict
        With the following keys:
        `axes`, `linecollection`, `step`, `duration`.
        `linecollection` refers to the :class:`matplotlib.collections.LineCollection` instance.
        The duration and step are also stored in the output.

    Example
    ------
    .. ipython:: python
        :okwarning:

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

        # Default duration based on a fraction of the area crossing time
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
    axes = kwargs.get("ax", axes)
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
    ref_dens=0,
    pres=None,
    potential=None,
    absolute=None,
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

    A TS diagram is a scatter plot with salinity (practical or absolute) as X axis
    and potential temperature as Y axis.
    The density is generally added as background contours.

    Parameters
    ----------
    temp: xarray.DataArray
        Temperature. If not potential, it will be converted into potential
        if `potential=None` or `potential=False`.
        Note that if temp is not potential and **contains a depth coordinate**, depth values must be negative
        (to compute pres with `gsw.p_from_z` if necessary)
    sal: xarray.DataArray
        Salinity (practical or absolute). If not absolute, it will be converted into absolute salinity
        if the potential temperature needs to be computed.
    dens: bool
        Add contours of density.
        The density is by default computed with function :func:`gsw.density.sigma0` (ref_dens=0).
    ref_dens: integer, 0
        choice of reference for density calculation (between 0 and 4)
        ref_dens=0 will consider func:`gsw.density.sigma0`, etc..
    pres: xarray.DataArray, None
        Pressure to compute potential temperature and absolute salinity.
    potential: bool, None
        Is the temperature potential? If None, infer from attributes.
    absolute: bool, None
        Is the salinity absolute? If None, infer from attributes.
    clabel: bool
        Add labels to density contours
    clabel_kwargs: dict, None
        Parameters that are passed to :func:`~matplotlib.pyplot.clabel`.
    colorbar: bool, None
        Should we add the colorbar? If None, check if scatter plot color is a data array.
    colorbar_kwargs: dict, None
        Parameters that are passed to :func:`add_colorbar`.
        The colorbar is shrunk and labelled from the scatter color array by default.
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

        # Load the Mercator data
        file_name = xoa.get_data_sample("MODELS/CMEMS-IBI/ibi-argo-7900573.nc")
        ds = xr.open_dataset(file_name)
        temp = ds.thetao
        sal = ds.so
        depth = ds.depth.broadcast_like(temp)

        # Plot
        @savefig api.plot.plot_ts.png
        plot_ts(temp, sal, potential=True, scatter_c=depth, contour_linewidths=0.2, clabel_fontsize=8, cmap="cmo.deep_r")

    """

    # Potential temperature
    metaspecs = xmeta.get_meta_specs(temp)
    # potential = POTENTIAL[potential]
    if potential is None:
        potential = metaspecs.match_data_var(temp, "ptemp")

    if not potential:
        import gsw

        if pres is None:
            lat = xcoords.get_lat(temp)
            depth = xcoords.get_depth(temp)
            lat, depth = xr.broadcast(lat, depth)
            pres = gsw.p_from_z(depth, lat)

        if absolute is None:
            absolute = metaspecs.match_data_var(sal, "asal")
        if not absolute:
            lon = xcoords.get_lon(temp)
            lat = xcoords.get_lat(temp)
            sal_abs = gsw.SA_from_SP(sal, pres, lon, lat)
        else:
            sal_abs = sal.copy()

        attrs = temp.attrs
        temp = gsw.pt0_from_t(sal_abs, temp, pres)
        temp.attrs.update(attrs)
        metaspecs.format_data_var(temp, meta_name="ptemp", copy=False, replace_attrs=True)

    # Init plot
    axes = kwargs.get("ax", axes)
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
        colorbar = isinstance(scatter_kwargs.get("c"), xr.DataArray)
    if colorbar:
        colorbar_kwargs = xmisc.dict_filter(
            kwargs, "colorbar_", defaults={}, **(colorbar_kwargs or {})
        )
        c = scatter_kwargs["c"]
        out["colorbar"] = add_colorbar(
            out["scatter"], axes, da=c if isinstance(c, xr.DataArray) else None, **colorbar_kwargs
        )

    # Labels
    axes.set_xlabel(get_label(sal, units=False))
    axes.set_ylabel(get_label(temp))

    # Density contours
    if dens is not False:

        import gsw

        (smin, tmin), (smax, tmax) = axes.viewLim.get_points()
        tt = np.linspace(tmin, tmax, 100)
        ss = np.linspace(smin, smax, 100)
        ss, tt = np.meshgrid(ss, tt)

        # absolute salinity and conservative temperature calculation
        lon_m = xcoords.get_lon(temp).mean()
        lat_m = xcoords.get_lat(temp).mean()
        if absolute is None:
            absolute = metaspecs.match_data_var(sal, "asal")
        if not absolute:
            ss_absolute = gsw.SA_from_SP(ss, 0, lon_m.values, lat_m.values)
        else:
            ss_absolute = ss.copy()
        tt_conservative = gsw.CT_from_pt(ss_absolute, tt)

        # Density as sigma{0-1-2-3-4} depending on ref_dens value
        if ref_dens == 0:
            func = gsw.sigma0
        elif ref_dens == 1:
            func = gsw.sigma1
        elif ref_dens == 2:
            func = gsw.sigma2
        elif ref_dens == 3:
            func = gsw.sigma3
        elif ref_dens == 4:
            func = gsw.sigma4

        dd = func(ss_absolute, tt_conservative)

        # Contours
        contour_kwargs = xmisc.dict_filter(
            kwargs, "contour_", defaults={"colors": ".3"}, **(contour_kwargs or {})
        )
        out["contour"] = axes.contour(ss, tt, dd, **contour_kwargs)

        # Contour labels
        clabel_kwargs = xmisc.dict_filter(kwargs, "clabel_", defaults={}, **(clabel_kwargs or {}))
        out["clabel"] = axes.clabel(out["contour"], **clabel_kwargs)

    return out


minimap_plot_coords = xmisc.Choices(
    {
        "auto": "infer from coords",
        "box": "rectangle of coords extent",
        "filledbox": "filled rectangle of coords extent",
        "points": "scatter plot",
        "lines": "as lines",
        "center": "a single point at the center",
        "Filled": "filled polygon",
        False: "do not plot",
    },
    parameter="plot_coords",
    description="Type of plot for coordinates",
    aliases={"auto": [True, None]},
)


@minimap_plot_coords.format_function_docstring
def plot_minimap(
    obj,
    ax=[0.9, 0.9, 0.09],
    fig=None,
    extent=1.0,
    min_extent=2.0,
    gridlines=True,
    ocean_color=None,
    land_color=None,
    land_scale="110m",
    plot_coords="auto",
    coords_markersize=10,
    coords_linewidth=2,
    coords_color="tab:red",
    coords_facecolor=0.2,
    **kwargs,
):
    """Plot a small map to show the geographic situation of coordinates

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset, tuple
        Object that contains lon and lat coordinates.
        The object must contain unique and identifiable geographic coordinates
        that can be retrieved with :func:`xoa.coords.get_lon` and :func:`xoa.coords.get_lat`.
        In case of a tuple, it should a couple of `(lon, lat)` :class:`xarray.DataArray`.
    ax: list, cartopy.mpl.geoaxes.GeoAxes
        A matplotlib axes instance or a list that defines the bounding box of the axes to be
        created ``[xmin, ymin, width, height]`` or ``[xmin, ymin, size]`` in figure coordinates.
    fig: figure
        Figure instance
    extent: str, float
        Either ``"global"`` for a global minimap, or the margin added to the coordinates
        bounding box expressed relative to the coordinates extent: a value of 1.0 means
        a margin equal to the extent.
    min_extent: None, float, (float, float)
        Minimal extent in degrees. See :func:`xoa.geo.get_extent`
    gridlines: bool
        Add gridlines.
    ocean_color: None, color
        A matplotlib color for oceans.
        If `None`, it defaults to ``cartopy.feature.COLORS["ocean"]``.
    land_color: None, color
        A matplotlib color for lands.
        If `None`, it defaults to ``cartopy.feature.COLORS["land"]``.
    {plot_coords}
    coords_markersize: float
        Size of markers in ``"points"`` mode.
    coords_linewidth: float
        Line width in ``"lines"``, `"box"`` and ``"filledbox"`` modes.
    coord_color: color
        Matplotlib color.
    coords_facecolor: color
        Matplotlib face color in ``"filled"`` and ``"filledbox"`` modes.
        If a float, it is interpreted as an alpha transparency that is applied to `coord_color`
        to get the face color.
    **kwargs:
        Parameters matching ``land_<param>`` are passed to the function that plots
        lands.
        Parameters matching ``coords_<param>`` are passed to the function that plots
        coordinates.
        Parameters matching ``gridlines_<param>`` are passed to
        :meth:`cartopy.mpl.geoaxes.GeoAxes.gridlines`.

    Return
    ------
    cartopy.mpl.geoaxes.GeoAxes
        An instance of cartopy geographic axes

    Example
    -------
    .. ipython:: python
        :okwarning:

        @suppress
        import xarray as xr, numpy as np, matplotlib.pyplot as plt
        @suppress
        from xoa.plot import plot_minimap
        ds = xr.Dataset(coords={{"lon": ("station", [-7, -5, -5]), "lat": ("station", [44, 44, 46])}})
        plt.plot([0, 2], [0, 2])
        @savefig api.plot.plot_minimap.global.png
        plot_minimap(ds, extent="global")
        plt.figure()
        plt.plot([0, 2], [0, 2])
        @savefig api.plot.plot_minimap.regional.png
        plot_minimap(ds, extent=1.2, color="tab:green", gridlines=False, land_color="k")

    See also
    --------
    plot_double_minimap
    xoa.coords.get_lon
    xoa.coords.get_lat
    xoa.geo.get_extent
    """
    # Create map
    ccrs, cfeature = _import_cartopy_()
    pcar = ccrs.PlateCarree()
    if isinstance(obj, tuple):
        lon, lat = obj
    else:
        lon = xcoords.get_lon(obj)
        lat = xcoords.get_lat(obj)
    if fig is None:
        fig = plt.gcf()
    if isinstance(ax, (list, tuple)):
        if ocean_color is None:
            ocean_color = cfeature.COLORS["water"]
        proj = ccrs.NearsidePerspective(
            central_longitude=float(lon.mean()),
            central_latitude=float(lat.mean()),  # satellite_height=50000
        )
        if len(ax) == 3:
            ax = ax + ax[-1:]
        ax = fig.add_axes(ax, projection=proj, facecolor=ocean_color)
        ax.spines["geo"].set_linewidth(0.2)
    if gridlines:
        kwgl = xmisc.dict_filter(kwargs, "gridlines_")
        ax.gridlines(**kwgl)
    if land_scale is None:
        land_scale = "50m" if extent == "global" else "110m"
    if extent == "global":
        ax.set_global()
    else:
        extent = np.array(xgeo.get_extent(obj, square=True, margin=extent, min_extent=min_extent))
        ax.set_extent(extent, pcar)

    # Add land
    kwland = xmisc.dict_filter(kwargs, "land_")
    if land_color is None:
        land_color = cfeature.COLORS["land"]
    kwland["facecolor"] = land_color
    ax.add_feature(cfeature.LAND.with_scale(land_scale), **kwland)

    # Add coordinates
    plot_coords = minimap_plot_coords[plot_coords]
    if plot_coords is not False:
        kwcoords = xmisc.dict_filter(kwargs, "coords_")
        kwcoords.update(transform=pcar)
        if plot_coords in ("box", "filledbox"):
            bbox = xgeo.get_extent((lon, lat))
            lon = [bbox[0], bbox[1], bbox[1], bbox[0], bbox[0]]
            lat = [bbox[2], bbox[2], bbox[3], bbox[3], bbox[2]]
            plot_coords = "lines" if plot_coords == "box" else "filled"
        elif plot_coords == "center":
            lon = [lon.mean()]
            lat = [lat.mean()]
            plot_coords = "points"
        if plot_coords is None:
            plot_coords = "auto"
        if plot_coords == "auto":
            if lon.dims == lat.dims and lon.ndim == 1 and lon.size > 0:
                plot_coords = "lines"
            else:
                lon, lat = xr.broadcast(lon, lat)
                plot_coords = "points"
        if plot_coords == "lines":
            kwcoords.update(linewidth=coords_linewidth, color=coords_color)
            ax.plot(lon, lat, **kwcoords)
        elif plot_coords == "filled":
            if isinstance(coords_facecolor, float) and coords_facecolor <= 1:
                fc_alpha = coords_facecolor
                coords_facecolor = mcolors.to_rgba(coords_color)
                fc_alpha *= coords_facecolor[-1]
                coords_facecolor = coords_facecolor[:3] + (fc_alpha,)
            kwcoords.update(
                linewidth=coords_linewidth, color=coords_color, facecolor=coords_facecolor
            )
            ax.fill(lon, lat, **kwcoords)
        else:
            kwcoords.update(s=coords_markersize, c=coords_color)
            ax.scatter(lon, lat, **kwcoords)

    return ax


def plot_double_minimap(obj, regional_ax="below", **kwargs):
    """Plot a global minimap and a regional minimap

    It consists in two calls to :func:`plot_minimap` with the first one with a "global" extent.
    By default, the coordinates are plotted as single point on the global minimap, and the
    regional minimap is placed below the global one.

    Parameters
    ----------
    regional_ax: axes, list, str
        If a string, it should be one of ``"below"``, ``"above"``, ``"left"`` or ``"right"``
        and it is interpreted as a relative position of the regional minimap with respect
        to the global one.
    **kwargs:
        Parameters matching ``global_<param>`` are passed to the global minimap while
        parameters matching ``regional_<param>`` are passed to the regional minimap.
        All other parameters are passed to both minimaps.

    Returns
    -------
    cartopy.mpl.geoaxes.GeoAxes, cartopy.mpl.geoaxes.GeoAxes
        A tuple of `(global_ax, regional_ax)`

    Example
    -------
    .. ipython:: python

        @suppress
        import xarray as xr, numpy as np, matplotlib.pyplot as plt
        @suppress
        from xoa.plot import plot_double_minimap
        ds = xr.Dataset(coords={"lon": ("station", [-7, -5, -5]), "lat": ("station", [44, 44, 46])})
        plt.plot([0, 2], [0, 2])
        @savefig api.plot.plot_double_minimap.png
        plot_double_minimap(ds)

    See also
    --------
    plot_minimap
    """

    # Filter keywords
    kwglobal = xmisc.dict_filter(kwargs, "global_")
    kwregional = xmisc.dict_filter(kwargs, "regional_")

    # Global minimap
    kw = kwargs.copy()
    kwglobal.update(extent="global")
    kwglobal.setdefault("plot_coords", "center")
    kw.update(kwglobal)
    global_ax = plot_minimap(obj, **kw)

    # Regional minimap
    if isinstance(regional_ax, str):
        bb = global_ax.bbox.transformed(global_ax.figure.transFigure.inverted())
        if regional_ax == "right":
            regional_ax = [bb.xmin + bb.width * 1.1, bb.ymin, bb.width, bb.height]
        elif regional_ax == "above":
            regional_ax = [bb.xmin, bb.ymax + bb.height * 0.1, bb.width, bb.height]
        elif regional_ax == "left":
            regional_ax = [bb.xmin - bb.width * 1.1, bb.ymin, bb.width, bb.height]
        else:
            regional_ax = [bb.xmin, bb.ymin - bb.height * 1.1, bb.width, bb.height]
    kw = kwargs.copy()
    kwregional.update(ax=regional_ax)
    kw.update(kwregional)
    regional_ax = plot_minimap(obj, **kw)

    return global_ax, regional_ax


# %% Labels and maps


def get_label(da, units=True):
    """Get a label like ``"Long name [units]"`` for a data array

    Missing attributes are completed from the current meta specs with
    :meth:`xoa.meta.MetaSpecs.fill_attrs`.

    Parameters
    ----------
    da: xarray.DataArray
    units: bool
        Add the units between brackets.

    Return
    ------
    str

    Example
    -------
    .. ipython:: python

        @suppress
        import xarray as xr
        @suppress
        from xoa.plot import get_label
        da = xr.DataArray([1.], dims="x", name="temp")
        get_label(da)
        get_label(da, units=False)
    """
    attrs = xmeta.get_meta_specs(da).fill_attrs(da).attrs
    label = attrs.get("long_name") or attrs.get("standard_name") or da.name or ""
    label = str(label)
    if label:
        label = label[0].upper() + label[1:]
    unit = attrs.get("units")
    if units and unit:
        label = f"{label} [{unit}]" if label else f"[{unit}]"
    return label


def _get_cbar_kwargs_(da=None, cbar_kwargs=None):
    """Get the colorbar parameters with the shrink factor and the label as defaults"""
    kwargs = dict(cbar_kwargs or {})
    kwargs.setdefault("shrink", cplot.CBAR_SHRINK)
    if da is not None:
        kwargs.setdefault("label", get_label(da))
    return kwargs


def add_colorbar(mappable, ax=None, da=None, **kwargs):
    """Add a colorbar that is shrunk and labelled by default

    Parameters
    ----------
    mappable: matplotlib.cm.ScalarMappable
        The plotted artist, as returned by the plotting functions of this module
    ax: None, matplotlib.axes.Axes, list(matplotlib.axes.Axes)
        Axes that are shrunk to make room for the colorbar, which is shared if there are
        several. It defaults to the axes of the artist.
    da: None, xarray.DataArray
        Array that is used to label the colorbar with :func:`get_label`
    kwargs:
        Extra parameters are passed to :func:`xoa.core.plot.add_colorbar`.
        They override the label and the default shrink factor.

    Return
    ------
    matplotlib.colorbar.Colorbar

    Example
    -------
    .. code-block:: python

        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": ccrs.Mercator()})
        for ax in axes:
            mappable = plot_field(da, ax=ax, add_colorbar=False)
        add_colorbar(mappable, axes, da=da)  # one shared colorbar
    """
    return cplot.add_colorbar(mappable, ax, **_get_cbar_kwargs_(da, kwargs))


# For the functions that have a parameter with the same name
_add_colorbar_ = add_colorbar


def _get_meta_var_(obj, meta_name):
    """Get a data variable from its generic meta name or return None"""
    return xmeta.get_meta_specs(obj).search(obj, meta_name, errors="ignore")


def plot_field(
    field,
    ax=None,
    method="pcolormesh",
    transform=None,
    title=None,
    margin=0.0,
    map_kw=None,
    overlay_contours=None,
    ds=None,
    **kwargs,
):
    """Plot a field on a map

    Longitudes and latitudes are found with :func:`xoa.coords.get_lon` and
    :func:`xoa.coords.get_lat`, and the extent of the map is computed with
    :func:`xoa.geo.get_extent`.
    The colorbar is labelled with :func:`get_label`.

    Parameters
    ----------
    field: xarray.DataArray
        Field with longitude and latitude coordinates, and no other dimension
        than the horizontal ones.
    ax: None, cartopy.mpl.geoaxes.GeoAxes
        Axes to plot on. A map is created if not provided.
    method: {"pcolormesh", "contourf", "contour"}
        Plot method of :class:`xarray.DataArray.plot`
    transform: None, cartopy.crs.CRS
        Coordinate system of the field, which defaults to ``PlateCarree``
    title: None, str
    margin: float
        Margin added to the extent. See :func:`xoa.geo.get_extent`.
    map_kw: None, dict
        Parameters that decorate the map, as accepted by :func:`setup_map_axes`.
        When ``ax`` is not provided, ``figsize`` and ``projection`` are also
        accepted, as in :func:`create_base_map`.
    overlay_contours: None, True, list(dict)
        Line contours drawn over the field. ``True`` is a shorthand for a single
        contour of the field. Each item is a dict of parameters passed to
        :meth:`matplotlib.axes.Axes.contour`, with an extra ``field`` key, which is
        either a data array on any grid, a generic meta name like ``"bathy"``
        or ``"mask"`` that is searched in ``ds`` with :mod:`xoa.meta`,
        or the plotted field by default. Colors default to black.
    ds: None, xarray.Dataset
        Dataset where the meta names of overlay fields are searched
    kwargs:
        Extra parameters are passed to the plot method, like ``cmap``, ``vmin``,
        ``vmax`` or ``add_colorbar``. The colorbar is shrunk by default, which
        can be changed with ``cbar_kwargs``.

    Return
    ------
    matplotlib.cm.ScalarMappable
        The artist returned by the plot method

    Example
    -------
    .. code-block:: python

        plot_field(ds.temp.isel(time=0), cmap="Spectral_r",
                   overlay_contours=[dict(field="bathy", levels=[100, 1000])], ds=ds)

    See also
    --------
    plot_grid
    plot_section
    """
    ccrs, _ = _import_cartopy_()
    if transform is None:
        transform = ccrs.PlateCarree()
    lon = xcoords.get_lon(field)
    lat = xcoords.get_lat(field)
    extent = xgeo.get_extent(field, margin=margin)
    kw = dict(map_kw or {})
    if ax is None:
        _, ax = create_base_map(extent, **kw)
    else:
        setup_map_axes(
            ax, extent, transform, **{k: v for k, v in kw.items() if k in _AX_SETUP_KEYS}
        )

    if kwargs.get("add_colorbar", True):
        kwargs["cbar_kwargs"] = _get_cbar_kwargs_(field, kwargs.get("cbar_kwargs"))
    mappable = getattr(field.plot, method)(
        x=lon.name, y=lat.name, ax=ax, transform=transform, add_labels=False, **kwargs
    )

    if overlay_contours is True:
        overlay_contours = [{}]
    for spec in overlay_contours or ():
        spec = dict(spec)
        ofield = spec.pop("field", field)
        if isinstance(ofield, str):
            if ds is None:
                raise exceptions.XoaError(
                    f"Cannot search the overlay field '{ofield}' without the ds parameter"
                )
            name = ofield
            ofield = _get_meta_var_(ds, name)
            if ofield is None:
                raise exceptions.XoaError(f"Overlay field '{name}' not found in the dataset")
        spec.setdefault("colors", "k")
        olon = xcoords.get_lon(ofield)
        olat = xcoords.get_lat(ofield)
        ax.contour(ofield[olon.name], ofield[olat.name], ofield, transform=transform, **spec)

    if title:
        ax.set_title(title)
    return mappable


def plot_grid(
    obj,
    ax=None,
    kind="mesh",
    stride="auto",
    min_spacing=12,
    edges=None,
    centers=None,
    strips=None,
    n_cells=1,
    transform=None,
    map_kw=None,
    edge_kw=None,
    center_kw=None,
    **kwargs,
):
    """Plot a horizontal grid

    A grid is made of cells whose centers are the points where the data are defined.
    Cell edges are computed from the centers with :func:`xoa.core.grid.centers2edges`,
    and both can be drawn, optionally over a field. Large grids are under-sampled to stay
    readable.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Object with longitude and latitude coordinates, that may be 1D or 2D.
        With the ``"mesh"`` and ``"resolution"`` kinds, the coordinates must be unambiguous,
        which means a single location on a staggered grid: pass a data array in
        case of doubt. With the ``"bathy"`` kind, the grid is the one of the bathymetry.
    ax: None, cartopy.mpl.geoaxes.GeoAxes
        Axes to plot on. A map is created if not provided.
    kind: {"mesh", "bathy", "resolution"}
        What to draw:

        - ``"mesh"``: only the edges and centers
        - ``"bathy"``: the bathymetry that is searched in ``obj`` with
          the ``"bathy"`` meta name and masked with the ``"mask"`` one if any.
          It falls back to the mesh with a warning if there is no bathymetry.
        - ``"resolution"``: the resolution computed with :func:`xoa.grid.get_resolution`
          as the geometric mean of the resolutions along x and y, in km.

    stride: str, int, tuple(int)
        Under-sampling of the edges and centers, either one value for both directions
        or a ``(y, x)`` tuple. With ``"auto"``, the stride is adapted to the size of the axes
        and to the number of cells, so that there are at least ``min_spacing`` pixels
        between consecutive cells, measured on the map. When the grid is under-sampled, the drawn centers
        are the ones of the coarse cells delimited by the drawn edges.
    min_spacing: float
        Minimal spacing in pixels between cells in the ``"auto"`` stride mode.
    edges: None, bool
        Draw the edges of the cells. It defaults to True for the ``"mesh"`` kind, and
        to False with the others. The first and last edges are always drawn.
    centers: None, bool
        Draw the centers of the cells. It defaults to True for the ``"mesh"`` kind, and
        to False with the others.
    strips: None, str, list(str)
        Draw the extent of the strips along these edges.
        See :func:`xoa.grid.get_edge_extents`.
    n_cells: int
        Number of cells of the strips
    transform: None, cartopy.crs.CRS
        Coordinate system, which defaults to ``PlateCarree``
    map_kw: None, dict
        Parameters that decorate the map. See :func:`plot_field`.
    edge_kw: None, dict
        Parameters passed to :class:`matplotlib.collections.LineCollection`
        to draw the edges, like ``colors`` or ``linewidths``.
    center_kw: None, dict
        Parameters passed to :meth:`matplotlib.axes.Axes.plot` to draw the centers,
        like ``color`` or ``markersize``.
    kwargs:
        With the ``"bathy"`` and ``"resolution"`` kinds, extra parameters are passed
        to :func:`plot_field`.

    Return
    ------
    cartopy.mpl.geoaxes.GeoAxes

    Example
    -------
    .. code-block:: python

        plot_grid(ds.temp)  # edges and centers, under-sampled if needed
        plot_grid(ds.temp, stride=(2, 4), centers=False)
        plot_grid(ds, kind="bathy", edges=True, strips="north")

    See also
    --------
    plot_field
    xoa.grid.get_resolution
    xoa.grid.get_edge_extents
    xoa.core.grid.centers2edges
    """
    from . import grid as xgrid

    ccrs, _ = _import_cartopy_()
    if transform is None:
        transform = ccrs.PlateCarree()
    if kind not in ("mesh", "bathy", "resolution"):
        raise exceptions.XoaError(f"Invalid kind '{kind}'. Choose among: mesh, bathy, resolution")

    # Background field, whose coordinates define the grid for the bathymetry
    field = None
    if kind == "bathy":
        field = _get_meta_var_(obj, "bathy") if isinstance(obj, xr.Dataset) else None
        if field is None:
            exceptions.xoa_warn("No bathymetry found: plotting the mesh instead")
            kind = "mesh"
        else:
            mask = _get_meta_var_(obj, "mask")
            if mask is not None and set(mask.dims) == set(field.dims):
                field = field.where(mask != 0)
            kwargs.setdefault("cmap", "terrain_r")
            obj = field
    if edges is None:
        edges = kind == "mesh"
    if centers is None:
        centers = kind == "mesh"

    # Axes
    lon, lat = xgrid._get_lonlat_yx_(obj)
    extent = xgeo.get_extent((lon.values, lat.values), margin=0.05)
    kw = dict(map_kw or {})
    if ax is None:
        _, ax = create_base_map(extent, **kw)
    else:
        setup_map_axes(
            ax, extent, transform, **{k: v for k, v in kw.items() if k in _AX_SETUP_KEYS}
        )

    # Background field
    if kind == "resolution":
        field = xr.DataArray(
            cgrid.compute_center_resolution(lon.values, lat.values) * 1e-3,
            dims=lon.dims,
            coords={lon.name: lon, lat.name: lat},
            attrs={"long_name": "Grid resolution", "units": "km"},
        )
    if field is not None:
        plot_field(field, ax=ax, transform=transform, **kwargs)

    # Edges and centers
    if edges or centers:
        cplot.plot_mesh(
            ax,
            lon.values,
            lat.values,
            stride=stride,
            min_spacing=min_spacing,
            edges=edges,
            centers=centers,
            transform=transform,
            edge_kw=edge_kw,
            center_kw=center_kw,
        )

    # Strips
    if strips:
        for xmin, xmax, ymin, ymax in xgrid.get_edge_extents(obj, strips, n_cells).values():
            ax.plot(
                [xmin, xmax, xmax, xmin, xmin],
                [ymin, ymin, ymax, ymax, ymin],
                color="tab:orange",
                linewidth=1.5,
                transform=transform,
            )
    return ax


# %% Sections and series


def plot_section(
    da,
    x=None,
    ax=None,
    method="pcolormesh",
    title=None,
    add_colorbar=True,
    cbar_kwargs=None,
    **kwargs,
):
    """Plot a vertical section

    The depth is found with :func:`xoa.coords.get_depth` and may vary with the
    horizontal dimension, like with terrain-following coordinates.
    The vertical axis is inverted when depths are positive down,
    according to :func:`xoa.coords.get_positive_attr`.

    Parameters
    ----------
    da: xarray.DataArray
        Array with a vertical and a horizontal dimension only
    x: None, str, xarray.DataArray
        Horizontal axis as one of:

        - ``None``: longitude or latitude, according to the largest extent
        - ``"lon"``, ``"lat"``: the coordinate found with :mod:`xoa.coords`
        - ``"distance"``: the distance in km along the section
        - a data array that is broadcastable to the horizontal dimension

    ax: None, matplotlib.axes.Axes
    method: {"pcolormesh", "contourf", "contour"}
        Plot method of :class:`matplotlib.axes.Axes`
    title: None, str
    add_colorbar: bool
    cbar_kwargs: None, dict
        Parameters passed to :meth:`matplotlib.figure.Figure.colorbar`.
        The colorbar is shrunk by default (``shrink=0.7``).
    kwargs:
        Extra parameters are passed to the plot method.

    Return
    ------
    matplotlib.cm.ScalarMappable

    Example
    -------
    .. code-block:: python

        plot_section(ds.temp.isel(time=0, eta_rho=10), cmap="Spectral_r")
    """
    zdim = xcoords.get_zdim(da, errors="raise")
    hdims = [dim for dim in da.dims if dim != zdim]
    if len(hdims) != 1:
        raise exceptions.XoaError(
            f"A section needs a single horizontal dimension, but got: {hdims}"
        )
    hdim = hdims[0]

    # Depth
    depth = xcoords.get_depth(da, errors="ignore")
    if depth is None:
        depth = (
            da[zdim] if zdim in da.coords else xr.DataArray(np.arange(da.sizes[zdim]), dims=zdim)
        )
    positive = xcoords.get_positive_attr(da, zdim=zdim)

    # Horizontal axis
    lon = xcoords.get_lon(da, errors="ignore")
    lat = xcoords.get_lat(da, errors="ignore")
    if isinstance(x, xr.DataArray):
        xaxis = x
    else:
        if x is None:
            if lon is not None and lat is not None:
                xlon = float(lon.max() - lon.min()) * np.cos(np.radians(float(lat.mean())))
                x = "lon" if xlon >= float(lat.max() - lat.min()) else "lat"
            else:
                x = "lon" if lon is not None else "lat"
        if x == "distance":
            if lon is None or lat is None:
                raise exceptions.XoaError("Longitude and latitude are needed to compute distances")
            lonv, latv = xr.broadcast(lon, lat)
            lonv = lonv.transpose(hdim, ...).values.reshape(da.sizes[hdim], -1)[:, 0]
            latv = latv.transpose(hdim, ...).values.reshape(da.sizes[hdim], -1)[:, 0]
            dist = np.concatenate(
                [[0], np.cumsum(xgeo.haversine(lonv[:-1], latv[:-1], lonv[1:], latv[1:]))]
            )
            xaxis = xr.DataArray(
                dist * 1e-3, dims=hdim, attrs={"long_name": "Distance", "units": "km"}
            )
        elif x in ("lon", "lat"):
            xaxis = lon if x == "lon" else lat
            if xaxis is None:
                raise exceptions.XoaError(f"No {x} coordinate found")
        else:
            raise exceptions.XoaError("x must be None, 'lon', 'lat', 'distance' or a data array")
    xaxis, depth = xr.broadcast(xaxis, depth)
    xaxis = xaxis.transpose(zdim, hdim)
    depth = depth.transpose(zdim, hdim)
    data = da.transpose(zdim, hdim)

    # Plot
    if ax is None:
        ax = plt.gca()
    mappable = cplot.plot_depth_section(
        ax,
        xaxis.values,
        depth.values,
        data.values,
        method=method,
        invert_yaxis=positive == "down",
        **kwargs,
    )
    ax.set_xlabel(get_label(xaxis))
    ax.set_ylabel(get_label(depth))
    if add_colorbar:
        _add_colorbar_(mappable, ax, da=da, **(cbar_kwargs or {}))
    if title:
        ax.set_title(title)
    return mappable


def plot_stick(u, v=None, ax=None, scale=None, color="steelblue", **kwargs):
    """Plot a current time series as sticks

    Each vector is drawn as a stick anchored at ``y=0``, oriented along the current
    and scaled by its speed. This shows the direction and intensity of
    the currents, like tidal ones, along a single axis.

    Parameters
    ----------
    u: xarray.DataArray, xarray.Dataset
        Eastward component with a single dimension. If it is a dataset,
        the ``"u"`` and ``"v"`` generic meta names are searched in it with :mod:`xoa.meta`.
    v: xarray.DataArray, None
        Northward component, with the same dimension
    ax: None, matplotlib.axes.Axes
    scale: None, float
        Scale passed to :meth:`matplotlib.axes.Axes.quiver`: the smaller, the longer
        the sticks.
    color: color
    kwargs:
        Extra parameters are passed to :meth:`matplotlib.axes.Axes.quiver`

    Return
    ------
    matplotlib.quiver.Quiver

    Example
    -------
    .. code-block:: python

        plot_stick(ds.u, ds.v, scale=2.0)
        plot_stick(ds)  # u and v are searched

    See also
    --------
    plot_flow
    """
    if v is None:
        ds = u
        u = _get_meta_var_(ds, "u")
        v = _get_meta_var_(ds, "v")
        if u is None or v is None:
            raise exceptions.XoaError("Cannot find the u and v components in the dataset")
    if u.ndim != 1 or v.ndim != 1:
        raise exceptions.XoaError("u and v must have a single dimension")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3))

    dim = u.dims[0]
    time = xcoords.get_time(u, errors="ignore")
    if time is not None and time.dims == (dim,):
        x = time.values
        xlabel = time.name
    elif dim in u.coords:
        x = u.coords[dim].values
        xlabel = dim
    else:
        x = np.arange(u.sizes[dim])
        xlabel = None

    quiver = cplot.plot_sticks(ax, x, u.values, v.values, scale=scale, color=color, **kwargs)
    if xlabel:
        ax.set_xlabel(xlabel)
    return quiver


def _taylor_points_(da, ref, dim):
    """Statistics of the points of a single data array, and their labels"""
    da, ref = xr.broadcast(da, ref)
    dims = list(da.dims) if dim is None else ([dim] if isinstance(dim, str) else list(dim))
    for d in dims:
        if d not in da.dims:
            raise exceptions.XoaError(f"Invalid dimension: {d}. Valid dimensions: {da.dims}")
    pdims = [d for d in da.dims if d not in dims]
    da = da.transpose(*pdims, *dims)
    ref = ref.transpose(*pdims, *dims)
    npoints = int(np.prod([da.sizes[d] for d in pdims])) if pdims else 1
    out = cstats.taylor_stats(da.values.reshape(npoints, -1), ref.values.reshape(npoints, -1))
    if pdims:
        index = da.stack(point=pdims).get_index("point")
        labels = [
            ", ".join(map(str, item if isinstance(item, tuple) else (item,))) for item in index
        ]
    else:
        labels = [None]
    return out, labels


def plot_taylor(
    obj, ref, dim=None, normalize=False, labels=None, values=None, diagram=None, **kwargs
):
    """Plot a Taylor diagram of arrays against a reference

    The standard deviation, the correlation with the reference and the centered root mean
    square difference are computed over the dimensions ``dim``, and the other dimensions
    create the points of the diagram. The samples that are missing in the array or in the
    reference are ignored. Correlations may be negative.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Array to evaluate. With a dataset, each variable gives at least one point.
    ref: xarray.DataArray, xarray.Dataset, str
        Reference, which is broadcast against the array. With a dataset, it can be a variable
        name, or a dataset whose variables have the names of the variables of ``obj``.
    dim: None, str, list(str)
        Dimensions where the statistics are computed, which are all the dimensions by default,
        giving a single point per variable. Dimensions that are not listed create points.
    normalize: bool
        Divide the standard deviations by the one of the reference, which is thus at 1.
        It is required when the reference has different standard deviations for the points.
    labels: None, list(str)
        Labels of the points, that are shown in the legend. They are
        made from the variable names and the coordinates of the point dimensions by default.
    values: None, array_like, xarray.DataArray
        One value per point, with the same order, that colors the markers
        instead of the legend, with a colorbar labelled with :func:`get_label`.
    diagram: None, xoa.core.plot.TaylorDiagram
        Existing diagram
    kwargs:
        Extra parameters, like ``markers``, ``colors``, ``rmax``, ``rms_levels`` or ``cmap``,
        are passed to :func:`xoa.core.plot.plot_taylor`.

    Return
    ------
    xoa.core.plot.TaylorDiagram

    Example
    -------
    .. code-block:: python

        # One point per model, statistics over time and space
        plot_taylor(ds.sst_models, ds.sst_obs, dim=("time", "lat", "lon"), markers=["o", "s"])
        # One point per variable of a dataset
        plot_taylor(ds_models, ref=ds_obs, normalize=True)

    See also
    --------
    xoa.core.plot.TaylorDiagram
    xoa.core.stats.taylor_stats
    """
    if isinstance(obj, xr.Dataset):
        items = []
        for name, da in obj.data_vars.items():
            if isinstance(ref, str):
                rda = obj[ref]
                if name == ref:
                    continue
            elif isinstance(ref, xr.Dataset):
                if name not in ref:
                    continue
                rda = ref[name]
            else:
                rda = ref
            items.append((name, da, rda))
        if not items:
            raise exceptions.XoaError("No variable to compare with the reference")
    else:
        items = [(None, obj, ref)]
    stats = []
    pts = []
    for name, da, rda in items:
        out, plabels = _taylor_points_(da, rda, dim)
        stats.append(out)
        for label in plabels:
            pts.append(", ".join(str(x) for x in (name, label) if x is not None) or None)
    std, std_ref, corr, _ = (np.concatenate([out[i] for out in stats]) for i in range(4))

    if normalize:
        std = std / std_ref
        ref_std = 1.0
    else:
        ref_std = float(np.nanmean(std_ref))
        if not np.allclose(std_ref, ref_std, equal_nan=True):
            raise exceptions.XoaError(
                "The reference has different standard deviations: use normalize=True"
            )

    if labels is None and len(pts) > 1:
        labels = pts
    if diagram is None:
        first = items[0][1]
        units = first.attrs.get("units")
        if normalize:
            kwargs.setdefault("std_label", "Normalized standard deviation")
        elif units:
            kwargs.setdefault("std_label", f"Standard deviation [{units}]")
    if values is not None:
        if isinstance(values, xr.DataArray):
            cbar_kwargs = dict(kwargs.get("cbar_kwargs") or {})
            cbar_kwargs.setdefault("label", get_label(values))
            kwargs["cbar_kwargs"] = cbar_kwargs
        values = np.ravel(values)
    return cplot.plot_taylor(
        std, corr, ref_std=ref_std, diagram=diagram, labels=labels, values=values, **kwargs
    )


# %% Filters


def _smooth2d_(A, sigma):
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(A, sigma, truncate=3)


class _BaseFilter_(object):
    def prepare_image(self, src_image, dpi, pad):
        ny, nx, depth = src_image.shape
        padded_src = np.zeros([pad * 2 + ny, pad * 2 + nx, depth], dtype="d")
        padded_src[pad:-pad, pad:-pad, :] = src_image[:, :, :]
        return padded_src

    def get_pad(self, dpi):
        return 0

    def __call__(self, im, dpi):
        pad = self.get_pad(dpi)
        padded_src = self.prepare_image(im, dpi, pad)
        tgt_image = self.process_image(padded_src, dpi)
        return tgt_image, -pad, -pad


class OffsetFilter(_BaseFilter_):
    def __init__(self, offsets=None):
        if offsets is None:
            self.offsets = (0, 0)
        else:
            self.offsets = offsets

    def get_pad(self, dpi):
        return int(max(*self.offsets) / 72.0 * dpi)

    def process_image(self, padded_src, dpi):
        ox, oy = self.offsets
        a1 = np.roll(padded_src, int(ox / 72.0 * dpi), axis=1)
        a2 = np.roll(a1, -int(oy / 72.0 * dpi), axis=0)
        return a2


class GaussianFilter(_BaseFilter_):
    """Gaussian filter"""

    def __init__(self, sigma, alpha=0.5, color=None):
        self.sigma = sigma
        self.alpha = alpha
        if color is None:
            self.color = (0, 0, 0)
        else:
            self.color = color

    def get_pad(self, dpi):
        return int(self.sigma * 3 / 72.0 * dpi)

    def process_image(self, padded_src, dpi):
        # offsetx, offsety = int(self.offsets[0]), int(self.offsets[1])
        tgt_image = np.zeros_like(padded_src)
        aa = _smooth2d_(padded_src[:, :, -1] * self.alpha, self.sigma / 72.0 * dpi)
        tgt_image[:, :, -1] = aa
        tgt_image[:, :, :-1] = self.color
        return tgt_image


class DropShadowFilter(_BaseFilter_):
    """Create a drop shadow"""

    def __init__(self, width, alpha=0.3, color=None, offsets=None):
        self.gauss_filter = GaussianFilter(width / 3, alpha, color)
        self.offset_filter = OffsetFilter(offsets)

    def get_pad(self, dpi):
        return max(self.gauss_filter.get_pad(dpi), self.offset_filter.get_pad(dpi))

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        t2 = self.offset_filter.process_image(t1, dpi)
        return t2


class GrowFilter(_BaseFilter_):
    "Enlarge the area"

    def __init__(self, pixels, color=None, alpha=1.0):
        self.pixels = pixels
        if color is None:
            self.color = (1, 1, 1)
        else:
            self.color = color
        self.alpha = alpha

    def __call__(self, im, dpi):
        pad = self.pixels
        ny, nx, depth = im.shape
        new_im = np.empty([pad * 2 + ny, pad * 2 + nx, depth], dtype="d")
        alpha = new_im[:, :, 3]
        alpha.fill(0)
        alpha[pad:-pad, pad:-pad] = im[:, :, -1]
        alpha2 = np.clip(_smooth2d_(alpha, self.pixels / 72.0 * dpi) * 5, 0, 1) * self.alpha
        new_im[:, :, -1] = alpha2
        new_im[:, :, :-1] = self.color
        offsetx, offsety = -pad, -pad

        return new_im, offsetx, offsety


class LightFilter(_BaseFilter_):
    """Apply a light filter"""

    def __init__(self, sigma, fraction=0.5, **kwargs):
        self.gauss_filter = GaussianFilter(sigma / 3, alpha=1)
        self.light_source = mcolors.LightSource(**kwargs)
        self.fraction = fraction

    def get_pad(self, dpi):
        return self.gauss_filter.get_pad(dpi)

    def process_image(self, padded_src, dpi):
        t1 = self.gauss_filter.process_image(padded_src, dpi)
        elevation = t1[:, :, 3]
        rgb = padded_src[:, :, :3]
        rgb2 = self.light_source.shade_rgb(rgb, elevation, fraction=self.fraction)
        tgt = np.empty_like(padded_src)
        tgt[:, :, :3] = rgb2
        tgt[:, :, 3] = padded_src[:, :, 3]

        return tgt


class FilteredArtistList(martist.Artist):
    """
    A simple container to draw filtered artist.
    """

    def __init__(self, artist_list, filter):
        self._artist_list = artist_list
        self._filter = filter
        super().__init__()

    def draw(self, renderer):
        renderer.start_rasterizing()
        if hasattr(renderer, 'start_filter'):
            renderer.start_filter()
        for a in self._artist_list:
            if hasattr(a, 'draw'):
                a.draw(renderer)
        renderer.stop_filter(self._filter)
        renderer.stop_rasterizing()


def add_agg_filter(objs, filter, zorder=None, ax=None, add=True):
    """Add a filtered version of objects to plot

    Parameters
    ----------

    objs: :class:`matplotlib.artist.Artist`
        Plotted objects.
    filter: :class:`BaseFilter`
    zorder: optional
        zorder (else guess from ``objs``).
    ax: optional, :class:`matplotlib.axes.Axes`
    """
    # Input
    if not isinstance(objs, (list, tuple)):
        objs = [objs]
    elif len(objs) == 0:
        return []

    # Filter
    if ax is None:
        ax = plt.gca()
    shadows = FilteredArtistList(objs, filter)
    if hasattr(add, 'add_artist'):
        add.add_artist(shadows)
    elif add:
        ax.add_artist(shadows)

    # Text
    for t in objs:
        if isinstance(t, mtext.Text):
            t.set_path_effects([mpatheffects.Normal()])

    # Adjust zorder
    if zorder is None or zorder is True:
        same = zorder is True
        if hasattr(objs, 'get_zorder'):
            zorder = objs.get_zorder()
        else:
            zorder = objs[0].get_zorder()
        if not same:
            zorder -= 0.1
    if zorder is not False:
        shadows.set_zorder(zorder)

    return shadows


def add_shadow(
    objs, width=3, xoffset=2, yoffset=-2, alpha=0.5, color='k', zorder=None, ax=None, add=True
):
    """Add a drop-shadow to objects

    Parameters
    ----------
    objs: :class:`matplotlib.artist.Artist`
        Plotted objects.
    width: optional
        Width of the gaussian filter in points.
    xoffset: optional
        Shadow offset along X in points.
    yoffset: optional
        Shadow offset along Y in points.
    color: optional
        Color of the shadow.
    zorder: optional
        zorder (else guess from ``objs``).
    ax: optional, :class:`matplotlib.axes.Axes`

    Inspired from http://matplotlib.sourceforge.net/examples/pylab_examples/demo_agg_filter.html .
    """
    if color is not None:
        color = mcolors.ColorConverter().to_rgb(color)
    try:
        gauss = DropShadowFilter(width, offsets=(xoffset, yoffset), alpha=alpha, color=color)
        return add_agg_filter(objs, gauss, zorder=zorder, ax=ax, add=add)
    except:
        exceptions.xoa_warn('Cannot plot shadows using agg filters')


def add_glow(objs, width=3, zorder=None, color='w', ax=None, alpha=1.0, add=True):
    """Add a glow effect to text

    Parameters
    ----------
    objs: :class:`matplotlib.artist.Artist`
        Plotted objects.
    width: optional
        Width of the gaussian filter in points.
    color: optional
        Color of the shadow.
    zorder: optional
        zorder (else guess from ``objs``).
    ax: optional, :class:`matplotlib.axes.Axes`

    Inspired from http://matplotlib.sourceforge.net/examples/pylab_examples/demo_agg_filter.html .
    """
    if color is not None:
        color = mcolors.ColorConverter().to_rgb(color)
    try:
        white_glows = GrowFilter(width, color=color, alpha=alpha)
        return add_agg_filter(objs, white_glows, zorder=zorder, ax=ax, add=add)
    except:
        exceptions.xoa_warn('Cannot add glow effect using agg filters')


def add_lightshading(objs, width=7, fraction=0.5, zorder=None, ax=None, add=True, **kwargs):
    """Add a light shading effect to objects

    Parameters
    ----------
    objs: :class:`matplotlib.artist.Artist`
        Plotted objects.
    width: optional
        Width of the gaussian filter in points.
    fraction: optional
        Unknown.
    zorder: optional
        zorder (else guess from ``objs``).
    ax: optional, :class:`matplotlib.axes.Axes`
    **kwargs
        Extra keywords are passed to :class:`matplotlib.colors.LightSource`

    Inspired from http://matplotlib.sourceforge.net/examples/pylab_examples/demo_agg_filter.html .
    """
    if zorder is None:
        zorder = True
    try:
        lf = LightFilter(width, fraction=fraction, **kwargs)
        return add_agg_filter(objs, lf, zorder=zorder, add=add, ax=ax)
    except:
        exceptions.xoa_warn('Cannot add light shading effect using agg filters')
