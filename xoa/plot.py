# -*- coding: utf-8 -*-
"""
Plotting utilities

Filters are adapted from https://matplotlib.org/stable/gallery/misc/demo_agg_filter.html?highlight=agg%20filter
and http://vacumm.github.io/vacumm/library/misc.core_plot.html

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
import matplotlib.artist as martist
import matplotlib.text as mtext
import matplotlib.patheffects as mpatheffects
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from .__init__ import xoa_warn
from . import misc as xmisc
from . import geo as xgeo
from . import cf as xcf
from . import coords as xcoords
from . import dyn

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
    pres=None,
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
    pres: xarray.DataArray, None
        Pressure to compute potential temperature.
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

        if pres is None:
            lat = xcoords.get_lat(temp)
            depth = xcoords.get_depth(temp)
            lat, depth = xr.broadcast(lat, depth)
            pres = gsw.p_from_z(depth, lat)
        attrs = temp.attrs
        temp = gsw.pt0_from_t(sal, temp, pres)
        temp.attrs.update(attrs)
        cfspecs.format_data_var(temp, cf_name="ptemp", copy=False, replace_attrs=True)

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
        The object must contains unique and identifiable geographic coordinates
        that can be retrieved xith :func:`xoa.coords.get_lon` and func:`xoa.coords.get_lat`.
        In case of a tuple, it should a couple of `(lon, lat)` :class:`xarray.DataArray`.
    ax: list, cartopy.mpl.geoaxes.GeoAxes
        A matplotlib axes instance or a list that defines the bounding box of the axes to be
        created ``[xmin, ymin, width, height]`` or ``[xmin, ymin, size]`` in figure coordinates.
    fig: figure
        Figure instance
    extent: str, float
        Either ``"global"`` for a global minimap, or the margin added to the coordinates
        bounding box expressed relative to the coordinates extent: a velue of 1.0 means
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

    It consists in two calls to :func:`plot_minimap` with the first one whith a "global" extent.
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
        A tuple of `(global_ax, regiona_ax)`

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
        xoa_warn('Cannot plot shadows using agg filters')


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
        xoa_warn('Cannot add glow effect using agg filters')


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
        xoa_warn('Cannot add light shading effect using agg filters')
