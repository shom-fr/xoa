"""
Low level plotting routines

These routines work on numpy arrays and matplotlib axes, and know nothing about
xarray objects or meta specifications.
They are used by the high level plotting functions of the :mod:`xoa.plot` module.
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
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .. import exceptions
from .. import misc
from .grid import centers2edges

# %% Maps


def _import_cartopy_():
    """Import cartopy lazily"""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as exc:
        raise ImportError(
            "cartopy is required for map plots. Install it with: pip install cartopy"
        ) from exc
    return ccrs, cfeature


def add_land(ax, scale="110m", color=None, **kwargs):
    """Add the land to a cartopy axes

    Parameters
    ----------
    ax: cartopy.mpl.geoaxes.GeoAxes
    scale: str
        Natural Earth scale like ``"110m"``, ``"50m"`` or ``"10m"``
    color: None, color
        Face color which defaults to ``cartopy.feature.COLORS["land"]``
    kwargs:
        Extra parameters are passed to :meth:`cartopy.mpl.geoaxes.GeoAxes.add_feature`

    Return
    ------
    cartopy.mpl.feature_artist.FeatureArtist
    """
    ccrs, cfeature = _import_cartopy_()
    if color is None:
        color = cfeature.COLORS["land"]
    kwargs.setdefault("facecolor", color)
    return ax.add_feature(cfeature.LAND.with_scale(scale), **kwargs)


def setup_map_axes(
    ax,
    extent=None,
    transform=None,
    gridlines=True,
    gridlines_labels_on=("bottom", "left"),
    land=False,
    land_scale="110m",
    coastlines=False,
    bbox=True,
    **kwargs,
):
    """Decorate a cartopy axes

    Parameters
    ----------
    ax: cartopy.mpl.geoaxes.GeoAxes
    extent: None, list
        Extent ``[xmin, xmax, ymin, ymax]`` in degrees, as returned by
        :func:`xoa.geo.get_extent`
    transform: None, cartopy.crs.CRS
        Coordinate system of the extent, which defaults to ``PlateCarree``
    gridlines: bool
        Add labelled gridlines
    gridlines_labels_on: list(str)
        Sides on which labels are drawn
    land: bool
        Add the land with :func:`add_land`
    land_scale: str
    coastlines: bool
        Add the coastlines
    bbox: bool
        Keep the frame of the axes visible
    kwargs:
        Extra parameters starting with ``"gridlines_"`` are passed
        to :meth:`cartopy.mpl.geoaxes.GeoAxes.gridlines`.

    Return
    ------
    cartopy.mpl.geoaxes.GeoAxes
    """
    ccrs, _ = _import_cartopy_()
    if transform is None:
        transform = ccrs.PlateCarree()
    if extent is not None:
        ax.set_extent(extent, crs=transform)
    if not bbox:
        ax.spines["geo"].set_visible(False)
    if land:
        add_land(ax, scale=land_scale)
    if coastlines:
        ax.coastlines()
    if gridlines:
        kwgl = {
            "draw_labels": list(gridlines_labels_on),
            "linewidth": 0.3,
            "linestyle": "--",
            "color": "gray",
            "rotate_labels": False,
        }
        kwgl.update(misc.dict_filter(kwargs, "gridlines_"))
        gl = ax.gridlines(**kwgl)
        gl.xlabel_style = {"size": 7}
        gl.ylabel_style = {"size": 7}
    return ax


def create_base_map(
    extent,
    figsize=(8, 8),
    projection=None,
    transform=None,
    title=None,
    **kwargs,
):
    """Create a decorated cartopy map

    Parameters
    ----------
    extent: list
        Extent ``[xmin, xmax, ymin, ymax]`` in degrees, as returned by
        :func:`xoa.geo.get_extent`
    figsize: tuple
        Figure size in inches
    projection: None, cartopy.crs.CRS
        Projection of the map, which defaults to ``Mercator``
    transform: None, cartopy.crs.CRS
        Coordinate system of the extent, which defaults to ``PlateCarree``
    title: None, str
    kwargs:
        Extra parameters are passed to :func:`setup_map_axes`

    Return
    ------
    matplotlib.figure.Figure
    cartopy.mpl.geoaxes.GeoAxes
    """
    ccrs, _ = _import_cartopy_()
    if projection is None:
        projection = ccrs.Mercator()
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    setup_map_axes(ax, extent, transform, **kwargs)
    if title:
        ax.set_title(title)
    return fig, ax


#: Keys of the parameters of :func:`setup_map_axes` that may be passed to a function
#: that draws on existing axes
AX_SETUP_KEYS = ("gridlines", "gridlines_labels_on", "land", "land_scale", "coastlines", "bbox")


# %% Colorbars

#: Default shrink factor of colorbars
CBAR_SHRINK = 0.7


def add_colorbar(mappable, ax=None, **kwargs):
    """Add a colorbar that is shrunk by default

    Parameters
    ----------
    mappable: matplotlib.cm.ScalarMappable
        The plotted artist
    ax: None, matplotlib.axes.Axes, list(matplotlib.axes.Axes)
        Axes that are shrunk to make room for the colorbar, which is shared if there are
        several. It defaults to the axes of the artist.
    kwargs:
        Extra parameters are passed to :meth:`matplotlib.figure.Figure.colorbar`,
        like ``label``. The ``shrink`` parameter defaults to :data:`CBAR_SHRINK`.

    Return
    ------
    matplotlib.colorbar.Colorbar
    """
    if ax is None:
        ax = mappable.axes
    axes = np.ravel(ax)
    kwargs.setdefault("shrink", CBAR_SHRINK)
    return axes[0].figure.colorbar(mappable, ax=ax if np.ndim(ax) else axes[0], **kwargs)


# %% Meshes


#: Maximal number of grid lines used to measure the spacing in the automatic stride mode
_MAX_SPACING_LINES = 64


def get_strides(stride, lon, lat, ax=None, transform=None, min_spacing=12):
    """Get the under-sampling strides along y and x

    Parameters
    ----------
    stride: str, int, tuple(int)
        Either ``"auto"``, one positive integer for both directions or a ``(y, x)`` tuple.
        With ``"auto"``, it is the smallest value that keeps at least ``min_spacing`` pixels
        between consecutive centers along each direction. The spacing is measured on the axes,
        so it accounts for its size, the extent, the projection and the rotation of the grid.
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes in degrees
    ax: None, matplotlib.axes.Axes
        Axes that is needed in ``"auto"`` mode
    transform: None, cartopy.crs.CRS
        Coordinate system of ``lon`` and ``lat``, when the axes have a projection
    min_spacing: float
        Minimal spacing in pixels for the ``"auto"`` mode

    Return
    ------
    int
        Stride along y
    int
        Stride along x
    """
    if isinstance(stride, str) and stride == "auto":
        if ax is None:
            raise exceptions.XoaError("The axes are needed to compute an automatic stride")
        ax.apply_aspect()
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        stride = []
        for axis in 0, 1:
            if lon.shape[axis] < 2:
                stride.append(1)
                continue
            # Measure on a bounded number of lines, to be cheap with large grids
            step = max(1, lon.shape[1 - axis] // _MAX_SPACING_LINES)
            sub = (
                (slice(None, None, step), slice(None))
                if axis == 1
                else (slice(None), slice(None, None, step))
            )
            sublon, sublat = lon[sub], lat[sub]
            xy = np.stack([sublon, sublat], axis=-1)
            if hasattr(ax, "projection"):
                xy = ax.projection.transform_points(transform, sublon, sublat)[..., :2]
            pix = ax.transData.transform(xy.reshape(-1, 2)).reshape(xy.shape)
            spacing = np.nanmedian(np.linalg.norm(np.diff(pix, axis=axis), axis=-1))
            stride.append(1 if not spacing > 0 else max(1, int(np.ceil(min_spacing / spacing))))
    elif np.isscalar(stride):
        stride = (stride, stride)
    try:
        sy, sx = (int(s) for s in stride)
    except (TypeError, ValueError):
        sy = sx = 0
    if sy < 1 or sx < 1:
        raise exceptions.XoaError(
            "stride must be 'auto', a positive integer or a tuple of two positive integers"
        )
    return sy, sx


def get_sampled_indices(n, stride):
    """Get the indices from 0 to ``n-1`` every ``stride``, always keeping the last one"""
    return sorted(set(range(0, n, stride)) | {n - 1})


def get_mesh(lon, lat, stride=(1, 1)):
    """Get the edges and centers of a grid for plotting, possibly under-sampled

    Edges are computed from the centers with :func:`xoa.core.grid.centers2edges`.
    When the grid is under-sampled, the first and last edges are always kept, and the
    centers are the ones of the coarse cells that are delimited by the kept edges.

    Parameters
    ----------
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes of the centers in degrees
    stride: int, tuple(int)
        Under-sampling strides along y and x

    Return
    ------
    list(array(n, 2))
        Polylines of the kept edges as ``(lon, lat)`` points, the ones that are constant
        along y first and the ones that are constant along x next
    array
        Longitudes of the centers
    array
        Latitudes of the centers
    """
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    sy, sx = (stride, stride) if np.isscalar(stride) else stride
    elon = centers2edges(lon)
    elat = centers2edges(lat)
    rows = get_sampled_indices(elon.shape[0], sy)
    cols = get_sampled_indices(elon.shape[1], sx)
    elon = elon[np.ix_(rows, cols)]
    elat = elat[np.ix_(rows, cols)]
    segments = [np.column_stack([elon[j], elat[j]]) for j in range(elon.shape[0])]
    segments += [np.column_stack([elon[:, i], elat[:, i]]) for i in range(elon.shape[1])]
    if sy > 1 or sx > 1:
        clon = 0.25 * (elon[:-1, :-1] + elon[1:, :-1] + elon[:-1, 1:] + elon[1:, 1:])
        clat = 0.25 * (elat[:-1, :-1] + elat[1:, :-1] + elat[:-1, 1:] + elat[1:, 1:])
    else:
        clon, clat = lon, lat
    return segments, clon, clat


def plot_mesh(
    ax,
    lon,
    lat,
    stride="auto",
    min_spacing=12,
    edges=True,
    centers=True,
    transform=None,
    edge_kw=None,
    center_kw=None,
):
    """Plot the edges and centers of a grid

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    lon, lat: array_like(ny, nx)
        Longitudes and latitudes of the centers in degrees
    stride: str, int, tuple(int)
        Under-sampling strides. See :func:`get_strides`.
    min_spacing: float
        Minimal spacing in pixels for the ``"auto"`` stride
    edges, centers: bool
        Draw the edges and the centers
    transform: None, cartopy.crs.CRS
        Coordinate system of ``lon`` and ``lat``
    edge_kw: None, dict
        Parameters passed to :class:`matplotlib.collections.LineCollection`
    center_kw: None, dict
        Parameters passed to :meth:`matplotlib.axes.Axes.plot`

    Return
    ------
    matplotlib.collections.LineCollection, None
        The edges if drawn
    matplotlib.lines.Line2D, None
        The centers if drawn

    See also
    --------
    get_mesh
    get_strides
    """
    stride = get_strides(stride, lon, lat, ax, transform, min_spacing)
    segments, clon, clat = get_mesh(lon, lat, stride)
    kwtransform = {} if transform is None else {"transform": transform}
    collection = line = None
    if edges:
        kwedge = {"colors": "0.4", "linewidths": 0.5, "zorder": 2}
        kwedge.update(edge_kw or {})
        collection = mcollections.LineCollection(segments, **kwtransform, **kwedge)
        ax.add_collection(collection)
    if centers:
        kwcenter = {
            "color": "tab:red",
            "marker": ".",
            "markersize": 2,
            "linestyle": "",
            "zorder": 3,
        }
        kwcenter.update(center_kw or {})
        (line,) = ax.plot(clon.ravel(), clat.ravel(), **kwtransform, **kwcenter)
    return collection, line


# %% Sections and series


def plot_depth_section(ax, x, z, data, method="pcolormesh", invert_yaxis=False, **kwargs):
    """Plot a vertical section from arrays

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x, z, data: array_like(nz, nh)
        Horizontal coordinates, depths and values, that all have the vertical
        dimension first. With the ``"flat"`` shading of ``pcolormesh``, the values
        may have one point less along each dimension.
    method: {"pcolormesh", "contourf", "contour"}
        Plot method of :class:`matplotlib.axes.Axes`
    invert_yaxis: bool
        Invert the vertical axis, which is useful when depths are positive down
    kwargs:
        Extra parameters are passed to the plot method

    Return
    ------
    matplotlib.cm.ScalarMappable
    """
    x = np.asarray(x)
    z = np.asarray(z)
    data = np.asarray(data)
    if x.shape != z.shape or data.ndim != 2:
        raise exceptions.XoaError("x and z must have the same shape and data must be 2D")
    if method == "pcolormesh":
        kwargs.setdefault("shading", "nearest")
    mappable = getattr(ax, method)(x, z, data, **kwargs)
    if invert_yaxis and not ax.yaxis_inverted():
        ax.invert_yaxis()
    return mappable


def plot_sticks(ax, x, u, v, scale=None, color="steelblue", **kwargs):
    """Plot vectors as sticks anchored at ``y=0``

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    x: array_like(n)
        Positions along the horizontal axis
    u, v: array_like(n)
        Components of the vectors
    scale: None, float
        Scale passed to :meth:`matplotlib.axes.Axes.quiver`: the smaller, the longer
        the sticks.
    color: color
    kwargs:
        Extra parameters are passed to :meth:`matplotlib.axes.Axes.quiver`

    Return
    ------
    matplotlib.quiver.Quiver
    """
    u = np.asarray(u)
    kwargs.setdefault("headaxislength", 0)
    kwargs.setdefault("headlength", 0)
    kwargs.setdefault("headwidth", 1)
    kwargs.setdefault("width", 0.002)
    quiver = ax.quiver(
        np.asarray(x), np.zeros(u.shape), u, np.asarray(v), scale=scale, color=color, **kwargs
    )
    ax.axhline(0, color="0.5", linewidth=0.5)
    ax.get_yaxis().set_visible(False)
    return quiver


# %% Taylor diagrams

#: Default ticks of the correlation axis
TAYLOR_CORR_TICKS = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1)


class TaylorDiagram:
    """A Taylor diagram on a polar axes

    The angle is the arc cosine of the correlation and the radius is the standard
    deviation. The reference is on the horizontal axis at ``ref_std``, and the distance
    to it is the centered root mean square difference, which is shown by contours.
    The diagram is a quarter of circle, or a half circle when there are negative
    correlations.

    Parameters
    ----------
    fig: None, matplotlib.figure.Figure
        A new figure by default
    subplot: int, matplotlib.gridspec.SubplotSpec
        Position of the axes in the figure
    ref_std: float
        Standard deviation of the reference, which is 1 for normalized data
    rmax: None, float
        Maximal standard deviation
    negative: bool
        Show negative correlations
    corr_ticks: None, array_like
        Positive correlations where ticks are drawn. The negative ones are mirrored.
    rms_levels: None, False, array_like
        Levels of the contours of centered root mean square difference.
        Automatic by default and no contours when ``False``.
    ref_label: None, str
        Label of the reference marker in the legend. No legend entry when None.
    std_label, corr_label: str
        Axis labels
    grid_kwargs: None, dict
        Parameters of :meth:`matplotlib.axes.Axes.grid`
    contour_kwargs: None, dict
        Parameters of :meth:`matplotlib.axes.Axes.contour`
    ref_kwargs: None, dict
        Parameters of :meth:`matplotlib.axes.Axes.plot` for the reference marker
        and the arc of constant standard deviation.
        Use ``arc=False`` to hide the arc.

    Attributes
    ----------
    ax: matplotlib.projections.polar.PolarAxes
    artists: list
        Artists of the points that are added
    legend: None, matplotlib.legend.Legend
    colorbar: None, matplotlib.colorbar.Colorbar
    """

    def __init__(
        self,
        fig=None,
        subplot=111,
        ref_std=1.0,
        rmax=None,
        negative=False,
        corr_ticks=None,
        rms_levels=None,
        ref_label="Reference",
        std_label="Standard deviation",
        corr_label="Correlation",
        grid_kwargs=None,
        contour_kwargs=None,
        ref_kwargs=None,
    ):
        self.ref_std = float(ref_std)
        self.rmax = float(rmax) if rmax is not None else 1.25 * self.ref_std
        self.negative = bool(negative)
        self.thetamax = np.pi if negative else 0.5 * np.pi
        self.artists = []
        self.legend = None
        self.colorbar = None
        if fig is None:
            fig = plt.figure(figsize=(7, 5))
        ax = self.ax = fig.add_subplot(subplot, projection="polar")
        ax.set_thetamin(0)
        ax.set_thetamax(np.degrees(self.thetamax))
        ax.set_ylim(0, self.rmax)

        # Correlation axis
        ticks = np.asarray(TAYLOR_CORR_TICKS if corr_ticks is None else corr_ticks, dtype="d")
        if negative:
            ticks = np.unique(np.concatenate([ticks, -ticks]))
        ax.set_xticks(np.arccos(ticks))
        ax.set_xticklabels([f"{t + 0.0:g}" for t in ticks])
        ax.text(
            0.5 * self.thetamax,
            1.2 * self.rmax,
            corr_label,
            rotation=np.degrees(0.5 * self.thetamax) - 90,
            ha="center",
            va="center",
        )
        ax.annotate(
            std_label,
            (0, 0.5 * self.rmax),
            xytext=(0, -26),
            textcoords="offset points",
            ha="center",
            va="top",
            annotation_clip=False,
        )
        ax.grid(**{"linestyle": ":", "alpha": 0.6, **(grid_kwargs or {})})

        # Contours of centered rms difference
        if rms_levels is not False:
            theta = np.linspace(0, self.thetamax, 181)
            radius = np.linspace(0, self.rmax, 101)
            rr, tt = np.meshgrid(radius, theta)
            dist = np.sqrt(rr**2 + self.ref_std**2 - 2 * rr * self.ref_std * np.cos(tt))
            if rms_levels is None:
                rms_levels = mticker.MaxNLocator(nbins=5, prune="both").tick_values(0, dist.max())
                rms_levels = [lev for lev in rms_levels if 0 < lev < dist.max()]
            if len(rms_levels):
                kw = {"colors": "0.55", "linestyles": "--", "linewidths": 0.7}
                kw.update(contour_kwargs or {})
                cs = ax.contour(tt, rr, dist, levels=rms_levels, **kw)
                ax.clabel(cs, fmt="%g", fontsize=7)

        # Reference
        kw = {
            "marker": "*",
            "color": "k",
            "markersize": 12,
            "linestyle": "none",
            "zorder": 5,
            "clip_on": False,
        }
        kw.update(ref_kwargs or {})
        arc = kw.pop("arc", True)
        if arc:
            ax.plot(
                np.linspace(0, self.thetamax, 181),
                np.full(181, self.ref_std),
                color=kw["color"],
                linestyle="--",
                linewidth=0.8,
            )
        ax.plot([0], [self.ref_std], label=ref_label or "_nolegend_", **kw)

    def add_points(
        self,
        std,
        corr,
        labels=None,
        values=None,
        markers="o",
        colors=None,
        cmap=None,
        vmin=None,
        vmax=None,
        legend=True,
        colorbar=True,
        legend_kwargs=None,
        cbar_kwargs=None,
        **kwargs,
    ):
        """Add points to the diagram

        Parameters
        ----------
        std, corr: array_like(n)
            Standard deviations and correlations
        labels: None, list(str)
            One label per point, added to the legend. With ``values``, they are
            written next to the points.
        values: None, array_like(n)
            Values that color the markers, with a colorbar.
        markers: str, list(str)
            One marker for all the points, or one per point, cycled if shorter.
        colors: None, color, list
            Colors when no values are given. They follow the color cycle by default.
        cmap, vmin, vmax:
            Colormap and its limits when values are given
        legend, colorbar: bool
            Add a legend (when there are labels and no values) or a colorbar
            (when there are values)
        legend_kwargs, cbar_kwargs: None, dict
            Parameters of :meth:`matplotlib.axes.Axes.legend` and
            :func:`add_colorbar`
        kwargs:
            Extra parameters are passed to :meth:`matplotlib.axes.Axes.plot` or
            :meth:`matplotlib.axes.Axes.scatter`

        Return
        ------
        list
            The artists, one per point without values and one per marker with values
        """
        std = np.atleast_1d(np.asarray(std, dtype="d"))
        corr = np.atleast_1d(np.asarray(corr, dtype="d"))
        n = len(std)
        if corr.shape != std.shape:
            raise exceptions.XoaError("std and corr must have the same shape")
        if (corr < -1).any() or (corr > 1).any():
            raise exceptions.XoaError("Correlations must be between -1 and 1")
        if (corr < 0).any() and not self.negative:
            raise exceptions.XoaError(
                "There are negative correlations: create the diagram with negative=True"
            )
        if labels is not None and len(labels) != n:
            raise exceptions.XoaError("labels must have one item per point")
        if values is not None:
            values = np.asarray(values, dtype="d")
            if values.shape != std.shape:
                raise exceptions.XoaError("values must have one item per point")
        markers = [markers] if isinstance(markers, str) else list(markers)
        markers = [markers[i % len(markers)] for i in range(n)]
        theta = np.arccos(corr)
        ax = self.ax
        artists = []

        if values is None:
            if colors is None:
                cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                colors = [cycle[i % len(cycle)] for i in range(n)]
            elif isinstance(colors, str) or not np.iterable(colors):
                colors = [colors] * n
            kw = {"linestyle": "none", "markersize": 8, "zorder": 6, "clip_on": False}
            kw.update(kwargs)
            for i in range(n):
                label = labels[i] if labels is not None else "_nolegend_"
                artists.extend(
                    ax.plot(
                        [theta[i]],
                        [std[i]],
                        marker=markers[i],
                        color=colors[i % len(colors)],
                        label=label,
                        **kw,
                    )
                )
        else:
            vmin = np.nanmin(values) if vmin is None else vmin
            vmax = np.nanmax(values) if vmax is None else vmax
            kw = {"s": 60, "zorder": 6, "edgecolors": "k", "linewidths": 0.5}
            kw.update(kwargs)
            for marker in dict.fromkeys(markers):
                idx = [i for i in range(n) if markers[i] == marker]
                artists.append(
                    ax.scatter(
                        theta[idx],
                        std[idx],
                        c=values[idx],
                        marker=marker,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                        **kw,
                    )
                )
            if labels is not None:
                for i in range(n):
                    ax.annotate(
                        labels[i],
                        (theta[i], std[i]),
                        xytext=(5, 5),
                        textcoords="offset points",
                        fontsize=8,
                    )
            if colorbar:
                self.colorbar = add_colorbar(artists[0], ax, **(cbar_kwargs or {}))
        self.artists.extend(artists)

        if legend and values is None and labels is not None:
            self.add_legend(**(legend_kwargs or {}))
        return artists

    def add_legend(self, **kwargs):
        """Add the legend outside of the diagram, on its right"""
        kw = {"loc": "upper left", "bbox_to_anchor": (1.04, 1.0), "frameon": False}
        kw.update(kwargs)
        self.legend = self.ax.legend(**kw)
        return self.legend


def plot_taylor(std, corr, ref_std=1.0, diagram=None, **kwargs):
    """Plot points in a Taylor diagram

    Parameters
    ----------
    std, corr: array_like(n)
        Standard deviations and correlations of ``n`` points.
        The correlations may be negative.
    ref_std: float
        Standard deviation of the reference. The default of 1 is for data
        that are already normalized by the reference.
    diagram: None, TaylorDiagram
        Existing diagram. It is created when not provided, with a half circle
        if there are negative correlations and a maximal standard deviation
        that fits the points.
    kwargs:
        Parameters of :class:`TaylorDiagram` when the diagram is created, or
        of :meth:`TaylorDiagram.add_points`, like ``labels``, ``values``
        and ``markers``.

    Return
    ------
    TaylorDiagram
        The diagram, which holds the axes in its ``ax`` attribute and the plotted artists

    Example
    -------
    .. code-block:: python

        plot_taylor([0.9, 1.2, 0.7], [0.95, 0.8, -0.3], labels=["a", "b", "c"],
                    markers=["o", "s", "^"])
    """
    std = np.atleast_1d(np.asarray(std, dtype="d"))
    corr = np.atleast_1d(np.asarray(corr, dtype="d"))
    if diagram is None:
        init = {key: kwargs.pop(key) for key in list(kwargs) if key in TAYLOR_INIT_KEYS}
        init.setdefault("negative", bool((corr < 0).any()))
        if "rmax" not in init:
            init["rmax"] = 1.25 * max(np.nanmax(std), ref_std)
        diagram = TaylorDiagram(ref_std=ref_std, **init)
    diagram.add_points(std, corr, **kwargs)
    return diagram


#: Parameters of :class:`TaylorDiagram` that can be passed to :func:`plot_taylor`
TAYLOR_INIT_KEYS = (
    "fig",
    "subplot",
    "rmax",
    "negative",
    "corr_ticks",
    "rms_levels",
    "ref_label",
    "std_label",
    "corr_label",
    "grid_kwargs",
    "contour_kwargs",
    "ref_kwargs",
)
