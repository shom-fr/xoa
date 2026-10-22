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
import inspect

import matplotlib.collections as mcollections
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import numpy as np

from .. import exceptions
from .. import misc
from .geo import EARTH_RADIUS, haversine
from .grid import centers2edges
from .stats import rose_hist

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


#: Short names of the projections
PROJ_ALIASES = {"merc": "mercator", "pc": "platecarree", "ortho": "orthographic"}


def get_projection(projection=None, extent=None):
    """Get a cartopy projection from its name or alias

    Parameters
    ----------
    projection: None, str, cartopy.crs.Projection
        A projection instance, which is returned as is, or the lower case name of a
        :mod:`cartopy.crs` projection class that has no mandatory parameter,
        like ``"mercator"``, ``"platecarree"``, ``"orthographic"``, ``"stereographic"``
        or ``"lambertazimuthalequalarea"``.
        The aliases ``"merc"``, ``"pc"`` and ``"ortho"`` are also accepted.
        The default is :class:`cartopy.crs.Mercator`.
        The orthographic projection shows the Earth as seen from above a point,
        which limits the deformation of the cells of regional grids.
    extent: None, list
        Extent ``[xmin, xmax, ymin, ymax]`` in degrees, whose center is used as the
        ``central_longitude`` and ``central_latitude`` of the projections that accept
        both, like the orthographic one.

    Return
    ------
    cartopy.crs.Projection
    """
    ccrs, _ = _import_cartopy_()
    if projection is None:
        projection = "mercator"
    if not isinstance(projection, str):
        return projection
    classes = {
        name.lower(): cls
        for name, cls in vars(ccrs).items()
        if isinstance(cls, type)
        and issubclass(cls, ccrs.Projection)
        and cls is not ccrs.Projection
        and not name.startswith("_")
    }
    name = PROJ_ALIASES.get(projection.lower(), projection.lower())
    if name not in classes:
        raise exceptions.XoaError(
            f"Invalid projection name: {projection}. Choose among: "
            + ", ".join(sorted(classes))
            + ", or one of the aliases: "
            + ", ".join(PROJ_ALIASES)
        )
    cls = classes[name]
    kw = {}
    params = inspect.signature(cls.__init__).parameters
    if extent is not None and "central_longitude" in params and "central_latitude" in params:
        kw = dict(
            central_longitude=0.5 * (extent[0] + extent[1]),
            central_latitude=0.5 * (extent[2] + extent[3]),
        )
    try:
        return cls(**kw)
    except TypeError as exc:
        raise exceptions.XoaError(
            f"The {cls.__name__} projection has mandatory parameters: "
            "create an instance and pass it instead of its name."
        ) from exc


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


def decorate_map(
    ax,
    extent=None,
    transform=None,
    gridlines=True,
    gridlines_labels_on=("bottom", "left"),
    land=False,
    land_scale="110m",
    coastlines=False,
    bbox=True,
    fancy=False,
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
    fancy: bool, dict
        Add a black and white frame with :func:`add_fancy_map_frame`.
        A dict gives its parameters.
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
    if fancy:
        add_fancy_map_frame(ax, **(fancy if isinstance(fancy, dict) else {}))
    return ax


def create_map(
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
    projection: None, str, cartopy.crs.CRS
        Projection of the map, or its name as accepted by :func:`get_projection`,
        which defaults to ``Mercator``
    transform: None, cartopy.crs.CRS
        Coordinate system of the extent, which defaults to ``PlateCarree``
    title: None, str
    kwargs:
        Extra parameters are passed to :func:`decorate_map`

    Return
    ------
    matplotlib.figure.Figure
    cartopy.mpl.geoaxes.GeoAxes
    """
    projection = get_projection(projection, extent)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection=projection)
    decorate_map(ax, extent, transform, **kwargs)
    if title:
        ax.set_title(title)
    return fig, ax


#: Former names of :func:`create_map` and :func:`decorate_map`
create_base_map = create_map
setup_map_axes = decorate_map

#: Keys of the parameters of :func:`decorate_map` that may be passed to a function
#: that draws on existing axes
AX_SETUP_KEYS = (
    "gridlines",
    "gridlines_labels_on",
    "land",
    "land_scale",
    "coastlines",
    "bbox",
    "fancy",
)


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
    """Get evenly spread indices from 0 to ``n-1``, about every ``stride``

    The first and last ones are always kept, and the number of points between two consecutive
    indices differs by at most one, so that there is no thin cell at the end.
    """
    nseg = max(1, int(np.ceil((n - 1) / stride)))
    return np.unique(np.round(np.linspace(0, n - 1, nseg + 1)).astype(int)).tolist()


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


def get_quiverkey_value(speed, percentile=80):
    """Get a round reference value for a quiver key

    It is the closest one, in a logarithmic sense, to a high percentile of the speeds
    among 1, 2 and 5 times a power of ten, so that the reference arrow is as long
    as the strong arrows of the plot.

    Parameters
    ----------
    speed: array_like
        Speeds, with possibly missing values
    percentile: float
        Percentile of the positive speeds that is rounded

    Return
    ------
    float
        ``1.0`` when there is no positive speed
    """
    speed = np.asarray(speed, dtype=float)
    speed = speed[np.isfinite(speed) & (speed > 0)]
    if not speed.size:
        return 1.0
    ref = np.percentile(speed, percentile)
    mag = 10 ** np.floor(np.log10(ref))
    return float(min((1, 2, 5, 10), key=lambda m: abs(np.log(m * mag / ref))) * mag)


def add_quiverkey(ax, quiver, value, units="", pos=(0.0, 1.03), **kwargs):
    """Add a reference arrow, labelled with its value, above the left corner of the axes

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    quiver: matplotlib.quiver.Quiver
    value: float
        Reference value, see :func:`get_quiverkey_value`
    units: str
        Units that are appended to the label
    pos: tuple(float)
        Position of the start of the arrow in axes coordinates,
        which is above the left corner by default
    kwargs:
        Extra parameters are passed to :meth:`matplotlib.axes.Axes.quiverkey`

    Return
    ------
    matplotlib.quiver.QuiverKey
    """
    kwargs.setdefault("labelpos", "E")
    kwargs.setdefault("coordinates", "axes")
    kwargs.setdefault("color", "k")
    kwargs.setdefault("labelcolor", "k")
    kwargs.setdefault("fontproperties", {"size": "medium", "weight": "bold"})
    qkey = ax.quiverkey(quiver, *pos, value, f"{value:g} {units}".strip(), **kwargs)

    # The arrow ends at the position with an eastern label: shift it to make it start there
    if qkey.labelpos == "E" and qkey.coord == "axes":
        try:
            ax.apply_aspect()
            qkey._init()
            xverts = np.asarray(qkey.verts[0])[:, 0]
            trans = qkey.vector.get_transform()
            width = trans.transform([[xverts.max(), 0]])[0, 0] - trans.transform([[xverts.min(), 0]])[0, 0]
            qkey.X = pos[0] + width / ax.bbox.width
        except (AttributeError, IndexError, ZeroDivisionError):  # pragma: no cover
            pass

    # Make the key count in the tight bounding box of the figure, which is not the case by default
    def get_window_extent(renderer=None):
        xy = qkey.get_transform().transform([(pos[0], pos[1])])
        return mtransforms.Bbox.union([qkey.text.get_window_extent(renderer), mtransforms.Bbox([xy[0], xy[0]])])

    qkey.get_window_extent = get_window_extent
    qkey.set_clip_on(False)
    return qkey


# %% Scale bar and compass

#: Length units of the scale bar in meters
SCALEBAR_UNITS = {"m": 1.0, "km": 1e3, "nmi": 1852.0}

_LOC_ANCHORS = {
    "lower left": (0, 0),
    "lower right": (1, 0),
    "upper left": (0, 1),
    "upper right": (1, 1),
}


def _get_anchor_(loc, pad):
    """Position in axes coordinates and signs of the corner ``loc``

    ``pad`` is a float or a pair of floats along x and y.
    """
    if loc not in _LOC_ANCHORS:
        raise exceptions.XoaError(f"Invalid loc '{loc}'. Choose among: {', '.join(_LOC_ANCHORS)}")
    ix, iy = _LOC_ANCHORS[loc]
    padx, pady = np.broadcast_to(pad, 2)
    return (padx if ix == 0 else 1 - padx), (pady if iy == 0 else 1 - pady), 1 - 2 * ix, 1 - 2 * iy


def _axes_to_lonlat_(ax, xy):
    """Longitudes and latitudes of points given in axes coordinates"""
    ccrs, _ = _import_cartopy_()
    ax.apply_aspect()
    x, y = (ax.transAxes + ax.transData.inverted()).transform(np.atleast_2d(xy)).T
    lon, lat = ccrs.Geodetic().transform_points(ax.projection, x, y)[:, :2].T
    return lon, lat


def _get_blur_effects_(kw):
    """Path effects of a soft shadow or glow, made of a few strokes of increasing widths"""
    nlayer = 4
    return [
        mpatheffects.PathPatchEffect(
            offset=kw["offset"],
            facecolor=kw["color"],
            edgecolor=kw["color"],
            linewidth=kw["width"] * (i + 1) / nlayer,
            alpha=kw["alpha"] / nlayer,
        )
        for i in reversed(range(nlayer))
    ]


#: Default parameters of the effects of the map decorations
EFFECT_DEFAULTS = {
    "shadow": {"offset": (1, -1), "alpha": 0.3, "width": 2.0, "color": "k"},
    "glow": {"offset": (0, 0), "alpha": 1.0, "width": 3.0, "color": "w"},
}


def _get_effect_names_(effect):
    """Names of the effects, in the drawing order, from True, a name or a sequence of names"""
    if effect is None or effect is False:
        return []
    names = ["shadow"] if effect is True else [effect] if isinstance(effect, str) else effect
    for name in names:
        if name not in EFFECT_DEFAULTS:
            raise exceptions.XoaError(
                f"Invalid effect '{name}'. Choose among: {', '.join(EFFECT_DEFAULTS)}"
            )
    return [name for name in EFFECT_DEFAULTS if name in names]


def _add_effects_(ax, artists, effect, effect_kw, zorder):
    """Add a soft drop shadow and/or a glow below artists with path effects

    The effects of a patch are drawn by an invisible copy placed below all the artists,
    so that they do not hide the neighbor patches.
    """
    effects = []
    for name in _get_effect_names_(effect):
        kw = dict(EFFECT_DEFAULTS[name])
        effect_kw = effect_kw or {}
        kw.update(effect_kw.get(name, {}) if set(effect_kw) & set(EFFECT_DEFAULTS) else effect_kw)
        effects += _get_blur_effects_(kw)
    if not effects:
        return
    for artist in artists:
        if isinstance(artist, mpatches.Patch):
            under = mpatches.PathPatch(
                artist.get_path(),
                transform=artist.get_transform(),
                fill=False,
                edgecolor="none",
                zorder=zorder - 0.5,
                clip_on=False,
            )
            ax.add_patch(under)
            under.set_path_effects(effects)
        else:
            artist.set_path_effects(effects + [mpatheffects.Normal()])


def get_scalebar_length(max_length):
    """Get the longest round length that is not above a maximal length

    It is 1, 2 or 5 times a power of ten.
    """
    mag = 10 ** np.floor(np.log10(max_length))
    return float(max(m for m in (1, 2, 5, 10) if m * mag <= max_length) * mag)


def add_scalebar(
    ax,
    length=None,
    units="km",
    loc="lower left",
    pad=0.04,
    segments=2,
    height=0.012,
    color="k",
    effect=None,
    effect_kw=None,
    zorder=10,
    **kwargs,
):
    """Add a scale bar to a map

    The length of the bar is the ground distance along the horizontal direction of the axes
    at the position of the bar, so it accounts for the distortion of the projection.
    It is computed when the function is called: call it after the extent is set.

    Parameters
    ----------
    ax: cartopy.mpl.geoaxes.GeoAxes
    length: None, float
        Length of the bar in ``units``. It defaults to a round value, see
        :func:`get_scalebar_length`, that is at most a fifth of the width of the axes.
    units: {"m", "km", "nmi"}
    loc: {"lower left", "lower right", "upper left", "upper right"}
        Corner of the axes where the bar is placed
    pad: float
        Distance to the corner in axes coordinates
    segments: int
        Number of alternating black and white segments
    height: float
        Height of the bar in axes coordinates
    color: color
        Color of the dark segments, the labels and the frame
    effect: None, bool, str, list(str)
        Add a ``"shadow"``, a ``"glow"`` or both, which are drawn below the artists
        to make them stand out. True is a shorthand for the shadow.
    effect_kw: None, dict
        Parameters of the effects: the blur ``width`` in points, the ``alpha``,
        the ``color`` and, for the shadow, the ``offset`` in points. Give a dict per effect
        to set them separately, like ``{"glow": {"color": "y"}}``.
    zorder: float
    kwargs:
        Extra parameters are passed to :class:`matplotlib.text.Text`, like ``fontsize``.

    Return
    ------
    list
        The artists: the segments and the label
    """
    if units not in SCALEBAR_UNITS:
        raise exceptions.XoaError(
            f"Invalid units '{units}'. Choose among: {', '.join(SCALEBAR_UNITS)}"
        )
    x0, y0, sx, sy = _get_anchor_(loc, pad)

    # Meters per unit of axes width at the position of the bar
    lon, lat = _axes_to_lonlat_(ax, [(x0, y0), (x0 + 0.01, y0)])
    meters = haversine(lon[0], lat[0], lon[1], lat[1]) * EARTH_RADIUS * 100  # per axes width
    scale = SCALEBAR_UNITS[units]
    if length is None:
        length = get_scalebar_length(0.2 * meters / scale)
    width = length * scale / meters
    xleft = x0 if sx > 0 else x0 - width
    ybar = y0 if sy > 0 else y0 - height - 0.04  # room for the label under the corner

    artists = []
    dx = width / segments
    for i in range(segments):
        artists.append(
            ax.add_patch(
                mpatches.Rectangle(
                    (xleft + i * dx, ybar),
                    dx,
                    height,
                    transform=ax.transAxes,
                    facecolor=color if i % 2 == 0 else "w",
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=zorder,
                    clip_on=False,
                )
            )
        )
    kwargs = {"fontsize": "small", "fontweight": "bold", "color": color, **kwargs}
    artists.append(
        ax.text(
            x0,
            ybar + height + 0.008,
            f"{length:g} {units}",
            transform=ax.transAxes,
            ha="left" if sx > 0 else "right",
            va="bottom",
            zorder=zorder,
            **kwargs,
        )
    )
    _add_effects_(ax, artists, effect, effect_kw, zorder)
    return artists


def _compass_polygons_(style):
    """Polygons of the compass styles pointing to the north, as (vertices, light fill)"""
    if style == "simple":
        return [([(0, 1), (-0.35, -0.7), (0, -0.35), (0.35, -0.7)], False)]
    if style == "fancy":
        polys = []
        for k, length in enumerate((1.0, 0.6, 0.6, 0.6)):
            a = np.radians(90.0 * k)  # clockwise from north
            tip = (length * np.sin(a), length * np.cos(a))
            for side, light in ((-1, False), (1, True)):
                b = a + side * np.radians(45.0)
                polys.append(([tip, (0, 0), (0.22 * np.sin(b), 0.22 * np.cos(b))], light))
        return polys
    raise exceptions.XoaError(f"Invalid style '{style}'. Choose among: simple, fancy")


def add_compass(
    ax,
    loc="upper right",
    size=40,
    style="simple",
    pad=None,
    color="k",
    effect=None,
    effect_kw=None,
    zorder=10,
    **kwargs,
):
    """Add a compass to a map

    The north points to the direction of the meridian at the position of the compass,
    which differs from the vertical in some projections.
    The angle is computed when the function is called.

    Parameters
    ----------
    ax: cartopy.mpl.geoaxes.GeoAxes
    loc: {"lower left", "lower right", "upper left", "upper right"}
        Corner of the axes where the compass is placed
    size: float
        Height of the compass in points
    style: {"simple", "fancy"}
        A single arrow, or a rose with four split points and a ring
    pad: None, float
        Distance of the center to the corner in axes coordinates. By default, it leaves
        room for the arrow and the "N" inside the axes.
    color: color
        Color of the dark parts and of the "N"
    effect: None, bool, str, list(str)
        Add a ``"shadow"``, a ``"glow"`` or both, which are drawn below the artists
        to make them stand out. True is a shorthand for the shadow.
    effect_kw: None, dict
        Parameters of the effects: the blur ``width`` in points, the ``alpha``,
        the ``color`` and, for the shadow, the ``offset`` in points. Give a dict per effect
        to set them separately, like ``{"glow": {"color": "y"}}``.
    zorder: float
    kwargs:
        Extra parameters are passed to :class:`matplotlib.text.Text`, like ``fontsize``.

    Return
    ------
    list
        The artists: the polygons, the ring of the fancy style and the "N"
    """
    polygons = _compass_polygons_(style)
    if pad is None:
        ax.apply_aspect()
        margin = 0.625 * size + 12  # points: the arrow, the "N" and some room
        bbox = ax.get_window_extent()
        pad = margin / (np.array([bbox.width, bbox.height]) * 72.0 / ax.figure.dpi)
    x0, y0, _, _ = _get_anchor_(loc, pad)

    # Angle of the north with respect to the vertical of the display, clockwise
    lon, lat = _axes_to_lonlat_(ax, [(x0, y0)])
    ccrs, _ = _import_cartopy_()
    xy = ax.projection.transform_points(
        ccrs.Geodetic(), np.array([lon[0]] * 2), np.array([lat[0], lat[0] + 0.01])
    )[:, :2]
    dxy = np.diff(ax.transData.transform(xy), axis=0)[0]
    angle = np.degrees(np.arctan2(dxy[0], dxy[1]))

    radius = 0.5 * size  # points
    trans = (
        mtransforms.Affine2D().rotate_deg(-angle).scale(radius / 72.0)
        + ax.figure.dpi_scale_trans
        + mtransforms.ScaledTranslation(x0, y0, ax.transAxes)
    )
    artists = []
    for verts, light in polygons:
        artists.append(
            ax.add_patch(
                mpatches.Polygon(
                    verts,
                    closed=True,
                    transform=trans,
                    facecolor="w" if light else color,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=zorder,
                    clip_on=False,
                )
            )
        )
    if style == "fancy":
        artists.append(
            ax.add_patch(
                mpatches.Circle(
                    (0, 0),
                    0.45,
                    transform=trans,
                    fill=False,
                    edgecolor=color,
                    linewidth=0.8,
                    zorder=zorder,
                    clip_on=False,
                )
            )
        )
    kwargs = {"fontsize": "small", "fontweight": "bold", "color": color, **kwargs}
    rn = radius * 1.25  # label beyond the tip of the north arrow
    artists.append(
        ax.annotate(
            "N",
            (x0, y0),
            xycoords="axes fraction",
            xytext=(rn * np.sin(np.radians(angle)), rn * np.cos(np.radians(angle))),
            textcoords="offset points",
            ha="center",
            va="center",
            zorder=zorder,
            **kwargs,
        )
    )
    _add_effects_(ax, artists, effect, effect_kw, zorder)
    return artists


#: Sides of a map frame: axis along which it runs, position in axes coordinates, outward direction
_FRAME_SIDES = {
    "bottom": (0, 0.0, (0, -1)),
    "top": (0, 1.0, (0, 1)),
    "left": (1, 0.0, (-1, 0)),
    "right": (1, 1.0, (1, 0)),
}


def _lonlat_to_axes_(ax, lon, lat):
    """Axes coordinates of points given in degrees"""
    ccrs, _ = _import_cartopy_()
    x, y = ax.projection.transform_points(ccrs.Geodetic(), lon, lat)[:, :2].T
    return (ax.transData + ax.transAxes.inverted()).transform(np.array([x, y]).T)


def _get_frame_gridliners_(ax):
    """Gridliners of an axes, created if needed, with the cartopy longitude and latitude locators

    A gridliner that is created has no line and no label.
    """
    from cartopy.mpl.gridliner import Gridliner, classic_locator
    from cartopy.mpl.ticker import LatitudeLocator, LongitudeLocator

    gridliners = [a for a in ax.artists if isinstance(a, Gridliner)]
    if not gridliners:
        gl = ax.gridlines()
        gl.xlines = gl.ylines = False
        gridliners = [gl]
    for gl in gridliners:  # Not PlateCarree gridliners get the generic locator
        if gl.xlocator is classic_locator:
            gl.xlocator = LongitudeLocator()
        if gl.ylocator is classic_locator:
            gl.ylocator = LatitudeLocator()
    return gridliners


def _get_frame_ticks_(ax, side, ticks):
    """Positions of the ticks along a side, in axes coordinates, and the extent of the side"""
    iaxis = _FRAME_SIDES[side][0]
    edge = np.linspace(0, 1, 101)
    ends = []
    for fixed in (0.0, 1.0):
        xy = np.array([edge, np.full_like(edge, fixed)]).T
        lon, lat = _axes_to_lonlat_(ax, xy if iaxis == 0 else xy[:, ::-1])
        ends.append(np.degrees(np.unwrap(np.radians(lon))) if iaxis == 0 else lat)
    if not np.allclose(*ends, atol=1e-6 * max(1.0, np.abs(ends[0]).max())):
        raise exceptions.XoaError(
            "A fancy frame needs a projection whose meridians and parallels are "
            "parallel to the edges of the map, like PlateCarree or Mercator."
        )
    vmin, vmax = ends[0].min(), ends[0].max()
    if ticks is None:
        gl = _get_frame_gridliners_(ax)[-1]
        ticks = (gl.xlocator, gl.ylocator)[iaxis]
    ticks = np.asarray(ticks.tick_values(vmin, vmax) if hasattr(ticks, "tick_values") else ticks)
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

    lon_mid, lat_mid = _axes_to_lonlat_(ax, np.array([[0.5, 0.5]]))
    ones = np.ones_like(ticks)
    pos = _lonlat_to_axes_(
        ax,
        ticks if iaxis == 0 else lon_mid * ones,
        lat_mid * ones if iaxis == 0 else ticks,
    )[:, iaxis]
    return np.sort(pos[(pos > 1e-6) & (pos < 1 - 1e-6)])


def add_fancy_map_frame(
    ax,
    width=5.0,
    sides=("bottom", "left", "top", "right"),
    colors=("k", "w"),
    xticks=None,
    yticks=None,
    inside=False,
    linewidth=0.6,
    edgecolor="k",
    zorder=5,
):
    """Add a black and white alternating frame to a map, like GMT does

    The frame is cut at the longitudes and latitudes of the gridlines, which are created
    without lines nor labels if the axes has none. Their locators are the cartopy
    longitude and latitude ones, for all projections.
    It requires a projection whose meridians and parallels are parallel
    to the edges of the map, like ``PlateCarree`` or ``Mercator``.
    Existing tick labels are moved away by the width of the frame.

    Parameters
    ----------
    ax: cartopy.mpl.geoaxes.GeoAxes
    width: float
        Width of the frame in points
    sides: list(str)
        Sides to decorate among "bottom", "left", "top" and "right"
    colors: tuple
        Colors of the first and second alternating segments
    xticks, yticks: None, array_like, matplotlib.ticker.Locator
        Longitudes and latitudes where the frame is cut.
        They default to the locators of the gridlines of the axes.
    inside: bool
        Draw the frame inside the map, rather than outside
    linewidth, edgecolor:
        Properties of the thin lines that outline the frame
    zorder: float

    Return
    ------
    list(matplotlib.artist.Artist)
        The artists of the frame

    Example
    -------
    .. code-block:: python

        ax = create_map([-8, 0, 46, 50], gridlines=False)[1]
        add_fancy_map_frame(ax, width=6)

    See also
    --------
    decorate_map
    """
    sides = (sides,) if isinstance(sides, str) else tuple(sides)
    for side in sides:
        if side not in _FRAME_SIDES:
            raise exceptions.XoaError(
                f"Invalid side '{side}'. Choose among: {', '.join(_FRAME_SIDES)}"
            )
    ax.apply_aspect()
    fig = ax.figure
    sign = -1 if inside else 1

    def get_transform(dx, dy):
        """Axes coordinates shifted by (dx, dy) points"""
        return ax.transAxes + mtransforms.ScaledTranslation(dx / 72, dy / 72, fig.dpi_scale_trans)

    def style(artist):
        artist.set_clip_on(False)
        artist.set_zorder(zorder)
        return ax.add_artist(artist)

    artists = []
    for side in sides:
        iaxis, fixed, out = _FRAME_SIDES[side]
        pos = _get_frame_ticks_(ax, side, (xticks, yticks)[iaxis])
        bounds = np.concatenate([[0.0], np.sort(pos), [1.0]])

        def seg(a, b):
            return [(a, fixed), (b, fixed)] if iaxis == 0 else [(fixed, a), (fixed, b)]

        dx, dy = sign * width * np.array(out)
        mid = get_transform(dx / 2, dy / 2)
        artists.append(
            style(
                mcollections.LineCollection(
                    [seg(0, 1)], colors=colors[0], linewidths=width, transform=mid
                )
            )
        )
        artists.append(
            style(
                mcollections.LineCollection(
                    [seg(a, b) for a, b in zip(bounds[1::2], bounds[2::2])],
                    colors=colors[1],
                    linewidths=width,
                    transform=mid,
                )
            )
        )
        for offset in ((0, 0), (dx, dy)):
            artists.append(
                style(
                    mcollections.LineCollection(
                        [seg(0, 1)],
                        colors=edgecolor,
                        linewidths=linewidth,
                        transform=get_transform(*offset),
                    )
                )
            )

    if not inside:
        for gl in _get_frame_gridliners_(ax):
            if {"bottom", "top"} & set(sides):
                gl.xpadding += width
            if {"left", "right"} & set(sides):
                gl.ypadding += width

    # Corners
    for horiz in ("bottom", "top"):
        for vert in ("left", "right"):
            if horiz in sides and vert in sides:
                cx, cy = _FRAME_SIDES[vert][1], _FRAME_SIDES[horiz][1]
                sx, sy = sign * (2 * cx - 1), sign * (2 * cy - 1)
                anchor = mtransforms.Affine2D().scale(1 / 72) + fig.dpi_scale_trans
                trans = anchor + mtransforms.ScaledTranslation(cx, cy, ax.transAxes)
                artists.append(
                    style(
                        mpatches.Rectangle(
                            (0, 0),
                            sx * width,
                            sy * width,
                            facecolor=colors[1],
                            edgecolor=edgecolor,
                            linewidth=linewidth,
                            transform=trans,
                        )
                    )
                )
    return artists


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


def plot_vectors(ax, lon, lat, u, v, stride=(1, 1), c=None, transform=None, **kwargs):
    """Plot vectors on a grid as arrows, possibly under-sampled

    Parameters
    ----------
    ax: matplotlib.axes.Axes
    lon, lat: array_like(ny, nx)
        Positions of the vectors
    u, v: array_like(ny, nx)
        Components of the vectors
    stride: int, tuple(int)
        Under-sampling strides along y and x, see :func:`get_strides`
    c: None, array_like(ny, nx)
        Values that are used to color the arrows
    transform: None, cartopy.crs.CRS
        Coordinate system of the positions, in which the vectors are also defined
    kwargs:
        Extra parameters are passed to :meth:`matplotlib.axes.Axes.quiver`.
        The arrows are centered on their position and thin by default.

    Return
    ------
    matplotlib.quiver.Quiver
    """
    sy, sx = (stride, stride) if np.isscalar(stride) else stride
    sub = (slice(None, None, sy), slice(None, None, sx))
    args = [np.ma.masked_invalid(np.asarray(a, dtype=np.float64)[sub]) for a in (lon, lat, u, v)]
    if c is not None:
        args.append(np.ma.masked_invalid(np.asarray(c, dtype=np.float64)[sub]))
    if transform is not None:
        kwargs["transform"] = transform
    kwargs.setdefault("pivot", "mid")
    kwargs.setdefault("width", 0.0025)
    kwargs.setdefault("headwidth", 3.5)
    kwargs.setdefault("headlength", 4.5)
    kwargs.setdefault("headaxislength", 4.0)
    return ax.quiver(*args, **kwargs)


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
    reference: matplotlib.lines.Line2D
        The marker of the reference
    ref_arc: None, matplotlib.lines.Line2D
        The arc of constant standard deviation of the reference
    contours: None, matplotlib.contour.QuadContourSet
        The contours of centered root mean square difference
    contour_labels: list
        The labels of the contours, as :class:`matplotlib.text.Text` objects
    corr_label, std_label: matplotlib.text.Text
        The axis labels
    markers: list
        Artists of the points that are added, to be customized quickly,
        for instance with :func:`xoa.plot.add_shadow`
    labels: list
        The :class:`matplotlib.text.Annotation` objects of the labels written next
        to the points, when ``values`` are given
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
        self.markers = []
        self.labels = []
        self.ref_arc = None
        self.contours = None
        self.contour_labels = []
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
        self.corr_label = ax.text(
            0.5 * self.thetamax,
            1.2 * self.rmax,
            corr_label,
            rotation=np.degrees(0.5 * self.thetamax) - 90,
            ha="center",
            va="center",
        )
        self.std_label = ax.annotate(
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
                self.contours = ax.contour(tt, rr, dist, levels=rms_levels, **kw)
                self.contour_labels = ax.clabel(self.contours, fmt="%g", fontsize=7)

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
            (self.ref_arc,) = ax.plot(
                np.linspace(0, self.thetamax, 181),
                np.full(181, self.ref_std),
                color=kw["color"],
                linestyle="--",
                linewidth=0.8,
            )
        (self.reference,) = ax.plot([0], [self.ref_std], label=ref_label or "_nolegend_", **kw)

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
                    self.labels.append(
                        ax.annotate(
                            labels[i],
                            (theta[i], std[i]),
                            xytext=(5, 5),
                            textcoords="offset points",
                            fontsize=8,
                        )
                    )
            if colorbar:
                self.colorbar = add_colorbar(artists[0], ax, **(cbar_kwargs or {}))
        self.markers.extend(artists)

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


COMPASS_POINTS = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")


class Rose:
    """A rose of frequencies on a polar axes

    The north is at the top and the directions are clockwise.
    Each sector is a stack of bars, one per bin of value.

    Parameters
    ----------
    fig: None, matplotlib.figure.Figure
        A new figure by default
    subplot: int, matplotlib.gridspec.SubplotSpec
        Position of the axes in the figure
    rmax: None, float
        Maximal frequency in percent
    rlabel_angle: float
        Direction in degrees of the frequency labels
    grid_kwargs: None, dict
        Parameters of :meth:`matplotlib.axes.Axes.grid`

    Attributes
    ----------
    ax: matplotlib.projections.polar.PolarAxes
    bars: list
        The :class:`matplotlib.container.BarContainer` objects, one per bin of value
    legend: None, matplotlib.legend.Legend
    """

    def __init__(self, fig=None, subplot=111, rmax=None, rlabel_angle=22.5, grid_kwargs=None):
        self.bars = []
        self.legend = None
        if fig is None:
            fig = plt.figure(figsize=(5, 5))
        ax = self.ax = fig.add_subplot(subplot, projection="polar")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_xticks(np.radians(np.arange(0, 360, 45)))
        ax.set_xticklabels(COMPASS_POINTS)
        ax.set_rlabel_position(rlabel_angle)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
        ax.set_ylim(bottom=0, top=rmax, auto=rmax is None)
        ax.grid(**{"linestyle": ":", "alpha": 0.6, **(grid_kwargs or {})})

    def add_bars(
        self, freq, edges, cmap="YlGnBu", colors=None, label="", legend=True, legend_kwargs=None
    ):
        """Add stacked bars

        Parameters
        ----------
        freq: array_like(nsector, nbin)
            Frequencies in percent, as returned by :func:`xoa.core.stats.rose_hist`
        edges: array_like(nbin)
            Lower edges of the bins of value
        cmap: str, matplotlib.colors.Colormap
            Colormap that gives the colors of the bins
        colors: None, list
            One color per bin, instead of the colormap
        label: str
            Label of the values, like ``"Hs [m]"``, used as the title of the legend
        legend: bool
            Add a legend of the bins
        legend_kwargs: None, dict
            Parameters of :meth:`matplotlib.axes.Axes.legend`

        Return
        ------
        list
            The :class:`matplotlib.container.BarContainer` objects
        """
        freq = np.asarray(freq, dtype="d")
        nsector, nbin = freq.shape
        edges = np.asarray(edges, dtype="d")
        if colors is None:
            colors = plt.get_cmap(cmap)(np.linspace(0.25, 1, nbin))
        theta = np.radians(np.arange(nsector) * 360.0 / nsector)
        bottom = np.zeros(nsector)
        bars = []
        for ibin in range(nbin):
            text = f"{edges[ibin]:g}–{edges[ibin + 1]:g}" if ibin < nbin - 1 else f"≥ {edges[ibin]:g}"
            bars.append(
                self.ax.bar(
                    theta,
                    freq[:, ibin],
                    width=np.radians(360.0 / nsector),
                    bottom=bottom,
                    color=colors[ibin],
                    edgecolor="w",
                    linewidth=0.4,
                    label=text,
                )
            )
            bottom = bottom + freq[:, ibin]
        self.bars.extend(bars)
        if legend:
            kw = {
                "title": label or None,
                "loc": "upper left",
                "bbox_to_anchor": (1.05, 1.0),
                "frameon": False,
            }
            kw.update(legend_kwargs or {})
            self.legend = self.ax.legend(**kw)
            self._fit_legend_()
        return bars

    def _fit_legend_(self):
        """Shrink the axes so that the legend fits in the figure"""
        fig = self.ax.figure
        for _ in range(3):
            fig.canvas.draw()
            overflow = self.legend.get_window_extent().x1 - fig.bbox.x1
            if overflow <= 0:
                break
            x0, y0, w, h = self.ax.get_position().bounds
            self.ax.set_position([x0, y0, w - overflow / fig.bbox.width * 1.02, h])


#: Parameters of :class:`Rose` that can be passed to :func:`plot_rose`
ROSE_INIT_KEYS = ("fig", "subplot", "rmax", "rlabel_angle", "grid_kwargs")


def plot_rose(direction, value, nsector=16, edges=None, rose=None, **kwargs):
    """Plot a rose of the frequencies of a direction and a value

    Parameters
    ----------
    direction: array_like
        Directions in degrees, clockwise from the north. Waves and winds usually
        come *from* a direction and currents go *to* it: this does not change the plot.
    value: array_like
        Values, like speeds or heights, with the same size as the directions
    nsector: int
        Number of sectors, centered on the north
    edges: None, array_like
        Lower edges of the bins of value, see :func:`xoa.core.stats.rose_hist`
    rose: None, Rose
        Existing rose. It is created when not provided.
    kwargs:
        Parameters of :class:`Rose` when it is created, or of :meth:`Rose.add_bars`,
        like ``cmap``, ``colors`` and ``label``.

    Return
    ------
    Rose
        The rose, which holds the axes in its ``ax`` attribute and the plotted artists

    Example
    -------
    .. code-block:: python

        plot_rose(wdir, wspd, edges=[0, 5, 10, 15, 20], label="Wind speed [m/s]")
    """
    freq, edges = rose_hist(direction, value, nsector=nsector, edges=edges)
    if rose is None:
        init = {key: kwargs.pop(key) for key in list(kwargs) if key in ROSE_INIT_KEYS}
        rose = Rose(**init)
    rose.add_bars(freq, edges, **kwargs)
    return rose
