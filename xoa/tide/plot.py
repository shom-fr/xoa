"""
Plots of tidal harmonic analysis results

- :func:`plot_tidal_ellipse`: one annotated ellipse
- :func:`plot_tidal_ellipses`: ellipse map of one constituent
- :func:`plot_tidal_constituents`: ellipse maps of several constituents
- :func:`plot_cotidal`: amplitude and phase of a sea level constituent
- :func:`plot_tidal_spectrum`: amplitude of each constituent against frequency
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

import cmocean  # noqa: F401 to register the cmo colormaps
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Arc

from .. import coords as xcoords
from .. import exceptions
from .. import plot as xplot
from ..core import plot as cplot
from ..core.tide import ellipse_properties, ellipse_vector

ROTATION_COLORS = {1: "tab:red", -1: "tab:blue", 0: "0.4"}
ROTATION_LABELS = {1: "counter-clockwise", -1: "clockwise", 0: "rectilinear"}

# %% helpers


def _select_constituent_(coef, constituent):
    if "constituent" in coef.dims:
        if constituent is None:
            raise exceptions.XoaError(
                "constituent is required when coef holds several constituents"
            )
        return coef.sel(constituent=constituent)
    return coef


def _lonlat_(da):
    """Longitudes and latitudes of ``da`` as numpy arrays of its shape"""
    _, lon, lat = xr.broadcast(da, xcoords.get_lon(da), xcoords.get_lat(da))
    return lon.transpose(*da.dims).values, lat.transpose(*da.dims).values


def _auto_spacing_(lon, lat):
    """Median distance in degrees to the nearest neighbor, longitudes being scaled"""
    from scipy.spatial import cKDTree

    x = np.column_stack([lon * np.cos(np.radians(lat)), lat])
    if len(x) < 2:
        return 1.0
    return float(np.median(cKDTree(x).query(x, k=2)[0][:, 1]))


def _is_color_by_(color_by, *names):
    return isinstance(color_by, str) and color_by in names


def _wrap_deg_(phase):
    return np.degrees(phase) % 360.0


# %% single ellipse


def plot_tidal_ellipse(coef, constituent=None, ax=None, n=200, annotate=True, color=None):
    """Plot one tidal current ellipse with its characteristics

    The ellipse is drawn in velocity space, east on x and north on y, with its
    major and minor axes, the inclination angle from east, an arrow for the
    sense of rotation and a marker for the current at the reference time of the
    analysis, i.e. ``ω (t - t_ref) = 0``.

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :meth:`xoa.tide.TidalHarmonic.analyse_uv` at a single point.
    constituent: str, None
        Constituent name, required when ``coef`` has several constituents.
    ax: matplotlib.axes.Axes, None
    n: int
        Number of points along the ellipse.
    annotate: bool
        Add a text box with the semi-axes, inclination, phase, ellipticity,
        rotation sense and period.
    color: str, None
        Ellipse color, which depends on the rotation sense by default.

    Return
    ------
    matplotlib.axes.Axes
    """
    c = _select_constituent_(coef, constituent)
    smaj, smin, incl, phase = (
        float(c[k]) for k in ("semi_major", "semi_minor", "inclination", "phase")
    )
    ellipticity, rotation = (float(a) for a in ellipse_properties(smaj, smin))
    rotation = int(rotation)
    if color is None:
        color = ROTATION_COLORS[rotation]
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)

    u, v = ellipse_vector(smaj, smin, incl, phase, np.linspace(0, 2 * np.pi, n))
    ax.plot(u, v, color=color, lw=2)
    ax.axhline(0, color="0.7", lw=0.6, zorder=0)
    ax.axvline(0, color="0.7", lw=0.6, zorder=0)

    # Axes of the ellipse
    cos_i, sin_i = np.cos(incl), np.sin(incl)
    ax.plot(
        [-smaj * cos_i, smaj * cos_i], [-smaj * sin_i, smaj * sin_i], color=color, lw=0.8, ls="--"
    )
    ax.plot(
        [smin * sin_i, -smin * sin_i], [-smin * cos_i, smin * cos_i], color=color, lw=0.8, ls=":"
    )

    # Inclination from east, shown modulo 180° in (-90°, 90°]
    incl_deg = (np.degrees(incl) + 90.0) % 180.0 - 90.0
    r_arc = 0.45 * smaj
    ax.add_patch(
        Arc((0, 0), 2 * r_arc, 2 * r_arc, theta1=min(0, incl_deg), theta2=max(0, incl_deg))
    )
    half = np.radians(incl_deg) / 2
    ax.annotate(
        f"{incl_deg:.0f}°", (0.6 * r_arc * np.cos(half), 0.6 * r_arc * np.sin(half)), fontsize=8
    )

    # Current at the reference time, and rotation arrow a bit further along the ellipse
    ax.plot(u[0], v[0], "o", color=color, ms=6, label="t = t_ref")
    k = max(1, n // 12)
    ax.annotate(
        "",
        xy=(u[2 * k], v[2 * k]),
        xytext=(u[k], v[k]),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
    )

    ax.set_aspect("equal")
    ax.set_xlabel("u [m s$^{-1}$]")
    ax.set_ylabel("v [m s$^{-1}$]")
    lim = 1.3 * smaj if smaj > 0 else 1.0
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    name = str(c["constituent"].values) if "constituent" in c.coords else (constituent or "")
    ax.set_title(f"{name} tidal current ellipse".strip())

    if annotate:
        lines = [
            f"semi-major  {smaj:.3f} m/s",
            f"semi-minor  {abs(smin):.3f} m/s",
            f"inclination {incl_deg:.1f}°",
            f"phase       {_wrap_deg_(phase):.1f}°",
            f"ellipticity {ellipticity:+.2f}",
            f"rotation    {ROTATION_LABELS[rotation]}",
        ]
        if "frequency" in c.coords and float(c["frequency"]) > 0:
            lines.append(f"period      {24.0 / float(c['frequency']):.2f} h")
        ax.text(
            0.02,
            0.98,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            family="monospace",
            fontsize=7.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )
    ax.legend(loc="lower right", fontsize=7, frameon=False)
    return ax


# %% ellipse maps


def _draw_reference_(ax, transform, lon, lat, spacing, scale, ref):
    """Circle of radius ``ref`` m s-1 in the lower-left corner, with its label"""
    x0 = lon.min() - 0.5 * spacing
    y0 = lat.min() - 0.5 * spacing
    coslat = np.cos(np.radians(y0))
    t = np.linspace(0, 2 * np.pi, 73)
    ax.plot(
        x0 + scale * ref * np.cos(t) / coslat,
        y0 + scale * ref * np.sin(t),
        color="k",
        lw=1,
        zorder=6,
        transform=transform,
    )
    ax.text(
        x0 + 1.3 * scale * ref / coslat,
        y0,
        f"{ref:g} m/s",
        va="center",
        fontsize=7,
        zorder=6,
        transform=transform,
    )


def plot_tidal_ellipses(
    coef,
    constituent=None,
    ax=None,
    step=1,
    scale=None,
    color_by="rotation",
    cmap=None,
    vmin=None,
    vmax=None,
    ref=None,
    phase_ticks=True,
    background=None,
    background_kw=None,
    lw=0.8,
    title=None,
    colorbar=True,
    transform=None,
    map_kw=None,
):
    """Map the tidal current ellipses of one constituent

    Each ellipse is centred on its grid point and drawn at geographic scale:
    longitudes are corrected by ``cos(lat)`` so that ellipses are not distorted.
    Optionally, a tick joins the centre to the current at the reference time of the
    analysis, which shows the phase.

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :meth:`xoa.tide.TidalHarmonic.analyse_uv` with longitude and
        latitude coordinates. Points with a NaN ``semi_major``, like land, are skipped.
    constituent: str, None
        Constituent name, required when ``coef`` has several constituents.
    ax: None, cartopy.mpl.geoaxes.GeoAxes
        A map is created when not provided.
    step: int
        Plot one point out of ``step`` along each spatial dimension.
    scale: None, float
        Degrees of latitude per m s-1. By default, the 95th percentile of the
        semi-major axis spans 90 % of the distance between plotted points.
    color_by: {"rotation", "phase", "ellipticity", "semi_major"}, str, array
        ``"rotation"`` colors counter-clockwise ellipses in red and clockwise ones
        in blue. The others color by value with ``cmap``. Any other string is
        a matplotlib color. An array of values, with the shape of one constituent,
        is also accepted.
    cmap, vmin, vmax:
        Color mapping when ``color_by`` selects values.
    ref: None, float
        Reference speed in m s-1: a circle of that radius is drawn in a corner of the map.
    phase_ticks: bool
        Draw the tick marking the current at ``t = t_ref``.
    background: None, xarray.DataArray
        Field drawn below the ellipses with :func:`xoa.plot.plot_field`,
        like the semi-major axis or the bathymetry.
    background_kw: None, dict
        Parameters of the call to :func:`xoa.plot.plot_field` for the background.
    lw: float
        Ellipse line width.
    title: None, str
    colorbar: bool
        Colorbar when ``color_by`` selects values, legend for ``"rotation"``.
    transform: None, cartopy.crs.CRS
        Coordinate system of the data, which defaults to ``PlateCarree``
    map_kw: None, dict
        Parameters that decorate the map, see :func:`xoa.plot.plot_field`.

    Return
    ------
    matplotlib.collections.LineCollection
        The ellipses. Use ``.axes`` and ``.figure`` to get the axes and figure.
    """
    ccrs, _ = cplot._import_cartopy_()
    transform = transform or ccrs.PlateCarree()
    c = _select_constituent_(coef, constituent)
    smaj = c["semi_major"]
    lon, lat = _lonlat_(smaj)

    sl = tuple(slice(None, None, step) for _ in smaj.dims)
    params = {
        k: c[k].transpose(*smaj.dims).values[sl]
        for k in ("semi_major", "semi_minor", "inclination", "phase")
    }
    lon, lat = lon[sl], lat[sl]
    valid = np.isfinite(params["semi_major"]) & np.isfinite(lon) & np.isfinite(lat)
    if not valid.any():
        raise exceptions.XoaError("No valid point to plot")
    params = {k: a[valid] for k, a in params.items()}
    lon_v, lat_v = lon[valid], lat[valid]

    spacing = _auto_spacing_(lon_v, lat_v)
    if scale is None:
        scale = 0.45 * spacing / np.nanpercentile(params["semi_major"], 95)
    extent = [
        lon_v.min() - 2 * spacing,
        lon_v.max() + 2 * spacing,
        lat_v.min() - 2 * spacing,
        lat_v.max() + 2 * spacing,
    ]
    ax = xplot._get_map_axes_(ax, extent, transform, map_kw)
    if background is not None:
        xplot.plot_field(background, ax=ax, **(background_kw or {}))

    # Ellipse polygons, with longitudes stretched by 1/cos(lat) to stay round on the map
    u, v = ellipse_vector(
        *(params[k] for k in ("semi_major", "semi_minor", "inclination", "phase")),
        np.linspace(0, 2 * np.pi, 73),
    )
    coslat = np.cos(np.radians(lat_v))[:, None]
    segs = np.stack([lon_v[:, None] + scale * u / coslat, lat_v[:, None] + scale * v], axis=-1)

    ellipticity, rotation = ellipse_properties(params["semi_major"], params["semi_minor"])
    values, label, norm_range, colors = None, None, (None, None), color_by
    if _is_color_by_(color_by, "rotation"):
        colors = [ROTATION_COLORS[int(r)] for r in rotation]
    elif _is_color_by_(color_by, "phase"):
        values, label, cmap_default, norm_range = (
            _wrap_deg_(params["phase"]),
            "Phase [°]",
            "cmo.phase",
            (0, 360),
        )
    elif _is_color_by_(color_by, "ellipticity"):
        values, label, cmap_default, norm_range = ellipticity, "Ellipticity", "cmo.balance", (-1, 1)
    elif _is_color_by_(color_by, "semi_major"):
        values, label, cmap_default = (
            params["semi_major"],
            "Semi-major axis [m s$^{-1}$]",
            "cmo.speed",
        )
    elif not isinstance(color_by, str):
        values = np.asarray(color_by).reshape(smaj.shape)[sl][valid]
        label, cmap_default = getattr(color_by, "name", None) or "Value", "cmo.amp"

    kw = dict(linewidths=lw, zorder=5, transform=transform)
    lc = LineCollection(segs, **kw, **({} if values is not None else {"colors": colors}))
    if values is not None:
        lc.set_cmap(cmap or cmap_default)
        lc.set_array(values)
        lc.set_clim(
            norm_range[0] if vmin is None else vmin, norm_range[1] if vmax is None else vmax
        )
    ax.add_collection(lc)

    if phase_ticks:
        tick = np.stack(
            [
                np.stack([lon_v, lat_v], axis=-1),
                np.stack(
                    [lon_v + scale * u[:, 0] / coslat[:, 0], lat_v + scale * v[:, 0]], axis=-1
                ),
            ],
            axis=1,
        )
        tc = LineCollection(tick, **{**kw, "linewidths": 0.6})
        if values is not None:
            tc.set_cmap(lc.get_cmap())
            tc.set_array(values)
            tc.set_clim(*lc.get_clim())
        else:
            tc.set_color(colors)
        ax.add_collection(tc)

    if ref is not None:
        _draw_reference_(ax, transform, lon_v, lat_v, spacing, scale, ref)

    lc.colorbar_label = label
    if colorbar:
        if values is not None:
            ticks = np.arange(0, 361, 60) if _is_color_by_(color_by, "phase") else None
            xplot.add_colorbar(lc, ax=ax, label=label, ticks=ticks)
        elif _is_color_by_(color_by, "rotation"):
            handles = [
                Line2D([], [], color=ROTATION_COLORS[s], label=ROTATION_LABELS[s])
                for s in (1, -1)
                if (rotation == s).any()
            ]
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9)

    name = str(c["constituent"].values) if "constituent" in c.coords else (constituent or "")
    ax.set_title(title if title is not None else f"{name} tidal current ellipses".strip())
    return lc


def plot_tidal_constituents(
    coef, constituents, ncols=None, scale=None, figsize=None, projection=None, **kwargs
):
    """Map the tidal current ellipses of several constituents side by side

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :meth:`xoa.tide.TidalHarmonic.analyse_uv`.
    constituents: list(str)
    ncols: None, int
        Number of columns, which defaults to all constituents on one row.
    scale: None, float, dict
        Ellipse scale, see :func:`plot_tidal_ellipses`: one value for all
        panels or a ``{constituent: scale}`` dict, with missing ones being automatic.
        Shallow-water constituents, much weaker than M2, usually need a larger one.
    figsize: None, tuple
    projection: None, str, cartopy.crs.Projection
        Projection of the maps, see :func:`xoa.core.plot.get_projection`.
    kwargs:
        Extra parameters are passed to :func:`plot_tidal_ellipses`. A ``"phase"``
        or ``"ellipticity"`` coloring shares a single colorbar between the panels.

    Return
    ------
    matplotlib.figure.Figure
    list(cartopy.mpl.geoaxes.GeoAxes)
    """
    n = len(constituents)
    ncols = ncols or n
    nrows = -(-n // ncols)
    colorbar = kwargs.pop("colorbar", True)
    shared = _is_color_by_(kwargs.get("color_by"), "phase", "ellipticity")
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize or (4.5 * ncols, 4.2 * nrows),
        subplot_kw={"projection": cplot.get_projection(projection)},
        constrained_layout=True,
        squeeze=False,
    )
    axes = list(axes.ravel())
    for ax in axes[n:]:
        fig.delaxes(ax)
    axes = axes[:n]
    for ax, name in zip(axes, constituents):
        lc = plot_tidal_ellipses(
            coef,
            name,
            ax=ax,
            scale=scale.get(name) if isinstance(scale, dict) else scale,
            colorbar=colorbar and not shared,
            **kwargs,
        )
    if colorbar and shared:
        ticks = np.arange(0, 361, 60) if kwargs["color_by"] == "phase" else None
        xplot.add_colorbar(lc, axes, label=lc.colorbar_label, ticks=ticks)
    return fig, axes


# %% sea level


def plot_cotidal(
    coef,
    constituent,
    ax=None,
    phase_levels=None,
    cmap="cmo.amp",
    title=None,
    phase_kw=None,
    **kwargs,
):
    """Co-tidal chart: amplitude in colors and phase in contour lines

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :meth:`xoa.tide.TidalHarmonic.analyse` on 2D fields with
        longitude and latitude coordinates.
    constituent: str
    ax: None, cartopy.mpl.geoaxes.GeoAxes
        A map is created when not provided.
    phase_levels: None, array-like
        Phase contour levels in degrees, with an automatic spacing by default.
    cmap: str
        Colormap of the amplitude.
    title: None, str
    phase_kw: None, dict
        Parameters of the phase :meth:`matplotlib.axes.Axes.contour`.
    kwargs:
        Extra parameters are passed to :func:`xoa.plot.plot_field` to plot the amplitude.

    Return
    ------
    matplotlib.axes.Axes
    """
    c = coef.sel(constituent=constituent)
    amp = c["amp"]
    if amp.ndim != 2:
        raise exceptions.XoaError("plot_cotidal needs 2D fields")
    amp = amp.assign_attrs(long_name=f"{constituent} amplitude", units="m")
    pc = xplot.plot_field(amp, ax=ax, cmap=cmap, **kwargs)
    ax = pc.axes
    lon, lat = _lonlat_(amp)

    # Mask the cells next to the 0/360° jump, where contours would be spurious
    phase = _wrap_deg_(c["phase"].transpose(*amp.dims).values)
    jump = np.zeros(phase.shape, dtype=bool)
    for axis in (0, 1):
        d = np.abs(np.diff(phase, axis=axis)) > 180.0
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        jump[tuple(a)] |= d
        jump[tuple(b)] |= d
    phase = np.where(jump, np.nan, phase)
    if phase_levels is None:
        # About 6 to 12 contours over the phase range
        span = np.nanmax(phase) - np.nanmin(phase)
        step = next((st for st in (1, 2, 5, 10, 15, 30, 45, 60) if span / st <= 12), 90)
        phase_levels = np.arange(0, 360, step)
    ccrs, _ = cplot._import_cartopy_()
    cs = ax.contour(
        lon,
        lat,
        phase,
        levels=phase_levels,
        colors="k",
        linewidths=0.7,
        transform=ccrs.PlateCarree(),
        **(phase_kw or {}),
    )
    ax.clabel(cs, fmt="%d°", fontsize=7)
    ax.set_title(title if title is not None else f"{constituent} co-tidal chart")
    return ax


# %% spectrum


def plot_tidal_spectrum(coef, ax=None, record_days=None, reduce="mean", label_min=0.0):
    """Amplitude of each constituent against its frequency

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :meth:`xoa.tide.TidalHarmonic.analyse` (``amp``)
        or :meth:`xoa.tide.TidalHarmonic.analyse_uv` (``semi_major``).
    ax: None, matplotlib.axes.Axes
    record_days: None, float
        Record length: the Rayleigh resolution ``1 / record_days`` is shown as a bar.
    reduce: {"mean", "max"}
        How spatial dimensions are reduced.
    label_min: float
        Only label constituents with an amplitude above this value.

    Return
    ------
    matplotlib.axes.Axes
    """
    var = "amp" if "amp" in coef else "semi_major"
    sdims = [d for d in coef[var].dims if d != "constituent"]

    def _reduce_(da):
        return getattr(da, reduce)(sdims, skipna=True) if sdims else da

    amp = _reduce_(coef[var]).values
    freq = coef["frequency"].values
    names = [str(n) for n in coef["constituent"].values]
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 3.5), constrained_layout=True)
    ax.vlines(freq, 0, amp, color="tab:blue", lw=2)
    ax.plot(freq, amp, "o", color="tab:blue", ms=4)
    if var == "amp" and "amp_std" in coef:
        ax.errorbar(
            freq,
            amp,
            yerr=_reduce_(coef["amp_std"]).values,
            fmt="none",
            ecolor="k",
            lw=0.8,
            capsize=2,
        )
    for f, a, name in zip(freq, amp, names):
        if a >= label_min:
            ax.annotate(
                name, (f, a), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=8
            )
    if record_days:
        y = -0.12 * np.nanmax(amp)
        x0 = ax.get_xlim()[0] + 0.02 * np.ptp(ax.get_xlim())
        ax.annotate(
            "",
            xy=(x0, y),
            xytext=(x0 + 1.0 / record_days, y),
            arrowprops=dict(arrowstyle="<->", lw=1),
            annotation_clip=False,
        )
        ax.text(
            x0 + 1.0 / record_days,
            y,
            f" Rayleigh {1 / record_days:.3f} cpd",
            va="center",
            fontsize=7,
        )
    ax.set_xlabel("Frequency [cycles per day]")
    ax.set_ylabel("Amplitude [m]" if var == "amp" else "Semi-major axis [m s$^{-1}$]")
    ax.set_ylim(bottom=-0.2 * np.nanmax(amp) if record_days else 0)
    ax.set_title("Tidal constituents")
    return ax
