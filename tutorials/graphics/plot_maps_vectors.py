#!/usr/bin/env python
"""
Plot fields and currents on maps
================================

In this tutorial, we show how to:

* plot a field on a map, whose extent and labels are found from the data,
* add a scale bar and a compass, optionally with a shadow or a glow,
* compare several fields in a single figure,
* overlay under-sampled arrows of the currents, optionally colored by the speed,
  with a reference arrow whose value is rounded, and uncertainty ellipses at their tip,
* plot a current time series as sticks,
* show the location of the domain with a minimap, placed so that the figure stays compact.

The concepts are explained in the :ref:`indepth.plot` guide.
See also: :ref:`sphx_glr_tutorials_graphics_plot_taylor_diagram.py`.
"""

# %%
# Initialisations
# ---------------

import cmocean
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xoa
from xoa import meta
from xoa.plot import (
    plot_double_minimap,
    plot_field,
    plot_fields,
    plot_minimap,
    plot_stick,
    plot_vectors,
)

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# Maps of a moderate size, with land
MAP_KW = {"land": True, "figsize": (6, 4.6)}

with meta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc").xoa.decode()
ds = ds.isel(time=0, depth=0, drop=True)
speed = np.hypot(ds.u, ds.v).rename("speed")
speed.attrs.update(long_name="Surface current speed", units="m s-1")

# %%
# A single field
# --------------
#
# Longitudes and latitudes are found in the coordinates, and the colorbar is labelled
# with the name and units of the field.

plot_field(ds.temp, cmap=cmocean.cm.thermal, map_kw=MAP_KW)

# %%
# Scale bar and compass
# ---------------------
#
# They are off by default. The scale bar is a ground distance, and the compass follows the
# meridians of the projection. Both accept a ``loc`` and an ``effect``, a shadow or a glow, for readability.
# See :ref:`sphx_glr_tutorials_graphics_plot_map_decorations.py` for a complete example.

plot_field(
    ds.temp,
    cmap=cmocean.cm.thermal,
    map_kw=MAP_KW,
    scalebar={"effect": "shadow"},
    compass={"style": "fancy", "effect": "shadow"},
)

# %%
# Locating the domain
# -------------------
#
# A minimap, drawn by :func:`~xoa.plot.plot_minimap`, shows where the data are.
# It is drawn from the coordinates of any object,
# here as a box over a small map. With ``parent``, it is an inset of the main axes, at the
# location ``loc``, which keeps the figure compact and the minimap close to what it locates.

ax = plot_field(ds.temp, cmap=cmocean.cm.thermal, map_kw=MAP_KW).axes
plot_minimap(ds, parent=ax, loc="lower right", plot_coords="box")

# %%
# A double minimap, drawn by :func:`~xoa.plot.plot_double_minimap`, adds a global map,
# with the location as a single point.
# The regional map is placed on the left of the global one.

ax = plot_field(ds.sal, cmap=cmocean.cm.haline, map_kw=MAP_KW).axes
plot_double_minimap(ds, parent=ax, loc="lower right", size=0.15, regional_ax="left")

# %%
# Several fields
# --------------
#
# A dict maps titles to fields, which are drawn on maps of a single figure.
# For publications, ``panel_labels=True`` adds the labels "a)", "b)"... to the maps.
# A string sets the style, like ``"({A})"``, and a dict sets the other parameters of
# :class:`~xoa.plot.PanelLabeler`, like ``loc="lower right"``.

fig, axes = plot_fields(
    {
        "Temperature": (ds.temp, {"cmap": cmocean.cm.thermal}),
        "Salinity": (ds.sal, {"cmap": cmocean.cm.haline}),
        "Sea surface height": ds.ssh,
        "Speed": (speed, {"cmap": cmocean.cm.speed}),
    },
    ncols=2,
    figsize=(8, 6),
    map_kw={"land": True},
    panel_labels=True,
)

# %%
# Currents as arrows
# ------------------
#
# The arrows of :func:`~xoa.plot.plot_vectors` are under-sampled so that they stay readable, with a
# stride that is adapted to the size of the axes. They are drawn over a field, here the speed.
# A reference arrow is added above the left corner of the map, with a round value
# that is close to the strong arrows, like 0.5 m s-1 here. Pass ``key=False`` to remove it.

ax = plot_field(speed, cmap=cmocean.cm.speed, map_kw=MAP_KW).axes
plot_vectors(ds.u, ds.v, ax=ax, color="w")

# %%
# The arrows can be colored by the speed, and the stride can be forced.

plot_vectors(ds.u, ds.v, color="speed", stride=3, map_kw=MAP_KW, title="Stride 3")

# %%
# Uncertainty ellipses
# --------------------
#
# With standard deviations of the components, and optionally their correlation,
# an ellipse is drawn at the tip of each arrow. Its radius is ``nsigma`` standard
# deviations, which includes 95% of a gaussian distribution by default. They are drawn
# with the color of the arrows, with a transparent face.
# Here the uncertainty is synthetic, and the correlation tilts the ellipses.

std = 0.03 + 0.15 * speed
plot_vectors(
    ds.u,
    ds.v,
    color="tab:blue",
    ustd=std,
    vstd=0.5 * std,
    uvcorr=0.6 + 0 * std,
    stride=3,
    map_kw=MAP_KW,
    ellipse_kwargs={"label": "95 %"},
).axes.legend(loc="lower right")

# %%
# Current time series
# -------------------
#
# Along a single axis, :func:`~xoa.plot.plot_stick` draws a current series as sticks anchored on a
# line, like a tidal current at a mooring. Here the series is synthetic.

time = np.array("2020-01-01", dtype="M8[h]") + np.arange(72)
phase = 2 * np.pi * np.arange(72) / 12.42
u = xr.DataArray(0.6 * np.cos(phase), dims="time", coords={"time": time}, name="u")
v = xr.DataArray(0.3 * np.sin(phase), dims="time", coords={"time": time}, name="v")
u.attrs["units"] = v.attrs["units"] = "m s-1"
fig, ax = plt.subplots(figsize=(8, 2.6), constrained_layout=True)
plot_stick(u, v, ax=ax)
ax.set_title("Rotating tidal current")
