#!/usr/bin/env python
"""
Decorate a map
==============

In this tutorial, we build a single map step by step, with:

* a background layer from a web map service,
* a GMT-like black and white frame,
* a scale bar and a compass,
* a minimap that locates the domain.

The functions are described in the :ref:`indepth.plot` guide.
See also: :ref:`sphx_glr_tutorials_graphics_plot_maps_vectors.py`.
"""

# %%
# Initialisations
# ---------------

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

import xoa
from xoa import meta
from xoa.plot import plot_field, plot_minimap

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

with meta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc").xoa.decode()
ds = ds.isel(time=0, depth=0, drop=True)

# %%
# The background layer
# --------------------
#
# The background is a layer of a web map service, which is added to the axes
# with :func:`~xoa.plot.add_layer`, or with ``layer=True`` in :func:`~xoa.plot.decorate_map`.
# The layers are defined in the ``layers`` section of the xoa config,
# and the default one, given by the ``plot.layer`` option, is the National Geographic map of Esri.
# See the :ref:`indepth.plot` guide to define your own.
# The land is drawn instead when the service is not reachable.
#
# The map is created by :func:`~xoa.plot.plot_field`, whose ``map_kw`` accepts the options of
# :func:`~xoa.plot.decorate_map`. The temperature is drawn as white contours,
# labelled with a formatter that appends the units,
# and an outline that keeps the labels readable.


def plot_temp(**kwargs):
    """Plot the temperature contours over the background layer and return the axes"""
    cs = plot_field(
        ds.temp,
        method="contour",
        levels=range(16, 29, 2),
        colors="w",
        linewidths=0.8,
        add_colorbar=False,
        **kwargs,
    )
    ax = cs.axes
    for label in ax.clabel(cs, fmt=StrMethodFormatter("{x:.0f}°C"), fontsize=7):
        label.set_path_effects([pe.withStroke(linewidth=2, foreground="0.2")])
    xoa.plot.add_layer(ax)
    return ax


ax = plot_temp(map_kw={"figsize": (6, 4.6)})

# %%
# The fancy frame
# ---------------
#
# ``fancy=True`` adds a black and white frame cut at the gridlines. The tick labels are
# moved away by its width in points, which is set with a dict like ``fancy={"width": 4}``.
# It requires a projection whose meridians and parallels are parallel to the map edges,
# like the default Mercator one.

ax = plot_temp(map_kw={"figsize": (6, 4.6), "fancy": {"width": 4}})

# %%
# Scale bar and compass
# ---------------------
#
# They are placed in the corners, with a glow and a shadow to stay readable over the background.

ax = plot_temp(
    map_kw={"figsize": (6, 4.6), "fancy": {"width": 4}},
    scalebar={"loc": "lower left", "effect": "glow"},
    compass={"style": "fancy", "loc": "upper left", "effect": "shadow"},
)

# %%
# The minimap
# -----------
#
# The minimap of :func:`~xoa.plot.plot_minimap` is an inset of the map that shows where
# the data are.

plot_minimap(ds, parent=ax, loc="lower right", plot_coords="box")
