#!/usr/bin/env python
# coding: utf-8
"""
Decorate a map
==============

In this tutorial, we build a single map step by step, with:

* a GEBCO bathymetry as a background,
* a GMT-like black and white frame,
* a scale bar and a compass,
* a minimap that locates the domain.

The functions are described in the :ref:`indepth.plot` guide.
See also: :ref:`sphx_glr_tutorials_graphics_plot_maps_vectors.py`.
"""

# %%
# Initialisations
# ---------------

import matplotlib.pyplot as plt

import xoa
from xoa import meta
from xoa.plot import plot_field, plot_minimap

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

with meta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc").xoa.decode()
ds = ds.isel(time=0, depth=0, drop=True)

# %%
# The GEBCO background
# --------------------
#
# The bathymetry is a layer of the GEBCO web map service, which is added to the axes
# with :meth:`cartopy.mpl.geoaxes.GeoAxes.add_wms`.
# It is requested at each draw, at the resolution of the figure.
# The land of Natural Earth is used instead when the service is not reachable.

GEBCO_URL = "https://wms.gebco.net/mapserv?"


def add_gebco(ax, layer="GEBCO_LATEST"):
    """Add the GEBCO layer below the other artists, or the land if it fails"""
    try:
        ax.add_wms(GEBCO_URL, layers=[layer]).set_zorder(0)
    except Exception as exc:
        print(f"GEBCO is not available ({type(exc).__name__}): the land is drawn instead")
        xoa.plot.add_land(ax, scale="50m")


# %%
# The map is created by :func:`~xoa.plot.plot_field`, whose ``map_kw`` accepts the options of
# :func:`~xoa.plot.decorate_map`. The temperature is drawn as white contours to let the
# bathymetry show through.

mappable = plot_field(
    ds.temp,
    method="contour",
    levels=range(16, 29, 2),
    colors="w",
    linewidths=0.8,
    add_colorbar=False,
    map_kw={"figsize": (6, 4.6)},
)
add_gebco(mappable.axes)

# %%
# The fancy frame
# ---------------
#
# ``fancy=True`` adds a black and white frame cut at the gridlines. The tick labels are
# moved away by its width in points, which is set with a dict like ``fancy={"width": 4}``.
# It requires a projection whose meridians and parallels are parallel to the map edges,
# like the default Mercator one.

mappable = plot_field(
    ds.temp,
    method="contour",
    levels=range(16, 29, 2),
    colors="w",
    linewidths=0.8,
    add_colorbar=False,
    map_kw={"figsize": (6, 4.6), "fancy": {"width": 4}},
)
add_gebco(mappable.axes)

# %%
# Scale bar and compass
# ---------------------
#
# They are placed in the corners, with a glow and a shadow to stay readable over the bathymetry.

mappable = plot_field(
    ds.temp,
    method="contour",
    levels=range(16, 29, 2),
    colors="w",
    linewidths=0.8,
    add_colorbar=False,
    map_kw={"figsize": (6, 4.6), "fancy": {"width": 4}},
    scalebar={"loc": "lower left", "effect": "glow"},
    compass={"style": "fancy", "loc": "upper left", "effect": "shadow"},
)
ax = mappable.axes
add_gebco(ax)

# %%
# The minimap
# -----------
#
# The minimap is an inset of the map that shows where the data are.

plot_minimap(ds, parent=ax, loc="lower right", plot_coords="box")
