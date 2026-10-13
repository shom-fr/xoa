#!/usr/bin/env python
# coding: utf-8
"""
Plot fields and currents on maps
================================

In this tutorial, we show how to:

* plot a field on a map, whose extent and labels are found from the data,
* compare several fields in a single figure,
* overlay under-sampled arrows of the currents, optionally colored by the speed,
* plot a current time series as sticks,
* show the location of the domain with a minimap.

The concepts are explained in the :ref:`indepth.plot` guide.
See also: :ref:`sphx_glr_tutorials_graphics_plot_taylor_diagram.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean

import xoa
from xoa.plot import (
    plot_field,
    plot_fields,
    plot_vectors,
    plot_stick,
    plot_minimap,
    plot_double_minimap,
)

xr.set_options(display_style="text")

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

plot_field(ds.ptemp, cmap=cmocean.cm.thermal, map_kw={"land": True})

# %%
# Locating the domain
# -------------------
#
# A minimap shows where the data are. It is drawn from the coordinates of any object,
# here as a box over a small map. A double minimap adds a global map
# with the location as a single point.

plot_field(ds.ptemp, cmap=cmocean.cm.thermal, map_kw={"land": True})
plot_minimap(ds, plot_coords="box")

# %%
# The double minimap, with a regional map below the global one.

plot_field(ds.sal, cmap=cmocean.cm.haline, map_kw={"land": True})
plot_double_minimap(ds, ax=[0.9, 0.55, 0.09])

# %%
# Several fields
# --------------
#
# A dict maps titles to fields, which are drawn on maps of a single figure.

fig, axes = plot_fields(
    {
        "Temperature": ds.ptemp,
        "Salinity": ds.sal,
        "Absolute dynamic topography": ds.adt,
        "Speed": speed,
    },
    ncols=2,
    figsize=(9, 7),
    map_kw={"land": True},
)

# %%
# Currents as arrows
# ------------------
#
# The arrows are under-sampled so that they stay readable, with a stride that is adapted to
# the size of the axes. They are drawn over a field, here the speed.

ax = plot_field(speed, cmap=cmocean.cm.speed, map_kw={"land": True}).axes
plot_vectors(ds.u, ds.v, ax=ax, color="w")

# %%
# The arrows can be colored by the speed, and the stride can be forced.

plot_vectors(ds.u, ds.v, color="speed", stride=3, map_kw={"land": True}, title="Stride 3")

# %%
# Current time series
# -------------------
#
# Along a single axis, a current series is drawn as sticks anchored on a line,
# like a tidal current at a mooring. Here the series is synthetic.

time = np.array("2020-01-01", dtype="M8[h]") + np.arange(72)
phase = 2 * np.pi * np.arange(72) / 12.42
u = xr.DataArray(0.6 * np.cos(phase), dims="time", coords={"time": time}, name="u")
v = xr.DataArray(0.3 * np.sin(phase), dims="time", coords={"time": time}, name="v")
plot_stick(u, v)
plt.title("Rotating tidal current")
