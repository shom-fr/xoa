#!/usr/bin/env python
# coding: utf-8
"""
Colormaps adapted to the data and graphical effects
===================================================

In this tutorial, we show how to:

* center a diverging colormap on zero for data that are not symmetric,
* use a circular colormap for directions and phases,
* make labels and lines stand out with shadows and glows.

The data are the surface fields of the GLORYS12 reanalysis of the Gulf Stream.
See also: :ref:`sphx_glr_tutorials_graphics_plot_maps_vectors.py`,
:ref:`sphx_glr_tutorials_graphics_plot_taylor_diagram.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean  # noqa

import xoa
import xoa.meta as xmeta
from xoa.color import CmapAdapter
from xoa.plot import add_glow, add_shadow

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

with xmeta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc").xoa.decode()
ds = ds.isel(time=0, depth=0)
lon, lat = ds.lon.values, ds.lat.values
ssh = ds.ssh - ds.ssh.mean()
print(float(ssh.min()), float(ssh.max()))

# %%
# Center a diverging colormap
# ---------------------------
#
# The sea surface height anomaly is mostly positive, so that the middle of a diverging colormap
# is not at zero, and its white would not mean "no anomaly". A :class:`~xoa.color.CmapAdapter`
# with the ``"piv0"`` specification crops the colormap so that its center is at the pivot, here
# zero, after the limits of the data are set. The adapted colormap, ``vmin`` and ``vmax``
# are given by :meth:`~xoa.color.CmapAdapter.get_dict`.

fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.6), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
asym = ssh.clip(max=0.25)
axes[0].pcolormesh(lon, lat, asym, cmap="cmo.balance", transform=ccrs.PlateCarree())
axes[0].set_title("Standard colormap")
cma = CmapAdapter("cmo.balance", "piv0")
cma.set_vlim(float(asym.min()), float(asym.max()))
mappable = axes[1].pcolormesh(lon, lat, asym, **cma.get_dict(), transform=ccrs.PlateCarree())
axes[1].set_title("Centered on zero")
fig.colorbar(mappable, ax=axes, shrink=0.8, label="Sea surface height anomaly [m]")

# %%
# A circular colormap for directions
# ----------------------------------
#
# The direction of the currents is an angle: 0 and 360 degrees are the same direction.
# A circular colormap like ``cmo.phase`` has the same color at both ends, which is what
# the ``"cyc360"`` specification expects.

direction = np.degrees(np.arctan2(ds.v, ds.u)) % 360
cyc = CmapAdapter("cmo.phase", "cyc360")
cyc.set_vlim(0, 360)
fig, ax = plt.subplots(
    figsize=(5.5, 4.2), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
mappable = ax.pcolormesh(lon, lat, direction, **cyc.get_dict(), transform=ccrs.PlateCarree())
cbar = fig.colorbar(mappable, ax=ax, shrink=0.8, label="Direction [°]", ticks=range(0, 361, 90))
ax.set_title("Direction of the surface currents")

# %%
# Make things stand out
# ---------------------
#
# :func:`xoa.plot.add_shadow` adds a drop shadow to lines or text, and :func:`~xoa.plot.add_glow`
# a halo that makes a text readable over any background. These effects use filters of the agg
# backend, which are applied at rendering. We use them on contour lines and labels.

speed = np.hypot(ds.u, ds.v)
fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.6), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
axes[0].contourf(
    lon, lat, speed, levels=np.linspace(0, 2, 17), cmap="cmo.deep", transform=ccrs.PlateCarree()
)
lines = axes[0].contour(
    lon, lat, speed, levels=[0.5, 1.0], colors="w", linewidths=1.2, transform=ccrs.PlateCarree()
)
add_shadow(lines, ax=axes[0], width=2, xoffset=1, yoffset=-1)
axes[0].set_title("Shadowed contours")

axes[1].pcolormesh(lon, lat, ds.temp, cmap="cmo.thermal", transform=ccrs.PlateCarree())
dark = axes[1].text(
    -67.5, 41.3, "Georges Bank", weight="bold", ha="center", transform=ccrs.PlateCarree()
)
add_glow(dark, ax=axes[1], width=4, color="w")
light = axes[1].text(
    -66.0, 40.0, "Gulf Stream", color="w", weight="bold", transform=ccrs.PlateCarree()
)
add_shadow(light, ax=axes[1], width=1.5, xoffset=1, yoffset=-1, alpha=0.8)
axes[1].set_title("A glow and a shadow on labels")
