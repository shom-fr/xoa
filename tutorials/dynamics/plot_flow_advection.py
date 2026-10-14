#!/usr/bin/env python
# coding: utf-8
"""
Advect particles in the Gulf Stream currents
============================================

In this tutorial, we show how to:

* find the surface currents and the sea level of a reanalysis,
* draw the flow as a "windy" plot of short lagrangian tracks,
* release particles across the stream and follow them,
* see how a patch of particles is stretched and dispersed, with distances on the sphere.

The data are surface fields of the GLORYS12 reanalysis of the Copernicus Marine Service,
averaged over 2 x 2 grid cells, on the 1st of October 2019, in the region of the Gulf Stream.
The currents are steady here, since there is a single date.
See also: :ref:`sphx_glr_tutorials_dynamics_plot_gulfstream_dynamics.py`.
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
from xoa.dyn import flow2d, get_sea_level
from xoa.geo import haversine
from xoa.plot import plot_field, plot_flow

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

MAP_KW = {"gridlines_labels_on": ["bottom", "left"], "land": True}
PCAR = ccrs.PlateCarree()

# %%
# The currents
# ------------
#
# The sea level is searched with :func:`xoa.dyn.get_sea_level`, which knows the
# usual variants (SSH, ADT, SLA...) from the naming specifications.

with xmeta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc").xoa.decode()
ds = ds.isel(time=0, depth=0)
u, v = ds.xoa.get("u"), ds.xoa.get("v")
ssh = get_sea_level(ds)
speed = np.hypot(u, v).rename("speed")
speed.attrs.update(long_name="Surface current speed", units="m s-1")
print(ssh.name, ":", ssh.attrs["long_name"])

# %%
# The flow as a windy plot
# ------------------------
#
# :func:`xoa.plot.plot_flow` advects a few thousand random particles during a short time
# and draws their tracks with a fading transparency: it reveals the structure of the flow, like
# the meanders and the eddies of the stream.

ax = plot_field(speed, cmap=cmocean.cm.deep, map_kw=MAP_KW).axes
plot_flow(u, v, axes=ax, color="w", particles=3000, linewidth=(0.3, 0.9), transform=PCAR)

# %%
# Release particles across the stream
# -----------------------------------
#
# :func:`xoa.dyn.flow2d` integrates the particle positions with a fourth-order Runge-Kutta
# scheme, from initial longitudes and latitudes, for a duration and with a time step.
# We release one particle every 0.1 degree along 69.5 degrees West, across the stream, and follow
# them during ten days. The output is a dataset of longitudes and latitudes with the
# ``time`` and ``particles`` dimensions.

lat0 = np.arange(35.5, 40.0, 0.1)
lon0 = np.full(lat0.shape, -69.5)
tracks = flow2d(u, v, (lon0, lat0), np.timedelta64(10, "D"), np.timedelta64(1, "h"))
print(tracks.sizes)

# %%
# The tracks are colored by the day, and their end points are marked: the particles that
# are caught by the core of the stream travel far to the east, and the others stay in the
# slow waters or recirculate in the eddies.

days = (tracks.time - tracks.time[0]) / np.timedelta64(1, "D")
fig, ax = plt.subplots(
    figsize=(7.5, 5.5), subplot_kw={"projection": PCAR}, constrained_layout=True
)
plot_field(speed, ax=ax, cmap="Greys", alpha=0.6, add_colorbar=False, map_kw=MAP_KW)
for k in range(tracks.sizes["particles"]):
    sc = ax.scatter(
        tracks.lon[:, k],
        tracks.lat[:, k],
        c=days,
        s=1.5,
        cmap=cmocean.cm.thermal,
        transform=PCAR,
    )
ax.plot(lon0, lat0, "k-", lw=1.5, transform=PCAR)
fig.colorbar(sc, ax=ax, shrink=0.75, label="Day")
ax.set_title("Particles released across the stream")

# %%
# Dispersion of a patch
# ---------------------
#
# A patch of 200 particles is released at the northern flank of the stream. The distance
# between them is computed on the sphere with :func:`xoa.geo.haversine`, in meters,
# and its growth shows how fast the stream stretches the patch.

rng = np.random.default_rng(0)
plon = rng.normal(-68.0, 0.15, 200)
plat = rng.normal(37.5, 0.15, 200)
patch = flow2d(u, v, (plon, plat), np.timedelta64(10, "D"), np.timedelta64(1, "h"))
cx = patch.lon.mean("particles").values[:, None]
cy = patch.lat.mean("particles").values[:, None]
dist = haversine(patch.lon.values, patch.lat.values, cx, cy)  # to the center of mass, in m
spread = np.sqrt((dist**2).mean(axis=1)) * 1e-3

fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.8), gridspec_kw={"width_ratios": (1.25, 1)}, constrained_layout=True
)
axes[0].remove()
axes[0] = fig.add_subplot(1, 2, 1, projection=PCAR)
plot_field(speed, ax=axes[0], cmap="Greys", alpha=0.5, add_colorbar=False, map_kw=MAP_KW)
for day, color in zip((0, 5, 10), ("tab:blue", "tab:orange", "tab:red")):
    step = int(day * 24)
    axes[0].scatter(
        patch.lon[step],
        patch.lat[step],
        s=4,
        color=color,
        label=f"Day {day}",
        transform=PCAR,
    )
axes[0].legend(loc="lower left", markerscale=2)
axes[0].set_title("A patch of particles")
axes[1].plot((patch.time - patch.time[0]) / np.timedelta64(1, "D"), spread, color="k")
axes[1].set_xlabel("Day")
axes[1].set_ylabel("Root mean square distance [km]")
axes[1].set_title("Dispersion")
axes[1].grid(alpha=0.3)

# %%
# The patch is first advected and stretched along the stream, and its size grows with time.
# With currents that vary in time, like a series of daily means, the same tools give
# realistic trajectories, for example to search the origin of a water mass.
