#!/usr/bin/env python
# coding: utf-8
"""
Accumulating statistics in time
===============================

In this tutorial, we show how to:

* accumulate temporal statistics step by step and check them against a manual computation,
* get volume statistics that are automatically weighted by the layer thicknesses and
  the grid cell sizes, whose weights are far from homogeneous,
* stop the accumulation and restart it from a file.

The data are synthetic: a temperature that is linear with depth, on sigma layers
over a sloping bottom, whose thickness varies with time.

The concepts are explained in the :ref:`indepth.stataccum` guide.
See also: :ref:`sphx_glr_tutorials_graphics_plot_taylor_diagram.py`.
"""

# %%
# Initialisations
# ---------------

import os
import tempfile

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from xoa.stats import StatAccum
from xoa.grid import get_dx_dy, get_dz

nt, nz, ny, nx = 10, 20, 12, 30
time = np.array("2020-01-01", dtype="M8[h]") + np.arange(nt) * 6
lon = xr.DataArray(np.linspace(-6, -4, nx), dims="x", attrs={"units": "degrees_east"})
lat = xr.DataArray(np.linspace(30, 60, ny), dims="y", attrs={"units": "degrees_north"})
bathy = 20 + 180 * (lon - lon[0]) / (lon[-1] - lon[0])
tide = xr.DataArray(1 + 0.1 * np.sin(np.arange(nt)), dims="time", coords={"time": time})
sigma = xr.DataArray((np.arange(nz) + 0.5) / nz, dims="nz")

# The depth of the layers varies in space and time
z = (-bathy * tide * sigma).transpose("time", "nz", "x")
z.attrs.update(units="m", positive="up")
temp = 15 + 0.05 * z + 0.1 * (lat - 45)
temp = temp.transpose("time", "nz", "y", "x").rename("temp")
temp.attrs.update(long_name="Sea temperature", units="degC")
temp = temp.assign_coords(z=z, lon=lon, lat=lat)
temp_ref = (temp + 0.5).rename("temp_ref")

# %%
# Accumulating step by step
# -------------------------
#
# The data are appended in five chunks of two time steps, as if they were
# produced along a run. Nothing but the moments is kept between two calls.
# After each chunk, we get the statistics at a given point, and we compare them
# with a manual computation on all the time steps available so far.

point = dict(nz=-1, y=0, x=-1)
sa = StatAccum()
steps, tmean, tstd = [], [], []
for i in range(0, nt, 2):
    sa.append(temp.isel(time=slice(i, i + 2)))
    stats = sa.get_stats(["tmean", "tstd"]).isel(point)
    steps.append(sa.nt)
    tmean.append(float(stats.tmean))
    tstd.append(float(stats.tstd))

series = temp.isel(point).values
manual_mean = [series[:n].mean() for n in steps]
manual_std = [series[:n].std() for n in steps]
np.testing.assert_allclose(tmean, manual_mean)
np.testing.assert_allclose(tstd, manual_std)

fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True)
for ax, acc, man, name in zip(axes, (tmean, tstd), (manual_mean, manual_std), ("mean", "std")):
    ax.plot(steps, man, "o-", label="Manual")
    ax.plot(steps, acc, "x--", ms=10, label="StatAccum")
    ax.set_title(f"Temporal {name}")
    ax.set_xlabel("Number of accumulated time steps")
ax.legend()

# %%
# The accumulated statistics match the manual ones at each step,
# although the data of the previous steps are not used anymore.

# %%
# Volume statistics
# -----------------
#
# We now reduce all the spatial dimensions at each time step. Statistics are weighted
# by the layer thicknesses, which are deduced from the ``z`` coordinate, and by the grid
# cell sizes, which are deduced from the longitudes and latitudes.
# These are the weights that ``StatAccum`` computes internally.
# They are very heterogeneous, as shown below.

dz = get_dz(temp)
dx, dy = get_dx_dy(temp)
weights = dz * dx * dy

fig, axes = plt.subplots(1, 3, figsize=(9, 2.5))
dz.isel(time=0).plot(ax=axes[0], x="x", y="nz")
axes[0].set_title("Layer thickness (m)")
(dx * dy).isel(x=0).plot(ax=axes[1])
axes[1].set_title("Cell area (m2)")
weights.isel(time=0, y=0).plot(ax=axes[2], x="x", y="nz")
axes[2].set_title("Cell volume (m3)")
fig.tight_layout()
print("Weights range over a factor", float(weights.max() / weights.min()))

# %%
# We accumulate the volume statistics and compare the mean with a manual weighted
# computation. We also compare with an unweighted mean, where thin layers on the shelf
# count as much as thick ones in the open ocean, and cells at high latitudes
# as much as the others.

sa = StatAccum(spatial=True)
sa.append(temp, temp_ref)
stats = sa.get_stats()
print(stats)

manual = (temp * weights).sum(["nz", "y", "x"]) / weights.sum(["nz", "y", "x"])
xr.testing.assert_allclose(stats.smean0, manual, check_dim_order=False)
unweighted = StatAccum(temporal=False, spatial=True, weights=False).append(temp).get_stats()

fig, ax = plt.subplots(figsize=(6, 3))
manual.plot(ax=ax, label="Manual weighted", lw=4, alpha=0.4)
stats.smean0.plot(ax=ax, label="StatAccum weighted", ls="--")
unweighted.smean.plot(ax=ax, label="Unweighted")
ax.legend()
ax.set_title("Volume mean of the temperature")

# %%
# Stop and restart
# ----------------
#
# The accumulation can be interrupted at any time and resumed later, for instance
# from one job of a long simulation to the next.

path = os.path.join(tempfile.mkdtemp(), "stats.nc")
part = StatAccum(spatial=True)
part.append(temp.isel(time=slice(0, 4)), temp_ref.isel(time=slice(0, 4)))
part.dump(path)

part = StatAccum.from_file(path)
part.append(temp.isel(time=slice(4, None)), temp_ref.isel(time=slice(4, None)))
xr.testing.assert_allclose(part.get_stats(), stats)

# %%
# The restarted accumulator gives the same statistics as the uninterrupted one.
# Times already accumulated are refused, or skipped with ``on_overlap="skip"``.

fig, ax = plt.subplots(figsize=(6, 3))
part.get_stats().tmean0.isel(y=0, x=-1).plot(ax=ax, y="z", yincrease=False)
ax.set_title("Temporal mean in the deepest column")
