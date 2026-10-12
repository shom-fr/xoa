#!/usr/bin/env python
# coding: utf-8
"""
Accumulating statistics in time
===============================

In this tutorial, we show how to:

* accumulate temporal and spatial statistics step by step,
* get volume statistics that are automatically weighted by the layer thicknesses and
  the grid cell sizes,
* stop the accumulation and restart it from a file.

The data are synthetic: a temperature that is linear with depth, on sigma layers
over a sloping bottom, whose thickness varies with time.
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
# Volume statistics
# -----------------
#
# We reduce all the spatial dimensions at each time step. Statistics are weighted by
# the layer thicknesses, which are deduced from the ``z`` coordinate, and by the grid
# cell sizes, which are deduced from the longitudes and latitudes.
# Temporal statistics are also accumulated at each point.

sa = StatAccum(spatial=True)
sa.append(temp, temp_ref)
stats = sa.get_stats()
print(stats)

# %%
# Without weights, thin layers on the shelf count as much as thick ones
# in the open ocean, and cells at high latitudes as much as the others.

unweighted = StatAccum(temporal=False, spatial=True, weights=False).append(temp).get_stats()

fig, ax = plt.subplots(figsize=(6, 3))
stats.smean0.plot(ax=ax, label="Weighted")
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
