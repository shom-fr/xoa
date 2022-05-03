#!/usr/bin/env python
# coding: utf-8
"""
Interpolate a meridional section of CROCO outputs to regular depths
===================================================================

This notebook, we show:

* how to compute the depths from s-coordinates,
* how to easily find the name of variables and coordinates,
* how to interpolate a 3D field with varying depths to regular depths,
* how to compute the mixed layer depth from temperature.
"""

# %%
# Initialisations
# ---------------
#
# Import needed modules.

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean
import xoa
from xoa.regrid import regrid1d
from xoa.thermdyn import mixed_layer_depth

xr.set_options(display_style="text")

# %%
# Register the :meth:`xarray.Dataset.decode_sigma` callable accessor.

xoa.register_accessors(decode_sigma=True)

# %%
# The :ref:`xoa <accessors>` accessor is also registered by default, and give access
# to most of the fonctionalities of the other accessors.

# %%
# Read the model
# --------------
# This sample is a meridional extraction of a full 3D CROCO output.

ds = xoa.open_data_sample("croco.south-africa.meridional.nc")
print(ds)

# %%
# Compute depths from s-coordinates
# ---------------------------------
#
# Decode the dataset according to the CF conventions:
#
# 1. Find sigma terms
# 2. Compute depths
# 3. Assign depths as coordinates
#
# Note that the :meth:`xarray.Dataset.decode_sigma` callable accessor
# calls the :func:`xoa.sigma.decode_cf_sigma` function.

ds = ds.decode_sigma()
print(ds.depth)

# %%
# Find coordinate names from CF conventions
# -----------------------------------------
#
# The `depth` was assigned as coordinates at the previous stage.
# We use the :ref:`xoa <accessors.dataset>` accessor to easily access the temperature, latitude and depth arrays.
# The default configuration exposes shortcuts for some variables and coordinates
# as shown in :cfsec:`accessors`.

temp = ds.xoa.temp.squeeze()
temp = temp.where(temp != 0)  # convert zeros to nans
lat_name = temp.xoa.lat.name

# %%
# Interpolate at regular depths
# -----------------------------
#
# We interpolate the temperature array from irregular to regular depths.
#
# Let's create the output depths.

depth = xr.DataArray(np.linspace(ds.depth.min(), ds.depth.max(), 1000), name="depth", dims="depth")

# %%
# Let's interpolate the temperature.

tempz = regrid1d(temp, depth, extrap="top")

# Compute the mixed layer depths
# -------------------------------
#
# The mixed layer depths are computed here as the depth at wich the temperature
# is `deltatemp` below the surface temperature,
# thanks to the :func:`xoa.thermdyn.mixed_layer_depth` function.

deltatemp = 0.2
mld = -mixed_layer_depth(temp, deltatemp=deltatemp)
mldz = -mixed_layer_depth(temp, deltatemp=deltatemp)

# %%
# Plots
# -----

# %%
# Plot the full section.

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 4))
kw = dict(levels=np.arange(0, 23))
temp.plot.contourf(lat_name, "depth", cmap="cmo.thermal", ax=axs[0], **kw)
temp.plot.contour(lat_name, "depth", colors='k', linewidths=0.3, ax=axs[0], **kw)
tempz.plot.contourf(lat_name, "depth", cmap="cmo.thermal", ax=axs[1], **kw)
tempz.plot.contour(lat_name, "depth", colors='k', linewidths=0.3, ax=axs[1], **kw)

# %%
# Plot a zoom near the surface and add the mixed layer depth isoline.

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 3))
kw = dict(levels=np.arange(0, 23))
temp.plot.contourf(lat_name, "depth", cmap="cmo.thermal", ax=axs[0], **kw)
temp.plot.contour(lat_name, "depth", colors='k', linewidths=0.3, ax=axs[0], **kw)
mld.plot.line(x=lat_name, color="k", linewidth=2, linestyle="--", ax=axs[0])
tempz.plot.contourf(lat_name, "depth", cmap="cmo.thermal", ax=axs[1], **kw)
tempz.plot.contour(lat_name, "depth", colors='k', linewidths=0.3, ax=axs[1], **kw)
mldz.plot.line(x=lat_name, color="k", linewidth=2, linestyle="--", ax=axs[1])
axs[0].set_ylim(-300, 0)

# # %%
# # Et voilà!
