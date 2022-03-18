#!/usr/bin/env python
# coding: utf-8
"""
Interpolate a meridional section of CROCO outputs to regular depths
===================================================================

This notebook, we show:

* how to compute the depths from s-coordinates,
* how to easily find the name of variables and coordinates,
* how to interpolate a 3D field with varying depths to regular depths.
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
lat_name = temp.xoa.lat.name

# %%
# Interpolate at regular depths
# -----------------------------
#
# We interpolate the temperature array from irregular to regular depths.
#
# Let's create the output depths.

depth = xr.DataArray(np.linspace(ds.depth.min(), ds.depth.max(), 100),
                     name="depth", dims="depth")

# %%
# Let's interpolate the temperature.

tempz = regrid1d(temp, depth)

# %%
# Plots
# -----
#
# Make a basic comparison plots.

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 4))
kw = dict(levels=np.arange(0, 23))
temp.plot.contourf(lat_name, "depth", cmap="cmo.thermal", ax=axs[0], **kw)
temp.plot.contour(lat_name, "depth", colors='w', linewidths=.3, ax=axs[0], **kw)
tempz.plot.contourf(lat_name, "depth", cmap="cmo.thermal", ax=axs[1], **kw)
tempz.plot.contour(lat_name, "depth", colors='w', linewidths=.3, ax=axs[1], **kw);


# %%
# Make iso slices
# ----------------------

from xoa.regrid import isoslice


# %%
# Let's say we want to slice the temperature at depth=-1200 m
# 
# The first argument of :func:`xoa.regrid.isoslice` is the array we want to slice , here the temperature.
# 
# The second one, is the array on which we search the isovalue. Since we look at given depth it must be the depth array
# 
# The last one is the isovalue
# 

isodepth = -1200.
isotemp = isoslice(temp, temp.depth, isodepth)


# %%
# Make a simple profil plot at a given latitude (index eta_rho=10)

plt.figure()
temp.isel(eta_rho=10).plot.line(y="depth")
plt.axhline(isodepth, ls='--')
plt.axvline(isotemp.isel(eta_rho=10), ls='--', c='r')


# %%
# Now we try to find the depth at which the temperature is 12°C for example
# 
# The order of the arguments are not the same as before !

isotemp=12.
isodep=isoslice(temp.depth,temp,isotemp)

# %%
# And the plot ...

plt.figure()
temp.isel(eta_rho=10).plot.line(y="depth")
plt.axvline(isotemp,ls='--')
plt.axhline(isodep.isel(eta_rho=10),ls='--',c='r')

# %%
# Et voilà!
