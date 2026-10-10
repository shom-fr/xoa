#!/usr/bin/env python
# coding: utf-8
"""
Interpolate a meridional section of CROCO outputs to regular heights
====================================================================

In this notebook, we show:

* how to compute the z heights from s-coordinates,
* how to easily find the name of variables and coordinates,
* how to interpolate a 3D field with varying heights to regular heights,
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
import cmocean  # noqa
import xoa
from xoa.plot import plot_section
from xoa.regrid import regrid1d
from xoa.thermdyn import mixed_layer_depth
import xoa.meta as xmeta

xr.set_options(display_style="text")

# %%
# Register the internal CROCO naming specifications

croco_cfg_file = xoa.get_meta_config_file("croco")
print(croco_cfg_file)
xmeta.register_meta_specs(croco_cfg_file)

# %%
# You can provide your own configuration file.
#
# In this way, the :mod:`xoa.meta` module will recognise the
# CROCO netcdf names.
# It would be equivalent to force loading name specs with
# `xoa.meta.set_meta_specs(xoa.get_meta_config_file("croco"))`.

# %%
# Read the model
# --------------
# This sample is a meridional extraction of a full 3D CROCO output.
sample_file = xoa.get_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.meridional.nc")
print(sample_file)
ds = xr.open_dataset(sample_file)
print(ds)

# %%
# Compute heights from s-coordinates
# ----------------------------------
#
# Decode the dataset according to the CF conventions:
#
# 1. Find sigma terms
# 2. Compute the ``z`` heights, that are negative in the ocean
# 3. Assign them as coordinates
#
# Note that the ``decode_sigma`` subaccessor of the :ref:`xoa <accessors>` accessor
# calls the :func:`xoa.sigma.decode_cf_sigma` function.

ds = ds.xoa.decode_sigma()
print(ds.z)

# %%
# Find coordinate names from CF conventions
# -----------------------------------------
#
# The `z` was assigned as coordinates at the previous stage.
# We use the :ref:`xoa <accessors.dataset>` accessor to easily access the temperature, latitude and z arrays.
# The default configuration exposes shortcuts for some variables and coordinates
# as shown in :metasec:`accessors`.

temp = ds.xoa.temp.squeeze()
temp = temp.where(temp != 0)  # convert zeros to nans
lat_name = temp.xoa.lat.name

# %%
# Interpolate at regular heights
# ------------------------------
#
# We interpolate the temperature array from irregular to regular heights.
#
# Let's create the output heights.

z = xr.DataArray(
    np.linspace(ds.z.values.min(), ds.z.values.max(), 1000), name="z", dims="z"
)

# %%
# Let's interpolate the temperature.

tempz = regrid1d(temp, z, extrap="top")

# %%
# Compute the mixed layer depths
# -------------------------------
#
# The mixed layer depths are computed here as the depth at which the temperature
# is `deltatemp` lower than the surface temperature,
# thanks to the :func:`xoa.thermdyn.mixed_layer_depth` function.

deltatemp = 0.2
mld = -mixed_layer_depth(temp, deltatemp=deltatemp)
mldz = -mixed_layer_depth(tempz, deltatemp=deltatemp)

# %%
# Plots
# -----

# %%
# Plot the full section with :func:`xoa.plot.plot_section`, that finds the depths and
# labels the axes from the meta-data.

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(13, 5),
                        constrained_layout=True)
kw = dict(levels=np.arange(0, 23), x="lat")
plot_section(temp, ax=axs[0], method="contourf", cmap="cmo.thermal", **kw)
plot_section(
    temp, ax=axs[0], method="contour", colors="k", linewidths=0.3, add_colorbar=False, **kw
)
plot_section(tempz, ax=axs[1], method="contourf", cmap="cmo.thermal", **kw)
plot_section(
    tempz, ax=axs[1], method="contour", colors="k", linewidths=0.3, add_colorbar=False, **kw
)
axs[0].set_title("Original")
axs[1].set_title("Interpolated")

# %%
# Plot a zoom near the surface and add the mixed layer depth isoline.

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(13, 4.5),
                        constrained_layout=True)
plot_section(temp, ax=axs[0], method="contourf", cmap="cmo.thermal", **kw)
plot_section(
    temp, ax=axs[0], method="contour", colors="k", linewidths=0.3, add_colorbar=False, **kw
)
axs[0].plot(mld[lat_name], mld, color="k", linewidth=2, linestyle="--")
plot_section(tempz, ax=axs[1], method="contourf", cmap="cmo.thermal", **kw)
plot_section(
    tempz, ax=axs[1], method="contour", colors="k", linewidths=0.3, add_colorbar=False, **kw
)
axs[1].plot(mldz[lat_name], mldz, color="k", linewidth=2, linestyle="--")
axs[0].set_ylim(-300, 0)
axs[0].set_title("Original")
axs[1].set_title("Interpolated")

# %%
# Et voilà!
