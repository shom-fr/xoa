#!/usr/bin/env python
"""
Interpolate a meridional section of CROCO outputs to regular heights
====================================================================

In this tutorial, we show how to:

* compute the z heights from s-coordinates,
* easily find the name of variables and coordinates,
* interpolate a 3D field with varying heights to regular heights,
* compute the mixed layer depth from temperature.

The concepts are explained in the :ref:`indepth.vertical` guide.
See also: :ref:`sphx_glr_tutorials_grids_plot_grid_tools.py`, :ref:`sphx_glr_tutorials_dynamics_plot_croco_rossby.py`.
"""

# %%
# Initialisations
# ---------------
#
# Import needed modules.

import cmocean  # noqa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xoa
import xoa.meta as xmeta
from xoa.plot import add_colorbar, plot_section
from xoa.regrid import regrid1d
from xoa.thermdyn import mixed_layer_depth

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

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
# 2. Compute the ``z_rho`` and ``z_w`` heights, that are negative in the ocean
# 3. Assign them as coordinates
#
# Note that the :attr:`~xarray.Dataset.xoa.decode_sigma` subaccessor of the :ref:`xoa <accessors>`
# accessor calls the :func:`xoa.sigma.decode_cf_sigma` function.

ds = ds.xoa.decode_sigma()
print(ds.z_rho)

# %%
# Find coordinate names from CF conventions
# -----------------------------------------
#
# The `z_rho` was assigned as coordinates at the previous stage.
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
    np.linspace(ds.z_rho.values.min(), ds.z_rho.values.max(), 1000), name="z", dims="z"
)

# %%
# Let's interpolate the temperature with :func:`xoa.regrid.regrid1d`.

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


def plot_both(axs, **extra):
    """Plot the original and the interpolated sections with a shared colorbar"""
    kw = dict(levels=np.arange(0, 23), x="lat", add_colorbar=False, **extra)
    for ax, field in zip(axs, (temp, tempz)):
        mappable = plot_section(field, ax=ax, method="contourf", cmap="cmo.thermal", **kw)
        plot_section(field, ax=ax, method="contour", colors="k", linewidths=0.3, **kw)
        ax.set_xlabel("Latitude [°N]")
        ax.set_ylabel("Height [m]")
    axs[0].set_title("Original")
    axs[1].set_title("Interpolated")
    add_colorbar(mappable, axs, da=temp)
    return mappable


fig, axs = plt.subplots(
    ncols=2, sharex=True, sharey=True, figsize=(9, 3.4), constrained_layout=True
)
plot_both(axs)

# %%
# Plot a zoom near the surface and add the mixed layer depth isoline.

fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(9, 3), constrained_layout=True)
plot_both(axs)
axs[0].plot(mld[lat_name], mld, color="k", linewidth=2, linestyle="--")
axs[1].plot(mldz[lat_name], mldz, color="k", linewidth=2, linestyle="--")
axs[0].set_ylim(-300, 0)

# %%
# Et voilà!
