#!/usr/bin/env python
# coding: utf-8
"""
A tour of xoa with the Gulf Stream
==================================

In this tutorial, we show with a single dataset how to:

* open a model output and find its variables and coordinates whatever their names,
* plot fields on maps and sections,
* compute dynamical diagnostics from the currents,
* regrid fields to another grid and interpolate them along a transect,
* find the other tutorials that go deeper.

The data are 3D fields of the GLORYS12 reanalysis of the Copernicus Marine Service,
averaged over 2 x 2 grid cells, on the 1st of October 2019, in the region of the Gulf Stream.
Most of the features are available as methods of the ``xoa`` accessor of xarray objects.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean  # noqa

import xoa
import xoa.meta as xmeta
from xoa.plot import plot_fields, plot_section

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
MAP_KW = {"gridlines_labels_on": ["bottom", "left"], "land": True}

# %%
# Open and decode
# ---------------
#
# The names are specific to the product: ``thetao`` for the temperature, ``uo`` for the zonal
# current... xoa knows them from the naming specifications, which are found from the
# attributes of the file, or set by the user like here for the Mercator products.
# ``decode`` renames the arrays to generic names and completes
# their attributes, so that all the models are accessed in the same way.
# See :ref:`sphx_glr_tutorials_meta_plot_naming.py`.

with xmeta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc")
    print(list(ds.data_vars))
    ds = ds.xoa.decode().isel(time=0)
    print(list(ds.data_vars))

# %%
# Even without decoding, a quantity is found from its generic name.

print(ds.xoa.get("lon").name, ds.xoa.get("depth").name, ds.xoa.u.attrs["standard_name"])

# %%
# Plot
# ----
#
# :func:`xoa.plot.plot_fields` draws several fields on maps of a single figure. The extent,
# the labels and the colorbars come from the data. Plots are also available from the
# accessor, like ``temp.xoa.plot.field()``. See :ref:`sphx_glr_tutorials_graphics_plot_maps_vectors.py`.

surf = ds.isel(depth=0)
plot_fields(
    {"Temperature": surf.temp, "Salinity": surf.sal, "Sea surface height": surf.ssh},
    ncols=3,
    figsize=(10, 3.4),
    cmap=cmocean.cm.thermal,
    map_kw=MAP_KW,
)

# %%
# A vertical section along a latitude is drawn with the depths that come with the dataset.

section = ds.temp.sel(lat=37.5, method="nearest")
fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
plot_section(section, ax=ax, method="contourf", cmap=cmocean.cm.thermal, levels=np.arange(2, 31, 2))
ax.set_ylim(1000, 0)  # depths are positive down

# %%
# Diagnostics
# -----------
#
# The ``xoa`` accessor of a dataset computes the diagnostics from its currents, found by their
# generic names, and the grid resolution from its coordinates:
# the relative vorticity, the Rossby number, the divergence, the strain
# and the Okubo-Weiss parameter. See :ref:`sphx_glr_tutorials_dynamics_plot_gulfstream_dynamics.py`.

rossby = surf.xoa.get_rossby()
okubo = surf.xoa.get_okuboweiss(normalize=True)

# %%
# The sea surface height gives the geostrophic currents. Here is the speed error with respect
# to the currents of the reanalysis.

ug, vg = surf.ssh.xoa.get_geos()
err = np.hypot(ug, vg) - np.hypot(surf.u, surf.v)
err.attrs.update(long_name="Geostrophic speed error", units="m s-1")
plot_fields(
    {"Rossby number": rossby, "Okubo-Weiss": okubo, "Geostrophic speed error": err},
    ncols=3,
    figsize=(10, 3.4),
    method="contourf",
    cmap=cmocean.cm.balance,
    map_kw=MAP_KW,
    levels=np.linspace(-1, 1, 21),
    extend="both",
)

# %%
# Regrid and interpolate
# ----------------------
#
# The same temperature is regridded to a regular grid that is twice coarser with the
# conservative method, and interpolated along a transect. See
# :ref:`sphx_glr_tutorials_grids_plot_regrid_interp.py`.

dst = xr.Dataset(
    coords={
        "lon": ("lon", np.arange(-72.0, -62.0, 0.5), {"standard_name": "longitude"}),
        "lat": ("lat", np.arange(34.0, 42.0, 0.5), {"standard_name": "latitude"}),
    }
)
coarse = surf.temp.xoa.regrid(dst, method="conservative")
tlon = xr.DataArray(np.linspace(-71.0, -63.0, 60), dims="s")
tlat = xr.DataArray(np.linspace(35.0, 41.0, 60), dims="s")
transect = surf.temp.xoa.interp(tlon, tlat)
print(coarse.shape, transect.shape)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), gridspec_kw={"width_ratios": (1, 1.3)}, constrained_layout=True)
coarse.plot(ax=axes[0], cmap=cmocean.cm.thermal)
axes[0].plot(tlon, tlat, "k--")
axes[0].set_title("Coarse grid and transect")
axes[1].plot(np.hypot(tlon - tlon[0], tlat - tlat[0]) * 111.0, transect, "k")
axes[1].set_xlabel("Distance along the transect [km]")
axes[1].set_ylabel("Temperature [°C]")
axes[1].set_title("Interpolation")
axes[1].grid(alpha=0.3)

# %%
# Go further
# ----------
#
# * Naming and configuration: :ref:`sphx_glr_tutorials_meta_plot_naming.py`,
#   :ref:`sphx_glr_tutorials_config_plot_naming_specs.py`.
# * Grids, vertical coordinates, filtering, kriging: the "Grids" tutorials.
# * Dynamics, vertical modes, mixed layer, advection: the "Dynamics" tutorials.
# * Comparison with observations: the "Evaluation" tutorials.
