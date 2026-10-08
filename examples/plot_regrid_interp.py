#!/usr/bin/env python
# coding: utf-8
"""
Regrid and interpolate horizontally
===================================

In this tutorial, we show :

* how to regrid data to another regular or curvilinear grid with the bilinear, bicubic and
  conservative methods,
* how to reuse the weights, in memory and in a file,
* how to interpolate to arbitrary points, such as a transect, with and without time,
* how to handle land points with a mask and the ``skipna`` option,
* how to use the same tools from the ``xoa`` accessors.

"""

# %%
# Initialisations
# ---------------

import os
import tempfile
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xoa
from xoa import geo, interp, regrid, weights
from xoa.plot import add_colorbar, plot_field
from xoa.core.grid import create_rotated_grid

xr.set_options(display_style="text")

# %%
# Register the :ref:`xoa <accessors>` accessors:

xoa.register_accessors()

# %%
# Read the source data
# --------------------
#
# We use the surface temperature of a regional model on a staggered grid.
# Longitudes and latitudes are 2D, and the temperature is defined at the "rho" points,
# which are the coordinates it carries.

ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.surf.nc")
temp = ds.temp.isel(time=0, s_rho=-1)
print(temp)

# %%
# The coordinates are found from their meta-data, whatever their names.

print(xoa.coords.get_lon(temp).name, xoa.coords.get_lat(temp).name)

# %%
# Regrid to a regular grid
# ------------------------
#
# A destination grid is any object with longitude and latitude coordinates,
# that can be 1D or 2D.

lon_attrs = {"standard_name": "longitude", "units": "degrees_east"}
lat_attrs = {"standard_name": "latitude", "units": "degrees_north"}
dst = xr.Dataset(
    coords={
        "lon": ("lon", np.linspace(15.5, 21.5, 61), lon_attrs),
        "lat": ("lat", np.linspace(-37.4, -34.2, 33), lat_attrs),
    }
)

# %%
# The :class:`~xoa.regrid.Regridder` class supports the ``bilinear``, ``bicubic`` and
# ``conservative`` methods.

results = {}
for method in "bilinear", "bicubic", "conservative":
    regridder = regrid.Regridder(temp, dst, method=method)
    results[method] = regridder.regrid(temp)
print(results["bilinear"])

# %%
# Compare the three methods with the source field.
# The conservative method is the best choice for fluxes or quantities that must be
# preserved when going to a coarser grid.

# We use :func:`xoa.plot.plot_field`, that finds the coordinates in the field
# and sets up the map, and give all panels the extent of the source.

extent = geo.get_extent(temp, margin=0.02)
fig, axes = plt.subplots(
    1, 4, figsize=(16, 3.5), subplot_kw={"projection": ccrs.Mercator()}, sharey=True
)
kw = dict(vmin=float(temp.min()), vmax=float(temp.max()), cmap="Spectral_r", add_colorbar=False)
fields = {"source": temp, **results}
for ax, (title, field) in zip(axes, fields.items()):
    mappable = plot_field(
        field, ax=ax, title=title, map_kw={"gridlines_labels_on": ["bottom"]}, **kw
    )
    ax.set_extent(extent, crs=ccrs.PlateCarree())
add_colorbar(mappable, axes, da=temp)

# %%
# Weights
# -------
#
# The expensive part, finding where destination points are in the source grid, is done
# once, at the first call, and reused for all next calls and all data arrays
# sharing the same horizontal dimensions, whatever the other ones.

regridder = regrid.Regridder(temp, dst, method="bilinear")
t0 = time.perf_counter()
regridder.regrid(temp)
t1 = time.perf_counter()
regridder.regrid(ds.temp)  # all times and levels at once
t2 = time.perf_counter()
print(f"first call with weights computation: {t1 - t0:.3f} s")
print(f"second call on the full 4D variable: {t2 - t1:.3f} s")
print(ds.temp.dims, "->", regridder.regrid(ds.temp).dims)

# %%
# They are also shared in memory: another regridder or interpolator, or a call of the
# ``xoa`` accessors, that has the same grids, method and parameters reuses them.
# Weights are never computed twice for the same grids, whatever the number of variables.

print(regrid.Regridder(temp, dst, "bilinear").core_regridder is regridder.core_regridder)

# %%
# The weights can also be saved to a netcdf file with the ``weights_file`` parameter,
# to be reused in another session. A single file can hold the weights of all your grids
# and methods: each one is saved in a group whose name contains the method and a
# fingerprint of the grids. The weights are thus found automatically, and can never be
# loaded for other grids. See :mod:`xoa.weights`.

with tempfile.TemporaryDirectory() as tmpdir:
    weights_file = os.path.join(tmpdir, "weights.nc")
    for method in "bilinear", "conservative":
        regrid.Regridder(temp, dst, method, weights_file=weights_file).regrid(temp)
    print(*weights.list_groups(weights_file), sep="\n")

    # In a new session, the weights are read from the file
    regrid.clear_weights_cache()
    reloaded = regrid.Regridder(temp, dst, "bilinear", weights_file=weights_file)
    print(reloaded.core_regridder.has_weights)
    print(np.allclose(reloaded.regrid(temp), results["bilinear"], equal_nan=True))

# %%
# Regrid to a curvilinear grid
# ----------------------------
#
# Grids may also be curvilinear, like this rotated one.
# Points that are outside the source domain are set to nan.

grid = create_rotated_grid(30, 20, 18.5, -36.0, 25.0, 4.0, 2.5)
rotated = xr.Dataset(
    {
        "lon": (("y", "x"), grid["lon"], lon_attrs),
        "lat": (("y", "x"), grid["lat"], lat_attrs),
    }
)
on_rotated = regrid.Regridder(temp, rotated, "bilinear").regrid(temp)
print(on_rotated.dims, int(on_rotated.isnull().sum()), "missing points")

# %%
# And back to the original grid, to check the loss of information.

src_grid = xr.Dataset(coords={"lon_rho": temp.lon_rho, "lat_rho": temp.lat_rho})
back = regrid.Regridder(on_rotated, src_grid, "bilinear").regrid(on_rotated)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), subplot_kw={"projection": ccrs.Mercator()})
kw = dict(map_kw={"gridlines_labels_on": ["bottom"]}, add_colorbar=False)
plot_field(on_rotated, ax=axes[0], title="rotated grid", cmap="Spectral_r", **kw)
plot_field(back, ax=axes[1], title="back to the source grid", cmap="Spectral_r", **kw)
diff = plot_field(back - temp, ax=axes[2], title="difference", cmap="RdBu_r", vmin=-1, vmax=1, **kw)
add_colorbar(diff, axes[2], label="Difference [Celsius]")

# %%
# Interpolate to points
# ---------------------
#
# The :class:`~xoa.interp.Interpolator` class interpolates to any set of points, like
# a transect or scattered positions, with the ``bilinear`` and ``bicubic`` methods.
# Dimensions that are not horizontal are preserved.

lons = np.linspace(16.5, 20.5, 40)
lats = np.linspace(-36.8, -35.2, 40)
transect = interp.Interpolator(temp, lons, lats, method="bicubic").interp(ds.temp.isel(time=0))
print(transect.dims, transect.shape)

# %%
# The result is a section along the transect, for all levels at once.

fig, ax = plt.subplots(figsize=(6, 3))
ax.plot(lons, transect.isel(s_rho=-1))
ax.set_xlabel("Longitude")
ax.set_ylabel("Surface temperature")
fig.tight_layout()

# %%
# Destination coordinates can also have their own dimensions, or be a grid.
# Here is a 2D array of points, that is a destination grid.

pts_lon, pts_lat = np.meshgrid(np.linspace(17, 20, 4), np.linspace(-36.5, -35.5, 3))
print(interp.Interpolator(temp, pts_lon, pts_lat).interp(temp).dims)

# %%
# Interpolate in space and time
# -----------------------------
#
# Positions may also come with times, like an observing platform.
# The temperature of a regional ocean model, here on a regular grid, is interpolated
# linearly between the two available dates.

ibi = xoa.open_data_sample("MODELS/CMEMS-IBI/ibi-argo-7900573.nc")
sst = ibi.thetao.isel(depth=0)
times = np.array(
    ["2022-02-06T18:00", "2022-02-07T00:00", "2022-02-07T06:00"], dtype="datetime64[ns]"
)
track = interp.Interpolator(sst, [-10.0, -9.8, -9.6], [43.8, 43.9, 44.0])
at_times = track.interp_with_time(sst, times)
print(at_times.values)

# %%
# Masks and missing values
# ------------------------
#
# Land points here are set to zero, and they leak into the interpolated values near
# the coast. Provide the mask of the source grid and use ``skipna`` to ignore them.

mask = ds.mask_rho.values.astype(bool)
coast = xr.Dataset(
    coords={
        "lon": ("lon", np.linspace(18.3, 20.5, 45), lon_attrs),
        "lat": ("lat", np.linspace(-34.8, -34.1, 15), lat_attrs),
    }
)
naive = regrid.Regridder(temp, coast, "bilinear").regrid(temp)
masked = regrid.Regridder(temp, coast, "bilinear", src_mask=mask).regrid(temp, skipna=True)
print("minimum without mask:", float(naive.min()), "- with mask:", float(masked.min()))

# %%
# The ``na_thres`` parameter controls how many missing neighbours are tolerated:
# 1 means that a single valid neighbour is enough to get a value, and values closer
# to 0 are stricter.

strict = regrid.Regridder(temp, coast, "bilinear", src_mask=mask).regrid(
    temp, skipna=True, na_thres=0.5
)
print(int(masked.isnull().sum()), "missing points against", int(strict.isnull().sum()))

# %%
# Use the accessors
# -----------------
#
# All of that is also available from the ``xoa`` accessors, which build the
# regridder or the interpolator on the fly.

print(temp.xoa.regrid(dst, method="bilinear").shape)
print(temp.xoa.interp(lons, lats, method="bilinear").shape)

# %%
# .. note::
#     To keep the weights between calls, create the :class:`~xoa.regrid.Regridder` or
#     :class:`~xoa.interp.Interpolator` object yourself, as shown above.
#     The numba kernels they rely on are available in :mod:`xoa.core.interp` and
#     :mod:`xoa.core.regrid` for use with plain numpy arrays.

plt.show()
