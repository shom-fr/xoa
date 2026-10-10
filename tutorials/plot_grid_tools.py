#!/usr/bin/env python
# coding: utf-8
"""
Inspect and manipulate horizontal and vertical grids
====================================================

In this tutorial, we show :

* how to compute the horizontal resolution of a regular or curvilinear grid,
* how to plot the edges and centers of a grid, with an adaptive under-sampling,
* how to get the geographic extent of the strips along the edges of a grid,
* how to convert a 2D longitude and latitude to 1D axes when the grid allows it,
* how to compute the edges of cells from their centers, with extrapolation at the ends,
* how to pad an array and its coordinates.

"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import xoa
from xoa import grid
from xoa.core.grid import create_rotated_grid
from xoa.plot import plot_grid

xr.set_options(display_style="text")

# %%
# Register the :ref:`xoa <accessors>` accessors:

xoa.register_accessors()

# %%
# Horizontal resolution
# ---------------------
#
# We use the surface temperature of a regional model, with 2D longitudes and latitudes.

ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.surf.nc")
temp = ds.temp.isel(time=0, s_rho=-1)

# %%
# :func:`~xoa.grid.get_resolution` gives the distance in meters between adjacent points
# along x and y. The arrays have one point less than the grid along the dimension
# of the difference.

dx, dy = grid.get_resolution(temp)
print(dx.dims, dx.shape, dy.shape)
print(f"dx: {dx.min().item() / 1e3:.1f} - {dx.max().item() / 1e3:.1f} km")
print(f"dy: {dy.min().item() / 1e3:.1f} - {dy.max().item() / 1e3:.1f} km")

# %%
# The resolution shrinks along x towards the poles, as meridians get closer.
# The ``get_median_resolution`` function gives a single value in degrees, which is
# convenient to choose the resolution of another dataset.

print(grid.get_median_resolution(temp))

# %%
# Plot the grid
# -------------
#
# The :func:`xoa.plot.plot_grid` function draws the edges and the centers of the cells.
# Edges are computed from the centers, with extrapolation at the ends of the grid.

plot_grid(temp)

# %%
# Large grids are under-sampled to stay readable. By default, the stride is adapted
# to the size of the map, so that cells are at least ``min_spacing`` pixels apart on screen,
# whatever the number of cells, the projection or the rotation of the grid.
# Here is a grid of 400 x 300 points, that is drawn with only a few tens of lines,
# and whose displayed centers are the ones of the coarse cells.

big = create_rotated_grid(400, 300, 18.5, -36.0, 25.0, 4.0, 2.5)
big = xr.Dataset(
    {
        "lon": (("y", "x"), big["lon"], {"standard_name": "longitude", "units": "degrees_east"}),
        "lat": (("y", "x"), big["lat"], {"standard_name": "latitude", "units": "degrees_north"}),
    }
)
plot_grid(big)

# %%
# The stride can also be set by hand, for each direction, and edges or centers
# can be switched off.

plot_grid(big, stride=(40, 60), centers=False)

# %%
# The resolution is shown on a map as the geometric mean of the resolutions along x and y,
# and the bathymetry of a dataset is found thanks to its meta-data.
# Edges can be drawn over these fields.

plot_grid(temp, kind="resolution")

# %%
plot_grid(ds, kind="bathy", edges=True, map_kw={"land": True})

# %%
# Let's check how the resolution varies on a larger domain, with a regular grid in degrees.

lon_attrs = {"standard_name": "longitude", "units": "degrees_east"}
lat_attrs = {"standard_name": "latitude", "units": "degrees_north"}
world = xr.Dataset(
    coords={
        "lon": ("lon", np.arange(0.0, 40.0, 1.0), lon_attrs),
        "lat": ("lat", np.arange(-80.0, 81.0, 1.0), lat_attrs),
    }
)
wdx, wdy = grid.get_resolution(world)
fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
ax.plot(wdx.isel(lon=0) / 1e3, world.lat[:], label="along x")
ax.plot(
    wdy.isel(lon=0) / 1e3, 0.5 * (world.lat[1:].values + world.lat[:-1].values), label="along y"
)
ax.set_xlabel("Resolution (km)")
ax.set_ylabel("Latitude")
ax.grid(alpha=0.3)
ax.legend()

# %%
# Edge extents
# ------------
#
# The extent of a rotated grid is much larger than the area it really covers.
# :func:`~xoa.grid.get_edge_extents` gives the bounding box of the strips made of the
# first or last cells along each edge, which follow the grid.

rotated_grid = create_rotated_grid(30, 20, 18.5, -36.0, 25.0, 4.0, 2.5)
rotated = xr.Dataset(
    {
        "lon": (("y", "x"), rotated_grid["lon"], lon_attrs),
        "lat": (("y", "x"), rotated_grid["lat"], lat_attrs),
    }
)
extents = grid.get_edge_extents(rotated, ["north", "west"], n_cells=3)
print(extents)

# %%
# :func:`xoa.plot.plot_grid` draws the mesh and these strips, which are compatible with
# :func:`xoa.geo.get_extent` that gives the extent of the whole grid.

plot_grid(rotated, strips=["north", "west"], n_cells=3)

# %%
# Convert to rectangular axes
# ---------------------------
#
# When 2D longitudes and latitudes are constant along one dimension, they are in fact
# 1D axes. :func:`~xoa.grid.to_rect` converts them, and the dimensions are renamed
# accordingly. The check is based on :func:`xoa.core.grid.check_grid_type`.

print(temp.dims, "->", grid.to_rect(temp).dims)

# %%
# It leaves curvilinear grids unchanged, and warns unless ``errors="ignore"``.

data = xr.DataArray(
    np.zeros(rotated.lon.shape),
    dims=rotated.lon.dims,
    coords={"lon": rotated.lon, "lat": rotated.lat},
)
print(grid.to_rect(data, errors="ignore").lon.ndim)

# %%
# Cell edges
# ----------
#
# :func:`~xoa.grid.get_edges` computes the edges of the cells from their centers.
# Inner edges are in the middle of the centers, and outer ones are extrapolated
# by default. This is what the conservative regridding along a dimension needs.

depth = xoa.open_data_sample("MODELS/CMEMS-IBI/ibi-argo-7900573.nc").depth
edges = grid.get_edges(depth, "depth")
print(depth.values[:4], edges.values[:5])

# %%
# The ``edge`` mode replicates the end values instead, so that the first and last
# cells are half as thick as their neighbours.

print(grid.get_edges(depth, "depth", mode="edge").values[:5])

# %%
# Conversely, :func:`~xoa.grid.get_centers` gives the middle of consecutive points.

print(grid.get_centers(edges, "depth").values[:4])

# %%
# Padding
# -------
#
# :func:`~xoa.grid.pad` adds points along dimensions, to the data and to their coordinates,
# which are linearly extrapolated by default.

sst = temp.isel(eta_rho=slice(0, 3), xi_rho=slice(0, 4))
padded = grid.pad(sst, {"eta_rho": 1, "xi_rho": (0, 2)}, mode="linear_extrap")
print(sst.shape, "->", padded.shape)
print(sst.lon_rho.values[0], padded.lon_rho.values[0])
