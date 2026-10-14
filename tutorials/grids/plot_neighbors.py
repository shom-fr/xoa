#!/usr/bin/env python
# coding: utf-8
"""
Find neighbors on the sphere
============================

In this tutorial, we show how to:

* build a :class:`~xoa.geo.GeoKDTree` once from a grid and query it several times,
* find the grid points nearest to scattered observations, or their k nearest neighbors,
* flag the grid points within a distance of the observations, as a mask or a distance map,
* build the tree on the observations to get the distance of a grid to them,
* search across the dateline and near the poles, where lon/lat distances fail.

The distances are great circle distances, and the search is fast thanks to a KD-tree.
See also: :ref:`sphx_glr_tutorials_grids_plot_kriging.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xoa  # noqa: F401
from xoa.geo import GeoKDTree
from xoa.plot import add_colorbar, plot_field

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
pc = ccrs.PlateCarree()
map_kw = {"gridlines_labels_on": ["left", "bottom"]}


def new_map(ncols=1, width=4.5, **kwargs):
    """Create a figure with maps"""
    return plt.subplots(
        1,
        ncols,
        figsize=(width * ncols, 4),
        subplot_kw={"projection": pc},
        constrained_layout=True,
        **kwargs,
    )


# %%
# The grid is a regular one, and the observations are 8 random positions.

rng = np.random.default_rng(1)
lon = xr.DataArray(np.linspace(-6, -4, 60), dims="lon", attrs={"units": "degrees_east"})
lat = xr.DataArray(np.linspace(46, 48, 50), dims="lat", attrs={"units": "degrees_north"})
grid = xr.DataArray(
    np.zeros((lat.size, lon.size)), dims=("lat", "lon"), coords={"lon": lon, "lat": lat}
)
obs = xr.Dataset(
    coords={
        "lon": ("npts", rng.uniform(-5.8, -4.2, 8)),
        "lat": ("npts", rng.uniform(46.2, 47.8, 8)),
    }
)

# %%
# Nearest grid points
# -------------------
#
# The tree is built once. A query returns the distances and the indices along each
# dimension of the grid, which are used to select the grid points pointwise.

tree = GeoKDTree(grid)
dist, idx = tree.query(obs, k=1, units="km")
nearest = grid.isel(idx)
print(dist.values.round(1))
print(idx)

# %%
# Each observation is linked to its nearest grid point.

fig, ax = new_map()
ax.set_extent([-6, -4, 46, 48], crs=pc)
ax.gridlines(draw_labels=True)
ax.plot(
    grid.lon.values[None].repeat(lat.size, 0),
    grid.lat.values[:, None].repeat(lon.size, 1),
    ",",
    color="0.7",
    transform=pc,
)
ax.plot([obs.lon, nearest.lon], [obs.lat, nearest.lat], "-", color="C3", lw=1, transform=pc)
ax.plot(obs.lon, obs.lat, "ko", ms=4, transform=pc, label="Observations")
ax.plot(nearest.lon, nearest.lat, "r+", ms=8, transform=pc, label="Nearest points")
ax.legend(loc="upper right")
ax.set_title("Nearest grid points")

# %%
# k nearest neighbors
# -------------------
#
# With ``k`` greater than one, the distances and indices get a ``neighbor`` dimension,
# sorted by increasing distance. The grid is coarsened to make the neighbors visible.

coarse = grid.isel(lon=slice(None, None, 6), lat=slice(None, None, 6))
ctree = GeoKDTree(coarse)
target = (-5.0, 47.0)
dist, idx = ctree.query(target, k=6, units="km")
neigh = coarse.isel(idx)
print(dist.values.round(1))

fig, ax = new_map()
ax.plot(
    coarse.lon.values[None].repeat(coarse.lat.size, 0),
    coarse.lat.values[:, None].repeat(coarse.lon.size, 1),
    ".",
    color="0.7",
    transform=pc,
)
sc = ax.scatter(neigh.lon, neigh.lat, c=dist, cmap="viridis_r", s=60, transform=pc, zorder=3)
ax.plot(*target, "r*", ms=12, transform=pc, label="Target")
for i in range(dist.size):
    ax.text(float(neigh.lon[i]), float(neigh.lat[i]) + 0.04, str(i + 1), transform=pc, ha="center")
add_colorbar(sc, ax, label="Distance [km]")
ax.legend(loc="upper right")
ax.set_title("The 6 nearest neighbors")

# %%
# Limiting the distance
# ---------------------
#
# With ``max_dist``, the neighbors that are too far are missing: their distance is infinite
# and their index is -1. Do not use these indices to select data.

dist, idx = ctree.query(target, k=6, max_dist=15, units="km")
print(dist.values.round(1))
print(idx["lon"].values)

# %%
# Within a distance
# -----------------
#
# The same tree is queried again, for the grid points that are within 30 km
# of at least one observation. With ``output="dist"``, we get the distance to the closest
# observation on the grid, and nan elsewhere. The ``"mask"`` output is its boolean version,
# and ``"index"`` gives the neighbors of each observation.

dist30 = tree.query_radius(obs, 30, units="km", output="dist")
mask30 = tree.query_radius(obs, 30, units="km", output="mask")
print(int(mask30.sum()), "points within 30 km")

fig, axes = new_map(2)
m = plot_field(
    dist30,
    ax=axes[0],
    title="Distance [km]",
    cmap="Blues_r",
    map_kw=map_kw,
    add_colorbar=False,
)
add_colorbar(m, axes[0], label="Distance [km]")
plot_field(
    mask30.astype("d"),
    ax=axes[1],
    title="Mask",
    cmap="Blues",
    vmin=0,
    vmax=1.5,
    map_kw=map_kw,
    add_colorbar=False,
)
for ax in axes:
    ax.plot(obs.lon, obs.lat, "r.", transform=pc)

# %%
# The mask is handy to select data, for instance to average a field around the observations.
# Here is the same query for several radii, which reuses the tree.

fig, axes = new_map(3, width=3.4)
for ax, radius in zip(axes, [10, 25, 50]):
    mask = tree.query_radius(obs, radius, units="km", output="mask")
    plot_field(
        mask.astype("d"),
        ax=ax,
        title=f"{radius} km ({int(mask.sum())} points)",
        cmap="Blues",
        vmin=0,
        vmax=1.5,
        map_kw=map_kw,
        add_colorbar=False,
    )
    ax.plot(obs.lon, obs.lat, "r.", transform=pc)

# %%
# Neighbors of each observation
# -----------------------------
#
# With ``output="index"``, a list is returned, with one item per observation,
# whose number of neighbors differs. The items are suited for pointwise selection.

found = tree.query_radius(obs, 20, units="km")
print([f["lon"].size for f in found])

fig, ax = new_map()
ax.set_extent([-6, -4, 46, 48], crs=pc)
for i, f in enumerate(found):
    sel = grid.isel(f)
    ax.plot(sel.lon, sel.lat, ".", color=f"C{i}", ms=3, transform=pc)
    ax.plot(obs.lon[i], obs.lat[i], "o", color=f"C{i}", mec="k", transform=pc)
ax.gridlines(draw_labels=True)
ax.set_title("Grid points within 20 km of each observation")

# %%
# Distance of a grid to the observations
# --------------------------------------
#
# The tree can be built on the observations instead, and queried with the grid.
# This gives the distance of every grid point to the closest observation,
# like a coverage map, and the index of this observation.
# This is also how neighbors are searched in a local interpolation of scattered data.

otree = GeoKDTree(obs)
dist, idx = otree.query(grid, k=1, units="km")
print(dist.dims, idx)

fig, axes = new_map(2)
m = plot_field(
    dist,
    ax=axes[0],
    title="Distance to the closest observation",
    cmap="magma_r",
    map_kw=map_kw,
    add_colorbar=False,
)
add_colorbar(m, axes[0], label="Distance [km]")
axes[0].plot(obs.lon, obs.lat, "c.", transform=pc)
m = plot_field(
    idx["npts"].astype("d"),
    ax=axes[1],
    title="Closest observation",
    cmap="tab10",
    vmin=0,
    vmax=10,
    map_kw=map_kw,
    add_colorbar=False,
)
axes[1].plot(obs.lon, obs.lat, "k.", transform=pc)

# %%
# The ``xoa`` accessor does the same in one call, and it caches the tree of the grid,
# so that successive calls with other targets do not rebuild it.

dist, idx = grid.xoa.get_neighbors(obs, k=3, units="km")
print(dist.dims)

# %%
# Across the dateline and near the poles
# --------------------------------------
#
# Longitudes and latitudes are converted to cartesian coordinates on the sphere, so
# there is no problem where the longitudes jump, or where the meridians converge.
# We search the neighbors of points that are on both sides of the dateline,
# and near the north pole, on a global grid.

glon = xr.DataArray(np.arange(-180, 180, 2.0), dims="lon")
glat = xr.DataArray(np.arange(-90, 91, 2.0), dims="lat")
globe = xr.DataArray(
    np.zeros((glat.size, glon.size)), dims=("lat", "lon"), coords={"lon": glon, "lat": glat}
)
targets = xr.Dataset(coords={"lon": ("p", [179.0, 0.0, -179.0]), "lat": ("p", [10.0, 89.0, -30.0])})
gtree = GeoKDTree(globe)
field = gtree.query_radius(targets, 800, units="km", output="dist")
east = int(field.sel(lon=slice(170, 180), lat=10).notnull().sum())
west = int(field.sel(lon=slice(-180, -170), lat=10).notnull().sum())
print(f"Points found at 10N on both sides of the dateline: {east} and {west}")

pc180 = ccrs.PlateCarree(central_longitude=180)
fig = plt.figure(figsize=(9, 4), constrained_layout=True)
axes = [
    fig.add_subplot(1, 2, 1, projection=pc180),
    fig.add_subplot(1, 2, 2, projection=ccrs.NorthPolarStereo()),
]
for ax in axes:
    m = plot_field(field, ax=ax, cmap="viridis_r", add_colorbar=False, vmax=800, transform=pc)
    ax.plot(targets.lon, targets.lat, "r*", ms=8, transform=pc)
axes[0].set_extent([-25, 25, -45, 25], crs=pc180)
axes[0].set_title("Across the dateline")
axes[1].set_extent([-180, 180, 78, 90], crs=pc)
axes[1].set_title("Close to the north pole")
add_colorbar(m, axes, label="Distance [km]")
