#!/usr/bin/env python
# coding: utf-8
"""
Select data in a box, a circle or a polygon
===========================================

In this tutorial, we show how to:

* get the ``isel`` indexers of a box, a circle or a polygon, on a rectangular grid,
  a curvilinear grid or a cloud of points,
* choose between the smallest window that covers the shape and a mask (``outer``),
  or the largest window with only points inside (``inner``),
* add a margin of grid points, down to the cell edges, or of a distance,
* find the nearest point, and reuse a selection for several arrays.

Distances are great circle ones, and the shapes can cross the dateline.
See also: :ref:`sphx_glr_tutorials_grids_plot_neighbors.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import xoa  # noqa: F401
from xoa.core.grid import create_rotated_grid
from xoa.geo import GeoSelector, geosel, get_geosel, with_geosel

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
pc = ccrs.PlateCarree()


def new_map(ncols=1, width=4.5, extent=(-7, -2, 45, 49.5)):
    """Create a figure with maps"""
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(width * ncols, 3.8),
        subplot_kw={"projection": pc},
        constrained_layout=True,
        squeeze=False,
    )
    for ax in axes[0]:
        ax.set_extent(extent, crs=pc)
        ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5")
    return fig, axes[0]


def plot_selection(ax, grid, sel, color="C3", title=None):
    """Plot the points of a grid, those selected and those masked"""
    lon, lat = grid.lon, grid.lat
    sub = sel.apply(grid.sst)
    ax.plot(lon.values.ravel(), lat.values.ravel(), ",", color="0.7", transform=pc)
    ax.plot(sub.lon.values.ravel(), sub.lat.values.ravel(), ".", color="0.4", ms=3, transform=pc)
    ok = sub.notnull().values.ravel()
    ax.plot(
        sub.lon.values.ravel()[ok], sub.lat.values.ravel()[ok], ".", color=color, ms=3, transform=pc
    )
    ax.set_title(title)


# %%
# Two grids are used: a rectangular one and a rotated curvilinear one.

lon = xr.DataArray(np.arange(-7.0, -2.0, 0.1), dims="lon", name="lon")
lat = xr.DataArray(np.arange(45.0, 49.5, 0.1), dims="lat", name="lat")
rect = xr.DataArray(
    np.random.default_rng(0).random((lat.size, lon.size)),
    dims=("lat", "lon"),
    coords={"lon": lon, "lat": lat},
    name="sst",
)

grid = create_rotated_grid(50, 40, -4.5, 47.0, 25.0, 4.5, 3.5)
curv = xr.Dataset(
    {"sst": (("y", "x"), np.random.default_rng(1).random(grid["lon"].shape))},
    coords={"lon": (("y", "x"), grid["lon"]), "lat": (("y", "x"), grid["lat"])},
)

# %%
# A box on a rectangular grid
# ---------------------------
#
# On a rectangular grid, a box is exactly a pair of slices, so no mask is needed.
# The result is a :class:`~xoa.geo.GeoSelection`, whose ``isel`` attribute is passed to
# :meth:`xarray.DataArray.isel`, or that is applied directly with ``apply``.
# The longitudes of a box can cross the dateline: ``lon=(170, -170)``.

sel = get_geosel(rect, lon=(-5.5, -3.5), lat=(46, 48))
print(sel)
print(sel.mask)
sub = sel.apply(rect)
print(sub.shape)

# %%
# A box on a curvilinear grid
# ---------------------------
#
# The same box is not a rectangle of indices anymore. With ``fit="outer"``, the smallest
# window that covers the box is returned with a mask of the points that are inside it,
# applied by ``apply`` (or ``mask=False`` to keep them). With ``fit="inner"``, the
# largest window with only points inside is returned, and no mask is needed.

box = dict(lon=(-5.2, -3.8), lat=(46.2, 47.4))
fig, axes = new_map(2)
for ax, fit in zip(axes, ["outer", "inner"]):
    sel = get_geosel(curv, fit=fit, **box)
    plot_selection(ax, curv, sel, title=f'fit="{fit}"')
    ax.plot([-5.2, -3.8, -3.8, -5.2, -5.2], [46.2, 46.2, 47.4, 47.4, 46.2], "k-", transform=pc)
    print(fit, sel.isel, None if sel.mask is None else int(sel.mask.sum()))

# %%
# Circles and polygons
# --------------------
#
# A circle is given by its center and radius, in meters or in kilometers with ``units``,
# and a polygon by its vertices. Several arrays are selected in a single call.

selector = GeoSelector(curv)
circle = selector.select(circle=(-4.5, 47.0, 100), units="km")
polygon = selector.select(polygon=[[-6, 46], [-4, 45.5], [-3.5, 47.5], [-5, 48.5]])

fig, axes = new_map(2)
plot_selection(axes[0], curv, circle, title="Circle of 100 km")
plot_selection(axes[1], curv, polygon, title="Polygon")
sst_circle, sst_polygon = circle.apply(curv.sst), polygon.apply(curv.sst)

# %%
# A cloud of points
# -----------------
#
# Scattered observations, like those of a ship or of drifters, share a single dimension.
# The selection is then a single array of indices, which is what ``isel`` needs, and there is
# no mask. The :func:`~xoa.geo.geosel` function selects an array directly.
# A margin in grid points makes no sense here: use a distance.

rng = np.random.default_rng(2)
obs = xr.Dataset(
    {"temp": ("obs", rng.normal(14, 2, 400))},
    coords={"lon": ("obs", rng.uniform(-7, -2, 400)), "lat": ("obs", rng.uniform(45, 49.5, 400))},
)
box = geosel(obs, lon=(-6, -4.5), lat=(46, 47.5))
circle = geosel(obs, circle=(-3.5, 48, 70), units="km")
polygon = geosel(obs, polygon=[[-6, 46], [-4, 45.5], [-3.5, 47.5], [-5, 48.5]])
padded = geosel(obs, circle=(-3.5, 48, 70), units="km", pad=30, pad_units="km")
print(box.sizes["obs"], circle.sizes["obs"], padded.sizes["obs"])

fig, axes = new_map(3, width=3.4)
for ax, sub, title in zip(axes, [box, circle, polygon], ["Box", "Circle", "Polygon"]):
    ax.plot(obs.lon, obs.lat, ".", color="0.7", ms=3, transform=pc)
    ax.plot(sub.lon, sub.lat, ".", color="C3", ms=4, transform=pc)
    ax.set_title(title)
axes[1].plot(padded.lon, padded.lat, ".", color="C1", ms=3, transform=pc, zorder=0.5)
axes[1].set_title("Circle, and with 30 km margin")

# %%
# The nearest observations to targets are found the same way, and the result is a pointwise
# selection.

sel = get_geosel(obs, nearest=([-6.0, -4.0], [46.0, 48.0]), units="km")
print(obs.isel(sel.isel).temp.values, sel.dist.values.round(1))

# %%
# Margins
# -------
#
# The ``pad`` argument grows the shape. In grid points, with ``pad_units="points"``, it uses
# a local estimate of the grid step, from the cell edges, so it follows variable and rotated
# cells. ``pad=1`` adds about one point around the shape, and ``pad="edges"`` (which is 0.5)
# selects the points whose cell touches the shape, which is what is needed to draw
# the selection with ``pcolormesh``. A distance is given with ``pad_units="km"``.

fig, axes = new_map(3, width=3.4)
for ax, pad, pad_units in zip(axes, [0, "edges", 30], ["points", "points", "km"]):
    sel = get_geosel(curv, circle=(-4.5, 47.0, 60), units="km", pad=pad, pad_units=pad_units)
    plot_selection(ax, curv, sel, title=f"pad={pad!r} {pad_units}")

# %%
# Nearest point
# -------------
#
# A scalar target gives indices, like ``isel(lon=35, lat=45)``. Several targets give a
# pointwise selection along a ``target`` dimension, and the distances to them.

sel = get_geosel(rect, nearest=(-4.53, 47.02), units="km")
print(sel.isel, float(sel.dist))
print(rect.isel(sel.isel).values)

targets = ([-6.0, -5.0, -4.0], [46.0, 47.0, 48.0])
sel = get_geosel(curv, nearest=targets, units="km")
print(curv.isel(sel.isel).sst.values, sel.dist.values.round(1))

# %%
# Reusing a selector, and the decorator
# -------------------------------------
#
# The selector analyses the grid once and makes as many selections as needed, which is
# also done by the ``get_geosel`` method of the ``xoa`` accessor, that caches it, while its
# ``geosel`` method applies the selection to the object directly.
# The :func:`~xoa.geo.with_geosel` decorator adds a ``geosel`` argument to any function
# that takes an array or dataset as first argument.


@with_geosel
def mean_sst(ds):
    return float(ds.sst.mean())


print(mean_sst(curv))
print(mean_sst(curv, geosel=dict(circle=(-4.5, 47.0, 50), units="km")))
print(curv.xoa.geosel(lon=(-5, -4), lat=(46, 47)).sst.shape)
print(curv.xoa.geosel(lon=-4.5, lat=46.5))