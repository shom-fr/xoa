#!/usr/bin/env python
# coding: utf-8
"""
Vorticity, strain and eddies in CROCO surface currents
======================================================

In this tutorial, we show how to:

* bring the currents of an Arakawa C grid to the same grid points,
* compute the resolution of the grid in meters,
* compute the relative vorticity, the divergence, the strain and the Okubo-Weiss parameter,
* compare the model currents with the geostrophic currents derived from the sea surface height,
* plot currents as arrows over a field,
* mask the cells that are inside a polygon, with a threshold on their covered fraction,
  and average the diagnostics there.

The data are surface outputs of a CROCO simulation of the Agulhas current region, south of Africa,
where the current meanders and sheds eddies.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean  # noqa
import contourpy

import xoa
import xoa.meta as xmeta
import xoa.dyn as xdyn
import xoa.grid as xgrid
from xoa.core.poly import points_in_polygon
from xoa.plot import plot_field, plot_vectors, add_colorbar

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Register the internal CROCO naming specifications, to find the variables
# and coordinates whatever their names.

xmeta.register_meta_specs(xoa.get_meta_config_file("croco"))

# %%
# Read the model
# --------------
#
# This sample contains a single time step of the surface layer of the simulation.
# The currents are not defined at the same positions as the sea surface height:
# ``u`` is on the cell edges along x, and ``v`` on the cell edges along y.

ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.surf.nc")
ds = ds.xoa.decode().isel(time=0, sig_rho=0)
print(ds.u.dims, ds.v.dims, ds.ssh.dims)

# %%
# Currents at the grid points
# ---------------------------
#
# The diagnostics of :mod:`xoa.dyn` are centered finite differences that work on
# collocated grids. We thus interpolate ``u`` and ``v`` to the points
# of the sea surface height, with a linear extrapolation on the first column or row.
# The land points are masked with the land-sea mask of the model.

mask = ds.mask_rho
lon, lat = ds.ssh.lon_rho, ds.ssh.lat_rho
u = (
    ds.u.drop_vars(["lon_u", "lat_u"])
    .interp(x_u=ds.x_rho, kwargs={"fill_value": "extrapolate"})
    .assign_coords(lon_rho=lon, lat_rho=lat)
    .where(mask)
    .rename("u")
)
v = (
    ds.v.drop_vars(["lon_v", "lat_v"])
    .interp(y_v=ds.y_rho, kwargs={"fill_value": "extrapolate"})
    .assign_coords(lon_rho=lon, lat_rho=lat)
    .where(mask)
    .rename("v")
)
for da in u, v:
    da.attrs = {"units": "m s-1", "long_name": da.name}
u.attrs["long_name"] = "Zonal velocity"
v.attrs["long_name"] = "Meridional velocity"
ssh = ds.ssh.where(mask)
speed = np.hypot(u, v).rename("speed")
speed.attrs = {"long_name": "Speed", "units": "m s-1"}

# %%
# Resolution of the grid
# ----------------------
#
# :func:`xoa.grid.get_dx_dy` computes the distance in meters between grid points along x and y,
# centered on the grid points, from the longitudes and latitudes.
# It is used by default by all the diagnostics, but you can pass your own ``dx`` and ``dy``.

dx, dy = xgrid.get_dx_dy(u)
print(f"dx = {float(dx.mean()) * 1e-3:.1f} km, dy = {float(dy.mean()) * 1e-3:.1f} km")

# %%
# A first look at the currents
# ----------------------------
#
# The speed is drawn with :func:`xoa.plot.plot_field` and the currents are added as arrows
# with :func:`xoa.plot.plot_vectors`, which under-samples them to keep the figure readable.
# The coast is drawn from the land-sea mask of the model.

MAP_KW = {"gridlines_labels_on": ["bottom", "left"]}
COAST = [{"field": mask, "levels": [0.5], "colors": "0.25", "linewidths": 0.8}]
SPEED_KW = {"cmap": "cmo.tempo", "vmin": 0, "vmax": 1, "overlay_contours": COAST}


def new_map(nrows=1, ncols=1, **kwargs):
    kwargs.setdefault("constrained_layout", True)
    kwargs.setdefault("subplot_kw", {"projection": ccrs.Mercator()})
    return plt.subplots(nrows, ncols, **kwargs)


fig, ax = new_map(figsize=(8.5, 4.6))
mappable = plot_field(speed, ax=ax, map_kw=MAP_KW, **SPEED_KW)
plot_vectors(u, v, ax=ax, color="0.1", min_spacing=22)
ax.set_title("Surface currents")

# %%
# Diagnostics
# -----------
#
# The functions of :mod:`xoa.dyn` are also available as methods of the ``xoa`` accessor
# of a dataset that contains the currents.
# All of them are computed with numba and accept extra leading dimensions, like time,
# and dask arrays.

uv = xr.Dataset({"u": u, "v": v})
relvort = uv.xoa.get_relvort()
div = uv.xoa.get_div()
sn, ss = uv.xoa.get_strain()
ow = uv.xoa.get_okuboweiss()
print(relvort)

# %%
# These quantities are best compared when normalized by the Coriolis parameter,
# which is given by :func:`xoa.dyn.get_coriolis`.
# The normalized relative vorticity is the Rossby number, which is positive for cyclonic
# rotation and negative for anticyclonic rotation in both hemispheres.
# The Okubo-Weiss parameter is normalized by its standard deviation.
# Rotation dominates strain where it is lower than a negative threshold, classically
# -0.2 times its standard deviation, which is where the eddy cores are.

f = xdyn.get_coriolis(lat)
rossby = (relvort / f).rename("rossby")
rossby.attrs = {"long_name": "Rossby number"}
div_n = (div / f).rename("div")
div_n.attrs = {"long_name": "Divergence / f"}
strain_n = (np.hypot(sn, ss) / abs(f)).rename("strain")
strain_n.attrs = {"long_name": "Strain / |f|"}
ow_n = (ow / ow.std()).rename("ow")
ow_n.attrs = {"long_name": "Okubo-Weiss / std"}
OW_THRESHOLD = -0.2
core = [
    {"field": ow_n, "levels": [OW_THRESHOLD], "colors": "k", "linewidths": 0.9, "linestyles": "-"}
]

fig, axes = new_map(2, 2, figsize=(12, 7.2))
kw = {"map_kw": MAP_KW}
panels = [
    (rossby, {"cmap": "cmo.balance", "vmin": -0.5, "vmax": 0.5, "title": "Rossby number"}),
    (div_n, {"cmap": "cmo.balance", "vmin": -0.15, "vmax": 0.15, "title": "Divergence"}),
    (strain_n, {"cmap": "cmo.amp", "vmin": 0, "vmax": 0.5, "title": "Strain"}),
    (ow_n, {"cmap": "cmo.curl_r", "vmin": -3, "vmax": 3, "title": "Okubo-Weiss"}),
]
for ax, (field, pkw) in zip(axes.ravel(), panels):
    plot_field(field, ax=ax, overlay_contours=COAST + core, **kw, **pkw)

# %%
# The black lines enclose the cells where Okubo-Weiss is lower than the threshold.
# They mostly match the strong vorticity patches, as expected.

# %%
# Geostrophic currents
# --------------------
#
# The geostrophic currents are derived from the sea surface height with
# :func:`xoa.dyn.get_geos`, which uses the Coriolis parameter computed from the latitudes.
# Let's compare them with the model currents.

ug, vg = xdyn.get_geos(ssh)
speed_g = np.hypot(ug, vg).rename("speed")
speed_g.attrs = speed.attrs
ageos = np.hypot(u - ug, v - vg).rename("ageos")
ageos.attrs = {"long_name": "Ageostrophic speed", "units": "m s-1"}

fig, axes = new_map(1, 3, figsize=(15, 4.4))
for ax, title, (fu, fv, fs) in zip(
    axes[:2], ("Model", "Geostrophy"), ((u, v, speed), (ug, vg, speed_g))
):
    plot_field(fs, ax=ax, title=title, map_kw=MAP_KW, **SPEED_KW)
    plot_vectors(fu, fv, ax=ax, color="0.1", min_spacing=22, key=title == "Model")
plot_field(
    ageos,
    ax=axes[2],
    cmap="cmo.matter",
    vmin=0,
    vmax=0.3,
    title="Model - geostrophy",
    map_kw=MAP_KW,
    overlay_contours=COAST,
)

# %%
# The currents are mostly geostrophic, with differences that are small compared to
# the strong currents and that grow along the coast and the domain boundaries.

# %%
# Averaging inside a polygon
# --------------------------
#
# Let's average the diagnostics inside the strongest vortex, whose outline is the closed
# Okubo-Weiss contour that surrounds the most negative value.
# The outline is extracted with :mod:`contourpy` from the longitudes and latitudes.

imin = np.unravel_index(np.nanargmin(ow_n.values), ow_n.shape)
xmin, ymin = float(lon.values[imin]), float(lat.values[imin])
generator = contourpy.contour_generator(lon.values, lat.values, ow_n.fillna(0).values)
lines = [line for line in generator.lines(OW_THRESHOLD) if np.allclose(line[0], line[-1])]
inside = [bool(points_in_polygon(xmin, ymin, line)) for line in lines]
polygon = lines[inside.index(True)]
print(f"The polygon has {len(polygon)} vertices")

# %%
# :func:`xoa.grid.get_polygon_mask` tells which cells are inside the polygon.
# A cell is selected when the fraction of its area that is covered by the polygon
# is at least the ``threshold`` parameter, so that cells that are cut by the polygon
# can be kept or not. A null threshold selects all the cells that touch the polygon.
# With a ``None`` threshold, only the position of the centers matters.

thresholds = [None, 0.0, 0.5, 1.0]
masks = {t: xgrid.get_polygon_mask(rossby, polygon, threshold=t) for t in thresholds}
x0, y0 = polygon.min(axis=0) - 0.35
x1, y1 = polygon.max(axis=0) + 0.35
fig, axes = new_map(2, 2, figsize=(9.5, 7.6))
kw = {"cmap": "cmo.balance", "vmin": -0.5, "vmax": 0.5, "map_kw": MAP_KW, "add_colorbar": False}
for ax, (t, m) in zip(axes.ravel(), masks.items()):
    label = "center inside" if t is None else "touching" if t == 0 else f"fraction >= {t:g}"
    plot_field(rossby, ax=ax, alpha=0.25, **kw)
    mappable = plot_field(rossby.where(m), ax=ax, title=f"{label}: {int(m.sum())} cells", **kw)
    ax.plot(*polygon.T, color="k", lw=1.4, transform=ccrs.PlateCarree())
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
add_colorbar(mappable, axes, da=rossby)

# %%
# The more demanding the threshold, the smaller the mask: from all the cells that touch
# the polygon to the ones that are entirely inside it.
# The selected cells, in full colors, can now be used to average the diagnostics over the
# vortex, which is cyclonic since its Rossby number is positive.

for t, m in masks.items():
    label = "centers" if t is None else f"{t:g}"
    print(
        f"threshold {label:>7}: {int(m.sum()):3d} cells,"
        f" Rossby = {float(rossby.where(m).mean()):+.3f},"
        f" Okubo-Weiss = {float(ow_n.where(m).mean()):+.2f}"
    )

# %%
# Polygons are not limited to vortices: you can use any shape, like the contour of a region
# or a shapely geometry, to extract or average any field. The ``get_polygon_mask`` method of the
# ``xoa`` accessor does the same from a data array or a dataset.
plt.show()
