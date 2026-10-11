#!/usr/bin/env python
# coding: utf-8
"""
Vorticity, strain and eddies in Gulf Stream currents
====================================================

In this tutorial, we show how to:

* find the currents and the sea surface height of a reanalysis with the naming specifications,
* compute the resolution of the grid in meters,
* compute the relative vorticity, the divergence, the strain and the Okubo-Weiss parameter,
* compare the reanalysis currents with the geostrophic and cyclogeostrophic currents derived
  from the sea surface height,
* plot currents as arrows over a field,
* mask the cells that are inside a polygon, with a threshold on their covered fraction,
  and average the diagnostics there.

The data are surface fields of the GLORYS12 reanalysis of the Copernicus Marine Service,
averaged over 2 x 2 grid cells, on the 1st of October 2019, in the region of the Gulf Stream
where the current meanders and sheds rings.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean  # noqa
import contourpy

import xoa
import xoa.meta as xmeta
import xoa.dyn as xdyn
import xoa.grid as xgrid
import xoa.mask as xmask
from xoa.core.poly import points_in_polygon
from xoa.plot import plot_field, plot_vectors, add_colorbar

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Register the internal Mercator naming specifications: they know the names of the
# Copernicus Marine variables, like ``uo``, ``vo`` and ``zos``, and they are used
# automatically for the files of the Mercator and GLORYS products.

xmeta.register_meta_specs(xoa.get_meta_config_file("mercator"))

# %%
# Read the reanalysis
# -------------------
#
# The dataset is decoded to use generic names, and the surface layer is selected.
# The currents and the sea surface height are on the same regular longitude and latitude
# grid, which is what the diagnostics of :mod:`xoa.dyn` need: they are centered finite
# differences that work on collocated grids.

ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc")
ds = ds.xoa.decode().isel(time=0, depth=0)
print(ds)

# %%
# The variables are found whatever their names, thanks to the specifications.
# The speed is added for plotting.

u, v, ssh = ds.xoa.get("u"), ds.xoa.get("v"), ds.xoa.get("ssh")
lon, lat = u.lon, u.lat
for da, name in (u, "Zonal velocity"), (v, "Meridional velocity"):
    da.attrs["long_name"] = name
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
# with :func:`xoa.plot.plot_vectors`, which under-samples them to keep the figure readable:
# here, there is one arrow every two grid points, so that each eddy still gets several.

MAP_KW = {"gridlines_labels_on": ["bottom", "left"], "land": True}
SPEED_KW = {
    "method": "contourf",
    "cmap": "cmo.tempo",
    "levels": np.linspace(0, 2, 21),
    "extend": "max",
}


def new_map(nrows=1, ncols=1, **kwargs):
    kwargs.setdefault("constrained_layout", True)
    kwargs.setdefault("subplot_kw", {"projection": ccrs.Mercator()})
    return plt.subplots(nrows, ncols, **kwargs)


fig, ax = new_map(figsize=(8.5, 6.2))
plot_field(speed, ax=ax, map_kw=MAP_KW, **SPEED_KW)
plot_vectors(u, v, ax=ax, color="0.15", stride=2)
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

fig, axes = new_map(2, 2, figsize=(12, 9))
kw = {"map_kw": MAP_KW, "method": "contourf", "extend": "both"}
panels = [
    (rossby, {"cmap": "cmo.balance", "levels": np.linspace(-0.6, 0.6, 25), "title": "Rossby number"}),
    (div_n, {"cmap": "cmo.balance", "levels": np.linspace(-0.06, 0.06, 25), "title": "Divergence"}),
    (strain_n, {"cmap": "cmo.amp", "levels": np.linspace(0, 0.5, 21), "title": "Strain"}),
    (ow_n, {"cmap": "cmo.curl_r", "levels": np.linspace(-3, 3, 25), "title": "Okubo-Weiss"}),
]
for ax, (field, pkw) in zip(axes.ravel(), panels):
    plot_field(field, ax=ax, overlay_contours=core, **kw, **pkw)

# %%
# The black lines enclose the cells where Okubo-Weiss is lower than the threshold.
# They mostly match the strong vorticity patches, as expected.

# %%
# Geostrophic currents
# --------------------
#
# The geostrophic currents are derived from the sea surface height with
# :func:`xoa.dyn.get_geos`, which uses the Coriolis parameter computed from the latitudes.
# Let's compare them with the currents of the reanalysis.

ug, vg = xdyn.get_geos(ssh)
speed_g = np.hypot(ug, vg).rename("speed")
speed_g.attrs = speed.attrs
ageos = np.hypot(u - ug, v - vg).rename("ageos")
ageos.attrs = {"long_name": "Ageostrophic speed", "units": "m s-1"}

fig, axes = new_map(1, 2, figsize=(14, 6))
for ax, title, (fu, fv, fs) in zip(
    axes, ("Reanalysis", "Geostrophy"), ((u, v, speed), (ug, vg, speed_g))
):
    mappable = plot_field(fs, ax=ax, title=title, map_kw=MAP_KW, add_colorbar=False, **SPEED_KW)
    plot_vectors(fu, fv, ax=ax, color="0.15", stride=2, key=title == "Reanalysis")
add_colorbar(mappable, axes, da=speed)

# %%
# The currents are mostly geostrophic, but the flow is strongly curved in the meanders and the
# rings, so that the advection terms of the momentum equations matter too.

# %%
# Cyclogeostrophic currents
# -------------------------
#
# The cyclogeostrophic currents include the advection terms in the balance with the pressure
# gradient and the Coriolis force. They are computed by :func:`xoa.dyn.get_cyclogeos`
# with an iterative method like the one of
# `Penven et al. (2014) <https://doi.org/10.1002/2013JC009528>`_: starting from the
# geostrophic currents, they are corrected with the advection terms of the previous iteration
# until they no longer change. By default, the correction has the four terms of the complete
# advection. With ``terms="penven"``, it only has the two terms :math:`u \\partial_x u` and
# :math:`v \\partial_y v`, which are the ones that are usually taught as the ones of the paper.
# The currents are missing where the iterations do not converge, like in very strong
# anticyclones that have no cyclogeostrophic solution.

ucf, vcf = xdyn.get_cyclogeos(ssh)
ucg, vcg = xdyn.get_cyclogeos(ssh, terms="penven")
print(f"Missing points: {int(ucf.isnull().sum())} and {int(ucg.isnull().sum())},"
      f" against {int(ug.isnull().sum())} for geostrophy")

# %%
# The error of each estimate is the norm of its difference with the currents of the reanalysis.
# They are compared on the points where all of them are defined.

estimates = {
    "Geostrophy": (ug, vg),
    "Cyclogeostrophy (two terms)": (ucg, vcg),
    "Cyclogeostrophy (full advection)": (ucf, vcf),
}
valid = u.notnull() & v.notnull()
for eu, ev in estimates.values():
    valid &= eu.notnull() & ev.notnull()
errors = {
    name: np.hypot(eu - u, ev - v).where(valid).rename("error")
    for name, (eu, ev) in estimates.items()
}
for da in errors.values():
    da.attrs = {"long_name": "Speed error", "units": "m s-1"}

fig, axes = new_map(1, 3, figsize=(15, 5.2))
kw = {
    "method": "contourf",
    "cmap": "cmo.matter",
    "levels": np.linspace(0, 0.4, 17),
    "extend": "max",
    "map_kw": MAP_KW,
    "add_colorbar": False,
}
for ax, (name, err) in zip(axes, errors.items()):
    mappable = plot_field(err, ax=ax, title=name, **kw)
add_colorbar(mappable, axes, da=errors["Geostrophy"])

# %%
# The errors are the largest along the stream and in the rings, where the advection terms of
# the cyclogeostrophic balance are significant, and the correction reduces them, in the strongest
# ring for instance. Points where an estimate is missing are left blank.
# Let's average the errors over regions, that are
# defined with the Rossby number and the Okubo-Weiss parameter that were computed above:
# the cores of the vortices, and the cyclonic and anticyclonic areas.
# The complete advection makes a much larger correction than the two terms.
# Note that the reanalysis currents have other ageostrophic components, like the Ekman
# drift, that are not in these balances, so that the errors do not vanish.

ow_core = (ow / ow.std()) < -0.2
regions = {
    "Everywhere": valid,
    "Vortex cores": valid & ow_core,
    "Cyclonic": valid & (rossby > 0.15),
    "Anticyclonic": valid & (rossby < -0.15),
}
table = pd.DataFrame(
    {
        name: {
            region: float(np.sqrt((err**2).where(mask_).mean()))
            for region, mask_ in regions.items()
        }
        for name, err in errors.items()
    }
)
print("Root mean square of the speed error [m/s]")
print(table.round(3).to_string())

# %%
# The complete advection reduces the error everywhere, especially in the cyclonic areas,
# where geostrophy overestimates the speed of the rotation. The two terms
# change the currents much less.
# The remaining error comes from the other ageostrophic components of the reanalysis,
# like the Ekman drift, and from the time averaging.

# %%
# Averaging inside a polygon
# --------------------------
#
# Let's average the diagnostics inside the strongest vortex, whose outline is the closed
# Okubo-Weiss contour that surrounds the most negative value.
# The outline is extracted with :mod:`contourpy` from the longitudes and latitudes.

imin = np.unravel_index(np.nanargmin(ow_n.values), ow_n.shape)
xmin, ymin = float(lon.values[imin[1]]), float(lat.values[imin[0]])
generator = contourpy.contour_generator(lon.values, lat.values, ow_n.fillna(0).values)
lines = [line for line in generator.lines(OW_THRESHOLD) if np.allclose(line[0], line[-1])]
inside = [bool(points_in_polygon(xmin, ymin, line)) for line in lines]
polygon = lines[inside.index(True)]
print(f"The polygon has {len(polygon)} vertices")

# %%
# :func:`xoa.mask.get_polygon_mask` tells which cells are inside the polygon.
# A cell is selected when the fraction of its area that is covered by the polygon
# is at least the ``threshold`` parameter, so that cells that are cut by the polygon
# can be kept or not. A null threshold selects all the cells that touch the polygon.
# With a ``None`` threshold, only the position of the centers matters.

thresholds = [None, 0.0, 0.5, 1.0]
masks = {t: xmask.get_polygon_mask(rossby, polygon, threshold=t) for t in thresholds}
x0, y0 = polygon.min(axis=0) - 0.35
x1, y1 = polygon.max(axis=0) + 0.35
fig, axes = new_map(2, 2, figsize=(10, 6))
kw = {"cmap": "cmo.balance", "vmin": -0.6, "vmax": 0.6, "map_kw": MAP_KW, "add_colorbar": False}
for ax, (t, m) in zip(axes.ravel(), masks.items()):
    label = "center inside" if t is None else "touching" if t == 0 else f"fraction >= {t:g}"
    plot_field(rossby, ax=ax, alpha=0.25, **kw)
    mappable = plot_field(rossby.where(m), ax=ax, title=f"{label}: {int(m.sum())} cells", **kw)
    ax.plot(*polygon.T, color="k", lw=1.6, transform=ccrs.PlateCarree())
    ax.set_extent([x0, x1, y0, y1], crs=ccrs.PlateCarree())
add_colorbar(mappable, axes, da=rossby)

# %%
# The more demanding the threshold, the smaller the mask: from all the cells that touch
# the polygon to the ones that are entirely inside it.
# The selected cells, in full colors, can now be used to average the diagnostics over the
# vortex. The sign of the Rossby number tells its rotation, which is cyclonic when it is positive.

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
