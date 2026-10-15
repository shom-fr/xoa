#!/usr/bin/env python
# coding: utf-8
"""
Depths and layer thicknesses
============================

In this tutorial, we show how to:

* integrate layer thicknesses to get the depths of the interfaces and of the layer centers,
* go back from depths to layer thicknesses,
* find or compute the thicknesses of a dataset, whatever the vertical coordinate,
* use the thicknesses as weights, to compute a depth average.

The first data are the thicknesses of the upper layers of the Hycom model, which vary with time
and position. The second are the currents of a CROCO simulation on sigma levels.
The concepts are explained in the :ref:`indepth.vertical` guide.
See also: :ref:`sphx_glr_tutorials_grids_plot_croco_section.py`,
:ref:`sphx_glr_tutorials_evaluation_plot_hycom_gdp.py`.
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
from xoa.grid import dz2depth, dz2z, get_dz, z2dz
from xoa.plot import plot_fields

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# From thicknesses to depths
# --------------------------
#
# Hycom provides the thickness of its layers, which vary in time and space.

with xmeta.set_meta_specs("hycom"):
    hycom = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.h.nc").xoa.decode()
dz = hycom.dz
print(dz.dims, dz.isel(time=0, x=0, y=0).values)

# %%
# :func:`xoa.grid.dz2z` integrates the thicknesses from the top, that is at ``z=0``, and
# gives ``z`` positive up at the interfaces: there is one more level than layers.
# :func:`xoa.grid.dz2depth` does the same with positive down depths.
# With ``centered=True``, the output is at the center of the layers.
# The reference of the integration is the surface by default, and it can be the sea surface
# height with ``top`` or the sea floor with ``bottom`` when the whole water column is described.

z_interfaces = dz2z(dz.isel(time=0))
depth_centers = dz2depth(dz.isel(time=0), centered=True)
print(z_interfaces.sizes["depth"], "interfaces for", depth_centers.sizes["depth"], "layers")
print(z_interfaces.isel(x=0, y=0).values.round(1))
print(depth_centers.isel(x=0, y=0).values.round(1))

# %%
# The layers of this hybrid-coordinate model are thin and stretch with depth, and they follow
# the changes of the mixed layer in time. Here are the interfaces at a given point.

point = dict(x=6, y=5)
zt = dz2z(dz.isel(**point))
fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
for k in range(zt.sizes["depth"]):
    ax.plot(zt.time, zt.isel(depth=k), color="0.2", lw=1)
ax.fill_between(
    zt.time, zt.isel(depth=0), zt.isel(depth=1), color=cmocean.cm.deep(0.3), alpha=0.5
)
ax.set_ylabel("Interface height [m]")
ax.set_title("Hycom layers at a point")

# %%
# From depths to thicknesses
# --------------------------
#
# :func:`xoa.grid.z2dz` does the opposite, from the depths of the layer centers. The interfaces
# are the middle of consecutive centers, and the outer ones are extrapolated, unless the
# top and the bottom are provided. The result is exact for evenly spaced levels, and
# approximate for others, like here where the layers thicken with depth.

centers = dz2z(dz.isel(time=0), centered=True)
dz_back = z2dz(centers, top=0.0)
rel_err = abs(dz_back - dz.isel(time=0)) / dz.isel(time=0)
print(f"Relative error: {float(rel_err.max()):.1%} at most")

# %%
# With sigma levels
# -----------------
#
# On the sigma levels of CROCO, the depth of each level is given at each horizontal point
# and the thicknesses are not stored. :func:`xoa.grid.get_dz` searches for a thickness
# variable, and else computes it from the depths with :func:`~xoa.grid.z2dz`, whose
# ``top`` is the sea surface height.

with xmeta.set_meta_specs("croco"):
    croco = xoa.open_data_sample("MODELS/CROCO/MED/pelops_3d.nc").xoa.decode()
croco.depth.attrs["positive"] = "up"  # The file does not say that the depths are negative
dzc = z2dz(croco.depth, top=croco.ssh)
print(dzc.dims)
print(f"Layer thicknesses from {float(dzc.min()):.1f} to {float(dzc.max()):.0f} m")

# %%
# Thicknesses are the weights of vertical integrals. The depth-averaged currents are very
# different from the surface ones: the former are the barotropic component, and
# the water columns are very different in depth over the steep topography.

u, v = croco.u, croco.v
ubar = (u * dzc).sum("sig_rho") / dzc.sum("sig_rho")
vbar = (v * dzc).sum("sig_rho") / dzc.sum("sig_rho")
speed_bar = np.hypot(ubar, vbar).rename("speed_bar")
speed_bar.attrs.update(long_name="Depth-averaged speed", units="m s-1")
speed_surf = np.hypot(u.isel(sig_rho=-1), v.isel(sig_rho=-1))  # The last level is the surface.rename("speed_surf")
speed_surf.attrs.update(long_name="Surface speed", units="m s-1")
depth_bottom = dzc.sum("sig_rho").rename("column")
depth_bottom.attrs.update(long_name="Water column thickness", units="m")
plot_fields(
    {"Surface": speed_surf, "Depth average": speed_bar, "Water column": depth_bottom},
    ncols=3,
    figsize=(10, 3.4),
    cmap=cmocean.cm.speed,
    map_kw={"land": True, "gridlines_labels_on": ["bottom", "left"]},
)
