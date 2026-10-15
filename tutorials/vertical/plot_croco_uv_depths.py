#!/usr/bin/env python
# coding: utf-8
"""
Depths of the sigma levels at the u and v points of a CROCO grid
================================================================

In this tutorial, we show how to compute the ``z`` heights at the u and v points
of a staggered CROCO grid, in two ways:

* move the sea surface height and the bathymetry to the u and v points, then compute ``z``,
* compute ``z`` at the rho points, then move it to the u and v points.

The concepts are explained in the :ref:`indepth.vertical` guide.
See also: :ref:`sphx_glr_tutorials_vertical_plot_croco_section.py`.
"""

# %%
# Initialisations
# ---------------

import xarray as xr
import matplotlib.pyplot as plt

import xoa
import xoa.meta as xmeta
from xoa.grid import shift
from xoa.sigma import decode_sigma

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
xmeta.register_meta_specs(xoa.get_meta_config_file("croco"))

# %%
# Read the model
# --------------
#
# The currents ``u`` and ``v`` are not at the same points as the sea surface height ``zeta``
# and the bathymetry ``h``, which are at the rho points.
# We compute the heights at the rho points, as in the :ref:`sigma section tutorial
# <sphx_glr_tutorials_vertical_plot_croco_section.py>`.

ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.surf.nc")
ds = ds.xoa.decode_sigma().isel(time=0)
print(ds.u.dims, ds.v.dims, ds.z_rho.dims)

# %%
# From rho to u and v points
# --------------------------
#
# A u point is half a cell east of a rho point and a v point is half a cell north.
# The value at a u or v point is the mean of the two surrounding rho points.
# On the full grid of a CROCO output, which has one u point less along ``xi`` and one v point
# less along ``eta``, this is what :func:`xoa.grid.get_centers` does.
# Our sample is a subsampled extraction that has as many u points as rho points,
# so we use :func:`xoa.grid.shift`, which pads the last column or row.
# Both are the same operation, so the helpers below can be used with either.
#
# .. note:: The sample was extracted with a stride of two along ``xi_rho`` and ``eta_rho``
#    (indices 86, 88, 90, ... in the model) and the u and v indices were kept
#    half a step further (86.5, 88.5, ...).
#    The rho neighbours of a u point are thus two cells apart here and the
#    next one is not in the file, so the values at u and v points are only illustrative.
#    On a full output, ``xi_u`` is ``xi_rho[:-1] + 0.5`` and ``get_centers`` aligns them.


def rho2u(da):
    return shift(da, {"xi_rho": "right"}).rename(xi_rho="xi_u").assign_coords(xi_u=ds.xi_u)


def rho2v(da):
    return shift(da, {"eta_rho": "right"}).rename(eta_rho="eta_v").assign_coords(eta_v=ds.eta_v)


# %%
# Method 1: move the sea surface height and the bathymetry, then compute z
# ------------------------------------------------------------------------
#
# :func:`xoa.sigma.decode_sigma` looks for the formula terms at a given horizontal location
# with the suffixed names ``zeta_u``, ``h_u`` and ``hc_u``, so we create them by renaming
# the moved arrays.
# The sigma at the w levels would need an invalid ``uw`` location, so we drop it.

hlocs = {"u": rho2u, "v": rho2v}
ds1 = ds.drop_vars(["sc_w", "Cs_w", "s_w"])
for loc, rho2x in hlocs.items():
    ds1[f"zeta_{loc}"] = rho2x(ds.zeta)
    ds1[f"h_{loc}"] = rho2x(ds.h)
    ds1[f"hc_{loc}"] = ds.hc
ds1 = decode_sigma(ds1, hlocs=list(hlocs))
z1 = {loc: ds1[f"z_{loc}"] for loc in hlocs}
print(z1["u"].dims)

# %%
# Method 2: compute z at rho points, then move it
# -----------------------------------------------

z2 = {"u": rho2u(ds.z_rho), "v": rho2v(ds.z_rho)}

# %%
# Compare the two methods
# -----------------------
#
# The two methods differ because ``z`` is not linear in the bathymetry,
# but the difference is small compared to the depth.
# Method 2 is the cheapest since ``z`` is computed once.
# Method 1 gives the heights that the formula yields from the bathymetry and sea surface height
# at these points.

for loc in "uv":
    diff = abs(z1[loc] - z2[loc])
    print(f"{loc}: max difference = {float(diff.max()):.2f} m "
          f"for max depth = {float(abs(z2[loc]).max()):.0f} m")

# %%
# The heights of the level near the bottom, along a west-east line across the slope. The u points
# are between the rho points, and the two methods are hardly distinguishable here: the differences
# are about ten centimeters at most at this level.

row = dict(eta_rho=14, s_rho=1)
fig, ax = plt.subplots(figsize=(7, 3.2), constrained_layout=True)
ax.plot(ds.lon_rho.isel(eta_rho=14), ds.z_rho.isel(**row), "o", ms=4, color="0.3", label="rho points")
ax.plot(ds.lon_u.isel(eta_rho=14), z1["u"].isel(**row), "s", ms=4, mfc="none", label="u, method 1")
ax.plot(ds.lon_u.isel(eta_rho=14), z2["u"].isel(**row), "x", ms=4, label="u, method 2")
ax.set_xlabel("Longitude [°E]")
ax.set_ylabel("Height [m]")
ax.legend()
ax.grid(alpha=0.3)

# %%
# The difference of the two methods near the bottom, over the domain.

fig, axs = plt.subplots(ncols=2, figsize=(8, 3), sharey=True, constrained_layout=True)
for ax, loc in zip(axs, "uv"):
    diff = (z1[loc] - z2[loc]).isel(s_rho=1)
    diff = diff.assign_coords(lon=ds[f"lon_{loc}"], lat=ds[f"lat_{loc}"])
    diff.plot(ax=ax, x="lon", y="lat", cbar_kwargs={"label": "m"})
    ax.set_title(f"z at {loc} points: method 1 - method 2")
    ax.set_xlabel("Longitude [°E]")
    ax.set_ylabel("Latitude [°N]")

# %%
# Which one to choose?
# --------------------
#
# Use method 2 by default: CROCO computes the heights and thicknesses at the rho points only
# (``set_depth.F``) and averages the thicknesses to the u and v points where it needs them,
# so moving ``z`` gives the thicknesses of the model.
# Use method 1 only if you need the sigma formula applied to the bathymetry and sea surface height
# at the u and v points.
