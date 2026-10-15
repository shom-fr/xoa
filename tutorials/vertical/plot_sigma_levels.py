#!/usr/bin/env python
# coding: utf-8
"""
Sigma levels over a slope: choose your stretching
=================================================

In this tutorial, we show how to:

* compute the heights of the levels of a pure sigma coordinate and of a stretched
  s-coordinate over a shelf break,
* see how the stretching parameters concentrate the levels near the surface or the bottom,
* get the stretching curve and use it with the functions of :mod:`xoa.sigma`.

The data are synthetic: a shelf, a slope and an abyssal plain, with a flat sea surface.
The concepts are explained in the :ref:`indepth.vertical` guide.
See also: :ref:`sphx_glr_tutorials_vertical_plot_croco_section.py`,
:ref:`sphx_glr_tutorials_vertical_plot_croco_uv_depths.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from xoa.sigma import get_cs, ocean_s_coordinate_g2, ocean_sigma_coordinate

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# The bathymetry
# --------------
#
# A shelf of 100 m that deepens to 3000 m, along a 200 km section.

x = xr.DataArray(np.linspace(0, 200, 201), dims="x", name="x", attrs={"units": "km"})
bathy = (100.0 + 2900.0 * 0.5 * (1 + np.tanh((x - 100) / 20))).rename("bathy")
ssh = xr.zeros_like(bathy).rename("ssh")
nlev = 20
sig = xr.DataArray((np.arange(nlev) + 0.5) / nlev - 1.0, dims="sig", name="sig")

# %%
# Pure sigma levels
# -----------------
#
# The levels are evenly spread between the sea surface and the bottom
# (:func:`xoa.sigma.ocean_sigma_coordinate`): they are very thin over the shelf and thick in
# deep water, which does not resolve the surface layer there.

z_sigma = ocean_sigma_coordinate(sig, ssh, bathy).transpose("sig", "x")
print(z_sigma.dims, float(z_sigma.isel(x=0, sig=-1) - z_sigma.isel(x=0, sig=-2)))

# %%
# Stretched s-coordinates
# -----------------------
#
# The generic form 2 of the s-coordinate (:func:`xoa.sigma.ocean_s_coordinate_g2`) has a
# critical depth ``hc`` above which the levels are evenly spread, and
# two parameters ``thetas`` and ``thetab`` that stretch the levels towards the surface and the
# bottom, between 0 and 1 for the latter. The stretching curve ``Cs`` is given by
# :func:`xoa.sigma.get_cs`.

params = {
    "thetas=1, thetab=0": dict(thetas=1.0, thetab=0.0, hc=100.0),
    "thetas=7, thetab=0": dict(thetas=7.0, thetab=0.0, hc=100.0),
    "thetas=7, thetab=0.5": dict(thetas=7.0, thetab=0.5, hc=100.0),
}
levels = {"Pure sigma": z_sigma}
curves = {}
for label, p in params.items():
    cs = get_cs(sig, p["thetas"], p["thetab"])
    curves[label] = cs
    levels[label] = ocean_s_coordinate_g2(sig, ssh, bathy, p["hc"], cs=cs).transpose("sig", "x")

# %%
# Their stretching curves show how much of the water column is covered by the
# upper levels: the larger ``thetas``, the more levels near the surface.

fig, ax = plt.subplots(figsize=(4.5, 3.6), constrained_layout=True)
ax.plot(sig, sig, "k--", label="Pure sigma")
for label, cs in curves.items():
    ax.plot(sig, cs, label=label)
ax.set_xlabel("Sigma")
ax.set_ylabel("Stretching curve")
ax.legend(frameon=False)
ax.grid(alpha=0.3)

# %%
# The levels over the section: the pure sigma levels leave 150 m between the upper levels
# in deep water, whereas the stretched coordinates keep the thin levels near the surface, and
# near the bottom with a ``thetab`` that is greater than zero.

fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True, sharey=True, constrained_layout=True)
for ax, (label, z) in zip(axes.flat, levels.items()):
    for k in range(nlev):
        ax.plot(x, z.isel(sig=k), color="0.2", lw=0.6)
    ax.fill_between(x, -bathy, -3200, color="0.7")
    ax.set_title(label)
    ax.set_ylim(-3200, 50)
for ax in axes[:, 0]:
    ax.set_ylabel("Height [m]")
for ax in axes[-1]:
    ax.set_xlabel("Distance [km]")

# %%
# The thickness of the upper level in deep water is a good indicator of the vertical resolution
# of the surface layer, which matters for the exchanges with the atmosphere.

for label, z in levels.items():
    top = float(z.isel(x=-1, sig=-1) - z.isel(x=-1, sig=-2))
    print(f"{label:20s} upper layer: {top:6.1f} m")
