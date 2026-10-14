#!/usr/bin/env python
# coding: utf-8
"""
Dimensionless numbers of the Gulf Stream
========================================

In this tutorial, we show how to:

* compute the Rossby number of the surface currents,
* compute the Burger number from the deformation radius of the vertical modes,
  or from the stratification,
* compute the internal Froude number of the upper ocean,
* compute the Ekman number of the surface layer.

The data are 3D fields of the GLORYS12 reanalysis of the Copernicus Marine Service,
averaged over 2 x 2 grid cells, on the 1st of October 2019, in the region of the Gulf Stream.

The concepts are explained in the :ref:`indepth.dyn` guide.
See also: :ref:`sphx_glr_tutorials_dynamics_plot_vertical_modes.py`, :ref:`sphx_glr_tutorials_dynamics_plot_gulfstream_dynamics.py`.
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
import xoa.dyn as xdyn
import xoa.numbers as xnum
import xoa.thermdyn as xthermdyn
from xoa.plot import plot_fields

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
xmeta.register_meta_specs(xoa.get_meta_config_file("mercator"))

MAP_KW = {"gridlines_labels_on": ["bottom", "left"], "land": True}

# %%
# Read the reanalysis
# -------------------

ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc")
ds = ds.xoa.decode().isel(time=0)
temp, sal, u, v = ds.temp, ds.sal, ds.u, ds.v
n2 = xthermdyn.get_n2(temp, sal)
open_ocean = n2.notnull().all("depth")

# %%
# Rossby number
# -------------
#
# The Rossby number :math:`Ro = \zeta / f` compares the relative vorticity to the planetary
# one. It is positive for cyclonic rotation and negative for anticyclonic rotation, and
# it is of order one for the strongest eddies and meanders.

rossby = xnum.get_rossby(u.isel(depth=0), v.isel(depth=0))
print(rossby)

# %%
# Burger number
# -------------
#
# The Burger number :math:`Bu = (R_d / L)^2` compares the deformation radius to the scale
# of the flow. We take here a scale of 50 km for the eddies. The deformation radius is either
# the one of the first baroclinic vertical mode, the quick WKB estimate of
# :func:`xoa.dyn.get_rossby_radius`, or the approximation :math:`N H / |f|` with
# the mean stratification of the upper 1000 m.

L = 50e3
vm = xdyn.VerticalModes(n2, nmodes=2)
burger = xnum.get_burger(L, rd=vm.rd.isel(mode=1)).where(open_ocean)
burger_wkb = xnum.get_burger(L, rd=xdyn.get_rossby_radius(n2)).where(open_ocean)

H = 1000.0
n2_mean = n2.where(n2.depth <= H).mean("depth")
burger_nh = xnum.get_burger(L, n2=n2_mean, h=H).where(open_ocean)

# %%
# Froude number
# -------------
#
# The internal Froude number :math:`Fr = U / (N H)` is the speed over the speed of the
# internal waves. It is small, i.e. the flow is subcritical, except where the stratification is
# weak, like in the mixed layers.

speed = np.hypot(u, v).isel(depth=0)
froude = xnum.get_froude(speed, n2_mean, H).where(open_ocean)

# %%
# Ekman number
# ------------
#
# The vertical Ekman number :math:`Ek = \nu / (|f| H^2)` is computed for a surface layer of
# 50 m and a constant eddy viscosity of :math:`10^{-2}` m2 s-1, so it only depends on the
# latitude.

ekman = xnum.get_ekman(1e-2, 50.0, u.lat)
print(float(ekman.min()), float(ekman.max()))

# %%
# Maps
# ----

fig, axes = plot_fields(
    [
        (rossby, {"cmap": "cmo.balance", "levels": np.linspace(-0.6, 0.6, 25)}),
        (burger, {"cmap": "cmo.matter", "levels": np.linspace(0, 1, 21)}),
        (burger_wkb, {"cmap": "cmo.matter", "levels": np.linspace(0, 1, 21)}),
        (burger_nh, {"cmap": "cmo.matter", "levels": np.linspace(0, 6, 25)}),
        (froude, {"cmap": "cmo.amp", "levels": np.linspace(0, 0.2, 21)}),
    ],
    titles=[
        "Rossby",
        "Burger from the modes",
        "Burger from the WKB radius",
        "Burger from N H / f",
        "Froude",
    ],
    ncols=3,
    figsize=(10, 6.2),
    map_kw=MAP_KW,
    method="contourf",
    extend="both",
)

# %%
# The Rossby number reaches 0.5 along the stream and its eddies, and the Burger number
# of the first mode is of order 0.1 to 0.5 for eddies of 50 km, and the WKB radius gives about
# the same. The :math:`N H / |f|` estimate
# is about :math:`\pi^2` times larger, since the first baroclinic radius is
# close to :math:`N H / (\pi |f|)` for a constant stratification: use the radius of the modes
# when it is available. The Froude number stays below one, i.e. the flow is subcritical,
# and it is largest where the stratification is weak.
