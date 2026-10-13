#!/usr/bin/env python
# coding: utf-8
"""
Mixed layer, thermocline and isotherms
======================================

In this tutorial, we show how to:

* compute the mixed layer depth with a temperature or a density criterion,
* find the depth of an isotherm and the temperature at a given depth with ``isoslice``,
* check the diagnostics on a vertical profile and on a section.

The data are a GLORYS reanalysis of the Gulf Stream region. The concepts are explained
in the :ref:`indepth.vertical` guide.
See also: :ref:`sphx_glr_tutorials_dynamics_plot_vertical_modes.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import gsw
import matplotlib.pyplot as plt
import cmocean

import xoa
from xoa.interp import isoslice
from xoa.plot import plot_fields, plot_minimap
from xoa.thermdyn import mixed_layer_depth

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc")
ds = ds.xoa.decode().isel(time=0, drop=True)
temp, sal = ds.ptemp, ds.sal
print(temp.depth.attrs["positive"], float(temp.depth[-1]))

# %%
# The potential density is needed for the density criterion. We compute it with the
# `gsw <https://teos-10.github.io/GSW-Python/>`_ package from the practical salinity and the
# potential temperature, with the pressure that is deduced from the depth.

press = gsw.p_from_z(-temp.depth, temp.lat)
asal = gsw.SA_from_SP(sal, press, temp.lon, temp.lat)
sigma0 = gsw.sigma0(asal, gsw.CT_from_pt(asal, temp)).rename("sigma0")
sigma0.attrs.update(long_name="Potential density anomaly", units="kg m-3")

# %%
# Mixed layer depth
# -----------------
#
# The mixed layer is the surface layer whose properties are homogeneous. Its depth is where
# the temperature has decreased by a given amount (``deltatemp``, 0.2 degC by default)
# or the density has increased by a given amount (``deltadens``, 0.03 kg m-3) with respect
# to the surface. The array must have a depth coordinate with a ``positive`` attribute.

mld_temp = mixed_layer_depth(temp, method="deltatemp")
mld_dens = mixed_layer_depth(sigma0, method="deltadens")
print(float(mld_temp.mean()), float(mld_dens.mean()))

fig, axes = plot_fields(
    {"Temperature criterion": mld_temp, "Density criterion": mld_dens},
    ncols=2,
    figsize=(9, 4),
    cmap=cmocean.cm.deep,
    vmin=0,
    vmax=150,
    extend="max",
    map_kw={"land": True},
)
plot_minimap(ds, ax=[0.9, 0.88, 0.1], fig=fig)

# %%
# Both criteria give the same patterns, with mixed layers of a few tens of meters that deepen
# along the fronts and the eddies of the stream. The density criterion is a bit shallower on
# average, since it also accounts for the salinity.

# %%
# Check on a profile
# ------------------
#
# On a profile, we see where the criterion is met. We choose a point north of the stream.

point = dict(lat=39.0, lon=-66.0, method="nearest")
fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), sharey=True, constrained_layout=True)
for ax, var, mld, label in (
    (axes[0], temp, mld_temp, "Temperature [degC]"),
    (axes[1], sigma0, mld_dens, "Potential density anomaly [kg m-3]"),
):
    prof = var.sel(**point)
    ax.plot(prof, prof.depth, "o-", ms=3)
    ax.axhline(float(mld.sel(**point)), color="C3", ls="--", label="Mixed layer depth")
    ax.set_xlabel(label)
axes[0].set_ylim(300, 0)
axes[0].set_ylabel("Depth [m]")
axes[0].legend()

# %%
# Isotherms
# ---------
#
# :func:`xoa.interp.isoslice` interpolates an array where another one crosses a value.
# With the depth as the interpolated array, we get the depth of an isotherm.
# Where the isotherm is above the surface, the result is meaningless and masked.

depth = temp.depth.broadcast_like(temp)
z18 = isoslice(depth, temp, 18.0, "depth").where(temp.isel(depth=0) > 18)
z18.attrs.update(long_name="Depth of the 18 degC isotherm", units="m")

# %%
# We can also do the other way round, to get the temperature at a given depth.

t200 = isoslice(temp, depth, 200.0, "depth")
t200.attrs.update(long_name="Temperature at 200 m", units="degC")

fig, axes = plot_fields(
    [(z18, {"cmap": cmocean.cm.deep, "vmin": 0, "vmax": 500}), (t200, {"cmap": cmocean.cm.thermal})],
    ncols=2,
    figsize=(9, 4),
    map_kw={"land": True},
)

# %%
# The 18 degC isotherm is deeper than 400 m in the warm waters south of the stream and it
# rises sharply across it, which is also visible on the temperature at 200 m.

# %%
# Section across the stream
# -------------------------
#
# On a meridional section, we draw the temperature with a few isotherms and the mixed
# layer depth.

sec = temp.sel(lon=-66.0, method="nearest")
fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
sec.plot.contourf(
    x="lat", y="depth", yincrease=False, ylim=(1000, 0), levels=np.arange(4, 29, 2), cmap=cmocean.cm.thermal, ax=ax
)
sec.plot.contour(
    x="lat", y="depth", yincrease=False, levels=[12, 18], colors="k", linewidths=1, ax=ax
)
mld_temp.sel(lon=-66.0, method="nearest").plot(ax=ax, color="w", lw=2, label="Mixed layer depth")
ax.set_ylim(1000, 0)
ax.set_ylabel("Depth [m]")
ax.legend(loc="lower left")
ax.set_title("Temperature at 66W")
