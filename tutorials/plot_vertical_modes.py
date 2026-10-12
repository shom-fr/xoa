#!/usr/bin/env python
# coding: utf-8
"""
Vertical modes of the Gulf Stream
=================================

In this tutorial, we show how to:

* compute the stratification from temperature and salinity,
* compute the vertical modes of each water column and the associated deformation radii,
* split the currents into the amplitudes of the modes, and put them back together,
* see how the kinetic energy is shared between the barotropic and baroclinic modes,
* focus on the upper ocean with a vertical zoom.

The data are 3D fields of the GLORYS12 reanalysis of the Copernicus Marine Service,
averaged over 2 x 2 grid cells, on the 1st of October 2019, in the region of the Gulf Stream.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean  # noqa

import xoa
import xoa.meta as xmeta
import xoa.dyn as xdyn
import xoa.thermdyn as xthermdyn
from xoa.plot import plot_field, add_colorbar

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Register the internal Mercator naming specifications, which are used automatically for
# the files of the Mercator and GLORYS products.

xmeta.register_meta_specs(xoa.get_meta_config_file("mercator"))

# %%
# Read the reanalysis
# -------------------
#
# The sample goes down to 1942 m, whereas the ocean is deeper than that in the middle of the
# domain: the modes that we compute are the ones of the upper 1942 m, and the water
# columns that are shallower have less levels.

ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc")
ds = ds.xoa.decode().isel(time=0)
temp, sal, u, v = ds.temp, ds.sal, ds.u, ds.v
print(ds)

# %%
# Stratification
# --------------
#
# The vertical modes are the solutions of a problem that depends only on the stratification
# of each water column, which is the squared Brunt-Vaisala frequency :math:`N^2`.
# :func:`xoa.thermdyn.get_n2` computes it from the potential temperature and the salinity
# with the `gsw <https://teos-10.github.io/GSW-Python/>`_ package, at the levels.

n2 = xthermdyn.get_n2(temp, sal)
print(n2)

# %%
# The vertical modes
# ------------------
#
# The :class:`xoa.dyn.VerticalModes` class computes the modes of all the water columns
# at once, with numba. The first mode is the barotropic one, which is constant along the
# vertical, and the next ones are the baroclinic modes, that have more and more nodes.
# With the default rigid lid, the barotropic mode has no speed.
# The deformation radii are computed since the latitudes are found in the coordinates.

vm = xdyn.VerticalModes(n2, nmodes=5)
print(vm)
print(vm.c.dims, vm.rd.dims)

# %%
# Let's look at the first modes in a water column of the open ocean, south of the stream,
# with the stratification that makes them.

LAT, LON = 35.0, -66.0
point = {"lat": LAT, "lon": LON, "method": "nearest"}
depth = ds.depth

fig, axes = plt.subplots(1, 2, figsize=(8, 5.5), sharey=True, constrained_layout=True)
axes[0].plot(n2.sel(**point).clip(min=1e-7), depth, color="0.2", lw=1.8)
axes[0].set_xscale("log")
axes[0].set_xlim(1e-7, 1e-3)
axes[0].set_xlabel("Squared Brunt-Vaisala frequency [s-2]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Stratification")
colors = cmocean.cm.deep(np.linspace(0.15, 0.85, 4))
for mode, color in zip(range(1, 5), colors):
    modes = vm.modes.sel(**point).isel(mode=mode)
    rd = float(vm.rd.sel(**point).isel(mode=mode)) * 1e-3
    axes[1].plot(modes, depth, color=color, lw=1.8, label=f"Mode {mode}: {rd:.0f} km")
axes[1].plot(vm.modes.sel(**point).isel(mode=0), depth, color="0.6", ls="--", label="Mode 0")
axes[1].axvline(0, color="0.8", lw=0.8, zorder=0)
axes[1].set_xlabel("Vertical structure [1]")
axes[1].set_title("Modes and deformation radii")
axes[1].legend(frameon=False, loc="lower right")
axes[0].invert_yaxis()

# %%
# The mixed layer, where the stratification vanishes, is not shown on the plot. The modes are
# normalized so that the mean of their squares is one, and they are positive at the surface. The first baroclinic mode has a single node, close to the bottom of the
# thermocline, and has its largest amplitude at the surface.

# %%
# Deformation radii
# -----------------
#
# The deformation radius of a mode is its speed divided by the Coriolis parameter:
# it is the scale at which the rotation matters. That of the first baroclinic
# mode is the one of the mesoscale eddies. Here it is mapped for the water columns
# that have all the levels, i.e. outside of the shelf.

open_ocean = n2.notnull().all("depth")
rd1 = (vm.rd.isel(mode=1) * 1e-3).where(open_ocean).rename("rd1")
rd1.attrs = {"long_name": "First deformation radius", "units": "km"}
print(f"Median first radius: {float(rd1.median()):.1f} km")

MAP_KW = {"gridlines_labels_on": ["bottom", "left"], "land": True}
fig, ax = plt.subplots(
    figsize=(7.5, 5.8), subplot_kw={"projection": ccrs.Mercator()}, constrained_layout=True
)
plot_field(
    rd1,
    ax=ax,
    method="contourf",
    cmap="cmo.matter",
    levels=np.linspace(18, 34, 17),
    extend="both",
    map_kw=MAP_KW,
    title="First baroclinic deformation radius",
    overlay_contours=[
        {"field": np.hypot(u, v).isel(depth=0), "levels": [0.8, 1.2], "colors": ["0.35", "k"], "linewidths": 0.8}
    ],
)

# %%
# The radius decreases northward with the Coriolis parameter and the warm waters south of the
# stream, whose stratification is strong, have larger radii than the colder waters on the
# shelf side. The contours show the surface currents of 0.8 and 1.2 m/s, which is the stream.

# %%
# Decompose the currents
# ----------------------
#
# :meth:`xoa.dyn.VerticalModes.decomp` projects a field on the modes, and returns the
# amplitudes with the ``mode`` dimension first. The amplitude of the barotropic mode is the
# vertical mean of the currents. It is computed for all the water columns at once, with
# :func:`xarray.apply_ufunc` so that extra dimensions like time and dask arrays work too.

au, av = vm.decomp(u), vm.decomp(v)
print(au.dims, au.shape)

# %%
# By orthogonality, the mean kinetic energy over the water column is the sum of the squared
# amplitudes. Here is how it is shared between the modes, for the open ocean.

ke = (au**2 + av**2).where(open_ocean)
total = ((u**2 + v**2) * vm.dz).sum("depth") / vm.dz.sum("depth")
share = (ke / total).where(open_ocean).mean(("lat", "lon"))
fig, axes = plt.subplots(
    1, 3, figsize=(15, 4.6), subplot_kw={"projection": ccrs.Mercator()}, constrained_layout=True
)
fig.delaxes(axes[0])
# Bars on a regular axes at the place of the first map
bar_ax = fig.add_subplot(1, 3, 1)
bar_ax.bar(share.mode, share * 100, color=cmocean.cm.deep(np.linspace(0.15, 0.85, 5)))
for k, value in enumerate(share.values):
    bar_ax.text(k, value * 100 + 1.5, f"{value * 100:.0f}%", ha="center", fontsize=8)
bar_ax.set_xlabel("Mode")
bar_ax.set_ylabel("Share of the kinetic energy [%]")
bar_ax.set_title("Kinetic energy")
bar_ax.set_ylim(0, 80)
for ax, k, vmax in zip(axes[1:], (0, 1), (1.2, 0.8)):
    amp = np.hypot(au.isel(mode=k), av.isel(mode=k)).where(open_ocean).rename("amp")
    amp.attrs = {"long_name": f"Speed of mode {k}", "units": "m s-1"}
    plot_field(
        amp,
        ax=ax,
        method="contourf",
        cmap="cmo.tempo",
        levels=np.linspace(0, vmax, 17),
        extend="max",
        map_kw=MAP_KW,
        title=("Barotropic mode" if k == 0 else f"Baroclinic mode {k}"),
    )

# %%
# The currents of the stream are mostly barotropic and first baroclinic: the first two modes
# carry most of the kinetic energy, as it is generally observed in the ocean. Note that the
# barotropic mode is here the vertical mean over the upper 1942 m only.

# %%
# Recompose the currents
# ----------------------
#
# :meth:`xoa.dyn.VerticalModes.recomp` puts the amplitudes back together. Using only the
# first modes gives a smooth version of the profiles, and all the modes give the exact profile.
# Let's see it in a column of the stream, where the currents are strong and sheared.

STREAM = {"lat": 37.5, "lon": -66.0, "method": "nearest"}
fig, axes = plt.subplots(1, 2, figsize=(8, 5.2), constrained_layout=True)
axes[0].plot(u.sel(**STREAM), depth, color="k", lw=2.2, label="Reanalysis")
colors = cmocean.cm.matter(np.linspace(0.2, 0.85, 3))
for n, color in zip((1, 2, 3), colors):
    axes[0].plot(
        vm.recomp(au, modes=n).sel(**STREAM), depth, color=color, lw=1.6,
        label=f"{n} mode" + ("s" if n > 1 else ""),
    )
axes[0].invert_yaxis()
axes[0].set_xlabel("Zonal current [m s-1]")
axes[0].set_ylabel("Depth [m]")
axes[0].set_title("Profile in the stream")
axes[0].legend(frameon=False)

# Error over the domain, as a function of the number of modes
vm_all = xdyn.VerticalModes(n2, nmodes=n2.sizes["depth"])
au_all = vm_all.decomp(u)
nmodes = [1, 2, 3, 5, 8, 12, 20]
errors = [
    float(np.sqrt(((vm_all.recomp(au_all, modes=n) - u) ** 2).mean())) for n in nmodes
]
axes[1].plot(nmodes, errors, "o-", color="0.2")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Number of modes")
axes[1].set_ylabel("Root mean square error [m s-1]")
axes[1].set_title("Truncation error")
axes[1].grid(True, which="both", alpha=0.3)

# %%
# The error decreases quickly with the number of modes: three of them are enough for an error
# of a few centimeters per second, which is why the modal decomposition is a compact
# description of the vertical structure of the currents.

# %%
# Focus on the upper ocean
# ------------------------
#
# The ``zrange`` parameter restricts the computation to a layer: only the levels that are in it
# are used, and the modes have no flux through its limits. The modes of the upper 500 m
# describe the thermocline, and their radii are smaller since they are faster to oscillate.

zoom = xdyn.VerticalModes(n2, nmodes=5, zrange=(0, 500))
rd1z = (zoom.rd.isel(mode=1) * 1e-3).where(open_ocean)
print(f"Median first radius: {float(rd1.median()):.1f} km for the water column, "
      f"{float(rd1z.median()):.1f} km for the upper 500 m")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 5), constrained_layout=True)
for ax, obj, title, ylim in zip(
    axes, (vm, zoom), ("Whole water column", "Upper 500 m"), ((2000, 0), (500, 0))
):
    for mode, color in zip(range(1, 4), colors):
        ax.plot(obj.modes.sel(**point).isel(mode=mode), depth, color=color, lw=1.8, label=f"Mode {mode}")
    ax.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax.set_title(title)
    ax.set_xlabel("Vertical structure [1]")
    ax.set_ylim(*ylim)
axes[0].set_ylabel("Depth [m]")
axes[0].legend(frameon=False, loc="lower left")

# %%
# The decomposition and the recomposition then only apply to the layer, and are missing
# outside of it, as the modes are. The amplitudes of the zoom therefore describe the upper
# 500 m only, like the amplitude of its barotropic mode, which is their vertical mean.

au_zoom = zoom.decomp(u.where(depth <= 500))
rec = zoom.recomp(au_zoom, modes=3)
upper = depth <= 500
error = float(np.sqrt(((rec - u) ** 2).where(upper).mean()))
print(f"Error of the first 3 modes of the upper 500 m: {error:.3f} m/s")

# %%
# With a free surface, the barotropic mode also gets a speed, that is close to
# :math:`\sqrt{g H}`: ``VerticalModes(n2, bc="free")``.
free = xdyn.VerticalModes(n2, nmodes=2, bc="free")
print(f"Barotropic speed: {float(free.c.sel(**point).isel(mode=0)):.0f} m/s for a depth of "
      f"{float(free.dz.sel(**point).sum()):.0f} m")
plt.show()
