#!/usr/bin/env python
# coding: utf-8
"""
Interpolate scattered data with kriging
=======================================

In this tutorial, we show how to:

* compute and fit an empirical semi-variogram of scattered observations,
* interpolate the observations on a regular grid with ordinary kriging,
* get the estimation error and check the result against the truth.

The data are synthetic, so that the interpolation can be evaluated: a spatially correlated
field is sampled at random positions and corrupted with a small noise.
See also: :ref:`sphx_glr_tutorials_filtering_plot_filtering.py` and
:ref:`sphx_glr_tutorials_regrid_plot_kriging_sst.py`.
"""

# %%
# Initialisations
# ---------------

import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean

from xoa.filter import generate_kernel, smooth
from xoa.krig import Kriger, LocalKriger, VariogramModel, empirical_variogram
from xoa.plot import add_colorbar, plot_field, plot_minimap

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# The truth is white noise that is smoothed with :func:`~xoa.filter.smooth` and a kernel from
# :func:`~xoa.filter.generate_kernel` to get a correlation length of about 40 km,
# on a regular grid. It is sampled at 150 random positions.

rng = np.random.default_rng(3)
nx, ny, nobs = 50, 40, 150
lon = xr.DataArray(np.linspace(-6, -4, nx), dims="lon", attrs={"units": "degrees_east"})
lat = xr.DataArray(np.linspace(46, 48, ny), dims="lat", attrs={"units": "degrees_north"})
noise = xr.DataArray(
    rng.normal(size=(ny, nx)), dims=("lat", "lon"), coords={"lon": lon, "lat": lat}
)
kernel = generate_kernel(15, noise, window_func="hanning", isotropic=True)
truth = smooth(noise, kernel, na_thres=1)
truth = 15 + 2 * truth / truth.std()
truth.attrs.update(long_name="Temperature", units="degC")

iy, ix = np.unravel_index(rng.choice(ny * nx, nobs, replace=False), (ny, nx))
obs = xr.DataArray(
    truth.values[iy, ix] + 0.1 * rng.normal(size=nobs),
    dims="npts",
    coords={"lon": ("npts", lon.values[ix]), "lat": ("npts", lat.values[iy])},
    name="temp",
    attrs=truth.attrs,
)

# %%
# Semi-variogram
# --------------
#
# The semi-variogram is half the mean squared difference of the pairs of points,
# as a function of their distance. It tells how fast the points get uncorrelated.
# :func:`~xoa.krig.empirical_variogram` computes it. We restrict it to the distances where
# there are enough pairs, and :class:`~xoa.krig.VariogramModel` fits a model
# with a nugget (noise), a sill (variance) and a range (correlation length).

ev = empirical_variogram(obs, nbin=15, dist_units="km", distmax=120)
vario = VariogramModel("spherical", dist_units="km")
vario.fit(ev)
print(vario)

fig, ax = plt.subplots(figsize=(6, 3.5))
ev.plot(ax=ax, marker="o", ls="", label="Empirical")
vario.plot(ax=ax, rmax=120)
ax.legend(loc="upper left")
ax.set_title("Semi-variogram")
ax.set_ylabel("Semi-variance [degC2]")

# %%
# Ordinary kriging
# ----------------
#
# The fitted model is given to the :class:`~xoa.krig.Kriger`, which returns the
# estimation and its error at the positions of a target dataset.
# The result is on the grid of the target.

kriger = Kriger(obs, "ordinary", vario.get_func(), dist_units="km")
out = kriger.interp(xr.Dataset(coords={"lon": lon, "lat": lat}))
est = out.temp
err = out.temp_error

vmin, vmax = float(truth.min()), float(truth.max())
kw = dict(cmap=cmocean.cm.thermal, vmin=vmin, vmax=vmax, add_colorbar=False)
map_kw = {"gridlines_labels_on": ["left", "bottom"]}
fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.8), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
mappable = plot_field(truth, ax=axes[0], title="Truth and observations", map_kw=map_kw, **kw)
axes[0].scatter(
    obs.lon,
    obs.lat,
    c=obs,
    s=18,
    edgecolors="w",
    linewidths=0.5,
    **{k: v for k, v in kw.items() if k != "add_colorbar"},
    transform=ccrs.PlateCarree(),
)
plot_field(est, ax=axes[1], title="Kriging", map_kw=map_kw, **kw)
add_colorbar(mappable, axes, da=truth)
plot_minimap(truth, parent=axes[0], loc="lower right", size=0.25)

# %%
# The kriged field retrieves the structures of the truth from the sparse observations.

# %%
# Error
# -----
#
# The kriging error only depends on the positions of the observations and on the variogram.
# It is small near the observations and it grows with the distance to them.
# The actual error is also shown, with its root mean square value.

diff = est - truth
rmse = float(np.sqrt((diff**2).mean()))
print(f"RMSE: {rmse:.2f} degC")
fig, axes = plt.subplots(
    1, 2, figsize=(9, 3.8), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
m0 = plot_field(
    err, ax=axes[0], title="Kriging error", cmap=cmocean.cm.amp, map_kw=map_kw, add_colorbar=False
)
add_colorbar(m0, axes[0], label="Error [degC]")
axes[0].scatter(obs.lon, obs.lat, c="k", s=4, transform=ccrs.PlateCarree())
m1 = plot_field(
    diff,
    ax=axes[1],
    title=f"Estimation - truth (RMSE {rmse:.2f})",
    cmap=cmocean.cm.balance,
    vmin=-2,
    vmax=2,
    map_kw=map_kw,
    add_colorbar=False,
)
add_colorbar(m1, axes[1], label="Difference [degC]")

# %%
# Local kriging
# -------------
#
# The :class:`~xoa.krig.Kriger` uses all the observations at once, which gets slow and heavy
# with many observations. The :class:`~xoa.krig.LocalKriger` only uses the ``nneigh`` nearest
# observations of each target point, found with a KD-tree, and solves a small system for each
# of them with numba. Beyond the range of the variogram, far observations hardly matter.

local = LocalKriger(obs, "ordinary", vario, nneigh=40, dist_units="km")
target = xr.Dataset(coords={"lon": lon, "lat": lat})
start = time.perf_counter()
out_local = local.interp(target)
print(f"Local kriging: {time.perf_counter() - start:.3f} s")
est_local = out_local.temp
print(f"Max difference with the global kriging: {float(abs(est_local - est).max()):.3f} degC")

# %%
# The local estimation is close to the global one, the largest differences being far from
# the observations. The number of neighbors and the optional ``maxdist`` set the compromise
# between speed and accuracy. Where there are fewer than ``nmin`` neighbors, for instance
# far from the observations with a small ``maxdist``, the result is masked.
