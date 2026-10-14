#!/usr/bin/env python
# coding: utf-8
"""
Map the Gulf Stream temperature from sparse observations
========================================================

In this tutorial, we show how to:

* sample a realistic sea surface temperature field along ship tracks and at buoys,
* remove a large scale trend and fit a variogram model on the residuals,
* krig the observations on a grid and evaluate the estimation and its uncertainty,
* use the :class:`~xoa.krig.LocalKriger` to go faster with many observations,
* choose the number of neighbors and compare the cost of the three kriging classes.

The truth is the surface temperature of a Mercator ocean reanalysis around the Gulf Stream,
which is sampled to mimic observations, so that the interpolation can be evaluated.
The basics are in the :ref:`sphx_glr_tutorials_grids_plot_kriging.py` tutorial.
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

import xoa
from xoa import meta
from xoa.krig import ClusteredKriger, Kriger, LocalKriger, VariogramModel, empirical_variogram
from xoa.plot import add_colorbar, plot_field, plot_minimap

xr.set_options(display_style="text")
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True})
proj = ccrs.PlateCarree()
map_kw = {"gridlines_labels_on": ["left", "bottom"]}
cmap_temp = cmocean.cm.thermal

# %%
# The truth
# ---------
#
# The Gulf Stream separates the cold slope waters from the warm Sargasso Sea
# with a sharp front and meanders. Such a field is a difficult test for an interpolation.

with meta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc").xoa.decode()
truth = ds.temp.isel(time=0, depth=0, drop=True)
truth.attrs.update(long_name="Sea surface temperature", units="degC")
land = truth.isnull()
lon, lat = truth.lon, truth.lat

# %%
# Synthetic observations
# ----------------------
#
# We mimic a campaign with six ship tracks sampled every 8 km or so, and 60 drifting buoys.
# The truth is interpolated at their positions, where it is defined,
# and a measurement noise of 0.15 degC is added.

rng = np.random.default_rng(4)
lon_lim = (float(lon.min()), float(lon.max()))
lat_lim = (float(lat.min()), float(lat.max()))


def sample_positions(rng, ntracks=6, nbuoys=60, step=0.08):
    """Get the positions of observations as ship tracks and buoys"""
    lons, lats, kinds = [], [], []
    for _ in range(ntracks):
        a = np.array([rng.uniform(*lon_lim), rng.uniform(*lat_lim)])
        b = np.array([rng.uniform(*lon_lim), rng.uniform(*lat_lim)])
        t = np.linspace(0, 1, int(np.hypot(*(b - a)) / step))
        lons.append(a[0] + t * (b[0] - a[0]))
        lats.append(a[1] + t * (b[1] - a[1]))
        kinds.append(np.zeros(t.size, "i"))
    lons.append(rng.uniform(*lon_lim, nbuoys))
    lats.append(rng.uniform(*lat_lim, nbuoys))
    kinds.append(np.ones(nbuoys, "i"))
    return np.concatenate(lons), np.concatenate(lats), np.concatenate(kinds)


def observe(rng, lons, lats, noise=0.15):
    """Sample the truth at positions, and add noise"""
    npts = xr.DataArray(np.arange(lons.size), dims="npts")
    pos = {"lon": ("npts", lons), "lat": ("npts", lats)}
    obs = truth.interp(lon=xr.DataArray(lons, dims="npts"), lat=xr.DataArray(lats, dims="npts"))
    obs = obs.drop_vars(["lon", "lat"], errors="ignore").assign_coords(npts=npts, **pos)
    obs = obs.where(obs.notnull(), drop=True)
    return (obs + rng.normal(0, noise, obs.size)).rename("sst").assign_attrs(truth.attrs)


lons, lats, kinds = sample_positions(rng)
obs = observe(rng, lons, lats)
kind = xr.DataArray(kinds, dims="npts")[obs.npts.values]
print(f"{obs.size} observations")

# %%
# Here are the truth and the observations.


def plot_obs(ax, obs, vmin, vmax, size=14, **kwargs):
    """Draw observations as dots coloured like the field"""
    return ax.scatter(
        obs.lon, obs.lat, c=obs, s=size, edgecolors="w", linewidths=0.4, cmap=cmap_temp,
        vmin=vmin, vmax=vmax, transform=proj, zorder=5, **kwargs
    )  # fmt: skip


vmin, vmax = float(truth.min()), float(truth.max())
kw = dict(cmap=cmap_temp, vmin=vmin, vmax=vmax, add_colorbar=False)
fig, axes = plt.subplots(
    1, 2, figsize=(10, 4.2), subplot_kw={"projection": proj}, constrained_layout=True
)
mappable = plot_field(truth, ax=axes[0], title="Truth", map_kw=map_kw, **kw)
plot_field(truth, ax=axes[1], title="Observations", map_kw=map_kw, alpha=0.35, **kw)
plot_obs(axes[1], obs, vmin, vmax)
add_colorbar(mappable, axes, da=truth)
plot_minimap(truth, parent=axes[0], loc="lower right", size=0.25)

# %%
# Remove the large scale trend
# ----------------------------
#
# Kriging assumes that the mean is constant, or that the field is stationary
# for the ordinary kriging within the neighborhood of each point.
# Here, the temperature drops by more than 10 degC from south to north, which
# would hide the variogram of the small scales and bias it.
# We remove a plane fitted to the observations, krig the residuals, and add the plane back.


A = np.c_[np.ones(obs.size), obs.lon.values, obs.lat.values]
coef = np.linalg.lstsq(A, obs.values, rcond=None)[0]


def plane(lo, la):
    return coef[0] + coef[1] * lo + coef[2] * la


resid = obs - plane(obs.lon, obs.lat)
resid.attrs.update(obs.attrs, long_name="Temperature anomaly")
trend = plane(lon, lat).where(~land)
print(f"Standard deviation: {float(obs.std()):.2f} degC before and {float(resid.std()):.2f} after")

fig, axes = plt.subplots(
    1, 2, figsize=(10, 4.2), subplot_kw={"projection": proj}, constrained_layout=True
)
m0 = plot_field(trend, ax=axes[0], title="Fitted trend", map_kw=map_kw, **kw)
add_colorbar(m0, axes[0], label="°C")
amax = float(np.abs(resid).quantile(0.98))
sc = axes[1].scatter(
    resid.lon, resid.lat, c=resid, s=16, cmap=cmocean.cm.balance, vmin=-amax, vmax=amax,
    edgecolors="w", linewidths=0.3, transform=proj,
)  # fmt: skip
plot_field(truth * np.nan, ax=axes[1], title="Observation anomalies", map_kw=map_kw, **kw)
axes[1].set_extent([*lon_lim, *lat_lim], crs=proj)
add_colorbar(sc, axes[1], label="°C")

# %%
# Fit the semi-variogram
# ----------------------
#
# The empirical semi-variogram of the anomalies is fitted with the available models.
# The nugget accounts for the noise and the unresolved small scales, the sill for the
# variance at large distances, and the range for the correlation length.
# The distances are in km, and the fit error helps choosing the model.
# The gaussian model is left out: it is so smooth that, without a nugget, the kriging
# system is ill-conditioned.

ev = empirical_variogram(resid, nbin=25, nbin0=5, dist_units="km", distmax=500)
models = {}
for mtype in ["exponential", "spherical"]:
    models[mtype] = VariogramModel(mtype, dist_units="km")
    models[mtype].fit(ev)
    p = models[mtype].params
    print(
        f"{mtype:12s} nugget={p['n']:.3f} sill={p['s']:.2f} range={p['r']:.0f} km "
        f"fit error={models[mtype]._fit_err:.3f}"
    )

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
ax.plot(ev.dist, ev, "o", color="0.3", label="Empirical")
dist = np.linspace(0, 500, 200)
for (mtype, model), color in zip(models.items(), ["tab:blue", "tab:orange"]):
    ax.plot(dist, model(dist), color=color, lw=2, label=mtype.title())
ax.set_xlabel("Distance [km]")
ax.set_ylabel("Semi-variance [°C$^2$]")
ax.set_title("Semi-variogram of the anomalies")
ax.legend(loc="lower right")

# %%
# The empirical variogram bumps at 100-200 km because of the front, which is a hint of
# non stationarity, and the models are a compromise. We keep the one that fits best.

best = min(models, key=lambda mtype: models[mtype]._fit_err)
vario = models[best]
print("Best model:", vario)

# %%
# Ordinary kriging
# ----------------
#
# The anomalies are kriged on the grid of the truth with the :class:`~xoa.krig.Kriger`,
# and the trend is added back. The result is a dataset on the target grid with the
# estimation and its kriging variance. The land points are masked.

target = xr.Dataset(coords={"lon": lon, "lat": lat})

start = time.perf_counter()
kriger = Kriger(resid, "ordinary", vario, dist_units="km")
out = kriger.interp(target, name="sst")
t_global = time.perf_counter() - start
est = (out.sst + plane(lon, lat)).where(~land).assign_attrs(truth.attrs)
std = np.sqrt(out.sst_error.clip(min=0)).where(~land)
diff = est - truth
rmse = float(np.sqrt((diff**2).mean()))
print(f"Global kriging: RMSE={rmse:.2f} degC in {t_global:.2f} s")

# %%
# The estimation retrieves the front and its meanders where the observations are close enough.
# The kriging standard deviation tells where to trust it: it is small near the tracks and
# it saturates at the level of the sill far from them. It only depends on the positions
# of the observations and on the variogram, not on the values.
# The actual error is also shown.

fig, axes = plt.subplots(
    2, 2, figsize=(10, 8), subplot_kw={"projection": proj}, constrained_layout=True
)
axes = axes.ravel()
plot_field(truth, ax=axes[0], title="Truth", map_kw=map_kw, **kw)
plot_field(est, ax=axes[1], title="Kriging", map_kw=map_kw, **kw)
add_colorbar(mappable, axes[:2], da=truth, shrink=0.8)
dmax = float(np.abs(diff).quantile(0.99))
m2 = plot_field(
    diff, ax=axes[2], title=f"Estimation - truth (RMSE {rmse:.2f})", cmap=cmocean.cm.balance,
    vmin=-dmax, vmax=dmax, map_kw=map_kw, add_colorbar=False,
)  # fmt: skip
add_colorbar(m2, axes[2], label="°C", shrink=0.8)
m3 = plot_field(
    std, ax=axes[3], title="Kriging standard deviation", cmap=cmocean.cm.amp, map_kw=map_kw,
    add_colorbar=False,
)  # fmt: skip
add_colorbar(m3, axes[3], label="°C", shrink=0.8)
for ax in axes[1:]:
    ax.scatter(obs.lon, obs.lat, c="k", s=1.5, transform=proj, zorder=5)

# %%
# Local kriging
# -------------
#
# The :class:`~xoa.krig.Kriger` uses all the observations, which means a dense matrix of
# distances and its inversion, with a cost that grows as the cube of the number of
# observations. The :class:`~xoa.krig.LocalKriger` finds the ``nneigh`` nearest observations
# of each grid point with a KD-tree on the sphere, and solves a small system for each of them,
# in parallel with numba. Beyond the range of the variogram, the far observations
# have almost no weight anyway.

local = LocalKriger(resid, "ordinary", vario, nneigh=32, dist_units="km")
local.interp(target.isel(lon=slice(2), lat=slice(2)))  # compilation
start = time.perf_counter()
out_local = local.interp(target, name="sst")
t_local = time.perf_counter() - start
est_local = (out_local.sst + plane(lon, lat)).where(~land)
std_local = np.sqrt(out_local.sst_error.clip(min=0)).where(~land)
rmse_local = float(np.sqrt(((est_local - truth) ** 2).mean()))
print(f"Local kriging: RMSE={rmse_local:.2f} degC in {t_local:.3f} s")

fig, axes = plt.subplots(
    1, 3, figsize=(13, 3.8), subplot_kw={"projection": proj}, constrained_layout=True
)
plot_field(est_local, ax=axes[0], title="Local kriging", map_kw=map_kw, **kw)
add_colorbar(mappable, axes[0], da=truth, shrink=0.8)
dd = est_local - est
dmax = max(float(np.abs(dd).max()), 1e-3)
m1 = plot_field(
    dd, ax=axes[1], title="Local - global", cmap=cmocean.cm.balance, vmin=-dmax, vmax=dmax,
    map_kw=map_kw, add_colorbar=False,
)  # fmt: skip
add_colorbar(m1, axes[1], label="°C", shrink=0.8)
ds_ = std_local - std
smax = max(float(np.abs(ds_).max()), 1e-3)
m2 = plot_field(
    ds_, ax=axes[2], title="Standard deviation: local - global", cmap=cmocean.cm.balance,
    vmin=-smax, vmax=smax, map_kw=map_kw, add_colorbar=False,
)  # fmt: skip
add_colorbar(m2, axes[2], label="°C", shrink=0.8)

# %%
# The two estimations agree where the observations are dense. They differ in the
# cold waters to the north, where the global kriging relaxes towards the mean of the whole
# domain, whereas the local one relaxes towards the mean of the neighbors, and it is faster.
# The local kriging standard deviation is slightly larger since it uses less information.

# %%
# How many neighbors?
# -------------------
#
# More neighbors give an estimation closer to the global one, but cost more.
# Too few neighbors lead to artefacts since the neighborhood becomes one-sided,
# for instance along a ship track. Let's look at the error with respect to the truth and
# to the global kriging, and at the cost, as functions of ``nneigh``.

nneighs = np.array([4, 8, 16, 32, 64, 128])
errs, gaps, costs = [], [], []
for nneigh in nneighs:
    lk = LocalKriger(resid, "ordinary", vario, nneigh=int(nneigh), dist_units="km")
    best_t = np.inf
    lk.interp(target.isel(lon=slice(2), lat=slice(2)))  # warm-up
    for _ in range(5):
        start = time.perf_counter()
        out_nn = lk.interp(target, name="sst")
        best_t = min(best_t, time.perf_counter() - start)
    res = (out_nn.sst + plane(lon, lat)).where(~land)
    errs.append(float(np.sqrt(((res - truth) ** 2).mean())))
    gaps.append(float(np.sqrt(((res - est) ** 2).mean())))
    costs.append(best_t)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
axes[0].semilogx(nneighs, errs, "o-", label="Local - truth")
axes[0].semilogx(nneighs, gaps, "s-", label="Local - global")
axes[0].axhline(rmse, color="k", ls="--", label="Global - truth")
axes[0].set_xlabel("Number of neighbors")
axes[0].set_ylabel("RMSE [degC]")
axes[0].set_title("Accuracy")
axes[0].legend()
axes[1].loglog(nneighs, costs, "o-", color="tab:red", label="Local")
axes[1].axhline(t_global, color="k", ls="--", label="Global")
axes[1].set_xlabel("Number of neighbors")
axes[1].set_ylabel("Time [s]")
axes[1].set_title("Cost")
axes[1].legend()
for ax in axes:
    ax.set_xticks(nneighs, labels=[str(n) for n in nneighs])
    ax.minorticks_off()

# %%
# The estimation gets closer to the global one as the neighbors increase, but the error with
# respect to the truth does not improve, and is even lower than the global one since the local
# mean is more suited to a non stationary field. The cost goes on increasing:
# a few tens of neighbors is a good compromise here.
# The ``maxdist`` argument also limits the search to a distance, in the units of the
# variogram, and the points with fewer than ``nmin`` neighbors are set to nan.

# %%
# Many observations
# -----------------
#
# Satellite swaths, gliders and ship thermosalinographs provide thousands of points.
# We compare the cost of the three kriging classes as the number of observations increases:
#
# * :class:`~xoa.krig.Kriger` uses all the points at once,
# * :class:`~xoa.krig.ClusteredKriger` splits them in clusters of at most ``npmax`` points
#   and merges the estimations, which limits the memory,
# * :class:`~xoa.krig.LocalKriger` only uses the nearest ones.
#
# The observations are here random points, and the cost includes the setup of the kriger.

counts = [250, 500, 1000, 2000]
timings = {"Kriger": [], "ClusteredKriger": [], "LocalKriger": []}
builders = {
    "Kriger": lambda o: Kriger(o, "ordinary", vario, dist_units="km"),
    "ClusteredKriger": lambda o: ClusteredKriger(
        o, "ordinary", vario, dist_units="km", npmax=500, nproc=1
    ),
    "LocalKriger": lambda o: LocalKriger(o, "ordinary", vario, nneigh=32, dist_units="km"),
}
for count in counts:
    lo = rng.uniform(*lon_lim, count)
    la = rng.uniform(*lat_lim, count)
    big = observe(rng, lo, la)
    big = big - plane(big.lon, big.lat)
    for name, build in builders.items():
        start = time.perf_counter()
        build(big).interp(target, name="sst")
        timings[name].append(time.perf_counter() - start)

fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
for (name, values), marker in zip(timings.items(), "os^"):
    ax.loglog(counts, values, marker + "-", lw=2, label=name)
ax.set_xticks(counts, labels=[str(c) for c in counts])
ax.minorticks_off()
ax.set_xlabel("Number of observations")
ax.set_ylabel("Time [s]")
ax.set_title("Cost of kriging the %d grid points" % (lon.size * lat.size))
ax.legend()

# %%
# The cost of the local kriging hardly depends on the number of observations, and it is the
# right tool for big datasets. The clustered kriging is not faster than the global one, but
# it avoids building a huge matrix: it is an alternative when all the points
# must contribute to a global solution with a limited memory.
