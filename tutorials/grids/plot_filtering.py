#!/usr/bin/env python
# coding: utf-8
"""
Filter fields in space and in time
==================================

In this tutorial, we show how to:

* smooth a masked 2D field with different kernels,
* control how missing values propagate near the coast,
* extend a field into the land mask,
* apply a Shapiro filter, a classic of ocean modelling, with your own kernel,
* remove the tidal signal of a time series with the Demerliac and Godin filters,
* decimate scattered observations that are too close to each other.

The concepts are explained in the :ref:`indepth.filtering` guide.
See also: :ref:`sphx_glr_tutorials_grids_plot_kriging.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean

import xoa
from xoa import filter as xfilter
from xoa.plot import plot_field, plot_minimap, add_colorbar

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.surf.nc")
ds["temp"] = ds.temp.where(ds.mask_rho.astype(bool))
temp = ds.temp.isel(time=0, s_rho=-1)

# %%
# Smoothing a field
# -----------------
#
# A kernel is given as a size, which makes a moving average, or with a window function.
# With ``smooth``, the result is divided by the sum of the local weights, so that
# missing points do not bias the average. With ``na_thres=0``, the default,
# a point is masked as soon as a single neighbour is missing, which erodes the coast
# by half the kernel size.

kw = dict(cmap=cmocean.cm.thermal, add_colorbar=False, vmin=float(temp.min()), vmax=float(temp.max()))
fields = {
    "Original": temp,
    "Mean, 5 points": xfilter.smooth(temp, 5),
    "Hanning, 9 points": xfilter.smooth(temp, 9, window_func="hanning", isotropic=True),
    "Gaussian, 9 points": xfilter.smooth(
        temp, 9, window_func="gaussian", window_kwargs={"std": 2}, isotropic=True
    ),
}
fig, axes = plt.subplots(
    2, 2, figsize=(9, 5.5), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
for ax, (title, field) in zip(axes.flat, fields.items()):
    mappable = plot_field(field, ax=ax, title=title, map_kw={"gridlines_labels_on": []}, **kw)
add_colorbar(mappable, axes, da=temp)
plot_minimap(temp, parent=axes[0, 0], loc="upper left", size=0.25)

# %%
# Tolerating missing values
# -------------------------
#
# ``na_thres`` is the fraction of missing neighbours that is accepted. The more tolerant,
# the less the coast is eroded, until the field is extended over the land with ``na_thres=1``.

fig, axes = plt.subplots(
    2, 2, figsize=(9, 5.5), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
for ax, thres in zip(axes.flat, (0, 0.3, 0.6, 1)):
    mappable = plot_field(
        xfilter.smooth(temp, 9, na_thres=thres),
        ax=ax,
        title=f"na_thres={thres}",
        map_kw={"gridlines_labels_on": []},
        **kw,
    )
add_colorbar(mappable, axes, da=temp)

# %%
# Extending a field into the land
# -------------------------------
#
# ``erode_coast`` fills the land points that are close to the coast with the smoothed field,
# which is useful before an interpolation or a derivative that needs valid neighbours.

fig, axes = plt.subplots(
    1, 2, figsize=(7, 3), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True
)
for ax, (title, field) in zip(axes, {"Original": temp, "Eroded": xfilter.erode_coast(temp, until=4)}.items()):
    plot_field(field, ax=ax, title=title, map_kw={"gridlines_labels_on": []}, **kw)

# %%
# Filtering a time series
# -----------------------
#
# Tidal filters remove the tidal signal from hourly series with a convolution whose weights
# are designed for it. Here a slow signal is mixed with the main semi-diurnal and
# diurnal tidal waves.

time = np.array("2020-01-01", dtype="M8[h]") + np.arange(24 * 30)
hours = np.arange(time.size)
slow = 0.3 * np.sin(2 * np.pi * hours / (24 * 15))
tide = 1.0 * np.cos(2 * np.pi * hours / 12.42) + 0.4 * np.cos(2 * np.pi * hours / 23.93)
ssh = xr.DataArray(
    slow + tide, dims="time", coords={"time": time}, name="ssh", attrs={"units": "m"}
)
filtered = {name: xfilter.tidal_filter(ssh, name) for name in ("demerliac", "godin")}

fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
ssh.plot(ax=ax, label="Raw", color="0.7")
for name, da in filtered.items():
    da.plot(ax=ax, label=name.title())
xr.DataArray(slow, dims="time", coords={"time": time}).plot(ax=ax, color="k", ls=":", label="Slow signal")
ax.legend()
ax.set_title("Removing the tide")
ax.set_xlabel("")

# %%
# Both filters recover the slow signal, except at the ends of the series where
# the convolution window is incomplete and the result is masked.

# %%
# Decimating scattered observations
# ---------------------------------
#
# Observations are often much denser along tracks than elsewhere, which is a problem for
# a statistical interpolation like kriging. :func:`xoa.filter.decimate` keeps the points that
# are at least ``radius`` meters from each other. With the ``"pick"`` method, the kept points are
# a sample of the original ones. With the ``"average"`` method, they get the average
# of the points that are within the radius. Here are points along three tracks of a ship.

rng = np.random.default_rng(1)
ny, nx = temp.shape
s_ = rng.uniform(size=2000)
ii = np.clip((nx * (0.1 + 0.8 * s_)).astype(int), 0, nx - 1)
jj = np.clip((ny * (0.2 + 0.6 * np.sin(2 * np.pi * s_) ** 2)).astype(int), 0, ny - 1)
valid = np.isfinite(temp.values[jj, ii])
ii, jj = ii[valid], jj[valid]
obs = xr.Dataset(
    {"temp": ("npts", temp.values[jj, ii])},
    coords={
        "lon": ("npts", temp.lon_rho.values[jj, ii], {"standard_name": "longitude"}),
        "lat": ("npts", temp.lat_rho.values[jj, ii], {"standard_name": "latitude"}),
    },
)
picked = xfilter.decimate(obs, 20e3, method="pick")
averaged = xfilter.decimate(obs, 20e3, method="average")
print(obs.sizes["npts"], "->", picked.sizes["npts"], "points")

fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True, sharey=True, constrained_layout=True)
for ax, (title, d) in zip(
    axes, {"Original": obs, "Picked": picked, "Averaged": averaged}.items()
):
    sc = ax.scatter(d.lon, d.lat, c=d.temp, s=6, vmin=kw["vmin"], vmax=kw["vmax"], cmap=kw["cmap"])
    ax.set_title(f"{title}: {d.sizes['npts']} points")
add_colorbar(sc, axes, da=temp)

# %%
# The decimated points are evenly spaced. The averaged ones are smoother, since they are the mean
# of the neighbors, while the picked ones keep the original values.
