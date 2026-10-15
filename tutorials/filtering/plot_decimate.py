#!/usr/bin/env python
# coding: utf-8
"""
Decimate scattered observations
===============================

In this tutorial, we show how to:

* build a synthetic set of observations that are much denser along tracks than elsewhere,
* thin them out with :func:`xoa.filter.decimate`, by picking points or by averaging them,
* choose the radius of selection and the amount of smoothing,
* measure the effect on the noise of the observations.

The concepts are explained in the :ref:`indepth.filtering` guide.
See also: :ref:`sphx_glr_tutorials_regrid_plot_kriging.py`,
:ref:`sphx_glr_tutorials_filtering_plot_filter_space.py`.
"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cmocean

from xoa import filter as xfilter
from xoa.plot import add_colorbar

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# A synthetic set of observations
# -------------------------------
#
# The truth is an eddy of 3 °C in a field of 15 °C on a box of 10° by 8°.
# The observations are random points, plus many more points that are measured along
# two ship tracks, with a noise of 0.3 °C. The order of the points is shuffled,
# which matters for the selection, as we'll see.

rng = np.random.default_rng(0)
lon0, lat0, radius_eddy = -15.0, 45.0, 2.0


def truth(lon, lat):
    return 15 + 3 * np.exp(-((lon - lon0) ** 2 + (lat - lat0) ** 2) / radius_eddy**2)


def track(lon_start, lat_start, lon_end, lat_end, npts):
    s = np.sort(rng.uniform(size=npts))
    return lon_start + s * (lon_end - lon_start), lat_start + s * (lat_end - lat_start)


lon_t1, lat_t1 = track(-19, 41, -11, 49, 1200)
lon_t2, lat_t2 = track(-19, 48, -11, 42, 800)
lon = np.concatenate([rng.uniform(-20, -10, 300), lon_t1, lon_t2])
lat = np.concatenate([rng.uniform(40, 48, 300), lat_t1, lat_t2])
order = rng.permutation(lon.size)
lon, lat = lon[order], lat[order]
obs = xr.Dataset(
    {"temp": ("npts", truth(lon, lat) + rng.normal(scale=0.3, size=lon.size), {"units": "°C"})},
    coords={
        "lon": ("npts", lon, {"standard_name": "longitude"}),
        "lat": ("npts", lat, {"standard_name": "latitude"}),
    },
)
print(obs.sizes["npts"], "observations")

# %%
# Pick or average
# ---------------
#
# :func:`~xoa.filter.decimate` loops over the points and keeps a point only if it is at least
# ``radius`` meters away from all the points that were kept before. With ``"pick"``, the kept points
# have their original values. With ``"average"``, the default, they get the average of all the points
# that are within ``radius * smooth_factor`` meters, which reduces the noise.
# The kept points are the same in both cases, and they depend on the order of the points,
# which is why we shuffled them: along a track, the points would be picked
# from one end to the other.

radius = 80e3
picked = xfilter.decimate(obs, radius, method="pick")
averaged = xfilter.decimate(obs, radius, method="average")
print(f"Kept: {picked.sizes['npts']} points, with a radius of {radius * 1e-3:.0f} km")

kw = dict(cmap=cmocean.cm.thermal, vmin=14.5, vmax=18, s=6)
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharex=True, sharey=True, constrained_layout=True)
for ax, (title, d) in zip(axes, {"Original": obs, "Picked": picked, "Averaged": averaged}.items()):
    sc = ax.scatter(d.lon, d.lat, c=d.temp, **kw)
    ax.set_title(f"{title}: {d.sizes['npts']} points")
add_colorbar(sc, axes, da=obs.temp)

# %%
# The decimated points are evenly spaced, and the tracks no longer dominate. The averaged ones
# are smoother, since the noise is averaged, while the picked ones keep the noise.
#
# The smoothing factor
# --------------------
#
# ``smooth_factor`` is the ratio of the averaging radius to the selection radius. A value
# of 0 is the same as ``"pick"``. A value of 1, the default, averages over the same radius,
# and a larger value smooths more but also erodes the eddy, until the average is that of the whole
# domain with an infinite factor. The error against the truth has a minimum between them.

factors = (0, 0.5, 1, 1.5, 2, 3, 5)
rmse = [
    float(
        np.sqrt(
            ((d := xfilter.decimate(obs, radius, smooth_factor=factor)).temp - truth(d.lon, d.lat))
            ** 2
        ).mean()
    )
    for factor in factors
]
fig, axes = plt.subplots(1, 2, figsize=(9, 2.8), constrained_layout=True)
axes[0].plot(factors, rmse, "o-")
axes[0].set_xlabel("smooth_factor")
axes[0].set_ylabel("RMSE against the truth (°C)")
axes[0].set_title("Error of the decimated points")
for factor, color in ((0, "0.7"), (1, "C0"), (3, "C1")):
    d = xfilter.decimate(obs, radius, smooth_factor=factor)
    axes[1].scatter(d.lat, d.temp, s=8, color=color, label=f"smooth_factor={factor}")
lats = np.linspace(40, 50, 100)
axes[1].plot(lats, truth(lon0, lats), "k", lw=1, label="Truth at -15°E")
axes[1].legend()
axes[1].set_xlabel("Latitude (°N)")
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("Smoothing the eddy")

# %%
# Choosing the radius
# -------------------
#
# The radius is the minimal distance between the kept points, hence the resolution of the
# decimated set. It must be smaller than the scale you want to resolve.
# Before a kriging, it is chosen smaller than the correlation length of the variogram,
# and large enough to keep a reasonable number of points: see :ref:`sphx_glr_tutorials_regrid_plot_kriging.py`.

fig, ax = plt.subplots(figsize=(5, 2.8), constrained_layout=True)
radii = np.array([10, 20, 40, 80, 160, 320]) * 1e3
ax.loglog(radii * 1e-3, [xfilter.decimate(obs, r).sizes["npts"] for r in radii], "o-")
ax.set_xlabel("Radius (km)")
ax.set_ylabel("Number of kept points")
ax.set_title("Kept points as a function of the radius")
