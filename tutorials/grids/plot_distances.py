#!/usr/bin/env python
"""
Distances and directions on the sphere
======================================

In this tutorial, we show how to:

* convert steps in degrees to meters and back, at a given latitude,
* compute great circle distances between points, and between two sets of points,
* follow a drifter: distance travelled, speed and direction,
* split scattered points into compact clouds.

The distances are computed with the haversine formula, which takes the curvature of the
earth into account, whereas a distance on a longitude-latitude plane fails at high latitudes,
across the dateline and close to the poles.
See also: :ref:`sphx_glr_tutorials_grids_plot_neighbors.py`, :ref:`sphx_glr_tutorials_grids_plot_grid_tools.py`.
"""

# %%
# Initialisations
# ---------------

import cmocean  # noqa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import xoa
from xoa import geo
from xoa.plot import add_colorbar

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Degrees and meters
# ------------------
#
# A degree of latitude is 111 km, and a degree of longitude shrinks with the cosine of
# the latitude, which is why the cells of a regular grid in degrees are narrower
# at high latitudes. :func:`xoa.geo.deg2m` and :func:`xoa.geo.m2deg` convert the steps.

lats = np.arange(0, 81, 20)
for lat in lats:
    print(
        f"{lat:2d}N: 1 degree of longitude = {geo.deg2m(1, lat) / 1e3:6.1f} km, "
        f"10 km = {geo.m2deg(10e3, lat):.3f} degrees of longitude"
    )
print(f"1 degree of latitude = {geo.deg2m(1) / 1e3:.1f} km")

# %%
# Distances between points
# ------------------------
#
# :func:`xoa.geo.haversine` gives the great circle distance in meters between points, which
# can be arrays that are broadcast. :func:`xoa.geo.cdist` and :func:`xoa.geo.pdist` give
# the distances between two sets of points and inside a set, like the functions of the
# same names of :mod:`scipy.spatial.distance`. Here are the distances between some ports
# of the Atlantic coast.

ports = {
    "Brest": (-4.49, 48.39),
    "Lisbon": (-9.14, 38.72),
    "Casablanca": (-7.59, 33.57),
    "Las Palmas": (-15.41, 28.12),
    "Dakar": (-17.44, 14.69),
}
xy = np.array(list(ports.values()))
dist = geo.cdist(xy, xy) / 1e3
print(f"Brest - Dakar: {dist[0, -1]:.0f} km")

fig, ax = plt.subplots(figsize=(4.8, 4), constrained_layout=True)
mappable = ax.imshow(dist, cmap=cmocean.cm.dense)
ax.set_xticks(range(len(ports)), list(ports), rotation=45, ha="right")
ax.set_yticks(range(len(ports)), list(ports))
for i in range(len(ports)):
    for j in range(len(ports)):
        ax.text(
            j,
            i,
            f"{dist[i, j]:.0f}",
            ha="center",
            va="center",
            color="w" if dist[i, j] > 2000 else "k",
            fontsize=8,
        )
fig.colorbar(mappable, ax=ax, shrink=0.8, label="Distance [km]")
ax.set_title("Distances between ports")

# %%
# The same distances are available for xarray objects with coordinates, with
# :func:`xoa.geo.get_distances`, which also accepts the units.

stations = xr.Dataset(
    coords={
        "lon": ("station", xy[:, 0], {"standard_name": "longitude"}),
        "lat": ("station", xy[:, 1], {"standard_name": "latitude"}),
    }
)
print(geo.get_distances(stations, units="km").values[0].round())

# %%
# Follow a drifter
# ----------------
#
# The drifter of the Global Drifter Program is sampled at irregular times in the Bay of Biscay.
# We resample it every 3 hours, and the distance between consecutive positions gives its
# speed. The bearing is in degrees counter-clockwise from the east, like the angle
# of the velocity vector.

csv_name = xoa.get_data_sample("OBS/DRIFTERS/gdp-6203641.csv")
drifter = pd.read_csv(
    csv_name, header=0, skiprows=[1], parse_dates=[0], index_col=0, usecols=[2, 3, 4, 5, 6, 7]
)
drifter = drifter.resample("3h").mean().dropna()
lon, lat = drifter.longitude.values, drifter.latitude.values
dt = np.diff(drifter.index.values) / np.timedelta64(1, "s")
step = geo.haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
speed = step / dt
angle = geo.bearing(lon[:-1], lat[:-1], lon[1:], lat[1:])
print(f"Travelled {step.sum() / 1e3:.0f} km in {dt.sum() / 86400:.0f} days")

fig, axes = plt.subplots(
    1, 3, figsize=(10, 3.2), gridspec_kw={"width_ratios": (1.1, 1.4, 0.8)}, constrained_layout=True
)
axes[2].remove()
axes[2] = fig.add_subplot(1, 3, 3, projection="polar")
sc = axes[0].scatter(lon[1:], lat[1:], c=speed, s=8, cmap=cmocean.cm.speed)
axes[0].set_aspect(1 / np.cos(np.radians(lat.mean())))
axes[0].set_title("Track")
add_colorbar(sc, axes[0], label="Speed [m s⁻¹]", shrink=0.8)
axes[1].plot(drifter.index[1:], speed, color="k", lw=0.8)
axes[1].set_ylabel("Speed [m s⁻¹]")
axes[1].set_title("Speed")
axes[1].tick_params(axis="x", rotation=30)
axes[2].hist(np.radians(angle), bins=24, range=(-np.pi, np.pi), color=cmocean.cm.speed(0.5))
axes[2].set_title("Directions")

# %%
# Split scattered points
# ----------------------
#
# :func:`xoa.geo.clusterize` splits points into compact clouds of at most ``npmax`` points,
# with a k-means on the sphere. It is the tool of the clustered kriging, to bound the
# size of the problem. Here the positions of the drifter are split in clouds of at most
# 100 points.

pos = xr.Dataset(
    coords={
        "lon": ("time", lon, {"standard_name": "longitude"}),
        "lat": ("time", lat, {"standard_name": "latitude"}),
    }
)
clusters = geo.clusterize(pos, 100, seed=0)
print(clusters.sizes, int(clusters.iclust.max()) + 1, "clusters")

fig, ax = plt.subplots(figsize=(4.5, 3.6), constrained_layout=True)
ax.scatter(lon, lat, c=clusters.iclust, s=8, cmap="tab10")
ax.plot(clusters.centroid.isel(xy=0), clusters.centroid.isel(xy=1), "k*", ms=9)
ax.set_aspect(1 / np.cos(np.radians(lat.mean())))
ax.set_title("Clusters of at most 100 points")
