#!/usr/bin/env python
# coding: utf-8
"""
Filter fields in space
======================

In this tutorial, we show how to:

* smooth a masked 2D field with different kernels, and choose between isotropic and orthogonal ones,
* control how missing values propagate near the coast,
* extend a field into the land mask,
* remove grid-scale noise with a Shapiro filter, and use your own kernel,
* beware that kernel sizes are in grid points, not in kilometers.

The concepts are explained in the :ref:`indepth.filtering` guide.
See also: :ref:`sphx_glr_tutorials_filtering_plot_filter_time.py`,
:ref:`sphx_glr_tutorials_filtering_plot_decimate.py`.
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
from xoa import grid as xgrid
from xoa.plot import plot_field, plot_minimap, add_colorbar

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.surf.nc")
ds["temp"] = ds.temp.where(ds.mask_rho.astype(bool))
temp = ds.temp.isel(time=0, s_rho=-1)

kw = dict(
    cmap=cmocean.cm.thermal, add_colorbar=False, vmin=float(temp.min()), vmax=float(temp.max())
)
map_kw = {"gridlines_labels_on": []}


def plot_maps(fields, shape, figsize, **kwargs):
    """Plot several fields on maps with a shared colorbar"""
    fig, axes = plt.subplots(
        *shape,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (title, field) in zip(axes.flat, fields.items()):
        mappable = plot_field(field, ax=ax, title=title, map_kw=map_kw, **{**kw, **kwargs})
    add_colorbar(mappable, axes, da=temp)
    return axes


# %%
# Smoothing a field
# -----------------
#
# A kernel is given as a size, which makes a moving average, or with a window function.
# With :func:`~xoa.filter.smooth`, the result is divided by the sum of the local weights,
# so that missing points do not bias the average. With ``na_thres=0``, the default,
# a point is masked as soon as a single neighbour is missing, which erodes the coast
# by half the kernel size.

axes = plot_maps(
    {
        "Original": temp,
        "Mean, 5 points": xfilter.smooth(temp, 5),
        "Hanning, 9 points": xfilter.smooth(temp, 9, window_func="hanning", isotropic=True),
        "Gaussian, 9 points": xfilter.smooth(
            temp, 9, window_func="gaussian", window_kwargs={"std": 2}, isotropic=True
        ),
    },
    (2, 2),
    (9, 5.5),
)
plot_minimap(temp, parent=axes.flat[0], loc="upper left", size=0.25)

# %%
# Orthogonal or isotropic kernels
# -------------------------------
#
# By default, a window function is applied along each dimension separately and the
# 1D windows are multiplied, which gives a kernel whose footprint is a square.
# With ``isotropic=True``, the window is a function of the distance to the center
# in grid points, and the footprint is a disk.
# The kernel size can also be different along each dimension.

dims = ("eta_rho", "xi_rho")
kernels = {
    "Orthogonal Hanning, 11": xfilter.generate_kernel(11, temp, window_func="hanning"),
    "Isotropic Hanning, 11": xfilter.generate_kernel(
        11, temp, window_func="hanning", isotropic=True
    ),
    "Anisotropic mean, 3 x 11": xfilter.generate_kernel({"eta_rho": 3, "xi_rho": 11}, temp),
}
fig, axes = plt.subplots(1, 3, figsize=(9, 2.6), constrained_layout=True)
for ax, (title, kernel) in zip(axes, kernels.items()):
    kernel.transpose(*dims).plot(ax=ax, cmap="viridis", add_colorbar=False)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")

# %%
# Tolerating missing values
# -------------------------
#
# ``na_thres`` is the fraction of missing neighbours that is accepted. The more tolerant,
# the less the coast is eroded, until the field is extended over the land with ``na_thres=1``.

plot_maps(
    {f"na_thres={thres}": xfilter.smooth(temp, 9, na_thres=thres) for thres in (0, 0.3, 0.6, 1)},
    (2, 2),
    (9, 5.5),
)

# %%
# Normalizing or not
# ------------------
#
# :func:`~xoa.filter.smooth` is :func:`~xoa.filter.convolve` with ``normalize=True``.
# Without normalization, the result is a weighted sum, which is not a smoothed field:
# it is biased by the number of valid points, and by the sum of the weights.
# Kernels are not normalized either, so you must do it yourself if you need a sum.

conv = xfilter.convolve(temp, 5, na_thres=1)
smooth = xfilter.smooth(temp, 5, na_thres=1)
print("Temperature at some point:", float(temp[10, 20]))
print("Smoothed:", float(smooth[10, 20]), "-- Convolved:", float(conv[10, 20]))
print("Smoothed at the coast:", float(smooth[1, 25]), "-- Convolved:", float(conv[1, 25]))

# %%
# Extending a field into the land
# -------------------------------
#
# :func:`~xoa.filter.erode_coast` fills the land points that are close to the coast
# with the smoothed field, which is useful before an interpolation or a derivative that needs
# valid neighbours. ``until`` is the number of iterations, or a mask where the data must remain
# missing.

plot_maps({"Original": temp, "Eroded 4 times": xfilter.erode_coast(temp, until=4)}, (1, 2), (7, 3))

# %%
# Removing grid-scale noise: Shapiro filters
# ------------------------------------------
#
# Ocean models use Shapiro filters to damp the noise at the scale of the grid, 2 grid
# points, with a minimal impact on larger scales. The :func:`~xoa.filter.shapiro_kernel`
# of xoa is a cross, with weights of 1 on the four neighbors, and it is the one used for
# the erosion of masks. The classic filter is the orthogonal one, ``[1, 2, 1]``
# along each dimension, that you can easily build with your own kernel.
# Let's add a checkerboard noise to the temperature, which is the worst case.

checkerboard = (-1.0) ** (np.arange(temp.shape[0])[:, None] + np.arange(temp.shape[1]))
noisy = temp + 0.5 * checkerboard
kernels = {
    "Shapiro cross": xfilter.shapiro_kernel(dims),
    "Orthogonal [1, 2, 1]": {"eta_rho": [1, 2, 1], "xi_rho": [1, 2, 1]},
}
filtered = {name: xfilter.smooth(noisy, kernel, na_thres=1) for name, kernel in kernels.items()}
plot_maps({"Noisy": noisy, **filtered}, (1, 3), (10, 3))
for name, kernel in kernels.items():
    residual = filtered[name] - xfilter.smooth(temp, kernel, na_thres=1)
    print(f"Typical remaining noise with {name}: {float(abs(residual).median()):.3f} °C")

# %%
# The orthogonal kernel removes the checkerboard noise entirely, while the cross
# only divides its amplitude by three at each pass.
#
# Kernel sizes are in grid points
# -------------------------------
#
# A kernel knows nothing about distances: a 9-point kernel has a footprint that depends
# on the resolution of the grid. This is not an issue for the CROCO grid above, whose
# resolution is almost constant (15 km), but it is for a regular longitude and latitude grid, where
# the zonal resolution decreases with latitude.
# Choose a kernel size from the resolution at the latitude of your data, or regrid your data to a
# uniform grid beforehand, and prefer a regional filtering for large domains.

lon = xr.DataArray(np.arange(-10, 10, 0.25), dims="lon", attrs={"standard_name": "longitude"})
lat = xr.DataArray(np.arange(0, 80, 0.25), dims="lat", attrs={"standard_name": "latitude"})
sst = xr.DataArray(
    np.zeros((lat.size, lon.size)), dims=("lat", "lon"), coords={"lon": lon, "lat": lat}
)
dx, dy = xgrid.get_dx_dy(sst)
kernel_size = 9

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
ax.plot(lat, kernel_size * dx.isel(lon=0) * 1e-3, label="Along x")
ax.plot(lat, kernel_size * dy.isel(lon=0) * 1e-3, label="Along y")
ax.set_xlabel("Latitude (°N)")
ax.set_ylabel("Footprint of the kernel (km)")
ax.set_title(f"A {kernel_size}-point kernel on a regular lon/lat grid")
ax.legend()
ax.set_ylim(bottom=0)
