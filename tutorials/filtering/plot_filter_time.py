#!/usr/bin/env python
"""
Filter time series: remove the tide
===================================

In this tutorial, we show how to:

* remove the tidal signal of an hourly series with the Demerliac, Godin and mean filters,
* check what each filter keeps and removes, as a function of the period,
* understand what happens at the ends of the series, and with gaps,
* apply the filters to series that are sampled more often than every hour,
* know when a filter refuses a series: irregular or too coarse time steps.

The concepts are explained in the :ref:`indepth.filtering` guide.
See also: :ref:`sphx_glr_tutorials_filtering_plot_filter_space.py`.
"""

# %%
# Initialisations
# ---------------

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xoa import tide as xtide

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

M2, K1 = 12.42, 23.93  # periods of the main semi-diurnal and diurnal tidal waves, in hours
FILTERS = ("demerliac", "godin", "mean")


def make_series(step=1.0, ndays=30, name="ssh"):
    """Slow signal plus tide, with a time step in hours"""
    hours = np.arange(0, 24 * ndays, step)
    time = np.array("2020-01-01", dtype="M8[s]") + (hours * 3600).astype("m8[s]")
    slow = 0.3 * np.sin(2 * np.pi * hours / (24 * 15))
    tide = np.cos(2 * np.pi * hours / M2) + 0.4 * np.cos(2 * np.pi * hours / K1)
    kw = dict(dims="time", coords={"time": time})
    return (
        xr.DataArray(slow + tide, name=name, attrs={"units": "m"}, **kw),
        xr.DataArray(slow, **kw),
    )


# %%
# Three filters for the tide
# --------------------------
#
# :func:`~xoa.tide.tidal_filter` convolves a series with weights that are designed for
# hourly data: ``"demerliac"`` (25 hours), ``"godin"`` (70 hours) and ``"mean"``, a
# 24 hour moving average. They are normalized so a constant signal is unchanged.

ssh, slow = make_series()
filtered = {name: xtide.tidal_filter(ssh, name) for name in FILTERS}

fig, ax = plt.subplots(figsize=(9, 3), constrained_layout=True)
ssh.plot(ax=ax, label="Raw", color="0.7")
for name, da in filtered.items():
    da.plot(ax=ax, label=name.title())
slow.plot(ax=ax, color="k", ls=":", label="Slow signal")
ax.legend(ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_title("Removing the tide")
ax.set_xlabel("")

# %%
# What do they keep?
# ------------------
#
# The frequency response of a filter is estimated by applying it to pure sinusoids and
# measuring the amplitude of what is left in the middle of the series. The Demerliac
# and Godin filters cancel the whole range of periods below about 1 day, and
# let the slower ones through. The simple mean cancels only the periods that divide 24 hours,
# which are close to the tidal waves, and leaks in between, with up to 20% of the amplitude at periods of 18 hours and 10% around 10 hours.

periods = np.geomspace(3, 24 * 10, 120)
hours = np.arange(24 * 60)
time = np.array("2020-01-01", dtype="M8[h]") + hours
response = {name: [] for name in FILTERS}
for period in periods:
    wave = xr.DataArray(np.cos(2 * np.pi * hours / period), dims="time", coords={"time": time})
    for name in FILTERS:
        out = xtide.tidal_filter(wave, name).isel(time=slice(24 * 20, 24 * 40))
        response[name].append(float(abs(out).max()))

fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
for name, values in response.items():
    ax.semilogx(periods, values, label=name.title())
for period, label in ((M2, "M2"), (K1, "K1")):
    ax.axvline(period, color="0.6", ls=":")
    ax.annotate(label, (period, 1.02), ha="center", va="bottom", color="0.4")
ax.set_xlabel("Period (hours)")
ax.set_ylabel("Amplitude of the response")
ax.set_ylim(0, 1.1)
ax.legend()
ax.set_title("What passes through the filters")

# %%
# The ends of the series
# ----------------------
#
# A filter that is 25 or 70 hours wide cannot be applied where the window sticks out
# of the series. With the default ``na_thres=0``, these points are masked, which removes
# about 12 and 35 hours at each end. With ``na_thres=1``, they are kept, and computed with
# the few points that are available, with a larger error. Intermediate values of ``na_thres``
# accept a fraction of missing points. The same rule applies to gaps inside a series.

ssh_gap = ssh.copy()
ssh_gap[24 * 10 : 24 * 10 + 6] = np.nan
fig, axes = plt.subplots(1, 3, figsize=(10, 2.8), sharey=True, constrained_layout=True)
for ax, na_thres in zip(axes, (0, 0.5, 1)):
    out = xtide.tidal_filter(ssh_gap, "godin", na_thres=na_thres)
    slow.plot(ax=ax, color="k", ls=":")
    out.plot(ax=ax, color="C1")
    ax.set_title(f"Godin, na_thres={na_thres}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=45)

# %%
# Series that are sampled more often
# ----------------------------------
#
# When the time step is less than an hour, the weights are interpolated to the time step of
# the series, so you can filter high-frequency data directly.

fig, ax = plt.subplots(figsize=(9, 2.8), constrained_layout=True)
for step in (1, 0.5, 0.1):
    ssh_hf, _ = make_series(step)
    out = xtide.tidal_filter(ssh_hf, "demerliac")
    out.plot(ax=ax, label=f"{int(step * 60)} min", lw=3 - 2 * step)
slow.plot(ax=ax, color="k", ls=":", label="Slow signal")
ax.legend(ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.set_title("Demerliac filter at different time steps")
ax.set_xlabel("")

# %%
# When a filter refuses your series
# ---------------------------------
#
# The weights make sense only for a regular time axis whose step is an hour or less.
# A series with a step that varies by more than ``dt_tol`` (1% by default) of its mean
# or that is coarser than an hour raises an error. Increasing ``dt_tol`` is not a fix, since
# the weights would then be applied to the wrong time steps.

ssh_3h, _ = make_series(3)
ssh_irregular = ssh.isel(time=np.r_[0:200, 205 : ssh.sizes["time"]])
for title, da, kwargs in (
    ("every 3 hours", ssh_3h, {}),
    ("with a gap in time", ssh_irregular, {}),
):
    try:
        xtide.tidal_filter(da, "demerliac", **kwargs)
        print(f"Series {title}: filtered")
    except Exception as exc:
        print(f"Series {title}: {type(exc).__name__}: {exc}")

# %%
# Prefer filling the gap of an irregular series with missing values, by reindexing it on
# a regular time axis, to a large ``dt_tol``: ``na_thres`` then controls how the gap
# propagates.

regular = ssh_irregular.reindex(time=ssh.time)
out = xtide.tidal_filter(regular, "demerliac", na_thres=0.2)
print(f"{int(out.isnull().sum())} points are masked out of {out.size}")
