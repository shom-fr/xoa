#!/usr/bin/env python
"""
Analyse and predict the tide of a station
=========================================

In this tutorial, we show how to:

* choose the tidal constituents that a record can resolve,
* analyse a sea level series with :func:`xoa.tide.harmonic_analysis`,
* predict the tide, also with the uncertainty of the analysis,
* extract the high waters and the skew surge, i.e. the meteorological part of the surge.

The concepts are explained in the :ref:`indepth.tide` guide.
The series is synthetic: a tide with a storm surge and some noise.
See also: :ref:`sphx_glr_tutorials_tides_plot_tide_map.py`.
"""

# %%
# Initialisations
# ---------------

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xoa import tide as xtide
from xoa.tide import plot as tplot

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})


def concise_dates(axs):
    """Use compact date labels on the x axes"""
    for ax in np.ravel(axs):
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


# %%
# Synthetic sea level
# -------------------
#
# Two months of hourly data: the main constituents, a storm surge of 0.6 m and some noise.

rng = np.random.default_rng(0)
time = np.datetime64("2024-01-01") + np.arange(60 * 24) * np.timedelta64(1, "h")
hours = np.arange(time.size)
true = {"M2": (0.9, 1.0), "S2": (0.3, 2.0), "N2": (0.18, 0.5), "K1": (0.1, 4.0), "O1": (0.08, 5.0)}
omega = xtide.get_frequencies(list(true)) * 3600  # rad per hour
tide = sum(a * np.cos(w * hours - p) for (a, p), w in zip(true.values(), omega))
surge = 0.6 * np.exp(-(((hours - 30 * 24) / 18.0) ** 2))
obs = xr.DataArray(
    tide + surge + 0.05 * rng.standard_normal(hours.size),
    dims="time",
    coords={"time": time},
    name="ssh",
    attrs={"units": "m"},
)

# %%
# Which constituents can be resolved?
# -----------------------------------
#
# Two constituents can be separated when the record is longer than their beat period.
# :func:`~xoa.tide.auto_constituents` applies this Rayleigh criterion.

record_days = (time[-1] - time[0]) / np.timedelta64(1, "D")
names = xtide.auto_constituents(record_days)
print(f"{record_days:.0f} days: {names}")
print("S2 and K2 need", round(xtide.beat_period("S2", "K2")), "days")

# %%
# Harmonic analysis
# -----------------
#
# The result is a dataset of amplitudes and phases indexed by constituent,
# with the white-noise uncertainties.

names = ["M2", "S2", "N2", "K1", "O1"]
coef = xtide.harmonic_analysis(obs, names=names)
print(coef[["amp", "amp_std", "phase"]].to_dataframe().round(3))

# %%
# The spectrum shows which constituents dominate, with their uncertainties and the
# Rayleigh resolution of the record.

tplot.plot_tidal_spectrum(coef, record_days=record_days)

# %%
# Prediction
# ----------
#
# The tide is predicted over the same period and the surge is what remains.
# An ensemble drawn from the uncertainties gives the spread of the prediction.
# Since the surge is not white noise, the uncertainties are inflated.

pred = xtide.harmonic_predict(coef, time)
ens = xtide.harmonic_predict(
    xtide.TidalHarmonic(time, names).ensemble(coef, n_members=30, seed=1, inflate=3), time
)

fig, axs = plt.subplots(2, 1, figsize=(7, 4.5), constrained_layout=True)
sel = slice(28 * 24, 34 * 24)
axs[0].plot(time[sel], obs[sel], "k", lw=1, label="observed")
axs[0].plot(time[sel], pred[sel], "C0", lw=1, label="predicted tide")
axs[0].fill_between(
    time[sel], ens[sel].min("member"), ens[sel].max("member"), color="C0", alpha=0.3
)
axs[0].set_ylabel("Sea level (m)")
axs[0].legend(loc="upper right", ncols=2)
axs[1].plot(time, obs - pred, "C3", lw=1)
axs[1].set_ylabel("Residual (m)")
axs[1].set_title("Surge and noise")
axs[0].set_title("Observed and predicted sea level")
concise_dates(axs)

# %%
# High waters and skew surge
# --------------------------
#
# The skew surge is the difference between the highest observed level around a predicted
# high water and the predicted high water, whatever their timing.
# It is the quantity that is used to characterize storm surges.
# :func:`~xoa.tide.high_waters` flags the predicted high waters and
# :func:`~xoa.tide.skew_surge` gives the skew surge at these times, and nan elsewhere.
# **Look for:** the observed levels that exceed the predicted tide at the storm, and the
# skew surge that peaks there and is about zero outside.

hw = xtide.high_waters(pred)
surge_hw = xtide.skew_surge(obs, pred, hw).dropna("time")
print(
    f"{int(hw.sum())} high waters, largest skew surge {float(surge_hw.max()):.2f} m "
    f"on {str(surge_hw.idxmax().values)[:10]}"
)

win = slice(27 * 24, 34 * 24)  # a week around the storm
tmax = surge_hw.idxmax().values
pmax = float(pred.sel(time=tmax))  # predicted high water of the largest skew surge
fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=True, constrained_layout=True)
axs[0].plot(time[win], obs[win], "C0", lw=1.2, label="Observed")
axs[0].plot(time[win], pred[win], "k", lw=1, ls="--", label="Predicted tide")
axs[0].fill_between(
    time[win], pred[win], obs[win], where=(obs > pred)[win], color="C3", alpha=0.25, lw=0
)
axs[0].plot(time[win], pred.where(hw)[win], "ko", ms=4, ls="", label="Predicted high waters")
axs[0].annotate(
    "",
    xy=(tmax, pmax + float(surge_hw.max())),
    xytext=(tmax, pmax),
    arrowprops=dict(arrowstyle="<->", color="C3", lw=1.5),
)
axs[0].text(tmax, pmax + float(surge_hw.max()) + 0.1, "skew surge", color="C3", ha="center")
axs[0].set_ylabel("Sea level (m)")
axs[0].set_ylim(top=pmax + 1.2)
axs[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncols=3, frameon=False)
axs[1].fill_between(surge_hw.time, 0, surge_hw, color="C3", alpha=0.25, lw=0)
axs[1].plot(surge_hw.time, surge_hw, "C3o-", ms=3, lw=0.8)
axs[1].axhline(0, color="k", lw=0.5)
axs[1].set_ylabel("Skew surge (m)")
axs[1].set_xlim(time[win][0], time[win][-1])
concise_dates(axs)
plt.show()
