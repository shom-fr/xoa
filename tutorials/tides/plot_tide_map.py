#!/usr/bin/env python
"""
Analyse the tide of a gridded field
===================================

In this tutorial, we show how to:

* analyse sea level and currents of shape ``(time, y, x)`` in a single call,
* draw a co-tidal chart and the spectrum of the constituents,
* predict the tide on the whole grid at one given moment,
* describe the tidal currents with ellipses on maps,
* classify the tide with the form number.

The concepts are explained in the :ref:`indepth.tide` guide.
The fields are synthetic: a standing wave in a gulf that is closed at its eastern end.
See also: :ref:`sphx_glr_tutorials_tides_plot_tide_station.py`.
"""

# %%
# Initialisations
# ---------------

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from xoa import plot as xplot
from xoa import tide as xtide
from xoa.tide import plot as tplot

xr.set_options(display_style="text")
MAP_KW = {"figsize": (6, 4)}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Synthetic sea level and currents
# --------------------------------
#
# One month of hourly data on a 20 x 30 grid. The amplitudes grow toward the closed
# end and the currents are strongest at the mouth, with ellipses that are more open
# in the south.

names = ["M2", "S2", "K1", "O1"]
base_amp = {"M2": 1.5, "S2": 0.5, "K1": 0.3, "O1": 0.2}
base_pha = {"M2": 0.5, "S2": 1.0, "K1": 2.0, "O1": 3.0}

x = xr.DataArray(np.linspace(-6, -2, 30), dims="x", name="lon", attrs={"units": "degrees_east"})
y = xr.DataArray(np.linspace(47, 49, 20), dims="y", name="lat", attrs={"units": "degrees_north"})
x, y = x.assign_coords(lon=x), y.assign_coords(lat=y)
s = (x - x.min()) / (x.max() - x.min())  # 0 at the mouth, 1 at the closed end
t = np.datetime64("2024-03-01") + np.arange(30 * 24) * np.timedelta64(1, "h")
hours = xr.DataArray(np.arange(t.size) * 1.0, dims="time", coords={"time": t})
omega = dict(zip(names, xtide.get_frequencies(names) * 3600))  # rad per hour

ssh = sum(
    base_amp[n]
    * (0.5 + 0.5 * s)
    * (1 + 0.1 * (y - 48))
    * np.cos(omega[n] * hours - base_pha[n] - 0.3 * s)
    for n in names
)
ssh += 0.05 * np.random.default_rng(0).standard_normal(ssh.shape)
ssh = ssh.transpose("time", "y", "x").rename("ssh").assign_attrs(units="m")

umax = (0.8 * (1 - s) + 0.1) * (1 + 0.2 * (y - 48))
u = sum(0.5 * base_amp[n] * umax * np.cos(omega[n] * hours - base_pha[n]) for n in names).transpose(
    "time", "y", "x"
)
v = sum(
    0.5 * base_amp[n] * umax * (y - 47) / 2 * np.sin(omega[n] * hours - base_pha[n]) for n in names
).transpose("time", "y", "x")
print(ssh.dims, ssh.shape)

# %%
# Harmonic analysis of the whole grid
# -----------------------------------
#
# The pseudo-inverse of the design matrix is computed once and applied to all the points.
# The ``greenwich`` option gives the phase lags that are used by tidal atlases.

coef = xtide.harmonic_analysis(ssh, names=names, greenwich=True)
print(coef)

# %%
# Co-tidal chart
# --------------
#
# :func:`~xoa.tide.plot.plot_cotidal` maps the amplitude in colors and the phase in lines.

tplot.plot_cotidal(coef, "M2", map_kw=MAP_KW)
plt.tight_layout()

# %%
# Prediction at one moment
# ------------------------
#
# The coefficients can be evaluated at any time, here a single one, which is not
# in the analysed period. The result is the tide on the whole grid at this moment.

t0 = np.array([np.datetime64("2024-04-02T06:00")])
snapshot = xtide.harmonic_predict(coef, t0).isel(time=0)
print(snapshot.dims, snapshot.shape)

xplot.plot_field(
    snapshot.assign_attrs(long_name="Sea level", units="m"), cmap="auto", map_kw=MAP_KW
)
plt.title("Predicted tide on 2 April 06:00")
plt.tight_layout()

# %%
# Tidal currents and ellipses
# ---------------------------
#
# For the currents, each constituent describes an ellipse at each point.
# The semi-minor axis is negative when the current rotates clockwise.

coef_uv = xtide.tidal_ellipses(xtide.harmonic_analysis_uv(u, v, names=names))
m2 = coef_uv.sel(constituent="M2")

tplot.plot_tidal_ellipses(coef_uv, "M2", ref=0.5, background=m2.semi_major, step=2, map_kw=MAP_KW)
plt.tight_layout()

# %%
# Constituents can be compared side by side, coloured by their phase, and one ellipse
# can be detailed at a given point.

tplot.plot_tidal_constituents(
    coef_uv, ["M2", "S2", "K1"], color_by="phase", step=2, figsize=(10, 3.2)
)

# %%
tplot.plot_tidal_ellipse(coef_uv.isel(x=5, y=10), "M2")

# %%
# Spectrum
# --------
#
# The mean amplitude of the constituents against their frequency tells which dominate.

tplot.plot_tidal_spectrum(coef, record_days=30)

# %%
# Form number
# -----------
#
# The form number ``(K1 + O1) / (M2 + S2)`` tells whether the tide is
# semi-diurnal (below 0.25), mixed or diurnal (above 3).

form = xtide.form_number(coef)
print(f"form number from {float(form.min()):.2f} to {float(form.max()):.2f}")
plt.show()
