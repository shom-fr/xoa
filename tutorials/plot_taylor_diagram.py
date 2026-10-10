#!/usr/bin/env python
# coding: utf-8
"""
Compare simulations with a reference using Taylor diagrams
==========================================================

In this tutorial, we show how to:

* plot one point per model, with a legend and different markers,
* add drop shadows to the markers,
* color the points with values,
* handle negative correlations,
* draw already normalized statistics without a reference array.

"""

# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from xoa.plot import plot_taylor, add_shadow
from xoa.core import plot as cplot

# %%
# Synthetic data: a reference and models that are more or less correlated with it,
# on dimensions ``time`` and ``x``.

rng = np.random.default_rng(0)
ref = xr.DataArray(
    np.sin(np.linspace(0, 12, 200))[:, None] + 0.3 * rng.normal(size=(200, 10)),
    dims=("time", "x"),
    attrs={"units": "m"},
)
noise = [0.2, 0.5, 1.0, 1.5]
scales = [1.0, 1.2, 0.7, 1.4]
models = xr.concat(
    [s * ref + n * rng.normal(size=ref.shape) for s, n in zip(scales, noise)],
    dim=xr.DataArray(["MARS", "CROCO", "HYCOM", "NEMO"], dims="model", name="model"),
)

# %%
# One point per model
# -------------------
# The statistics are computed over ``time`` and ``x``. The remaining ``model``
# dimension creates the points, whose labels are the coordinates.
# A blurred drop shadow is added to the markers with :func:`xoa.plot.add_shadow`.
# The artists of the diagram are available as attributes, like ``markers`` and ``reference``.
# It must be called in the same cell as the plot, since the figure is captured
# at the end of the cell.

diagram = plot_taylor(models, ref, dim=("time", "x"), markers=["o", "s", "^", "D"])
add_shadow(
    diagram.markers + [diagram.reference], width=3, xoffset=2, yoffset=-2, alpha=0.4, ax=diagram.ax
)

# %%
# Normalized, with colored points
# -------------------------------
# With ``values``, the markers are colored and the labels are written next to them.

cost = xr.DataArray([3.0, 5.0, 2.0, 8.0], dims="model", attrs={"long_name": "Cost", "units": "h"})
plot_taylor(models, ref, dim=("time", "x"), normalize=True, values=cost, cmap="viridis")

# %%
# Negative correlations
# ---------------------
# The diagram becomes a half circle when needed.

diagram = plot_taylor(
    xr.concat([models, -models.isel(model=[1])], dim="model"),
    ref,
    dim=("time", "x"),
    labels=["MARS", "CROCO", "HYCOM", "NEMO", "-CROCO"],
    normalize=True,
)
diagram.ax.figure.set_size_inches(8, 4.3)
diagram.ax.figure.subplots_adjust(left=0.02, right=0.82, top=0.9, bottom=0.13)

# %%
# Without a reference array
# -------------------------
# Normalized statistics can be drawn directly, the reference standard deviation being 1.

diagram = cplot.plot_taylor([0.9, 1.2, 0.7], [0.95, 0.8, -0.3], labels=["a", "b", "c"], rmax=1.5)
diagram.ax.figure.set_size_inches(8, 4.3)
diagram.ax.figure.subplots_adjust(left=0.02, right=0.82, top=0.9, bottom=0.13)
