#!/usr/bin/env python
"""
Wave roses of a buoy
====================

In this tutorial, we show how to:

* plot a rose of the significant wave height and the wave direction of a buoy,
* mask the bad quality data,
* compare the seasons and the peak period.

The concepts are explained in the :ref:`indepth.plot` guide.
"""

# %%
# Initialisations
# ---------------

import matplotlib.pyplot as plt

import xoa
from xoa.plot import plot_rose

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# The hourly wave parameters of the Pierres Noires buoy, in the Iroise Sea, from 2013 to 2024.
# The quality flag is 1 for the good values.

ds = xoa.open_data_sample("OBS/WAVES/wave_buoy_6200069_2013-2024.nc")
ds = ds.where(ds.hs_qc == 1)
ds

# %%
# The waves come mostly from the west. The direction is the one they come from.
# :func:`~xoa.plot.plot_rose` draws the rose, which is also available as the
# :meth:`~xoa.accessors.XoaPlotAccessor.rose` method of the ``plot`` accessor.

rose = plot_rose(ds.hs, ds.direction)
rose.ax.set_title("Significant wave height")

# %%
# Winter and summer
# -----------------
#
# Waves are higher and more regular in winter. The same bins are used for both roses.

edges = [0, 1, 2, 3, 4, 6, 8]
fig = plt.figure(figsize=(9, 4.5))
fig.subplots_adjust(right=0.72)  # room for the legend
for i, (name, months) in enumerate({"Winter": [12, 1, 2], "Summer": [6, 7, 8]}.items()):
    sub = ds.isel(time=ds.time.dt.month.isin(months))
    rose = plot_rose(
        sub.hs, sub.direction, edges=edges, rmax=40, fig=fig, subplot=121 + i, legend=i == 1
    )
    rose.ax.set_title(name)

# %%
# Peak period
# -----------
#
# Long periods are swell and short ones are wind sea, which has wider directions.

rose = plot_rose(ds.tp, ds.direction, edges=[0, 6, 8, 10, 12, 14, 16], cmap="YlOrBr")
rose.ax.set_title("Peak period")
