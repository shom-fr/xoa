#!/usr/bin/env python
"""
Compare Hycom3d with a GDP drifter
==================================

In this tutorial, we show how to:

* decode dataset so that it is easy to access generic coordinates and variables,
* compute depths from layer thicknesses,
* interpolate currents from U and V positions to T position on an arakawa C grid,
* perform a 4D interpolation with a variable depth coordinate to random positions,
* make an horizontal slice of a 4D variable with a variable depth coordinate.

The concepts are explained in the :ref:`indepth.meta` guide.
See also: :ref:`sphx_glr_tutorials_evaluation_plot_mercator_argo.py`, :ref:`sphx_glr_tutorials_vertical_plot_croco_section.py`.
"""
# %%
# Initialisations
# ---------------

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import xoa
import xoa.geo as xgeo
import xoa.meta as xmeta
from xoa.core.plot import add_quiverkey, get_quiverkey_value
from xoa.grid import dz2depth, shift
from xoa.plot import plot_double_minimap, plot_flow
from xoa.regrid import grid2loc, regrid1d

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Set the internal Hycom naming specifications as the current ones
hycom_cfg_file = xoa.get_meta_config_file("hycom")
print(hycom_cfg_file)
xmeta.set_meta_specs(hycom_cfg_file)

# %%
# Note that, since this file is internal, you can do it simply with::
#
#        xmeta.set_meta_specs("hycom")
#
# Here is what these CF specifications contain

with open(hycom_cfg_file) as f:
    print(f.read())

# %%
# Note that most of these specifications are not needed to decode the dataset, i.e to find
# known variables and coordinates, since the default specifications help for that.

# %%
# Read and decode the datasets
# ----------------------------
#
# Hycom velocities
# ~~~~~~~~~~~~~~~~
#
# U and V are stored in separate files at U and V locations on the Hycom Arakawa C-Grid.
# We choose to rename the dataset contents so that we can use generic names to access it.
# We call for that the :meth:`~xarray.Dataset.xoa.decode` accessor method.
# Note that the staggered grid location is automatically appended to horizontal
# coordinates and dimensions of the U and V velocity components since they are
# placed at U and V locations. This prevent any conflict with the bathymetry and
# layer thickness horizontal coordinates and dimensions, which point to the T location.

# %%
# U velocity component
u_hycom = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.u.nc").xoa.decode().u.squeeze(drop=True)
print(u_hycom)

# %%
# V velocity component
v_hycom = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.v.nc").xoa.decode().v.squeeze(drop=True)

# %%
# Layer thicknesses dataset
hycom = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.h.nc").xoa.decode().squeeze(drop=True)
print(hycom)

# %%
# We now compute the depths from the layer thicknesses at the T location
# thanks to the :func:`xoa.grid.dz2depth` function:
# we integrate from a null SSH and interpolate the depth from W locations to T locations.

hycom.coords["depth"] = dz2depth(hycom.dz, centered=True)

# %%
# Then we interpolate the velocity components to the T location with
# the :func:`xoa.grid.shift` function.
# The call to :meth:`reloc <xoa.meta.MetaSpecs.reloc>` helps removing the
# staggered grid location prefixes.

ut_hycom = shift(u_hycom, {"x": "left", "y": "left"}).xoa.reloc(u=False)
vt_hycom = shift(v_hycom, {"x": "left", "y": "left"}).xoa.reloc(v=False)
hycom["u"] = ut_hycom.assign_coords(**hycom.coords)
hycom["v"] = vt_hycom.assign_coords(**hycom.coords)

# %%
# So, finally we obtain:

print(hycom)

# %%
# GDP drifter
# ~~~~~~~~~~~
#
# The drifter comes as a `csv` file and we read it as :class:`pandas.DataFrame` instance.

csv_name = xoa.get_data_sample("OBS/DRIFTERS/gdp-6203641.csv")
drifter = pd.read_csv(
    csv_name, header=0, skiprows=[1], parse_dates=[0], index_col=0, usecols=[2, 3, 4, 5, 6, 7]
)

# %%
# Since the sampling is not that nice, we resample it to 3-hour intervals.

drifter = drifter.resample("3h").mean()

# %%
# We switch to default naming conventions for the drifter.

xmeta.set_meta_specs("default")


# %%
# We convert it to and :class:`xarray.Dataset`, fix the time coordinate and decode it.

drifter = drifter.to_xarray().assign_coords(time=drifter.index.values).xoa.decode()

# %%
# We drop missing values.

drifter = drifter.where(~drifter.lon.isnull() & ~drifter.lat.isnull(), drop=True)

# %%
# We add a constant depth of 15 m.

drifter.coords["depth"] = (
    "time",
    np.full(drifter.sizes["time"], 15.0),
    {"standard_name": "ocean_depth", "units": "m", "positive": "down"},
)

# %%
# Here is what we obtain.

print(drifter)

# %%
# Compute and interpolate velocities
# ----------------------------------
#
# We compute the drifter velocity components

drifter["u"] = (
    drifter.lon.differentiate("time", datetime_unit="s") * xgeo.EARTH_RADIUS * np.pi / 180
)
drifter["u"] *= np.cos(np.radians(drifter.lat.values))
drifter["v"] = (
    drifter.lat.differentiate("time", datetime_unit="s") * xgeo.EARTH_RADIUS * np.pi / 180
)

# %%
# We make a 4D linear interpolation the Hycom velocity to the drifter positions with :func:`xoa.interp.grid2loc`.

uloc = grid2loc(hycom["u"], drifter)
vloc = grid2loc(hycom["v"], drifter)

# %%
# Instead of just showing the velocities along the drifter positions,
# we can plot the mean model velocities over the same period as a background.
# So we interpolate them at 15 m with :func:`xoa.regrid.regrid1d` and compute a time average.

d15 = xr.DataArray([15.0], dims="depth", name="depth")
uh15 = regrid1d(hycom["u"], d15).squeeze(drop=True).mean(dim="time")
vh15 = regrid1d(hycom["v"], d15).squeeze(drop=True).mean(dim="time")

# %%
# Plot
# ----
#
# The geographic extent is easily computed with the :func:`xoa.geo.get_extent` function.
# The background currents are plotted with the :func:`xoa.plot.plot_flow` function.
# A reference arrow with a round value is added with :func:`xoa.core.plot.get_quiverkey_value`
# and :func:`xoa.core.plot.add_quiverkey`, and a minimap with :func:`xoa.plot.plot_double_minimap`.

pmerc = ccrs.Mercator()
pcarr = ccrs.PlateCarree()
fig, ax = plt.subplots(
    figsize=(6.5, 6.5),
    subplot_kw={"facecolor": "teal", "projection": pmerc},
    constrained_layout=True,
)

ax.gridlines(draw_labels=True, dms=True)
ax.set_extent(xgeo.get_extent(uh15))
kwqv = dict(scale_units="dots", scale=0.1 / 20, units="dots", transform=pcarr)
plot_flow(uh15, vh15, axes=ax, transform=pcarr, color="w", alpha=(0.3, 1), linewidth=0.6)
qv = ax.quiver(
    uloc.lon.values,
    uloc.lat.values,
    uloc.values,
    vloc.values,
    color="w",
    width=2,
    label="Model",
    **kwqv,
)
ax.plot(drifter.lon.values, drifter.lat.values, '-', color="C1", transform=pcarr, lw=0.5)
ax.quiver(
    drifter.lon.values,
    drifter.lat.values,
    drifter.u.values,
    drifter.v.values,
    color="C1",
    label="Drifter",
    width=2,
    **kwqv,
)
qvalue = get_quiverkey_value(np.hypot(uloc, vloc))
add_quiverkey(ax, qv, qvalue, r"$m\,s^{-1}$", pos=(0.0, 1.07))
plt.legend()
plot_double_minimap(drifter, parent=ax, loc="lower left", size=0.15, regional_ax="right")

# %%
# The discrepancies between the lagrangian and mean eulerian currents highlight
# the variability due to the mesocale activity.
# The differences between the observed and modeled currents are partly due to
# the lack of data assimilation in the model.
