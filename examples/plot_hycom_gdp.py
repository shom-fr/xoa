#!/usr/bin/env python
# coding: utf-8
"""
Compare Hycom3d with a GDP drifter
==================================

In this notebook, we show :

* how to decode dataset so that it is easy to access generic coordinates and variables,
* how to compute depths from layer thicknesses,
* how to interpolate currents from U and V positions to T position on an arakawa C grid,
* how to perform a 4D interpolation with a variable depth coordinate to random positions,
* how to make an horizontal slice of a 4D variable with a variable depth coordinate.

"""
# %%
# Initialisations
# ---------------

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xoa
from xoa.grid import dz2depth, shift
from xoa.regrid import grid2loc, regrid1d
import xoa.cf as xcf
import xoa.geo as xgeo
from xoa.plot import plot_flow

xr.set_options(display_style="text")

# %%
# Register the :ref:`xoa <accessors>` accessors :

xoa.register_accessors()

# %%
# Register the Hycom naming specifications

xcf.register_cf_specs(xoa.get_data_sample("hycom.cfg"))

# %%
# Here is what these CF specifications contain

with open(xoa.get_data_sample("hycom.cfg")) as f:
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
u_hycom = xoa.open_data_sample("hycom.gdp.u.nc").xoa.decode().u.squeeze(drop=True)
print(u_hycom)

# %%
# V velocity component
v_hycom = xoa.open_data_sample("hycom.gdp.v.nc").xoa.decode().v.squeeze(drop=True)

# %%
# Layer thicknesses dataset
hycom = xoa.open_data_sample("hycom.gdp.h.nc").xoa.decode().squeeze(drop=True)
print(hycom)

# %%
# We now compute the depths from the layer thicknesses at the T location
# thanks to the :func:`xoa.grid.dz2depth` function:
# we integrate from a null SSH and interpolate the depth from W locations to T locations.

hycom.coords["depth"] = dz2depth(hycom.dz, centered=True)

# %%
# Then we interpolate the velocity components to the T location with
# the :func:`xoa.grid.shift` function.
# The call to :meth:`reloc <xoa.cf.CFSpecs.reloc>` helps removing the
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

drifter = xoa.open_data_sample("gdp-6203641.csv", header=0, skiprows=[1], parse_dates=[2], index_col=2)

# %%
# Since the sampling is not that nice, we resample it to 3-hour intervals.

drifter = drifter.resample("3H").mean()

# %%
# We convert it to and :class:`xarray.Dataset`, fix the time coordinate and decode it.

drifter = drifter.to_xarray().assign_coords(time=drifter.index.values).xoa.decode()

# %%
# We drop missing values.

drifter = drifter.where(~drifter.lon.isnull() & ~drifter.lat.isnull(), drop=True)

# %%
# We add a constant depth of 15 m.

drifter.coords["depth"] = drifter.lon*0 + 15

# %%
# Here is what we obtain.

print(drifter)

# %%
# Compute and interpolate velocities
# ----------------------------------
#
# We compute the drifter velocity components

drifter["u"] = drifter.lon.differentiate("time", datetime_unit="s") * xgeo.EARTH_RADIUS*np.pi/180
drifter["u"] *= np.cos(np.radians(drifter.lat.values))
drifter["v"] = drifter.lat.differentiate("time", datetime_unit="s") * xgeo.EARTH_RADIUS*np.pi/180

# %%
# We make a 4D linear interpolation the Hycom velocity to the drifter positions with :func:`xoa.regrid.grid2loc`.

uloc = grid2loc(hycom["u"], drifter)
vloc = grid2loc(hycom["v"], drifter)

# %%
# Instead of just showing the velocities along the drifter positions,
# we can plot the mean model velocities over the same period as a background.
# So we interpolate them at 15 m with :func:`xoa.regrid.regrid1d` and compute a time average.

d15 = xr.DataArray([15.], dims="depth", name="depth")
uh15 = regrid1d(hycom["u"], d15).squeeze(drop=True).mean(dim="time")
vh15 = regrid1d(hycom["v"], d15).squeeze(drop=True).mean(dim="time")

# %%
# Plot
# ----
#
# The geographic extent is easily computed with the :func:`xoa.geo.get_extent` function.
# The background currents are plotted with the :func:`xoa.plot.plot_flow` function.

pmerc = ccrs.Mercator()
pcarr = ccrs.PlateCarree()
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"facecolor": "teal", "projection": pmerc})
ax.gridlines(draw_labels=True, dms=True)
ax.set_extent(xgeo.get_extent(uh15))
kwqv = dict(scale_units="dots", scale=0.1/20, units="dots", transform=pcarr)
plot_flow(uh15, vh15, axes=ax, transform=pcarr, color="w", alpha=(.3, 1), linewidth=.6)
qv = ax.quiver(uloc.lon.values, uloc.lat.values, uloc.values, vloc.values,
    color="w", width=2, label="Model", **kwqv)
ax.plot(drifter.lon.values, drifter.lat.values, '-', color="C1", transform=pcarr, lw=.5)
ax.quiver(drifter.lon.values, drifter.lat.values, drifter.u.values, drifter.v.values,
    color="C1", label="Drifter", width=2, **kwqv)
plt.quiverkey(qv, 0.1, 1.06, 0.1, r"0.1 $m\,s^{-1}$", color="k", alpha=1, labelpos="E")
plt.legend();

# %%
# The discrepancies between the lagrangian and mean eulerian currents highlight
# the variability due to the mesocale activity.
# The differences between the observed and modeled currents are partly due to
# the lack of data assimilation in the model.
