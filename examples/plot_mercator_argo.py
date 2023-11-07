#!/usr/bin/env python
# coding: utf-8
"""
Compare Mercator to ARGO
========================

This notebook, we show:

* how to easily rename variables and coordinates in an ARGO and a Mercator datasets,
* how to plot a T-S diagram,
* how to stack geographical coordinates,
* how to compute a geographical distance between two datasets with coordinates,
* how to interpolate a vertical profile with different methods.
"""


# %%
# Initialisations
# ---------------
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import gsw
import cmocean

import xoa
import xoa.grid as xgrid
import xoa.regrid as xregrid
import xoa.cf as xcf
import xoa.coords as xcoords
import xoa.geo as xgeo
import xoa.plot as xplot

mpl.rc("axes", grid=True)
xr.set_options(display_style="text")

# %%
# Register the :ref:`xoa <accessors>` accessors :
xoa.register_accessors()

# %%
# Register the Mercator and ARGO naming specifications
xcf.register_cf_specs(xoa.get_data_sample("mercator.cfg"))
xcf.register_cf_specs(xoa.get_data_sample("argo.cfg"))

# %%
# Here is what these CF specifications contain
with open(xoa.get_data_sample("mercator.cfg")) as f:
    print(f.read())
with open(xoa.get_data_sample("argo.cfg")) as f:
    print(f.read())

# %%
# Read and decode the datasets
# ----------------------------

# %%
# ARGO profiles
# ~~~~~~~~~~~~~
#
# The ARGO data were downloaded with the `argopy <https://argopy.readthedocs.io/en/latest/>`_
# package with a piece of code close to the following::
#
#     from argopy import DataFetcher
#     adf = DataFetcher().float(7900573)
#     ds = adf.to_xarray().argo.point2profile().isel(N_PROF=slice(-10, None))
#     ds.to_netcdf("argo-7900573.nc")
#
# Here is the web page of this float: https://fleetmonitoring.euro-argo.eu/float/7900573
ds_argo = xoa.open_data_sample("argo-7900573.nc")

# %%
# We decode the names to make them more generic thanks to the CF specifications
# and select only the variables of interest.
ds_argo = ds_argo.xoa.decode()[["temp", "sal", "pres"]]

# %%
# We compute depths with the :mod:`gsw` package and assign them
# as coordinates of the ARGO dataset.
alat = ds_argo.lat.broadcast_like(ds_argo.pres)
adepth = gsw.z_from_p(ds_argo.pres, alat)
adepth = adepth.where(adepth.notnull(), adepth.min())
adepth.attrs.update({"long_name": "Depth", "units": "m"})
ds_argo = ds_argo.assign_coords(depth=adepth)
ds_argo = ds_argo.isel(level=slice(None, None, -1))
ds_argo = ds_argo.set_index(N_PROF="time").rename(N_PROF="time")

# %%
# Quick plot of the salinity that highligths the mediterranean water.
fig, axs = plt.subplots(ncols=2, figsize=(14, 6))
ds_argo.sal.plot.contourf("time", "depth", cmap="cmo.haline", levels=20, ax=axs[0])
ds_argo.sal.plot.contour("time", "depth", levels=[35.65], linewidths=1, colors="k", ax=axs[0])
xplot.plot_ts(
    ds_argo.temp,
    ds_argo.sal,
    potential=False,
    scatter_c=ds_argo.depth,
    scatter_cmap="cmo.deep",
    axes=axs[1],
)
axs[1].axvline(x=35.65, ls="--", color="k");

# %%
# Here we select the last ARGO profile as a reference profile
# since it better samples de mediterranean water vein.
ds_argo_prof = ds_argo.isel(time=-1).squeeze(drop=True)

# %%
# Mercator
# ~~~~~~~~
#
# The Mercator data come from the IBI configuration and were downloaded
# from the CMEMS site:
# https://resources.marine.copernicus.eu/product-detail/IBI_ANALYSISFORECAST_PHY_005_001/INFORMATION
# The dataset has been undersampled by a factor 3 in longitude and latitude
# to limit disk usage.
ds_merc = xoa.open_data_sample("ibi-argo-7900573.nc").xoa.decode()
ds_merc = ds_merc.isel(depth=slice(None, None, -1))
ds_merc = ds_merc.assign_coords(depth=-ds_merc.depth)
print(ds_merc)

# %%
# By analogy with the ensemble data assimilation,
# this block of data can be viewed as a ensemble of profiles that emulates
# the uncertainties of the model, so we stack geographical coordinates in a `member` dimension
# thanks to :func:`~xoa.coords.geo_stack`.
ds_merc_ens_rect = xcoords.geo_stack(ds_merc, "member")
print(ds_merc_ens_rect)

# %%
# This is a rectangular selection and we find it more appropriate to retain
# only profiles that are in a given radius of proximity, that is chosen
# close to the surface radius of deformation.
# We compute the distance from each model profile to the selected ARGO profile
# with :func:`~xoa.geo.get_distances`.
radius = 35e3  # 35 km
dist_merc2argo = (
    xgeo.get_distances(ds_merc_ens_rect, ds_argo_prof)
    .squeeze()
    .rename(npts0="member")
    .assign_coords(member=ds_merc_ens_rect.member)
)
ds_merc_ens = ds_merc_ens_rect.where(dist_merc2argo < radius, drop=True)


# %%
# Compare Mercator and ARGO
# -------------------------
#

# %%
# Let's plot the situation.
pmerc = ccrs.Mercator()
pcar = ccrs.PlateCarree()
fig, ax = plt.subplots(subplot_kw={"projection": pmerc}, figsize=(8, 8))
ax.set_extent(xgeo.get_extent(ds_merc, margin=1, square=True))
ax.gridlines(draw_labels=True)
ax.add_wms("https://ows.emodnet-bathymetry.eu/wms", "emodnet:mean_atlas_land", alpha=0.5)
kws = dict(c="C3", s=15, marker="s", transform=pcar)
ax.scatter(ds_merc_ens_rect.lon, ds_merc_ens_rect.lat, label="Rejected", alpha=0.15, **kws)
ax.scatter(ds_merc_ens.lon, ds_merc_ens.lat, label='Mercator', **kws)
ax.scatter(ds_argo_prof.lon, ds_argo_prof.lat, s=100, c="C2", transform=pcar, label='ARGO')
plt.legend();

# %%
# We used the :func:`xoa.geo.get_extent` to compute a square geographical extent based
# on the coordinates of a dataset with an added margin.

# %%
# Now interpolate Mercator onto ARGO time and position for both the ensemble and
# the reference profile. All interpolations can be performed
# with the :meth:`~xarray.Dataset.interp` method since we are dealing with
# axis (1D) coordinates.
ds_merc_ens = ds_merc_ens.interp(time=ds_argo_prof.time.values)
ds_merc_ens_std = ds_merc_ens.std("member")
ds_merc_prof = ds_merc.interp(time=ds_argo_prof.time, lon=ds_argo_prof.lon, lat=ds_argo_prof.lat)

# %%
# We plot now the model, its simulated uncertainty and the observations.
plt.figure(figsize=(4, 7))
plt.fill_betweenx(
    ds_merc_prof.sal.depth,
    ds_merc_prof.sal - 1.96 * ds_merc_ens_std.sal,
    ds_merc_prof.sal + 1.96 * ds_merc_ens_std.sal,
    fc="C1",
    alpha=0.2,
)
ds_merc_prof.sal.plot(y="depth", color="C1", label="Mercator")
plt.plot(ds_merc_prof.sal.values, ds_merc_prof.depth, "o-", color="C1")
ds_argo_prof.sal.plot(y="depth", label="ARGO", color="C0")
plt.title("Uncertain profile")
plt.legend()
xplot.plot_double_minimap(ds_argo_prof, regional_ax="left", global_ax=(0.88, 0.88, 0.11),);

# %%
# Therefore, if we accept an uncertainty in the positioning of the ocean
# structure in the model, it differences with (assimilated) observations
# become acceptables.

# %%
# As you can see on the previous figures, the ARGO profiles come with
# a very high vertical resolution at all depths, whereas the model
# comes with a crude vertical resolution far from the surface.
# Each model thermodynamical model value is representative of the
# cube that defines its XYZ cell.
# This means that despite the model does simulate the small scale variability,
# the cell average may be accurate.
# So we propose here to perform comparisons in the model
# and the observational spaces. We use the function :func:`xoa.regrid.regrid1d`
# that supports the linear (:attr:`~xoa.regrid.regrid1d_methods.linear` method) interpolation
# to higher resolutions and the cell averaging
# (:attr:`~xoa.regrid.regrid1d_methods.cellave` method) to lower resolutions.
sal_merc_ens_o = xregrid.regrid1d(ds_merc_ens.sal, ds_argo_prof.depth, method="linear")
sal_merc_prof_o = xregrid.regrid1d(ds_merc_prof.sal, ds_argo_prof.depth, method="linear")
sal_argo_prof_m = xregrid.regrid1d(ds_argo_prof.sal, ds_merc_prof.depth, method="cellave")
merc_prof_edges = xgrid.get_edges(ds_merc_prof.depth, "depth")

# %%
# Let's plot them.
fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 5))
sal_merc_prof_o.plot(y="depth", color="C1", label="Mercator", ax=axes[0])
ds_argo_prof.sal.plot(y="depth", label="ARGO", color="C0", ax=axes[0])
axes[0].set_title("On ARGO depths")
axis = axes[0].axis()
ds_merc_prof.sal.plot(y="depth", color="C1", label="Mercator", ax=axes[1])
plt.stairs(ds_merc_prof.sal, merc_prof_edges, orientation="horizontal", color="C1", lw=0.5)
sal_argo_prof_m.plot(y="depth", label="ARGO", color="C0", ax=axes[1])
plt.stairs(sal_argo_prof_m, merc_prof_edges, orientation="horizontal", color="C0", lw=0.5)
axes[1].set_title("On Mercator depths")
axes[1].axis(axis);

# %%
# These plots show that the model is too haline, except at the
# mediterranean water depth where the model is very close to the observations
# once an appropriate vertical regridding is performed.
# However, the vertical resolution is not sufficient enough to simulate
# the maximum of salinity.
