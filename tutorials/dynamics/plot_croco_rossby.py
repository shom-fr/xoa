#!/usr/bin/env python
"""
Rossby number on the sigma levels of a CROCO simulation
=======================================================

In this tutorial, we show how to:

* find the currents and the depths of the sigma levels of a CROCO simulation,
* compute the Rossby number on levels that follow the topography,
* see why the horizontal derivatives must be taken at constant depth,
* interpolate the Rossby number to a regular depth,
* validate it against the same diagnostic computed from currents that were interpolated
  to regular depths by the model.

The data are 3D averaged outputs of a CROCO simulation of the eastern Mediterranean
at 1/60 degree, south of Crete, where the topography is steep.
The currents are already at the same points as the sea surface height,
so that the diagnostics of :mod:`xoa.dyn` can be applied.

The concepts are explained in the :ref:`indepth.dyn` guide.
See also: :ref:`sphx_glr_tutorials_vertical_plot_croco_section.py`, :ref:`sphx_glr_tutorials_dynamics_plot_gulfstream_dynamics.py`.
"""

# %%
# Initialisations
# ---------------

import cmocean  # noqa
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import xoa
import xoa.coords as xcoords
import xoa.mask as xmask
import xoa.meta as xmeta
import xoa.numbers as xnum
import xoa.stats as xstats
from xoa.plot import add_colorbar, plot_fields, plot_section
from xoa.regrid import regrid1d

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Use the internal CROCO naming specifications.

xmeta.set_meta_specs("croco")

# %%
# Read the simulation
# -------------------
#
# The sample contains the currents and the depth of each sigma level at each horizontal point.
# We also read the same currents that the model interpolated at nine regular depths, which
# will be our reference.

ds = xoa.open_data_sample("MODELS/CROCO/MED/pelops_3d.nc").xoa.decode()
ref = xoa.open_data_sample("MODELS/CROCO/MED/pelops_3d_interp.nc").xoa.decode()
print(ds)

# %%
# The depths vary in space: the levels are thin over the shelf and thick in deep water.
# They are negative, like heights above the sea surface, but the file does not say it with a
# ``positive`` attribute, so we add it. See :ref:`indepth.vertical` for the two conventions.
# The sign does not matter for the diagnostics, but it does for the plots.

ds.depth.attrs["positive"] = "up"
u, v = ds.xoa.get("u"), ds.xoa.get("v")
z = ds.depth
print(u.dims, z.dims)
print(
    f"The first level is at {float(z.isel(sig_rho=0).min()):.0f} to "
    f"{float(z.isel(sig_rho=0).max()):.0f} m"
)

# %%
# Rossby number on the levels
# ---------------------------
#
# The Rossby number is the relative vorticity divided by the Coriolis parameter.
# It is positive for a cyclonic rotation and negative for an anticyclonic one.
# :func:`xoa.numbers.get_rossby` computes it for each level.
# Without the ``z`` parameter, the horizontal derivatives are taken along the levels.

rossby_s = xnum.get_rossby(u, v)

# %%
# Over a slope, the levels are not horizontal and the derivatives along them
# are contaminated by the vertical shear of the currents:
# :math:`\partial_x u|_z = \partial_x u|_s - \partial_x z|_s \, \partial_z u`.
# With the ``z`` parameter, the derivatives are taken at constant depth, by correcting
# the ones along the levels with the slope of the levels and the vertical shear.

rossby_z = xnum.get_rossby(u, v, z=z)

# %%
# Here is a west-east section at 36.7°N, across steep slopes, with the two estimates and
# their difference. They are close where the levels are flat, and they differ along the walls,
# where the levels are tilted and the currents are sheared.

jsec = int(abs(ds.lat_rho.isel(x_rho=0) - 36.7).argmin())
kw = {"method": "contourf", "cmap": "cmo.balance", "extend": "both", "x": "lon"}
fig, axes = plt.subplots(
    1,
    3,
    figsize=(9, 2.5),
    sharey=True,
    constrained_layout=True,
    gridspec_kw={"width_ratios": [1, 1, 1.15]},
)
for ax, da, title in zip(axes[:2], (rossby_s, rossby_z), ("Along the levels", "At constant depth")):
    mappable = plot_section(
        da.isel(y_rho=jsec), ax=ax, add_colorbar=False, levels=np.linspace(-1, 1, 21), **kw
    )
    ax.set_title(title)
add_colorbar(mappable, axes[:2], da=rossby_z)
diff = (rossby_z - rossby_s).rename("rossby")
diff.attrs["long_name"] = "Difference of Rossby number"
plot_section(diff.isel(y_rho=jsec), ax=axes[2], levels=np.linspace(-0.3, 0.3, 21), **kw)
axes[2].set_title("Difference")
for ax in axes:
    ax.set_xlabel("Longitude")
for ax in axes[1:]:
    ax.set_ylabel("")

# %%
# Interpolate to a regular depth
# ------------------------------
#
# To map the Rossby number at a given depth, we interpolate it from the sigma levels with
# :func:`xoa.regrid.regrid1d`, which handles depths that vary in space.
# Let's choose a depth that is in the reference, where the topography is steep.

DEPTH = -100.0
depth = xr.DataArray([DEPTH], dims="depth", name="depth")


def at_depth(da):
    out = regrid1d(
        da.assign_coords(depth=z), depth, dim=("sig_rho", "depth"), coord_in_name="depth"
    )
    # The regridding drops the attributes of the coordinates, that are needed to find the longitudes
    return xcoords.copy_coord_attrs(out.isel(depth=0), da)


rossby_s100 = at_depth(rossby_s).rename("rossby")
rossby_z100 = at_depth(rossby_z).rename("rossby")

# %%
# The reference is the Rossby number of the currents that the model interpolated
# at the same depth, whose derivatives are at constant depth by construction.

rossby_ref = xnum.get_rossby(ref.xoa.get("u").sel(depth=DEPTH), ref.xoa.get("v").sel(depth=DEPTH))

# %%
# Compare them. The land and the points closer than three cells from the edges are not used.

estimates = {
    "Reference": rossby_ref,
    "Along the levels": rossby_s100,
    "At constant depth": rossby_z100,
}
valid = xmask.get_common_mask(*estimates.values(), edge=3)

MAP_KW = {"gridlines_labels_on": ["bottom", "left"], "land": True}
kw = {
    "method": "contourf",
    "cmap": "cmo.balance",
    "levels": np.linspace(-1, 1, 21),
    "extend": "both",
    "map_kw": MAP_KW,
    "colorbar": "shared",
}
fig, axes = plot_fields(
    {name: da.where(valid) for name, da in estimates.items()}, figsize=(9, 3), **kw
)

# %%
# The estimate along the levels is noisy over the slopes, where the levels are the most
# tilted, while the one at constant depth is much closer to the reference.
# Let's see where, with the differences with the reference.

kw.update(
    {
        "levels": np.linspace(-0.3, 0.3, 21),
        "cbar_kwargs": {"label": "Difference with the reference"},
    }
)
estimates.pop("Reference")
fig, axes = plot_fields(
    {name: (da - rossby_ref).where(valid) for name, da in estimates.items()},
    figsize=(6.5, 3.3),
    **kw,
)
print(f"Standard deviation of the reference: {float(rossby_ref.where(valid).std()):.3f}")
for name, da in estimates.items():
    print(
        f"Root mean square error {name.lower()}: {float(xstats.rms(da, rossby_ref, mask=valid)):.3f}"
    )

# %%
# The correction reduces the error, which remains because the vertical interpolation is not
# done at the same stage: the reference was interpolated before the derivatives were computed,
# and we interpolate the Rossby number afterwards.

# %%
# All the diagnostics of :mod:`xoa.dyn` that need horizontal derivatives accept the ``z``
# parameter, like the divergence, the strains and the Okubo-Weiss parameter, and they
# are available as methods of the ``xoa`` accessor of a dataset that contains
# the currents: ``ds.xoa.get_rossby(z=ds.depth)``.
plt.show()
