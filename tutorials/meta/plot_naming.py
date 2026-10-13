#!/usr/bin/env python
# coding: utf-8
"""
Find variables whatever their names
===================================

In this tutorial, we show how to:

* access the same quantity in datasets that name it differently,
* write a function once and apply it to several models,
* decode datasets to generic names and encode them back,
* use the naming specifications of a model, or write your own.

The concepts are explained in the :ref:`indepth.meta` guide.
See also: :ref:`sphx_glr_tutorials_graphics_plot_maps_vectors.py`.
"""

# %%
# Initialisations
# ---------------

import xarray as xr
import cmocean

import xoa
from xoa import meta
from xoa.coords import get_tdim, get_zdim
from xoa.plot import plot_fields

xr.set_options(display_style="text")

paths = {
    "GLORYS": "MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc",
    "IBI": "MODELS/CMEMS-IBI/ibi-argo-7900573.nc",
    "HYCOM": "MODELS/HYCOM/hycom.gdp.u.nc",
    "CROCO": "MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.surf.nc",
}
datasets = {name: xoa.open_data_sample(path) for name, path in paths.items()}

# %%
# Same quantity, different names
# ------------------------------
#
# Each model has its own names for the variables and the coordinates.

for name, ds in datasets.items():
    print(f"{name:7}", list(ds.data_vars)[:6], list(ds.coords)[:4])

# %%
# The ``xoa`` accessor searches a dataset for a quantity from its generic name, like
# ``"u"`` or ``"lon"``, by scanning the names and the attributes of the arrays.

for name, ds in datasets.items():
    print(f"{name:7} u -> {ds.xoa.get('u').name:3} sal -> {getattr(ds.xoa.get('sal'), 'name', None)}")

# %%
# A function that works with any model
# ------------------------------------
#
# Since the search relies on the generic names, a function does not need to know
# which model it is applied to. Here we map the surface speed of the two CMEMS models.


def get_surface_speed(ds):
    u, v = ds.xoa.get("u"), ds.xoa.get("v")
    speed = (u**2 + v**2) ** 0.5
    speed = speed.isel({get_zdim(speed): 0, get_tdim(speed): 0}, drop=True)
    speed.attrs.update(long_name="Surface current speed", units="m s-1")
    return speed


speeds = {name: get_surface_speed(datasets[name]) for name in ("GLORYS", "IBI")}
fig, axes = plot_fields(
    speeds, ncols=2, figsize=(9, 4), cmap=cmocean.cm.speed, map_kw={"land": True}
)

# %%
# Naming specifications
# ---------------------
#
# The specifications that are used are inferred from the dataset or set by the user.
# Some come with xoa for a few models, like Mercator for the CMEMS products,
# which know their own names. ``decode`` renames the arrays to their generic names
# and completes their attributes, and ``encode`` gives back the specialized names.
# The accessor reads the specifications when it is first used on a dataset,
# so we open the dataset within the context.

with meta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample(paths["GLORYS"])
    decoded = ds.xoa.decode()
    print(list(ds.data_vars), "->", list(decoded.data_vars))
    print(list(decoded.xoa.encode().data_vars))

# %%
# The attributes are completed with the conventional ones.

print(decoded.temp.attrs["standard_name"], "|", decoded.temp.attrs["long_name"])

# %%
# The names of CROCO are explicit about the grid location: ``u`` lives on the ``u`` points,
# with its own longitudes and latitudes. They are known by the ``croco`` specifications.

with meta.set_meta_specs("croco"):
    croco = xoa.open_data_sample(paths["CROCO"])
    print(croco.xoa.get_meta_specs().name)
    print(croco.xoa.get("u").dims, croco.xoa.get("v").dims)
    print(list(croco.xoa.decode().data_vars)[-6:])

# %%
# Writing your own specifications
# -------------------------------
#
# Specifications are written as a dict or a configuration file. We declare here
# that ``T_surf`` is a temperature and that ``mylon`` is a longitude, which
# the default specifications do not know.


def make_dataset():
    return xr.Dataset(
        {"T_surf": ("n", [10.0, 11.0, 12.0], {"units": "degC"})},
        coords={"mylon": ("n", [1.0, 2.0, 3.0])},
    )


specs = {
    "data_vars": {"temp": {"alt_names": ["T_surf"]}},
    "coords": {"lon": {"alt_names": ["mylon"]}},
}
print(make_dataset().xoa.get("temp"))
with meta.set_meta_specs(specs):
    custom = make_dataset()
    print(custom.xoa.get("temp").name, custom.xoa.get("lon").name)
    print(list(custom.xoa.decode().variables))
