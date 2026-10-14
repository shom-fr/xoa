#!/usr/bin/env python
# coding: utf-8
"""
Write your own naming specifications
====================================

In this tutorial, we show how to:

* declare that your names are known quantities, with a dict or a configuration file,
* use the specifications temporarily or for a given dataset,
* register them so that they are inferred from the names of a dataset,
* assign them to a dataset once and for all.

The use of the specifications is shown in :ref:`sphx_glr_tutorials_meta_plot_naming.py`
and the concepts are explained in the :ref:`indepth.meta` guide.
The configuration files are read with :mod:`xoa.cfgm`, see
:ref:`sphx_glr_tutorials_config_plot_cfgm.py`.
"""

# %%
# Initialisations
# ---------------

import pathlib
import tempfile

import xarray as xr

import xoa
from xoa import meta

xr.set_options(display_style="text")

# %%
# A dataset with names that are not known by default: ``T_surf`` is a temperature and
# ``mylon`` is a longitude.


def make_dataset():
    return xr.Dataset(
        {"T_surf": ("n", [10.0, 11.0, 12.0], {"units": "degC"})},
        coords={"mylon": ("n", [1.0, 2.0, 3.0])},
    )


print(make_dataset().xoa.get("temp"))

# %%
# With a dict
# -----------
#
# Each item of the specifications is a generic name, here ``temp`` and ``lon``, that is declared
# in a section of data variables or of coordinates, with its alternative names.
# The specifications are set for the content of a ``with`` block, and the accessor reads them
# when it is first used on a dataset.

specs = {
    "data_vars": {"temp": {"alt_names": ["T_surf"]}},
    "coords": {"lon": {"alt_names": ["mylon"]}},
}
with meta.set_meta_specs(specs):
    ds = make_dataset()
    print(ds.xoa.get("temp").name, ds.xoa.get("lon").name)

# %%
# With a configuration file
# -------------------------
#
# The same specifications are written in a configuration file, which is better for a model
# that you use often. The ``name`` is the specialized name and ``decode`` renames
# the arrays to the generic names and completes their attributes. The ``register`` section
# gives a name to the specifications, which is needed to recall them.
# We write the file in a temporary directory.

content = """
[register]
name=mymodel

[data_vars]
    [[temp]]
    name=T_surf
    [[sal]]
    name=S_surf

[coords]
    [[lon]]
    name=mylon
"""
path = pathlib.Path(tempfile.mkdtemp()) / "mymodel.cfg"
path.write_text(content)

with meta.set_meta_specs(str(path)):
    ds = make_dataset()
    decoded = ds.xoa.decode()
    print(list(ds.variables), "->", list(decoded.variables))
    print(list(decoded.xoa.encode().variables))

# %%
# Internal specifications, like the ones of CROCO, are files like this one, and
# :func:`xoa.get_meta_config_file` gives their path if you want to start from one of them.
# The other sections are described in :ref:`appendix.meta.sections`.

print(xoa.get_meta_config_file("croco"))

# %%
# Registering them
# ----------------
#
# Registered specifications are recalled by their name and are candidates when
# the ones that match a dataset are looked for, so that ``set_meta_specs`` is not needed
# any more. The ones that know the most names of a dataset are inferred.

meta.register_meta_specs(str(path))
print(meta.is_registered_meta_specs("mymodel"))

ds = make_dataset()
print(meta.infer_meta_specs(ds, named=True).name)
print(ds.xoa.get("temp").name)
print(list(ds.xoa.decode().variables))

# %%
# Assigning them
# --------------
#
# To avoid the inference, assign the specifications to the dataset: this sets
# the ``meta_specs`` encoding of its arrays, that is saved with the file.

meta.assign_meta_specs(ds, "mymodel")
print(ds.T_surf.encoding)
print(meta.get_meta_specs(ds.T_surf).name)

# %%
# Registering changes how the next datasets are decoded. Since the other tutorials share
# this session, we restore the initial state.

meta.reset_cache(memory=True)
print(meta.is_registered_meta_specs("mymodel"))
