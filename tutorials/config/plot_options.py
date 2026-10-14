#!/usr/bin/env python
# coding: utf-8
"""
Set the options of xoa
======================

In this tutorial, we show how to:

* list the options of xoa and read them,
* let the maps choose their colormap from the sign of the data and from the options,
* change the options for a block of code, for the session or from a file,
* find the user configuration files.

The options are managed with :mod:`xoa.cfgm`, see :ref:`sphx_glr_tutorials_config_plot_cfgm.py`.
The list of the options and the way to set them permanently are in the
:doc:`configuration </configure>` page. The same information is printed in a terminal by
the ``xoa info`` command.
"""

# %%
# Initialisations
# ---------------

import pathlib
import tempfile

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import xoa
import xoa.meta as xmeta
from xoa.plot import plot_fields

xr.set_options(display_style="text")
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# List and read the options
# -------------------------
#
# The options are organised in sections, and each of them has a type and a default value.
# They are printed with their specifications, and read with
# :func:`xoa.get_option`, with the flat ``section.option`` or the ``section, option`` form.

xoa.show_options(specs=True)

# %%

print(xoa.get_option("plot.cmapdiv"), xoa.get_option("plot", "cmappos"))

# %%
# Colormaps that follow the data
# ------------------------------
#
# With ``cmap="auto"``, :func:`xoa.plot.plot_field` and :func:`xoa.plot.plot_fields`
# choose the colormap from the sign of the data: the diverging one of ``plot.cmapdiv``
# with limits that are symmetric around zero when the data have both signs, the
# positive one of ``plot.cmappos`` or the negative one of ``plot.cmapneg`` otherwise.

with xmeta.set_meta_specs("mercator"):
    ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc").xoa.decode()
ds = ds.isel(time=0, depth=0)
fields = {
    "Speed (positive)": np.hypot(ds.u, ds.v).rename("speed").assign_attrs(
        long_name="Speed", units="m s-1"
    ),
    "Sea surface height (both signs)": (ds.ssh - ds.ssh.mean()).assign_attrs(
        long_name="Anomaly", units="m"
    ),
    "Minus the temperature (negative)": (-ds.temp).assign_attrs(
        long_name="Minus temperature", units="degC"
    ),
}
kw = dict(cmap="auto", figsize=(10, 3.4), map_kw={"land": True, "gridlines_labels_on": ["bottom", "left"]})
plot_fields(fields, **kw)

# %%
# Change the options
# ------------------
#
# :class:`xoa.set_options` is a context manager that changes the options within a block.
# Here the diverging colormap is changed to make a different figure.

with xoa.set_options("plot", cmapdiv="cmo.curl", cmappos="cmo.ice"):
    plot_fields(fields, **kw)
print("After the block:", xoa.get_option("plot.cmapdiv"))

# %%
# Used without ``with``, it changes the options for the rest of the session, as does
# :func:`xoa.set_option` for a single option in the flat form. :func:`xoa.reset_options`
# restores the options of the configuration file, or the default ones.

xoa.set_option("plot.cmapdiv", "cmo.delta")
print(xoa.get_option("plot.cmapdiv"))
xoa.reset_options()
print(xoa.get_option("plot.cmapdiv"))

# %%
# From a file
# -----------
#
# The options that you always want are set in your user configuration file, whose location
# depends on your system and is given by :func:`xoa.show_paths`. A file, or the lines
# of a file, can also be loaded for the session with :func:`xoa.load_options`.

cfgfile = pathlib.Path(tempfile.mkdtemp()) / "xoa.cfg"
cfgfile.write_text("[plot]\ncmappos = cmo.speed\n")
xoa.load_options(str(cfgfile))
print(xoa.get_option("plot.cmappos"))

# %%
# The user configuration file and the other paths of xoa are printed by
# :func:`xoa.show_paths`, which is also the ``xoa info paths`` command.

print(xoa.get_default_user_config_file())
xoa.reset_options()
