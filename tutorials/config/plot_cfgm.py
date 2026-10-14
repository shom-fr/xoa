#!/usr/bin/env python
# coding: utf-8
"""
Manage a configuration with validation and command line options
================================================================

In this tutorial, we show how to:

* describe the options of a tool in specifications that give their types and defaults,
* load and validate a user configuration, with values like lists of floats,
* add your own validation function,
* let the command line options override the user configuration and the defaults,
* export the specifications to the documentation.

This is the :mod:`xoa.cfgm` module, which extends
`configobj <https://configobj.readthedocs.io/en/latest/index.html>`_, and that xoa
uses for its own options and its naming specifications.
The concepts are explained in the :ref:`indepth.cfgm` guide.
See also: :ref:`sphx_glr_tutorials_config_plot_naming_specs.py`.
"""

# %%
# Initialisations
# ---------------

import pathlib
import tempfile
from argparse import ArgumentParser
from pprint import pprint

import numpy as np
from validate import VdtTypeError

from xoa import cfgm

# %%
# The specifications
# ------------------
#
# They are written as lines of a configuration file: the sections are in brackets, and each
# option has a validation function, its parameters and a comment that serves as a help.
# The ``floats`` function is the plural form of ``float``: it validates a list of floats,
# here with ``n=2`` values. In a library, the specifications are typically in a static file
# that sits next to the module.

specs = """
[logger] # logging system
level=option(debug,info,error,default=info) # logging level
name=string(default=root) # logger name
[domain]
xlim=floats(n=2, default=list(-20,0)) # min and max longitudes
ylim=floats(n=2, default=list(40,50)) # min and max latitudes
[plot]
figsize=figsize(default=list(6,4)) # figure size in inches
"""
CFGM = cfgm.ConfigManager(specs.strip().splitlines())
pprint(CFGM.defaults.dict())

# %%
# Load and validate a configuration
# ---------------------------------
#
# The user configuration only holds what differs from the defaults. It is validated,
# which converts the strings, and completed with the defaults.

user_lines = ["[logger]", "name=xoa", "[domain]", "xlim=-10,0"]
cfg = CFGM.load(user_lines)
pprint(cfg.dict())

# %%
# A value that is not acceptable raises an explicit error.

try:
    CFGM.load(["[logger]", "level=verbose"])
except cfgm.XoaValidateError as exc:
    print(exc)

# %%
# Add a validation function
# -------------------------
#
# The available functions are listed with :func:`xoa.cfgm.print_validation_functions`,
# which accepts a pattern.

cfgm.print_validation_functions("*figsize*")

# %%
# A new function is registered with :func:`xoa.cfgm.register_validation_functions`.
# Here it converts an angle to degrees within [0, 360), or to radians.


def is_angle(value, radians=False):
    """Validate an angle with optional conversion to radians"""
    try:
        value = float(value) % 360.0
    except Exception:
        raise VdtTypeError("Invalid angle")
    return float(np.radians(value)) if radians else value


cfgm.register_validation_functions(angle=is_angle)
print(cfgm.get_validator().check("angle(radians=True)", 180 + 360))

# %%
# The command line
# ----------------
#
# The :meth:`~xoa.cfgm.ConfigManager.arg_parse` method adds one option to a parser for each
# option of the specifications, like ``--logger-level``. The command line takes precedence
# over the user configuration, which takes precedence over the defaults. Here the command
# line sets the level, the file sets the name, and the defaults give the rest.

cfgfile = pathlib.Path(tempfile.mkdtemp()) / "config.cfg"
cfgfile.write_text("\n".join(user_lines))

parser = ArgumentParser(description="My script")
parser.add_argument("ncfile", help="netcdf file")
cfg, args = CFGM.arg_parse(
    parser=parser,
    getargs=True,
    args=["--logger-level", "error", "myfile.nc"],
    cfgfile=str(cfgfile),
)
print(args.ncfile)
pprint(cfg.dict())

# %%
# The parser knows the options, grouped by section, and has also a ``--long-help``
# and a ``--short-help`` options in addition to ``--help``.

parser.print_help()

# %%
# Document the options
# --------------------
#
# The specifications are exported as `Sphinx <https://www.sphinx-doc.org>`_ declarations
# of sections and options, to build the documentation of the options of a tool,
# like :doc:`/configure` does for xoa. The :mod:`xoa.cfgm` module is also a Sphinx extension
# that does it at build time, see the :ref:`indepth.cfgm` guide.

print(CFGM.to_rst())
