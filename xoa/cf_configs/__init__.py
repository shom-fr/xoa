#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal configuration files for the :mod:`xoa.cf` modules.

They define names and attributes specifications for finding and formatting array and datasets.

The list of available cf config files is stored in :data:`CF_CONFIGS`.

.. seealso:: :ref:`appendix.cf.specialized`.


.. autodata:: CF_CONFIGS
"""
import os
import glob


_THIS_DIR = os.path.dirname(__file__)


#: Dictionary that contains the absolute path of internal CF configuration files
CF_CONFIGS = dict(
    (os.path.basename(path[:-4]), path) for path in glob.glob(os.path.join(_THIS_DIR, "*.cfg"))
)


def get_cf_config_file(name):
    """Get the path of a CF config file given its short name"""
    if name.endswith(".cfg"):
        name = name[:-4]
    if name not in CF_CONFIGS:
        from ..__init__ import XoaError

        raise XoaError(
            "fInvalid CF config name '{name}'.\n" + "Please use on of: " + ", ".join(CF_CONFIGS)
        )
    return CF_CONFIGS[name]
