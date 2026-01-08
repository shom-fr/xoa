#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Internal configuration files for the :mod:`xoa.meta` module.

They define names and attributes specifications for finding and formatting array and datasets.

The list of available cf config files is stored in :data:`META_CONFIGS`.

.. seealso:: :ref:`appendix.meta.specialized`.


.. autodata:: META_CONFIGS
"""
import os
import glob


_THIS_DIR = os.path.dirname(__file__)


#: Dictionary that contains the absolute path of internal CF configuration files
META_CONFIGS = dict(
    (os.path.basename(path[:-4]), path) for path in glob.glob(os.path.join(_THIS_DIR, "*.cfg"))
)
