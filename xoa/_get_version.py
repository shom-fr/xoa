#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get version
"""


def _get_version():
    __version__ = "unknown"
    try:
        from ._version import __version__
    except ImportError:
        pass
    return __version__
