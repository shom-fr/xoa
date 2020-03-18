#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xarray-base Ocean Analysis library

The successor of Vacumm.
"""
# Copyright or © or Copr. Shom/Ifremer/Actimar
#
# stephane.raynaud@shom.fr, charria@ifremer.fr, wilkins@actimar.fr
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import importlib
import os
import re
import warnings

import appdirs
import configobj
import validate

__project__ = "xoa"
__version__ = "0.1.0"
__release__ = "0"
__date__ = "2020-02-04"
__author__ = "Shom/Ifremer/Actimar"
__email__ = "stephane.raynaud@shom.fr, charria@ifremer.fr, wilkins@actimar.fr"
__copyright__ = "Copyright (c) 2020 Shom/Ifremer/Actimar"
__description__ = __doc__

_RE_OPTION_MATCH = re.compile(r"^(\w+)\W(\w+)$").match

#: Specifications of configuration options
CONFIG_SPECS = """
[accessors] # xarray accessors
cf=boolean(default=True) # automatically load the CF accessors?

[plot] # plot parameters
cmapdiv = string(default="cmo.balance") # defaut diverging colormap
cmappos = string(default="cmo.amp")     # default positive colormap
cmapneg = string(default="cmo.tempo_r") # default negative colormap
cmapcyc = string(default="cmo.phase")   # default cyclic colormap

[cache] # cache systems
cf=boolean(default=True) # use the xoa.cf in memory and file caches
"""

#: Default xoa user configuration file
DEFAULT_USER_CONFIG_FILE = os.path.join(
    appdirs.user_data_dir("xoa"), "xoa.cfg"
)

_REQUIREMENTS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "requirements.txt"
)

_CACHE = {}


class XoaError(Exception):
    pass


class XoaConfigError(XoaError):
    pass


class XoaWarning(UserWarning):
    pass


def xoa_warn(message):
    """Issue a :class:`XoaWarning` warning

    Example
    -------
    .. ipython:: python
        :okwarning:

        @suppress
        import xoa
        xoa.xoa_warn('Be careful!')
    """
    warnings.warn(message, XoaWarning, stacklevel=2)


def load_options(cfgfile=None):
    """Load specified options

    Parameters
    ----------
    cfgfile: file, list(str), dict

    Example
    -------
    .. ipython:: python

        @suppress
        import xoa
        # Dict
        xoa.load_options({'plot': {'cmappos': 'mycmap'}})

        # Lines
        optlines = "[plot]\\n cmappos=mycmap".split('\\n')
        xoa.load_options(optlines)
    """

    if "cfgspecs" not in _CACHE:
        _CACHE["cfgspecs"] = configobj.ConfigObj(
            CONFIG_SPECS.split("\n"),
            list_values=False,
            interpolation=False,
            raise_errors=True,
            file_error=True,
        )
    if "options" not in _CACHE:
        _CACHE["options"] = configobj.ConfigObj(
            (
                DEFAULT_USER_CONFIG_FILE
                if os.path.exists(DEFAULT_USER_CONFIG_FILE)
                else None
            ),
            configspec=_CACHE["cfgspecs"],
            file_error=False,
            raise_errors=True,
            list_values=True,
        )
    if cfgfile:
        _CACHE["options"].merge(
            configobj.ConfigObj(
                cfgfile, file_error=True, raise_errors=True, list_values=True
            )
        )
    _CACHE["options"].validate(validate.Validator(), copy=True)


def _get_options_():
    if "options" not in _CACHE:
        load_options()
    return _CACHE["options"]


def get_option(section, option=None):
    """Get a config option

    Example
    -------
    .. ipython:: python

        @suppress
        import xoa
        print(xoa.get_option('plot', 'cmapdiv'))
        print(xoa.get_option('plot.cmapdiv'))
    """
    options = _get_options_()
    if option is None:
        m = _RE_OPTION_MATCH(section)
        if m:
            section, option = m.groups()
        else:
            raise XoaConfigError(
                "You must provide an option name to get_option"
            )
    try:
        value = options[section][option]
    except Exception:
        return XoaConfigError(f"Invalid section/option: {section}/{option}")
    return value


class set_options(object):
    """Set configuration options

    Parameters
    ----------
    section: str, None
    **options: dict
        If a key is in the format "<section>.<option>", then the section
        is overwritten.


    Example
    -------
    .. ipython:: python

        @suppress
        import xoa

        # Classic: for the session
        xoa.set_options('plot', cmapdiv='cmo.balance', cmappos='cmo.amp')

        # With dict
        opts = {"plot.cmapdiv": "cmo.balance"}
        xoa.set_options(**opts)

        # Context: temporary
        with xoa.set_options('plot', cmapdiv='cmo.delta'):
            print('within context:', xoa.get_option('plot.cmapdiv'))
        print('after context:', xoa.get_option('plot.cmapdiv'))

    """

    def __init__(self, section=None, **options):
        # Fromat before being ingested
        self.old_options = _CACHE.get("options")
        del _CACHE["options"]
        opts = {}
        for option, value in options.items():
            m = _RE_OPTION_MATCH(option)
            if m:
                sec, option = m.groups()
            else:
                if section is None:
                    raise XoaConfigError(
                        "You must specify the section explicitly or "
                        "through the the option name")
                sec = section
            opts.setdefault(sec, {})[option] = value

        # Ingest options
        load_options(opts)

    def __enter__(self):
        return _CACHE["options"]

    def __exit__(self, type, value, traceback):
        if self.old_options:
            _CACHE["options"] = self.old_options
        else:
            del _CACHE["options"]


def set_option(option, value):
    """Set a single option using the flat format, i.e ``section.option``

    Parameters
    ----------
    option: str
        Option name in the ``section.option`` format
    value:
        Value to set

    Example
    -------
    .. ipython:: python

        @suppress
        import xoa
        xoa.set_option('plot.cmapdiv', 'cmo.balance');
    """
    return set_options(None, **{option: value})


def reset_options():
    """Restore options to their default values

    Example
    -------
    .. ipython:: python

        @suppress
        import xoa
        print(xoa.get_option('plot.cmapdiv'))
        xoa.set_options('plot', cmapdiv='mycmap')
        print(xoa.get_option('plot.cmapdiv'))
        xoa.reset_options()
        print(xoa.get_option('plot.cmapdiv'))
    """
    del _CACHE['options']


def show_options(specs=False):
    """Print current xoa configuration

    Parameters
    ----------
    specs: bool
        Print option specifications instead

    Example
    -------
    .. ipython:: python

        @suppress
        import xoa
        xoa.show_options()
        xoa.show_options(specs=True)
    """
    if specs:
        print(CONFIG_SPECS.strip("\n"))
    else:
        print("\n".join(_get_options_().write())
              .strip("\n").replace('#', ' #'))


def _parse_requirements_(reqfile):
    re_match_specs_match = re.compile(r"^(\w+)(\W+.+)?$").match
    reqs = {}
    with open(reqfile) as f:
        for line in f:
            line = line.strip().strip("\n")
            if line and not line.startswith("#"):
                m = re_match_specs_match(line)
                if m:
                    reqs[m.group(1)] = m.group(2)
    return reqs


def show_versions():
    """Print the versions of xoa and of some dependencies

    Example
    -------
    .. ipython:: python
        :okexcept:

        @suppress
        import xoa
        xoa.show_versions()
    """
    print("- xoa:", __version__)
    for package in _parse_requirements_(_REQUIREMENTS_FILE):
        try:
            pp = importlib.import_module(package)
            if hasattr(pp, "__version__"):
                version = pp.__version__
            else:
                version = "UNKNOWN"
        except ImportError:
            version = 'ERROR'
        if hasattr(pp, "__version__"):
            print(f"- {package}: {version}")


def show_paths():
    """Print some xoa paths"""
    print("- xoa library dir:", os.path.dirname(__file__))
    from . import cf
    asterix = False
    for label, path in [("user config file", DEFAULT_USER_CONFIG_FILE),
                        ("user CF specs file", cf.USER_CF_FILE),
                        ("user CF cache file", cf.USER_CF_CACHE_FILE)]:
        if not os.path.exists(path):
            asterix = True
            path = path + " [*]"
        print("-", label+":", path)
    if asterix:
        print("*: file not present")


def show_info(opt_specs=True):
    """Print xoa related info

    Example
    -------
    .. ipython:: python
        :okexcept:

        @suppress
        import xoa
        xoa.show_info()
    """
    print("# VERSIONS")
    show_versions()
    print("\n# FILES AND DIRECTORIES")
    show_paths()
    print("\n# OPTIONS")
    show_options(specs=opt_specs)
