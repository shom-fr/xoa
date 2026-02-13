#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xoa options
"""
# Copyright 2020-2026 Shom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import os
import warnings
import platform
from importlib.metadata import version as im_get_version

from . import exceptions

_RE_OPTION_MATCH = re.compile(r"^(\w+)\W(\w+)$").match

#: Specifications of configuration options
CONFIG_SPECS = """
[cf] # cf module
cache=boolean(default=False) # use the :mod:`~xoa.cf` in memory and file caches

[plot] # plot parameters
cmapdiv = string(default="cmo.balance") # defaut diverging colormap
cmappos = string(default="cmo.amp")     # default positive colormap
cmapneg = string(default="cmo.tempo_r") # default negative colormap
cmapcyc = string(default="cmo.phase")   # default cyclic colormap

"""

_PACKAGES = [
    "platformdirs",
    "cartopy",
    "cmocean",
    "configobj",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "xarray",
]


def get_default_user_config_file():
    """Get the default user config file name"""
    try:
        from platformdirs import user_config_dir
    except ImportError:
        from appdirs import user_config_dir

        warnings.warn(
            "appdirs is deprecated. Please install platformdirs.",
            warnings.DeprecationWarning,
        )
    return os.path.join(user_config_dir("xoa"), "xoa.cfg")


def load_options(cfgfile=None):
    """Load specified options

    Parameters
    ----------
    cfgfile: file, list(str), dict

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa import load_options
        # Dict
        load_options({'plot': {'cmappos': 'mycmap'}})

        # Lines
        optlines = "[plot]\\n cmappos=mycmap".split('\\n')
        load_options(optlines)
    """
    import configobj
    import validate

    from . import _XOA_CACHE

    if "cfgspecs" not in _XOA_CACHE:
        _XOA_CACHE["cfgspecs"] = configobj.ConfigObj(
            CONFIG_SPECS.split("\n"),
            list_values=False,
            interpolation=False,
            raise_errors=True,
            file_error=True,
        )
    if "options" not in _XOA_CACHE:
        default_user_config_file = get_default_user_config_file()
        _XOA_CACHE["options"] = configobj.ConfigObj(
            (default_user_config_file if os.path.exists(default_user_config_file) else None),
            configspec=_XOA_CACHE["cfgspecs"],
            file_error=False,
            raise_errors=True,
            list_values=True,
        )
    if cfgfile:
        _XOA_CACHE["options"].merge(
            configobj.ConfigObj(cfgfile, file_error=True, raise_errors=True, list_values=True)
        )
    _XOA_CACHE["options"].validate(validate.Validator(), copy=True)


def _get_options_():
    from . import _XOA_CACHE

    if "options" not in _XOA_CACHE:
        load_options()
    return _XOA_CACHE["options"]


def get_option(section, option=None):
    """Get a config option

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa import get_option
        print(get_option('plot', 'cmapdiv'))
        print(get_option('plot.cmapdiv'))
    """
    options = _get_options_()
    if option is None:
        m = _RE_OPTION_MATCH(section)
        if m:
            section, option = m.groups()
        else:
            raise exceptions.XoaConfigError("You must provide an option name to get_option")
    try:
        value = options[section][option]
    except Exception:
        raise exceptions.XoaConfigError(f"Invalid section/option: {section}/{option}")
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
        from xoa import set_options, get_option

        # Classic: for the session
        set_options('plot', cmapdiv='cmo.balance', cmappos='cmo.amp')

        # With dict
        opts = {"plot.cmapdiv": "cmo.balance"}
        set_options(**opts)

        # Context: temporary
        with set_options('plot', cmapdiv='cmo.delta'):
            print('within context:', get_option('plot.cmapdiv'))
        print('after context:', get_option('plot.cmapdiv'))

    """

    def __init__(self, section=None, **options):
        # Format before being ingested
        from . import _XOA_CACHE

        self.xoa_cache = _XOA_CACHE
        self.old_options = self.xoa_cache.get("options")
        if "options" in self.xoa_cache:
            del self.xoa_cache["options"]
        opts = {}
        for option, value in options.items():
            m = _RE_OPTION_MATCH(option)
            if m:
                sec, option = m.groups()
            else:
                if section is None:
                    raise exceptions.XoaConfigError(
                        "You must specify the section explicitly or through the option name"
                    )
                sec = section
            opts.setdefault(sec, {})[option] = value

        # Ingest options
        load_options(opts)

    def __enter__(self):
        return self.xoa_cache["options"]

    def __exit__(self, type, value, traceback):
        if self.old_options:
            self.xoa_cache["options"] = self.old_options
        else:
            del self.xoa_cache["options"]


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
        from xoa import set_option
        set_option('plot.cmapdiv', 'cmo.balance');
    """
    return set_options(None, **{option: value})


def reset_options():
    """Restore options to their default values in the current session

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa import get_option, set_options, reset_options
        print(get_option('plot.cmapdiv'))
        set_options('plot', cmapdiv='mycmap')
        print(get_option('plot.cmapdiv'))
        reset_options()
        print(get_option('plot.cmapdiv'))
    """
    from . import _XOA_CACHE

    del _XOA_CACHE["options"]


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
        from xoa import show_options
        show_options()
        show_options(specs=True)
    """
    if specs:
        print(CONFIG_SPECS.strip("\n"))
    else:
        print("\n".join(_get_options_().write()).strip("\n").replace("#", " #"))


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
        from xoa import show_versions
        show_versions()
    """
    from . import __version__

    print("- python:", platform.python_version())
    print("- xoa:", __version__)
    for package in _PACKAGES:
        try:
            version = im_get_version(package)
        except Exception:
            version = "NOT INSTALLED or UKNOWN"
        print(f"- {package}: {version}")


def show_paths():
    """Print some xoa paths

    Example
    -------
    .. ipython:: python
        :okexcept:

        @suppress
        from xoa import show_paths
        show_paths()
    """
    print("- xoa library dir:", os.path.dirname(__file__))
    from . import meta
    from .data_samples import get_data_sample

    asterix = False
    default_user_config_file = get_default_user_config_file()
    for label, path in [
        ("user config file", default_user_config_file),
        ("user CF specs file", meta.USER_META_FILE),
    ]:
        if not os.path.exists(path):
            asterix = True
            path = path + " [*]"
        print("-", label + ":", path)
    print("- data samples:", " ".join(get_data_sample()))
    if asterix:
        print("*: file not present")


def show_info(opt_specs=True):
    """Print xoa related info

    Example
    -------
    .. ipython:: python
        :okexcept:

        @suppress
        from xoa import show_info
        show_info()
    """
    print("# VERSIONS")
    show_versions()
    print("\n# FILES AND DIRECTORIES")
    show_paths()
    print("\n# OPTIONS")
    show_options(specs=opt_specs)
