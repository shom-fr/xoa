#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xarray-based ocean analysis library

The successor of Vacumm.
"""
# Copyright 2020-2021 Shom
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


import os
import re
import warnings
import platform

import pkg_resources
import appdirs
import configobj
import validate


# Taken from xarray
try:
    __version__ = pkg_resources.get_distribution("xoa").version
except Exception:
    # Local copy or not installed with setuptools.
    # Disable minimum version checks on downstream libraries.
    __version__ = "999"

_RE_OPTION_MATCH = re.compile(r"^(\w+)\W(\w+)$").match

#: Specifications of configuration options
CONFIG_SPECS = """
[cf] # cf module
cache=boolean(default=True) # use the :mod:`~xoa.cf` in memory and file caches

[plot] # plot parameters
cmapdiv = string(default="cmo.balance") # defaut diverging colormap
cmappos = string(default="cmo.amp")     # default positive colormap
cmapneg = string(default="cmo.tempo_r") # default negative colormap
cmapcyc = string(default="cmo.phase")   # default cyclic colormap

"""

#: Default xoa user configuration file
DEFAULT_USER_CONFIG_FILE = os.path.join(
    appdirs.user_config_dir("xoa"), "xoa.cfg"
)

# Directory of sample files
_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), '_samples')

_PACKAGES = [
    "appdirs",
    "cartopy",
    "cmocean",
    "configobj",
    "matplotlib",
    "numpy",
    "pandas",
    "scipy",
    "xarray",
    "xesmf"
    ]


class XoaError(Exception):
    pass


class XoaConfigError(XoaError):
    pass


class XoaWarning(UserWarning):
    pass


def xoa_warn(message, stacklevel=2):
    """Issue a :class:`XoaWarning` warning

    Example
    -------
    .. ipython:: python
        :okwarning:

        @suppress
        from xoa import xoa_warn
        xoa_warn('Be careful!')
    """
    warnings.warn(message, XoaWarning, stacklevel=stacklevel)


def _get_cache_():
    from . import __init__
    if not hasattr(__init__, "_XOA_CACHE"):
        __init__._XOA_CACHE = {}
    return  __init__._XOA_CACHE


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
    _get_cache_()
    xoa_cache = _get_cache_()

    if "cfgspecs" not in xoa_cache:
        xoa_cache["cfgspecs"] = configobj.ConfigObj(
            CONFIG_SPECS.split("\n"),
            list_values=False,
            interpolation=False,
            raise_errors=True,
            file_error=True,
        )
    if "options" not in xoa_cache:
        xoa_cache["options"] = configobj.ConfigObj(
            (
                DEFAULT_USER_CONFIG_FILE
                if os.path.exists(DEFAULT_USER_CONFIG_FILE)
                else None
            ),
            configspec=xoa_cache["cfgspecs"],
            file_error=False,
            raise_errors=True,
            list_values=True,
        )
    if cfgfile:
        xoa_cache["options"].merge(
            configobj.ConfigObj(
                cfgfile, file_error=True, raise_errors=True, list_values=True
            )
        )
    xoa_cache["options"].validate(validate.Validator(), copy=True)


def _get_options_():
    xoa_cache = _get_cache_()
    if "options" not in xoa_cache:
        load_options()
    return xoa_cache["options"]


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
        self.xoa_cache = _get_cache_()
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
                    raise XoaConfigError(
                        "You must specify the section explicitly or through the option name")
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
    xoa_cache = _get_cache_()
    del xoa_cache['options']


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
        from xoa import show_versions
        show_versions()
    """
    print("- python:", platform.python_version())
    print("- xoa:", __version__)
    for package in _PACKAGES:
        try:
            version = pkg_resources.get_distribution(package).version
        except pkg_resources.DistributionNotFound:
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
    from . import cf
    asterix = False
    for label, path in [("user config file", DEFAULT_USER_CONFIG_FILE),
                        ("user CF specs file", cf.USER_CF_FILE),
                        ("user CF cache file", cf.USER_CF_CACHE_FILE)]:
        if not os.path.exists(path):
            asterix = True
            path = path + " [*]"
        print("-", label+":", path)
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


def get_data_sample(filename=None):
    """Get the absolute path to a sample file

    Parameters
    ----------
    filename: str, None
        Name of the sample. If ommited, a list of available samples
        name is returned.

    Returns
    -------
    str OR list(str)

    Example
    -------
    .. .ipython:: python

        @suppress
        from xoa import get_data_sample
        get_data_sample("croco.south-africa.surf.nc")
        get_data_sample()

    See also
    --------
    show_data_samples
    open_data_sample
    """
    if not os.path.exists(_SAMPLE_DIR):
        filenames = []
    else:
        filenames = os.listdir(_SAMPLE_DIR)
    if filename is None:
        return filenames
    if filename not in filenames:
        raise XoaError("Invalid data sample: "+filename)
    return os.path.join(_SAMPLE_DIR, filename)


def open_data_sample(filename, **kwargs):
    """Open a data sample with :func:`xarray.open_dataset` or :func:`pandas.read_csv`

    A shortcut to::

        xr.open_dataset(get_data_sample(filename))

    Parameters
    ----------
    filename: str
        File name of the sample

    Returns
    -------
    xarray.Dataset, pandas.DataFrame

    Example
    -------
    .. .ipython:: python

        @suppress
        from xoa import open_data_sample
        open_data_sample("croco.south-africa.nc")


    See also
    --------
    get_data_sample
    show_data_samples
    """
    fname = get_data_sample(filename)
    if fname.endswith("nc"):
        import xarray as xr
        return xr.open_dataset(fname, **kwargs)
    import pandas as pd
    return pd.read_csv(fname, **kwargs)


def show_data_samples():
    """Print the list of data samples

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa import show_data_samples
        show_data_samples()

    See also
    --------
    get_data_samples
    open_data_sample
    """
    print(' '.join(get_data_sample()))


def register_accessors(xoa=True, xcf=False, decode_sigma=False):
    """Register xarray accessors

    Parameters
    ----------
    xoa: bool, str
        Register the main accessors with
        :func:`~xoa.cf.register_xoa_accessors`.
    xcf: bool, str
        Register the :mod:`xoa.cf` module accessors with
        :func:`~xoa.cf.register_cf_accessors`.
    decode_sigma: bool, str
        Register the :mod:`xoa.sigma` module accessor with
        :func:`~xoa.cf.register_sigma_accessor`.

    See also
    --------
    xoa.accessors
    """
    if xoa:
        from .accessors import register_xoa_accessors
        kw = {"name": xoa} if isinstance(xoa, str) else {}
        register_xoa_accessors(**kw)
    if xcf:
        from .accessors import register_cf_accessors
        kw = {"name": xcf} if isinstance(xcf, str) else {}
        register_cf_accessors(**kw)
    if decode_sigma:
        from .accessors import register_sigma_accessor
        kw = {"name": decode_sigma} if isinstance(decode_sigma, str) else {}
        register_sigma_accessor(**kw)
