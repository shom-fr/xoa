#!/usr/bin/env python3
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
import copy
import os
import platform
import re
import warnings
from importlib.metadata import version as im_get_version

from . import exceptions

_RE_OPTION_MATCH = re.compile(r"^(\w+)\W(\w+)$").match
_RE_SECTION_MATCH = re.compile(r"^\w+$").match

#: Specifications of configuration options
CONFIG_SPECS = """
[meta] # meta module
cache=boolean(default=False) # use the :mod:`~xoa.meta` in memory cache

[plot] # plot parameters
cmapdiv = string(default="cmo.balance") # default diverging colormap
cmappos = string(default="cmo.amp")     # default positive colormap
cmapneg = string(default="cmo.tempo_r") # default negative colormap
cmapcyc = string(default="cmo.phase")   # default cyclic colormap
field_method = string(default="pcolormesh") # default method to plot fields
section_method = string(default="pcolormesh") # default method to plot sections
colorbar_shrink = float(default=0.7) # shrink factor of colorbars
mesh_min_spacing = float(default=12) # minimal spacing in pixels between mesh centers in auto stride mode
vectors_min_spacing = float(default=16) # minimal spacing in pixels between vectors in auto stride mode
key_pos = float_list(min=2, max=2, default=list(0.0, 1.03)) # position of the reference arrow in axes coordinates
key_percentile = float(default=80) # percentile of the speed used to choose the reference arrow value
stick_color = string(default="steelblue") # color of sticks

map_figsize = float_list(min=2, max=2, default=list(8, 8)) # figure size in inches
map_gridlines = boolean(default=True) # add labelled gridlines
map_gridlines_labels_on = string_list(default=list("bottom", "left")) # sides of the gridline labels
map_gridlines_linewidth = float(default=0.3) # line width of the gridlines
map_gridlines_linestyle = string(default="--") # line style of the gridlines
map_gridlines_color = string(default="gray") # color of the gridlines
map_gridlines_labelsize = float(default=7) # font size of the gridline labels
map_land = boolean(default=False) # add the land
map_land_scale = string(default="110m") # Natural Earth scale of the land
map_land_color = string(default=None) # face color of the land, cartopy default if not set
map_coastlines = boolean(default=False) # add the coastlines
map_bbox = boolean(default=True) # keep the frame of the axes visible
map_fancy = boolean(default=False) # add a fancy black and white frame
map_layer = boolean(default=False) # add the background layer

layer = string(default="natgeo") # name of the default background layer, see the layers section
layer_alpha = float(default=1.0) # opacity of the layer
layer_zorder = float(default=0) # drawing order
layer_attribution = boolean(default=True) # draw the attribution of the layer
layer_attribution_size = float(default=5) # font size of the attribution
layer_attribution_color = string(default="0.4") # color of the attribution

scalebar_units = option("m", "km", "nmi", default="km") # units of the length
scalebar_loc = string(default="lower left") # corner of the axes
scalebar_pad = float(default=0.04) # distance to the corner in axes coordinates
scalebar_segments = integer(min=1, default=2) # number of alternating segments
scalebar_height = float(default=0.012) # height in axes coordinates
scalebar_color = string(default="k") # color of the dark segments, labels and frame
scalebar_zorder = float(default=10) # drawing order

compass_loc = string(default="upper right") # corner of the axes
compass_size = float(default=40) # height in points
compass_style = option("simple", "fancy", default="simple") # style of the compass
compass_color = string(default="k") # color of the dark parts
compass_zorder = float(default=10) # drawing order

fancy_width = float(default=5.0) # width in points
fancy_colors = string_list(min=2, default=list("k", "w")) # alternating colors
fancy_linewidth = float(default=0.6) # line width of the edges
fancy_edgecolor = string(default="k") # color of the edges
fancy_zorder = float(default=5) # drawing order

flow_particles = integer(min=1, default=2000) # number of particles
flow_alpha = float_list(min=1, max=2, default=list(0.2, 1)) # opacity, or its ramp along the tracks
flow_linewidth = float(default=0.3) # line width of the tracks
flow_color = string(default="k") # color of the tracks

minimap_loc = string(default="upper right") # location in the parent axes
minimap_size = float(default=0.2) # size relative to the parent axes
minimap_extent = float(default=1.0) # extent margin
minimap_min_extent = float(default=2.0) # minimal extent in degrees
minimap_gridlines = boolean(default=True) # add gridlines
minimap_coords_markersize = float(default=10) # marker size of the point coordinates
minimap_coords_linewidth = float(default=1.5) # line width of the coordinates
minimap_coords_color = string(default="tab:red") # color of the coordinates
minimap_coords_facecolor = float(default=0.2) # opacity of the coordinates area

shadow_width = float(default=3) # shadow blur width in points
shadow_xoffset = float(default=2) # shadow horizontal offset in points
shadow_yoffset = float(default=-2) # shadow vertical offset in points
shadow_alpha = float(default=0.5) # shadow opacity
shadow_color = string(default="k") # shadow color
glow_width = float(default=3) # glow width in points
glow_alpha = float(default=1.0) # glow opacity
glow_color = string(default="w") # glow color
lightshading_width = float(default=7) # light shading width in points
lightshading_fraction = float(default=1.0) # light shading fraction

panel_style = string(default="{a})") # label format
panel_loc = string(default="upper left") # label location
panel_pad = float(default=0.03) # distance to the axes corner

[layers] # background layers of maps, one subsection per layer
    [[__many__]]
    kind = option("wms", "wmts", default="wmts") # type of web map service
    url = string() # url of the service, or of its capabilities document for wmts
    layer = string() # name of the layer in the service
    attribution = string(default=None) # credit drawn on the map, quoted if it has commas
"""

#: Default layers, which can be overridden or completed in the user config
CONFIG_DEFAULTS = """
[layers]
    [[natgeo]]
    kind = wmts
    url = https://services.arcgisonline.com/arcgis/rest/services/NatGeo_World_Map/MapServer/WMTS/1.0.0/WMTSCapabilities.xml
    layer = NatGeo_World_Map
    attribution = "Esri, National Geographic"
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


def _get_specs_():
    import configobj

    from . import _XOA_CACHE

    if "cfgspecs" not in _XOA_CACHE:
        _XOA_CACHE["cfgspecs"] = configobj.ConfigObj(
            CONFIG_SPECS.split("\n"),
            list_values=False,
            interpolation=False,
            raise_errors=True,
            file_error=True,
        )
    return _XOA_CACHE["cfgspecs"]


def _check_options_(options):
    """Check that options are known and valid"""
    import configobj
    import validate

    specs = _get_specs_()
    for section in options.sections:
        if section not in specs:
            raise exceptions.XoaConfigError(f"Invalid section: {section}")
        for option in options[section].scalars:
            if option not in specs[section]:
                raise exceptions.XoaConfigError(f"Invalid option: {section}.{option}")
    results = options.validate(validate.Validator(), copy=True, preserve_errors=True)
    if results is not True:
        errors = [
            f"{'.'.join(sections + [key])}: {err if err is not False else 'missing'}"
            for sections, key, err in configobj.flatten_errors(options, results)
        ]
        raise exceptions.XoaConfigError("Invalid option value: " + "; ".join(errors))


def load_options(cfgfile=None):
    """Load specified options and keep them for the session

    The options are merged into the current ones.
    Use :class:`set_options` to change them temporarily.

    Parameters
    ----------
    cfgfile: str, :class:`os.PathLike`, list(str), dict
        Path of a config file, lines or a dict

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

        # File
        load_options("/path/to/my/xoa.cfg")
    """
    import configobj

    from . import _XOA_CACHE

    if "options" not in _XOA_CACHE:
        default_user_config_file = get_default_user_config_file()
        options = configobj.ConfigObj(
            CONFIG_DEFAULTS.split("\n"),
            configspec=_get_specs_(),
            raise_errors=True,
            list_values=True,
        )
        if os.path.exists(default_user_config_file):
            options.merge(
                configobj.ConfigObj(default_user_config_file, raise_errors=True, list_values=True)
            )
    else:
        options = copy.deepcopy(_XOA_CACHE["options"])
    if cfgfile:
        if isinstance(cfgfile, os.PathLike):
            cfgfile = os.fspath(cfgfile)
        options.merge(
            configobj.ConfigObj(cfgfile, file_error=True, raise_errors=True, list_values=True)
        )
    _check_options_(options)
    _XOA_CACHE["options"] = options


def get_options():
    """Get all the current options

    Return
    ------
    configobj.ConfigObj
        The validated options, with a section per option section.
        It is the live object that :func:`get_option` reads, so do not modify it:
        use :func:`set_options` or :func:`load_options` instead.

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa import get_options
        list(get_options()["layers"])
    """
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
    options = get_options()
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


def resolve_option(value, option):
    """Get the value of an option if a value is None, and lists as tuples"""
    if value is None:
        value = get_option(option)
        if isinstance(value, list):
            value = tuple(value)
    return value


class set_options:
    """Set configuration options

    Parameters
    ----------
    section: str, :class:`os.PathLike`, None
        Section name, or path of a config file to load, which is recognised
        when it is not a plain section name.
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

        # From a file, temporary
        with set_options("/path/to/my/xoa.cfg"):
            pass
    """

    def __init__(self, section=None, **options):
        from . import _XOA_CACHE

        # Format before being ingested
        cfgfile = None
        if section is not None and (
            isinstance(section, os.PathLike) or not _RE_SECTION_MATCH(section)
        ):
            cfgfile, section = section, None
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
        self.xoa_cache = _XOA_CACHE
        self.old_options = get_options()
        try:
            if cfgfile:
                load_options(cfgfile)
            load_options(opts)
        except Exception:
            self.xoa_cache["options"] = self.old_options
            raise

    def __enter__(self):
        return self.xoa_cache["options"]

    def __exit__(self, type, value, traceback):
        self.xoa_cache["options"] = self.old_options


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

    _XOA_CACHE.pop("options", None)


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
        print("\n".join(get_options().write()).strip("\n").replace("#", " #"))


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
            version = "NOT INSTALLED or UNKNOWN"
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
