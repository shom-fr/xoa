#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naming convention tools for reading and formatting variables

.. rubric:: How to use it

See the :ref:`uses.cf` section.

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
import pickle
import re
import operator
import pprint
import fnmatch
import copy

import appdirs

from .__init__ import XoaError, xoa_warn, get_option
from .misc import dict_merge, match_string, ERRORS, Choices

_THISDIR = os.path.dirname(__file__)

# Joint variables and coords config specification file
_INIFILE = os.path.join(_THISDIR, "cf.ini")

# Base config file for CF specifications
_CFGFILE = os.path.join(_THISDIR, "cf.cfg")

_user_cache_dir = appdirs.user_cache_dir("xoa")

#: User cache file for cf specs
USER_CF_CACHE_FILE = os.path.join(_user_cache_dir, "cf.pyk")

#: User CF config file
USER_CF_FILE = os.path.join(appdirs.user_config_dir("xoa"), "cf.cfg")

# Argument passed to dict_merge to merge CF configs
_CF_DICT_MERGE_KWARGS = dict(
    mergesubdicts=True,
    mergelists=True,
    skipnones=False,
    skipempty=False,
    overwriteempty=True,
    mergetuples=True,
    unique=True
)

ATTRS_PATCH_MODE = Choices(
    {'fill': 'do not erase existing attributes, just fill missing ones',
     'replace': 'replace existing attributes'},
    parameter="mode",
    description="policy for patching existing attributes"
    )


class XoaCFError(XoaError):
    pass


def _get_cache_():
    from . import cf
    if not hasattr(cf, "_CF_CACHE"):
        cf._CF_CACHE = {
            "current": None,     # current active specs
            "default": None,     # default xoa specs
            "loaded_dicts": {},  # for pure caching of dicts by key
            "registered": []     # for registration and matching purpose
        }
    return  cf._CF_CACHE


def _compile_sg_match_(re_match, attrs, formats, root_patterns,
                       location_pattern):
    for attr in attrs:
        root_pattern = root_patterns[attr]
        re_match[attr] = re.compile(
            formats[attr].format(
                root=rf"(?P<root>{root_pattern})",
                loc=rf"(?P<loc>{location_pattern})"
            ),
            re.I,
        ).match


class SGLocator(object):
    """Staggered grid location parsing and formatting utility

    Parameters
    ----------
    name_format: str
        A string containing the string patterns ``{root}`` and ``{loc}``,
        which defaults to ``"{root}_{loc}"``
    valid_locations: list(str), None
        Valid location strings that are used when parsing
    """
    valid_attrs = ("name", "standard_name", "long_name")

    formats = {
        "name": "{root}_{loc}",
        "standard_name": "{root}_at_{loc}_location",
        "long_name": "{root} at {loc} location",
    }

    root_patterns = {
        "name": r"\w+",
        "standard_name": r"\w+",
        "long_name": r"[\w ]+"
        }

    location_pattern = "[a-z]+"

    re_match = {}
    _compile_sg_match_(re_match, valid_attrs, formats, root_patterns,
                       location_pattern)

    def __init__(self, name_format=None, valid_locations=None):

        # Init
        self.formats = self.formats.copy()
        self.re_match = self.re_match.copy()
        self._name_format = name_format
        if valid_locations:
            valid_locations = list(valid_locations)
        self.valid_locations = valid_locations

        # Formats and regexps
        to_recompile = set()
        if name_format:
            for pat in ("{root}", "{loc}"):
                if pat not in name_format:
                    raise XoaCFError("name_format must contain strings "
                                     "{root} and {loc}: "+name_format)
            if len(name_format) == 10:
                xoa_warn('No separator found in "name_format" and '
                         'and no "valid_locations" specified: '
                         f'{name_format}. This leads to ambiguity during '
                         'regular expression parsing.')
            self.formats["name"] = name_format
            to_recompile.add("name")
        if self.valid_locations:
            self.location_pattern = "|".join(self.valid_locations)
            to_recompile.update(("name", "standard_name", "long_name"))
        _compile_sg_match_(self.re_match, to_recompile, self.formats,
                           self.root_patterns, self.location_pattern)

    def parse_attr(self, attr, value):
        """Parse an attribute string to get its root and location

        Parameters
        ----------
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        value: str
            Attribute value

        Return
        ------
        str
            Root
        str, None
            Lower case location

        Example
        -------
        .. ipython:: python

            @suppress
            from xoa.cf import SGLocator
            sg = SGLocator(name_format="{root}_{loc}")
            sg.parse_attr("name", "super_banana_t")
            sg.parse_attr("standard_name", "super_banana_at_rhum_location")
            sg.parse_attr("standard_name", "super_banana_at_rhum_place")

            sg = SGLocator(valid_locations=["u", "rho"])
            sg.parse_attr("name", "super_banana_t")
            sg.parse_attr("name", "super_banana_rho")
        """
        m = self.re_match[attr](value)
        if m is None:
            return value, None
        return m.group("root"), m.group("loc").lower()

    def get_location(self, da):
        """Parse a data array name and attributes to find location

        Parameters
        ----------
        da: xarray.DataArray
            Data array to scan

        Return
        ------
        None, str
            ``None`` if no location is found, else the corresponding string

        Raises
        ------
        XoaCFError
            When locations parsed from name and attributes are conflicting.
            Thus, this method method is a way to check location consistency.
        """
        loc = None
        src = []

        # Name
        if da.name:
            loc = self.parse_attr("name", da.name)[1]
            src.append("name")

        # Standard name
        if "standard_name" in da.attrs:
            loc_ = self.parse_attr("standard_name", da.standard_name)[1]
            if loc_:
                if loc and loc_ != loc:
                    raise XoaCFError(
                        "The location parsed from standard_name attribute "
                        f"[{loc_}] conflicts the location parsed from the "
                        f"name [{loc}]")
                else:
                    loc = loc_
                    src.append("standard_name")

        # Long name
        if "long_name" in da.attrs:
            loc_ = self.parse_attr("long_name", da.long_name)[1]
            if loc_:
                if loc and loc_ != loc:
                    src = ' and '.join(src)
                    raise XoaCFError(
                        "The location parsed from long_name attribute "
                        f"[{loc_}] conflicts the location parsed from the "
                        f"{src} [{loc}]")
                else:
                    loc = loc_

        return loc

    def match_attr(self, attr, value, root, loc=None):
        """Check if an attribute is matching a location

        Parameters
        ----------
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        root: str
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location

        Return
        ------
        bool or loc
        """
        if attr not in self.valid_attrs:
            return
        value = value.lower()
        vroot, vloc = self.parse_attr(attr, value)
        root = root.lower()
        if vroot.lower() != root:
            return False
        if loc is None or loc == "any":  # any loc
            return True
        if not loc:  # not loc explicit
            return vloc is None
        return vloc in loc  # one of these locs

    def format_attr(self, attr, value, loc, standardize=True):
        """Format a single attribute at a specified location

        Parameters
        ----------
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        value: str
            Current attribute value. It is parsed to get current ``root``.
        loc: {True, None}, str, {False, ""}
            If None, location is left unchanged;
            if a str, it is set;
            else, it is removed.
        standardize: bool
            If True, standardize ``root`` and ``loc`` values.

        Return
        ------
        str

        Example
        -------
        .. ipython:: python

            @suppress
            from xoa.cf import SGLocator
            sg = SGLocator()
            sg.format_attr('standard_name', 'sea_water_temperature', 't')
            sg.format_attr('standard_name', 'sea_water_temperature', False)
            sg.format_attr('name', 'banana_t', None)

        """
        if value is None:
            return value
        if attr not in self.valid_attrs:
            return value
        root, ploc = self.parse_attr(attr, value)
        if standardize:
            if attr == "long_name":
                root = root.capitalize().replace("_", " ")
            else:
                root = root.replace(" ", "_")
                if attr == "standard_name":
                    root = root.lower()
        if (
            loc
            and self.valid_locations
            and loc.lower() not in self.valid_locations
        ):
            raise XoaCFError(
                "Invalid location: {}. Please one use of: {}.".format(
                    loc, ', '.join(self.valid_locations))
            )
        elif loc is False or loc == '':
            ploc = None
        elif loc is True or loc is None:
            loc = ploc
        loc = loc or ploc
        if not loc:
            return root
        if standardize:
            if attr == "long_name":
                loc = loc.upper()
            elif attr == "standard_name":
                loc = loc.lower()
        return self.formats[attr].format(root=root, loc=loc)

    def format_attrs(self, attrs, loc=None, standardize=True):
        """Copy and format a dict of attributes

        Parameters
        ----------
        attrs: dict
        loc: {True, None}, letter, {False, ""}
            If None, location is left unchanged;
            if a letter, it is set;
            else, it is removed.
        standardize: bool
            If True, standardize ``root`` and ``loc`` values.

        Return
        ------
        dict

        Example
        -------
        .. ipython:: python

            @suppress
            from xoa.cf import SGLocator
            sg = SGLocator()
            attrs = dict(standard_name='sea_water_temperature_at_t_location',
                         long_name='sea_water_temperature',
                         other_attr=23.)
            sg.format_attrs(attrs, loc='t') # force t loc
            sg.format_attrs(attrs) # keep loc
            sg.format_attrs(attrs, loc=False) # force no loc
        """
        attrs = attrs.copy()
        for attr, value in attrs.items():
            if attr in self.valid_attrs and attr != 'name':
                if isinstance(value, list):
                    value = [self.format_attr(
                        attr, v, loc, standardize=standardize) for v in value]
                else:
                    value = self.format_attr(
                        attr, value, loc, standardize=standardize)
                attrs[attr] = value
        return attrs

    def merge_attr(self, attr, value0, value1, loc=None, standardize=True):
        """Merge two attribute values taking care of location

        Parameters
        ----------
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        value0: str, None
            First attribute value
        value1: str, None
            Second attribute value, which is prioritary over the first one
        loc: letters, {"any", None} or {"", False}
            - letters: one of these locations
            - None or "any": any
            - False or '': no location

        Returns
        -------
        str, None

        Example
        -------
        .. ipython:: python

            @suppress
            from xoa.cf import SGLocator
            sg = SGLocator()
            sg.merge_attr('name', 'temp_t', 'mytemp')
            sg.merge_attr('name', 'temp', 'mytemp_t')
            sg.merge_attr('name', 'temp_u', 'mytemp_t', 'v')
        """
        assert attr in self.valid_attrs
        if value0 is None and value1 is None:
            return
        if value0 is None:
            return self.format_attr(attr, value1, loc, standardize=standardize)
        if value1 is None:
            return self.format_attr(attr, value0, loc, standardize=standardize)

        if loc in (None, "any"):
            loc0 = self.parse_attr(attr, value0)[1]
            loc1 = self.parse_attr(attr, value1)[1]
            if loc1 is not None:
                loc = loc1
            elif loc0 is not None:
                loc = loc0
        return self.format_attr(attr, value1, loc, standardize=standardize)

    def patch_attrs(self, attrs, patch, loc=None, standardize=True,
                    replace=False):
        """Patch a dict of attribute with another dict taking care of loc

        Parameters
        ----------
        attrs: dict
            Dictionary of attributes to patch
        patch: dict
            Patch to apply
        loc: {None, "any"}, letter, {False, ""}
            If None, location is left unchanged;
            if a letter, it is set;
            else, it is removed.
        standardize: bool
            If True, standardize ``root`` and ``loc`` values.
        replace: bool
            Replace existing attributes?

        Returns
        -------
        dict
            A new dictionary of attributes

        See also
        --------
        merge_attr
        """
        # Reloc if needed
        attrs = attrs.copy()
        if attrs:
            attrs.update(self.format_attrs(
                attrs, loc, standardize=standardize))

        # Loop patching on attributes
        for attr, value in patch.items():

            # Skip
            if value is None or (attr in attrs and not replace):
                continue

            # Attr with loc
            if attr in self.valid_attrs:

                # List
                if isinstance(value, list):
                    if attr in attrs:
                        for val in value:
                            if self.match_attr(
                                    attr, attrs[attr], val, loc=None):
                                value = attrs[attr]  # don't change, it's ok
                                break
                        else:
                            value = value[0]
                    else:
                        value = value[0]

                # Location
                value = self.merge_attr(
                    attr, attrs.get(attr, None), value, loc,
                    standardize=standardize)

            if value is not None:
                attrs[attr] = value

        return attrs

    def format_dataarray(
        self, da, loc=None, standardize=True, name=None, attrs=None,
        rename=True, copy=True, replace_attrs=False, add_loc_to_name=True
    ):
        """Format name and attributes of a copy of DataArray

        Parameters
        ----------
        da: xarray.DataArray
        loc: {True, None}, letter, {False, ""}
            If None, location is left unchanged;
            if a letter, it is set;
            else, it is removed.
        standardize: bool
            If True, standardize ``root`` and ``loc`` values.
        name: str, None
            Substitute for data array name
        attrs: str, None
            Substitute for dataarray attributes.
            If ``standard_name`` and ``long_name`` values are a list,
            and if the dataarray has its attribute included in the list,
            it is left unchanged since it is considered compatible.
        rename:
            Allow renaming the array if its name is already set
        add_loc_to_name: bool
            Add the location to the name
        replace_attrs: bool
            Replace existing attributes?
        copy: bool
            Make sure to work on a copy

        Return
        ------
        xarray.DataArray

        See also
        --------
        format_attrs
        patch_attrs
        """
        if copy:
            da = da.copy()

        # Attributes
        if attrs:
            da.attrs.update(self.patch_attrs(
                da.attrs, attrs, loc, standardize=standardize,
                replace=replace_attrs))
        else:
            da.attrs.update(self.format_attrs(
                da.attrs, loc, standardize=standardize))

        # Name
        if rename or da.name is None:
            kw = {"loc": loc} if add_loc_to_name else {}
            da.name = self.merge_attr("name", da.name, name, **kw)

        # Check location consistency
        if loc is None or loc == "any" or loc is True:
            loc = self.get_location(da)
            if loc:
                da = self.format_dataarray(da, loc=loc, rename=True, replace_attrs=True)

        return da


def _solve_rename_conflicts_(rename_args):
    """Skip renaming items that overwride previous items"""
    used = {}
    for old_name in list(rename_args):
        new_name = rename_args[old_name]
        if new_name in used:
            del rename_args[old_name]
            xoa_warn(
                f"Cannot rename {old_name} to {new_name} since "
                f"{used[new_name]} will also be renamed to {new_name}. Skipping...")
        else:
            used[new_name] = old_name
    return rename_args


class CFSpecs(object):
    """Manager for CF specifications

    CF specifications are defined here an extension of a subset of
    CF conventions: known variables and coordinates are described through
    a generic name, a specialized name, alternates names, some properties
    and attributes like standard_name, long_name, axis.

    Have a look to the :ref:`default specifications <appendix.cf.default>`
    and to the :ref:`uses.cf` section.


    Parameters
    ----------
    cfg: str, list, CFSpecs, dict
        A config file name or string or dict or CF Specs, or a list of them.
        It may contain the "data_vars", "coords"  and "sglocator" sections.
        When a list is provided, specs are merged with the firsts having
        priority over the lasts.
    default: bool
        Load also the default internal specs
    user: bool
        Load also the user specs stored in :data:`USER_CF_FILE`
    name: str, None
        Assign a shortcut name. It defaults the the `[register] name`
        option of the specs.
    cache: bool
        Use in-memory cache system?

    See also
    --------
    CFCoordSpecs
    CFVarSpecs
    SGLocator
    :ref:`uses.cf`
    :ref:`appendix.cf.default`
    """

    def __init__(self, cfg=None, default=True, user=True, name=None, cache=None):

        # Initialiase categories
        self._cfs = {}
        catcls = {"data_vars": CFVarSpecs, "coords": CFCoordSpecs}
        for category in self.categories:
            self._cfs[category] = catcls[category](self)

        # Load config
        self._dict = None
        self._name = name
        self._cfgspecs = _get_cfgm_().specs.dict()
        self._cfgs = []
        self._load_default = default
        self._load_user = user
        self.load_cfg(cfg, cache=cache)

    def _load_cfg_as_dict_(self, cfg, cache=None):
        """Load a single cfg, validate it and return it as a dict

        When the config source is a tuple or a file name, the loaded config
        is in-memory cached by default. The cache key is the first item
        of the tuple or the file name.

        Parameters
        ----------
        cf: str, dict, CFSpecs
            Config source
        cache: bool
            Use in-memory cache system?

        """
        # Config manager to get defaults and validation
        cfgm = _get_cfgm_()

        # Get it from cache if from str or CFSpecs with registration name
        if cache is None:
            cache = get_option("cf.cache")
        cache = cache and ((isinstance(cfg, str) and '\n' not in cfg) or
                           (isinstance(cfg, dict) and "register" in cfg and
                            cfg["register"]["name"]))
        if cache:

            # Init cache
            if isinstance(cfg, str):
                cache_key = cfg
            elif (isinstance(cfg, dict) and "register" in cfg and cfg["register"]["name"]):
                cache_key = cfg["register"]["name"]
            cf_cache = _get_cache_()
            if cache_key in cf_cache["loaded_dicts"]:
                # a copy is needed because of the post processing
                return copy.deepcopy(cf_cache["loaded_dicts"][cache_key])

        # Check input type
        if isinstance(cfg, str) and '\n' in cfg:
            cfg = cfg.split("\n")
        elif isinstance(cfg, CFSpecs):
            cfg = cfg._dict

        # Load, validate and convert to dict
        cfg_dict = cfgm.load(cfg).dict()

        # Cache it
        if cache:
            # a copy is needed because of the post processing
            cf_cache["loaded_dicts"][cache_key] = copy.deepcopy(cfg_dict)

        return cfg_dict

    def load_cfg(self, cfg=None, cache=None):
        """Load a single or a list of configurations

        Parameters
        ----------
        cfg: ConfigObj init or list
            Single or a list of either:

            - config file name,
            - multiline config string or a list of lines,
            - config dict,
            - tuple of the previous.
        cache: bool, None
            In in-memory cache system?
            Defaults to option boolean :xoaoption:`cf.cache`.

        """
        # Get the list of validated configurations
        to_load = []
        if cfg:
            if not isinstance(cfg, tuple):
                cfg = (cfg,)
            to_load.extend([c for c in cfg if c])
        if self._load_user and os.path.exists(USER_CF_FILE):
            to_load.append(USER_CF_FILE)
        if self._load_default:
            to_load.append(_CFGFILE)
        if not to_load:
            to_load = [None]

        # Load them
        dicts = [self._load_cfg_as_dict_(cfg, cache) for cfg in to_load]

        # Merge them, except "register"
        self._dict = dict_merge(*dicts, **_CF_DICT_MERGE_KWARGS)
        self._dict["register"] = dicts[0]["register"]

        # SG locator
        self._sgl_settings = self._dict["sglocator"]
        self._sgl = SGLocator(**self._sgl_settings)

        # Post process
        self._post_process_()

    def copy(self):
        return CFSpecs(self, default=False, user=False)

    def __copy__(self):
        return self.copy()

    @property
    def dict(self):
        """Dictionary copy of the specs"""
        return self._dict.copy()

    def get_name(self):
        return self._dict["register"]["name"]

    def set_name(self, name):
        self._dict["register"]["name"] = self._name = name

    name = property(fset=set_name, fget=get_name, doc="Name")

    def pprint(self, **kwargs):
        """Pretty print the specs as dict using :func:`pprint.pprint`"""
        pprint.pprint(self.dict, **kwargs)

    @property
    def categories(self):
        """List of cf specs categories"""
        return ["data_vars", "coords"]

    @property
    def sglocator(self):
        """:class:`SGLocator` instance"""
        return self._sgl

    @property
    def cfgspecs(self):
        return self._cfgspecs

    def __getitem__(self, section):
        if section in self.categories:
            return self._cfs[section]
        for cat in self.categories:
            if section in self._cfs[cat]:
                return self._cfs[cat][section]
        return self._dict[section]

    def __contains__(self, category):
        return category in self.categories

    def __getattr__(self, name):
        if "_cfs" in self.__dict__ and name in self.__dict__["_cfs"]:
            return self.__dict__["_cfs"][name]
        if name == "dims":
            return self._dict['dims']
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name
            )
        )

    @property
    def dims(self):
        """Dictionary of dims per dimension type within x, y, z, t and f"""
        return self._dict['dims']

    @property
    def coords(self):
        """Specifications for coords :class:`CFCoordSpecs`"""
        return self._cfs['coords']

    @property
    def data_vars(self):
        """Specification for data_vars :class:`CFVarSpecs`"""
        return self._cfs['data_vars']

    # def __str__(self):
    #     return pformat(self._dict)

    def set_specs_from_cfg(self, cfg):
        """Update or create specs from a config"""
        self.load_cfg(cfg)

    def set_specs(self, category, name, **specs):
        """Update or create specs for an item"""
        data = {category: {name: specs}}
        self.load_cfg(data)

    def _post_process_(self):

        # Inits
        items = {}
        for category in self.categories:
            items[category] = []

        # Process
        for category in self.categories:
            for name in self[category].names:
                for item in self._process_entry_(category, name):
                    if item:
                        pcat, pname, pspecs = item
                        items[pcat].append((pname, pspecs))

        # Refill
        for category in items:
            self._dict[category].clear()
            self._dict[category].update(items[category])

        # Updates valid dimensions
        alt_names = {}
        for name, coord_specs in self._dict['coords'].items():
            if coord_specs["attrs"]["axis"]:
                axis = coord_specs["attrs"]['axis'].lower()
                self._dict['dims'][axis].append(name)  # generic name
                if coord_specs['name']:  # specialized names
                    self._dict['dims'][axis].append(coord_specs['name'])
                alt_names.setdefault(axis, []).extend(coord_specs['alt_names'])  # alternate names
        for axis in self._dict['dims']:
            if axis in alt_names:
                self._dict['dims'][axis].extend(alt_names[axis])

        # Force the registration name
        if self._name:
            self._dict["register"]["name"] = self._name

    def _process_entry_(self, category, name):
        """Process an entry

        - Check inheritance
        - Set a default long_name from standard_name
        - Process the "select" key

        Yield
        -----
        category, name, entry
        """
        # Dict of entries for this category
        entries = self._dict[category]

        # Wrong entry!
        if name not in entries:
            yield

        # Already processed
        if "processed" in entries[name]:
            yield

        # Get the specs as pure dict
        if hasattr(entries[name], "dict"):
            entries[name] = entries[name].dict()
        specs = entries[name]

        # Inherits from other specs (merge specs with dict_merge)
        if "inherit" in specs and specs["inherit"]:

            # From what?
            from_name = specs["inherit"]
            if ":" in from_name:
                from_cat, from_name = from_name.split(":")[:2]
            else:
                from_cat = category
            assert (
                from_cat != category or name != from_name
            ), "Cannot inherit cf specs from it self"

            # Parents must be already processed
            for item in self._process_entry_(from_cat, from_name):
                yield item

            # Inherit with merging
            entries[name] = specs = dict_merge(
                specs,
                self._dict[from_cat][from_name],
                cls=dict,
                **_CF_DICT_MERGE_KWARGS,
            )

            # Check compatibility of keys when not from same category
            if category != from_cat:
                for key in list(specs.keys()):
                    # print(self._cfgspecs[category])
                    # print(self._cfgspecs[from_cat])
                    if (key not in self._cfgspecs[category]["__many__"] and
                            key in self._cfgspecs[from_cat]["__many__"]):
                        del specs[key]

            # Switch off inheritance now
            specs["inherit"] = None

        # Long name from name or standard_name
        if not specs["attrs"]["long_name"]:
            if specs["attrs"]["standard_name"]:
                long_name = specs["attrs"]["standard_name"][0]
            else:
                long_name = name.title()
            long_name = long_name.replace("_", " ").capitalize()
            specs["attrs"]["long_name"].append(long_name)

        # Select
        if specs.get("select", None):
            for key in specs["select"]:
                try:
                    specs["select"] = eval(specs["select"])
                except Exception:
                    pass

        specs['processed'] = True
        yield category, name, specs

    def get_loc(self, da):
        """Get the staggered grid location from name and standard_name

        Parameters
        ----------
        da: xarray.DataArray

        Return
        ------
        str, None

        See also
        --------
        SGLocator.get_location
        """
        return self.sglocator.get_location(da)

    get_location = get_loc

    def format_coord(self, da, name=None, loc=None, copy=True,
                     standardize=True, rename=True, rename_dim=True,
                     specialize=False, attrs=True, replace_attrs=False,
                     add_loc_to_name=None, rename_dims=True):
        """Format a coordinate array

        Parameters
        ----------
        da: xarray.DataArray
        name: str, None
            A CF name. If not provided, it guessed with :meth:`match`.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        rename: bool
            Rename arrays
        add_loc_to_name: bool
            Add loc to the name
        specialize: bool
            Does not use the CF name for renaming, but the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.
        standardize: bool
        rename_dim: bool
            For a 1D array, rename the dimension if it has the same name
            as the array.
            Note that it is set to False, if ``rename`` is False.
        attrs: bool, dict
            If False, does not change attributes at all.
            If True, use Cf attributes.
            If a dict, use this dict.
        replace_attrs: bool
            Replace existing attributes?

        Returns
        -------
        xarray.DataArray, str, None
            The formatted array or copy of it.
            The CF name, given or matching, if rename if False; and None
            if not matching.

        See also
        --------
        CFCoordSpecs.format_dataarray
        """
        return self.coords.format_dataarray(
            da, name=name, loc=loc, copy=copy, standardize=standardize,
            rename=rename, rename_dim=rename_dim,
            replace_attrs=replace_attrs, attrs=attrs,
            specialize=specialize, add_loc_to_name=add_loc_to_name)

    def format_data_var(
        self,
        da,
        name=None,
        rename=True,
        specialize=False,
        attrs=True,
        replace_attrs=False,
        format_coords=True,
        coords=None,
        loc=None,
        standardize=True,
        copy=True,
        add_loc_to_name=None,
        coord_add_loc_to_name=None,
        rename_dims=True
    ):
        """Format array name and attributes according to currents CF specs

        Parameters
        ----------
        da: xarray.DataArray
        name: str, None
            CF name. If None, it is guessed.
        loc: str, {"any", None}, {"", False}

            - str: one of these locations
            - None or "any": any
            - False or '"": no location

        rename: bool
            Not only format attribute, but also rename arrays,
            thus making a copies.
        add_loc_to_name: bool
            Add loc to the name
        coord_add_loc_to_name: bool
            Add loc to the coordinates name
        specialize: bool
            Does not use the CF name for renaming, but the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.
        rename_dims:
            Also rename dimensions that are not coordinates
        attrs: bool, dict
            If False, does not change attributes at all.
            If True, use Cf attributes.
            If a dict, use this dict.
        replace_attrs: bool
            Replace existing attributes?
        format_coords: bool
            Also format coords.
        coords: dict, None
            Dict whose keys are coord names, and values are CF coord names.
            Used only if ``format_coords`` is True.
        standardize: bool
        copy: bool
            Make sure to work on a copy of the array.
            Note that a copy is always returned when ``rename`` is True.

        Returns
        -------
        xarray.DataArray
            Formatted array

        See also
        --------
        CFCoordSpecs.format_dataarray
        CFVarSpecs.format_dataarray
        """
        # Copy
        if copy:
            da = da.copy()

        # Init rename dict
        rename_args = {}

        # Data var
        for cat in "data_vars", "coords":
            new_name = self[cat].format_dataarray(
                da, name=name, loc=loc, standardize=standardize,
                rename=False, copy=False, replace_attrs=replace_attrs,
                attrs=attrs, specialize=specialize,
                add_loc_to_name=add_loc_to_name
            )
            if new_name:
                break
        if rename and new_name:
            da.name = new_name

        # Coordinates
        if format_coords:
            coords = coords or {}
            for cname, cda in list(da.coords.items()):
                rename_args[cda.name] = self.format_coord(
                    cda,
                    name=coords.get(cname),
                    loc=loc,
                    standardize=standardize,
                    rename=False,
                    copy=False,
                    replace_attrs=replace_attrs,
                    add_loc_to_name=coord_add_loc_to_name
                )

        # Dimensions
        if rename and rename_dims:
            rename_dims_args = self.coords.get_rename_dims_args(
                da, loc=loc, specialize=specialize)#, exclude=list(rename_args.keys()))
            rename_args.update(rename_dims_args)

        # Final renaming
        if rename and rename_args:
            _solve_rename_conflicts_(rename_args)
            da = da.rename(rename_args)

        # Return the guessed name
        if not rename and name is None:
            return new_name

        return da

    def format_dataset(
            self, ds, loc=None, rename=True, standardize=True, format_coords=True,
            coords=None, copy=True, replace_attrs=False, add_loc_to_name=None,
            coord_add_loc_to_name=None, specialize=False,
            rename_dims=True):
        """Auto-format a whole xarray.Dataset

        See also
        --------
        format_data_var
        format_coord
        """
        # Copy
        if copy:
            ds = ds.copy()

        # Init rename dict
        rename_args = {}

        # Data arrays
        for name, da in list(ds.items()):
            new_name = self.format_data_var(
                da, loc=loc, rename=False, standardize=True,
                format_coords=False, copy=False, replace_attrs=replace_attrs,
                add_loc_to_name=add_loc_to_name, specialize=specialize)
            if rename and new_name:
                rename_args[da.name] = new_name

        # Coordinates
        if format_coords:
            for cname, cda in list(ds.coords.items()):
                new_name = self.format_coord(
                    cda, loc=loc, standardize=True, rename=False,
                    rename_dim=False, copy=False, replace_attrs=replace_attrs,
                    add_loc_to_name=coord_add_loc_to_name, specialize=specialize)
                if rename and new_name:
                    rename_args[cda.name] = new_name

        # Dimensions
        if rename_dims:
            rename_dims_args = self.coords.get_rename_dims_args(
                ds, loc=loc, specialize=specialize)#, exclude=list(rename_args.keys()))
            rename_args.update(rename_dims_args)

        # Final renaming
        if rename and rename_args:
            _solve_rename_conflicts_(rename_args)
            ds = ds.rename(rename_args)

        return ds

    def auto_format(self, dsa, **kwargs):
        """Rename variables and coordinates and fill their attributes

        See also
        --------
        encode
        format_dataarray
        format_dataset
        fill_attrs
        """
        if hasattr(dsa, "data_vars"):
            return self.format_dataset(dsa, format_coords=True, **kwargs)
        return self.format_data_var(dsa, **kwargs)

    def decode(self, dsa, **kwargs):
        """Auto format, infer coordinates and rename to generic names

        See also
        --------
        auto_format
        encode
        format_dataarray
        format_dataset
        fill_attrs
        """
        # Names and attributes
        dsa = self.auto_format(dsa, **kwargs)

        # Coordinates
        dsa = self.infer_coords(dsa)

        # Assign cf specs
        if self.name and self in get_registered_cf_specs():
            dsa = assign_cf_specs(dsa, self.name)

        return dsa

    def encode(self, dsa, **kwargs):
        """Same as :meth:`decode` but rename with the specialized name

        See also
        --------
        decode
        auto_format
        format_dataarray
        format_dataset
        fill_attrs
        """
        kwargs.setdefault("specialize", True)
        return self.decode(dsa, **kwargs)

    def fill_attrs(self, dsa, **kwargs):
        """Fill missing attributes of a xarray.Dataset or xarray.DataArray

        .. note:: It does not rename anything

        See also
        --------
        format_dataarray
        format_dataset
        auto_format
        """
        kwargs.update(rename=False, replace_attrs=False)
        if hasattr(dsa, "data_vars"):
            return self.format_dataset(dsa, format_coords=True, **kwargs)
        return self.format_data_var(dsa, **kwargs)

    def match_coord(self, da, name=None, loc="any"):
        """Check if an array matches a given or any coord specs

        Parameters
        ----------
        da: xarray.DataArray
        name: str, dict, None
            Cf name.
            If None, all names are used.
            If a dict, name is interpreted as an explicit set of
            specifications.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location

        Returns
        -------
        bool, str
            True or False if name is provided, else found name or None

        See also
        --------
        CFCoordSpecs.match
        """
        return self.coords.match(da, name=name, loc=loc)

    def match_data_var(self, da, name=None, loc="any"):
        """Check if an array matches given or any data_var specs

        Parameters
        ----------
        da: xarray.DataArray
        name: str, dict, None
            Cf name.
            If None, all names are used.
            If a dict, name is interpreted as an explicit set of
            specifications.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location

        Returns
        -------
        bool, str
            True or False if name is provided, else found name or None

        See also
        --------
        CFVarSpecs.match
        """
        return self.data_vars.match(da, name=name, loc=loc)

    @staticmethod
    def get_category(da):
        """Guess if a datarray belongs to data_vars or coords

        It belongs to coords if one of its dimensions or
        coordinates has its name.

        Parameters
        ----------
        da: xarray.DataArray

        Returns
        -------
        str
        """
        if da.name is not None and (da.name in da.dims or
                                    da.name in da.coords):
            return "coords"
        return "data_vars"

    def match(self, da, loc="any"):
        """Check if an array matches any data_var or coord specs

        Parameters
        ----------
        da: xarray.DataArray
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location

        Return
        ------
        str, None
            Category
        str, None
            Name
        """
        category = self.get_category(da)
        categories = list(self.categories)
        if category != categories[0]:
            categories = categories[::-1]
        for category in categories:
            name = self[category].match(da, loc=loc)
            if name:
                return category, name
        return None, None

    @ERRORS.format_method_docstring
    def search_coord(
            self, dsa, name=None, loc="any", get="obj", single=True,
            errors="warn"):
        """Search for a coord that maches given or any specs

        Parameters
        ----------
        dsa: DataArray or Dataset
        name: str, dict
            A CF name. If not provided, all CF names are scaned.
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or "": no location
        get: {{"obj", "name"}}
            When found, get the object found or its name.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
            A warning is emitted when set to True and multiple item are found.
        {errors}

        Returns
        -------
        None or str or object

        Example
        -------
        .. ipython:: python

            @suppress
            from xoa.cf import get_cf_specs
            @suppress
            import xarray as xr, numpy as np
            lon = xr.DataArray([2, 3], dims='foo',
                               attrs={{'standard_name': 'longitude'}})
            data = xr.DataArray([0, 1], dims=('foo'), coords=[lon])
            cfspecs = get_cf_specs()
            cfspecs.search_coord(data, "lon")
            cfspecs.search_coord(data, "lon", get="name")
            cfspecs.search_coord(data, "lat", errors="ignore")

        See also
        --------
        search_data_var
        CFCoordSpecs.search
        """
        return self.coords.search(
            dsa, name=name, loc=loc, get=get, single=single, errors=errors)

    @ERRORS.format_method_docstring
    def search_dim(self, da, dim_type=None, loc="any", errors="ignore"):
        """Search for a dimension from its type

        Parameters
        ----------
        da: xarray.DataArray
        dim_type: None, {{"x", "y", "z", "t", "f"}}
            Dimension type
        loc:
            Location
        {errors}

        Return
        ------
        str, (str, str)
            Dim name OR, (dim name, dim_type) if dim_type is None

        See also
        --------
        CFCoordSpecs.search_dim
        """
        return self.coords.search_dim(
            da, dim_type=dim_type, loc=loc, errors=errors)

    @ERRORS.format_method_docstring
    def search_coord_from_dim(self, da, dim, errors="ignore"):
        """Search a dataarray for a coordinate from a dimension name

        It first searches for a coordinate with the same name and that is
        the only one having this dimension.
        Then look for coordinates with the same type like x, y, etc.

        Parameters
        ----------
        da: xarray.DataArray
        dim: str
        {errors}

        Return
        ------
        xarray.DataArray, None
            An coordinate array or None
        """
        return self.coords.search_from_dim(da, dim, errors=errors)

    @ERRORS.format_method_docstring
    def search_data_var(
            self, dsa, name=None, loc="any", get="obj", single=True,
            errors="warn"):
        """Search for a data_var that maches given or any specs

        Parameters
        ----------
        dsa: DataArray or Dataset
        name: str, dict
            A CF name. If not provided, all CF names are scaned.
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {{"obj", "name"}}
            When found, get the object found or its name.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
            A warning is emitted when set to True and multiple item are found.
        {errors}

        Returns
        -------
        None or str or object

        Example
        -------
        .. ipython:: python
            :okwarning:

            @suppress
            from xoa.cf import get_cf_specs
            @suppress
            import xarray as xr, numpy as np
            data = xr.DataArray(
                [0, 1], dims=('x'),
                attrs={{'standard_name': 'sea_water_temperature'}})
            ds = xr.Dataset({{'foo': data}})
            cfspecs = get_cf_specs()
            cfspecs.search_data_var(ds, "temp")
            cfspecs.search_data_var(ds, "temp", get="name")
            cfspecs.search_data_var(ds, "sal")

        See also
        --------
        search_coord
        CFVarSpecs.search
        """
        return self.data_vars.search(
            dsa, name=name, loc=loc, get=get, single=single, errors=errors)

    @ERRORS.format_method_docstring
    def search(self, dsa, name=None, loc="any", get="obj",
               single=True, categories=None, errors="warn"):
        """Search for a dataarray with data_vars and/or coords


        Parameters
        ----------
        dsa: xarray.DataArray, xarray.Dataset
            Array or dataset to scan
        name: str
            Name to search for.
        categories: str, list, None
            Explicty categories with "coords" and "data_vars".
        {errors}

        Return
        ------
        None, xarray.DataArray, list

        """
        if not categories:
            categories = (self.categories if hasattr(dsa, "data_vars")
                          else ["coords", "data_vars"])
        elif isinstance(categories, str):
            categories = [categories]
        else:
            categories = self.categories
        errors = ERRORS[errors]
        if not single:
            found = []
        for category in categories:
            res = self[category].search(dsa, name=name, loc=loc, get=get,
                                        single=single, errors="ignore")
            if not single:
                res = [r for r in res if r not in found]
                found.extend(res)
            elif res is not None:
                return res
        if not single:
            return found
        msg = "Search failed"
        if errors == "warn":
            xoa_warn(msg)
        elif errors == "raise":
            raise XoaCFError(msg)

    def get(self, dsa, name, get="obj"):
        """A shortcut to :meth:`search` with an explicit name

        A single element is searched for into all :attr:`categories`
        and errors are ignored.
        """
        return self.search(dsa, name, errors="ignore", single=True, get=get)
        # if da is None:
        #     raise XoaCFError("Search failed for the following cf name: "
        #                      + name)
        # return da

    @ERRORS.format_method_docstring
    def get_dims(self, da, dim_types, allow_positional=False,
                 positions='tzyx', errors="warn"):
        """Get the data array dimensions names from their type

        Parameters
        ----------
        da: xarray.DataArray
            Array to scan
        dim_types: str, list
            Letters among "x", "y", "z", "t" and "f".
        allow_positional: bool
            Fall back to positional dimension of types is unkown.
        positions: str
            Default position per type starting from the end.
        {errors}

        Return
        ------
        tuple
            Tuple of dimension name or None when the dimension if not found

        See also
        --------
        CFCoordSpecs.get_dims
        """
        return self.coords.get_dims(
            da, dim_types, allow_positional=allow_positional,
            positions=positions, errors=errors)

    def get_axis(self, coord, lower=False):
        return self.coords.get_axis(coord, lower=lower)

    def get_dim_type(self, dim, da=None, lower=True):
        return self.coords.get_dim_type(dim, da=da, lower=lower)

    def get_dim_types(self, da, unknown=None, asdict=False):
        return self.coords.get_dim_types(
            da, unknown=unknown, asdict=asdict)

    def parse_dims(self, dims, dsa):
        return self.coords.parse_dims(dims, dsa)

    def infer_coords(self, ds):
        """Search for known coords and make sure they are set as coords

        Parameters
        ----------
        ds: xarray.Dataset

        Return
        ------
        xarray.Dataset
            New dataset with potentially updated coordinates
        """
        if hasattr(ds, "data_vars"):
            for da in ds.data_vars.values():
                if self.coords.match(da):
                    ds = ds.set_coords(da.name)
        return ds.copy()

class _CFCatSpecs_(object):
    """Base class for loading data_vars and coords CF specifications"""

    category = None

    attrs_exclude = [
        "name",
        "inherit",
        "coords",
        "select",
        "search_order",
        "cmap",
    ]

    attrs_first = [
        "name",
        "standard_name",
        "long_name",
        "units",
    ]

    def __init__(self, parent):
        assert self.category in parent
        self.parent = parent

    @property
    def sglocator(self):
        """:class:`SGLocator` instance"""
        return self.parent.sglocator

    @property
    def _dict(self):
        return self.parent._dict[self.category]

    def __getitem__(self, key):
        return self.get_specs(key)

    def __iter__(self):
        return iter(self._dict)

    def __len__(self):
        return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    # def __str__(self):
    #     return pformat(self._dict)

    def _validate_name_(self, name):
        if name in self:
            return name

    def _assert_known_(self, name, errors="raise"):
        if name not in self._dict:
            if errors == "raise":
                raise XoaCFError(
                    f"Invalid {self.category} CF specs name: "+name)
            return False
        return True

    @property
    def names(self):
        return [key for key in self._dict.keys() if not key.startswith("_")]

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def get_specs(self, name, errors="warn"):
        """Get the specs of a cf item

        Parameters
        ----------
        name: str
        errors: "silent", "warning" or "error".

        Return
        ------
        dict or None
        """
        errors = ERRORS[errors]
        if name not in self._dict:
            if errors == "raise":
                raise XoaCFError("Can't get cf specs from: " + name)
            if errors == "warn":
                xoa_warn("Invalid cf name: " + str(name))
            return
        return self._dict[name]

    @property
    def dims(self):
        """Dims per dimension types"""
        return self.parent._dict['dims']

    def set_specs(self, item, **specs):
        """Update or create specs for an item"""
        data = {self.category: {item: specs}}
        self.parent.load_cfg(data)

    def set_specs_from_cfg(self, cfg):
        """Update or create specs for several item with a config specs"""
        if isinstance(cfg, dict) and self.category not in cfg:
            cfg = {self.category: cfg}
        self.parent.load_cfg(cfg)

    def _get_ordered_match_specs_(self, name):
        specs = self[name]
        match_specs = {}
        for sm in specs["search_order"]:
            for attr in (
                "name",
                "standard_name",
                "long_name",
                "axis",
                "units",
            ):
                if attr[0] != sm:
                    continue
                # if attr == "name" and "name" in specs:
                if attr == "name":
                    match_specs["name"] = [name]
                    if "name" in specs and specs["name"]:
                        match_specs["name"].append(specs["name"])
                    if "alt_names" in specs:
                        match_specs["name"].extend(specs["alt_names"])
                elif "attrs" in specs and attr in specs["attrs"]:
                    match_specs[attr] = specs["attrs"][attr]
        return match_specs

    def match(self, da, name=None, loc="any"):
        """Check if da attributes match given or any specs

        Parameters
        ----------
        da: xarray.DataArray
        name: str, dict, None
            CF name.
            If None, all names are used.
            If a dict, name is interpreted as an explicit set of
            specifications.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location

        Return
        ------
        bool, str
            True or False if name is provided, else found name or None
        """
        if name:
            if isinstance(name, str):
                self._assert_known_(name)
            names = [name]
        else:
            names = self.names
        for name_ in names:

            # Get match specs
            if isinstance(name_, dict):
                match_specs = name_
            else:
                match_specs = self._get_ordered_match_specs_(name_)

            # Loop on match specs
            for attr, refs in match_specs.items():
                value = getattr(da, attr, None)
                if value is None:
                    continue
                for ref in refs:
                    if (
                        attr in self.sglocator.valid_attrs
                        and self.sglocator.match_attr(
                            attr, value, ref, loc=loc
                        )
                    ) or match_string(value, ref, ignorecase=True):
                        return True if name else name_
        return False if name else None

    @ERRORS.format_method_docstring
    def search(self, dsa, name=None, loc="any", get="obj", single=True,
               errors="raise"):
        """Search for a data_var or coord that maches given or any specs

        Parameters
        ----------
        dsa: DataArray or Dataset
        name: str, dict
            A CF name. If not provided, all CF names are scaned.
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {{"obj", "name"}}
            When found, get the object found or its name.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
            A warning is emitted when set to True and multiple item are found.
        {errors}

        Returns
        -------
        None or str or object
        """

        # Get target objects
        if self.category and hasattr(dsa, self.category):
            objs = getattr(dsa, self.category)
        else:
            objs = dsa.keys() if hasattr(dsa, "keys") else dsa.coords

        # Get match specs
        if name:  # Explicit name so we loop on search specs
            if isinstance(name, str):
                if not self._assert_known_(name, errors):
                    return
            match_specs = []
            for attr, refs in self._get_ordered_match_specs_(name).items():
                match_specs.append({attr: refs})
        else:
            match_specs = [None]

        # Loops
        assert get in ("name", "obj", "both"), (
            "'get' must be either 'name' or 'obj' or 'both'")
        found = []
        found_objs = []
        for match_arg in match_specs:
            for obj in objs.values():
                m = self.match(obj, match_arg, loc=loc)
                if m:
                    if obj.name in found_objs:
                        continue
                    found_objs.append(obj.name)
                    name = name if name else m
                    if get == "both":
                        found.append((obj, name))
                    else:
                        found.append(obj if get == "obj" else name)

        # Return
        if not single:
            return found
        errors = ERRORS[errors]
        if errors != "ignore" and len(found) > 1:
            msg = "Multiple items found while you requested a single one"
            if errors == "raise":
                raise XoaCFError(msg)
            xoa_warn(msg)
        if found:
            return found[0]
        if errors != "ignore":
            msg = "No match item found"
            if errors == "raise":
                raise XoaCFError(msg)
            xoa_warn(msg)

    def get(self, dsa, name):
        """Call to :meth:`search` with an explicit name and ignoring errors"""
        return self.search(dsa, name, errors="ignore")

    @ERRORS.format_method_docstring
    def get_attrs(
        self, name, select=None, exclude=None, errors="warn", loc=None,
        multi=False, standardize=True, **extra
    ):
        """Get the default attributes from cf specs

        Parameters
        ----------
        name: str
            Valid CF name
        select: str, list
            Include only these attributes
        exclude: str, list
            Exclude these attributes
        multi: bool
            Get standard_name and long_name attribute as a list of possible
            values
        {errors}
        extra: dict
          Extra params are included as extra attributes

        Return
        ------
        dict
        """

        # Get specs
        specs = self.get_specs(name, errors=errors)
        if not specs:
            return {}

        # Which attributes
        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
        # exclude.extend(self.attrs_exclude)
        exclude.extend(extra.keys())
        exclude = set(exclude)
        keys = set(specs["attrs"].keys())
        keys -= exclude
        if select:
            keys = keys.intersection(select)

        # Loop
        attrs = {}
        for key in specs["attrs"].keys():

            # No lists or tuples
            value = specs["attrs"][key]
            if isinstance(value, list):
                if len(value) == 0:
                    continue
                if not multi or key not in ("standard_name", "long_name"):
                    value = value[0]

            # Store it
            attrs[key] = value

        # Extra attributes
        attrs.update(extra)

        # Finalize and optionally change location
        attrs = self.parent.sglocator.format_attrs(
            attrs, loc=loc, standardize=standardize)

        return attrs

    def get_name(self, name, specialize=False, loc=None):
        """Get the name of the matching CF specs

        Parameters
        ----------
        name: str, xarray.DataArray
            Either a cf name or a data array
        specialize: bool
            Get the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.
        loc: str, None
            Format it at this location

        Return
        ------
        None or str
        """
        if not isinstance(name, str):
            name = self.match(name)
        if name is None:
            return
        if specialize and self[name]["name"]:
            name = self[name]["name"]
        return self.sglocator.format_attr("name", name, loc=loc)

    def format_dataarray(
            self, da, name=None, loc=None, rename=True, attrs=True, standardize=True,
            specialize=False, rename_dim=True, replace_attrs=False, copy=True, add_loc_to_name=None):
        """Format a DataArray's name and attributes

        Parameters
        ----------
        da: xarray.DataArray
        name: str, None
            A CF name. If not provided, it guessed with :meth:`match`.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        rename: bool
            Rename arrays when their name is set?
        add_loc_to_name: None, bool
            Add the loc to the name, overriding the specs.
        attrs: bool, dict
            If False, does not change attributes at all.
            If True, use Cf attributes.
            If a dict, use this dict.
        replace_attrs: bool
            Replace existing attributes?
        standardize: bool
        specialize: bool
            Does not use the CF name for renaming, but the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.
        rename_dim: bool
            For a 1D array, rename the dimension if it has the same name
            as the array.
            Note that it is set to False, if ``rename`` is False.
        copy: bool
            Force a copy (not of the data) in any case?

        Returns
        -------
        xarray.DataArray, str, None
            The formatted array or copy of it.
            The CF name, given or matching, if rename if False; and None
            if not matching.

        """
        if rename:
            copy = True
        if copy:
            da = da.copy()

        # Get names
        if name is None:
            name = self.match(da, loc="any")
        if name is None:
            if not rename:
                return
            return da.copy() if copy else None
        assert name in self.names
        old_name = da.name
        cf_name = name
        new_name = self.get_name(cf_name, specialize=specialize)
        if add_loc_to_name is None:
            add_loc_to_name = self[cf_name]["add_loc"]
        if add_loc_to_name is None and old_name:
            add_loc_to_name = bool(self.sglocator.parse_attr("name", old_name)[1])

        # Attributes
        if attrs is True:
            attrs = self.get_attrs(cf_name, loc=None, standardize=False, multi=True)
            if da.name and da.name not in da.indexes and "axis" in attrs:
                del attrs["axis"]
        elif not attrs:
            attrs = {}

        # Format array
        new_da = self.sglocator.format_dataarray(
            da,
            loc=loc,
            name=new_name,
            attrs=attrs,
            standardize=standardize,
            rename=rename,
            add_loc_to_name=add_loc_to_name,
            replace_attrs=replace_attrs,
            copy=False
        )

        # Return renaming name but don't rename
        if not rename:
            if not add_loc_to_name:
                return new_name
            if loc is None or loc == "any":
                loc = self.sglocator.get_location(new_da)
            return self.sglocator.merge_attr("name", old_name, new_name, loc)

        # Rename dim if axis coordinate
        rename_dim = rename and rename_dim
        if (rename_dim and old_name and old_name in da.indexes):
            new_da = new_da.rename({old_name: new_da.name})
        return new_da

    def rename_dataarray(
            self, da, name=None, specialize=False, loc=None, standardize=True, rename_dim=True,
            copy=True, add_loc_to_name=None):
        """Rename a DataArray

        It is a specialized call to :meth:`format_dataarray` where
        attributes are left unchanged.

        Parameters
        ----------
        da: xarray.DataArray
        name: str, None
            A CF name. If not provided, it guessed with :meth:`match`.
        specialize: bool
            Does not use the CF name for renaming, but the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        standardize: bool
        rename_dim: bool
            For a 1D array, rename the dimension if it has the same name
            as the array.
            Note that it is set to False, if ``rename`` is False.
        copy: bool
            Force a copy (not of the data) in any case?


        See also
        --------
        format_dataarray
        """
        return self.format_dataarray(
            da, name=name, specialize=specialize, loc=loc, attrs=False,
            standardize=standardize, rename_dim=rename_dim, copy=copy,
            add_loc_to_name=add_loc_to_name)


class CFVarSpecs(_CFCatSpecs_):
    """CF specification for data_vars"""

    category = "data_vars"


class CFCoordSpecs(_CFCatSpecs_):
    """CF specification for coords"""

    category = "coords"

    def get_axis(self, coord, lower=False):
        """Get the dimension type, either from axis attr or from match Cf coord

        .. note:: The ``axis`` is the uppercase version of the ``dim_type``

        Parameters
        ----------
        coord: xarray.DataArray
        lower: bool
            Lower case?

        Return
        ------
        {"x", "y", "z", "t", "f"}, None

        See also
        --------
        get_dim_type
        get_dim_types
        """
        axis = None
        if "axis" in coord.attrs:
            axis = coord.attrs["axis"]
        else:
            cfname = self.match(coord)
            if cfname:
                axis = self[cfname]["attrs"]["axis"]
        if axis is not None:
            if lower:
                return axis.lower()
            return axis.upper()

    def get_dim_type(self, dim, da=None, lower=True):
        """Get the type of a dimension

        Three cases:

        - This dimension is registered in CF dims.
        - da has dim as dim and has an axis attribute inferred with :meth:`get_axis`.
        - da has a coord named dim with an axis attribute inferred with :meth:`get_axis`.

        Parameters
        ----------
        dim: str
            Dimension name
        da: xarray.DataArray
            Data array that the dimension belongs to, to help inferring
            the type
        lower: bool
            For lower case

        Return
        ------
        str, None
            Letter as one of x, y, z, t or f, if found, else None

        See also
        --------
        get_axis
        """
        # Remove location
        dim_loc = dim
        dim = self.sglocator.parse_attr('name', dim)[0]

        # Loop on types
        if dim.lower() in self.dims:
            return dim.lower()
        for dim_type, dims in self.dims.items():
            if dim.lower() in dims:
                return dim_type

        # Check if a coordinate have the same name and an axis type
        if da is not None:

            # Check dim validity
            if dim_loc not in da.dims:
                raise XoaCFError(f"dimension '{dim}' does not belong to da")

            # Check axis from coords
            if dim in da.indexes:
                return self.get_axis(da.coords[dim], lower=True)

            # Check da axis itself
            axis = self.get_axis(da, lower=True)
            if axis:
                return axis

    def get_dim_types(self, da, unknown=None, asdict=False):
        """Get a tuple of the dimension types of an array

        Parameters
        ----------
        da: xarray.DataArray or tuple(str)
            Data array or tuple of dimensions
        unknown:
            Value to assign when type is unknown
        asdict: bool

        Return
        ------
        tuple, dict
            Tuple of dimension types and of length ``da.ndim``.
            A dimension type is either a letter or the ``unkown`` value
            when the inference of type has failed.
            If ``asdict`` is True, a dict is returned instead,
            ``(dim, dim_type)`` as key-value pairs.

        See also
        --------
        get_dim_type
        """
        dim_types = {}
        if isinstance(da, tuple):
            dims = da
            da = None
        else:
            dims = da.dims
        for dim in dims:
            dim_type = self.get_dim_type(dim, da=da)
            if dim_type is None:
                dim_type = unknown
            dim_types[dim] = dim_type
        if asdict:
            return dim_types
        return tuple(dim_types.values())

    @ERRORS.format_method_docstring
    def search_dim(self, da, dim_type=None, loc="any", errors="ignore"):
        """Search a dataarray for a dimension name according to its type

        First, scan the dimension names.
        Then, look for coordinates: either it has an 'axis' attribute,
        or it a known CF coordinate.

        Parameters
        ----------
        da: xarray.DataArray
            Coordinate or data array
        dim_type: {{"x", "y", "z", "t", "f"}}, None
            When set to None, it is inferred with :meth:`get_axis`
        loc: "any", letter
            Staggered grid location
        {errors}

        Return
        ------
        str, (str, str), None
            Dim name OR, (dim name, dim_type) if dim_type is None.
            None if nothing found.
        """
        # Fixed dim type?
        with_dim_type = dim_type is not None
        if with_dim_type:
            dim_type = dim_type.lower()
        else:
            dim_type = self.get_axis(da, lower=True)

        # Loop on dims
        for dim in da.dims:

            # Filter by loc
            pname, ploc = self.sglocator.parse_attr('name', dim)
            if loc != "any" and ploc and loc and loc != ploc:
                continue

            # Type of the current dim
            this_dim_type = self.get_dim_type(dim, da=da)

            # This must match dim_type
            if with_dim_type:
                if this_dim_type and this_dim_type == dim_type:
                    return dim
                continue

            # Any dim_type but no ambiguity because same as da
            elif dim_type and this_dim_type and this_dim_type == dim_type:
                return dim, dim_type

        # Not found but only 1d and no dim_type specified
        if da.ndim == 1 and not with_dim_type:
            #FIXME: loop on coordinates?
            return dim, this_dim_type

        # Failed
        errors = ERRORS[errors]
        if errors != "ignore":
            msg = f"No dimension found in dataarray matching type: {dim_type}"
            if errors == "raise":
                raise XoaCFError(msg)
            xoa_warn(msg)
        if with_dim_type:
            return
        return None, None

    @ERRORS.format_method_docstring
    def search_from_dim(self, da, dim, errors="ignore"):
        """Search a dataarray for a coordinate from a dimension name

        It first searches for a coordinate with a different name and that is
        the only one having this dimension.
        Then check if it is an index.
        Then look for coordinates with the same type like x, y, etc.

        Parameters
        ----------
        da: xarray.DataArray
        dim: str
        {errors}

        Return
        ------
        xarray.DataArray, None
            An coordinate array or None

        See also
        --------
        get_axis
        get_dim_type
        """
        if dim not in da.dims:
            raise XoaError(f"Invalid dimension: {dim}")

        # A coord with a different name
        coords = [coord for name, coord in da.coords.items() if name != dim and dim in coord.dims]
        if len(coords) == 1:
            return coords[0]

        # As an index
        if dim in da.indexes:
            return da.coords[dim]

        # Get dim_type from known dim name
        dim_type = self.get_dim_type(dim, da=da, lower=True)

        # So we can do something
        if dim_type is not None:

            # Look for a coordinate with this dim_type
            #  starting from coordinates with a higher number of dimensions
            #  like depth that have more dims than level
            for coord in sorted(da.coords.values(), key=operator.attrgetter('ndim'), reverse=True):
                if dim in coord.dims:
                    coord_dim_type = self.get_axis(coord, lower=True)
                    if coord_dim_type and coord_dim_type == dim_type:
                        return coord

        # Nothing found
        errors = ERRORS[errors]
        if errors != "ignore":
            msg = f"No dataarray coord found from dim: {dim}"
            if errors == "raise":
                raise XoaCFError(msg)
            xoa_warn(msg)

    @ERRORS.format_method_docstring
    def get_dims(self, da, dim_types, allow_positional=False,
                 positions='tzyx', errors="warn"):
        """Get the data array dimensions names from their type

        Parameters
        ----------
        da: xarray.DataArray
            Array to scan
        dim_types: str, list
            Letters among "x", "y", "z", "t" and "f".
        allow_positional: bool
            Fall back to positional dimension of types is unkown.
        positions: str
            Default expected position of dim per type in da
            starting from the end.
        {errors}

        Return
        ------
        tuple
            Tuple of dimension names or None when the dimension is not found

        See also
        --------
        search_dim
        get_dim_type
        """
        # Check shape
        errors = ERRORS[errors]
        if len(dim_types) > da.ndim:
            msg = (f"this data array has less dimensions ({da.ndim})"
                   " than requested ({})".format(len(dim_types)))
            if errors == "raise":
                raise XoaError(msg)
            if errors == "warn":
                xoa_warn(msg)

        # Loop on types
        scanned = {}
        for dim_type in dim_types:
            scanned[dim_type] = self.search_dim(da, dim_type)

        # Guess from position
        if allow_positional:
            not_found = [item[0] for item in scanned.items()
                         if item[1] is None]
            for i, dim_type in enumerate(positions[::-1]):
                if dim_type in not_found:
                    scanned[dim_type] = da.dims[-i-1]

        # Final check
        if errors != 'ignore':
            for dim_type, dim in scanned.items():
                if dim is None:
                    msg = f"no dimension found of type '{dim_type}'"
                    if errors == 'raise':
                        raise XoaError(msg)
                    xoa_warn(msg)

        return tuple(scanned.values())

    def get_rename_dims_args(self, dsa, loc=None, specialize=False):
        """Get args for renaming dimensions that are not coordinates"""
        rename_args = {}
        for dim in dsa.dims:
            if dim in dsa.coords:
                continue
            dim_type = self.get_dim_type(dim, dsa)
            if dim_type:
                if specialize and self.dims[dim_type]:
                    new_name = self.dims[dim_type][0]
                else:
                    new_name = dim_type
                new_name = self.sglocator.merge_attr('name', dim, new_name, loc=loc)
                rename_args[dim] = new_name
        return rename_args


    def parse_dims(self, dims, dsa):
        """Convert from generic dim names to specialized names

        Parameters
        ----------
        dims: str, tuple, list, dict
        dsa: xarray.Dataset, xarray.DataArray

        Return
        ------
        Same type as dims
        """
        dim_types = self.get_dim_types(dsa, asdict=True)
        def _parse_dim_(dim):
            if dim not in dsa.dims:
                for dsadim, dim_type in dim_types.items():
                    if dim == dim_type:
                        dim = dsadim
                        break
            return dim

        if isinstance(dims, str):
            return _parse_dim_(dims)
        if isinstance(dims, dict):
            return dict((_parse_dim_(dim), value) for dim, value in dims.items())
        return type(dims)(_parse_dim_(dim) for dim in dims)


for meth in ('get_axis', 'get_dim_type', 'get_dim_types',
             'search_dim', 'get_dims'):
    doc = getattr(CFCoordSpecs, meth).__doc__
    getattr(CFSpecs, meth).__doc__ = doc


def _get_cfgm_():
    """Get a :class:`~xoa.cfgm.ConfigManager` instance to manage
    coords and data_vars spécifications"""
    cf_cache = _get_cache_()
    if "cfgm" not in cf_cache:
        from .cfgm import ConfigManager

        cf_cache["cfgm"] = ConfigManager(_INIFILE)
    return cf_cache["cfgm"]


def get_matching_item_specs(da, loc="any"):
    """Get the item CF specs that match this data array

    Parameters
    ----------
    da: xarray.DataArray

    Return
    ------
    dict or None

    See also
    --------
    CFSpecs.match
    """
    cfspecs = get_cf_specs(da)
    cat, name =  cfspecs.match(da, loc=loc)
    if cat:
        return cfspecs[cat][name]


def _same_attr_(da0, da1, attr):
    return (attr in da0.attrs and attr in da1.attrs and
            da0.attrs[attr].lower() == da1.attrs[attr].lower())


def are_similar(da0, da1):
    """Check if two DataArrays are similar

    Verifications are performed in the following order:

    - ``standard_name`` attribute,
    - Matching CFSpecs item name.
    - ``name`` attribute.
    - ``long_name`` attribute.

    Parameters
    ----------
    da0: xarray.DataArray
    da1: xarray.DataArray

    Return
    ------
    bool
    """
    # Standard name
    if _same_attr_(da0, da1, "standard_name"):
        return True

    # Cf name
    cf0 = get_matching_item_specs(da0)
    cf1 = get_matching_item_specs(da1)
    if (cf0 and cf1 and cf0.name == cf1.name):
        return True

    # Name
    if da0.name and da0.name and da0.name == da1.name:
        return True

    # Long name
    return _same_attr_(da0, da1, "long_name")


def search_similar(dsa, da):
    """Search in ds for a similar DataArray

    See :func:`is_similar` for what means "similar".

    Parameters
    ----------
    dsa: xarray.Dataset, xarray.DataArray
        Dataset that must be scanned.
    da: xarray.DataArray
        Array that must be compared to the content of ``ds``

    Return
    ------
    xarray.DataArray or None

    See also
    --------
    is_similar
    get_matching_item_specs
    """
    targets = list(da.coords.values())
    if hasattr(dsa, "data_vars"):
        targets = list(da.data_vars.values()) + targets
    for ds_da in targets:
        if are_similar(ds_da, da):
            return ds_da


class set_cf_specs(object):
    """Set the current CF specs

    Parameters
    ----------
    cf_source: CFSpecs, str, list, dict
        Either a :class:`CFSpecs` instance or the name of a registered one,
        or an argument to instantiante one.

    See also
    --------
    get_cf_specs
    register_cf_specs
    get_registered_cf_specs
    """

    def __init__(self, cf_source):
        if isinstance(cf_source, str):
            cfspecs = get_cf_specs_from_name(cf_source, errors="ignore")
            if cfspecs:
                cf_source = cfspecs
        if not isinstance(cf_source, CFSpecs):
            cf_source = CFSpecs(cf_source)
        self.cf_cache = _get_cache_()
        self.old_specs = self.cf_cache["current"]
        self.cf_cache["current"] = self.specs = cf_source

    def __enter__(self):
        return self.specs

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_specs is None:
            del self.cf_cache["current"]
        else:
            self.cf_cache["current"] = self.old_specs


def reset_cache(disk=True, memory=False):
    """Reset the on disk and/or in memory cf specs cache

    Parameters
    ----------
    disk: bool
        Remove the cf specs cahce file (:data:`USER_CF_CACHE_FILE`)
    memory: bool
        Remove the in-memory cache.

        .. warning:: This may lead to unpredicted behaviors.

    """
    if disk and os.path.exists(USER_CF_CACHE_FILE):
        os.remove(USER_CF_CACHE_FILE)

    if memory:
        cf_cache = _get_cfgm_()
        cf_cache["loaded_dicts"].clear()
        cf_cache["current"] = None
        cf_cache["default"] = None
        cf_cache["registered"].clear()


def show_cache():
    """Show the cf specs cache file"""
    print(USER_CF_CACHE_FILE)


@ERRORS.format_function_docstring
def get_cf_specs_from_name(name, errors="warn"):
    """Get a registered CF specs instance from its name

    Parameters
    ----------
    name: str
    {errors}

    Return
    ------
    CFSpecs or None
        Issue a warning if not found
    """
    cf_cache = _get_cache_()
    for cfspecs in cf_cache["registered"][::-1]:
        if cfspecs["register"]["name"] and cfspecs["register"]["name"] == name.lower():
            return cfspecs
    errors = ERRORS[errors]
    msg = f"Invalid registration name for CF specs: {name}"
    if errors == "raise":
        raise XoaCFError(msg)
    elif errors == "warn":
        xoa_warn(msg)


def get_cf_specs_encoding(ds):
    """Get the ``cfspecs`` encoding value

    Parameters
    ----------
    ds: xarray.DataArray, xarray.Dataset

    Return
    ------
    str or None

    See also
    --------
    get_cf_specs_from_encoding
    """
    if ds is not None and not isinstance(ds, str):
        for source in ds.encoding, ds.attrs:
            for attr, value in source.items():
                if attr.lower() == "cfspecs":
                    return value


def get_cf_specs_from_encoding(ds):
    """Get a registered CF specs instance from the ``cfspecs`` encoding value

    Parameters
    ----------
    ds: xarray.DataArray, xarray.Dataset

    Return
    ------
    CFSpecs or None

    See also
    --------
    get_cf_specs_encoding
    """
    if ds is not None and not isinstance(ds, str):
        name = get_cf_specs_encoding(ds)
        if name is not None:
            return get_cf_specs_from_name(name, errors="warn")


def get_default_cf_specs(cache="rw"):
    """Get the default CF specifications

    Parameters
    ----------
    cache: str, bool, None
        Cache default specs on disk with pickling for fast loading.
        If ``None``, it defaults to boolean option :xoaoption:`cf.cache`.
        Possible string values: ``"ignore"``, ``"rw"``, ``"read"``,
        ``"write"``, ``"clean"``.
        If ``True``, it is set to ``"rw"``.
        If ``False``, it is set to ``"ignore"``.
    """
    if cache is None:
        cache = get_option('cf.cache')
    if cache is True:
        cache = "rw"
    elif cache is False:
        cache = "ignore"
    assert cache in ("ignore", "rw", "read", "write", "clean")
    cf_cache = _get_cache_()
    if cf_cache["default"] is not None:
        return cf_cache["default"]
    cfspecs = None

    # Try from disk cache
    if cache in ("read", "rw"):
        if os.path.exists(USER_CF_CACHE_FILE) and (
            os.stat(_CFGFILE).st_mtime <
            os.stat(USER_CF_CACHE_FILE).st_mtime
        ):
            try:
                with open(USER_CF_CACHE_FILE, "rb") as f:
                    cfspecs = pickle.load(f)
            except Exception as e:
                xoa_warn(
                    "Error while loading cached cf specs: " + str(e.args)
                )

    # Compute it from scratch
    if cfspecs is None:

        # Setup
        cfspecs = CFSpecs()

        # Cache it on disk
        if cache in ("write", "rw"):
            try:
                cachedir = os.path.dirname(USER_CF_CACHE_FILE)
                if not os.path.exists(cachedir):
                    os.makedirs(cachedir)
                with open(USER_CF_CACHE_FILE, "wb") as f:
                    pickle.dump(cfspecs, f)
            except Exception as e:
                xoa_warn("Error while caching cf specs: " + str(e.args))

    cf_cache["default"] = cfspecs
    if not is_registered_cf_specs(cfspecs):
        register_cf_specs(cfspecs)
    return cfspecs


def get_cf_specs(name=None, cache="rw"):
    """Get the current or a registered CF specifications instance

    Parameters
    ----------
    name: str, "current", "default", None, xarray.Dataset, xarray.DataArray
        "default" means the default xoa specs.
        "current" is equivalent to None and means the currents specs,
        which defaults to the xoa defaults!
        Else registration name for these specs or a data array or dataset
        that can be used to get the registration name if it set in the
        :attr:`cfspecs` attribute or encoding.
        When set, ``cache`` is ignored.
        Raises a :class:`XoaCFError` is case of invalid name.
    cache: str, bool, None
        Cache default specs on disk with pickling for fast loading.
        If ``None``, it defaults to boolean option :xoaoption:`cf.cache`.
        Possible string values: ``"ignore"``, ``"rw"``, ``"read"``, ``"write"``.
        If ``True``, it is set to ``"rw"``.
        If ``False``, it is set to ``"ignore"``.

    Return
    ------
    CFSpecs
        None is return if no specs are found

    Raise
    -----
    XoaCFError
        When ``name`` is provided as a string and is invalid.
    """
    # Explicit request
    if name is None:
        name = "current"
    if not isinstance(name, str) or name not in ("current", "default"):

        # Registered name
        if isinstance(name, str):
           return get_cf_specs_from_name(name, errors="raise")

        # Name as dataset or data array so we guess the name
        cfspecs = get_cf_specs_from_encoding(name)
        if cfspecs:
            return cfspecs
        else:
            name = "current"

    # Not named => current or default specs
    if name == "current":
        cf_cache = _get_cache_()
        if cf_cache["current"] is None:
            cf_cache["current"] = get_default_cf_specs()
        cfspecs =  cf_cache["current"]
    else:
        cfspecs = get_default_cf_specs()

    return cfspecs


def register_cf_specs(*args, **kwargs):
    """Register :class:`CFSpecs` in a bank optionally with a name"""
    args = list(args)
    for name, cfspecs in kwargs.items():
        if not isinstance(cfspecs, CFSpecs):
            cfspecs = CFSpecs(cfspecs)
        cfspecs.name = name
        args.append(cfspecs)
    for cfspecs in args:
        if not isinstance(cfspecs, CFSpecs):
            cfspecs = CFSpecs(cfspecs)
        cf_cache = _get_cache_()
        if cfspecs not in cf_cache["registered"]:
            cf_cache["registered"].append(cfspecs)


def get_registered_cf_specs(current=True, reverse=True, named=False):
    """Get the list of registered CFSpecs

    Parameters
    ----------
    current: bool
        Also include the current specs if any, always at the last position
    reverse: bool
        Reverse the list
    named: bool
        Make sure the returned CFSpecs have a valid registration name

    Return
    ------
    list

    See also
    --------
    register_cf_specs
    """
    cf_cache = _get_cache_()
    cfl = cf_cache["registered"]
    if reverse:
        cfl = cfl[::-1]
    if current and cf_cache["current"] is not None:
        cfl.append(cf_cache["current"])
    if named:
        cfl = [c for c in cfl if c.name]
    return cfl


def is_registered_cf_specs(name):
    """Check if given cf specs set is registered

    Parameters
    ----------
    name: str, CFSpecs

    Return
    ------
    bool
    """
    for cfspecs in get_registered_cf_specs():
        if (isinstance(name, str) and cfspecs["register"]["name"] and
                cfspecs["register"]["name"] == name):
            return True
        if isinstance(name, CFSpecs) and name is cfspecs:
            return True
    return False


def get_cf_specs_matching_score(ds, cfspecs):
    """Get the matching score between ds data_vars and coord names and a CFSpecs instance names

    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
    cf_specs: CFSpecs

    Return
    ------
    float
        A percentage of the number of identified data arrays vs
        the total number of data arrays
    """
    hit = 0
    total = 0
    for cat in "data_vars", "coords":
        cfnames = [cfspecs[cat].get_name(name, specialize=True)
                   for name in cfspecs[cat].names]
        if not hasattr(ds, "data_vars"):  # DataArray
            dsnames = [ds.name] if ds.name else []
        else:
            dsnames = list(getattr(ds, cat).keys())
        dsnames = [cfspecs.sglocator.parse_attr("name", dsname)[0] for dsname in dsnames]
        total += len(dsnames)
        hit += len(set(dsnames).intersection(cfnames))
    if total == 0:
        return 0
    return 100 * hit / total


def infer_cf_specs(ds, named=False):
    """Get the registered CFSpecs that are best matching this dataset

    This accomplished with some heurestics.
    First, the :attr:`cfspecs` global attribute or encoding of the dataset is compared
    with the name of all registered datasets.
    Second, a score based on the number of data_vars and coord names
    that are both in the cfspecs and the dataset is computed by :func:`get_cf_specs_matching_score`
    for the registered instances.
    Finally, if no matching dataset is found, the current one is returned.


    Parameters
    ----------
    ds: xarray.Dataset, xarray.DataArray
    named: bool
        Make sure the candidate CFSpecs have a name

    Return
    ------
    CFSpecs
        The matching cf specs or the current ones

    See also
    --------
    register_cf_specs
    get_registered_cf_specs
    get_cf_specs_matching_score
    get_cf_specs
    get_cf_specs
    get_cf_specs_from_name
    get_cf_specs_from_encoding
    """
    # By registration name first
    cfspecs = get_cf_specs_from_encoding(ds)
    if cfspecs:
        return cfspecs

    # Candidates
    candidates = get_registered_cf_specs(named=named)

    # By attributes
    attrs = dict(ds.attrs)
    attrs.update(ds.encoding)
    if attrs:
        for cfspecs in candidates:
            for attr, pattern in cfspecs["register"]["attrs"].items():
                if attr in attrs and fnmatch.fnmatch(str(attrs[attr]).lower(), pattern.lower()):
                    return cfspecs

    # By matching score
    best_score = -1
    for cfspecs in candidates:
        score = get_cf_specs_matching_score(ds, cfspecs)
        if score != 0 and score > best_score:
            best_cfspecs = cfspecs
            best_score = score
    if best_score != -1:
        return best_cfspecs

    # Fallback to default specs
    cfspecs = get_cf_specs()
    if named and not cfspecs.name:
        return
    return cfspecs


def assign_cf_specs(ds, name=None, register=False):
    """Set the ``cfspecs`` encoding to ``name`` in all data vars and coords

    Parameters
    ----------
    ds: xarray.DataArray, xarray.Dataset
    name: None, str, CFSpecs, xarray.DataArray, xarray.Dataset
        If a :class:`CFSpecs`, it must have a registration name :

        .. code-block:: ini

            [register]
            name=registration_name

        If not provided, :func:`infer_cf_specs` is called to infer
        the best named registered specs.

    register: bool
        Register the specs if name is a named, unregistered :class:`CFSpecs` instance.

    Return
    ------
    xarray.Dataset, xarray.DataArray

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.cf import assign_cf_specs
        @suppress
        import xarray as xr
        ds = xr.Dataset({'temp': ('lon', [5])}, coords={'lon': [6]})
        assign_cf_specs(ds, "mycroco");
        ds.encoding
        ds.temp.encoding
        ds.lon.encoding

    """
    # Name as a CFSpecs instance
    if name is None:
        cfspecs = infer_cf_specs(ds, named=True)
        if cfspecs.name:
            name = cfspecs.name
        else:
            return ds
    elif hasattr(name, "coords"):  # from a dataset/dataarray
        name = get_cf_specs_encoding(ds)
        if name is None:
            return ds
    if not isinstance(name, str):
        if not name.name:
            xoa_warn("CFSpecs instance has no registration name")
            return ds
        if register and not is_registered_cf_specs(name):
            register_cf_specs(name)
        name = name.name

    # Set as encoding
    targets = [ds] + list(ds.coords.values())
    if hasattr(ds, "data_vars"):
        targets.extend(list(ds.data_vars.values()))
    for target in targets:
        target.encoding.update(cfspecs=name)
    return ds


def infer_coords(ds):
    return get_cf_specs(ds).infer_coords(ds)


infer_coords.__doc__ = CFSpecs.infer_coords.__doc__