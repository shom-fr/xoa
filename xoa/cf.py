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

from .__init__ import XoaError, xoa_warn, get_option, __version__
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
    unique=True,
)

ATTRS_PATCH_MODE = Choices(
    {
        'fill': 'do not erase existing attributes, just fill missing ones',
        'replace': 'replace existing attributes',
    },
    parameter="mode",
    description="policy for patching existing attributes",
)


class XoaCFError(XoaError):
    pass


def _get_cache_():
    from . import cf

    if not hasattr(cf, "_CF_CACHE"):
        cf._CF_CACHE = {
            "current": None,  # current active specs
            "default": None,  # default xoa specs
            "loaded_dicts": {},  # for pure caching of dicts by key
            "registered": [],  # for registration and matching purpose
        }
    return cf._CF_CACHE


def _compile_sg_match_(re_match, attrs, formats, root_patterns, location_pattern):
    for attr in attrs:
        root_pattern = root_patterns[attr]
        re_match[attr] = re.compile(
            formats[attr].format(
                root=rf"(?P<root>{root_pattern})", loc=rf"(?P<loc>{location_pattern})"
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

    root_patterns = {"name": r"\w+", "standard_name": r"\w+", "long_name": r"[\w ]+"}

    location_pattern = "[a-z]+"

    re_match = {}
    _compile_sg_match_(re_match, valid_attrs, formats, root_patterns, location_pattern)

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
                    raise XoaCFError(
                        "name_format must contain strings " "{root} and {loc}: " + name_format
                    )
            if len(name_format) == 10:
                xoa_warn(
                    'No separator found in "name_format" and '
                    'and no "valid_locations" specified: '
                    f'{name_format}. This leads to ambiguity during '
                    'regular expression parsing.'
                )
            self.formats["name"] = name_format
            to_recompile.add("name")
        if self.valid_locations:
            self.location_pattern = "|".join(self.valid_locations)
            to_recompile.update(("name", "standard_name", "long_name"))
        _compile_sg_match_(
            self.re_match, to_recompile, self.formats, self.root_patterns, self.location_pattern
        )

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

    @ERRORS.format_method_docstring
    def get_loc(self, name=None, attrs=None, errors="warn"):
        """Parse name and attributes to find a unique location

        Parameters
        ----------
        name: None, str
            Name to parse
        attrs: None, dict
            Dict with `standard_name` and/or `long_name` attributes.
        {errors}


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
        errors = ERRORS[errors]
        src = []

        # Name
        if name:
            loc = self.parse_attr("name", name)[1]
            src.append("name")
        if not attrs:
            return loc

        # Standard name
        if "standard_name" in attrs and attrs["standard_name"]:
            standard_name = attrs["standard_name"]
            if isinstance(standard_name, list):
                standard_name = standard_name[0]
            loc_ = self.parse_attr("standard_name", standard_name)[1]
            if loc_:
                if not loc or loc_ == loc:
                    loc = loc_
                    src.append("standard_name")
                elif errors != "ignore":
                    msg = (
                        "The location parsed from standard_name attribute "
                        f"[{loc_}] conflicts the location parsed from the "
                        f"name [{loc}]"
                    )
                    if errors == "raise":
                        raise XoaCFError(msg)
                    else:
                        xoa_warn(msg)

        # Long name
        if "long_name" in attrs and attrs["long_name"]:
            long_name = attrs["long_name"]
            if isinstance(long_name, list):
                long_name = long_name[0]
            loc_ = self.parse_attr("long_name", long_name)[1]
            if loc_:
                if not loc or loc_ == loc:
                    loc = loc_
                elif errors != "ignore":
                    src = ' and '.join(src)
                    msg = (
                        "The location parsed from long_name attribute "
                        f"[{loc_}] conflicts the location parsed from the "
                        f"{src} [{loc}]"
                    )
                    if errors == "raise":
                        raise XoaCFError(msg)
                    else:
                        xoa_warn(msg)

        return loc

    @ERRORS.format_method_docstring
    def get_loc_from_da(self, da, errors="warn"):
        """Parse a data array name and attributes to find a unique location

        Parameters
        ----------
        da: xarray.DataArray
            Data array to scan
        {errors}

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
        return self.get_loc(name=da.name, attrs=da.attrs, errors=errors)

    def parse_loc_arg(self, loc):
        """Parse the location argument

        Return values as function of input values:

            * `None`: None, True, "any"
            * `False`: `False`, ""
            * str: str
        """
        if loc is None or loc is True or loc == "any":
            return
        if loc is False or loc == "":
            return False
        if not isinstance(loc, str):
            raise XoaCFError(
                'Invalid loc argument. Must one of: '
                'None, Trye, "any", False, "" or a location string'
            )
        if self.valid_locations is not None and loc not in self.valid_locations:
            raise XoaCFError(
                f'Location "{loc}" is not recognised by the currents specifications. '
                'Registered locations are: ' + ', '.join(self.valid_locations)
            )
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
        loc = self.parse_loc_arg(loc)
        if loc is None:  # any loc
            return True
        if not loc:  # explicit no loc
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
            if a non empty str, it is set;
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
        loc = self.parse_loc_arg(loc)
        if loc is False:
            return root
            # loc = None
        if loc is None:
            loc = ploc
        # if (
        #     loc
        #     and self.valid_locations
        #     and loc.lower() not in self.valid_locations
        # ):
        #     raise XoaCFError(
        #         "Invalid location: {}. Please one use of: {}.".format(
        #             loc, ', '.join(self.valid_locations))
        #     )
        # elif loc is False or loc == '':
        #     ploc = None
        # elif loc is True or loc is None:
        #     loc = ploc
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
                    value = [self.format_attr(attr, v, loc, standardize=standardize) for v in value]
                else:
                    value = self.format_attr(attr, value, loc, standardize=standardize)
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

        loc = self.parse_loc_arg(loc)
        if loc is None:
            loc0 = self.parse_attr(attr, value0)[1]
            loc1 = self.parse_attr(attr, value1)[1]
            if loc1 is not None:
                loc = loc1
            elif loc0 is not None:
                loc = loc0
        return self.format_attr(attr, value1, loc, standardize=standardize)

    def patch_attrs(self, attrs, patch, loc=None, standardize=True, replace=False):
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
            attrs.update(self.format_attrs(attrs, loc, standardize=standardize))

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
                            if self.match_attr(attr, attrs[attr], val, loc=None):
                                value = attrs[attr]  # don't change, it's ok
                                break
                        else:
                            value = value[0]
                    else:
                        value = value[0]

                # Location
                value = self.merge_attr(
                    attr, attrs.get(attr, None), value, loc, standardize=standardize
                )

            if value is not None:
                attrs[attr] = value

        return attrs

    def format_dataarray(
        self,
        da,
        loc=None,
        standardize=True,
        name=None,
        attrs=None,
        rename=True,
        copy=True,
        replace_attrs=False,
        add_loc_to_name=True,
        add_loc_to_attrs=True,
    ):
        """Format name and attributes of a copy of DataArray

        Parameters
        ----------
        da: xarray.DataArray
        loc: {True, None}, str, {False, ""}
            If None, location is left unchanged;
            if a string, it is set;
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

        # Location argument
        loc = self.parse_loc_arg(loc)  # specified
        if loc is None:
            loc0 = self.get_loc_from_da(da)  # parsed from
            loc1 = self.get_loc(name=name, attrs=attrs)  # parsed from specs
            loc = loc0 if loc1 is None else loc1

        # Attributes
        kwloc = {"loc": loc if add_loc_to_attrs else False}
        if attrs:
            da.attrs.update(
                self.patch_attrs(
                    da.attrs, attrs, standardize=standardize, replace=replace_attrs, **kwloc
                )
            )
        else:
            da.attrs.update(self.format_attrs(da.attrs, standardize=standardize, **kwloc))

        # Name
        if rename:
            kwloc = {"loc": loc if add_loc_to_name else False}
            if da.name:  # change the name
                da.name = self.merge_attr("name", da.name, name, **kwloc)
            else:  # set it
                da.name = self.format_attr("name", name, **kwloc)

        # # Check location consistency
        # loc = self.parse_loc_arg(loc)
        # if loc is None:
        #     loc = self.get_loc(da)
        #     if loc:
        #         da = self.format_dataarray(
        #             da, loc=loc, rename=True, replace_attrs=True,
        #             add_loc_to_name=add_loc_to_name)

        return da

    def add_loc(self, da, loc, to_name=True, to_attrs=True):
        """A shortcut to :meth:`format_dataarray` to update `da` with loc without copy"""
        return self.format_dataarray(
            da,
            loc,
            copy=False,
            replace_attrs=True,
            add_loc_to_name=to_name,
            add_loc_to_attrs=to_attrs,
        )


def _solve_rename_conflicts_(rename_args):
    """Skip renaming items that overwride previous items"""
    used = {}
    for old_name in list(rename_args):
        new_name = rename_args[old_name]
        if old_name == new_name:
            del rename_args[old_name]
            continue
        if new_name in used:
            del rename_args[old_name]
            xoa_warn(
                f"Cannot rename {old_name} to {new_name} since "
                f"{used[new_name]} will also be renamed to {new_name}. Skipping..."
            )
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
        cache = cache and (
            (isinstance(cfg, str) and '\n' not in cfg)
            or (isinstance(cfg, dict) and "register" in cfg and cfg["register"]["name"])
        )
        if cache:

            # Init cache
            if isinstance(cfg, str):
                cache_key = cfg
            elif isinstance(cfg, dict) and "register" in cfg and cfg["register"]["name"]:
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
            "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
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
            assert from_cat != category or name != from_name, "Cannot inherit cf specs from it self"

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
                    if (
                        key not in self._cfgspecs[category]["__many__"]
                        and key in self._cfgspecs[from_cat]["__many__"]
                    ):
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
        SGLocator.get_loc
        """
        return self.sglocator.get_loc(da)

    get_location = get_loc

    def get_loc_mapping(self, obj, cf_names=None, categories=["coords", "data_vars"]):
        """Associate a location to each identified variables, coordinates and dimensions of obj

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
        cf_names: dict, None
            Dict with names as keys and generic CF names as values.
            If not provided, :meth:`match` is used to guess CF names.

        Return
        ------
        dict
            Keys are item names and values are location.
            This dict is also stored in the `cf_locs` key of the `encoding`
            attribute of `obj`.
        """
        # # Check cache
        # if "cf_locs" in obj.encoding:
        #     return obj.encoding["cf_locs"]

        locations = {}
        das = obj.values() if hasattr(obj, "data_vars") else [obj]

        def check_coords(da, specs, locations):
            """Scan the add_coords_loc section and the coordinates and dimensions"""
            for cf_coord_name, coord_loc in specs["add_coords_loc"].items():
                if self.coords[cf_coord_name]["attrs"]["axis"].lower() not in 'xyz':
                    continue

                loc = specs["loc"] if coord_loc is True else coord_loc

                # Coordinates
                coord = self.search_coord(da, cf_coord_name, errors="ignore")
                if coord is not None and locations.get(coord.name) is None:
                    locations[coord.name] = loc
                    continue

                # Dimensions
                dim = self.search_dim(da, cf_coord_name, errors="ignore")
                if dim is not None and locations.get(dim) is None:
                    locations[dim] = loc

        # Loop on data vars
        for da in das:
            for cat in categories:
                cf_name = cf_names.get(da.name) if cf_names else self[cat].match(da)
                if cf_name and cf_name in self[cat]:
                    specs = self[cat][cf_name]
                    if specs["add_loc"] is not False:
                        if specs["loc"] is None:  # infer from da
                            locations[da.name] = self.sglocator.get_loc_from_da(da)
                            if locations[da.name] is None and specs["add_loc"] is True:
                                locations[da.name] = True
                        else:
                            locations[da.name] = specs["loc"]  # from config
                    else:
                        locations[da.name] = False
                    check_coords(da, specs, locations)  # check coords from config
                    break  # good category
            else:
                continue

        # Loop on coordinates
        for coord in obj.coords.values():
            cf_coord_name = cf_names.get(coord.name) if cf_names else self.coords.match(coord)
            if (cf_coord_name and
                    (self.coords[cf_coord_name]["attrs"]["axis"] or "").lower() in 'xyz'):
                if locations.get(coord.name) is None:
                    if self.coords[cf_coord_name]["add_loc"] is not False:
                        if self.coords[cf_coord_name]["loc"] is None:
                            # infer from da
                            locations[coord.name] = self.sglocator.get_loc_from_da(coord)
                            if (locations[coord.name] is None and
                                    self.coords[cf_coord_name]["add_loc"] is True):
                                locations[coord.name] = True
                        else:
                            # from config
                            locations[coord.name] = self.coords[cf_coord_name]["loc"]
                    else:
                        locations[coord.name] = False
                check_coords(coord, self.coords[cf_coord_name], locations)

        # Loop again on data vars: if loc is True for an datarray and loc is unique
        # for all dims, it is set to the array too
        for da in das:
            if locations.get(da.name) is True:
                loc = None
                for dim in list(da.dims) + list(da.coords):
                    dim_loc = locations.get(dim)
                    if dim_loc is not None:
                        if loc is None:
                            loc = dim_loc
                        elif loc != dim_loc:
                            break  # multiple locs so no loc
                else:
                    locations[da.name] = loc

        # # Store in encoding
        # obj.encoding["cf_locs"] = locations
        # for name, loc in locations.items():
        #     if name in obj.coords or name not in obj.dims:
        #         obj[name].encoding["cf_loc"] = loc

        return locations

    def _format_obj_(
        self,
        obj,
        cf_names=None,
        rename=True,
        standardize=True,
        format_coords=True,
        copy=True,
        replace_attrs=False,
        attrs=True,
        # loc=None, add_loc_to_name=None, add_loc_to_coord_names=None,
        specialize=False,
        rename_dims=True,
        categories=["coords", "data_vars"],
    ):
        """Auto-format a whole xarray.Dataset

        See also
        --------
        format_data_var
        format_coord
        """
        # Copy
        if copy:
            obj = obj.copy()

        # Init rename dict
        rename_args = {}

        # Common formatting kwargs
        kwargs = dict(
            copy=False,
            rename=False,
            replace_attrs=replace_attrs,
            standardize=True,
            specialize=specialize,
        )

        # Staggared grid locations
        locations = self.get_loc_mapping(obj, cf_names=cf_names, categories=categories)

        # Data arrays
        is_dataset = hasattr(obj, "data_vars")
        if is_dataset:  # dataset
            data_vars = obj.values()
        else:
            data_vars = [obj]
        for da in data_vars:
            for cat in categories:
                cf_name = cf_names.get(da.name) if cf_names else None
                if cf_name and cf_name not in self[cat]:
                    continue
                new_name = self[cat].format_dataarray(
                    da,
                    cf_name=cf_name,
                    attrs=attrs if isinstance(attrs, bool) else attrs.get(da.name),
                    # format_coords=False,
                    loc=locations.get(da.name),
                    **kwargs,
                )
                if new_name:
                    break
            else:
                new_name = None
            if rename and new_name:
                if is_dataset:
                    rename_args[da.name] = new_name
                else:
                    da.name = new_name

        # Coordinates
        if format_coords:
            for cname, cda in list(obj.coords.items()):
                new_coord_name = self.coords.format_dataarray(
                    cda,
                    cf_name=cf_names.get(cname) if isinstance(cf_names, dict) else None,
                    attrs=attrs if isinstance(attrs, bool) else attrs.get(cname, True),
                    # related=obj,
                    # format_coords=False,
                    # loc=loc, add_loc_to_name=add_loc_to_coord_names,
                    loc=locations.get(cda.name),
                    # rename_dim=False,
                    **kwargs,
                )
                if rename and new_coord_name:
                    rename_args[cda.name] = new_coord_name

        # Dimensions
        if rename_dims:
            rename_dims_args = self.coords.get_rename_dims_args(
                obj, locations=locations, specialize=specialize
            )  # , exclude=list(rename_args.keys()))
            rename_args.update(rename_dims_args)

        # Final renaming
        if rename and rename_args:
            _solve_rename_conflicts_(rename_args)
            obj = obj.rename(rename_args)

        return obj

    def format_coord(
        self,
        da,
        cf_name=None,
        # loc=None,
        copy=True,
        format_coords=True,
        standardize=True,
        rename=True,
        rename_dim=True,
        specialize=False,
        attrs=True,
        replace_attrs=False,
        # add_loc_to_name=None,
        rename_dims=True,
    ):
        """Format a coordinate array

        Parameters
        ----------
        da: xarray.DataArray
        cf_name: str, None
            A generic CF name. If not provided, it guessed with :meth:`match`.
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
        return self._format_obj_(
            da,
            cf_names={da.name: cf_name},
            copy=copy,
            standardize=standardize,
            rename=rename,
            rename_dims=rename_dims,
            format_coords=format_coords,
            replace_attrs=replace_attrs,
            attrs=attrs if isinstance(attrs, bool) else {da.name: attrs},
            specialize=specialize,
            categories=["coords"],
        )

    def format_data_var(
        self,
        da,
        cf_name=None,
        # loc=None,
        copy=True,
        rename=True,
        rename_dims=True,
        specialize=False,
        format_coords=True,
        attrs=True,
        replace_attrs=False,
        standardize=True,
        # add_loc_to_name=None,
    ):
        """Format a data_var array

        Parameters
        ----------
        da: xarray.DataArray
        cf_name: str, None
            A generic CF name. If not provided, it guessed with :meth:`match`.
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
        rename_dims: bool
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
        return self._format_obj_(
            da,
            cf_names={da.name: cf_name},
            copy=copy,
            rename=rename,
            rename_dims=rename_dims,
            specialize=specialize,
            format_coords=format_coords,
            replace_attrs=replace_attrs,
            attrs=attrs if isinstance(attrs, bool) else {da.name: attrs},
            standardize=standardize,
            categories=["coords", "data_vars"],
        )

    def format_dataset(
        self,
        ds,
        cf_names=None,
        # loc=None,
        copy=True,
        format_coords=True,
        standardize=True,
        rename=True,
        rename_dims=True,
        specialize=False,
        attrs=True,
        replace_attrs=False,
        # add_loc_to_name=None
    ):
        """Format a whole dataset

        Parameters
        ----------
        ds: xarray.Dataset
        cf_names: dict, None
            Dict of names as keys and generic CF names as values.
            If not provided, CF names are guessed with :meth:`match`.
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
        attrs: bool, dict of dict
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
        return self._format_obj_(
            ds,
            cf_names=cf_names,
            copy=copy,
            standardize=standardize,
            rename=rename,
            rename_dims=rename_dims,
            format_coords=format_coords,
            replace_attrs=replace_attrs,
            attrs=attrs,
            specialize=specialize,
            categories=["coords", "data_vars"],
        )

    def auto_format(self, obj, **kwargs):
        """Rename variables and coordinates and fill their attributes

        See also
        --------
        encode
        format_data_var
        format_dataset
        fill_attrs
        """
        if hasattr(obj, "data_vars"):
            return self.format_dataset(obj, **kwargs)
        return self.format_data_var(obj, **kwargs)

    def decode(self, obj, **kwargs):
        """Auto format, infer coordinates and rename to generic names

        See also
        --------
        auto_format
        encode
        format_dataarray
        format_dataset
        fill_attrs
        """
        # Coordinates
        obj = self.infer_coords(obj)

        # Names and attributes
        obj = self.auto_format(obj, **kwargs)

        # Assign cf specs
        if self.name and self in get_registered_cf_specs():
            obj = assign_cf_specs(obj, self.name)

        return obj

    def encode(self, obj, **kwargs):
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
        return self.decode(obj, **kwargs)

    def to_loc(self, obj, **locs):
        """Set the staggered grid location for specified names

        .. note:: It only changes the names, not the attributes.

        Parameters
        ----------
        obj: xarray.Dataset, xarray.DataArray
        locs: dict
            **Keys are root names**, values are new locations.
            A value of `False`, remove the location.
            A value of `None` left it as is.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        reloc
        sglocator
        SGLocator.format_attr
        """
        rename_args = {}
        names = set(obj.dims).union(obj.coords)
        if hasattr(obj, "data_vars"):
            names = names.union(obj.data_vars)
        for name in names:
            if name not in rename_args:
                root_name, old_loc = self.sglocator.parse_attr("name", name)
                if root_name in locs and locs[root_name] is not None:
                    rename_args[name] = self.sglocator.format_attr(
                        "name", root_name, locs[root_name])
        return obj.rename(rename_args)

    def reloc(self, obj, **locs):
        """Convert given staggered grid locations to other locations

        .. note:: It only changes the names, not the attributes.

        Parameters
        ----------
        obj: xarray.Dataset, xarray.DataArray
        locs: dict
            **Keys are locations**, values are new locations.
            A value of `False` or `None`, remove the location.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        to_loc
        sglocator
        SGLocator.format_attr
        """
        rename_args = {}
        names = set(obj.dims).union(obj.coords)
        if hasattr(obj, "data_vars"):
            names = names.union(obj.data_vars)
        for name in names:
            if name not in rename_args:
                root_name, old_loc = self.sglocator.parse_attr("name", name)
                if old_loc and old_loc in locs:
                    rename_args[name] = self.sglocator.format_attr(
                        "name", root_name, locs[old_loc])

        return obj.rename(rename_args)

    def fill_attrs(self, obj, **kwargs):
        """Fill missing attributes of a xarray.Dataset or xarray.DataArray

        .. note:: It does not rename anything

        See also
        --------
        format_dataarray
        format_dataset
        auto_format
        """
        kwargs.update(rename=False, replace_attrs=False)
        if hasattr(obj, "data_vars"):
            return self.format_dataset(obj, format_coords=True, **kwargs)
        return self.format_data_var(obj, **kwargs)

    def match_coord(self, da, cf_name=None, loc="any"):
        """Check if an array matches a given or any coord specs

        Parameters
        ----------
        da: xarray.DataArray
        cf_name: str, dict, None
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
        return self.coords.match(da, cf_name=cf_name, loc=loc)

    def match_data_var(self, da, cf_name=None, loc="any"):
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
        return self.data_vars.match(da, cf_name=cf_name, loc=loc)

    def match_dim(self, dim, cf_name=None, loc=None):
        """Check if a dimension name matches given or any coord specs

        Parameters
        ----------
        dim: str
            Dimension name
        cf_name: str, None
            Cf name.
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
        CFVarSpecs.match_from_name
        """
        return self.coords.match_from_name(dim, cf_name=cf_name, loc=loc)

    @staticmethod
    def get_category(da):
        """Guess if a dataarray belongs to data_vars or coords

        It belongs to coords if one of its dimensions or
        coordinates has its name.

        Parameters
        ----------
        da: xarray.DataArray

        Returns
        -------
        str
        """
        if da.name is not None and (da.name in da.dims or da.name in da.coords):
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
            cf_name = self[category].match(da, loc=loc)
            if cf_name:
                return category, cf_name
        return None, None

    @ERRORS.format_method_docstring
    def search_coord(self, obj, cf_name=None, loc="any", get="obj", single=True, errors="warn"):
        """Search for a coord that maches given or any specs

        Parameters
        ----------
        obj: DataArray or Dataset
        cf_name: str, dict
            A generic CF name. If not provided, all CF names are scaned.
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
            cfspecs.search_coord(data, "lon", get="cf_name")
            cfspecs.search_coord(data, "lat", errors="ignore")

        See also
        --------
        search_data_var
        CFCoordSpecs.search
        """
        return self.coords.search(
            obj, cf_name=cf_name, loc=loc, get=get, single=single, errors=errors
        )

    @ERRORS.format_method_docstring
    def search_dim(self, da, cf_arg=None, loc="any", errors="ignore"):
        """Search for a dimension from its type

        Parameters
        ----------
        da: xarray.DataArray
        cf_arg: None, str, {{"x", "y", "z", "t", "f"}}
            Dimension type or generic CF name
        loc:
            Staggered grid location
        {errors}

        Return
        ------
        None, str, dict
            Dim name OR, dict with dim, type and cf_name keys if cf_arg is None

        See also
        --------
        CFCoordSpecs.search_dim
        """
        return self.coords.search_dim(da, cf_arg=cf_arg, loc=loc, errors=errors)

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
    def search_data_var(self, obj, cf_name=None, loc="any", get="obj", single=True, errors="warn"):
        """Search for a data_var that maches given or any specs

        Parameters
        ----------
        obj: DataArray or Dataset
        cf_name: str, dict
            A generic CF name. If not provided, all CF names are scaned.
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
            cfspecs.search_data_var(ds, "temp", get="cf_name")
            cfspecs.search_data_var(ds, "sal")

        See also
        --------
        search_coord
        CFVarSpecs.search
        """
        return self.data_vars.search(
            obj, cf_name=cf_name, loc=loc, get=get, single=single, errors=errors
        )

    @ERRORS.format_method_docstring
    def search(
        self, obj, cf_name=None, loc="any", get="obj", single=True, categories=None, errors="warn"
    ):
        """Search for a dataarray with data_vars and/or coords


        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array or dataset to scan
        cf_name: str
            Generic CF name to search for.
        categories: str, list, None
            Explicty categories with "coords" and "data_vars".
        {errors}

        Return
        ------
        None, xarray.DataArray, list

        """
        if not categories:
            categories = self.categories if hasattr(obj, "data_vars") else ["coords", "data_vars"]
        elif isinstance(categories, str):
            categories = [categories]
        else:
            categories = self.categories
        errors = ERRORS[errors]
        if not single:
            found = []
        for category in categories:
            res = self[category].search(
                obj, cf_name=cf_name, loc=loc, get=get, single=single, errors="ignore"
            )
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

    def get(self, obj, cf_name, get="obj"):
        """A shortcut to :meth:`search` with an explicit generic CF name

        A single element is searched for into all :attr:`categories`
        and errors are ignored.
        """
        return self.search(obj, cf_name, errors="ignore", single=True, get=get)
        # if da is None:
        #     raise XoaCFError("Search failed for the following cf name: "
        #                      + name)
        # return da

    @ERRORS.format_method_docstring
    def get_dims(self, da, cf_args, allow_positional=False, positions='tzyx', errors="warn"):
        """Get the data array dimensions names from their type

        Parameters
        ----------
        da: xarray.DataArray
            Array to scan
        cf_args: str, list
            Letters among "x", "y", "z", "t" and "f",
            or generic CF names.
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
            da, cf_args, allow_positional=allow_positional, positions=positions, errors=errors
        )

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
        return self.coords.get_axis(coord, lower=lower)

    def get_dim_type(self, dim, da=None, lower=True):
        """Get the type of a dimension

        Three cases:

        - This dimension is registered in CF dims.
        - obj has dim as dim and has an axis attribute inferred with :meth:`get_axis`.
        - obj has a coord named dim with an axis attribute inferred with :meth:`get_axis`.

        Parameters
        ----------
        dim: str
            Dimension name
        obj: xarray.DataArray, xarray.Dataset
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
        return self.coords.get_dim_type(dim, da=da, lower=lower)

    def get_dim_types(self, da, unknown=None, asdict=False):
        """Get a tuple of the dimension types of an array

        Parameters
        ----------
        obj: xarray.DataArray, tuple(str), xarray.Dataset
            Data array, dataset or tuple of dimensions
        unknown:
            Value to assign when type is unknown
        asdict: bool

        Return
        ------
        tuple, dict
            Tuple of dimension types and of length ``obj.ndim``.
            A dimension type is either a letter or the ``unkown`` value
            when the inference of type has failed.
            If ``asdict`` is True, a dict is returned instead,
            ``(dim, dim_type)`` as key-value pairs.

        See also
        --------
        get_dim_type
        """
        return self.coords.get_dim_types(da, unknown=unknown, asdict=asdict)

    def parse_dims(self, dims, obj):
        """Convert from generic dim names to specialized names

        Parameters
        ----------
        dims: str, tuple, list, dict
        obj: xarray.Dataset, xarray.DataArray

        Return
        ------
        Same type as dims
        """
        return self.coords.parse_dims(dims, obj)

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
    def coords(self):
        return self.parent.coords

    @property
    def data_vars(self):
        return self.parent.data_vars

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
                raise XoaCFError(f"Invalid {self.category} CF specs name: " + name)
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
        errors: "ignore", "warning" or "raise".

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
        """Dict of dim names per type"""
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

    def get_allowed_names(self, cf_name):
        """Get the list of allowed names for a given `cf_name`

        It includes de `cf_name` itself, the `name` alt_names` specification values

        Parameters
        ----------
        cf_name: str
            Valid CF name

        Return
        ------
        list
        """
        specs = self[cf_name]
        allowed_names = [cf_name]
        if "name" in specs and specs["name"]:
            allowed_names.append(specs["name"])
        if "alt_names" in specs:
            allowed_names.extend(specs["alt_names"])
        return allowed_names

    def get_loc_mapping(self, obj, cf_names=None):
        return self.parent.get_loc_mapping(
            obj, cf_names=cf_names, categories=["coords", "data_vars"]
        )

    def _get_ordered_match_specs_(self, cf_name):
        specs = self[cf_name]
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
                    match_specs["name"] = self.get_allowed_names(cf_name)
                elif "attrs" in specs and attr in specs["attrs"]:
                    match_specs[attr] = specs["attrs"][attr]
        return match_specs

    def match(self, da, cf_name=None, loc=None):
        """Check if da attributes match given or any specs

        Parameters
        ----------
        da: xarray.DataArray
        cf_name: str, dict, None
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
        if cf_name:
            if isinstance(cf_name, str):
                self._assert_known_(cf_name)
            names = [cf_name]
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
                        and self.sglocator.match_attr(attr, value, ref, loc=loc)
                    ) or match_string(value, ref, ignorecase=True):
                        da.encoding["cf_name"] = name_
                        da.encoding["cf_category"] = self.category
                        return True if cf_name else name_
        return False if cf_name else None

    def match_from_name(self, name, cf_name=None, loc=None):
        """Get the generic CF name of an object knowing only its name

        It compares `name` to the `name` and `alt_names` config values.

        Parameters
        ----------
        name: str
            Actual name
        cf_name: str, None
            A target generic CF name. If not provided, all items are considered.

        Return
        ------
        None, str, True, False
            If `cf_name` is provided, returns a boolean, else returns
            the matching CF name or None.
        """
        explicit = cf_name is not None
        if cf_name:
            self._assert_known_(cf_name)
            cf_names = [cf_name]
        else:
            cf_names = self.names

        for cf_name in cf_names:
            for allowed_name in self.get_allowed_names(cf_name):
                if self.sglocator.match_attr("name", name, allowed_name, loc=loc):
                    return cf_name if not explicit else True
        return None if not explicit else False

    @ERRORS.format_method_docstring
    def search(self, obj, cf_name=None, loc=None, get="obj", single=True, errors="raise"):
        """Search for a data_var or coord that maches given or any specs

        Parameters
        ----------
        obj: DataArray or Dataset
        cf_name: str, dict
            A generic CF name. If not provided, all CF names are scaned.
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {{"obj", "cf_name", "both"}}
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
        if self.category and hasattr(obj, self.category):
            objs = getattr(obj, self.category)
        else:
            objs = obj.keys() if hasattr(obj, "keys") else obj.coords

        # Get match specs
        if cf_name:  # Explicit name so we loop on search specs
            if isinstance(cf_name, str):
                if not self._assert_known_(cf_name, errors):
                    return
            match_specs = []
            for attr, refs in self._get_ordered_match_specs_(cf_name).items():
                match_specs.append({attr: refs})
        else:
            match_specs = [None]

        # Loops
        assert get in (
            "cf_name",
            "obj",
            "both",
        ), f"'get' must be either 'cf_name' or 'obj' or 'both', NOT '{get}'"
        found = []
        found_objs = []
        for match_arg in match_specs:
            for obj in objs.values():
                m = self.match(obj, match_arg, loc=loc)
                if m:
                    if obj.name in found_objs:
                        continue
                    found_objs.append(obj.name)
                    cf_name = cf_name if cf_name else m
                    if get == "both":
                        found.append((obj, cf_name))
                    else:
                        found.append(obj if get == "obj" else cf_name)

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

    def get(self, obj, cf_name):
        """Call to :meth:`search` with an explicit name and ignoring errors"""
        return self.search(obj, cf_name, errors="ignore")

    @ERRORS.format_method_docstring
    def get_attrs(
        self,
        cf_name,
        select=None,
        exclude=None,
        errors="warn",
        loc=None,
        multi=False,
        standardize=True,
        **extra,
    ):
        """Get the default attributes from cf specs

        Parameters
        ----------
        cf_name: str
            Valid generic CF name
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
        specs = self.get_specs(cf_name, errors=errors)
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
        attrs = self.parent.sglocator.format_attrs(attrs, loc=loc, standardize=standardize)

        return attrs

    def get_name(self, name, specialize=False, loc=None):
        """Get the name of the matching CF specs

        Parameters
        ----------
        name: str, xarray.DataArray
            Either a data array, a known cf name or a data var name
        specialize: bool
            Get the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.
        loc: str, None
            Format it at this location

        Return
        ------
        None or str
            Either the CF name or the specialized name
        """
        if not isinstance(name, str):
            name = name.encoding.get("cf_name", self.match(name))
            # FIXME: category?
        elif name not in self:
            name = self.match_from_name(name)
        if name is None:
            return
        if specialize and self[name]["name"]:
            name = self[name]["name"]
        return self.sglocator.format_attr("name", name, loc=loc)

    def get_loc_arg(self, da, cf_name=None, locations=None):
        """Get the `loc` argument from a name or data array with name

        Parameters
        ----------
        da: xarray.DataArray
        cf_name: None, str
            A generic CF name
        locations: None, dict

        Return
        ------
        None, False, str
        """
        # Just a name
        # FIXME: really?
        if isinstance(da, str):
            return self.sglocator.parse_attr("name", da)[1]

        # From config
        # if "cf_loc" in da.encoding:  # Already infered and stored in encoding
        #     loc = da.encoding["cf_loc"]
        # else:  # Infer from config
        if locations is None:
            locations = self.get_loc_mapping(da, cf_names={da.name: cf_name})
        loc = locations.get(da.name)
        if loc is not None:
            return loc

        # From the array attributes
        return self.sglocator.get_loc_from_da(da)

    def format_dataarray(
        self,
        da,
        cf_name=None,
        rename=True,
        specialize=False,
        # rename_dim=True,
        loc=None,
        attrs=True,
        standardize=True,
        replace_attrs=False,
        copy=True,
        # add_loc_to_name=None,
        bound_to=None,
    ):
        """Format a DataArray's name and attributes

        Parameters
        ----------
        da: xarray.DataArray
        cf_name: str, None
            A generic CF name. If not provided, it guessed with :meth:`match`.
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
            Do not use the CF name for renaming, but the value of the "name" entry,
            which is generally a specialized one, like a name adopted by specialized dataset.
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

        # Names
        if cf_name is None:
            cf_name = self.match(da)
        if cf_name is None:
            if not rename:
                return
            return da.copy() if copy else None
        assert cf_name in self.names
        old_name = da.name
        new_name = self.get_name(cf_name, specialize=specialize)  # if specialize else cf_name

        # Location
        if loc is None:
            loc = self.get_loc_arg(da)  # from config

        # Attributes
        if attrs is True:

            # Get attributes from Cf specs
            attrs = self.get_attrs(cf_name, loc=None, standardize=False, multi=True)

            # Remove axis attribute for auxilary coordinates
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
            # add_loc_to_name=add_loc_to_name,
            replace_attrs=replace_attrs,
            copy=False,
        )
        # new_da.encoding["cf_name"] = cf_name

        # Return new name but don't rename
        if not rename:
            if old_name is None:
                return self.sglocator.format_attr("name", new_name, loc)
            return self.sglocator.merge_attr("name", old_name, new_name, loc)

        # # Rename dim if axis coordinate
        # rename_dim = rename and rename_dim
        # if (rename_dim and old_name and old_name in da.indexes):
        #     new_da = new_da.rename({old_name: new_da.name})

        return new_da

    def rename_dataarray(
        self,
        da,
        name=None,
        specialize=False,
        loc=None,
        standardize=True,
        rename_dim=True,
        copy=True,
        add_loc_to_name=None,
    ):
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
            da,
            name=name,
            specialize=specialize,
            loc=loc,
            attrs=False,
            standardize=standardize,
            rename_dim=rename_dim,
            copy=copy,
            add_loc_to_name=add_loc_to_name,
        )


class CFVarSpecs(_CFCatSpecs_):
    """CF specification for data_vars"""

    category = "data_vars"


class CFCoordSpecs(_CFCatSpecs_):
    """CF specification for coords"""

    category = "coords"

    def get_loc_mapping(self, da, cf_names=None):
        return self.parent.get_loc_mapping(da, cf_names=cf_names, categories=["coords"])

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

    def get_dim_type(self, dim, obj=None, lower=True):
        """Get the type of a dimension

        Three cases:

        - This dimension is registered in CF dims.
        - obj has dim as dim and has an axis attribute inferred with :meth:`get_axis`.
        - obj has a coord named dim with an axis attribute inferred with :meth:`get_axis`.

        Parameters
        ----------
        dim: str
            Dimension name
        obj: xarray.DataArray, xarray.Dataset
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
        if obj is not None:

            # Check dim validity
            if dim_loc not in obj.dims:
                raise XoaCFError(f"dimension '{dim}' does not belong to obj")

            # Check axis from coords
            if dim in obj.indexes:
                return self.get_axis(obj.coords[dim], lower=True)

            # Check obj axis itself
            if not hasattr(obj, "data_vars"):
                axis = self.get_axis(obj, lower=True)
                if axis:
                    return axis

    def get_dim_types(self, obj, unknown=None, asdict=False):
        """Get a tuple of the dimension types of an array

        Parameters
        ----------
        obj: xarray.DataArray, tuple(str), xarray.Dataset
            Data array, dataset or tuple of dimensions
        unknown:
            Value to assign when type is unknown
        asdict: bool

        Return
        ------
        tuple, dict
            Tuple of dimension types and of length ``obj.ndim``.
            A dimension type is either a letter or the ``unkown`` value
            when the inference of type has failed.
            If ``asdict`` is True, a dict is returned instead,
            ``(dim, dim_type)`` as key-value pairs.

        See also
        --------
        get_dim_type
        """
        dim_types = {}
        if isinstance(obj, tuple):
            dims = obj
            obj = None
        else:
            dims = obj.dims
        for dim in dims:
            dim_type = self.get_dim_type(dim, obj=obj)
            if dim_type is None:
                dim_type = unknown
            dim_types[dim] = dim_type
        if asdict:
            return dim_types
        return tuple(dim_types.values())

    @ERRORS.format_method_docstring
    def search_dim(self, obj, cf_arg=None, loc=None, errors="ignore"):
        """Search a dataarray/dataset for a dimension name according to its generic name or type

        First, scan the dimension names.
        Then, look for coordinates: either it has an 'axis' attribute,
        or it a known CF coordinate.

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Coordinate or data array
        cf_arg: str, {{"x", "y", "z", "t", "f"}}, None
            One-letter imension type or generic name.
            When set to None, dmension type is inferred with :meth:`get_axis`
            applied to `obj`
        loc: "any", letter
            Staggered grid location
        {errors}

        Return
        ------
        str, dict, None
            Dim name OR, dict with dim, type and cf_name keys if dim_type is None.
            None if nothing found.
        """
        # Explicit?
        cf_name = dim_type = None
        if cf_arg:
            if cf_arg in obj.dims:
                return cf_arg
            if len(cf_arg) == 1:
                dim_type = cf_arg.lower()
                if cf_arg in self.names:
                    cf_name = cf_arg
            else:
                self._assert_known_(cf_arg)
                cf_name = cf_arg
        isds = hasattr(obj, "data_vars")
        if not isds and dim_type is None:
            dim_type = self.get_axis(obj, lower=True)
        loc = self.sglocator.parse_loc_arg(loc)

        # Loop on dims
        for dim in obj.dims:

            # Filter-out by loc
            pname, ploc = self.sglocator.parse_attr('name', dim)
            ploc = self.sglocator.parse_loc_arg(ploc)
            if loc is not None and ploc and loc and loc != ploc:
                continue

            # From generic name
            if dim in obj.coords:
                this_cf_name = self.match(obj.coords[dim])
            else:
                this_cf_name = self.match_from_name(dim)
            if cf_name:
                if this_cf_name == cf_name:
                    return dim
                continue

            # From dimension type
            this_dim_type = self.get_dim_type(dim, obj=obj)
            out = {"dim": dim, "type": this_dim_type, "cf_name": this_cf_name}
            if this_dim_type and this_dim_type == dim_type:
                if cf_arg:
                    return dim
                return out

        # Not found but only 1d and no dim_type specified
        if len(obj.dims) == 1 and not cf_arg:
            # FIXME: loop on coordinates?
            return out

        # Failed
        errors = ERRORS[errors]
        if errors != "ignore":
            msg = f"No dimension found in dataarray matching: {cf_arg}"
            if errors == "raise":
                raise XoaCFError(msg)
            xoa_warn(msg)
        # if cf_arg is None:
        #     return
        # return None, None

    @ERRORS.format_method_docstring
    def search_from_dim(self, obj, dim, errors="ignore"):
        """Search a dataarray/dataset for a coordinate from a dimension name

        It first searches for a coordinate with a different name and that is
        the only one having this dimension.
        Then check if it is an index.
        Then look for coordinates with the same type like x, y, etc.

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
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
        if dim not in obj.dims:
            raise XoaError(f"Invalid dimension: {dim}")

        # A coord with a different name
        coords = [coord for name, coord in obj.coords.items() if name != dim and dim in coord.dims]
        if len(coords) == 1:
            return coords[0]

        # As an index
        if dim in obj.indexes:
            return obj.coords[dim]

        # Get dim_type from known dim name
        dim_type = self.get_dim_type(dim, obj, lower=True)

        # So we can do something
        def get_ndim(o):
            return len(o.dims)
        if dim_type is not None:

            # Look for a coordinate with this dim_type
            #  starting from coordinates with a higher number of dimensions
            #  like depth that have more dims than level
            for coord in sorted(obj.coords.values(), key=get_ndim, reverse=True):
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
    def get_dims(self, obj, cf_args, allow_positional=False, positions='tzyx', errors="warn"):
        """Get the data array dimensions names from their type

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array/dataset to scan
        cf_args: list
            List of letters among "x", "y", "z", "t" and "f",
            or generic names.
        allow_positional: bool
            Fall back to positional dimension of types is unkown.
        positions: str
            Default expected position of dim per type in `obj`
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
        dims = list(obj.dims)
        ndim = len(dims)
        if len(cf_args) > len(dims):
            msg = f"this data array has less dimensions ({ndim})" " than requested ({})".format(
                len(cf_args)
            )
            if errors == "raise":
                raise XoaError(msg)
            if errors == "warn":
                xoa_warn(msg)

        # Loop on types
        scanned = {}
        for cf_arg in cf_args:
            scanned[cf_arg] = self.search_dim(obj, cf_arg)

        # Guess from position
        if allow_positional:
            not_found = [item[0] for item in scanned.items() if item[1] is None]
            for i, cf_arg in enumerate(positions[::-1]):
                if cf_arg in not_found:
                    scanned[cf_arg] = dims[-i - 1]

        # Final check
        if errors != 'ignore':
            for cf_arg, dim in scanned.items():
                if dim is None:
                    msg = f"no dimension found matching: {cf_arg}"
                    if errors == 'raise':
                        raise XoaError(msg)
                    xoa_warn(msg)

        return tuple(scanned.values())

    def get_rename_dims_args(self, obj, locations=None, specialize=False):
        """Get args for renaming dimensions that are not coordinates

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array or dataset
        locations: dict, None
            Dict of staggerd grid location with dim names as keys
        specialize: bool
            Does not use the CF name for renaming, but the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.

        Return
        ------
        dict:
            Argument compatible with :meth:`xarray.Dataset.rename`
        """

        # Get location specs
        if locations is None:
            locations = self.get_loc_mapping(obj)

        # Loop on dims
        rename_args = {}
        for dim in obj.dims:

            # Skip effective coordinate dims
            if dim in obj.coords:
                continue

            # Is it known?
            cf_dim_name = self.match_from_name(dim)  # known coordinate name
            dim_type = self.get_dim_type(dim, obj)  # known dimension type
            dim_loc = locations.get(dim)

            # Root name
            if cf_dim_name:
                new_name = self.get_name(cf_dim_name, specialize=specialize)
            elif dim_type:
                new_name = dim_type
            else:
                new_name = dim

            # Add loc
            if dim_loc is not False:
                new_name = self.sglocator.merge_attr('name', dim, new_name, loc=dim_loc)

            # Register arg
            if new_name != dim:
                rename_args[dim] = new_name

        return rename_args

    def parse_dims(self, dims, obj):
        """Convert from generic dim names to specialized names

        Parameters
        ----------
        dims: str, tuple, list, dict
        obj: xarray.Dataset, xarray.DataArray

        Return
        ------
        Same type as dims
        """
        # dim_types = self.get_dim_types(obj, asdict=True)

        def _parse_dim_(cf_arg):
            return self.search_dim(obj, cf_arg) or cf_arg

        if isinstance(dims, str):
            return _parse_dim_(dims)
        if isinstance(dims, dict):
            return dict((_parse_dim_(dim), value) for dim, value in dims.items())
        return type(dims)(_parse_dim_(dim) for dim in dims)


for meth in ('get_axis', 'get_dim_type', 'get_dim_types', 'search_dim', 'get_dims'):
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
    cat, name = cfspecs.match(da, loc=loc)
    if cat:
        return cfspecs[cat][name]


def _same_attr_(da0, da1, attr):
    return (
        attr in da0.attrs
        and attr in da1.attrs
        and da0.attrs[attr].lower() == da1.attrs[attr].lower()
    )


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
    if cf0 and cf1 and cf0.name == cf1.name:
        return True

    # Name
    if da0.name and da0.name and da0.name == da1.name:
        return True

    # Long name
    return _same_attr_(da0, da1, "long_name")


def search_similar(obj, da):
    """Search in ds for a similar DataArray

    See :func:`is_similar` for what means "similar".

    Parameters
    ----------
    obj: xarray.Dataset, xarray.DataArray
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
    if hasattr(obj, "data_vars"):
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
    """Get the ``cf_specs`` encoding value

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
                if attr.lower() == "cf_specs":
                    return value


def get_cf_specs_from_encoding(ds):
    """Get a registered CF specs instance from the ``cf_specs`` encoding value

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
            os.stat(_CFGFILE).st_mtime < os.stat(USER_CF_CACHE_FILE).st_mtime
        ):
            try:
                with open(USER_CF_CACHE_FILE, "rb") as f:
                    cfspecs = pickle.load(f)
            except Exception as e:
                xoa_warn("Error while loading cached cf specs: " + str(e.args))

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
        :attr:`cf_specs` attribute or encoding.
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
        cfspecs = cf_cache["current"]
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
        if (
            isinstance(name, str)
            and cfspecs["register"]["name"]
            and cfspecs["register"]["name"] == name
        ):
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
        cfnames = [cfspecs[cat].get_name(name, specialize=True) for name in cfspecs[cat].names]
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
    First, the :attr:`cf_specs` global attribute or encoding of the dataset is compared
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
    """Set the ``cf_specs`` encoding to ``name`` in all data vars and coords

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
        target.encoding.update(cf_specs=name)
    return ds


def infer_coords(ds):
    """Infer which of the data arrays of a dataset are coordinates

    When coordinates are found, it makes sure they are registered in the dataset
    as coordindates.

    Parameters
    ----------
    ds: xarray.Dataset

    See also
    --------
    CFSpecs.infer_coords
    """
    return get_cf_specs(ds).infer_coords(ds)


infer_coords.__doc__ = CFSpecs.infer_coords.__doc__
