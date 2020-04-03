#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naming convention tools for reading and formatting variables
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

import os
import pickle
import re
import operator
import warnings
import pprint

import appdirs

from .__init__ import XoaError, xoa_warn, get_option
from .misc import dict_merge, match_string

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
)

_CACHE = {}


class XoaCFError(XoaError):
    pass


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
        """Parse datarray name and attribute to find location

        Parameters
        ----------
        da: xarray.DataArray
            Dataarray to scan

        Returns
        None, str
            ``None`` if no location is found, else the correspoing string

        Raises
        ------
        XoaCFError
            When locations parsed from name and attributes are conflicting.
            Thus, this method method is a way check location consistency.
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

    def patch_attrs(self, attrs, patch, loc=None, standardize=True):
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

        Returns
        -------
        dict
            A new dictionary of attributes

        See also
        --------
        merge_attr
        """
        attrs = attrs.copy()

        # Loop on attributes
        for attr, value in patch.items():
            if value is None:
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
        rename=True, copy=True
    ):
        """Format name and attributes of copy of DataArray

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
            Substitute for dataarray name
        attrs: str, None
            Substitute for dataarray attributes.
            If ``standard_name`` and ``long_name`` values are a list,
            and if the dataarray has its attribute included in the list,
            it is left unchanged since it is considered compatible.
        rename:
            Prevent renaming the array
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
        if da.attrs:
            da.attrs.update(self.format_attrs(
                da.attrs, loc, standardize=standardize))
        if attrs:
            da.attrs.update(self.patch_attrs(da.attrs, attrs, loc,
                                             standardize=standardize))

        # Name
        if rename:
            da.name = self.merge_attr("name", da.name, name, loc)

        return da


class CFSpecs(object):
    """CF specification manager

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

    """

    def __init__(self, cfg=None, default=True, user=True):

        # Initialiase categories
        self._cfs = {}
        catcls = {"data_vars": CFVarSpecs, "coords": CFCoordSpecs}
        for category in self.categories:
            self._cfs[category] = catcls[category](self)

        # Load config
        self._dict = None
        self._cfgspecs = _get_cfgm_().specs.dict()
        self._cfgs = []
        self._load_default = default
        self._load_user = user
        self.load_cfg(cfg)

    def _load_cfg_as_dict_(self, cfg, cache=None):
        """Load a single cfg, validate it and return it as a dict

        When the config source is a tuple or a file name, the loaded config
        is in-memory cached by default. The cache key is the first item
        of the tuple or the file name.

        Parameters
        ----------
        cf: str, tuple, dict, CFSpecs
            Config source
        cache: bool
            Use in-memory cache system?

        """
        # Config manager to get defaults and validation
        cfgm = _get_cfgm_()

        # Get it from cache
        if cache is None:
            cache = get_option("cf.cache")
        cache = cache and ((isinstance(cfg, str) and '\n' not in cfg) or
                           isinstance(cfg, tuple))
        if cache:
            # Init cache
            if "cfgs" not in _CACHE:
                _CACHE["cfgs"] = {}
            if isinstance(cfg, str):
                cache_key = cfg
            else:
                cache_key, cfg = cfg
            if cache_key in _CACHE["cfgs"]:
                # print("CFG FROM CACHE: " + cfg)
                return _CACHE["cfgs"][cfg]

        # Check input type
        if isinstance(cfg, str) and '\n' in cfg:
            cfg = cfg.split("\n")
        elif isinstance(cfg, CFSpecs):
            cfg = cfg._dict

        # Load, validate and convert to dict
        cfg_dict = cfgm.load(cfg).dict()

        # Cache it
        if cache:
            _CACHE["cfgs"][cache_key] = cfg_dict

        return cfg_dict

    def load_cfg(self, cfg=None, cache=None):
        """Load a single or a list of configurations

        Parameters
        ----------
        cfg: ConfigObj init or list
            Single or a list of either:

            - config file name,
            - multiline config string,
            - config dict,
            - two-element tuple with the first item of one of the above types,
              and second item as a cache key for faster reloading.
        cache: bool, None
            In in-memory cache system?
            Defaults to option boolean :xoaoption:`cf.cache`.

        """
        # Get the list of validated configurations
        to_load = []
        if cfg:
            if not isinstance(cfg, list):
                cfg = [cfg]
            to_load.extend([c for c in cfg if c])
        if self._load_user and os.path.exists(USER_CF_FILE):
            to_load.append(USER_CF_FILE)
        if self._load_default:
            to_load.append(_CFGFILE)
        if not to_load:
            to_load = [None]

        # Load them
        dicts = [self._load_cfg_as_dict_(cfg, cache) for cfg in to_load]

        # Merge them
        self._dict = dict_merge(*dicts, **_CF_DICT_MERGE_KWARGS)

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

    def __getitem__(self, section):
        assert section in self._dict
        if section in self.categories:
            return self._cfs[section]
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
        for coord_specs in self._dict['coords'].values():
            if coord_specs['axis']:
                axis = coord_specs['axis'].lower()
                self._dict['dims'][axis].extend(coord_specs['name'])

    def _process_entry_(self, category, name):
        """Process an entry

        - Makes sure to have lists, except for 'axis' and 'inherit'
        - Check geo coords
        - Check inheritance
        - Makes sure that coords specs have no 'coords' key
        - Makes sure that specs key is the first entry of 'names'
        - Add standard_name to list of names
        - Check duplications to other locations ('toto' -> 'toto_u')

        Yield
        -----
        category, name, entry
        """
        # Dict of entries for this category
        entries = self._dict[category]

        # # Wrong entry!
        # if name not in entries:
        #     print('bad name', name)
        #     yield

        # # Already processed
        # if "processed" in entries[name] and name=='temp':
        #     print('already processed', name)
        #     yield

        # Get the specs as pure dict
        if hasattr(entries[name], "dict"):
            entries[name] = entries[name].dict()
        specs = entries[name]

        # Long name from name or standard_name
        if not specs["long_name"]:
            if specs["standard_name"]:
                long_name = specs["standard_name"][0]
            else:
                long_name = name.title()
            long_name = long_name.replace("_", " ").capitalize()
            specs["long_name"].append(long_name)

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

            # Check compatibility of keys
            for key in list(specs.keys()):
                if key not in self._cfgspecs[category]["__many__"]:
                    del specs[key]

            specs["inherit"] = None  # switch off inheritance now

        # Names
        if name in specs["name"]:
            specs["name"].remove(name)
        specs["name"].insert(0, name)

        # Select
        if specs.get("select", None):
            for key in specs["select"]:
                try:
                    specs["select"] = eval(specs["select"])
                except Exception:
                    pass

        # Standard_names in names
        if specs["standard_name"]:
            for standard_name in specs["standard_name"]:
                if standard_name not in specs["name"]:
                    specs["name"].append(standard_name)

        # if name=="depth":
        #     print('>>>>',category, specs['name'])

        specs['processed'] = True
        yield category, name, specs

    def format_coord(self, da, name=None, loc=None, copy=True,
                     standardize=True, rename=True, rename_dim=True):
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
        standardize: bool
        rename_dim: bool
            For a 1D array, rename the dimension if it has the same name
            as the array.
            Note that it is set to False, if ``rename`` is False.

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
            rename=rename, rename_dim=rename_dim)

    def format_data_var(
        self,
        da,
        name=None,
        rename=True,
        format_coords=True,
        coords=None,
        loc=None,
        standardize=True,
        copy=True
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
        new_name = self.data_vars.format_dataarray(
            da, name=name, loc=loc, standardize=standardize,
            rename=False, copy=False
        )
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
                    copy=False
                )

        # Final renaming
        if rename and rename_args:
            da = da.rename(rename_args)

        # Return the guessed name
        if not rename and name is None:
            return new_name

        return da

    def format_dataset(self, ds, loc=None, rename=True, standardize=True,
                       format_coords=True, coords=None, copy=True):
        """Auto-format a whole xarray.Dataset"""
        # Copy
        if copy:
            ds = ds.copy()

        # Init rename dict
        rename_args = {}

        # Data arrays
        for name, da in list(ds.items()):
            new_name = self.format_data_var(
                da, loc=loc, rename=False, standardize=True,
                format_coords=False, copy=False)
            if rename and new_name:
                rename_args[da.name] = new_name

        # Coordinates
        if format_coords:
            for cname, cda in list(ds.coords.items()):
                new_name = self.format_coord(
                    cda, loc=loc, standardize=True, rename=False,
                    rename_dim=False, copy=False)
                if rename:
                    rename_args[cda.name] = new_name

                # if cda.ndim == 1 and cname == cda.dims[0]:
                #     ds.coords[cname] = cda
                #     ds = ds.rename({cname: cda.name})
                # else:
                #     ds.coords[cda.name] = cda

        # Final renaming
        if rename and rename_args:
            ds = ds.rename(rename_args)

        return ds

    def auto_format(self, dsa, loc=None, standardize=True):
        """Auto-format the xarray.Dataset or xarray.DataArray

        See also
        --------
        format_dataarray
        format_dataset
        """
        if hasattr(dsa, "data_vars"):
            return self.format_dataset(dsa, loc=loc, standardize=standardize,
                                       format_coords=True)
        return self.format_dataarray(dsa, loc=loc, standardize=standardize)

    __call__ = auto_format

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

        See also
        --------
        CFVarSpecs.match
        """
        return self.data_vars.match(da, name=name, loc=loc)

    def search_coord(self, dsa, name=None, loc="any", get="obj", single=True):
        """Search for a coord that maches given or any specs

        Parameters
        ----------
        dsa: DataArray or Dataset
        name: str, dict
            A CF name. If not provided, all CF names are scaned.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {"obj", "name"}
            When found, get the object found or its name.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
            A warning is emitted when set to True and multiple item are found.

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
                               attrs={'standard_name': 'longitude'})
            data = xr.DataArray([0, 1], dims=('foo'), coords=[lon])
            cfspecs = get_cf_specs()
            cfspecs.search_coord(data, "lon")
            cfspecs.search_coord(data, "lon", get="name")
            cfspecs.search_coord(data, "lat")

        See also
        --------
        search_data_var
        CFCoordSpecs.search
        """
        return self.coords.search(dsa, name=name, loc=loc, get=get,
                                  single=single)

    def search_dim(self, da, dim_type=None, loc="any"):
        """Search for a dimension from its type

        Parameters
        ----------
        da: xarray.DataArray
        dim_type: None, {"x", "y", "z", "t", "f"}
            Dimension type
        loc:
            Location

        See also
        --------
        CFCoordSpecs.search_dim
        """
        return self.coords.search_dim(da, dim_type=dim_type, loc=loc)

    def search_coord_from_dim(self, da, dim):
        """Search a dataarray for a coordinate from a dimension name

        It first searches for a coordinate with the same name and that is
        the only one having this dimension.
        Then look for coordinates with the same type like x, y, etc.

        Parameters
        ----------
        da: xarray.DataArray
        dim: str

        Return
        ------
        xarray.DataArray, None
            An coordinate array or None
        """
        return self.coords.search_from_dim(da, dim)

    def search_data_var(self, dsa, name=None, loc="any", get="obj",
                        single=True):
        """Search for a data_var that maches given or any specs

        Parameters
        ----------
        dsa: DataArray or Dataset
        name: str, dict
            A CF name. If not provided, all CF names are scaned.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {"obj", "name"}
            When found, get the object found or its name.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
            A warning is emitted when set to True and multiple item are found.

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
            data = xr.DataArray(
                [0, 1], dims=('x'),
                attrs={'standard_name': 'sea_water_temperature'})
            ds = xr.Dataset({'foo': data})
            cfspecs = get_cf_specs()
            cfspecs.search_data_var(ds, "temp")
            cfspecs.search_data_var(ds, "temp", get="name")
            cfspecs.search_data_var(ds, "sal")

        See also
        --------
        search_coord
        CFVarSpecs.search
        """
        return self.data_vars.search(dsa, name=name, loc=loc, get=get,
                                     single=single)

    def search(self, dsa, name=None, loc="any", get="obj",
               single=True, categories=None):
        """Search for a dataarray with data_vars and/or coords"""
        if not categories:
            categories = (self.categories if hasattr(dsa, "data_vars")
                          else ["coords"])
        else:
            categories = self.categories
        for category in categories:
            res = self[category].search(dsa, name=name, loc=loc, get=get,
                                        single=single)
            if res is not None:
                return res


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
        return self._dict.values

    # def __len__(self):
    #     return len(self._dict)

    def __contains__(self, key):
        return key in self._dict

    # def __str__(self):
    #     return pformat(self._dict)

    def _validate_name_(self, name):
        if name in self:
            return name

    def _assert_known_(self, name):
        if name not in self._dict:
            raise XoaCFError(f"Invalid {self.category} CF specs name: "+name)

    @property
    def names(self):
        return list(self._dict.keys())

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def get_specs(self, name, errors="warn"):
        """Get the specs of acf item

        Parameters
        ----------
        name: str
        errors: "silent", "warning" or "error".

        Return
        ------
        dict or None
        """
        assert errors in ("ignore", "warn", "raise")
        if name not in self._dict:
            if errors == "raise":
                raise XoaCFError("Can't get cf specs from: " + name)
            if errors == "warn":
                xoa_warn("Invalid cf name: " + str(name))
            return
        return self._dict[name]

    @property
    def dims(self):
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
                if attr in specs and specs[attr] and attr[0] == sm:
                    match_specs[attr] = specs[attr]
        return match_specs

    def match(self, da, name=None, loc="any"):
        """Check if da attributes match given or any specs

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

    def search(self, dsa, name=None, loc="any", get="obj", single=True):
        """Search for a data_var or coord that maches given or any specs

        Parameters
        ----------
        dsa: DataArray or Dataset
        name: str, dict
            A CF name. If not provided, all CF names are scaned.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {"obj", "name"}
            When found, get the object found or its name.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
            A warning is emitted when set to True and multiple item are found.

        Returns
        -------
        None or str or object
        """

        # Get target objects
        if self.category:
            objs = getattr(dsa, self.category)
        else:
            objs = dsa.keys() if hasattr(dsa, "keys") else dsa.coords

        # Get match specs
        if name:  # Explicit name so we loop on search specs
            if isinstance(name, str):
                self._assert_known_(name)
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
                    if id(obj) in found_objs:
                        continue
                    found_objs.append(id(obj))
                    name = name if name else m
                    if get == "both":
                        found.append((obj, name))
                    else:
                        found.append(obj if get == "obj" else name)

        # Return
        if not single:
            return found
        if len(found) > 1:
            xoa_warn("Multiple items found while you requested a single one")
        if found:
            return found[0]

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
        mode: "silent", "warning" or "error".
        multi: bool
            Get standard_name and long_name attribute as a list of possible
            values
        **extra
          Extra params as included as extra attributes

        Return
        ------
        dict
        """

        # Get specs
        specs = self.get_specs(name, errors=errors) or {}

        # Which attributes
        attrs = {}
        if exclude is None:
            exclude = []
        elif isinstance(exclude, str):
            exclude = [exclude]
        exclude.extend(self.attrs_exclude)
        exclude.extend(extra.keys())
        exclude = set(exclude)
        keys = set(self.attrs_first)
        set(specs.keys())
        keys -= exclude
        if select:
            keys = keys.intersection(select)

        # Loop
        for key in keys:

            # No lists or tuples
            value = specs[key]
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

    def format_dataarray(self, da, name=None, loc=None, rename=True,
                         standardize=True, rename_dim=True, copy=True):
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
            Rename arrays
        standardize: bool
        rename_dim: bool
            For a 1D array, rename the dimension if it has the same name
            as the array.
            Note that it is set to False, if ``rename`` is False.

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

        # Get name
        if name is None:
            name = self.match(da, loc="any")
        if name is None:
            if not rename:
                return
            return da.copy() if copy else None
        assert name in self.names

        # Format array
        old_name = da.name
        cf_attrs = self.get_attrs(
            name, loc=None, standardize=False, multi=True)
        new_da = self.sglocator.format_dataarray(
            da,
            loc=loc,
            name=name,
            attrs=cf_attrs,
            standardize=standardize,
            rename=rename,
            copy=False,
        )

        # Return renaming name not renaming
        if not rename:
            return self.sglocator.merge_attr("name", da.name, name, loc)

        # Rename dim if axis coordinate
        rename_dim = rename and rename_dim
        if (rename_dim and old_name is not None and da.ndim == 1 and
                old_name == da.dims[0]):
            new_da = new_da.rename({old_name: new_da.name})
        return new_da


class CFVarSpecs(_CFCatSpecs_):
    """CF specification for data_vars"""

    category = "data_vars"


class CFCoordSpecs(_CFCatSpecs_):
    """CF specification for coords"""

    category = "coords"

    def get_dim_type(self, coord, lower=True):
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

        """
        axis = None
        if "axis" in coord.attrs:
            axis = coord.attrs["axis"]
        else:
            cfname = self.match(coord)
            if cfname:
                axis = self[cfname]["axis"]
        if axis is not None:
            if lower:
                return axis.lower()
            return axis.upper()

    def search_dim(self, da, dim_type=None, loc="any"):
        """Search a dataarray for a dimension name according to its type

        First, scan the dimension names.
        Then, look for coordinates: either it has an 'axis' attribute,
        or it a known CF coordinate.

        Parameters
        ----------
        da: xarray.DataArray
        dim_type: {"x", "y", "z", "t", "f"}, None
            When
        loc: "any", letter
            Staggered grid location

        Return
        ------
        str, (str, str)
            Dim name OR, (dim name, dim_type) if dim_type is None
        """
        if dim_type is not None:
            dim_type = dim_type.lower()
        else:
            da_axis = self.get_dim_type(da)

        # From dimension names
        for dim in da.dims:
            pname, ploc = self.sglocator.parse_attr('name', dim)
            if loc != "any" and ploc and loc and loc != ploc:
                continue
            dim_types = (self.dims.keys()
                         if not dim_type and not da_axis else
                         [dim_type or da_axis])
            for dt in dim_types:
                if pname.lower() in self.dims[dt]:
                    if dim_type is not None:
                        return dim
                    if da.ndim == 1 or da_axis:
                        return dim, dt

        # From coordinates
        for coord in da.coords.values():
            if coord.ndim == 1 and coord.name in da.dims:
                co_axis = self.get_dim_type(coord)
                if co_axis:
                    if dim_type and co_axis == dim_type:
                        return coord.name
                    if dim_type is None and da_axis and (
                            da_axis == co_axis):
                        return coord.name, co_axis

        # Fallback
        if dim_type is None and da.ndim == 1:
            return da.dims[0], da_axis

    def search_from_dim(self, da, dim):
        """Search a dataarray for a coordinate from a dimension name

        It first searches for a coordinate with the same name and that is
        the only one having this dimension.
        Then look for coordinates with the same type like x, y, etc.

        Parameters
        ----------
        da: xarray.DataArray
        dim: str

        Return
        ------
        xarray.DataArray, None
            An coordinate array or None
        """

        assert dim in da.dims, 'Invalid dimension'
        dim_type = None

        # Coord as dim only
        if dim in da.coords:

            # Check if another coordinate has this dim
            for coord in da.coords.values():
                if coord.name != dim and dim in coord.dims:
                    break
            else:
                return da.coords[dim]

            # Get the dim_type if any
            dim_type = self.get_dim_type(da.coords[dim])

        # Get dim_type from known dim name
        if dim_type is None:
            pname, ploc = self.sglocator.parse_attr('name', dim)
            for dim_type, dim_names in self.dims.items():
                if pname.lower() in dim_names:
                    break

        # Nothing to do there
        if dim_type is None:
            return

        # Look for a coordinate with this dim_type
        for coord in sorted(da.coords.values(),
                            key=operator.attrgetter('ndim'),
                            reverse=True):
            if dim in coord.dims:
                coord_dim_type = self.get_dim_type(coord)
                if coord_dim_type and coord_dim_type == dim_type:
                    return coord


def _get_cfgm_():
    """Get a :class:`~xoa.cfgm.ConfigManager` instance to manage
    coords and data_vars spécifications"""
    if "cfgm" not in _CACHE:
        from .cfgm import ConfigManager

        _CACHE["cfgm"] = ConfigManager(_INIFILE)
    return _CACHE["cfgm"]


def _same_attr_(da0, da1, attr):
    return (attr in da0.attrs and attr in da1.attrs and
            da0.attrs[attr].lower() == da1.attrs[attr].lower())


def are_similar(da0, da1):
    """Check if two DataArrays are similar

    Verifications are performed in the following order:

    - ``standard_name`` attribute,
    - ``cf_name`` as the results of ``da.cf.name``.
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
    if (da0.cf.name and da1.cf.name and da0.cf.name == da1.cf.name):
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
    """
    import xarray as xr
    targets = list(da.coords.values())
    if isinstance(dsa, xr.Dataset):
        targets = list(da.data_vars.values()) + targets
    for ds_da in targets:
        if are_similar(ds_da, da):
            return ds_da


class set_cf_specs(object):
    """Set the current CF specs"""

    def __init__(self, cf_source):
        assert isinstance(cf_source, CFSpecs)
        self.old_specs = _CACHE.get("specs", None)
        _CACHE["specs"] = self.specs = cf_source

    def __enter__(self):
        return self.specs

    def __exit__(self, exc_type, exc_value, traceback):
        if self.old_specs is None:
            del _CACHE["specs"]
        else:
            _CACHE["specs"] = self.old_specs


def clean_cf_cache():
    """Remove the cf specs cache file"""
    if os.path.exists(USER_CF_CACHE_FILE):
        os.remove(USER_CF_CACHE_FILE)


def show_cf_cache():
    """Show the cf specs cache file"""
    print(USER_CF_CACHE_FILE)


def get_cf_specs(name=None, category=None, cache="rw"):
    """Get the CF specifications for a target in a category

    Parameters
    ----------
    name: str or None
        A target name like "sst". If not provided, return all specs.
    category: str or None
        Select a category with "coords" or "data_vars".
        If not provided, search first in "data_vars", then "coords".
    cache: str, bool, None
        Cache specs on disk with pickling for fast loading.
        If ``None``, it defaults to boolean option :xoaoption:`cf.cache`.
        Possible string values: ``"ignore"``, ``"rw"``, ``"read"``,
        ``"write"``, ``"clean"``.
        If ``True``, it is set to ``"rw"``.
        If ``False``, it is set to ``"ignore"``.

    Return
    ------
    dict or None
        None is return if no specs are found
    """
    if cache is None:
        cache = get_option('cf.cache')
    if cache is True:
        cache = "rw"
    elif cache is False:
        cache = "ignore"
    assert cache in ("ignore", "rw", "read", "write", "clean")

    # Get the source of specs
    if "specs" not in _CACHE:

        # Try from disk cache
        if cache in ("read", "w"):
            if os.path.exists(USER_CF_CACHE_FILE) and (
                os.stat(_CFGFILE).st_mtime <
                os.stat(USER_CF_CACHE_FILE).st_mtime
            ):
                try:
                    with open(USER_CF_CACHE_FILE, "rb") as f:
                        set_cf_specs(pickle.load(f))
                except Exception as e:
                    xoa_warn(
                        "Error while loading cached cf specs: " + str(e.args)
                    )

        # Compute it from scratch
        if "specs" not in _CACHE:

            # Setup
            set_cf_specs(CFSpecs())

            # Cache it on disk
            if cache in ("write", "rw"):
                try:
                    cachedir = os.path.dirname(USER_CF_CACHE_FILE)
                    if not os.path.exists(cachedir):
                        os.makedirs(cachedir)
                    with open(USER_CF_CACHE_FILE, "wb") as f:
                        pickle.dump(_CACHE["specs"], f)
                except Exception as e:
                    xoa_warn("Error while caching cf specs: " + str(e.args))

    # Clean cache
    if cache == "clean":
        clean_cf_cache()

    cf_source = _CACHE["specs"]

    # Select categories
    if category is not None:
        if isinstance(cf_source, CFSpecs):
            cf_source = cf_source[category]
        if name is None:
            return cf_source
        toscan = [cf_source]
    else:
        if name is None:
            return cf_source
        toscan = [cf_source["data_vars"], cf_source["coords"]]

    # Search
    for ss in toscan:
        if name in ss:
            return ss[name]


# def get_cf_coord_specs(name=None):
#     """Shortcut to ``get_cf_specs(name=name, category='coords')``"""
#     return get_cf_specs(name=name, category="coords")


# def get_cf_var_specs(name=None):
#     """Shortcut to ``get_cf_specs(name=name, category='data_vars')``"""
#     return get_cf_specs(name=name, category="data_vars")


class _CFAccessor_(object):
    _category = None

    def __init__(self, dsa):
        self._cfspecs = get_cf_specs()
        self._dsa = dsa

    def set_cf_specs(self, cfspecs):
        """Set the :class:`CFSpecs` using by this accessor"""
        assert isinstance(cfspecs, CFSpecs)
        self._cfspecs = cfspecs

    def get(self, name, loc="any", single=True):
        """Search for a CF item with :meth:`CFSpecs.search`"""
        return self._cfspecs[self._category].search(
            self._dsa, name=name, loc=loc, get="obj", single=single)

    def get_coord(self, name, loc="any", single=True):
        """Search for a CF coord with :meth:`CFCoordSpecs.search`"""
        return self._cfspecs.coords.search(
            self._dsa, name=name, loc=loc, get="obj", single=single)

    def __getattr__(self, attr):
        if attr in self._cfspecs['accessors']['properties'][self._category]:
            return self.get(attr)
        return object.__getattribute__(self, attr)

    def __getitem__(self, name):
        return self.get(name)

    def auto_format(self, loc=None, standardize=True):
        """Auto-format attributes with :meth:`CFSpecs.auto_format`"""
        return self._cfspecs.auto_format(self._dsa, loc=loc,
                                         standardize=standardize)

    __call__ = auto_format


class DataArrayCFAccessor(_CFAccessor_):
    _category = 'coords'

    @property
    def name(self):
        if not hasattr(self, '_name'):
            self._name = self._cfspecs.match(self._dsa)
        return self._name

    @property
    def dim(self):
        try:
            return self._cfspecs.coords.search_dim(self._dsa)[0]
        except XoaError:
            return

    def get_dim(self, dim_type):
        dim_type = dim_type.lower()
        if not hasattr(self, '_dims'):
            self._dims = {}
            if dim_type not in self._dims:
                self._dims[dim_type] = self._cfspecs.coords.search_dim(
                    self._dsa, dim_type)
        return self._dims[dim_type]

    @property
    def xdim(self):
        return self.get_dim("x")

    @property
    def ydim(self):
        return self.get_dim("y")

    @property
    def zdim(self):
        return self.get_dim("z")

    @property
    def tdim(self):
        return self.get_dim("t")


class DatasetCFAccessor(_CFAccessor_):
    _category = 'data_vars'


def register_cf_accessors(name='cf'):
    """Register xarray accessors"""
    # cfspecs = cfspecs or get_cf_specs()
    # name = cfspecs['accessors']['name']
    import xarray as xr
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            xr.core.extensions.AccessorRegistrationWarning)
        xr.register_dataarray_accessor(name)(DataArrayCFAccessor)
        xr.register_dataset_accessor(name)(DatasetCFAccessor)


register_cf_accessors()
