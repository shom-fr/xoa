#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naming convention tools
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

import appdirs

from .__init__ import XoaError, xoa_warn
from .misc import dict_merge, match_string

_THISDIR = os.path.dirname(__file__)

# Joint variables and coords config specification file
_INIFILE = os.path.join(_THISDIR, "cf.ini")

# Base config file for CF specifications
_CFGFILE = os.path.join(_THISDIR, "cf.cfg")

# File used for pickling cf specs
_user_cache_dir = appdirs.user_cache_dir("xoa")
_PYKFILE = os.path.join(_user_cache_dir, "cf.pyk")

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


class SGLocator(object):
    """Staggered grid location parsing and formatting utility

    Parameters
    ----------
    name_format: str
        A string containing the string patterns ``{root}`` and ``{loc}}``,
        which defaults to ``"{root}_{loc}"``
    valid_locations: str
        Valid location letters that are checked when parsing
    """

    formats = {
        "name": "{root}_{loc}",
        "standard_name": "{root}_at_{loc}_location",
        "long_name": "{root} at {loc} location",
    }

    re_match = {
        "standard_name": re.compile(
            formats["standard_name"].format(
                root=r"(?P<root>\w+)", loc=r"(?P<loc>[a-zA-Z])"
            ),
            re.I,
        ).match,
        "long_name": re.compile(
            formats["long_name"].format(
                root=r"(?P<root>[\w ]+)", loc=r"(?P<loc>[a-zA-Z])"
            ),
            re.I,
        ).match,
    }

    valid_attrs = ("name", "standard_name", "long_name")

    def __init__(self, name_format=None, valid_locations=None):

        # Init
        self.formats = self.formats.copy()
        self.re_match = self.re_match.copy()
        self._name_format = name_format
        if valid_locations:
            valid_locations = "".join([s.lower() for s in valid_locations])
        self.valid_locations = valid_locations

        # The name case
        if name_format:
            self.formats["name"] = name_format
        self.re_match["name"] = re.compile(
            self.formats["name"].format(
                root=r"(?P<root>\w+)", loc=r"(?P<loc>[a-zA-Z])"
            ),
            re.I,
        ).match

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
        letter or None
            Location
        """
        m = self.re_match[attr](value)
        if m is None:
            return value, None
        root = m.group("root")
        loc = m.group("loc").lower()
        if self.valid_locations and loc not in self.valid_locations:
            return value, None
        return root, loc

    def match_attr(self, attr, value, root, loc=None):
        """Check if an attribute is matching a location

        Parameters
        ----------
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        root: str
        loc: letters, {"any", None} or {"", False}
            - letters: one of these locations
            - None or "any": any
            - False or '': no location

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
        """Format a attribute at a specified location

        Parameters
        ----------
        attr: {'name', 'standard_name', 'long_name'}
            Attribute name
        value: str
            Current attribute value. It is parsed to get current ``root``.
        loc: {False, ""}, letter, None
            If None, location is left unchanged;
            if a letter, it is set;
            else, it is removed.
        standardize: bool
            If True, standardize ``root`` and ``loc`` values.

        Example
        -------
        >>> sg = SGLocator()
        >>> sg.format_attr('standard_name', 'sea_water_temperature', 't')
        'sea_water_temperature_at_t_location'
        >>> sg.format_attr('standard_name', 'sea_water_temperature', False)
        'sea_water_temperature'

        Return
        ------
        str
        """
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
                f"Invalid location: {loc}. "
                f"Please one use of: {self.valid_locations}."
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
        """Copy and format a dict of attributes"""
        attrs = attrs.copy()
        for attr, value in attrs.items():
            if attr in self.valid_attrs:
                attrs[attr] = self.format_attr(
                    attr, value, loc, standardize=standardize
                )
        return attrs

    def format_dataarray(
        self, da, loc=None, standardize=True, name=None, attrs=None
    ):
        """Format name and attributes of copy of DataArray

        Parameters
        ----------
        da: xarray.DataArray
        loc: str, {None, False}
        standardize: bool
        name: str, None
            Substitute for dataarray name
        attrs: str, None
            Substitute for dataarray attributes

        Return
        ------
        xarray.DataArray
        """
        da = da.copy()
        if attrs is not None:
            da.attrs.update(attrs)
        da.attrs.update(
            self.format_attrs(da.attrs, loc, standardize=standardize)
        )
        if name:
            da.name = name
        da.name = self.format_attr(
            "name", da.name, loc, standardize=standardize
        )
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


    Attributes
    ----------
    sglocator: :class:`SGLocator`
        Used to manage stagerred grid locations
    categories: list
        Types of specs
    """

    def __init__(self, cfg=None):

        # Initialiase categories
        self._cfs = {}
        catcls = {"coords": CFCoordSpecs, "data_vars": CFVarSpecs}
        for category in self.categories:
            self._cfs[category] = catcls[category](self)

        # Load config
        self._dict = None
        self._cfgspecs = _get_cfgm_().specs.dict()
        self.load_cfg(cfg)

    def load_cfg(self, cfg=None, update=True, cache=True):
        """Load a single or a list of configuration

        Parameters
        ----------
        cfg: ConfigObj init or list
            Single or a list of either:

            - config file name,
            - multiline config string,
            - config dict,
            - two-element tuple with the first item of one of the above types,
              and second item as a cache key for faster reloading.
        update: bool
            Update the current configuration or replace it.
        """
        # Get the list of validated configurations
        cfgm = _get_cfgm_()
        cfgs = [cfg] if not isinstance(cfg, list) else cfg
        if update:
            cfgs.append(self._dict)
        cfgs = [cfg for cfg in cfgs if cfg is not None]
        if not cfgs:
            cfgs = [None]
        if cache and "cfgs" not in _CACHE:
            _CACHE["cfgs"] = {}
        for i, cfg in enumerate(cfgs):

            # get from it cache?
            if isinstance(cfg, tuple):
                cfg, cache_key = cfg
                if cache and cache_key in _CACHE["cfgs"]:
                    cfgs.append(_CACHE["cfgs"][cache_key])
                    print("CFG FROM CACHE: " + cache_key)
                    continue
            else:
                cache_key = None

            # load and validate
            if isinstance(cfg, str) and cfg.strip().startswith("["):
                cfg = cfg.split("\n")
            elif isinstance(cfg, CFSpecs):
                cfg = cfg._dict
            cfgs[i] = cfgm.load(cfg).dict()

            # cache it
            if cache and cache_key:
                _CACHE["cfgs"][cache_key] = cfgs[-1]

        # Merge
        self._dict = dict_merge(*cfgs, **_CF_DICT_MERGE_KWARGS)

        # SG locator
        self._sgl_settings = self._dict["sglocator"]
        self._sgl = SGLocator(**self._sgl_settings)

        # Post process
        self._post_process_()

    def copy(self):
        return CFSpecs(self)

    def __copy__(self):
        return self.copy()

    @property
    def categories(self):
        return ["coords", "data_vars"]

    @property
    def sglocator(self):
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
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                self.__class__.__name__, name
            )
        )

    # def __str__(self):
    #     return pformat(self._dict)

    def register_from_cfg(self, cfg):
        """"Register new elements from a :class:`ConfigObj` instance
        or a config file"""
        self.load_cfg(cfg, update=True)

    def _post_process_(self):

        # Inits
        items = {}
        self._from_atlocs = {}
        for category in self.categories:
            items[category] = []
            self._from_atlocs[category] = []

        # Process
        for category in self.categories:
            self._from_atlocs[category] = []
            for name in self[category].names:
                for item in self._process_entry_(category, name):
                    if item:
                        pcat, pname, pspecs = item
                        items[pcat].append((pname, pspecs))

        # Refill
        for category, contents in items.items():
            self._dict[category].clear()
            self._dict[category].update(items[category])

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

        # Wrong entry!
        if name not in entries:
            yield

        # Get the specs as pure dict
        if hasattr(entries[name], "dict"):
            entries[name] = entries[name].dict()
        specs = entries[name]

        # Ids
        if name in specs["name"]:
            specs["name"].remove(name)
        specs["name"].insert(0, name)

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
            from_name = specs["inherit"]
            if ":" in from_name:
                from_cat, from_name = from_name.split(":")[:2]
            else:
                from_cat = category
            assert (
                from_cat != category or name != from_name
            ), "Cannot inherit cf specs from it self"
            # to handle high level inheritance
            for item in self._process_entry_(from_cat, from_name):
                yield item
            from_specs = None
            to_scan = []
            if "inherit" in entries:
                to_scan.append(self._inherit)
            to_scan.append(entries)

            for from_specs in to_scan:
                if from_name in from_specs:

                    # Merge
                    entries[name] = specs = dict_merge(
                        specs,
                        from_specs[from_name],
                        cls=dict,
                        **_CF_DICT_MERGE_KWARGS,
                    )

                    # Check compatibility of keys
                    for key in list(specs.keys()):
                        if key not in self._cfgspecs[category]["__many__"]:
                            del specs[key]
            specs["inherit"] = None  # switch off inheritance now

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

        yield category, name, specs

    def format_coord(self, da, name=None, loc=None, standardize=True):
        """Format a coordinate variable"""
        return self.coords.format_dataarray(
            da, name=name, loc=loc, standardize=standardize)

    def format_data_var(
        self,
        da,
        name=None,
        coords=None,
        format_coords=True,
        loc=None,
        standardize=True,
    ):
        # Data var
        da = self.data_vars.format_dataarray(
            da, loc=loc, standardize=standardize
        )

        # Coordinates
        if format_coords:
            coords = coords or {}
            for cname, cda in list(da.coords.items()):
                cda = self.format_coord(
                    cda,
                    name=coords.get(cname),
                    loc=loc,
                    standardize=standardize,
                )
                da.coords[cname] = cda
                da = da.rename({cname: cda.name})

        return da

    def format_dataset(self, ds, loc=None, standardize=True):
        """Auto-format a whole xarray.Dataset"""
        ds = ds.copy()
        for name, da in ds.items():
            del ds[name]
            da = self.format_data_var(da, loc=loc, standardize=True)
            ds[da.name] = da
        return ds

    def match_coord(self, da, name=None, loc="any"):
        return self.coords.match(da, name=name, loc=loc)

    def match_data_var(self, da, name=None, loc="any"):
        return self.data_vars.match(da, name=name, loc=loc)

    def search_coord(self, dsa, name=None, loc="any", get="obj"):
        return self.coords.search(dsa, name=name, loc=loc, get=get)

    def search_data_var(self, dsa, name=None, loc="any", get="obj"):
        return self.data_vars.search(dsa, name=name, loc=loc, get=get)


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
        assert name in self, "Invalid entry name:" + name

    @property
    def names(self):
        return list(self._dict.keys())

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def get_specs(self, name, errors="warn"):
        """Get the specs of cf item or xarray.DataArray

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

    def register(self, name, **specs):
        """Register a new element from its name and explicit specs"""
        data = {self.category: {name: specs}}
        self.parent.register_from_cfg(data)

    def register_from_cfg(self, cfg):
        """Register new elements from a config specs"""
        if isinstance(cfg, dict) and self.category not in cfg:
            cfg = {self.category: cfg}
        self.parent.register_from_cfg(cfg)

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
        name: str, dict
        loc: str, False
        """
        names = self.names if name is None else [name]
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

    def search(self, dsa, name=None, loc="any", get="obj"):
        """Search for a data_var or coord that maches given or any specs

        Parameters
        ----------
        dsa: DataArray or Dataset
        name: str, dict
        loc: "any", str
        get: {"obj", "name"}
            When found, get the object found or its name.

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
            match_specs = []
            for attr, refs in self._get_ordered_match_specs_(name).items():
                match_specs.append({attr: refs})
        else:
            match_specs = [None]

        # Loops
        assert get in ("name", "obj", "both"), (
            "'get' must be either 'name' or 'obj' or 'both'")
        for match_arg in match_specs:
            for obj in objs.values():
                m = self.match(obj, match_arg, loc=loc)
                if m:
                    if get == "both":
                        return obj, name
                    return obj if get == "obj" else (name if name else m)

    def get_attrs(
        self, name, select=None, exclude=None, errors="warn", loc=None, **extra
    ):
        """Get the default attributes from cf specs

        Parameters
        ----------
        name: str
            Valid cf name
        select: str, list
            Include only these attributes
        exclude: str, list
            Exclude these attributes
        mode: "silent", "warning" or "error".
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
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    continue
                value = value[0]

            # Store it
            attrs[key] = value

        # Extra attributes
        attrs.update(extra)

        # Finalize and optionally change location
        attrs = self.parent.sglocator.format_attrs(attrs, loc=loc)

        return attrs

    def format_dataarray(self, da, name=None, loc=None, standardize=True):
        if name is None:
            name = self.match(da, loc="any")
        if name is None:
            return da.copy()
        assert name in self.names
        return self.sglocator.format_dataarray(
            da,
            loc=loc,
            name=name,
            attrs=self.get_attrs(name),
            standardize=standardize,
        )


class CFVarSpecs(_CFCatSpecs_):

    category = "data_vars"


class CFCoordSpecs(_CFCatSpecs_):

    category = "coords"


def _get_cfgm_():
    """Get a :class:`~xoa.cfgm.ConfigManager` instance to manage
    coords and data_vars spécifications"""
    if "cfgm" not in _CACHE:
        from .cfgm import ConfigManager

        _CACHE["cfgm"] = ConfigManager(_INIFILE)
    return _CACHE["cfgm"]


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


def clean_cf_specs_cache():
    """Clean the cf specs cahche file"""
    if os.path.exists(_PYKFILE):
        os.remove(_PYKFILE)


def show_cf_specs_cache():
    """Show the cf specs cahche file"""
    print(_PYKFILE)


def get_cf_specs(name=None, category=None, cache="rw"):
    """Get the CF specifications for a target in a category

    Parameters
    ----------
    name: str or None
        A target name like "sst". If not provided, return all specs.
    category: str or None
        Select a category with "coords" or "data_vars".
        If not provided, search first in "data_vars", then "coords".
    cache: bool
        Cache specs on disk with pickling for fast loading

    Return
    ------
    dict or None
        None is return if no specs are found
    """
    if cache is True:
        cache = "rw"
    elif cache is False:
        cache = "ignore"
    assert cache in ("ignore", "rw", "read", "write", "clean")

    # Get the source of specs
    if "specs" not in _CACHE:

        # Try from disk cache
        if cache in ("read", "w"):
            if os.path.exists(_PYKFILE) and (
                os.stat(_CFGFILE).st_mtime < os.stat(_PYKFILE).st_mtime
            ):
                try:
                    with open(_PYKFILE, "rb") as f:
                        set_cf_specs(pickle.load(f))
                except Exception as e:
                    xoa_warn(
                        "Error while loading cached cf specs: " + str(e.args)
                    )

        # Compute it from scratch
        if "specs" not in _CACHE:

            # Setup
            set_cf_specs(CFSpecs((_CFGFILE, "default")))

            # Cache it on disk
            if cache in ("write", "rw"):
                try:
                    cachedir = os.path.dirname(_PYKFILE)
                    if not os.path.exists(cachedir):
                        os.makedirs(cachedir)
                    with open(_PYKFILE, "wb") as f:
                        pickle.dump(_CACHE["specs"], f)
                except Exception as e:
                    xoa_warn("Error while caching cf specs: " + str(e.args))

    # Clean cache
    if cache == "clean":
        clean_cf_specs_cache()

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


def get_cf_coord_specs(name=None):
    """Shortcut to ``get_cf_specs(name=name, category='coords')``"""
    return get_cf_specs(name=name, category="coords")


def get_cf_var_specs(name=None):
    """Shortcut to ``get_cf_specs(name=name, category='data_vars')``"""
    return get_cf_specs(name=name, category="data_vars")


class _CFAccessor_(object):
    _category = None

    def __init__(self, dsa):
        self._cfspecs = get_cf_specs()
        self._dsa = dsa

    def get(self, name, loc="any"):
        return self._cfspecs[self._category].search(
            self._dsa, name=name, loc=loc, get="obj")

    def __getattr__(self, attr):
        if attr in self._cfspecs['accessors']['properties'][self._category]:
            return self.get(attr)
        return object.__getattribute__(self, attr)

    def __getitem__(self, name):
        return self.get(name)


class DataArrayCFAccessor(_CFAccessor_):
    _category = 'coords'


class DatasetCFAccessor(_CFAccessor_):
    _category = 'data_vars'


def register_accessors(name='cf'):
    """Register xarray accessors"""
    # cfspecs = cfspecs or get_cf_specs()
    # name = cfspecs['accessors']['name']
    import xarray as xr
    xr.register_dataarray_accessor(name)(DataArrayCFAccessor)
    xr.register_dataset_accessor(name)(DatasetCFAccessor)
