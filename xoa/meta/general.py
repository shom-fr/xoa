#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naming convention tools for reading and formatting variables

.. rubric:: How to use it

See the :ref:`uses.meta` section.

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

import os
import pprint
import copy

from .. import exceptions
from .. import options
from .. import misc
from . import configs

from . import sglocator
from . import categories

# Argument passed to dict_merge to merge Meta configs
_META_DICT_MERGE_KWARGS = dict(
    mergesubdicts=True,
    mergelists=True,
    skipnones=False,
    skipempty=False,
    overwriteempty=True,
    mergetuples=True,
    unique=True,
)

ATTRS_PATCH_MODE = misc.Choices(
    {
        "fill": "do not erase existing attributes, just fill missing ones",
        "replace": "replace existing attributes",
    },
    parameter="mode",
    description="policy for patching existing attributes",
)


def _get_cfgm_():
    """Get a :class:`~xoa.cfgm.ConfigManager` instance to manage
    coords and data_vars spécifications"""
    from . import get_cache

    meta_specs_cache = get_cache()
    if "cfgm" not in meta_specs_cache:
        from ..cfgm import ConfigManager
        from . import INI_FILE

        meta_specs_cache["cfgm"] = ConfigManager(INI_FILE)
    return meta_specs_cache["cfgm"]


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
            exceptions.xoa_warn(
                f"Cannot rename {old_name} to {new_name} since "
                f"{used[new_name]} will also be renamed to {new_name}. Skipping..."
            )
        else:
            used[new_name] = old_name
    return rename_args


class MetaSpecs(categories._MetaBase_):
    """Manager for Meta specifications

    Meta specifications are defined here an extension of a subset of
    Meta conventions: known variables and coordinates are described through
    a generic name, a categories name, alternates names, some properties
    and attributes like standard_name, long_name, axis.

    Have a look to the :ref:`default specifications <appendix.meta_specs.default>`
    and to the :ref:`uses.meta_specs` section.


    Parameters
    ----------
    cfg: str, list, MetaSpecs, dict
        A config file name or string or dict or Meta Specs, or a list of them.
        It may contain the "data_vars", "coords"  and "sglocator" sections.
        When a list is provided, specs are merged with the firsts having
        priority over the lasts.
    default: bool
        Load also the default internal specs
    user: bool
        Load also the user specs stored in :data:`USER_META_FILE`
    name: str, None
        Assign a shortcut name. It defaults the the `[register] name`
        option of the specs.
    cache: bool
        Use in-memory cache system?

    See also
    --------
    MetaCoordSpecs
    MetaVarSpecs
    SGLocator
    :ref:`uses.meta_specs`
    :ref:`appendix.meta_specs.default`
    """

    def __init__(self, cfg=None, default=True, user=True, name=None, cache=None):
        # Initialiase categories
        self._metas = {}
        catcls = {"data_vars": categories.MetaVarSpecs, "coords": categories.MetaCoordSpecs}
        for category in self.categories:
            self._metas[category] = catcls[category](self)

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
        meta: str, dict, MetaSpecs
            Config source
        cache: bool
            Use in-memory cache system?

        """
        from . import get_cache

        # Config manager to get defaults and validation
        cfgm = _get_cfgm_()

        # Get it from cache if from str or MetaSpecs with registration name
        if cache is None:
            cache = options.get_option("cf.cache")
        cache = cache and (
            (isinstance(cfg, str) and "\n" not in cfg)
            or (isinstance(cfg, dict) and "register" in cfg and cfg["register"]["name"])
        )
        if cache:
            # Init cache
            if isinstance(cfg, str):
                cache_key = cfg
            elif isinstance(cfg, dict) and "register" in cfg and cfg["register"]["name"]:
                cache_key = cfg["register"]["name"]
            meta_cache = get_cache()
            if cache_key in meta_cache["loaded_dicts"]:
                # a copy is needed because of the post processing
                return copy.deepcopy(meta_cache["loaded_dicts"][cache_key])

        # Check input type
        if isinstance(cfg, str) and "\n" in cfg:  # multi-line content
            cfg = cfg.split("\n")
        elif isinstance(cfg, MetaSpecs):
            cfg = cfg._dict
        elif isinstance(cfg, str) and cfg in configs.META_CONFIGS:
            cfg = configs.META_CONFIGS[cfg]  # full path to internal config file

        # Load, validate and convert to dict
        cfg_dict = cfgm.load(cfg).dict()

        # Cache it
        if cache:
            # a copy is needed because of the post processing
            meta_cache["loaded_dicts"][cache_key] = copy.deepcopy(cfg_dict)

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
        from . import USER_META_FILE, get_meta_config_file

        # Get the list of validated configurations
        to_load = []
        if cfg:
            if not isinstance(cfg, tuple):
                cfg = (cfg,)
            to_load.extend([c for c in cfg if c])
        if self._load_user and os.path.exists(USER_META_FILE):
            to_load.append(USER_META_FILE)
        if self._load_default:
            to_load.append(get_meta_config_file("default"))
        if not to_load:
            to_load = [None]

        # Load them
        dicts = [self._load_cfg_as_dict_(cfg, cache) for cfg in to_load]

        # Merge them, except "register"
        self._dict = misc.dict_merge(*dicts, **_META_DICT_MERGE_KWARGS)
        self._dict["register"] = dicts[0]["register"]

        # SG locator
        self._sgl_settings = self._dict["sglocator"]
        self._sgl = sglocator.SGLocator(**self._sgl_settings)

        # Post process
        self._post_process_()

    def copy(self):
        return MetaSpecs(self, default=False, user=False)

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

    @property
    def excluded_names(self):
        """Data array names that are excluded from all parsings"""
        return self._dict["exclude_names"] or []

    def pprint(self, **kwargs):
        """Pretty print the specs as dict using :func:`pprint.pprint`"""
        pprint.pprint(self.dict, **kwargs)

    @property
    def categories(self):
        """List of meta specs categories"""
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
            return self._metas[section]
        for cat in self.categories:
            if section in self._metas[cat]:
                return self._metas[cat][section]
        return self._dict[section]

    def __contains__(self, category):
        return category in self.categories

    def __getattr__(self, name):
        if "_metas" in self.__dict__ and name in self.__dict__["_metas"]:
            return self.__dict__["_metas"][name]
        if name == "dims":
            return self._dict["dims"]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(self.__class__.__name__, name)
        )

    @property
    def dims(self):
        """Dictionary of dims per dimension type within x, y, z, t and f"""
        return self._dict["dims"]

    @property
    def coords(self):
        """Specifications for coords :class:`MetaCoordSpecs`"""
        return self._metas["coords"]

    @property
    def data_vars(self):
        """Specification for data_vars :class:`MetaVarSpecs`"""
        return self._metas["data_vars"]

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

        # Exclude
        for category in self.categories:
            items[category] = dict(items[category])
            for name in self[category].names:
                if items[category][name]["exclude"]:
                    del items[category][name]

        # Refill
        for category in items:
            self._dict[category].clear()
            self._dict[category].update(items[category])

        # Updates valid dimensions
        alt_names = {}
        for name, coord_specs in self._dict["coords"].items():
            if coord_specs["attrs"]["axis"]:
                axis = coord_specs["attrs"]["axis"].lower()
                self._dict["dims"][axis].append(name)  # generic name
                if coord_specs["name"]:  # categories names
                    self._dict["dims"][axis].append(coord_specs["name"])
                alt_names.setdefault(axis, []).extend(coord_specs["alt_names"])  # alternate names
        for axis in self._dict["dims"]:
            if axis in alt_names:
                self._dict["dims"][axis].extend(alt_names[axis])

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
            ), "Cannot inherit meta specs from it self"

            # Parents must be already processed
            for item in self._process_entry_(from_cat, from_name):
                yield item

            # Inherit with merging
            entries[name] = specs = misc.dict_merge(
                specs,
                self._dict[from_cat][from_name],
                cls=dict,
                **_META_DICT_MERGE_KWARGS,
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

        specs["processed"] = True
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

    def get_loc_mapping(self, obj, meta_names=None, loc=None, categories=["coords", "data_vars"]):
        """Associate a location to each identified variables, coordinates and dimensions of obj

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
        meta_names: dict, None
            Dict with names as keys and generic meta names as values.
            If not provided, :meth:`match` is used to guess meta names.

        Return
        ------
        dict
            Keys are item names and values are location.
            This dict is also stored in the `meta_locs` key of the `encoding`
            attribute of `obj`.
        """
        locations = {}
        isdataset = hasattr(obj, "data_vars")
        das = obj.values() if isdataset else [obj]

        # Process data variables
        self._process_data_var_locations_(das, categories, meta_names, isdataset, loc, locations)

        # Process coordinates
        self._process_coord_locations_(obj, meta_names, locations)

        # Infer locations from dimensions
        self._infer_locations_from_dims_(das, locations)

        return locations

    def _check_coord_locations_(self, da, specs, locations):
        """Scan add_coords_loc section and update locations for coordinates and dimensions.

        Parameters
        ----------
        da: xarray.DataArray
            Data array to scan
        specs: dict
            Meta specs for the data array
        locations: dict
            Dictionary to update with location mappings
        """
        for meta_coord_name, coord_loc in specs["add_coords_loc"].items():
            if self.coords[meta_coord_name]["attrs"].get("axis", "").lower() not in 'xyz':
                continue

            loc = specs["loc"] if coord_loc is True else coord_loc

            # Coordinates
            coord = self.search_coord(da, meta_coord_name, errors="ignore")
            if coord is not None and locations.get(coord.name) is None:
                locations[coord.name] = loc
                continue

            # Dimensions
            dim = self.search_dim(da, meta_coord_name, errors="ignore")
            if dim is not None and locations.get(dim) is None:
                locations[dim] = loc

    def _process_data_var_locations_(self, das, categories, meta_names, isdataset, loc, locations):
        """Process data variables and assign locations.

        Parameters
        ----------
        das: list
            List of data arrays
        categories: list
            Categories to search in (coords, data_vars)
        meta_names: dict, None
            Meta names mapping
        isdataset: bool
            Whether obj is a dataset
        loc: str, None
            Location override for DataArray
        locations: dict
            Dictionary to update with location mappings
        """
        for da in das:
            for cat in categories:
                meta_name = meta_names.get(da.name) if meta_names else self[cat].match(da)
                if meta_name and meta_name in self[cat]:
                    specs = self[cat][meta_name].copy()
                    if not isdataset and loc is not None:
                        specs["loc"] = loc
                    if specs["add_loc"] is not False:
                        if specs["loc"] is None:  # infer from da
                            locations[da.name] = self.sglocator.get_loc_from_da(da)
                            if locations[da.name] is None and specs["add_loc"] is True:
                                locations[da.name] = True
                        else:
                            locations[da.name] = specs["loc"]  # from config
                    else:
                        locations[da.name] = False
                    self._check_coord_locations_(da, specs, locations)
                    break  # good category

    def _process_coord_locations_(self, obj, meta_names, locations):
        """Process coordinates and assign locations.

        Parameters
        ----------
        obj: xarray.Dataset or xarray.DataArray
            Object with coordinates
        meta_names: dict, None
            meta names mapping
        locations: dict
            Dictionary to update with location mappings
        """
        for coord in obj.coords.values():
            meta_coord_name = meta_names.get(coord.name) if meta_names else self.coords.match(coord)
            if (
                meta_coord_name
                and (self.coords[meta_coord_name]["attrs"]["axis"] or "").lower() in "xyz"
            ):
                if locations.get(coord.name) is None:
                    if self.coords[meta_coord_name]["add_loc"] is not False:
                        if self.coords[meta_coord_name]["loc"] is None:
                            # infer from da
                            locations[coord.name] = self.sglocator.get_loc_from_da(coord)
                            if (
                                locations[coord.name] is None
                                and self.coords[meta_coord_name]["add_loc"] is True
                            ):
                                locations[coord.name] = True
                        else:
                            # from config
                            locations[coord.name] = self.coords[meta_coord_name]["loc"]
                    else:
                        locations[coord.name] = False
                self._check_coord_locations_(coord, self.coords[meta_coord_name], locations)

    def _infer_locations_from_dims_(self, das, locations):
        """Infer locations for data vars from their dimensions when location is True.

        If a data array has location True and all its dimensions have the same
        unique location, assign that location to the array.

        Parameters
        ----------
        das: list
            List of data arrays
        locations: dict
            Dictionary to update with location mappings
        """
        for da in das:
            if locations.get(da.name) is True:
                loc = None
                for dim in self._list_xr_names_(da):
                    dim_loc = locations.get(dim)
                    if dim_loc is not None:
                        if loc is None:
                            loc = dim_loc
                        elif loc != dim_loc:
                            break  # multiple locs so no loc
                else:
                    locations[da.name] = loc

    def _format_data_arrays_(
        self,
        obj,
        data_vars,
        categories,
        meta_names,
        attrs,
        locations,
        kwargs,
        rename,
        rename_args,
        is_dataset,
    ):
        """Format data arrays and collect rename arguments.

        Parameters
        ----------
        obj: xarray.Dataset or xarray.DataArray
        data_vars: list
            List of data arrays to format
        categories: list
            Categories to search in (coords, data_vars)
        meta_names: dict, None
            Meta names mapping
        attrs: bool, dict
            Attributes to apply
        locations: dict
            Location mapping
        kwargs: dict
            Common formatting arguments
        rename: bool
            Whether renaming is enabled
        rename_args: dict
            Dictionary to collect rename arguments
        is_dataset: bool
            Whether obj is a dataset
        """
        for da in data_vars:
            if da.name in self.excluded_names:
                continue
            for cat in categories:
                meta_name = meta_names.get(da.name) if meta_names else None
                if meta_name and meta_name not in self[cat]:
                    continue
                new_name = self[cat].format_dataarray(
                    da,
                    meta_name=meta_name,
                    attrs=attrs if isinstance(attrs, bool) else attrs.get(da.name),
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

    def _format_coordinates_(self, obj, meta_names, attrs, locations, kwargs, rename, rename_args):
        """Format coordinates and collect rename arguments.

        Parameters
        ----------
        obj: xarray.Dataset or xarray.DataArray
        meta_names: dict, None
            Meta names mapping
        attrs: bool, dict
            Attributes to apply
        locations: dict
            Location mapping
        kwargs: dict
            Common formatting arguments
        rename: bool
            Whether renaming is enabled
        rename_args: dict
            Dictionary to collect rename arguments
        """

        for cname in self._list_xr_names_(obj, data_vars=False, dims=False):
            if cname in self.excluded_names:
                continue
            cda = obj.coords[cname]
            new_coord_name = self.coords.format_dataarray(
                cda,
                meta_name=meta_names.get(cname) if isinstance(meta_names, dict) else None,
                attrs=attrs if isinstance(attrs, bool) else attrs.get(cname, True),
                loc=locations.get(cda.name),
                **kwargs,
            )
            if rename and new_coord_name:
                rename_args[cda.name] = new_coord_name

    def _apply_renames_(self, obj, rename_args, rename_dims, locations, specialize, rename):
        """Apply dimension and final renaming.

        Parameters
        ----------
        obj: xarray.Dataset or xarray.DataArray
        rename_args: dict
            Dictionary of rename mappings
        rename_dims: bool
            Whether to rename dimensions
        locations: dict
            Location mapping
        specialize: bool
            Whether to use categories names
        rename: bool
            Whether renaming is enabled

        Returns
        -------
        xarray.Dataset or xarray.DataArray
            Renamed object
        """
        if rename_dims:
            rename_dims_args = self.coords.get_rename_dims_args(
                obj, locations=locations, specialize=specialize
            )
            rename_args.update(rename_dims_args)

        if rename and rename_args:
            _solve_rename_conflicts_(rename_args)
            obj = obj.rename(rename_args)

        return obj

    def _format_obj_(
        self,
        obj,
        meta_names=None,
        rename=True,
        standardize=True,
        format_coords=True,
        copy=True,
        replace_attrs=False,
        attrs=True,
        loc=None,
        specialize=False,
        rename_dims=None,
        categories=["coords", "data_vars"],
        rename_args=None,
    ):
        """Auto-format a whole xarray.Dataset

        See also
        --------
        format_data_var
        format_coord
        """
        # Copy and initialization
        if copy:
            obj = obj.copy(deep=False)
        if rename_args is None:
            rename_args = {}
        if rename_dims is None:
            rename_dims = format_coords

        # Common formatting kwargs
        kwargs = dict(
            copy=False,
            rename=False,
            replace_attrs=replace_attrs,
            standardize=True,
            specialize=specialize,
        )

        # Get staggered grid locations
        locations = self.get_loc_mapping(obj, meta_names=meta_names, loc=loc, categories=categories)

        # Format data arrays
        is_dataset = hasattr(obj, "data_vars")
        data_vars = obj.values() if is_dataset else [obj]
        self._format_data_arrays_(
            obj,
            data_vars,
            categories,
            meta_names,
            attrs,
            locations,
            kwargs,
            rename,
            rename_args,
            is_dataset,
        )

        # Format coordinates
        if format_coords:
            self._format_coordinates_(
                obj, meta_names, attrs, locations, kwargs, rename, rename_args
            )

        # Apply renaming
        obj = self._apply_renames_(obj, rename_args, rename_dims, locations, specialize, rename)

        return obj

    def format_coord(
        self,
        da,
        meta_name=None,
        loc=None,
        copy=True,
        format_coords=True,
        standardize=True,
        rename=True,
        specialize=False,
        attrs=True,
        replace_attrs=False,
        # add_loc_to_name=None,
        rename_dims=True,
        rename_args=None,
    ):
        """Format a coordinate array

        Parameters
        ----------
        da: xarray.DataArray
        meta_name: str, None
            A generic meta name. If not provided, it guessed with :meth:`match`.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        rename: bool
            Rename arrays
        add_loc_to_name: bool
            Add loc to the name
        specialize: bool
            Does not use the meta name for renaming, but the first name
            as listed in specs, which is generally a categories one,
            like a name adopted by categories dataset.
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
        rename_args: dict, None
            Dictionay to collect args to rename

        Returns
        -------
        xarray.DataArray, str, None
            The formatted array or copy of it.
            The meta name, given or matching, if rename if False; and None
            if not matching.

        See also
        --------
        MetaCoordSpecs.format_dataarray
        """
        return self._format_obj_(
            da,
            meta_names={da.name: meta_name},
            copy=copy,
            standardize=standardize,
            rename=rename,
            rename_dims=rename_dims,
            format_coords=format_coords,
            replace_attrs=replace_attrs,
            attrs=attrs if isinstance(attrs, bool) else {da.name: attrs},
            specialize=specialize,
            categories=["coords"],
            loc=loc,
            rename_args=rename_args,
        )

    def format_data_var(
        self,
        da,
        meta_name=None,
        loc=None,
        copy=True,
        rename=True,
        rename_dims=True,
        specialize=False,
        format_coords=True,
        attrs=True,
        replace_attrs=False,
        standardize=True,
        rename_args=None,
        # add_loc_to_name=None,
    ):
        """Format a data_var array

        Parameters
        ----------
        da: xarray.DataArray
        meta_name: str, None
            A generic meta name. If not provided, it guessed with :meth:`match`.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        rename: bool
            Rename arrays
        add_loc_to_name: bool
            Add loc to the name
        specialize: bool
            Does not use the meta name for renaming, but the first name
            as listed in specs, which is generally a categories one,
            like a name adopted by categories dataset.
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
        rename_args: dict, None
            Dictionay to collect args to rename

        Returns
        -------
        xarray.DataArray, str, None
            The formatted array or copy of it.
            The meta name, given or matching, if rename if False; and None
            if not matching.

        See also
        --------
        MetaCoordSpecs.format_dataarray
        """
        return self._format_obj_(
            da,
            meta_names={da.name: meta_name},
            copy=copy,
            rename=rename,
            rename_dims=rename_dims,
            specialize=specialize,
            format_coords=format_coords,
            replace_attrs=replace_attrs,
            attrs=attrs if isinstance(attrs, bool) else {da.name: attrs},
            standardize=standardize,
            categories=["coords", "data_vars"],
            loc=loc,
            rename_args=rename_args,
        )

    def format_dataset(
        self,
        ds,
        meta_names=None,
        # loc=None,
        copy=True,
        format_coords=True,
        standardize=True,
        rename=True,
        rename_dims=True,
        specialize=False,
        attrs=True,
        replace_attrs=False,
        rename_args=None,
        # add_loc_to_name=None
    ):
        """Format a whole dataset

        Parameters
        ----------
        ds: xarray.Dataset
        meta_names: dict, None
            Dict of names as keys and generic meta names as values.
            If not provided, meta names are guessed with :meth:`match`.
        loc: str, {"any", None}, {"", False}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        rename: bool
            Rename arrays
        add_loc_to_name: bool
            Add loc to the name
        specialize: bool
            Does not use the meta name for renaming, but the first name
            as listed in specs, which is generally a categories one,
            like a name adopted by categories dataset.
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
        rename_args: dict, None
            Dictionay to collect args to rename

        Returns
        -------
        xarray.DataArray, str, None
            The formatted array or copy of it.
            The meta name, given or matching, if rename if False; and None
            if not matching.

        See also
        --------
        MetaCoordSpecs.format_dataarray
        """
        return self._format_obj_(
            ds,
            meta_names=meta_names,
            copy=copy,
            standardize=standardize,
            rename=rename,
            rename_dims=rename_dims,
            format_coords=format_coords,
            replace_attrs=replace_attrs,
            attrs=attrs,
            specialize=specialize,
            categories=["coords", "data_vars"],
            rename_args=rename_args,
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

    def decode(self, obj, set_encoding=True, **kwargs):
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

        # Assign meta specs
        if self.name:
            from . import is_registered_meta_specs, register_meta_specs, assign_meta_specs

            if not is_registered_meta_specs(self.name):
                register_meta_specs(self)
            obj = assign_meta_specs(obj, self.name, set_encoding=set_encoding)

        return obj

    def encode(self, obj, **kwargs):
        """Same as :meth:`decode` but rename with the categories name

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
        names = self._list_xr_names_(obj)
        if hasattr(obj, "name"):
            names = names.union({obj.name})
        for name in names:
            if name not in rename_args:
                root_name, old_loc = self.sglocator.parse_attr("name", name)
                if root_name in locs and locs[root_name] is not None:
                    rename_args[name] = self.sglocator.format_attr(
                        "name", root_name, locs[root_name]
                    )
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
        names = self._list_xr_names_(obj)
        for name in names:
            if name not in rename_args:
                root_name, old_loc = self.sglocator.parse_attr("name", name)
                if old_loc and old_loc in locs:
                    rename_args[name] = self.sglocator.format_attr("name", root_name, locs[old_loc])

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

    def match_coord(self, da, meta_name=None, loc="any"):
        """Check if an array matches a given or any coord specs

        Parameters
        ----------
        da: xarray.DataArray
        meta_name: str, dict, None
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
        MetaCoordSpecs.match
        """
        return self.coords.match(da, meta_name=meta_name, loc=loc)

    def match_data_var(self, da, meta_name=None, loc="any"):
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
        MetaVarSpecs.match
        """
        return self.data_vars.match(da, meta_name=meta_name, loc=loc)

    def match_dim(self, dim, meta_name=None, loc=None):
        """Check if a dimension name matches given or any coord specs

        Parameters
        ----------
        dim: str
            Dimension name
        meta_name: str, None
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
        MetaVarSpecs.match_from_name
        """
        return self.coords.match_from_name(dim, meta_name=meta_name, loc=loc)

    @classmethod
    def get_category(cls, da):
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
        if da.name is not None and misc.list_xr_names(da, data_vars=False):
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
            meta_name = self[category].match(da, loc=loc)
            if meta_name:
                return category, meta_name
        return None, None

    @misc.ERRORS.format_method_docstring
    def search_coord(
        self,
        obj,
        meta_name=None,
        loc="any",
        get="obj",
        single=True,
        errors="warn",
    ):
        """Search for a coord that maches given or any specs

        Parameters
        ----------
        obj: DataArray or Dataset
        meta_name: str, dict
            A generic meta name. If not provided, all meta names are scaned.
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
            from xoa.meta import get_meta_specs
            @suppress
            import xarray as xr, numpy as np
            lon = xr.DataArray([2, 3], dims='foo',
                               attrs={{'standard_name': 'longitude'}})
            data = xr.DataArray([0, 1], dims=('foo'), coords=[lon])
            meta_specs = get_meta_specs()
            meta_specs.search_coord(data, "lon")
            meta_specs.search_coord(data, "lon", get="meta_name")
            meta_specs.search_coord(data, "lat", errors="ignore")

        See also
        --------
        search_data_var
        MetaCoordSpecs.search
        """
        return self.coords.search(
            obj,
            meta_name=meta_name,
            loc=loc,
            get=get,
            single=single,
            errors=errors,
        )

    @misc.ERRORS.format_method_docstring
    def search_dim(self, da, meta_arg=None, loc="any", errors="ignore"):
        """Search for a dimension from its type

        Parameters
        ----------
        da: xarray.DataArray
        meta_arg: None, str, {{"x", "y", "z", "t", "f"}}
            Dimension type or generic meta name
        loc:
            Staggered grid location
        {errors}

        Return
        ------
        None, str, dict
            Dim name OR, dict with dim, type and meta_name keys if meta_arg is None

        See also
        --------
        MetaCoordSpecs.search_dim
        """
        return self.coords.search_dim(da, meta_arg=meta_arg, loc=loc, errors=errors)

    @misc.ERRORS.format_method_docstring
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

    @misc.ERRORS.format_method_docstring
    def search_data_var(
        self,
        obj,
        meta_name=None,
        loc="any",
        get="obj",
        single=True,
        errors="warn",
    ):
        """Search for a data_var that maches given or any specs

        Parameters
        ----------
        obj: DataArray or Dataset
        meta_name: str, dict
            A generic meta name. If not provided, all meta names are scaned.
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {{"obj", "name"}}
            When found, get the object found or its meta_name.
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
            from xoa.meta import get_meta_specs
            @suppress
            import xarray as xr, numpy as np
            data = xr.DataArray(
                [0, 1], dims=('x'),
                attrs={{'standard_name': 'sea_water_temperature'}})
            ds = xr.Dataset({{'foo': data}})
            meta_specs = get_meta_specs()
            meta_specs.search_data_var(ds, "temp")
            meta_specs.search_data_var(ds, "temp", get="meta_name")
            meta_specs.search_data_var(ds, "sal")

        See also
        --------
        search_coord
        MetaVarSpecs.search
        """
        return self.data_vars.search(
            obj,
            meta_name=meta_name,
            loc=loc,
            get=get,
            single=single,
            errors=errors,
        )

    @misc.ERRORS.format_method_docstring
    def search(
        self,
        obj,
        meta_name=None,
        loc="any",
        get="obj",
        single=True,
        categories=None,
        within=None,
        errors="warn",
    ):
        """Search for a dataarray with data_vars and/or coords


        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array or dataset to scan
        meta_name: str
            Generic meta name to search for.
        categories: str, list, None
            Explicty categories with "coords" and "data_vars".
        get: {{"obj", "meta_name"}}
            "Getthe object or its meta_name.
        within: str, None
            Object types to search within: "coords", "data_vars".
            Data vars are search only with "data_vars" and coordinates
            are both in "coords" and "data_vars".
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
        errors = misc.ERRORS[errors]
        if not single:
            found = []
        for category in categories:
            res = self[category].search(
                obj,
                meta_name=meta_name,
                loc=loc,
                get=get,
                single=single,
                within=within,
                errors="ignore",
            )
            if res is None:
                continue
            if not single:
                res = [r for r in res if r.name not in [f.name for f in found]]
                found.extend(res)
            elif res is not None:
                return res
        if not single:
            return found
        msg = "Search failed"
        if errors == "warn":
            exceptions.xoa_warn(msg)
        elif errors == "raise":
            raise exceptions.XoaMetaError(msg)

    @misc.ERRORS.format_method_docstring
    def get(self, obj, meta_name, get="obj", within=None, errors="ignore"):
        """A shortcut to :meth:`search` with an explicit generic meta name or a list of them

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array or dataset to scan
        meta_name: str, list(str)
            Generic meta name to search for.
            When a list, loop over possible meta_names and stop at the first found.
        get: {{"obj", "name"}}
            "Getthe object or its meta_name.
        within: str, None
            Object types to search within: "coords", "data_vars".
            Data vars are searched only with "data_vars" and coordinates
            are both in "coords" and "data_vars".
        {errors}

        A single element is searched for into all :attr:`categories`
        and errors are ignored.
        """
        meta_names = [meta_name] if isinstance(meta_name, str) else meta_name
        found = []
        for meta_name in meta_names:
            found.extend(
                self.search(obj, meta_name, errors="ignore", single=False, get=get, within=within)
            )
        return self._check_single_(errors, found, "item", meta_names)

    @misc.ERRORS.format_method_docstring
    def get_dims(
        self,
        da,
        meta_args,
        allow_positional=False,
        positions="tzyx",
        single=True,
        errors="warn",
    ):
        """Get the data array dimensions names from their type

        Parameters
        ----------
        da: xarray.DataArray
            Array to scan
        meta_args: str, list
            Letters among "x", "y", "z", "t" and "f",
            or generic meta names.
        allow_positional: bool
            Fall back to positional dimension of types is unkown.
        positions: str
            Default position per type starting from the end.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
        {errors}

        Return
        ------
        tuple
            Tuple of dimension name or None when the dimension if not found

        See also
        --------
        MetaCoordSpecs.get_dims
        """
        return self.coords.get_dims(
            da,
            meta_args,
            allow_positional=allow_positional,
            positions=positions,
            single=single,
            errors=errors,
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

        - This dimension is registered in Meta dims.
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
        return self.coords.get_dim_type(dim, obj=da, lower=lower)

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
        """Convert from generic dim names to categories names

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
        return ds.copy(deep=False)
