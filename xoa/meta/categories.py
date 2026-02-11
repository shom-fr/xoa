#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naming convention tools for reading and formatting variables

.. rubric:: How to use it

See the :ref:`uses.cf` section.

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


from .. import exceptions
from .. import misc


class _MetaBase_:

    @staticmethod
    def _list_xr_names_(obj, data_vars=True, coords=True, dims=True):
        """List the data vars, coords and dims names of a xarray dataset or data array"""
        return misc.list_xr_names(obj, data_vars, coords, dims)

    @staticmethod
    def _check_single_(errors, found, target_type, targets=None):
        """Check that we found a single item"""

        singletons = []
        for item in found:
            if hasattr(item, "name"):
                item = item.name
            if item not in singletons:
                singletons.append(item)
        nitems = len(singletons)

        # Single one so its ok
        if nitems == 1:
            return found[0]

        if targets is not None:
            if not isinstance(targets, str):
                targets = ", ".join(targets)
            suffix = f" matching '{targets}'"
        errors = misc.ERRORS[errors]

        # Multiple
        if nitems > 1:
            if errors != "ignore":
                msg = f"Found multiple {target_type}s{suffix} instead of single one"
                if errors == "raise":
                    raise exceptions.XoaMetaError(msg)
                msg += ". Returning the first item."
                exceptions.xoa_warn(msg, stacklevel=3)
                return found[0]
            else:
                return found[0]

        # No one
        if errors != "ignore":
            msg = f"No {target_type} found{suffix}"
            if errors == "raise":
                raise exceptions.XoaMetaError(msg)
            exceptions.xoa_warn(msg, stacklevel=3)


class _MetaCatSpecs_(_MetaBase_):
    """Base class for loading data_vars and coords Meta specifications"""

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

    def _validate_name_(self, name):
        if name in self:
            return name

    def _assert_known_(self, name, errors="raise"):
        if name not in self._dict:
            if errors == "raise":
                raise exceptions.XoaMetaError(f"Invalid {self.category} specs name: " + name)
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
        """Get the specs of a meta item

        Parameters
        ----------
        name: str
        errors: "ignore", "warning" or "raise".

        Return
        ------
        dict or None
        """
        errors = misc.ERRORS[errors]
        if name not in self._dict:
            if errors == "raise":
                raise exceptions.XoaMetaError("Can't get meta specs from: " + name)
            if errors == "warn":
                exceptions.xoa_warn("Invalid meta name: " + str(name))
            return
        return self._dict[name]

    @property
    def dims(self):
        """Dict of dim names per type"""
        return self.parent._dict["dims"]

    def set_specs(self, item, **specs):
        """Update or create specs for an item"""
        data = {self.category: {item: specs}}
        self.parent.load_cfg(data)

    def set_specs_from_cfg(self, cfg):
        """Update or create specs for several item with a config specs"""
        if isinstance(cfg, dict) and self.category not in cfg:
            cfg = {self.category: cfg}
        self.parent.load_cfg(cfg)

    def get_allowed_names(self, meta_name):
        """Get the list of allowed names for a given `meta_name`

        It includes de `meta_name` itself, the `name` alt_names` specification values

        Parameters
        ----------
        meta_name: str
            Valid Meta name

        Return
        ------
        list
        """
        specs = self[meta_name]
        allowed_names = []
        if "name" in specs and specs["name"]:
            allowed_names.append(specs["name"])
        if "alt_names" in specs:
            allowed_names.extend(specs["alt_names"])
        allowed_names.append(meta_name)
        return allowed_names

    def get_loc_mapping(self, obj, meta_names=None):
        return self.parent.get_loc_mapping(
            obj, meta_names=meta_names, categories=["coords", "data_vars"]
        )

    def _get_ordered_match_specs_(self, meta_name):
        specs = self[meta_name]
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
                if attr == "name":
                    match_specs["name"] = self.get_allowed_names(meta_name)
                elif "attrs" in specs and attr in specs["attrs"]:
                    match_specs[attr] = specs["attrs"][attr]
        return match_specs

    def _get_search_scope_(self, obj, within):
        """Determine search scope and get list of names to search.

        Parameters
        ----------
        obj: xarray.DataArray or xarray.Dataset
            Object to search in
        within: str, list, None
            Object types to search within

        Returns
        -------
        list
            List of names to search

        Raises
        ------
        XoaMetaError
            If within parameter is invalid
        """
        if within is None:
            if not hasattr(obj, "data_vars"):
                within = ["coords"]
            else:
                within = ["coords", "data_vars"]
                if self.category == "data_vars":
                    within = within[::-1]
        else:
            if isinstance(within, str):
                within = [within]
            for ot in within:
                if ot not in ["coords", "data_vars"]:
                    raise exceptions.XoaMetaError(
                        "with parameter must one or a list of: coords, data_vars"
                    )

        return self._list_xr_names_(
            obj, dims=False, coords="coords" in within, data_vars="data_vars" in within
        )

    def _build_match_specs_(self, meta_name, errors):
        """Build match specifications for search.

        Parameters
        ----------
        meta_name: str, dict, None
            Meta name to search for
        errors: str
            Error handling mode

        Returns
        -------
        list
            List of match specifications
        """
        if meta_name:  # Explicit name so we loop on search specs
            if isinstance(meta_name, str):
                if not self._assert_known_(meta_name, errors):
                    return None
            match_specs = []
            for attr, refs in self._get_ordered_match_specs_(meta_name).items():
                match_specs.append({attr: refs})
        else:
            match_specs = [None]
        return match_specs

    def _search_in_objects_(self, obj, names, match_specs, loc, get, meta_name):
        """Search for matching objects in the given list of names.

        Parameters
        ----------
        obj: xarray.DataArray or xarray.Dataset
            Object containing the arrays to search
        names: list
            List of names to search through
        match_specs: list
            List of match specifications
        loc: str, None
            Location parameter
        get: str
            What to return ("obj", "meta_name", or "both")
        meta_name: str, None
            Original meta_name argument

        Returns
        -------
        list
            List of found items
        """
        assert get in (
            "meta_name",
            "obj",
            "both",
        ), f"'get' must be either 'meta_name' or 'obj' or 'both', NOT '{get}'"

        found = []
        found_objs = []
        for match_arg in match_specs:
            for this_name in names:
                this_obj = obj[this_name]
                m = self.match(this_obj, match_arg, loc=loc)
                if m:
                    if this_name in found_objs:
                        continue
                    found_objs.append(this_name)
                    matched_meta_name = meta_name if meta_name else m
                    if get == "both":
                        found.append((this_obj, matched_meta_name))
                    else:
                        found.append(this_obj if get == "obj" else matched_meta_name)
        return found

    def match(self, da, meta_name=None, loc=None):
        """Check if da attributes match given or any specs

        Parameters
        ----------
        da: xarray.DataArray
        meta_name: str, dict, None
            Meta name.
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
        if meta_name:
            if isinstance(meta_name, (str, dict)):
                names = [meta_name]
            else:
                names = []
                for name in meta_name:
                    self._assert_known_(name)
                    names.append(name)
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
                    ) or misc.match_string(value, ref, ignorecase=True):
                        da.encoding["meta_name"] = name_
                        da.encoding["meta_category"] = self.category
                        return True if meta_name else name_
        return False if meta_name else None

    def match_from_name(self, name, meta_name=None, loc=None):
        """Get the generic Meta name of an object knowing only its name

        It compares `name` to the `name` and `alt_names` config values.

        Parameters
        ----------
        name: str
            Actual name
        meta_name: str, None
            A target generic Meta name. If not provided, all items are considered.

        Return
        ------
        None, str, True, False
            If `meta_name` is provided, returns a boolean, else returns
            the matching Meta name or None.
        """
        explicit = meta_name is not None
        if meta_name:
            self._assert_known_(meta_name)
            meta_names = [meta_name]
        else:
            meta_names = self.names

        for meta_name in meta_names:
            for allowed_name in self.get_allowed_names(meta_name):
                if self.sglocator.match_attr("name", name, allowed_name, loc=loc):
                    return meta_name if not explicit else True
        return None if not explicit else False

    @misc.ERRORS.format_method_docstring
    def search(
        self, obj, meta_name=None, loc=None, get="obj", single=True, within=None, errors="raise"
    ):
        """Search for a data_var or coord that maches given or any specs

        Parameters
        ----------
        obj: DataArray or Dataset
        meta_name: str, dict
            A generic Meta name. If not provided, all Meta names are scaned.
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or '"": no location
        get: {{"obj", "meta_name", "both"}}
            When found, get the object found or its name.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
            A warning is emitted when set to True and multiple item are found.
        within: str, None
            Object types to search within: "coords", "data_vars".
            Data vars are searched only with "data_vars" and coordinates
            are both in "coords" and "data_vars".
        {errors}

        Returns
        -------
        None or str or object
        """
        # Get search scope
        names = self._get_search_scope_(obj, within)

        # Build match specifications
        match_specs = self._build_match_specs_(meta_name, errors)
        if match_specs is None:
            return

        # Search in objects
        found = self._search_in_objects_(obj, names, match_specs, loc, get, meta_name)

        # Return result
        if not single:
            return found
        return self._check_single_(errors, found, "item", meta_name)

    @misc.ERRORS.format_method_docstring
    def get(self, obj, meta_name, get="obj", errors="ignore"):
        """A shortcut to :meth:`search` with an explicit generic Meta name or a list of them

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array or dataset to scan
        meta_name: str, list(str)
            Generic Meta name to search for.
            When a list, loop over possible meta_names and stop at the first found.
        get: {{"obj", "name"}}
            "Getthe object or its meta_name.
        {errors}

        A single element is searched for into all :attr:`categories`
        and errors are ignored.
        """
        meta_names = [meta_name] if isinstance(meta_name, str) else meta_name
        found = []
        for meta_name in meta_names:
            found.extend(self.search(obj, meta_name, errors="ignore", single=False, get=get))
        return self._check_single_(errors, found, "item", meta_names)

    @misc.ERRORS.format_method_docstring
    def get_attrs(
        self,
        meta_name,
        select=None,
        exclude=None,
        errors="warn",
        loc=None,
        multi=False,
        standardize=True,
        **extra,
    ):
        """Get the default attributes from meta specs

        Parameters
        ----------
        meta_name: str
            Valid generic Meta name
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
        specs = self.get_specs(meta_name, errors=errors)
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
        attrs = self.sglocator.format_attrs(attrs, loc=loc, standardize=standardize)

        return attrs

    def get_name(self, name, specialize=False, loc=None):
        """Get the name of the matching Meta specs

        Parameters
        ----------
        name: str, xarray.DataArray
            Either a data array, a known meta name or a data var name
        specialize: bool
            Get the first name
            as listed in specs, which is generally a specialized one,
            like a name adopted by specialized dataset.
        loc: str, None
            Format it at this location

        Return
        ------
        None or str
            Either the Meta name or the specialized name
        """
        if not isinstance(name, str):
            name = name.encoding.get("meta_name", self.match(name))
            # FIXME: category?
        elif name not in self:
            name = self.match_from_name(name)
        if name is None:
            return
        if specialize and self[name]["name"]:
            name = self[name]["name"]
        return self.sglocator.format_attr("name", name, loc=loc)

    def get_loc_arg(self, da, meta_name=None, locations=None):
        """Get the `loc` argument from a name or data array with name

        Parameters
        ----------
        da: xarray.DataArray
        meta_name: None, str
            A generic Meta name
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
        if locations is None:
            locations = self.get_loc_mapping(da, meta_names={da.name: meta_name})
        loc = locations.get(da.name)
        if loc is not None:
            return loc

        # From the array attributes
        return self.sglocator.get_loc_from_da(da)

    def _resolve_meta_name_(self, da, meta_name, specialize, rename, copy):
        """Resolve and validate Meta name for a data array.

        Parameters
        ----------
        da: xarray.DataArray
            Data array to process
        meta_name: str, None
            Meta name or None to match
        specialize: bool
            Use specialized name
        rename: bool
            Whether renaming is enabled
        copy: bool
            Whether copying is enabled

        Returns
        -------
        tuple: (meta_name, old_name, new_name) or (None, None, None)
            Meta name, original name, and formatted new name
        """
        if meta_name is None:
            meta_name = self.match(da)
        if meta_name is None:
            if not rename:
                return None, None, None
            return None, None, da.copy(deep=False) if copy else None
        assert meta_name in self.names
        old_name = da.name
        new_name = self.get_name(meta_name, specialize=specialize)
        return meta_name, old_name, new_name

    def _resolve_location_(self, da, loc):
        """Resolve location for a data array.

        Parameters
        ----------
        da: xarray.DataArray
            Data array to process
        loc: str, None
            Location or None to infer

        Returns
        -------
        str, None
            Resolved location
        """
        if loc is None:
            loc = self.get_loc_arg(da)
        return loc

    def _get_format_attrs_(self, meta_name, da, attrs):
        """Get attributes for formatting a data array.

        Parameters
        ----------
        meta_name: str
            Meta name
        da: xarray.DataArray
            Data array to process
        attrs: bool, dict
            Attributes specification

        Returns
        -------
        dict
            Attributes to apply
        """
        if attrs is True:
            # Get attributes from Meta specs
            attrs = self.get_attrs(meta_name, loc=None, standardize=False, multi=True)

            # Remove axis attribute for auxiliary coordinates
            if da.name and da.name not in da.indexes and "axis" in attrs:
                del attrs["axis"]

        elif not attrs:
            attrs = {}

        return attrs

    def format_dataarray(
        self,
        da,
        meta_name=None,
        rename=True,
        specialize=False,
        loc=None,
        attrs=True,
        standardize=True,
        replace_attrs=False,
        copy=True,
    ):
        """Format a DataArray's name and attributes

        Parameters
        ----------
        da: xarray.DataArray
        meta_name: str, None
            A generic Meta name. If not provided, it guessed with :meth:`match`.
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
            Do not use the Meta name for renaming, but the value of the "name" entry,
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
            The Meta name, given or matching, if rename if False; and None
            if not matching.

        """
        # Handle copy
        if rename:
            copy = True
        if copy:
            da = da.copy(deep=False)

        # Resolve Meta name
        meta_name, old_name, new_name = self._resolve_meta_name_(
            da, meta_name, specialize, rename, copy
        )
        if meta_name is None:
            if not rename:
                return
            return new_name  # Returns None or copied da

        # Resolve location
        loc = self._resolve_location_(da, loc)

        # Get attributes
        attrs = self._get_format_attrs_(meta_name, da, attrs)

        # Format array
        new_da = self.sglocator.format_dataarray(
            da,
            loc=loc,
            name=new_name,
            attrs=attrs,
            standardize=standardize,
            rename=rename,
            replace_attrs=replace_attrs,
            copy=False,
        )

        # Return new name but don't rename
        if not rename:
            if old_name is None:
                return self.sglocator.format_attr("name", new_name, loc)
            return self.sglocator.merge_attr("name", old_name, new_name, loc)

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
            A Meta name. If not provided, it guessed with :meth:`match`.
        specialize: bool
            Does not use the Meta name for renaming, but the first name
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


class MetaVarSpecs(_MetaCatSpecs_):
    """Meta specification for data_vars"""

    category = "data_vars"


class MetaCoordSpecs(_MetaCatSpecs_):
    """Meta specification for coords"""

    category = "coords"

    def get_loc_mapping(self, da, meta_names=None):
        return self.parent.get_loc_mapping(da, meta_names=meta_names, categories=["coords"])

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
            metaname = self.match(coord)
            if metaname:
                axis = self[metaname]["attrs"]["axis"]
        if axis is not None:
            if lower:
                return axis.lower()
            return axis.upper()

    def get_dim_type(self, dim, obj=None, lower=True):
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
        # Remove location
        dim_loc = dim
        dim = self.sglocator.parse_attr("name", dim)[0]

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
                raise exceptions.XoaMetaError(f"dimension '{dim}' does not belong to obj")

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

    def _parse_dim_search_args_(self, obj, meta_arg, loc, errors):
        """Parse and validate search arguments for dimension search.

        Parameters
        ----------
        obj: xarray.DataArray or xarray.Dataset
            Object to search in
        meta_arg: str, None
            Meta argument (name or type)
        loc: str, None
            Location argument
        errors: str
            Error handling mode

        Returns
        -------
        tuple: (meta_name, dim_type, loc, isds)
            Parsed arguments
        """
        meta_name = dim_type = None
        if meta_arg:
            if len(meta_arg) == 1:
                dim_type = meta_arg.lower()
                if meta_arg in self.names:
                    meta_name = meta_arg
            else:
                if not self._assert_known_(meta_arg, errors=errors):
                    return None, None, None, None
                meta_name = meta_arg
        isds = hasattr(obj, "data_vars")
        if not isds and dim_type is None:
            dim_type = self.get_axis(obj, lower=True)
        loc = self.sglocator.parse_loc_arg(loc)
        return meta_name, dim_type, loc, isds

    def _match_dimension_(self, obj, dim, meta_name, dim_type, loc, meta_arg):
        """Match a single dimension against search criteria.

        Parameters
        ----------
        obj: xarray.DataArray or xarray.Dataset
            Object containing the dimension
        dim: str
            Dimension name to match
        meta_name: str, None
            Target Meta name
        dim_type: str, None
            Target dimension type
        loc: str, None
            Target location
        meta_arg: str, None
            Original meta_arg for output formatting

        Returns
        -------
        tuple: (match, out)
            match: str, dict, or None if no match
            out: dict with dim info (always returned for fallback)
        """
        # Filter-out by loc
        pname, ploc = self.sglocator.parse_attr('name', dim)
        ploc = self.sglocator.parse_loc_arg(ploc)
        if loc is not None and loc != ploc:
            return None, None

        # From generic name
        if dim in obj.coords:
            this_meta_name = self.match(obj.coords[dim])
        else:
            this_meta_name = self.match_from_name(dim)
        if meta_name:
            if this_meta_name == meta_name:
                return dim, None
            if dim_type is None:  # keep searching if dim_type is not None
                # Still compute out for potential fallback
                this_dim_type = self.get_dim_type(dim, obj=obj)
                out = {"dim": dim, "type": this_dim_type, "meta_name": this_meta_name}
                return None, out

        # From dimension type
        this_dim_type = self.get_dim_type(dim, obj=obj)
        out = {"dim": dim, "type": this_dim_type, "meta_name": this_meta_name}
        if this_dim_type and this_dim_type == dim_type:
            if meta_arg:
                return dim, out
            else:
                return out, out
        return None, out

    @misc.ERRORS.format_method_docstring
    def search_dim(self, obj, meta_arg=None, loc=None, single=True, errors="ignore"):
        """Search a dataarray/dataset for a dimension name according to its generic name or type

        First, scan the dimension names.
        Then, look for coordinates: either it has an 'axis' attribute,
        or it a known Meta coordinate.

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Coordinate or data array
        meta_arg: str, {{"x", "y", "z", "t", "f"}}, None
            One-letter dimension type or generic Meta dim name.
            When set to None, dmension type is inferred with :meth:`get_axis`
            applied to `obj`
        loc: "any", letter
            Staggered grid location
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
        {errors}

        Return
        ------
        str, dict, None
            Dim name OR, dict with dim, type and meta_name keys if dim_type is None.
            None if nothing found.
        """
        # Parse search arguments
        result = self._parse_dim_search_args_(obj, meta_arg, loc, errors)
        if result == (None, None, None, None):  # Validation error
            return
        meta_name, dim_type, loc, isds = result

        # Search dimensions
        found = []
        out = None
        for dim in obj.dims:
            match, dim_out = self._match_dimension_(obj, dim, meta_name, dim_type, loc, meta_arg)
            if match is not None:
                found.append(match)
            # Track last dim_out for fallback
            if dim_out is not None:
                out = dim_out

        # Not found but only 1d and no dim_type specified
        if not found and len(obj.dims) == 1 and not meta_arg and out is not None:
            # FIXME: loop on coordinates?
            found.append(out)

        # Return result
        if single:
            return self._check_single_(errors, found, "dimension", meta_arg)
        return found

    @misc.ERRORS.format_method_docstring
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
            raise exceptions.XoaMetaError(f"Invalid dimension: {dim}")

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
        errors = misc.ERRORS[errors]
        if errors != "ignore":
            msg = f"No dataarray coord found from dim: {dim}"
            if errors == "raise":
                raise exceptions.XoaMetaError(msg)
            exceptions.xoa_warn(msg)

    @misc.ERRORS.format_method_docstring
    def get_dims(
        self,
        obj,
        meta_args,
        allow_positional=False,
        positions="tzyx",
        single=True,
        errors="warn",
    ):
        """Get the data array dimensions names from their type or generic Meta name

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array/dataset to scan
        meta_args: list
            List of letters among "x", "y", "z", "t" and "f",
            or generic names.
        allow_positional: bool
            Fall back to positional dimension of types is unkown.
        positions: str
            Default expected position of dim per type in `obj`
            starting from the end.
        single: bool
            If True, return the first item found or None.
            If False, return a possible empty list of found items.
        {errors}

        Return
        ------
        tuple, tuple(list)
            Tuple of dimension names or None when the dimension is not found

        See also
        --------
        search_dim
        get_dim_type
        """
        # Check shape
        errors = misc.ERRORS[errors]
        dims = list(obj.dims)
        ndim = len(dims)
        single_arg = isinstance(meta_args, str)
        if single_arg:
            meta_args = [meta_args]
        if len(meta_args) > len(dims):
            msg = f"This data array has less dimensions ({ndim})" " than requested ({})".format(
                len(meta_args)
            )
            if errors == "raise":
                raise exceptions.XoaMetaError(msg)
            if errors == "warn":
                exceptions.xoa_warn(msg)

        # Loop on args
        scanned = {}
        for meta_arg in meta_args:
            scanned[meta_arg] = self.search_dim(obj, meta_arg, single=False, errors="ignore")

        # Guess from position
        if allow_positional:
            not_found = [item[0] for item in scanned.items() if not item[1]]
            for i, meta_arg in enumerate(positions[::-1]):
                if meta_arg in not_found:
                    scanned[meta_arg] = [dims[-i - 1]]

        # Final check
        if single:
            for meta_arg, dim in scanned.items():
                if not dim:
                    if errors != "ignore":
                        msg = f"No dimension found matching: {meta_arg}"
                        if errors == "raise":
                            raise exceptions.XoaMetaError(msg)
                        exceptions.xoa_warn(msg)
                    scanned[meta_arg] = None
                else:
                    if len(dim) > 1 and errors != "ignore":
                        msg = f"Multiple candidates dimensions matching: {meta_arg}"
                        if errors == "raise":
                            raise exceptions.XoaMetaError(msg)
                        exceptions.xoa_warn(msg)
                    scanned[meta_arg] = dim[0]

        values = tuple(scanned.values())
        if single_arg and scanned:
            return values[0]
        return values

    def get_rename_dims_args(self, obj, locations=None, specialize=False):
        """Get args for renaming dimensions that are not coordinates

        Parameters
        ----------
        obj: xarray.DataArray, xarray.Dataset
            Array or dataset
        locations: dict, None
            Dict of staggerd grid locations with names as keys
        specialize: bool
            Do not use the Meta name for renaming, but the first name
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
            meta_dim_name = self.match_from_name(dim)  # known coordinate name
            dim_type = self.get_dim_type(dim, obj)  # known dimension type
            dim_loc = locations.get(dim)

            # Root name
            if meta_dim_name:
                new_name = self.get_name(meta_dim_name, specialize=specialize)
            elif dim_type:
                new_name = dim_type
            else:
                new_name = dim

            # Add loc
            if dim_loc is not False:
                new_name = self.sglocator.merge_attr("name", dim, new_name, loc=dim_loc)

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

        def _parse_dim_(meta_arg):
            if meta_arg in obj.dims:
                return meta_arg
            dim = self.search_dim(obj, meta_arg)
            if not dim:
                raise exceptions.XoaMetaError(f"Invalid argument for dimension: {meta_arg}")
            return dim

        if isinstance(dims, str):
            return _parse_dim_(dims)
        if isinstance(dims, dict):
            return dict((_parse_dim_(dim), value) for dim, value in dims.items())
        return type(dims)(_parse_dim_(dim) for dim in dims)


# for meth in (
#     "get_axis",
#     "get_dim_type",
#     "get_dim_types",
#     "search_dim",
#     "get_dims",
# ):
#     doc = getattr(MetaCoordSpecs, meth).__doc__
#     getattr(MetaSpecs, meth).__doc__ = doc
