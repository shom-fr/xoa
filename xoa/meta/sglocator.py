#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Naming convention tools for reading and formatting variables

.. rubric:: How to use it

See the :ref:`indepth.meta` section.

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


from .. import exceptions
from .. import misc


def _compile_sg_match_(re_match, formats, root_patterns, location_pattern):
    for attr in ("name", "standard_name", "long_name"):
        if formats[attr]:
            root_pattern = root_patterns[attr]
            re_match[attr] = re.compile(
                formats[attr].format(
                    root=rf"(?P<root>{root_pattern})",
                    loc=rf"(?P<loc>{location_pattern})",
                ),
                re.I,
            ).fullmatch


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

    default_formats = {
        "name": "{root}",
        "standard_name": "{root}_at_{loc}_location",
        "long_name": "{root} at {loc} location",
    }

    root_patterns = {
        "name": r"\w+",
        "standard_name": r"\w+",
        "long_name": r"[\w ]+",
    }

    location_pattern = "[a-z]+"

    re_match = {}
    _compile_sg_match_(re_match, default_formats, root_patterns, location_pattern)

    def __init__(self, name_format=None, valid_locations=None, encoding=None):
        # Init
        self.formats = self.default_formats.copy()
        self.re_match = self.re_match.copy()
        self.update(name_format, valid_locations, encoding)

    def update(self, name_format=None, valid_locations=None, encoding=None):
        if valid_locations:
            valid_locations = set(valid_locations)
        self.valid_locations = valid_locations or set()

        # Formats and regexps
        if name_format:
            for pat in ("{root}",):  # "{loc}"):
                if pat not in name_format:
                    raise exceptions.XoaMetaError(
                        "name_format must contain string {root}: " + name_format
                    )
            if len(name_format) == 10:
                exceptions.xoa_warn(
                    'No separator found in "name_format" and '
                    'and no "valid_locations" specified: '
                    f"{name_format}. This leads to ambiguity during "
                    "regular expression parsing."
                )
            self.formats["name"] = name_format
        elif name_format is not None:
            self.formats["name"] = "{root}"
        location_pattern = "|".join(self.valid_locations) or "[a-z]+"
        _compile_sg_match_(self.re_match, self.formats, self.root_patterns, location_pattern)

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
            from xoa.meta.sglocator import SGLocator
            sg = SGLocator(name_format="{root}_{loc}")
            sg.parse_attr("name", "super_banana_t")
            sg.parse_attr("standard_name", "super_banana_at_rhum_location")
            sg.parse_attr("standard_name", "super_banana_at_rhum_place")

            sg = SGLocator(valid_locations=["u", "rho"])
            sg.parse_attr("name", "super_banana_t")
            sg.parse_attr("name", "super_banana_rho")
        """
        if not self.formats[attr]:
            return value, None
        m = self.re_match[attr](value)
        if m is None or "loc" not in m.groupdict():
            return value, None
        return m.group("root"), m.group("loc").lower()

    @misc.ERRORS.format_method_docstring
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
        XoaMetaError
            When locations parsed from name and attributes are conflicting.
            Thus, this method method is a way to check location consistency.
        """
        loc = None
        errors = misc.ERRORS[errors]
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
                        raise exceptions.XoaMetaError(msg)
                    else:
                        exceptions.xoa_warn(msg)

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
                    src = " and ".join(src)
                    msg = (
                        "The location parsed from long_name attribute "
                        f"[{loc_}] conflicts the location parsed from the "
                        f"{src} [{loc}]"
                    )
                    if errors == "raise":
                        raise exceptions.XoaMetaError(msg)
                    else:
                        exceptions.xoa_warn(msg)

        return loc

    @misc.ERRORS.format_method_docstring
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
        XoaMetaError
            When locations parsed from name and attributes are conflicting.
            Thus, this method method is a way to check location consistency.
        """
        return self.get_loc(name=da.name, attrs=da.attrs, errors=errors)

    def parse_loc_arg(self, loc):
        """Parse the location argument

        Return values as function of input values:

            * `None`: None, True, "any"
            * `False`: `False`, ""
            * [01][01][01]: `encoding[loc][0]` or False
            * str: str
        """
        if loc is None or loc is True or loc == "any":
            return
        if loc is False or loc == "":
            return False
        if not isinstance(loc, str):
            raise exceptions.XoaMetaError(
                "Invalid loc argument. Must one of: "
                'None, Trye, "any", False, "" or a location string'
            )
        if self.valid_locations and loc not in self.valid_locations:
            raise exceptions.XoaMetaError(
                f'Location "{loc}" is invalid. '
                "Valid locations are: " + ", ".join(self.valid_locations)
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
            from xoa.meta.sglocator import SGLocator
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
        if loc is None:
            loc = ploc
        loc = loc or ploc
        if not loc:
            return root

        # Better looking loc
        if standardize:
            if attr == "long_name":
                loc = loc.upper()
            elif attr == "standard_name":
                loc = loc.lower()

        if not self.formats[attr]:
            return root
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
            from xoa.meta.sglocator import SGLocator
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
            if attr in self.valid_attrs and attr != "name":
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
            from xoa.meta.sglocator import SGLocator
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
                    attr,
                    attrs.get(attr, None),
                    value,
                    loc,
                    standardize=standardize,
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
            da = da.copy(deep=False)

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
                    da.attrs,
                    attrs,
                    standardize=standardize,
                    replace=replace_attrs,
                    **kwloc,
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
