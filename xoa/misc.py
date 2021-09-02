#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneaous low level utilities
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

import types
from enum import IntEnum, EnumMeta

import numpy as np

from .__init__ import XoaError


class XEnumMeta(EnumMeta):
    """Exented version ofnum meta-class

    This version supports:

    - :meth:`__contains__` (``in``) with strings
    - a better :meth:`__str__` (``str()``) method
    - a :meth:`get_rst` methods and :attr`str` and
      :attr:`rst_with_links` properties.

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.misc import XEnumMeta
        from enum import IntEnum

        class regrid_methods(IntEnum, metaclass=XEnumMeta):
            linear = 1
            bilinear = 1
            nearest = 0
            cellave = -1

        regrid_methods
        'linear' in regrid_methods
        'xxx' in regrid_methods
        1 in regrid_methods
        str(regrid_methods)
        regrid_methods.get_rst(with_links=True, link_module="xoa.tutu")
        regrid_methods.rst

    """
    default = None

    # def __call__(cls, value=None, *args, **kwargs):
    #     if value is None:
    #         return next(iter(cls))
    #     return super().__call__(value, *args, **kwargs)

    def __getitem__(cls, name):
        if not isinstance(name, str):
            return cls(name)
        return cls._member_map_[name]

    def __contains__(cls, value):
        if isinstance(value, str):
            return value in list(cls._member_map_.keys())
        return super().__call__(value)

    def _get_groups_(cls):
        groups = {}
        for name, number in cls._member_map_.items():
            groups.setdefault(number, []).append(name)
        return groups

    def _get_choices_(cls, es=""):
        choices = []
        for number, names in cls._get_groups_().items():
            cc = [f"{number}"] + [f'"{name}"' for name in names]
            choices.append(es + "|".join(cc) + es)
        return choices

    def __str__(cls):
        return ", ".join(cls._get_choices_())

    def __repr__(cls):
        return f"{cls.__name__}: {cls}"

    def get_rst(cls, with_links=False, link_module=None):
        if with_links:
            choices = []
            prefix = ((link_module+".") if link_module else "")
            prefix += cls.__name__ + "."
            for number, names in cls._get_groups_().items():
                number = int(number)
                attr = names[0]
                choice = "|".join([f"{number:d}"] +
                                  [f'"{name}"' for name in names])
                choices.append(f":attr:`{choice}<{prefix}{attr}>`")
            return ", ".join(choices)
        else:
            return ", ".join(cls._get_choices_("``"))

    @property
    def rst(cls):
        return cls.get_rst()

    @property
    def rst_with_links(cls):
        return cls.get_rst(with_links=True)


class DefaultEnumMeta(XEnumMeta):
    """Enum meta-class that support default value and None

    When the item is not provided or equal to ``None``, the first
    declared item is returned

    Inspired from: https://stackoverflow.com/questions/44867597/is-there-a-way-to-specify-a-default-value-for-python-enums

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.misc import DefaultEnumMeta
        from enum import IntEnum

        class regrid_methods(IntEnum, metaclass=DefaultEnumMeta):
            linear = 1
            bilinear = 1
            nearest = 0
            cellave = -1

        regrid_methods() # default method
        regrid_methods(None) # default method
        regrid_methods(1)
        regrid_methods[None] # default method
        regrid_methods['linear']
        regrid_methods['cellave']

    """
    default = None

    def __call__(cls, value=None, *args, **kwargs):
        if value is None:
            return next(iter(cls))
        return super().__call__(value, *args, **kwargs)

    def __getitem__(cls, name):
        if name is None:
            return next(iter(cls))
        if not isinstance(name, str):
            return cls(name)
        return cls._member_map_[name]


class IntEnumChoices(IntEnum):

    def __str__(self):
        return self.name


class Choices(object):
    """Choice management for a function or method parameter

    Parameters
    ----------
    choices: dict, list
        Allowed choices. When a dict, values are considered as the
        description of choices.
    case_insensitive: bool
        Wether the treatment of string type choice should be case
        sensitive or not.
    parameter: None, str
        Parameter name, which defaults to the lower case class name
    description: str
        Short description of the parameter.
    """

    def __init__(self, choices, case_insensitive=True, parameter=None,
                 description="Choices"):
        self._ci = case_insensitive
        self._docs = {}
        if isinstance(choices, dict):
            self._choices = []
            for value, doc in choices.items():
                value = self._reformat_value_(value)
                self._docs[value] = doc
                self._choices.append(value)
        else:
            self._choices = [self._reformat_value_(value) for value in choices]
        self._parameter = (self.__class__.__name__.lower()
                           if parameter is None else parameter)
        self._description = description

    @property
    def choices(self):
        return self._choices

    def _reformat_value_(self, value):
        if self._ci and isinstance(value, str):
            return value.lower()
        return value

    def __getitem__(self, choice):
        choice = self._reformat_value_(choice)
        if choice not in self._choices:
            desc = self._description if self._description else 'choice'
            raise XoaError(f"Invalid choice for \"{desc}\": {choice}. "
                           f"Please choose one of: {self}")
        return choice

    def __str__(self):
        return ", ".join(self._choices)

    def to_docstring(self, indent=4):
        """Convert to numpy-like docstring

        Parameters
        ----------
        indent: str, int
            Base indentation. Integers are multiplied by a space char.

        Return
        ------
        str
            Docstring of this parameter
        """
        indent = (" " * indent) if isinstance(indent, int) else indent
        pindent = indent + 4 * " "
        types = "{" + ", ".join([repr(c) for c in self.choices]) + "}"
        rst = f"{self._parameter}: {types}\n"
        if self._description:
            rst += f"{pindent}{self._description}\n"
        if self._docs:
            rst += "\n"
            for choice, doc in self._docs.items():
                rst += f"{pindent}- ``{choice}``: {doc}\n"
            # rst += "\n"
        return rst

    def format_function_docstring(self, func):
        func.__doc__ = func.__doc__.format(
            **{self._parameter: self.to_docstring(4)})
        return func

    def format_method_docstring(self, func):
        func.__doc__ = func.__doc__.format(
            **{self._parameter: self.to_docstring(8)})
        return func


ERRORS = Choices({"ignore": "silently ignore",
                  "warn": "emit a warning",
                  "raise": "raise an exception"},
                 parameter="errors",
                 description="In case of errors"
                 )


def get_axis_slices(ndim, axis, **kwargs):
    """Get standard slices for an axis of a ndim array

    Parameters
    ----------
    ndim:
        The number of dimensions. It can also be
        a tuple (like an array shape) or an array.
    axis:
        Index of the axis.

    Return
    ------
    A dictionary of tuples of slices. All tuples have a
        length of ndim, and can be used has a slice for the array
        (see exedges =ample).

        - ``"all"``: Select everything.
        - ``"first"``/``"last"``: First and last.
        - ``"firstp1"``: Second element.
        - ``"firstp2"``: Third element.
        - ``"lastm1"``: Element before the last one.
        - ``"lastm2"``: Second element before the last one.
        - ``"firsts"``: All but the last.
        - ``"lasts"``: All but the first.
        - ``"firstsm1"``: All but the last two.
        - ``"lastsp1"``: All but the first two.
        - ``"mid"``: All but the first and last.

    Example
    -------
    .. ipython:: python

        @suppress
        import numpy as np, pprint
        @suppress
        from xoa.misc import get_axis_slices

        var = np.arange(2*5*4).reshape(2, 5, 4)
        pprint.pprint(get_axis_slices(var, axis=1))
    """
    if not isinstance(ndim, int):
        ndim = np.ndim(ndim)
    sel = [slice(None)]*ndim
    selmid = list(sel)
    selmid[axis] = slice(1, -1)
    selmid = tuple(selmid)
    sellasts = list(sel)
    sellasts[axis] = slice(1, None)
    sellasts = tuple(sellasts)
    selfirsts = list(sel)
    selfirsts[axis] = slice(0, -1)
    selfirsts = tuple(selfirsts)
    sellastsp1 = list(sel)
    sellastsp1[axis] = slice(2, None)
    sellastsp1 = tuple(sellastsp1)
    selfirstsm1 = list(sel)
    selfirstsm1[axis] = slice(0, -2)
    selfirstsm1 = tuple(selfirstsm1)
    sellast = list(sel)
    sellast[axis] = -1
    sellast = tuple(sellast)
    selfirst = list(sel)
    selfirst[axis] = 0
    selfirst = tuple(selfirst)
    sellastm1 = list(sel)
    sellastm1[axis] = -2
    sellastm1 = tuple(sellastm1)
    sellastm2 = list(sel)
    sellastm2[axis] = -3
    sellastm2 = tuple(sellastm2)
    selfirstp1 = list(sel)
    selfirstp1[axis] = 1
    selfirstp1 = tuple(selfirstp1)
    selfirstp2 = list(sel)
    selfirstp2[axis] = 2
    selfirstp2 = tuple(selfirstp2)
    if kwargs:
        for key, val in kwargs.items():
            if isinstance(val, (list, tuple)):
                val = slice(*val)
            ksel = list(sel)
            ksel[axis] = val
            kwargs[key] = ksel
    return dict(all=sel, mid=selmid, lasts=sellasts, firsts=selfirsts,
                lastsp1=sellastsp1, firstsm1=selfirstsm1,
                last=sellast, first=selfirst, lastm1=sellastm1,
                firstp1=selfirstp1,
                lastm2=sellastm2, firstp2=selfirstp2, **kwargs)


def is_iterable(obj, nostr=True, nogen=True):
    """Check if an object is iterable or not

    Parameters
    ----------
    obj:
        Object to check

    Return
    ------
    bool
    """

    if not nogen and isinstance(obj, types.GeneratorType):
        return True
    if not (hasattr(obj, "__len__") and callable(obj.__len__)):
        return False
    if nostr:
        return not isinstance(obj, str)
    return True


def dict_check_defaults(dd, **defaults):
    """Check that a dictionary has some default values

    Parameters
    ----------
    dd: dict
        Dictionary to check
    **defs: dict
        Dictionary of default values

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.misc import dict_check_defaults

        dd = dict(color='blue')
        dict_check_defaults(dd, color='red', size=10)
    """
    if defaults is None:
        defaults = {}
    for item in defaults.items():
        dd.setdefault(*item)
    return dd


def dict_filter(
    kwargs,
    filters,
    defaults=None,
    copy=False,
    short=False,
    keep=False,
    **kwadd,
):
    """Filter out kwargs (typically extra calling keywords)

    Parameters
    ----------
    kwargs:
        Dictionnary to filter.
    filters:
        Single or list of prefixes.
    defaults:
        dictionnary of default values for output fictionnary.
    copy:
        Simply copy items, do not remove them from kwargs.
    short:
        Allow prefixes to not end with ``"_"``.
    keep:
        Keep prefix filter in output keys.

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.misc import dict_filter
        kwargs = {'basemap':'f', 'basemap_fillcontinents':True, 'quiet':False,'basemap_plot':False}
        dict_filter(kwargs,'basemap', defaults=dict(drawcoastlines=True,plot=True), good=True)
        kwargs

    Return
    ------
    dict
    """

    if isinstance(filters, str):
        filters = [filters]
    if copy:
        kwread = kwargs.get
    else:
        kwread = kwargs.pop

    # Set initial items
    kwout = {}
    for filter_ in filters:
        if not filter_.endswith("_") and filter_ in kwargs:
            if isinstance(kwargs[filter_], dict):
                kwout.update(kwread(filter_))
            else:
                kwout[filter_] = kwread(filter_)
        if not short and not filter_.endswith("_"):
            filter_ += "_"
        for att, val in list(kwargs.items()):
            if att.startswith(filter_) and att != filter_:
                if keep:
                    kwout[att] = kwread(att)
                else:
                    kwout[att[len(filter_):]] = kwread(att)

    # Add some items
    kwout.update(kwadd)

    # Set some default values
    if defaults is not None:
        for att, val in defaults.items():
            kwout.setdefault(att, val)
    return kwout


def dict_merge(*dd, mergesubdicts=True, mergelists=False, mergetuples=False,
               uniquify=False, skipnones=True, overwriteempty=False, cls=None,
               **kwargs):
    """Merge dictionaries

    First dictionaries have priority over next

    Parameters
    ----------
    dd:
        Argument are interpreted as dictionary to merge.
        Those who are not dictionaries are skipped.
    mergesubdicts: optional
        Also merge dictionary items
        (like in a tree).
    mergetuples: optional
        Also merge tuple items.
    mergelists: optional
        Also merge list items.
    uniquify: optional
        Uniquify lists and tuples.
    skipnones: optional
        Skip Nones.
    overwriteempty: optional
        Overwrite value that does are not True when converted to bool.
    cls: optional
        Class to use. Default to the first class found in arguments
        that is not a :class:`dict`, else defaults to :class:`dict`.

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.misc import dict_merge
        d1 = dict(a=3, b=5, e=[1, 2])
        d2 = dict(a=5, c=7, e=[3, 4])
        print(dict_merge(d1, d2, mergelists=True))

    """
    # Options
    dd = [_f for _f in dd if _f]

    # Get the class
    if cls is None:
        cls = dict
        for d in dd:
            if d.__class__ is not dict:
                cls = d.__class__
                break

    # Init
    from configobj import Section, ConfigObj

    if cls is Section:
        for d in dd:
            if isinstance(d, Section):
                break
        else:
            raise XoaError("Can't initialise Section for merging")
        outd = Section(d.parent, d.depth, d.main, name=d.name)
    else:
        outd = cls()
    kwargs.update(
        mergesubdicts=mergesubdicts,
        mergelists=mergelists,
        mergetuples=mergetuples,
        uniquify=uniquify,
        skipnones=skipnones,
        overwriteempty=overwriteempty,
        cls=cls)

    # Loop
    for d in dd:
        if not isinstance(d, dict):
            continue

        # Content
        for key, val in d.items():
            if skipnones and val is None:
                continue

            # Not set so we set
            if key not in outd or (overwriteempty and is_empty(outd[key])):
                outd[key] = val

            # Merge subdict
            elif (
                mergesubdicts
                and isinstance(outd[key], dict)
                and isinstance(val, dict)
            ):
                outd[key] = dict_merge(outd[key], val, **kwargs)

            # Merge lists
            elif (
                mergelists
                and isinstance(outd[key], list)
                and isinstance(val, list)
            ):
                outd[key] += val
                if uniquify:
                    outd[key] = gunique(list(outd[key]))

            # Merge tuples
            elif (
                mergetuples
                and isinstance(outd[key], tuple)
                and isinstance(val, tuple)
            ):
                outd[key] += val
                if uniquify:
                    outd[key] = tuple(gunique(outd[key]))

    # Comments for ConfigObj instances
    if cls is ConfigObj:
        if not outd.initial_comment and hasattr(d, "initial_comment"):
            outd.initial_comment = d.initial_comment
        if not outd.final_comment and hasattr(d, "final_comment"):
            outd.final_comment = d.final_comment
        if hasattr(d, "inline_comments") and d.inline_comments:
            outd.inline_comments = dict_merge(
                outd.inline_comments, d.inline_comments, overwriteempty=True
            )

    return outd


def is_empty(x):
    """Check if empty"""
    if isinstance(x, bool):
        return False
    try:
        return not bool(x)
    except Exception:
        return False


def match_string(ss, checks, ignorecase=True, transform=None):
    """Check that a string verify a check list that consists of
    a list of either strings or callables

    Parameters
    ----------
    ss: str
    checks: str, callable, list of {str or callable}
    ignorecase: bool
    transform: callable

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.misc import match_string
        import re
        match_string('sst', 'sst')
        match_string('sst', [re.compile(r'ss.$').match])

    Return
    ------
    True
    """
    # Nothing
    if not ss or not checks:
        return False

    # Setup
    ss = ss.strip()
    if ignorecase:
        ss = ss.lower()
    if not is_iterable(checks, nogen=False):
        checks = [checks]
    checks = [x for x in checks if x is not None]

    # Callables
    sss = []
    for check in checks:
        if callable(transform) and not callable(check):
            check = transform(check)
        if callable(check) and check(ss):
            return True
        if isinstance(check, str):
            sss.append(check)

    # Strings
    sss = [s.strip() for s in sss]
    if ignorecase:
        sss = [s.lower() for s in sss]
    return ss in sss


def match_attrs(obj, checks, ignorecase=True, transform=None):
    """Check that at least one of the attributes matches check list

    Parameters
    ----------
    obj: object
    checks: dict
        A dictionary of (attribute name, checklist), checklist being an
        iterable as accepted by :func:`match_string`.
    """
    if obj is None or checks is None:
        return False
    for attname, attchecks in checks.items():
        if hasattr(obj, attname) and match_string(
            getattr(obj, attname),
            attchecks,
            ignorecase=ignorecase,
            transform=transform,
        ):
            return True
    return False


def gunique(seq):
    """Create a generator that yields unique item whlist presrving the order

    Parameters
    ----------
    seq: sequence

    Yields
    ------
    item

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.misc import gunique
        print(list(gunique([1, 6, 1, 8])))
    """
    seen = set()
    seen_add = seen.add
    for x in seq:
        if not (x in seen or seen_add(x)):
            yield x


class ArgList(object):
    """Utility to always manage arguments as list and return results as input

    Examples
    --------
    .. ipython:: python

        @suppress
        from xoa.misc import ArgList

        # Scalar
        a = 'a'
        al = ArgList(a)
        al.get() # input for function as tuple
        al.put(['aa']) # output as input

        # Iterable
        a = ('a','b')
        al = ArgList(a)
        al.get()
        al.put(['aa'])

    """

    def __init__(self, argsi):
        self.single = not isinstance(argsi, list)
        self.argsi = argsi

    def get(self):
        return [self.argsi] if self.single else self.argsi

    def put(self, argso):
        so = not isinstance(argso, list)
        if (so and self.single) or (not so and not self.single):
            return argso
        if so and not self.single:
            return [argso]
        return argso[0]


class ArgTuple(object):
    """Utility to always manage arguments as tuple and return results as input

    Examples
    --------
    .. ipython:: python

        @suppress
        from xoa.misc import ArgTuple

        # Scalar
        a = 'a'
        al = ArgTuple(a)
        al.get() # input for function as tuple
        al.put(('aa',)) # output as input

        # Iterable
        a = ('a','b')
        al = ArgTuple(a)
        al.get()
        al.put(('aa',))

    """

    def __init__(self, argsi):
        self.single = not isinstance(argsi, tuple)
        self.argsi = argsi

    def get(self):
        return (self.argsi,) if self.single else self.argsi

    def put(self, argso):
        so = not isinstance(argso, tuple)
        if (so and self.single) or (not so and not self.single):
            return argso
        if so and not self.single:
            return (argso,)
        return argso[0]
