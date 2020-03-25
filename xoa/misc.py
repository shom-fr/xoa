#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miscellaneaous low level utilities
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

import types
from enum import EnumMeta

from .__init__ import XoaError

import numpy as np


class DefaultEnumMeta(EnumMeta):
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
            nearest = 0
            cellave = -1

        regrid_methods
        str(regrid_methods)
        regrid_methods.rst
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
        return cls._member_map_[name]

    def __str__(cls):
        return ", ".join([f"{name}|{number}"
                          for name, number in cls._member_map_.items()])

    def __repr__(cls):
        return f"{cls.__name__}: {cls}"

    @property
    def rst(cls):
        return " ".join([f'``"{name}"|{number}``'
                         for name, number in cls._member_map_.items()])


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


def dict_merge(*dd, **kwargs):
    """Merge dictionaries

    First dictionaries have priority over next

    Parameters
    ----------
    dd:
        Argument are interpreted as dictionary to merge.
        Those who are not dictionaries are skipped.
    mergesubdicts: optional
        Also merge dictionary items
        (like in a tree) [default: True].
    mergetuples: optional
        Also merge tuple items [default: False].
    mergelists: optional
        Also merge list items [default: False].
    unique: optional
        Uniquify lists and tuples [default: True].
    skipnones: optional
        Skip Nones [default: True].
    skipempty: optional
        Skip everything is not converted to False
        using bool [default: False].
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
    mergesubdicts = kwargs.get("mergesubdicts", True)
    mergelists = kwargs.get("mergelists", False)
    mergetuples = kwargs.get("mergetuples", False)
    unique = kwargs.get("unique", True)
    skipnones = kwargs.get("skipnones", True)
    overwriteempty = kwargs.get("overwriteempty", False)
    cls = kwargs.get("cls")
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
                if unique:
                    outd[key] = list(set((outd[key])))
            # Merge tuples
            elif (
                mergetuples
                and isinstance(outd[key], tuple)
                and isinstance(val, tuple)
            ):
                outd[key] += val
                if unique:
                    outd[key] = tuple(set(outd[key]))

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
