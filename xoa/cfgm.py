# -*- coding: utf-8 -*-
"""Configuration management utilities based on :mod:`configobj`

.. rubric:: Usage

See the :ref:`uses.cfgm` section.

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

import inspect
import os
import re
import shutil
import sys
import traceback
from argparse import Action as AP_Action
from argparse import ArgumentParser
from argparse import HelpFormatter as ArgHelpFormatter
from argparse import _HelpAction
from collections import OrderedDict
from warnings import warn
import logging

import validate
from configobj import ConfigObj, flatten_errors
from validate import (
    ValidateError, Validator, VdtTypeError,
    VdtValueTooBigError, VdtValueTooSmallError)

from .__init__ import XoaError, XoaWarning, xoa_warn

try:
    import numpy
except ImportError:
    numpy = None


class VdtWarning(XoaWarning):
    pass


class XoaValidateError(ValidateError, XoaError):
    pass


class VdtSizeError(XoaValidateError):
    """List size is incorrect (nmin, nmax, odd, even or shape)"""

    pass


def _valwrap_(validator):
    """
    Wrap a validation function to allow extraneous named arguments in specfile,
    this is usefull when getting specification with
    validator._parse_with_caching(configspec[section][option])
    """
    # Already wrapped
    if validator.__name__.startswith(
        "validator_wrapper-"
    ) or validator.__name__.startswith("list_validator_wrapper-"):
        return validator

    # Wrapper
    def validator_wrapper(value, *args, **kwargs):
        # Remove extraneous arguments the validator can't handle
        argspec = inspect.getfullargspec(validator)
        kwargs = kwargs.copy()
        for k in list(kwargs.keys()):
            if k not in argspec.args:
                kwargs.pop(k)
        return validator(value, *args, **kwargs)

    validator_wrapper.__name__ += "-" + validator.__name__
    return validator_wrapper


def _valwraplist_(validator):
    """
    Wrap a validation function to handle list value using an existing validator

    This adds the following list checking behaviors that can be used as named
    arguments in specifications

    Parameters
    ----------
    n:
        required fixed number of elements
    nmin:
        required minimum number of elements
    nmax:
        required maximum number of elements
    odd:
        number of elements must be odd
    even:
        number of elements must be even
    shape:
        check value shape (requires numpy)
    """
    # Already wrapped
    if validator.__name__.startswith("list_validator_wrapper-"):
        return validator

    # Wrapper
    def list_validator_wrapper(value, *args, **kwargs):
        # Handle None and default
        if str(value) == "None":
            return None
        default = args[0] if len(args) else kwargs.get("default", ())
        if value == "":
            value = default

        # Handle single value
        if not isinstance(value, (list, tuple)):
            value = [value]

        # Do list checks
        n = kwargs.pop("n", None)
        if n is not None and len(value) != int(n):
            raise VdtSizeError(
                "Incorrect size: {}, {} values expected".format(len(value), n)
            )
        nmin = kwargs.pop("nmin", None)
        if nmin is not None and len(value) < int(nmin):
            raise VdtSizeError(
                "Incorrect size: {}, at least {} values expected".format(
                    len(value), nmin
                )
            )
        nmax = kwargs.pop("nmax", None)
        if nmax is not None and len(value) > int(nmax):
            raise VdtSizeError(
                "Incorrect size: {}, at most {} values expected".format(
                    len(value), nmax
                )
            )
        odd = validate.is_boolean(kwargs.pop("odd", False))
        if odd and not len(value) % 2:
            raise VdtSizeError(
                "Incorrect size: {}, odd number of values expected".format(
                    len(value)
                )
            )
        even = validate.is_boolean(kwargs.pop("even", False))
        if even and len(value) % 2:
            raise VdtSizeError(
                "Incorrect size: {}, even number of values expected".format(
                    len(value)
                )
            )

        shape = kwargs.pop("shape", None)
        if shape is not None:
            if numpy is None:
                warn("Cannot check shape, numpy package is missing")
            else:
                try:
                    shape, vshape = list(map(int, shape)), numpy.shape(value)
                    if vshape != shape:
                        raise VdtSizeError(
                            "Incorrect shape: {}, {} shape expected".format(
                                vshape, shape
                            )
                        )
                except Exception:
                    raise ValidateError(
                        "Cannot test value shape, this may be caused by "
                        "an irregular array-like shape. Error was:\n{}".format(
                            traceback.format_exc()
                        )
                    )

        # Preserve tuple type
        istuple = isinstance(value, tuple)

        # Validate each values
        # FIXME: why *args[1:] as previously instead of *args?
        value = [validator(v, *args, **kwargs) for v in value]
        return tuple(value) if istuple else value

    list_validator_wrapper.__name__ += "-" + validator.__name__
    return list_validator_wrapper


def is_bbox(value, default=None):
    """Parse bbox coordinates with value format: x1,y1,x2,y2"""
    if str(value) == "None":
        return None
    if value == "":
        value = default
    c = []
    # two possible delimiters: whitespaces and ','
    for v in value.split():
        c.extend(v.split(","))
    if len(c) != 4:
        raise VdtTypeError(value)
    return list(map(float, c))


def is_numerics(
    value, default=None, min=None, max=None, type="float", n=None
):
    """Validation function of a tuple of numeric values"""
    if isinstance(value, str):
        value = value.strip("()[] ")
    if str(value) == "None":
        return None
    if isinstance(value, list):
        value = tuple(value)
    elif isinstance(value, str):
        try:
            value = eval(value)
        except Exception:
            raise VdtTypeError(value)
    if not isinstance(value, tuple):
        try:
            value = tuple(value)
        except Exception:
            value = (value,)
    if n is not None:
        if isinstance(n, str):
            n = int(n)
        if len(value) != n:
            raise VdtTypeError(value)
    out = ()
    type = eval(type)
    if min is not None and isinstance(min, str):
        min = type(min)
    if max is not None and isinstance(max, str):
        max = type(max)
    for val in value:
        if isinstance(val, str):
            try:
                val = type(val)
            except Exception:
                raise VdtTypeError(value)
        if min is not None and val < min:
            val = type(min)
        if max is not None and val > max:
            val = type(max)
        out += (val,)
    return out


def is_minmax(
    value, min=None, max=None, default=(0, 100), type="float"
):
    """Validation function of a min,max pair"""
    value = is_numerics(
        value, min=min, max=max, default=default, type=type, n=2
    )
    if value is not None:
        out = list(value)
        out.sort()
        value = tuple(value)
    return value


def is_figsize(value, default=(6, 6), min=0, max=20):
    """Validation function of a figure size (xsize,ysize)"""
    return is_numerics(
        value, default=default, min=min, max=max, type="float", n=2
    )


def is_interval(value, default=None):
    """Validation function of an interval of coordinates (min,max[,bounds])"""
    if isinstance(value, str):
        value = value.strip("()[] ")
    if str(value) == "None":
        return None
    if not isinstance(value, str):
        if not isinstance(value, list):
            raise VdtTypeError(value)
        value = ",".join(value)
    if value.startswith("("):
        value = value[1:]
    if value.endswith(")"):
        value = value[:-1]
    values = value.split(",")
    if len(values) < 2 or len(values) > 3:
        raise VdtTypeError(value)
    out = ()
    for val in values[:2]:
        try:
            val = eval(val)
        except Exception:
            pass
        out += (val,)
    if len(values) == 3 and values[2]:
        m = re.search("([co]{1,2}[ne]{0,2})", values[2])
        if m is None:
            raise VdtTypeError(value)
        out += (m.group(1),)
    return out


def is_cmap(value, default=None):
    """Validation function that return a :`xoa.color.CmapAdapter`"""
    if str(value) == "None":
        return None
    from .color import CmapAdapter

    # if isinstance(value, str) and value.startswith("["):
    #     value = eval(value)
    if isinstance(value, list):
        from .color import cmap_custom

        return cmap_custom(value)
    return CmapAdapter(value)


class VdtDateTimeError(ValidateError):
    pass


def is_path(value, default="", expand=None):
    """Parse a value as a path

    Parameters
    ----------
    expand:
        expandvars and expandhome on loaded path

    **Warning: expand currently can't work with interpolation**
    """
    # TODO: fix interpolation and expand !
    if str(value) == "None":
        return None
    if value == "":
        value = default
    if expand and isinstance(value, str):
        return os.path.expandvars(os.path.expanduser(value))
    return value


def is_pydatetime(value, default=None, fmt="%Y-%m-%dT%H:%M:%S"):
    """Parse value as a :class:`datetime.datetime` object"""
    import datetime

    if str(value) == "None":
        return None
    if value == "":
        value = default
    try:
        return datetime.datetime.strptime(value, fmt)
    except ValueError as e:
        raise VdtDateTimeError(e)


# def is_timeunits(value, default="days since 1950-01-01"):
#     """Validation function of standard time units"""
#     from .atime import are_valid_units
#     value = str(value)
#     if value == "None" or not value:
#         value = default
#     if not are_valid_units(value):
#         raise VdtTypeError(value)
#     return value


def is_cdtime(value, min=None, max=None, default=None):
    """Validation function of a date (compatible with :func:`cdtime.s2c`)"""
    import cdtime

    value = str(value).strip()
    if not value[0].isdigit():
        return value.upper()
    try:
        value = cdtime.s2c(value)
    except Exception:
        raise VdtTypeError(value)
    if min is not None and val < cdtime.s2c(min):
        raise VdtValueTooSmallError(value)
    if max is not None and val > cdtime.s2c(max):
        raise VdtValueTooBigError(value)
    return value


def is_timestamp(value, default=None):
    """Validation function of date as parsable by :func:`pandas.Timestamp`"""
    if str(value) == "None":
        return
    import pandas as pd

    return pd.Timestamp(value)


def is_timedelta(value, default=None):
    """Validation function of date as parsable by :func:`pandas.Timedelta`"""
    if str(value) == "None":
        return
    if not isinstance(value, list) or len(value) != 2:
        raise VdtDateTimeError("Need value and unit")
    import pandas as pd

    return pd.Timedelta(float(value[0]), value[1])


def is_timedelta64(value, default=None):
    """Validation function of date as parsable by :func:`numpy.timedelta64`"""
    if str(value) == "None":
        return
    if not isinstance(value, list) or len(value) != 2:
        raise VdtDateTimeError("Need value and unit")
    import numpy as np

    return np.timedelta64(float(value[0]), value[1])


def is_datetime(value, default=None):
    """Validation function to magically create a :class:`datetime.datetime`"""
    value = is_timestamp(value, default=default)
    if str(value) == "None":
        return
    return value.to_pydatetime()


def is_datetime64(value, default=None):
    """Validation function to create a :class:`numpy.datetime64`"""
    if str(value) == "None":
        return
    return is_timestamp(value, default=default).to_datetime64()


def is_daterange(value, default=None):
    """Validation function of a :func:`pandas.date_range`"""
    if str(value) == "None":
        return
    if not isinstance(value, list) or len(value) not in [3, 4]:
        raise VdtDateTimeError("Need startdate, end date and step")
    closed = None if len(value) == 3 else eval(value[3])
    import pandas as pd

    return pd.time_range(value[0], value[1], value[2], closed)


def is_daterange64(value, default=None):
    """Validation function of a date range created with :func:`numpy.arange`"""
    if str(value) == "None":
        return
    if not isinstance(value, list) or len(value) != 3:
        raise VdtDateTimeError("Need startdate, end date and step")
    import numpy as np

    return np.arange(value[0], value[1], dtype="M8[{}]".format(value[2]))


def is_boolstr(value, default=None):
    """Validation function of a boolean or a string"""
    if str(value) == "None":
        return
    if isinstance(value, str):
        try:
            return validate.bool_dict[value.lower()]
        except KeyError:
            return value
    if value == False:
        return False
    elif value == True:
        return True
    else:
        raise VdtTypeError(value)


def is_eval(value, default=None, unchanged_if_failed=True):
    """Validate a string that can be evaluated"""
    try:
        value = eval(str(value))
    except Exception:
        if unchanged_if_failed:
            return value
        else:
            raise VdtTypeError(value)
    return value


def is_color(value, default="k", alpha=False, as256=None):
    """Validate a matplotlib compatible color"""
    if str(value) == "None":
        return None
    from matplotlib.colors import ColorConverter

    CC = ColorConverter()
    if alpha:
        cc = CC.to_rgba
    else:
        cc = CC.to_rgb
    try:
        return cc(_check256_(value, as256=as256))
    except Exception:
        if isinstance(value, str):
            try:
                return cc(_check256_(eval(value), as256=as256))
            except Exception:
                raise VdtTypeError(value)

    raise VdtTypeError(value)


def _check256_(val, as256=None):
    if isinstance(val, str):
        return val
    if as256 is None:
        as256 = any([v > 1 for v in val[:3]])
    if as256:
        val = tuple([v / 256.0 for v in val[:3]]) + tuple(val[3:])
    return val


_re_funccall = re.compile(r"^\s*(\w+\(.*\))\s*$").match  # func(...)
_re_acc = re.compile(r"^\s*(\{.*\})\s*$").match  # {...}
_re_set = re.compile(r"^\s*(\w+)\s*=\s*(.+)\s*$").match  # a=b


def is_dict(value, default={}, vtype=None):
    """Validation function for dictionaries

    Examples
    --------
    value:

        - dict(a=2, b="x")
        - {"a":2, "b":"x"}
        - a=2
        - a=2, b="x"
        - OrderedDict(b=2)
        - dict([("a",2),("b","x")])
    """
    if str(value) == "None":
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        value = ", ".join(value)
    value = value.strip()
    if value == "":
        return {}
    m = _re_funccall(value) or _re_acc(value)
    if not m:
        m = _re_set(value)
        if not m:
            raise VdtTypeError(value)
        value = "dict(" + value + ")"
    if m:
        try:
            value = eval(value)
        except Exception:
            raise VdtTypeError(value)
        if not isinstance(value, dict):
            raise VdtTypeError(value)
        else:
            return value
    raise VdtTypeError(value)


# Define additionnal specifications
# Value should be dict for internal use of this module (iterable, opttype, ...)
# If value is not a dict, it is supposed to be the validator function


#: Available VACUMM :mod:`configobj` validator specifications
VALIDATOR_SPECS = {
    # copy of some validate.Validator.functions to later build plural forms
    "integer": validate.is_integer,
    "float": validate.is_float,
    "boolean": validate.is_boolean,
    "string": validate.is_string,
    "option": validate.is_option,
    # single value
    "date": is_cdtime,
    "cdtime": is_cdtime,
    "pydatetime": is_pydatetime,
    "datetime": is_datetime,
    "datetime64": is_datetime64,
    "timestamp": is_timestamp,
    "daterange": is_daterange,
    # "timeunits": is_timeunits,
    "minmax": is_minmax,
    "numerics": is_numerics,
    "figsize": is_figsize,
    "bbox": is_bbox,
    "datetime": is_datetime,
    "file": is_path,
    "path": is_path,
    "directory": is_path,
    "interval": is_interval,
    "eval": is_eval,
    "cmap": is_cmap,
    "color": is_color,
    "dict": is_dict,
    "boolstr": is_boolstr
    # lists validators for these scalars will be automatically generated
}

#: Available :mod:`validate.Validator` validator functions
VALIDATOR_FUNCTIONS = {}

#: Available VACUMM :mod:`configobj` validator type names
VALIDATOR_TYPES = []


def _update_registry_():

    # 1. Fix specs dicts
    # 2. Generate list validators
    for k, v in list(VALIDATOR_SPECS.items()):

        # Check type of spec
        if not isinstance(v, dict):
            v = dict(func=v)
            # Update specs mapping
            VALIDATOR_SPECS[k] = v

        # Check minimum settings
        v.setdefault("func", validate.is_string)
        v.setdefault("base_func", v["func"])
        v["func"] = _valwrap_(v["func"])
        v.setdefault("iterable", False)
        v.setdefault("opttype", k)
        v.setdefault("argtype", v["func"])

        # Add plural forms and validators to handle list values
        # TODO: we should remove this plural form which is not really correct
        if k.endswith("y"):
            nk = k[:-1] + "ies"
        elif k.endswith("x"):
            nk = k + "es"
        else:
            nk = k + "s"
        for nk in (nk, k + "_list"):
            if nk not in VALIDATOR_SPECS:
                nv = v.copy()
                nv["func"] = _valwraplist_(v["func"])
                nv["iterable"] = True
                VALIDATOR_SPECS[nk] = nv

    # List of names
    VALIDATOR_TYPES.clear()
    VALIDATOR_TYPES.extend(list(VALIDATOR_SPECS.keys()))
    VALIDATOR_TYPES.sort()

    # Dict of functions
    VALIDATOR_FUNCTIONS.clear()
    VALIDATOR_FUNCTIONS.update(
        ((k, v["func"]) for k, v in VALIDATOR_SPECS.items() if "func" in v)
    )


_update_registry_()

_for_doc = []
for key in VALIDATOR_TYPES:
    spec = VALIDATOR_SPECS[key]
    if not spec["func"].__name__.startswith("list_validator_wrapper"):
        val = ":func:`{} <{}>`".format(key, spec["base_func"].__name__)
    else:
        val = ":func:`{}`".format(key)
    _for_doc.append(val)

__doc__ = __doc__.format(" ".join(_for_doc))


def register_validation_functions(**kwargs):
    """Add a new configobj validator functions

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.cfgm import register_validation_functions

        def is_upper_string(mystr, default=""):
            mystr = str(mystr)
            if mystr == "None":
                mystr = ""
            return str(mystr).upper()

        register_validation_functions(is_upper_string=is_upper_string)

    """
    VALIDATOR_SPECS.update(**kwargs)
    _update_registry_()


def print_validation_functions(pattern="*"):
    """Print available xoa validator functions

    Parameters
    ----------
    pattern: str
        Only print function that matches this string

    Example
    -------

    .. ipython:: python

        @suppress
        from xoa.cfgm import print_validation_functions

        print_validation_functions("*time*")

    """
    from fnmatch import fnmatch
    for name in VALIDATOR_TYPES:
        specs = VALIDATOR_SPECS[name]
        func = specs["base_func"]
        if fnmatch(name, pattern):
            sig = str(inspect.signature(func))
            doc = inspect.getdoc(func)
            if doc:
                doc = doc.split("\n")[0]
                doc = f"\n    {doc}"
            else:
                doc = ""
            print(f"{name}{sig}{doc}")


class ConfigManager(object):
    """A configuration management class based on :class:`configobj`

    It supports content verification and default values thanks to
    :class:`validate.Validator`.

    Example
    -------
    >>> Cfg = Config('config.ini', interpolation='template')
    >>> Cfg.opt_parse()
    >>> cfg = Cfg.load('config.cfg')

    See also
    --------
    :class:`configobj.ConfigObj`, :class:`validate.Validator`

    """

    def __init__(
        self,
        cfgspecfile=None,
        validator=None,
        interpolation="template",
        encoding=None,
        boolean_false=True,
        splitsecdesc=False,
        cfgfilter=None,
        cfgfilter_default=False,
        warn_empty_specs=False,
    ):
        """
        Parameters
        ----------
        cfgspecfile: optional
            The specification file to be used with this.
        validator: :class:`validate.Validator`
            A custom :class:`validate.Validator`
            to use or a mapping dict of validator functions.
        interpolation: optional
            See :class:`configobj.ConfigObj`.
        boolean_false: optional
            Make sure that booleans have a default value.
        splitsecdesc: optional
            Section descriptions are split in two
            components separated by ':'.
        """
        # Specifications
        # - load
        self._encoding = encoding
        if isinstance(cfgspecfile, ConfigObj):
            self._configspec = cfgspecfile
        else:
            self._configspec = ConfigObj(
                cfgspecfile,
                list_values=False,
                interpolation=False,
                encoding=encoding,
                raise_errors=True,
                file_error=True,
            )
        if not self._configspec:
            if warn_empty_specs:
                warn("Empty Config specifications")
        else:
            # - filter
            if isinstance(cfgfilter, dict):
                filter_section(self._configspec, cfgfilter, cfgfilter_default)
            else:
                self._cfgfilter = None
            if not self._configspec and warn_empty_specs:
                xoa_warn("Empty Config specifications after filtering")
        self._cfgfilter = cfgfilter
        self._cfgfilter_default = cfgfilter_default
        self._configspecfile = self._configspec.filename

        # Validator
        if isinstance(validator, Validator):
            self._validator = validator
        else:
            self._validator = get_validator(functions=validator)

        # Makes sure that booleans have a default value
        self._boolean_false = boolean_false
        if boolean_false:
            self._configspec.walk(
                _walker_set_boolean_false_by_default_,
                validator=self._validator,
            )

        # Interpolation
        if interpolation is True:
            interpolation = "template"
        self._interpolation = interpolation

    @property
    def specs(self):
        return self._configspec

    cfgspecs = configspecs = specs

    @property
    def validator(self):
        return self._validator

    def get_spec(self, sec, key, **kwargs):
        """See :func:`get_spec`

        If sec is a basestring, use ``configspec[sec][key]``
        Otherwise use sec as a ``configspec, sec[key]``
        """
        return get_spec(
            (self._configspec[sec] if isinstance(sec, str) else sec)[key],
            validator=self._validator,
        )

    def get_defaults(self, nocomments=False, interpolation=None):
        """Get the default config

        Parameters
        ----------
        nocomments: boolean
            Do not include option comments in config file.
            If equal to 2, remove section comments too.
        interpolation: optional
            If True, interpolate values.

        Return
        ------
        A :class:`~configobj.ConfigObj` instance
        """
        if interpolation is None or interpolation is True:
            interpolation = self._interpolation
        cfg = ConfigObj(
            interpolation=interpolation,
            configspec=self._configspec,
            encoding=self._encoding,
        )
        cfg.validate(self._validator, copy=True)
        if nocomments:
            cfg.walk(
                _walker_remove_all_comments_,
                call_on_sections=int(nocomments) == 2,
            )
        elif self._configspec:
            cfg.inline_comments = self._configspec.inline_comments
        return cfg

    defaults = property(fget=get_defaults, doc="Default config")

    def reset(
        self, cfgfile="config.cfg", backup=True, nocomments=True, verbose=True
    ):
        """Reset a config file to default values

        Parameters
        ----------
        cfgfile: optional
            The configuration file to reset.
        backup: optional
            Backup the old config file.
        nocomments: optional
            Do not include comment in config file.

        Return
        ------
        :class:`~configobj.ConfigObj`
            A :class:`~configobj.ConfigObj` instance

        See also
        --------
        :meth:`defaults`
        """

        # Load defaults
        cfg = self.get_defaults(nocomments=nocomments, interpolation=False)

        # Remove old file
        if os.path.exists(cfgfile):
            if backup:
                shutil.copy(cfgfile, cfgfile + ".bak")
            os.remove(cfgfile)
        else:
            backup = False

        # Write to new one
        cfg.filename = cfgfile
        cfg.write()
        if verbose:
            print("Created default config file: {}".format(cfgfile))
            if backup:
                print("Backuped old config file to: {}.bak".format(cfgfile))
        return cfg

    def load(
        self,
        cfgfile="config.cfg",
        patch=None,
        validate=True,
        force=True,
        cfgfilter=False,
        **kwpatch,
    ):
        """Get a :class:`~configobj.ConfigObj` instance loaded from a file

        Parameters
        ----------
        cfgfile: optional
            config file

            - a config file name
            - a :class:`~configobj.ConfigObj` instance
            - ``None``: defaults to ``"config.cfg"``

        patch:
            A :class:`~configobj.ConfigObj` instance, a config file or
            a dictionary, used for patching.
        validate: optional
            Type of validation

            - ``False``: no validation
            - ``"fix"``: validation fixes and reports errors
            - ``"report"``: validation reports errors
            - ``"raise"``: validation raises errors

        force: optional
            Force re-instantiation of ``cfgfile`` when it is already
            a :class:`ConfigObj` instance.

        Return
        ------
        tuple, :class:`~configobj.ConfigObj`
            Depends on ``geterr``

            - if ``True``: ``(cfg, err)`` where is the result
              of :meth:`~configobj.ConfigObj.validate`
            - else: ``cfg`` (:class:`~configobj.ConfigObj` instance)
        """

        # Load the config
        if (
            cfgfile is not None
            and isinstance(cfgfile, str)
            and not os.path.exists(cfgfile)
        ):
            cfgfile = None

        # Instantiate / Copy
        if not isinstance(cfgfile, ConfigObj) or force:
            cfg = ConfigObj(
                cfgfile,
                interpolation=self._interpolation,
                configspec=self._configspec,
                encoding=self._encoding,
            )
        else:
            cfg = cfgfile

        # Patch
        if kwpatch and not patch:
            patch = {}
        if patch is not None:
            if kwpatch:  # Patch the patch!
                patch = self.patch(patch, kwpatch, validate=False)
            self.patch(cfg, patch, validate=False)

        # Filter
        if self._cfgfilter and cfgfilter:
            if not isinstance(cfgfilter, dict):
                cfgfilter = self._cfgfilter
            filter_section(cfg, cfgfilter, self._cfgfilter_default)

        # Validation
        if validate and self._configspec:

            # Validation itself
            err = cfg.validate(self._validator, preserve_errors=True)

            # Loop on errors
            if isinstance(err, dict):

                for sections, key, error in flatten_errors(cfg, err):

                    # Format explicit message
                    if len(sections):
                        section = "[" + "][".join(sections) + "] "
                    else:
                        section = ""
                    msg = "Config value error: {}{}: {}".format(
                        section, key, getattr(error, "message", error),
                    )

                    # Raise explicit error
                    raise XoaValidateError(msg)

        return cfg

    def patch(self, cfg, cfgpatch, validate=False):
        """Replace config values of ``cfg`` by those of ``cfgpatch``

        Parameters
        ----------
        cfg:
            A :class:`~configobj.ConfigObj` instance, a config file or
            a dictionary, that must be patched.
        cfgpatch:
            A :class:`~configobj.ConfigObj` instance, a config file or
            a dictionary, used for patching.
        validate: optional
            If ``True``, validate configs if they have a valid config spec.
        """
        if not isinstance(cfg, ConfigObj):
            cfg = ConfigObj(
                cfg, configspec=self._configspec, encoding=self._encoding
            )  # , interpolation=False)
        if not isinstance(cfgpatch, ConfigObj):
            cfgpatch = ConfigObj(
                cfgpatch,
                configspec=self._configspec,
                interpolation=False,
                encoding=self._encoding,
            )
        else:
            cfgpatch.interpolation = False

        # Merging based on specs with type check
        self.cfgspecs.walk(_walker_patch_, cfg=cfg, cfgpatch=cfgpatch)

        # Merging of missing stuff
        cfgpatch.walk(_walker_patch_, cfg=cfg, cfgpatch=cfgpatch)

        if validate and cfg.configspec is not None:
            cfg.validate(self.validator)

        return cfg

    def arg_parse(
        self,
        parser=None,
        exc=[],
        parse=True,
        args=None,
        getparser=False,
        getargs=False,
        cfgfile="config.cfg",
        patch=None,
        cfgfileopt="--cfgfile",
        cfgfilepatch="before",
        nested=None,
        extraopts=None,
    ):
        """Commandline options (:mod:`argparse`) and config mixer.

            1. Creates command-line options from config defaults
            2. Parse command-line argument and create a configuration patch

        For instance, the following config define the commandline option
        ``--section1-my-section2-my-key`` with ``value`` as a default value,
        stored in a special group of options with a short name
        and a long description::

            [section1] # Short name : long description of the group
                [[my_section2]]
                    my_key=value

        .. warning:: Section and option names must not contain any
            space-like character !

        .. note::

            If you want to prevent conflict of options, don't use ``"_"`` in
            section and option names.

        Parameters
        ----------
        parser:
            optional, a default one is created if not given. This can be:
                - a :class:`OptionParser` instance
                - a :class:`dict` with keyword arguments for the one
                  to be created
        exc: optional, list
            List of keys to be excluded from parsing.
        parse: optional, bool
            If ``True``, parse commande line options and arguments
        args: optional
            List of arguments to parse instead of default sys.argv[1:]
        getparser: optional, bool
            Allow getting the parser in addition to the config if parse=True
        getargs: optional, bool
            allow getting the parsed arguments in addition to the
            config if parse=True
        patch: optional
            Used if parse is True.
            Can take the following values:

            - a :class:`bool` value indicating wheter to apply
              defaults on the returned config
              before applying the command line config
            - a :class:`ConfigObj` instance to apply on the returned config
              before applying the command line config

        cfgfileopt: optional
            If present a config file option will be added.
            Can be a :class:`string` or couple of strings to use
            as the option short and/or long name
        cfgfilepatch:
            specify if the returned config must be patched if a
            config file command line option is provided and when to patch it.
            Can take the following values:

            - True or 'before': the config file would be used before
              command line options
            - 'after': the config file would be used after command line options
            - Any False like value: the config file would not be used

        nested: str
            Name of a section whose defines the configuration.
            It must be used when the configuration in nested in more g
            eneral configuration.
        extraopts: dict
            Extra options to declare in the form
            ``[(args1, kwargs1), ((args2, kwargs2), ...]``

        Return
        ------
        :class:`OptionParser`
            if parse is False
        :class:`~configobj.ConfigObj`
            if parse is True and getparser is not True
        (:class:`~configobj.ConfigObj`, :class:`OptionParser`)
            if both parse and getparser are True
        """

        # Prepare the option parser
        if parser is None:
            parser = ArgumentParser(add_help=False)
        elif isinstance(parser, dict):
            parser["add_help"] = False
            parser = ArgumentParser(**parser)

        # Add short and long helps
        old_conflict_handler = parser._optionals.conflict_handler
        parser._optionals.conflict_handler = "resolve"
        parser.add_argument(
            "-h",
            "--help",
            action=_AP_ShortHelpAction,
            help="show a reduced help and exit",
        )
        parser.add_argument(
            "--long-help",
            action=_HelpAction,
            help="show an extended help and exit",
        )
        parser.add_argument(
            "--short-help",
            action=_AP_VeryShortHelpAction,
            help="show a very reduced help and exit",
        )
        parser._optionals.conflict_handler = old_conflict_handler

        # Add extra options first
        if extraopts:
            for eopt in extraopts:
                if len(eopt) == 1:
                    if not isinstance(eopt, dict):
                        eargs = eopt
                        ekwargs = {}
                    else:
                        eargs = []
                        ekwargs = eopt
                else:
                    eargs, ekwargs = eopt
            parser.add_argument(*eargs, **ekwargs)

        # Add the cfgfile option (configurable)
        if cfgfileopt:
            if isinstance(cfgfileopt, str):
                if not cfgfileopt.startswith("-"):
                    if len(cfgfileopt) == 1:
                        cfgfileopt = "-" + cfgfileopt
                    else:
                        cfgfileopt = "--" + cfgfileopt
                cfgfileopt = (cfgfileopt,)
            parser.add_argument(
                *cfgfileopt,
                dest="cfgfile",
                help="user configuration file that overrides defauts"
                ' [default: "{default}"]',
                default=cfgfile,
            )

        # Default config
        defaults = self.defaults

        # Create global group of options from defaults
        # - inits
        re_match_initcom = re.compile(
            r"#\s*-\*-\s*coding\s*:\s*\S+\s*-\*-\s*"
        ).match
        if (
            len(defaults.initial_comment) == 0
            or re_match_initcom(defaults.initial_comment[0]) is None
        ):
            desc = ["global configuration options"]
        else:
            re_match_initcom(
                defaults.initial_comment[0]
            ), defaults.initial_comment[0]
            icom = int(
                re_match_initcom(defaults.initial_comment[0]) is not None
            )
            if len(defaults.initial_comment) > icom:
                desc = defaults.initial_comment[icom].strip("# ").split(":", 1)
        group = parser.add_argument_group(*desc)
        # - global options
        for key in defaults.scalars:
            if key not in exc:
                _walker_argcfg_setarg_(
                    defaults,
                    key,
                    group=group,
                    exc=exc,
                    nested=nested,
                    boolean_false=self._boolean_false,
                )
            #                group.add_argument('--'+_cfg2optname_(key, nested), help=_shelp_(defaults, key))
            else:
                pass

        # Create secondary option groups from defaults
        for key in defaults.sections:
            desc = [key]
            comment = defaults.inline_comments[key]  # FIXME: always empty!
            if comment is not None:
                desc = comment.strip("# ")
                if ":" in desc:
                    desc = desc.split(":", 1)
                else:
                    desc = [key.lower(), desc]
            group = parser.add_argument_group(*desc)
            defaults[key].walk(
                _walker_argcfg_setarg_,
                raise_errors=True,
                call_on_sections=False,
                group=group,
                exc=exc,
                nested=nested,
                boolean_false=self._boolean_false,
                validator=self.validator
            )

        # Now create a configuration instance from passed options
        if parse:

            # Which args ?
            if args is None:
                args = sys.argv[1:]

            # Parse
            options = parser.parse_args(list(args))

            # Create a configuration to feed
            cfg = ConfigObj(
                interpolation=self._interpolation, encoding=self._encoding
            )

            # Initial config from defaults or the one supplied
            if patch:
                self.patch(
                    cfg, patch if isinstance(patch, ConfigObj) else defaults
                )
            if cfgfilepatch:
                if isinstance(
                    cfgfilepatch, str
                ) and cfgfilepatch.strip().lower().startswith("a"):
                    cfgfilepatch = "after"
                else:
                    cfgfilepatch = "before"

            # Feed config with cfgfile before command line options
            if cfgfilepatch == "before" and getattr(options, "cfgfile", None):
                cfg = self.patch(cfg, self.load(options.cfgfile))

            # Feed config with command line options
            defaults.walk(
                _walker_argcfg_setcfg_,
                raise_errors=True,
                call_on_sections=False,
                cfg=cfg,
                options=options,
                nested=nested,
            )

            # Feed config with cfgfile after command line options
            if cfgfilepatch == "after" and getattr(options, "cfgfile", None):
                cfg = self.patch(cfg, self.load(options.cfgfile))

            if not getparser and not getargs:
                return cfg
            out = (cfg,)
            if getparser:
                out += (parser,)
            if getargs:
                # options.xoa_cfg = cfg
                out += (options,)
            return out

        else:
            return parser

    def arg_patch(self, parser, exc=[], cfgfileopt="cfgfile"):
        """Call to :meth:`arg_parse` and :meth:`patch`

        Return
        ------
        :class:`~configobj.ConfigObj`
        :class:`OptionParser`
        """

        # Create a patch configuration from commandline arguments
        cfgpatch, args = self.arg_parse(
            parser, exc=exc, cfgfileopt=cfgfileopt, getargs=True
        )

        #  Load personal config file and default values
        cfg = self.load(args.cfgfile)

        #  Patch it with commandline options
        self.patch(cfg, cfgpatch)

        return cfg, args

    def arg_long_help(
        self,
        rst=True,
        usage=None,
        description="Long help based on config specs",
    ):
        """Get the generic long help from config specs

        Parameters
        ----------
        rst: optional
            Reformat output in rst.
        """

        # Standard options
        parser = ArgumentParser(
            usage=usage, description=description, add_help=False
        )
        parser.add_argument(
            "-h", "--help", action="store_true", help="show a reduced help"
        )
        parser.add_argument(
            "--long-help", action="store_true", help="show an extended help"
        )
        parser.add_argument(
            "--short-help",
            action="store_true",
            help="show a very reduced help",
        )

        # Configuration options
        self.arg_parse(parser, parse=False)
        if rst:
            formatter = ArgHelpFormatter(max_help_position=0)
            for action_group in parser._action_groups:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()
            shelp = formatter.format_help()
        else:
            shelp = parser.format_help()

        # Encoding
        shelp = shelp.encode(sys.getdefaultencoding(), "replace")

        # Convert to rst
        if rst:
            shelp = opt2rst(shelp)

        return shelp

    def to_rst(self, mode="specs", **kwargs):
        """Convert the default config to rst with :func:`cfg2rst`"""
        return cfg2rst(self, mode=mode, **kwargs)

    get_rst = to_rst  # compat


def filter_section(sec, cfgfilter, default=False):
    """Recursively filter a section according to a dict of specifications

    When encountering an option of ``sec``, it removed if its value
    in ``cfgfilter`` is set to False. When not found, it default to the
    ``__default__`` key of ``cfgfilter``. And if the ``__default__`` is not
    found, it defaults to ``False`` (filtered out).
    When an option is a section and its value in ``cfgfilter`` is a dictionary,
    this subsection is filtered in the same way with the value as restrictions
    (``cfgfilter[subsection]``).

    Parameters
    ----------
    sec:
        A :class:`configobj.Section` instance.
    cfgfilter:
        A dictionary tree with the same structure as ``sec``.

    """
    # Default behavior
    default = cfgfilter.get("__default__", default)

    # Exceptions
    excepts = cfgfilter.get("__excepts__", None)
    if excepts is not None and not isinstance(excepts, list):
        excepts = [excepts]

    # First pass on level 0
    for key in sec:
        kdefault = (
            default if excepts is None or key not in excepts else not default
        )
        if not cfgfilter.get(key, kdefault):
            del sec[key]

    # Filter subsections
    for subsec in sec.sections:
        if subsec in cfgfilter:
            if isinstance(cfgfilter[subsec], dict):
                filter_section(sec[subsec], cfgfilter[subsec])
    return sec


def cfgargparse(
    cfgspecfile,
    parser,
    cfgfileopt="cfgfile",
    cfgfile="config.cfg",
    exc=[],
    extraopts=None,
    args=None,
    **kwargs,
):
    """Merge configuration and commandline arguments

    Parameters
    ----------
    cfgspecfile:
        Config specification file (.ini).
    parser:
        :class:`~argpase.ArgumentParser` instance.
    cfgfileopt: optional
        Name of the option used to specify the
        user config file. Example: ``'cfgfile'`` creates the option
        ``--cfgfile=<config file>``.
    cfgfile: optional
        Default name for the loaded config file.
    exc: optional
        Config option name that must not be used to generated
        a commandline option.
    **kwargs
        Extra params are passed to :class:`ConfigManager` initialization.

    Return
    ------
    :class:`ConfigObj`

    Tasks:

        1. Initialize a default configuration (:class:`ConfigManager`)
           from the specification file given by ``cfgspecfile``.
        2. Generate associated commandline options.
        3. Load a user configuration file (specified with the option
           whose name is given by ``cfgfileopt``).
        4. Patch this configuration with user supplied options retrieved
           using the :class:`~argpase.ArgumentParser` parser ``parser``.

        Technically it combines :class:`ConfigManager` and
        :meth:`ConfigManager.arg_parse`
    """
    return ConfigManager(cfgspecfile, **kwargs).arg_parse(
        parser,
        cfgfileopt=cfgfileopt,
        exc=exc,
        cfgfile=cfgfile,
        getargs=True,
        extraopts=extraopts,
        args=args,
    )


def opt2rst(shelp, prog=None, secfmt=":{secname}:", descname="Description"):
    """Convert --help str to rst

    This is useful for autodocumenting executable python scripts
    that show a formatted help.

    Parameters
    ----------
    shelp: str
        Help string showing options (results from :option:``--help``).
    prog: optional, str
        Program name, otherwise guess it from usage.

    Return
    ------
    str
        String converted to rst format (with :rst:dir:`cmdoption` directives).

    """
    rhelp = []
    multiline = False
    s_param = r"(?:\{[\w,]+\}|\w+)"
    s_sopt = rf"(?:-\w+(?: {s_param})?)"  # short option (-t)
    # long option (--toto)
    s_lopt = rf"(?:--[\w\-]+(?:[= ]+{s_param})?)"
    s_optsep = r"(?:, +)"  # option separator
    s_desc = r"(?:  (.+))"
    s_tot = (
        rf"^  (?:  )?((?:{s_sopt}|{s_lopt})(?:{s_optsep}"
        rf"(?:{s_sopt}|{s_lopt}))*){s_desc}?$"
    )
    re_opt = re.compile(s_tot).match
    re_sec = re.compile(r"^(?:  )?([\w\s]+):(?: (.+))?$").match
    secname = None
    for line in shelp.splitlines():

        # Sections
        m = re_sec(line)
        if m and not line.lower().endswith("ex:"):

            secname = m.group(1).title().strip()

            # Usage
            if secname == "Usage" and m.group(2) is not None:
                usage = m.group(2).strip()
                if prog is None:
                    prog = os.path.basename(usage.split()[0])
                rhelp.append(f".. program:: {prog}\n")
                rhelp.extend(
                    [
                        secfmt.format(locals()),
                        "\n\t.. code-block:: bash\n\n\t\t" + usage,
                    ]
                )
                multiline = True
            else:
                rhelp.extend([secfmt.format(locals()), ""])
                if m.group(2) is not None:
                    rhelp.extend(["", "\t" + m.group(2)])
                    multiline = True
            continue

        # Options and other lines
        m = re_opt(line)
        if m:

            rhelp.extend(["", "\t.. cmdoption:: " + m.group(1), ""])
            multiline = True
            if m.group(2) is not None:
                rhelp.append("\t\t" + m.group(2).strip())

        elif (
            secname
            and secname.lower() == "positional arguments"
            and line.startswith(" " * 2)
        ):

            sline = line.split()
            rhelp.extend(["", "\t.. cmdoption:: " + sline[0], ""])
            multiline = True
            if len(sline) > 1:
                rhelp.append("\t\t" + " ".join(sline[1:]))

        elif multiline and len(line.strip()) and line.startswith(" " * 3):

            indent = "\t\t"
            if secname == "Usage":
                indent += "\t"
            rhelp.append(indent + line.strip())
        # elif secname==descname:
        #    rhelp.append('\t'+line)

        else:

            indent = ""
            if secname and secname == descname:
                indent += "\t"
            rhelp.append(indent + line)
            multiline = False
            if secname == "Usage":
                secname = descname
                rhelp.extend([secfmt.format(locals()), ""])

    return "\n".join(rhelp)


def _opt2cfgname_(name, nested):
    cfgkey = name.replace("-", "_")
    if nested and cfgkey.startswith(nested + "_"):
        cfgkey = cfgkey[len(nested + "_"):]
    return cfgkey


_re_cfg2optname_sub = re.compile(r"[_\s]").sub


def _cfg2optname_(name, nested=None):
    optkey = _re_cfg2optname_sub("-", name)
    if nested:
        optkey = nested + "-" + optkey
    return optkey.lower()


class _attdict_(dict):
    def __getattr__(self, name):
        if name in self.__dict__:
            return object.__getattribute__(self, name)
        else:
            return self[name]


def get_spec(spec, validator=None):
    """ Get an option specification

    Parameters
    ----------
    spec:
        the specification string
    validator:
        (optional) the validator to use

    Return
    ------
    dict
        A dict with keys:

        funcname:
            the validation type name
        type:
            same as funcname
        args:
            the positionnal arguments
        kwargs:
            the named arguments
        default:
            the default value
        iterable:
            if the value is list-like
        func:
            the validation function
        argtype:
            the function used with :mod:`argparse`


    Read access to these keys can also be done as attribute
    of the returned dict (``d.funcname == d['funcname']``, ...)

    For example, a specification file containing::

        [section]
            option = integer(default=0, min=-10, max=10)

    Would return::

        {'funcname': integer, 'args': [],
        'kwargs': {'min': '-10', 'max': '10'}, 'default:' 0,
        'opttype': 'int', 'argtype': int,
        'func':is_integer, 'iterable': None}

    This can be usefull when you added extraneous named arguments into your
    specification file for your own use.

    """
    if not validator:
        validator = get_validator()
    funcname, args, kwargs, default = validator._parse_with_caching(spec)
    spec = VALIDATOR_SPECS.get(
        funcname, dict(func=None, iterable=None, opttype=None, argtype=None)
    ).copy()
    spec.update(
        dict(funcname=funcname, args=args, kwargs=kwargs, default=default, type=funcname)
    )
    return _attdict_(spec)


def get_validator(functions=None, cls=Validator, **kwargs):
    """Get a default validator"""

    # Init
    validator = cls(**kwargs)

    # This modules's validator functions are already wrapped
    validator.functions.update(VALIDATOR_FUNCTIONS)

    # User defined functions
    if functions:
        validator.functions.update(functions)

    # Wrap default functions to handle none and extra args
    for k, v in validator.functions.items():
        validator.functions[k] = _valwrap_(v)

    return validator


def _walker_remove_all_comments_(sec, key):
    sec.comments[key] = ""
    sec.inline_comments[key] = ""


def _walker_remove_comments_(sec, key):
    sec.comments[key] = ""


def _walker_remove_inline_comments_(sec, key):
    sec.inline_comments[key] = ""


def _walker_unchanged_options_(sec, key):
    if not sec.configspec:
        return
    spec = get_spec(sec.configspec.get(key, ""))
    return spec.default == sec[key]


def remove_defaults(cfg):
    defaults = cfg.walk(_walker_unchanged_options_, call_on_sections=False)

    def remove(c, d):
        for k, v in d.items():
            if isinstance(v, dict):
                remove(c[k], v)
            elif v:
                c.pop(k)

    remove(cfg, defaults)


def _walker_argcfg_setcfg_(sec, key, cfg=None, options=None, nested=None):
    """Walker to set config values"""
    # Find genealogy
    parents = _parent_list_(sec, names=False)
    cfgkey = "_".join([p.name.strip("_") for p in parents] + [key])
    for option, value in options._get_kwargs():

        # Option not set
        if value is None:
            continue

        # Option matches key?
        if _opt2cfgname_(option, nested) != cfgkey.lower():
            continue

        # Check or create cfg genealogy
        s = cfg
        for p in parents:
            if p.name not in s:
                s[p.name] = {}
            s = s[p.name]
        s[key] = value


def _walker_argcfg_setarg_(
    sec,
    key,
    group=None,
    exc=None,
    nested=None,
    encoding=None,
    boolean_false=True,
    validator=None
):
    """Walker to set options"""
    # Find option key and output var name
    key = key.strip("_")
    pp = _parent_list_(sec)
    varname = "_".join(pp + [key])
    optkey = _cfg2optname_(varname, nested)
    if validator is None:
        validator = get_validator()

    # Check exceptions
    if key in exc:
        return
    if sec.configspec is None or key not in sec.configspec:
        return
    spec = VALIDATOR_SPECS.get(sec.configspec[key].split("(", 1)[0], {})

    # Define the wrapping function for argparse argument types
    # which also handle list values
    def wrap_argparse_type(func, islist):
        def wrapper_argparse_type(value):
            if islist:  # Use configobj list parser
                value, comment = ConfigObj(
                    list_values=True, interpolation=False, encoding=encoding
                )._handle_value(value)
                return func(value)
            else:
                return func(value)

        wrapper_argparse_type.__name__ += "-" + func.__name__
        return wrapper_argparse_type

    # Add argument to group
    def func(value):
        return validator.check(sec.configspec[key], value)
    # func = spec.get("func", lambda s: s)
    argtype = wrap_argparse_type(func, spec.get("iterable", None))
    kw = {}
    if boolean_false and spec.get("opttype", "") == "boolean":
        default = sec[key]
        action = "store_true" if default is False else "store_false"
    else:
        action = "store"
        kw["type"] = argtype
    group.add_argument(
        "--" + optkey, action=action, help=_shelp_(sec, key), **kw
    )


def _shelp_(
    sec,
    key,
    format="{shelp} [default: {default}]",
    mode="auto",
    undoc="Undocumented",
    adddot=True,
):
    """Get help string

    Parameters
    ----------
    mode: string

        - inline: inline comment only,
        - above: above comments only,
        - merge: merge inline and above comments,
        - auto: if one is empty use the other one, else use inline

    """
    # filter
    def strip(c):
        return c.strip().strip("#").strip()

    abcoms = list(map(strip, [c for c in sec.comments[key] if c is not None]))
    incoms = list(
        map(strip, [c for c in [sec.inline_comments[key]] if c is not None])
    )

    # Merge comments above item and its inline comment
    if mode == "merge":
        comments = abcoms + incoms
    elif mode == "auto":
        if not incoms:
            comments = abcoms
        else:
            comments = incoms
    elif mode == "above":
        comments = abcoms
    else:
        comments = incoms

    # Force comments to end with a dot '.'
    if adddot:
        comments = [c.endswith(".") and c or f"{c}." for c in comments]
    shelp = "\n".join(comments)

    # If no comments
    if not shelp:
        shelp = undoc
    default = _sdefault_(sec, key)
    return format.format(**locals())


def _sdefault_(sec, key):
    """Get default string or None"""
    default = sec.get(key, None)
    if isinstance(default, (list, tuple)):
        default = ",".join(map(str, default))
    else:
        default = str(default).strip("()")
    default = default.replace("%", "%%")
    if default == "None":
        default = None
    return default


def _parent_list_(sec, names=True):
    parents = []
    while sec.name is not None:
        parents.insert(0, names and sec.name.strip("_") or sec)
        sec = sec.parent
    return parents


def _walker_patch_(sec, patch_key, cfg, cfgpatch):
    """Walk through the patch to apply it"""
    psec = cfgpatch
    csec = cfg
    for key in _parent_list_(sec):
        if key not in psec.sections:  # nothing to patch
            return
        if key not in csec.sections:
            csec[key] = {}
        csec = csec[key]
        psec = psec[key]
    if patch_key not in psec:  # nothing to patch
        return
    if patch_key in csec:
        try:
            psec[patch_key] = type(csec[patch_key])(psec[patch_key])
        except Exception:
            pass
    csec[patch_key] = psec[patch_key]


def _walker_set_boolean_false_by_default_(sec, key, validator=None):
    if validator is None:
        return
    check = sec[key]
    fun_name, fun_args, fun_kwargs, default = validator._parse_with_caching(
        check
    )
    if fun_name == "boolean" and default is None:
        if fun_args or fun_kwargs:
            check = check[:-1] + ", default=False)"
        else:
            check = check + "(default=False)"
        sec[key] = check


def get_sec_names(cfg):
    """Get section names as list from top to bottom ['sec0','sec1',...]"""
    if cfg.depth == 0:
        return []
    secnames = [cfg.name]
    for i in range(cfg.depth - 1):
        cfg = cfg.parent
        secnames.append(cfg.name)
    return secnames[::-1]


def get_cfg_path(cfg, entry=None, sep="."):
    """Get a string representing the path of a ConfigObj through sections

    Parameters
    ----------
    cfg: configobj.ConfigObj
    entry: str, None
    sep: str

    Example
    -------
    >>> c=configobj.ConfigObj({'a':{'b':{'c':'d'}}})
    >>> get_cfg_path(c['a']['b'])
    'a.b'
    >>> get_cfg_path(c['a'], 'b')
    'a.b'
    >>> get_cfg_path(c['a']['b'], 'c', '::')
    'a::b::c'

    """
    ancestors = []
    if entry is not None:
        ancestors.append(entry)
    curcfg = cfg
    while curcfg.depth > 0 and curcfg.parent is not curcfg:
        ancestors.append(curcfg.name)
        curcfg = curcfg.parent
    return sep.join(reversed(ancestors))


def _redent(text, n=1, indent="    "):
    lines = text.split("\n")
    lines = [(n * indent + line) for line in lines]
    return "\n".join(lines)


def cfg2rst(cfg, mode="basic", optrole="confopt", secrole="confsec", **kwargs):
    """Convert a configuration to rst format

    Configuration sections are declared by default with the rst :rst:dir:`confsec` directive
    and options are declared with the rst :rst:dir:`confopt` directive

    For instance:

    .. code-block:: ini

        a=1 # desc a
        [s1] # desc s1
            b=2  # desc b
            [[s2]] # desc s2
                c=$a-$b # desd c
                [sec1] # section 1

    is converted to:

    .. code-block:: rst

        .. confopt:: a

            desc a

        .. confsec:: [s1]

            desc s1

            .. confopt:: [s1] b

                desc b

            .. confsec:: [s1][s2]

                desc s2

                .. confopt:: [s1][s2] c

                    desd c

    Then one can reference an option with for example ``:confopt:`[s1][s2]c```.

    Parameters
    ----------
    cfg: configobj.ConfigObj, ConfigManager
        In the case of a :class:`ConfigManager`,
        the :meth:`~ConfigManager.get_defaults`
        are used.
    mode: {"basic", "values", "specs"}

        ``"basic"``:
            Only display the config section and option names and description.
        ``"values"``:
            Also display the option values.
            A classic config.
        ``"specs"``:
            Also display the type and default value.
            Config specifications are expected for cfg.

    Return
    ------
    str
    """
    if isinstance(cfg, ConfigManager):
        if mode == "specs":
            cfg = cfg.specs
        else:
            cfg = cfg.defaults
    lines = []
    cfg.walk(
        _walker_cfg2rst_,
        call_on_sections=True,
        lines=lines,
        optrole=optrole,
        secrole=secrole,
        mode=mode,
        **kwargs,
    )
    return "\n".join(lines)


def _walker_cfg2rst_(
    cfg,
    key,
    lines,
    optrole="cfgopt",
    secrole="cfgsec",
    mode="basic",
    validator=None,
    dir_fmt=".. {conftype}:: {name}\n\n{desc}\n",
    desc_fmt_desc_item="| {key}: {val}\n",
):

    assert mode in ("basic", "values", "specs")

    # Name, values and type
    secnames = get_sec_names(cfg)
    specs = OrderedDict()
    if key in cfg.sections:  # section

        secnames.append(key)
        name = "[{}]".format("][".join(secnames))
        conftype = secrole

    else:  # option

        if secnames:
            name = "[{}] {}".format("][".join(secnames), key)
        else:
            name = key
        conftype = optrole

        if mode == "values":

            specs["default"] = cfg[key]

        elif mode == "specs":

            spec = get_spec(cfg[key], validator=validator)
            specs["default"] = spec["default"]
            func = spec["base_func"]
            funcpath = f"{func.__module__}.{func.__name__}"
            funcname = spec['funcname']
            specs["type"] = f":func:`{funcname} <{funcpath}>`"
            if spec["args"]:
                skey = "possible choices" if name == "choice" else "args"
                specs[skey] = spec["args"]
            if spec["kwargs"]:
                specs.update(spec["kwargs"])
    if specs:  # join lists
        for k, v in specs.items():
            if isinstance(v, list):
                specs[k] = ", ".join(map(str, v))

    # Description
    desc = cfg.inline_comments.get(key)
    if desc is None:
        desc = ""
    desc = desc.strip("#").strip().capitalize()

    # Formatting
    # - description with specs
    if specs:
        sdesc = ""
        for key, val in specs.items():
            if val == "":
                val = " "
            elif not isinstance(val, str) or not val.startswith(":"):
                val = f"``{val!s}``"
            sdesc = sdesc + desc_fmt_desc_item.format(**locals())
        desc = sdesc + "\n" + desc
    # - directive
    desc = _redent(desc, 1)
    text = dir_fmt.format(**locals())
    text = _redent(text, cfg.depth)
    lines.append(text)


def print_short_help(parser, formatter=None, compressed=False):
    """Print all help of a parser instance but those of groups"""
    if isinstance(parser, ArgumentParser):

        if formatter is None:
            formatter = parser._get_formatter()

        # usage
        if compressed is True:
            compressed = [
                "-h",
                "-help",
                "--long-help",
                "--short-help",
                "--cfgfile",
            ]
        elif compressed and isinstance(compressed, str):
            compressed = [compressed]
        if compressed:
            fake_action = AP_Action(["other-options"], dest="")

            def valid_option(opt):
                if not opt.option_strings:
                    return True
                for optstr in opt.option_strings:
                    if optstr in compressed:
                        return True

            actions = list(filter(valid_option, parser._actions))
            actions.append(fake_action)
        else:
            actions = parser._actions
        formatter.add_usage(
            parser.usage, actions, parser._mutually_exclusive_groups
        )

        # description
        formatter.add_text(parser.description)

        # positionals, optionals and user-defined groups
        for action_group in parser._action_groups:
            if action_group.title in [
                "positional arguments",
                "optional arguments",
            ]:
                formatter.start_section(action_group.title)
                formatter.add_text(action_group.description)
                formatter.add_arguments(action_group._group_actions)
                formatter.end_section()

        # epilog
        formatter.add_text(parser.epilog)

        # determine help from format above
        parser._print_message(formatter.format_help(), sys.stdout)


class _AP_ShortHelpAction(_HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        print_short_help(parser)
        parser.exit()


class _AP_VeryShortHelpAction(_HelpAction):
    def __call__(self, parser, namespace, values, option_string=None):
        print_short_help(parser, compressed=True)
        parser.exit()


# %% Sphinx extension

def gen_cfgm_rst(app):

    logging.info("Generating rst declarations for the ConfigManager")

    rst = app.config.cfgm_get_cfgm_func().get_rst(secrole="cfgmsec", optrole="cfgmopt")

    outfile = os.path.abspath(app.config.cfgm_rst_file)
    outdir = os.path.dirname(outfile)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    with open(outfile, "w") as f:
        f.write(rst)

    logging.info("Created: " + app.config.cfgm_rst_file)


def setup(app):

    app.add_object_type('cfgmopt', 'cfgmopt',
                        objname='configuration option',
                        indextemplate='pair: %s; configuration option')
    app.add_object_type('cfgmsec', 'cfgmsec',
                        objname='configuration section',
                        indextemplate='pair: %s; configuration section')

    app.add_config_value('cfgm_get_cfgm_func', None, 'html')
    app.add_config_value('cfgm_rst_file', 'cfgm.rst', 'html', types=[str])

    app.connect('builder-inited', gen_cfgm_rst)

    return {'version': '0.1'}
