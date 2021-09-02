#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xarray and pandas xoa accessors

"""
# Copyright 2020-2021 Shom

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

from .misc import ERRORS


class _BasicCFAccessor_(object):
    def __init__(self, obj, cfspecs=None):
        from . import cf

        if cfspecs is None:
            cfspecs = cf.infer_cf_specs(obj)
        self._obj = obj
        self.set_cf_specs(cfspecs)

    def _assign_cf_specs_(self):
        from . import cf

        if self._cfspecs.name:
            self._obj = cf.assign_cf_specs(self._obj, self._cfspecs, register=True)

    def set_cf_specs(self, cfspecs):
        """Set the internal :class:`~xoa.cf.CFSpecs` used by this accessor

        If the specs object has a :cfsec:`registration name <register>`,
        it is assigned to
        the current object and its children with function
        :func:`~xoa.cf.assign_cf_specs`. This set the "cfspecs" encoding
        to the name.

        Parameters
        ----------
        cfspecs: xoa.cf.CFSpecs

        See also
        --------
        :ref:`uses.cf`
        """
        from . import cf

        assert isinstance(cfspecs, cf.CFSpecs)
        self._cfspecs = cfspecs
        self._assign_cf_specs_()

    def get_cf_specs(self):
        """Get the internal :class:`~xoa.cf.CFSpecs` instance used by this accessor

        If not provided at the initialization, it is infered with :func:`xoa.cf.infer_cf_specs`.

        Return
        ------
        xoa.cf.CFSpecs

        See also
        --------
        :ref:`uses.cf`
        """
        return self._cfspecs

    cfspecs = property(fget=get_cf_specs, fset=set_cf_specs, doc="The CFSpecs instance")


class _CFAccessor_(_BasicCFAccessor_):
    _search_category = None

    def __init__(self, obj, cfspecs=None):
        _BasicCFAccessor_.__init__(self, obj, cfspecs)
        self._coords = None
        self._data_vars = None
        self._cache = {}
        self._obj = self.infer_coords()
        self._assign_cf_specs_()

    @ERRORS.format_method_docstring
    def get(self, cf_name, loc="any", single=True, errors="ignore"):
        """Search for a data array or coordinate knowing its generic CF name

        Search is made with the :meth:`~xoa.cf.CFSpecs.search` method of the
        :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Parameters
        ----------
        name: str
        Generic CF name
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or "": no location
        {errors}

        See also
        --------
        :meth:`xoa.cf.CFSpecs.search`
        :meth:`xoa.cf.CFCoordSpecs.search`
        :meth:`xoa.cf.CFVarSpecs.search`
        """
        kwargs = dict(cf_name=cf_name, loc=loc, get="obj", single=single, errors=errors)
        if self._search_category is None:
            return self._cfspecs.search(self._obj, **kwargs)
        return self._cfspecs[self._search_category].search(self._obj, **kwargs)

    @ERRORS.format_method_docstring
    def get_coord(self, cf_name, loc="any", single=True, errors="ignore"):
        """Search for a coordinate knowing its generic CF name

        Search is made with the :meth:`~xoa.cf.CFSpecs.search_coord` method of the
        :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Parameters
        ----------
        cf_name: str
            Generic CF name
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or "": no location
        {errors}

        See also
        --------
        :meth:`xoa.cf.CFSpecs.search`
        :meth:`xoa.cf.CFCoordSpecs.search`
        :meth:`xoa.cf.CFVarSpecs.search`
        """
        return self._cfspecs.search_coord(
            self._obj, cf_name=cf_name, loc=loc, get="obj", single=single, errors="ignore"
        )

    def __getattr__(self, cf_name):
        """Shortcut to :meth:`get`"""
        return self.get(cf_name, errors="ignore")

    def __getitem__(self, cf_name):
        """Shortcut to :meth:`get`"""
        return self.get(cf_name, errors="ignore")

    def auto_format(self, **kwargs):
        """Rename variables and coordinates and fill their attributes

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.cf.CFSpecs.auto_format`
        :ref:`uses.cf`
        """
        return self._cfspecs.auto_format(self._obj, **kwargs)

    def fill_attrs(self, **kwargs):
        """Fill missing attributes

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.cf.CFSpecs.fill_attrs`
        :ref:`uses.cf`
        """
        return self._cfspecs.fill_attrs(self._obj, **kwargs)

    def to_loc(self, **locs):
        """Set the staggered grid location for specified names

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Parameters
        ----------
        locs: dict
            **Keys are root names**, values are new locations.
            A value of `False`, remove the location.
            A value of `None` left it as is.

        See also
        --------
        :meth:`xoa.cf.CFSpecs.to_loc`
        reloc
        """
        return self._cfspecs.to_loc(self._obj, **locs)

    def reloc(self, **locs):
        """Convert given staggered grid locations to other locations

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Parameters
        ----------
        obj: xarray.Dataset, xarray.DataArray
        locs: dict
            **Keys are locations**, values are new locations.
            A value of `False` or `None`, remove the location.

        See also
        --------
        :meth:`xoa.cf.CFSpecs.reloc`
        to_loc
        """
        return self._cfspecs.reloc(self._obj, **locs)

    def infer_coords(self, **kwargs):
        """Infer coordinates and set them as coordinates

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.cf.CFSpecs.infer_coords`
        :ref:`uses.cf`
        """
        return self.cfspecs.infer_coords(self._obj, **kwargs)

    def decode(self, **kwargs):
        """Rename variables and coordinates to generic names

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.cf.CFSpecs.decode`
        :meth:`xoa.cf.CFSpecs.encode`
        :ref:`uses.cf`
        """
        return self.cfspecs.decode(self._obj, **kwargs)

    def encode(self, **kwargs):
        """Rename variables and coordinates to specialized names

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.cf.CFSpecs.encode`
        :meth:`xoa.cf.CFSpecs.decode`
        :ref:`uses.cf`
        """
        return self.cfspecs.encode(self._obj, **kwargs)

    @ERRORS.format_method_docstring
    def get_depth(self, errors="ignore"):
        """Get the depth as computed or recognized by the :meth:`~xoa.cf.CFSpecs`

        If a depth variable cannot be found, it tries to compute either
        from sigma-like coordinates or from layer thinknesses.

        Parameters
        ----------
        {errors}

        Return
        ------
        xarray.DataArray, None

        See also
        --------
        :func:`xoa.coords.get_depth`
        :func:`xoa.grid.decode_cf_dz2depth`
        :func:`xoa.sigma.decode_cf_sigma`
        :ref:`uses.cf`
        """
        from .coords import get_depth

        return get_depth(self._obj, errors=errors)

    @property
    def coords(self):
        """Sub-accessor for coords only"""
        if self._coords is None:
            self._coords = _CoordAccessor_(self._obj, self._cfspecs)
        return self._coords

    @property
    def data_vars(self):
        """Sub-accessor for data_vars only"""
        if self._data_vars is None:
            self._data_vars = _DataVarAccessor__(self._obj, self._cfspecs)
        return self._data_vars


class _CoordAccessor_(_CFAccessor_):
    _search_category = 'coords'

    @property
    def dim(self):
        from .cf import XoaError

        try:
            return self._cfspecs.coords.search_dim(self._obj)[0]
        except XoaError:
            return

    def get_dim(self, cf_arg, loc="any"):
        """Get a dimension name knowing its type or generic CF name

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.

        Parameters
        ----------
        cf_arg: None, {{"x", "y", "z", "t", "f"}}
            Dimension type
        loc:
            Location

        Return
        ------
        str, None
            Dimension name or None of not found

        See also
        --------
        :meth:`xoa.cf.CFSpecs.search_dim`
        """
        cf_arg = cf_arg.lower()
        if not hasattr(self, '_dims'):
            self._dims = {}
            if cf_arg not in self._dims:
                self._dims[cf_arg] = self._cfspecs.coords.search_dim(self._obj, cf_arg, loc=loc)
        return self._dims[cf_arg]

    @property
    def xdim(self):
        """X dimension

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.cf.CFSpecs.search_dim`
        """
        return self.get_dim("x")

    @property
    def ydim(self):
        """Y dimension name

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.cf.CFSpecs.search_dim`
        """
        return self.get_dim("y")

    @property
    def zdim(self):
        """Z dimension name

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.cf.CFSpecs.search_dim`
        """
        return self.get_dim("z")

    @property
    def tdim(self):
        """T (time) dimension name

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.cf.CFSpecs.search_dim`
        """
        return self.get_dim("t")

    @property
    def fdim(self):
        """F (forecast) dimension name

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.cf.CFSpecs.search_dim`
        """
        return self.get_dim("f")


class _DataVarAccessor__(_CFAccessor_):
    _search_category = "data_vars"


class CFDatasetAccessor(_CFAccessor_):
    @property
    def ds(self):
        return self._obj


class CFDataArrayAccessor(_CoordAccessor_):
    @property
    def da(self):
        return self._obj

    @property
    def cf_name(self):
        """Get the generic name that matches this array

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.

        See also
        --------
        :meth:`xoa.cf.CFSpecs.match`
        """
        if 'name' not in self._cache:
            category, name = self._cfspecs.match(self._obj)
            self._cache["category"] = category
            self._cache["name"] = name
        return self._cache["name"]

    name = cf_name

    @property
    def attrs(self):
        """Get the generic attributes that matches this array

        This makes use of the :meth:`~xoa.cf.CFSpecs` instance was set at initialisation or
        inferred with :func:`xoa.cf.infer_cf_specs`.
        If no specialized name is declared in the specs, generic names are used.

        See also
        --------
        :meth:`xoa.cf.CFSpecs.get_attrs`
        """
        if "attrs" not in self._cache:
            if self.name:
                cf_attrs = self._cfspecs[self._cache["category"]].get_attrs(
                    self._cache["name"], multi=True
                )
                self._cache["attrs"] = self._cfspecs.sglocator.patch_attrs(
                    self._obj.attrs, cf_attrs
                )
            else:
                self._cache["attrs"] = {}
        return self._cache["attrs"]

    def __getattr__(self, attr):
        if self.name and self.attrs and attr in self.attrs:
            return self._cache["attrs"][attr]
        return _CoordAccessor_.__getattr__(self, attr)


class SigmaAccessor(_BasicCFAccessor_):
    """Dataset accessor to compute depths from sigma-like coordinates

    This follows the CF cnventions.

    Example
    -------
    >>> ds = xr.open_dataset('croco.nc')
    >>> ds = ds.decode_sigma()
    """

    def __init__(self, ds, cfspecs=None):
        assert hasattr(ds, "data_vars"), "ds must be a xarray.Dataset"
        _BasicCFAccessor_.__init__(self, ds, cfspecs)
        self._ds = self._obj

    @ERRORS.format_method_docstring
    def decode(self, rename=False, errors="raise"):
        """Compute depth from sigma coordinates

        Parameters
        ----------
        rename: bool
            Rename and format arrays ot make them compliant with
            :mod:`xoa.cf`
        {errors}

        Return
        ------
        xarray.Dataset

        See also
        --------
        :func:`xoa.sigma.decode_cf_sigma`
        """
        from .sigma import decode_cf_sigma

        return decode_cf_sigma(self._ds, rename=rename, errors=errors)

    def __call__(self):
        """Shortcut to :meth:`decode`"""
        return self.decode()

    def get_sigma_terms(self, loc=None, rename=False):
        """Call :func:`get_sigma_terms` on the dataset


        It operates like this:

        1. Search for the sigma variables.
        2. Parse their ``formula_terms`` attribute.
        3. Create a dict for each locations from names in datasets to
           :mod:`xoa.cf` compliant names that are also used in conversion
           functions.

        Parameters
        ----------
        ds: xarray.Dataset
        loc: str, {"any", None}
            Staggered grid location.
            If any or None, results for all locations are returned.

        Returns
        -------
        dict, dict of dict
            A dict is generated for a given sigma variable,
            whose keys are array names, like ``"sc_r"``,
            and values are :mod:`~xoa.cf` names, like ``"sig"``.
            A special key is the ``"type"`` whose corresponding value
            is the ``standard_name``, stripped from its potential staggered grid
            location indicator.
            If ``loc`` is ``"any"`` or ``None``,
            each dict is embedded in a master dict
            whose keys are staggered grid location. If no location is found,
            the key is set ``None``.

        Raises
        ------
        xoa.sigma.XoaSigmaError
            In case of:

            - inconsistent staggered grid location in dataarrays
              as checked by :meth:`xoa.cf.SGLocator.get_location`
            - no standard_name in sigma/s variable
            - a malformed formula
            - a formula term variable that is not found in the dataset
            - an unknown formula term name

        See also
        --------
        :func:`xoa.sigma.get_sigma_terms`
        """
        from .sigma import get_sigma_terms

        return get_sigma_terms(self._ds, loc=loc, rename=rename)


class XoaDataArrayAccessor(CFDataArrayAccessor):
    @property
    def cf(self):
        """The :class:`CFDataArrayAccessor` subaccessor"""
        if not hasattr(self, "_cf"):
            self._cf = CFDataArrayAccessor(self._ds, self._cfspecs)
        return self._cf


class XoaDatasetAccessor(CFDatasetAccessor):
    @property
    def cf(self):
        """The :class:`~xoa.accessors.CFDatasetAccessor` subaccessor"""
        if not hasattr(self, "_cf"):
            self._cf = CFDatasetAccessor(self._ds, self._cfspecs)
        return self._cf

    @property
    def decode_sigma(self):
        """The :class:`~xoa.accessors.SigmaAccessor` subaccessor for sigma coordinates"""
        if not hasattr(self, "_sigma"):
            self._sigma = SigmaAccessor(self._ds)
        return self._sigma


def _register_xarray_accessors_(dataarrays=None, datasets=None):
    """Silently register xarray accessors"""
    import xarray as xr

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xr.core.extensions.AccessorRegistrationWarning)
        if dataarrays:
            for name, cls in dataarrays.items():
                xr.register_dataarray_accessor(name)(cls)
        if datasets:
            for name, cls in datasets.items():
                xr.register_dataset_accessor(name)(cls)


def register_cf_accessors(name='xcf'):
    """Register the cf accessors"""
    _register_xarray_accessors_(
        dataarrays={name: CFDataArrayAccessor},
        datasets={name: CFDatasetAccessor},
    )


def register_sigma_accessor(name='decode_sigma'):
    """Register the sigma decoding accessor"""
    _register_xarray_accessors_(datasets={name: SigmaAccessor})


def register_xoa_accessors(name='xoa'):
    """Register the main xoa accessors"""
    _register_xarray_accessors_(
        dataarrays={name: XoaDataArrayAccessor},
        datasets={name: XoaDatasetAccessor},
    )
