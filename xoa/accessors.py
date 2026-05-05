#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xarray and pandas xoa accessors

"""
# Copyright 2020-2026 Shom

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

from . import exceptions
from .misc import ERRORS


class _BasicMetaAccessor_(object):
    """Base class for meta-aware xarray accessors"""

    def __init__(self, obj, meta_specs=None):
        from . import meta

        if meta_specs is None:
            meta_specs = meta.infer_meta_specs(obj)
        self._obj = obj
        self.set_meta_specs(meta_specs)

    def _assign_meta_specs_(self):
        from . import meta

        if self._meta_specs.name:
            self._obj = meta.assign_meta_specs(self._obj, self._meta_specs, register=True)

    def set_meta_specs(self, meta_specs):
        """Set the internal :class:`~xoa.meta.MetaSpecs` used by this accessor

        If the specs object has a :metasec:`registration name <register>`,
        it is assigned to
        the current object and its children with function
        :func:`~xoa.meta.assign_meta_specs`. This set the "meta_specs" encoding
        to the name.

        Parameters
        ----------
        meta_specs: xoa.meta.MetaSpecs

        See also
        --------
        :ref:`indepth.meta`
        """
        from .meta.general import MetaSpecs

        assert isinstance(meta_specs, MetaSpecs)
        self._meta_specs = meta_specs
        self._assign_meta_specs_()

    def get_meta_specs(self):
        """Get the internal :class:`~xoa.meta.MetaSpecs` instance used by this accessor

        If not provided at the initialization, it is inferred with :func:`xoa.meta.infer_meta_specs`.

        Return
        ------
        xoa.meta.MetaSpecs

        See also
        --------
        :ref:`indepth.meta`
        """
        return self._meta_specs

    meta_specs = property(fget=get_meta_specs, fset=set_meta_specs, doc="The MetaSpecs instance")

    # Backward compatibility
    def set_cf_specs(self, cfspecs):
        """Deprecated: use set_meta_specs instead"""
        warnings.warn(
            "set_cf_specs is deprecated. Use set_meta_specs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.set_meta_specs(cfspecs)

    def get_cf_specs(self):
        """Deprecated: use get_meta_specs instead"""
        warnings.warn(
            "get_cf_specs is deprecated. Use get_meta_specs instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_meta_specs()

    cfspecs = property(
        fget=get_cf_specs, fset=set_cf_specs, doc="Deprecated: use meta_specs instead"
    )


class _MetaAccessor_(_BasicMetaAccessor_):
    """Meta accessor with search and formatting capabilities"""

    _search_category = None

    def __init__(self, obj, meta_specs=None):
        _BasicMetaAccessor_.__init__(self, obj, meta_specs)
        self._coords = None
        self._data_vars = None
        self._cache = {}
        self._obj = self.infer_coords()
        self._assign_meta_specs_()

    @ERRORS.format_method_docstring
    def get(self, meta_name, loc="any", single=True, errors="ignore"):
        """Search for a data array or coordinate knowing its generic CF name

        Search is made with the :meth:`~xoa.meta.MetaSpecs.search` method of the
        :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

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
        :meth:`xoa.meta.MetaSpecs.search`
        :meth:`xoa.meta.MetaCoordSpecs.search`
        :meth:`xoa.meta.MetaVarSpecs.search`
        """
        kwargs = dict(loc=loc, get="obj", single=single, errors=errors)
        if self._search_category is None:
            return self._meta_specs.search(self._obj, meta_name, **kwargs)
        return self._meta_specs[self._search_category].search(self._obj, meta_name, **kwargs)

    @ERRORS.format_method_docstring
    def get_coord(self, meta_name, loc="any", single=True, errors="ignore"):
        """Search for a coordinate knowing its generic CF name

        Search is made with the :meth:`~xoa.meta.MetaSpecs.search_coord` method of the
        :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

        Parameters
        ----------
        meta_name: str
            Generic CF name
        loc: str, {{"any", None}}, {{"", False}}
            - str: one of these locations
            - None or "any": any
            - False or "": no location
        {errors}

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.search`
        :meth:`xoa.meta.MetaCoordSpecs.search`
        :meth:`xoa.meta.MetaVarSpecs.search`
        """
        return self._meta_specs.search_coord(
            self._obj, meta_name, loc=loc, get="obj", single=single, errors="ignore"
        )

    def __getattr__(self, meta_name):
        """Shortcut to :meth:`get`"""
        return self.get(meta_name, errors="ignore")

    def __getitem__(self, meta_name):
        """Shortcut to :meth:`get`"""
        return self.get(meta_name, errors="ignore")

    def auto_format(self, **kwargs):
        """Rename variables and coordinates and fill their attributes

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.auto_format`
        :ref:`indepth.meta`
        """
        return self._meta_specs.auto_format(self._obj, **kwargs)

    def fill_attrs(self, **kwargs):
        """Fill missing attributes

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.fill_attrs`
        :ref:`indepth.meta`
        """
        return self._meta_specs.fill_attrs(self._obj, **kwargs)

    def to_loc(self, **locs):
        """Set the staggered grid location for specified names

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

        Parameters
        ----------
        locs: dict
            **Keys are root names**, values are new locations.
            A value of `False`, remove the location.
            A value of `None` left it as is.

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.to_loc`
        reloc
        """
        return self._meta_specs.to_loc(self._obj, **locs)

    def reloc(self, **locs):
        """Convert given staggered grid locations to other locations

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

        Parameters
        ----------
        obj: xarray.Dataset, xarray.DataArray
        locs: dict
            **Keys are locations**, values are new locations.
            A value of `False` or `None`, remove the location.

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.reloc`
        to_loc
        """
        return self._meta_specs.reloc(self._obj, **locs)

    def infer_coords(self, **kwargs):
        """Infer coordinates and set them as coordinates

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.infer_coords`
        :ref:`indepth.meta`
        """
        return self.meta_specs.infer_coords(self._obj, **kwargs)

    def decode(self, **kwargs):
        """Rename variables and coordinates to generic names

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.decode`
        :meth:`xoa.meta.MetaSpecs.encode`
        :ref:`indepth.meta`
        """
        return self.meta_specs.decode(self._obj, **kwargs)

    def encode(self, **kwargs):
        """Rename variables and coordinates to specialized names

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.

        Return
        ------
        xarray.Dataset, xarray.DataArray

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.encode`
        :meth:`xoa.meta.MetaSpecs.decode`
        :ref:`indepth.meta`
        """
        return self.meta_specs.encode(self._obj, **kwargs)

    @ERRORS.format_method_docstring
    def get_depth(self, errors="ignore"):
        """Get the depth as computed or recognized by the :meth:`~xoa.meta.MetaSpecs`

        If a depth variable cannot be found, it tries to compute either
        from sigma-like coordinates or from layer thicknesses.

        Parameters
        ----------
        {errors}

        Return
        ------
        xarray.DataArray, None

        See also
        --------
        :func:`xoa.coords.get_depth`
        :func:`xoa.grid.decode_dz2depth`
        :func:`xoa.sigma.decode_sigma`
        :ref:`indepth.meta`
        """
        from .coords import get_depth

        return get_depth(self._obj, errors=errors)

    @property
    def coords(self):
        """Sub-accessor for coords only"""
        if self._coords is None:
            self._coords = _MetaCoordAccessor_(self._obj, self._meta_specs)
        return self._coords

    @property
    def data_vars(self):
        """Sub-accessor for data_vars only"""
        if self._data_vars is None:
            self._data_vars = _MetaDataVarAccessor_(self._obj, self._meta_specs)
        return self._data_vars


class _MetaCoordAccessor_(_MetaAccessor_):
    """Meta accessor specialized for coordinates"""

    _search_category = 'coords'

    @property
    def dim(self):
        from .exceptions import XoaError

        try:
            return self._meta_specs.coords.search_dim(self._obj)[0]
        except XoaError:
            return

    def get_dim(self, cf_arg, loc="any"):
        """Get a dimension name knowing its type or generic CF name

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
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
        :meth:`xoa.meta.MetaSpecs.search_dim`
        """
        cf_arg = cf_arg.lower()
        if not hasattr(self, '_dims'):
            self._dims = {}
            if cf_arg not in self._dims:
                self._dims[cf_arg] = self._meta_specs.coords.search_dim(self._obj, cf_arg, loc=loc)
        return self._dims[cf_arg]

    @property
    def xdim(self):
        """X dimension

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.meta.MetaSpecs.search_dim`
        """
        return self.get_dim("x")

    @property
    def ydim(self):
        """Y dimension name

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.meta.MetaSpecs.search_dim`
        """
        return self.get_dim("y")

    @property
    def zdim(self):
        """Z dimension name

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.meta.MetaSpecs.search_dim`
        """
        return self.get_dim("z")

    @property
    def tdim(self):
        """T (time) dimension name

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.meta.MetaSpecs.search_dim`
        """
        return self.get_dim("t")

    @property
    def fdim(self):
        """F (forecast) dimension name

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.


        See also
        -------
        :meth:`xoa.meta.MetaSpecs.search_dim`
        """
        return self.get_dim("f")


class _MetaDataVarAccessor_(_MetaAccessor_):
    """Meta accessor specialized for data variables"""

    _search_category = "data_vars"


class MetaDatasetAccessor(_MetaAccessor_):
    """Meta accessor registered on :class:`xarray.Dataset`"""

    @property
    def ds(self):
        return self._obj


class MetaDataArrayAccessor(_MetaCoordAccessor_):
    """Meta accessor registered on :class:`xarray.DataArray`"""

    @property
    def da(self):
        return self._obj

    @property
    def meta_name(self):
        """Get the generic name that matches this array

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.match`
        """
        if 'name' not in self._cache:
            category, name = self._meta_specs.match(self._obj)
            self._cache["category"] = category
            self._cache["name"] = name
        return self._cache["name"]

    name = meta_name
    cf_name = meta_name

    @property
    def attrs(self):
        """Get the generic attributes that matches this array

        This makes use of the :meth:`~xoa.meta.MetaSpecs` instance was set at initialisation or
        inferred with :func:`xoa.meta.infer_meta_specs`.
        If no specialized name is declared in the specs, generic names are used.

        See also
        --------
        :meth:`xoa.meta.MetaSpecs.get_attrs`
        """
        if "attrs" not in self._cache:
            if self.name:
                cf_attrs = self._meta_specs[self._cache["category"]].get_attrs(
                    self._cache["name"], multi=True
                )
                self._cache["attrs"] = self._meta_specs.sglocator.patch_attrs(
                    self._obj.attrs, cf_attrs
                )
            else:
                self._cache["attrs"] = {}
        return self._cache["attrs"]

    def __getattr__(self, attr):
        if self.name and self.attrs and attr in self.attrs:
            return self._cache["attrs"][attr]
        return _MetaCoordAccessor_.__getattr__(self, attr)


class SigmaAccessor(_BasicMetaAccessor_):
    """Dataset accessor to compute depths from sigma-like coordinates

    This follows the CF conventions.

    Example
    -------
    >>> ds = xr.open_dataset('croco.nc')
    >>> ds = ds.decode_sigma()
    """

    def __init__(self, ds, meta_specs=None):
        assert hasattr(ds, "data_vars"), "ds must be a xarray.Dataset"
        _BasicMetaAccessor_.__init__(self, ds, meta_specs)
        self._ds = self._obj

    @ERRORS.format_method_docstring
    def decode(self, rename=False, errors="raise"):
        """Compute depth from sigma coordinates

        Parameters
        ----------
        rename: bool
            Rename and format arrays to make them compliant with
            :mod:`xoa.meta`
        {errors}

        Return
        ------
        xarray.Dataset

        See also
        --------
        :func:`xoa.sigma.decode_sigma`
        """
        from .sigma import decode_sigma

        return decode_sigma(self._ds, rename=rename, errors=errors)

    def __call__(self):
        """Shortcut to :meth:`decode`"""
        return self.decode()

    def get_sigma_terms(self, loc=None, rename=False):
        """Call :func:`get_sigma_terms` on the dataset


        It operates like this:

        1. Search for the sigma variables.
        2. Parse their ``formula_terms`` attribute.
        3. Create a dict for each locations from names in datasets to
           :mod:`xoa.meta` compliant names that are also used in conversion
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
            and values are :mod:`~xoa.meta` names, like ``"sig"``.
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
              as checked by :meth:`xoa.meta.SGLocator.get_location`
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


class XoaDataArrayAccessor(MetaDataArrayAccessor):
    """Main xoa accessor registered on :class:`xarray.DataArray`"""

    @property
    def meta(self):
        """The :class:`MetaDataArrayAccessor` subaccessor

        Since XoaDataArrayAccessor inherits from MetaDataArrayAccessor,
        this property simply returns self to provide the meta functionality.
        """
        return self

    @property
    def cf(self):
        """The :class:`MetaDataArrayAccessor` subaccessor

        .. deprecated::
            Use :attr:`meta` instead.
        """
        warnings.warn(
            "The 'cf' accessor is deprecated. Use 'meta' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.meta


class XoaDatasetAccessor(MetaDatasetAccessor):
    """Main xoa accessor registered on :class:`xarray.Dataset`"""

    @property
    def meta(self):
        """The :class:`~xoa.accessors.MetaDatasetAccessor` subaccessor

        Since XoaDatasetAccessor inherits from MetaDatasetAccessor,
        this property simply returns self to provide the meta functionality.
        """
        return self

    @property
    def cf(self):
        """The :class:`~xoa.accessors.MetaDatasetAccessor` subaccessor

        .. deprecated::
            Use :attr:`meta` instead.
        """
        warnings.warn(
            "The 'cf' accessor is deprecated. Use 'meta' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.meta

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


def register_meta_accessors(name='meta'):
    """Register the meta accessors"""
    _register_xarray_accessors_(
        dataarrays={name: MetaDataArrayAccessor},
        datasets={name: MetaDatasetAccessor},
    )


def register_cf_accessors(name='xcf'):
    """Register the cf accessors

    .. deprecated::
        Use :func:`register_meta_accessors` instead.
    """
    import warnings
    from . import XoaDeprecationWarning

    warnings.warn(
        "register_cf_accessors is deprecated. Use register_meta_accessors instead.",
        XoaDeprecationWarning,
        stacklevel=2,
    )
    _register_xarray_accessors_(
        dataarrays={name: MetaDataArrayAccessor},
        datasets={name: MetaDatasetAccessor},
    )


# Backward compatibility aliases
CFDatasetAccessor = MetaDatasetAccessor
CFDataArrayAccessor = MetaDataArrayAccessor


def register_sigma_accessor(name='decode_sigma'):
    """Register the sigma decoding accessor"""
    _register_xarray_accessors_(datasets={name: SigmaAccessor})


def register_xoa_accessors(name='xoa'):
    """Register the main xoa accessors"""
    _register_xarray_accessors_(
        dataarrays={name: XoaDataArrayAccessor},
        datasets={name: XoaDatasetAccessor},
    )


def register_accessors(xoa=True, xcf=False, meta=True, decode_sigma=True):
    """Register xarray accessors

    Parameters
    ----------
    xoa: bool, str
        Register the main accessors with
        :func:`~xoa.meta.register_xoa_accessors`.
    xcf: bool, str
        Register the :mod:`xoa.meta` module accessors with
        :func:`~xoa.meta.register_meta_accessors` using the deprecated name "xcf".

        .. deprecated::
            Use ``meta`` instead.
    meta: bool, str
        Register the :mod:`xoa.meta` module accessors with
        :func:`~xoa.meta.register_meta_accessors`.
    decode_sigma: bool, str
        Register the :mod:`xoa.sigma` module accessor with
        :func:`~xoa.meta.register_sigma_accessor`.

    See also
    --------
    xoa.accessors
    """
    if xoa:
        from .accessors import register_xoa_accessors

        kw = {"name": xoa} if isinstance(xoa, str) else {}
        register_xoa_accessors(**kw)
    if xcf:
        from .accessors import register_meta_accessors

        exceptions.xoa_warn(
            "The 'xcf' parameter is deprecated. Use 'meta' instead.",
            category="deprecation",
            stacklevel=2,
        )
        kw = {"name": xcf} if isinstance(xcf, str) else {}
        # Register with "xcf" name for backward compatibility
        if not kw:
            kw = {"name": "xcf"}
        register_meta_accessors(**kw)
    if meta:
        from .accessors import register_meta_accessors

        kw = {"name": meta} if isinstance(meta, str) else {}
        register_meta_accessors(**kw)
    if decode_sigma:
        from .accessors import register_sigma_accessor

        kw = {"name": decode_sigma} if isinstance(decode_sigma, str) else {}
        register_sigma_accessor(**kw)
