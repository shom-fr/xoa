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


class _CFAccessor_(object):
    _search_category = None

    def __init__(self, dsa):
        from .cf import get_cf_specs
        self._cfspecs = get_cf_specs()
        self._dsa = dsa
        self._coords = None
        self._data_vars = None
        self._cache = {}

    def set_cf_specs(self, cfspecs):
        """Set the :class:`CFSpecs` using by this accessor"""
        from .cf import CFSpecs
        assert isinstance(cfspecs, CFSpecs)
        self._cfspecs = cfspecs

    def get(self, name, loc="any", single=True, errors="ignore"):
        """Search for a CF item with :meth:`CFSpecs.search`"""
        kwargs = dict(name=name, loc=loc, get="obj", single=single,
                      errors=errors)
        if self._search_category is None:
            return self._cfspecs.search(self._dsa, **kwargs)
        return self._cfspecs[self._search_category].search(self._dsa, **kwargs)

    def get_coord(self, name, loc="any", single=True):
        """Search for a CF coord with :meth:`CFCoordSpecs.search`"""
        return self._cfspecs.coords.search(
            self._dsa, name=name, loc=loc, get="obj", single=single,
            errors="ignore")

    def __getattr__(self, name):
        return self.get(name, errors="warn")

    def __getitem__(self, name):
        return self.get(name, errors="warn")

    def auto_format(self, loc=None, standardize=True):
        """Auto-format attributes with :meth:`CFSpecs.auto_format`"""
        return self._cfspecs.auto_format(self._dsa, loc=loc,
                                         standardize=standardize)

    __call__ = auto_format

    def fill_attrs(self, loc=None, standardize=True):
        """Fill attributes with :meth:`CFSpecs.fill_attrs`"""
        return self._cfspecs.fill_attrs(self._dsa, loc=loc,
                                        standardize=standardize)

    @property
    def coords(self):
        """Sub-accessor for coords only"""
        if self._coords is None:
            self._coords = _CoordAccessor_(self._dsa)
            self._coords.set_cf_specs(self._cfspecs)
        return self._coords

    @property
    def data_vars(self):
        """Sub-accessor for data_vars only"""
        if self._data_vars is None:
            self._data_vars = _DataVarAccessor__(self._dsa)
            self._data_vars.set_cf_specs(self._cfspecs)
        return self._data_vars


class _CoordAccessor_(_CFAccessor_):
    _search_category = 'coords'

    @property
    def dim(self):
        from .cf import XoaError
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

    @property
    def fdim(self):
        return self.get_dim("f")


class _DataVarAccessor__(_CFAccessor_):
    _search_category = "data_vars"


class CFDatasetAccessor(_CFAccessor_):
    pass


class CFDataArrayAccessor(_CoordAccessor_):

    @property
    def name(self):
        if 'name' not in self._cache:
            category, name = self._cfspecs.match(self._dsa)
            self._cache["category"] = category
            self._cache["name"] = name
        return self._cache["name"]

    @property
    def attrs(self):
        if "attrs" not in self._cache:
            if self.name:
                cf_attrs = self._cfspecs[self._cache["category"]].get_attrs(
                    self._cache["name"], multi=True)
                self._cache["attrs"] = self._cfspecs.sglocator.patch_attrs(
                    self._dsa.attrs, cf_attrs)
            else:
                self._cache["attrs"] = {}
        return self._cache["attrs"]

    def __getattr__(self, attr):
        if self.name and self.attrs and attr in self.attrs:
            return self._cache["attrs"][attr]
        return _CoordAccessor_.__getattr__(self, attr)


class SigmaAccessor(object):
    """Dataset accessor to compute depths from sigma-like coordinates

    This follows the CF cnventions.

    Example
    -------
    >>> ds = xr.open_dataset('croco.nc')
    >>> ds = ds.decode_sigma()
    """

    def __init__(self, ds):
        self._ds = ds

    def decode(self, rename=False, errors="raise"):
        """Call :func:`decode_cf_sigma` on the dataset"""
        from .sigma import decode_cf_sigma
        return decode_cf_sigma(self._ds, rename=rename, errors=errors)

    def __call__(self):
        """Shortcut to :meth:`decode`"""
        return self.decode()

    def get_sigma_terms(self, loc=None, rename=False):
        """Call :func:`get_sigma_terms` on the dataset"""
        from .sigma import get_sigma_terms
        return get_sigma_terms(self._ds, loc=loc, rename=rename)


class XoaDataArrayAccessor(CFDataArrayAccessor):

    @property
    def cf(self):
        """CF subaccessor"""
        if not hasattr(self, "_cf"):
            self._cf = CFDataArrayAccessor(self._ds)
        return self._cf


class XoaDatasetAccessor(CFDatasetAccessor):

    @property
    def cf(self):
        """CF subaccessor"""
        if not hasattr(self, "_cf"):
            self._cf = CFDatasetAccessor(self._ds)
        return self._cf

    @property
    def decode_sigma(self):
        """Sigma coordinate subacessor"""
        if not hasattr(self, "_sigma"):
            self._sigma = SigmaAccessor(self._ds)
        return self._sigma


def _register_xarray_accessors_(dataarrays=None, datasets=None):
    """Silently register xarray accessors"""
    import xarray as xr
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            xr.core.extensions.AccessorRegistrationWarning)
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
