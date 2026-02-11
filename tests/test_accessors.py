#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.accessors` module
"""

import warnings
import pytest
import numpy as np
import xarray as xr

import xoa
from xoa import accessors


class TestAccessorRegistration:
    """Tests for accessor registration functions"""

    def test_register_meta_accessors(self):
        """Test that register_meta_accessors creates meta accessor"""
        accessors.register_meta_accessors(name='meta')

        # Create a simple dataset
        ds = xr.Dataset({
            'temperature': (['x', 'y'], np.random.rand(3, 4)),
        })

        # Check that meta accessor is available
        assert hasattr(ds, 'meta')

    def test_register_cf_accessors_deprecated(self):
        """Test that register_cf_accessors issues deprecation warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            accessors.register_cf_accessors(name='xcf_test')

            # Check that a deprecation warning was issued
            assert len(w) == 1
            assert issubclass(w[0].category, (DeprecationWarning, xoa.XoaDeprecationWarning))
            assert "deprecated" in str(w[0].message).lower()

    def test_register_meta_accessors_custom_name(self):
        """Test registering meta accessors with custom name"""
        accessors.register_meta_accessors(name='mymeta')

        ds = xr.Dataset({
            'temp': (['x'], np.array([1, 2, 3])),
        })

        # Check that custom name accessor is available
        assert hasattr(ds, 'mymeta')


class TestBackwardCompatibility:
    """Tests for backward compatibility aliases"""

    def test_cf_dataset_accessor_alias(self):
        """Test that CFDatasetAccessor is an alias for MetaDatasetAccessor"""
        assert accessors.CFDatasetAccessor is accessors.MetaDatasetAccessor

    def test_cf_dataarray_accessor_alias(self):
        """Test that CFDataArrayAccessor is an alias for MetaDataArrayAccessor"""
        assert accessors.CFDataArrayAccessor is accessors.MetaDataArrayAccessor


class TestMetaAccessorMethods:
    """Tests for MetaAccessor methods"""

    def setup_method(self):
        """Set up test fixtures"""
        # Register accessors
        accessors.register_meta_accessors(name='meta')

        # Create test dataset with CF-like names
        self.ds = xr.Dataset({
            'SST': (['time', 'lat', 'lon'], np.random.rand(2, 3, 4), {
                'standard_name': 'sea_surface_temperature',
                'long_name': 'Sea Surface Temperature',
                'units': 'degC'
            }),
        }, coords={
            'lon': (['lon'], np.linspace(-180, 180, 4), {
                'standard_name': 'longitude',
                'units': 'degrees_east'
            }),
            'lat': (['lat'], np.linspace(-90, 90, 3), {
                'standard_name': 'latitude',
                'units': 'degrees_north'
            }),
            'time': (['time'], np.array([0, 1])),
        })

    def test_metaspecs_property(self):
        """Test that meta_specs property works"""
        assert hasattr(self.ds.meta, 'meta_specs')
        assert self.ds.meta.meta_specs is not None

    def test_set_meta_specs(self):
        """Test setting meta specs"""
        from xoa import meta

        new_specs = meta.get_meta_specs()
        self.ds.meta.set_meta_specs(new_specs)
        assert self.ds.meta.get_meta_specs() is new_specs

    def test_get_meta_specs(self):
        """Test getting meta specs"""
        specs = self.ds.meta.get_meta_specs()
        assert specs is not None

    def test_deprecated_cfspecs_property(self):
        """Test that cfspecs property issues deprecation warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = self.ds.meta.cfspecs

            # Check warning was issued
            assert len(w) >= 1
            assert any(issubclass(warn.category, DeprecationWarning) for warn in w)

    def test_deprecated_set_cf_specs(self):
        """Test that set_cf_specs issues deprecation warning"""
        from xoa import meta

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            specs = meta.get_meta_specs()
            self.ds.meta.set_cf_specs(specs)

            # Check warning was issued
            assert len(w) >= 1
            assert any(issubclass(warn.category, DeprecationWarning) for warn in w)

    def test_deprecated_get_cf_specs(self):
        """Test that get_cf_specs issues deprecation warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = self.ds.meta.get_cf_specs()

            # Check warning was issued
            assert len(w) >= 1
            assert any(issubclass(warn.category, DeprecationWarning) for warn in w)


class TestXoaAccessorSubaccessors:
    """Tests for xoa accessor subaccessors"""

    def setup_method(self):
        """Set up test fixtures"""
        # Register accessors
        xoa.register_accessors(xoa=True, meta=True)

        self.ds = xr.Dataset({
            'temp': (['x'], np.array([20, 21, 22])),
        })

    def test_xoa_meta_subaccessor(self):
        """Test that .xoa.meta subaccessor works"""
        assert hasattr(self.ds.xoa, 'meta')
        # The meta property should work - it may create a new accessor
        meta_accessor = self.ds.xoa.meta
        # Since the return is based on the object state, just check it's accessible
        assert meta_accessor is not None or hasattr(self.ds.xoa, '_meta')

    def test_xoa_cf_subaccessor_deprecated(self):
        """Test that .xoa.cf subaccessor issues deprecation warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = self.ds.xoa.cf

            # Check warning was issued
            assert len(w) >= 1
            assert any(issubclass(warn.category, DeprecationWarning) for warn in w)
            assert any("deprecated" in str(warn.message).lower() for warn in w)

    def test_xoa_cf_redirects_to_meta(self):
        """Test that .xoa.cf redirects to .xoa.meta"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # They should be the same object
            assert self.ds.xoa.cf is self.ds.xoa.meta


class TestRegisterAccessorsFunction:
    """Tests for the main register_accessors function in xoa.__init__"""

    def test_register_with_meta(self):
        """Test register_accessors with meta=True"""
        xoa.register_accessors(xoa=False, meta=True)

        ds = xr.Dataset({'data': (['x'], [1, 2, 3])})
        assert hasattr(ds, 'meta')

    def test_register_with_xcf_deprecated(self):
        """Test register_accessors with xcf=True issues warning"""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            xoa.register_accessors(xoa=False, xcf=True)

            # Check warning was issued
            assert len(w) >= 1
            assert any(issubclass(warn.category, (DeprecationWarning, xoa.XoaDeprecationWarning))
                      for warn in w)

    def test_register_both_meta_and_xcf(self):
        """Test that both meta and xcf can be registered (for compatibility)"""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xoa.register_accessors(xoa=False, meta=True, xcf=True)

        ds = xr.Dataset({'data': (['x'], [1, 2, 3])})
        assert hasattr(ds, 'meta')
        assert hasattr(ds, 'xcf')


class TestAccessorClassHierarchy:
    """Tests for accessor class hierarchy"""

    def test_meta_dataset_accessor_base(self):
        """Test that MetaDatasetAccessor inherits from correct base"""
        assert issubclass(accessors.MetaDatasetAccessor, accessors._MetaAccessor_)

    def test_meta_dataarray_accessor_base(self):
        """Test that MetaDataArrayAccessor inherits from correct base"""
        assert issubclass(accessors.MetaDataArrayAccessor, accessors._MetaCoordAccessor_)

    def test_xoa_dataset_accessor_base(self):
        """Test that XoaDatasetAccessor inherits from MetaDatasetAccessor"""
        assert issubclass(accessors.XoaDatasetAccessor, accessors.MetaDatasetAccessor)

    def test_xoa_dataarray_accessor_base(self):
        """Test that XoaDataArrayAccessor inherits from MetaDataArrayAccessor"""
        assert issubclass(accessors.XoaDataArrayAccessor, accessors.MetaDataArrayAccessor)
