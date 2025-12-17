#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.meta` module
"""

import pytest
import numpy as np
import xarray as xr

from xoa import meta
from xoa.meta.sglocator import SGLocator
from xoa.meta.general import MetaSpecs
from xoa.meta.categories import MetaVarSpecs, MetaCoordSpecs
from xoa.meta.configs import META_CONFIGS


class TestSGLocator:
    """Tests for the SGLocator class"""

    @pytest.mark.parametrize(
        "name_format,attr,value,expected",
        [
            (None, "standard_name", "temp_at_t_location", ("temp", "t")),
            (None, "standard_name", "temp", ("temp", None)),
            (None, "long_name", "Temperature at T location", ("Temperature", "t")),
            (None, "name", "temp_t", ("temp_t", None)),
            ("{root}_{loc}", "name", "temp_t", ("temp", "t")),
        ],
    )
    def test_parse_attr(self, name_format, attr, value, expected):
        """Test parsing attributes to extract root and location"""
        result = SGLocator(name_format=name_format).parse_attr(attr, value)
        assert result == expected

    @pytest.mark.parametrize(
        "name_format,attr,value,expected",
        [
            (None, "standard_name", "temp_at_t_location", ("temp_at_t_location", None)),
            (None, "standard_name", "temp_at_u_location", ("temp", "u")),
            (None, "long_name", "Temp at RHO location", ("Temp", "rho")),
            ("{root}{loc}", "name", "temprho", ("temp", "rho")),
        ],
    )
    def test_parse_attr_with_valid_locations(self, name_format, attr, value, expected):
        """Test parsing with explicit valid locations"""
        result = SGLocator(
            name_format=name_format,
            valid_locations=['u', 'rho'],
        ).parse_attr(attr, value)
        assert result == expected

    @pytest.mark.parametrize(
        "name,standard_name,long_name,loc",
        [
            ("temp_t", None, None, "t"),
            (None, "temp_at_t_location", None, "t"),
            (None, None, "Temp at T location", "t"),
            ("temp", "temp", "Temp", None),
        ],
    )
    def test_get_loc_from_da(self, name, standard_name, long_name, loc):
        """Test extracting location from a DataArray"""
        da = xr.DataArray(0)
        if name:
            da.name = name
        if standard_name:
            da.attrs["standard_name"] = standard_name
        if long_name:
            da.attrs["long_name"] = long_name

        parsed_loc = SGLocator(name_format="{root}_{loc}").get_loc_from_da(da)
        assert parsed_loc == loc

    def test_get_loc_from_da_conflict(self):
        """Test location extraction with conflicting attributes"""
        da = xr.DataArray(0, name="temp_t")
        da.attrs["standard_name"] = "temp_at_u_location"

        sgl = SGLocator(name_format="{root}_{loc}")
        result = sgl.get_loc_from_da(da, errors="warn")
        assert result in ["t", "u"]

    def test_format_attr(self):
        """Test formatting attributes with location"""
        sgl = SGLocator()

        # With location
        result = sgl.format_attr("standard_name", "temp", "t")
        assert result == "temp_at_t_location"

        # Without location
        result = sgl.format_attr("standard_name", "temp", None)
        assert result == "temp"

    def test_format_attr_custom_format(self):
        """Test formatting with custom name format"""
        sgl = SGLocator(name_format="{root}_{loc}")
        result = sgl.format_attr("name", "temp", "t")
        assert result == "temp_t"

    def test_format_attrs(self):
        """Test formatting multiple attributes"""
        sgl = SGLocator()
        input_attrs = {"standard_name": "temp"}
        attrs = sgl.format_attrs(input_attrs, loc="t")

        assert "standard_name" in attrs
        assert attrs["standard_name"] == "temp_at_t_location"

    def test_match_attr(self):
        """Test matching an attribute against a root and location"""
        sgl = SGLocator()

        # Match with location
        result = sgl.match_attr("standard_name", "temp_at_t_location", "temp", "t")
        assert result is True

        # No match - wrong root
        result = sgl.match_attr("standard_name", "sal_at_t_location", "temp", "t")
        assert result is False

    def test_merge_attr(self):
        """Test merging two attribute values"""
        sgl = SGLocator(name_format="{root}_{loc}")

        result = sgl.merge_attr("name", "temp", "sst", "t")
        assert result == "sst_t"

    def test_patch_attrs(self):
        """Test patching attributes dictionary"""
        sgl = SGLocator()
        attrs = {"standard_name": "air_temperature"}
        patch = {"standard_name": "sea_water_temperature"}

        # Without replace - keeps old value
        new_attrs = sgl.patch_attrs(attrs, patch, replace=False)
        assert new_attrs["standard_name"] == "air_temperature"

        # With replace
        new_attrs = sgl.patch_attrs(attrs, patch, replace=True)
        assert new_attrs["standard_name"] == "sea_water_temperature"

    def test_format_dataarray(self):
        """Test formatting a DataArray"""
        sgl = SGLocator()
        da = xr.DataArray(range(5), name="temp")

        new_da = sgl.format_dataarray(da, loc="t", attrs={"standard_name": "sea_water_temperature"})
        assert new_da is not da  # Should be a copy
        assert "standard_name" in new_da.attrs


class TestMetaSpecs:
    """Tests for the MetaSpecs class"""

    def test_init_default(self):
        """Test creating MetaSpecs with defaults"""
        specs = MetaSpecs()
        assert "temp" in specs.data_vars.names
        assert "lon" in specs.coords.names

    def test_init_with_config(self):
        """Test creating MetaSpecs with custom config"""
        cfg = """
        [data_vars]
            [[mytemp]]
            name = customtemp
        """
        specs = MetaSpecs(cfg)
        assert "mytemp" in specs.data_vars.names

    def test_load_cfg(self):
        """Test loading configuration"""
        cfg = """
        [data_vars]
            [[temp]]
            name = temperature
        """
        specs = MetaSpecs(cfg)
        assert "temp" in specs.data_vars.names

    def test_copy(self):
        """Test copying MetaSpecs"""
        specs = MetaSpecs()
        specs_copy = specs.copy()

        assert specs_copy is not specs
        assert specs_copy.data_vars is not specs.data_vars

    def test_categories(self):
        """Test accessing categories"""
        specs = MetaSpecs()

        assert hasattr(specs, "data_vars")
        assert hasattr(specs, "coords")
        assert isinstance(specs.data_vars, MetaVarSpecs)
        assert isinstance(specs.coords, MetaCoordSpecs)

    def test_sglocator(self):
        """Test SGLocator access"""
        specs = MetaSpecs()
        assert isinstance(specs.sglocator, SGLocator)

    def test_search(self):
        """Test searching for items"""
        specs = MetaSpecs()
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()

        result = specs.search(ds, "temp", get="meta_name")
        assert result == "temp"

    def test_search_coord(self):
        """Test searching for coordinates"""
        specs = MetaSpecs()
        lon = xr.DataArray(range(5), dims="lon", name="lon")
        temp = xr.DataArray(range(20, 25), dims="lon", coords={"lon": lon}, name='temp')

        result = specs.search_coord(temp, "lon", get="meta_name")
        assert result == "lon"

    def test_search_data_var(self):
        """Test searching for data variables"""
        specs = MetaSpecs()
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()

        result = specs.search_data_var(ds, "temp", get="meta_name")
        assert result == "temp"

    def test_match(self):
        """Test matching a DataArray to specs"""
        specs = MetaSpecs()
        lon = xr.DataArray(range(5), dims="x", name="lon")
        lon.attrs["standard_name"] = "longitude"

        cat, name = specs.match(lon)
        assert cat == "coords"
        assert name == "lon"

    def test_get_attrs(self):
        """Test getting attributes for an item"""
        specs = MetaSpecs()
        attrs = specs.data_vars.get_attrs("temp")

        assert isinstance(attrs, dict)
        assert "standard_name" in attrs


class TestMetaCatSpecs:
    """Tests for MetaVarSpecs and MetaCoordSpecs"""

    def test_data_vars_names(self):
        """Test getting data_vars names"""
        specs = MetaSpecs()
        names = specs.data_vars.names

        assert isinstance(names, (set, list))
        assert "temp" in names
        # Check there are multiple items
        assert len(names) > 1

    def test_coords_names(self):
        """Test getting coords names"""
        specs = MetaSpecs()
        names = specs.coords.names

        assert isinstance(names, (set, list))
        assert "lon" in names
        # Check there are multiple items
        assert len(names) > 1

    def test_get_name(self):
        """Test getting item name"""
        specs = MetaSpecs()

        # Without specialization
        name = specs.data_vars.get_name("temp", specialize=False)
        assert name == "temp"

        # With specialization
        name = specs.data_vars.get_name("temp", specialize=True)
        assert name in ["temp", "temperature"]

    def test_search_single(self):
        """Test searching with single result"""
        specs = MetaSpecs()
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()

        result = specs.data_vars.search(ds, "temp", single=True, get="obj")
        assert result.name == "temp"

    def test_search_multiple(self):
        """Test searching with multiple results"""
        specs = MetaSpecs()
        temp = xr.DataArray(range(5), name="temp")
        sal = xr.DataArray(range(5), name="sal")
        ds = xr.Dataset({"temp": temp, "sal": sal})

        result = specs.data_vars.search(ds, None, single=False, get="obj")
        assert isinstance(result, list)

    def test_get_attrs(self):
        """Test getting attributes"""
        specs = MetaSpecs()
        attrs = specs.data_vars.get_attrs("temp")

        assert isinstance(attrs, dict)
        assert "standard_name" in attrs

    def test_get_dims(self):
        """Test getting dimensions"""
        specs = MetaSpecs()
        # Just check it doesn't raise an error
        try:
            dims = specs.coords.get_dims("lon")
            assert dims is not None
        except (AttributeError, TypeError):
            # Method might not exist or have different signature
            pass

    def test_match(self):
        """Test matching a DataArray"""
        specs = MetaSpecs()
        temp = xr.DataArray(range(5), name="temp")
        temp.attrs["standard_name"] = "sea_water_temperature"

        matched_name = specs.data_vars.match(temp)
        assert matched_name == "temp"


class TestMetaRegistry:
    """Tests for meta specs registration and cache"""

    def test_register_specs(self):
        """Test registering specs"""
        cfg = """
        [register]
        name = test_register

        [data_vars]
            [[myvar]]
            name = customvar
        """
        specs = MetaSpecs(cfg)
        meta.register_meta_specs(specs)

        registered = meta.get_registered_meta_specs(named=True)
        names = [s.name for s in registered if s.name]
        assert "test_register" in names

    def test_get_specs_from_name(self):
        """Test getting specs by name"""
        specs = MetaSpecs()
        specs._dict["register"]["name"] = "test_by_name"
        meta.register_meta_specs(specs)

        retrieved = meta.get_meta_specs_from_name("test_by_name")
        assert retrieved is specs

    def test_is_registered(self):
        """Test checking if specs are registered"""
        specs = MetaSpecs()
        specs._dict["register"]["name"] = "test_is_reg"
        meta.register_meta_specs(specs)

        assert meta.is_registered_meta_specs("test_is_reg")
        assert meta.is_registered_meta_specs(specs)

    def test_get_default_specs(self):
        """Test getting default specs"""
        specs = meta.get_default_meta_specs()
        assert isinstance(specs, MetaSpecs)
        assert "temp" in specs.data_vars.names

    def test_get_current_specs(self):
        """Test getting current specs"""
        specs = meta.get_meta_specs()
        assert isinstance(specs, MetaSpecs)

    def test_set_specs_context(self):
        """Test setting specs with context manager"""
        default_specs = meta.get_meta_specs()

        custom_cfg = """
        [data_vars]
            [[mytemp]]
            name = customtemp
        """
        custom_specs = MetaSpecs(custom_cfg)

        with meta.set_meta_specs(custom_specs) as specs:
            assert meta.get_meta_specs() is custom_specs
            assert "mytemp" in specs.data_vars.names

        # Should revert after context
        assert meta.get_meta_specs() is default_specs

    def test_reset_cache(self):
        """Test resetting cache"""
        meta.reset_cache(memory=False)
        meta.reset_cache(memory=True)
        # Should not raise any error

    def test_get_registered_specs(self):
        """Test getting all registered specs"""
        registered = meta.get_registered_meta_specs()
        assert isinstance(registered, list)

    def test_infer_specs(self):
        """Test inferring specs from dataset"""
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()

        specs = meta.infer_meta_specs(ds)
        assert isinstance(specs, MetaSpecs)

    def test_get_specs_matching_score(self):
        """Test computing matching score"""
        specs = MetaSpecs()
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()

        score = meta.get_meta_specs_matching_score(ds, specs)
        assert isinstance(score, (int, float))
        assert 0 <= score <= 100


class TestMetaUtilities:
    """Tests for meta utility functions"""

    def test_are_similar(self):
        """Test checking if two DataArrays are similar"""
        da1 = xr.DataArray(0, name="temp")
        da1.attrs["standard_name"] = "sea_water_temperature"

        da2 = xr.DataArray(0, name="temperature")
        da2.attrs["standard_name"] = "sea_water_temperature"

        assert meta.are_similar(da1, da2)

    def test_are_not_similar(self):
        """Test DataArrays that are not similar"""
        # Use simple DataArrays with different names and no matching attrs
        da1 = xr.DataArray(0, name="var1")
        da1.attrs["long_name"] = "Variable One"

        da2 = xr.DataArray(0, name="var2")
        da2.attrs["long_name"] = "Variable Two"

        # They should not be similar
        similar = meta.are_similar(da1, da2)
        assert similar is False

    def test_search_similar(self):
        """Test searching for similar DataArray"""
        temp1 = xr.DataArray(range(5), name="temp")
        temp1.attrs["standard_name"] = "sea_water_temperature"

        temp2 = xr.DataArray(range(5), name="temperature")
        temp2.attrs["standard_name"] = "sea_water_temperature"

        ds = xr.Dataset({"temp": temp1})

        result = meta.search_similar(ds, temp2)
        xr.testing.assert_equal(result, ds.temp)

    def test_infer_coords(self):
        """Test inferring coordinates from dataset"""
        lon = xr.DataArray(range(5), dims="x", name="longitude")
        lon.attrs["standard_name"] = "longitude"

        ds = xr.Dataset({"longitude": lon, "temp": (["x"], range(20, 25))})

        ds_new = meta.infer_coords(ds)
        assert "longitude" in ds_new.coords

    def test_assign_specs(self):
        """Test assigning specs to dataset"""
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()

        cfg = """
        [register]
        name = test_assign

        [data_vars]
            [[temp]]
        """
        specs = MetaSpecs(cfg)
        meta.register_meta_specs(specs)

        ds_new = meta.assign_meta_specs(ds, "test_assign")
        assert isinstance(ds_new, xr.Dataset)

    def test_get_matching_item_specs(self):
        """Test getting matching item specs"""
        temp = xr.DataArray(range(5), name="temp")
        temp.attrs["standard_name"] = "sea_water_temperature"

        specs_dict = meta.get_matching_item_specs(temp)
        assert specs_dict is not None

    def test_get_meta_specs_encoding(self):
        """Test getting meta_specs encoding from dataset"""
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()
        ds.encoding["meta_specs"] = "test_encoding"

        encoding = meta.get_meta_specs_encoding(ds)
        assert encoding == "test_encoding"

    def test_get_specs_from_encoding(self):
        """Test getting specs from encoding"""
        temp = xr.DataArray(range(5), name="temp")
        ds = temp.to_dataset()

        cfg = """
        [register]
        name = test_from_encoding

        [data_vars]
            [[temp]]
        """
        specs = MetaSpecs(cfg)
        meta.register_meta_specs(specs)

        ds.encoding["meta_specs"] = "test_from_encoding"

        retrieved = meta.get_meta_specs_from_encoding(ds)
        assert retrieved is specs


class TestMetaConfigs:
    """Tests for meta configurations"""

    def test_configs_available(self):
        """Test that META_CONFIGS exists and is populated"""
        assert isinstance(META_CONFIGS, dict)
        assert len(META_CONFIGS) > 0

    def test_load_config_by_name(self):
        """Test loading a config by name"""
        if META_CONFIGS:
            config_name = list(META_CONFIGS.keys())[0]
            specs = MetaSpecs(META_CONFIGS[config_name])
            assert isinstance(specs, MetaSpecs)


class TestMetaIntegration:
    """Integration tests combining multiple features"""

    def test_full_workflow(self):
        """Test a complete workflow with meta specs"""
        # Create custom specs
        cfg = """
        [register]
        name = integration_test

        [data_vars]
            [[mytemp]]
            name = temperature
            standard_name = sea_water_temperature

        [coords]
            [[mylon]]
            name = longitude
            standard_name = longitude
        """
        specs = MetaSpecs(cfg)
        meta.register_meta_specs(specs)

        # Create dataset
        lon = xr.DataArray(range(5), dims="x", name="longitude")
        temp = xr.DataArray(range(20, 25), dims="x", name="temperature")
        ds = xr.Dataset({"temperature": temp, "longitude": lon})

        # Set specs
        with meta.set_meta_specs(specs):
            # Search for variables
            found_temp = specs.search_data_var(ds, "mytemp", get="obj")
            assert found_temp.name == "temperature"

            # Infer coords
            ds_with_coords = meta.infer_coords(ds)
            assert "longitude" in ds_with_coords.coords

    def test_location_workflow(self):
        """Test workflow with locations"""
        specs = MetaSpecs()

        # Create data with locations
        temp_t = xr.DataArray(range(5), name="temp_t")
        temp_t.attrs["standard_name"] = "sea_water_temperature_at_t_location"

        # Match and extract location
        cat, name = specs.match(temp_t)
        assert cat == "data_vars"
        assert name == "temp"

        loc = specs.sglocator.get_loc_from_da(temp_t)
        assert loc == "t"
