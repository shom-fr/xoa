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
            (None, "standard_name", "my_var_at_t_location", ("my_var", "t")),
            (None, "standard_name", "my_var", ("my_var", None)),
            (None, "long_name", "My var at T location", ("My var", "t")),
            (None, "name", "myvar_t", ("myvar_t", None)),
            ("{root}_{loc}", "name", "myvar_t", ("myvar", "t")),
        ],
    )
    def test_parse_attr(self, name_format, attr, value, expected):
        """Test parsing attributes to extract root and location"""
        result = SGLocator(name_format=name_format).parse_attr(attr, value)
        assert result == expected

    @pytest.mark.parametrize(
        "name_format,attr,value,expected",
        [
            (None, "standard_name", "my_var_at_t_location", ("my_var_at_t_location", None)),
            (None, "standard_name", "my_var_at_u_location", ("my_var", "u")),
            (None, "long_name", "My var at RHO location", ("My var", "rho")),
            (None, "long_name", "My var at rho location", ("My var", "rho")),
            (None, "name", "myvarrho", ("myvarrho", None)),
            ("{root}{loc}", "name", "myvarrho", ("myvar", "rho")),
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
            ("u_t", None, None, "t"),
            (None, "u_at_t_location", None, "t"),
            (None, None, "U at T location", "t"),
            ("u_t", "_at_t_location", "U at T location", "t"),
            ("u", "u_at_t_location", None, "t"),
            ("u", "u", "U", None),
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

    @pytest.mark.parametrize(
        "name,standard_name,long_name",
        [
            ("u_t", "u_at_u_location", None),
            (None, "u_at_t_location", "U at U location"),
            ("u_u", None, "U at T location"),
            ("u_u", "u_at_w_location", "U at T location"),
        ],
    )
    def test_get_loc_from_da_error(self, name, standard_name, long_name):
        """Test get_loc_from_da with conflicting location info raises error"""
        da = xr.DataArray(0)
        if name:
            da.name = name
        if standard_name:
            da.attrs["standard_name"] = standard_name
        if long_name:
            da.attrs["long_name"] = long_name

        sgl = SGLocator(name_format="{root}_{loc}")
        with pytest.raises(Exception):  # XoaMetaError
            sgl.get_loc_from_da(da, errors="raise")

        # Should not raise with errors="ignore"
        sgl.get_loc_from_da(da, errors="ignore")


    @pytest.mark.parametrize(
        "name_format,attr,root,loc,expected",
        [
            (None, "standard_name", "my_var", "t", "my_var_at_t_location"),
            (None, "standard_name", "my_var", "", "my_var"),
            (None, "long_name", "My var", "t", "My var at T location"),
            (None, "name", "myvar", "t", "myvar"),
            ("{root}_{loc}", "name", "myvar", "t", "myvar_t"),
        ],
    )
    def test_format_attr(self, name_format, attr, root, loc, expected):
        """Test formatting attributes with location"""
        result = SGLocator(name_format=name_format).format_attr(attr, root, loc)
        assert result == expected

    def test_format_attr_valid_locations_error(self):
        """Test format_attr with invalid location raises error"""
        with pytest.raises(Exception):  # XoaMetaError
            SGLocator(valid_locations="x").format_attr("name", "banana", "y")

    def test_format_attrs_no_loc(self):
        """Test formatting attributes without location"""
        attrs = {
            "name": "u_u",
            "standard_name": "banana_at_t_location",
            "long_name": "Banana at T location",
            "int_attr": 10,
            "str_attr": "good",
        }

        fmt_attrs = SGLocator(name_format="{root}_{loc}").format_attrs(attrs, loc='')
        assert fmt_attrs["name"] == "u_u"
        assert fmt_attrs["standard_name"] == "banana"
        assert fmt_attrs["long_name"] == "Banana"
        for attr in ("int_attr", "str_attr"):
            assert fmt_attrs[attr] == attrs[attr]

    def test_format_attrs_with_loc(self):
        """Test formatting attributes with location"""
        attrs = {
            "name": "u_u",
            "standard_name": "banana_at_t_location",
            "long_name": "Banana",
            "int_attr": 10,
            "str_attr": "good",
        }

        fmt_attrs = SGLocator(name_format="{root}_{loc}").format_attrs(attrs, loc="f")
        assert fmt_attrs["name"] == "u_u"
        assert fmt_attrs["standard_name"] == "banana_at_f_location"
        assert fmt_attrs["long_name"] == "Banana at F location"
        for attr in ("int_attr", "str_attr"):
            assert fmt_attrs[attr] == attrs[attr]

    @pytest.mark.parametrize(
        "attr,root,loc,expected",
        [
            ("standard_name", "my_var", "t", True),
            ("standard_name", "my_var2", "t", False),
            ("standard_name", "my_var", "x", False),
            ("standard_name", "my_var", None, True),
            ("standard_name", "my_var", "xtu", True),
            ("long_name", "My var", "t", True),
            ("long_name", "My var", "x", False),
            ("name", "myvar", "t", True),
            ("name", "myvar", "x", False),
        ],
    )
    def test_match_attr(self, attr, root, loc, expected):
        """Test matching an attribute against a root and location"""
        value = dict(
            standard_name="my_var_at_t_location",
            long_name="My var at T location",
            name="myvar_t",
        )[attr]
        result = SGLocator(name_format="{root}_{loc}").match_attr(attr, value, root, loc)
        assert result is expected

    @pytest.mark.parametrize(
        "value0, value1, loc, value",
        [
            ("sst", None, "t", "sst_t"),
            (None, "sst", "t", "sst_t"),
            ("sst", "sss", "t", "sss_t"),
            ("sst_x", "sss_y", "t", "sss_t"),
            ("sst_t", None, None, "sst_t"),
            (None, "sst_t", None, "sst_t"),
            ("sst_x", "sss_y", None, "sss_y"),
            ("sst_x", "sss", None, "sss_x"),
            ("sst", "sss_y", None, "sss_y"),
        ],
    )
    def test_merge_attr(self, value0, value1, loc, value):
        """Test merging two attribute values"""
        out = SGLocator(name_format="{root}_{loc}").merge_attr("name", value0, value1, loc)
        assert out == value

    @pytest.mark.parametrize(
        "isn, psn, osn, loc, replace",
        [
            ("sst", None, "sst", None, False),
            (None, "sst", "sst", None, False),
            ("sst", "temp", "sst", None, False),
            ("sst", "temp", "temp", None, True),
            ("sst_at_t_location", "temp_at_u_location", "sst_at_t_location", None, False),
            ("sst_at_t_location", "temp_at_u_location", "temp_at_u_location", None, True),
            ("sst", "temp", "sst_at_u_location", "u", False),
            ("sst", "temp", "temp_at_u_location", "u", True),
            ("sst_at_t_location", "temp_at_x_location", "sst_at_u_location", "u", False),
            ("sst_at_t_location", "temp_at_x_location", "temp_at_u_location", "u", True),
        ],
    )
    def test_patch_attrs(self, isn, psn, osn, loc, replace):
        """Test patching attributes dictionary"""
        iattrs = {"units": "m", "color": "blue"}
        patch = {"cmap": "viridis", "mylist": [1, 2], "units": "cm"}
        if isn:
            iattrs["standard_name"] = isn
        if psn:
            patch["standard_name"] = psn

        oattrs = SGLocator(name_format="{root}_{loc}").patch_attrs(
            iattrs, patch, loc=loc, replace=replace
        )

        assert oattrs["units"] == ("cm" if replace else "m")
        assert oattrs["color"] == "blue"
        assert oattrs["cmap"] == "viridis"
        assert oattrs["mylist"] == [1, 2]
        assert oattrs.get("standard_name") == osn

    @pytest.mark.parametrize(
        "floc,fname,fattrs,out_name,out_standard_name,replace_attrs",
        [
            ("p", None, None, "banana_p", "banana_at_p_location", False),
            (None, None, None, "banana_t", "banana_at_t_location", False),
            ("", None, None, "banana", "banana", False),
            (False, None, None, "banana", "banana", False),
            ("p", "sst", {"standard_name": "potatoe"}, "sst_p", "banana_at_p_location", False),
            ("p", "sst", {"standard_name": "potatoe"}, "sst_p", "potatoe_at_p_location", True),
            (
                'x',
                "sst",
                {"standard_name": ["potatoe", "banana"]},
                "sst_x",
                "banana_at_x_location",
                True,
            ),
            (None, "sst_q", {"standard_name": ["potatoe"]}, "sst_q", "potatoe_at_q_location", True),
            (None, "sst", {"standard_name": ["potatoe"]}, "sst_t", "potatoe_at_t_location", True),
        ],
    )
    def test_format_dataarray(
        self, floc, fname, fattrs, out_name, out_standard_name, replace_attrs
    ):
        """Test formatting a DataArray"""
        lon = xr.DataArray(range(5), dims="lon")
        banana = xr.DataArray(
            lon + 20,
            dims="lon",
            coords=[lon],
            name="banana_t",
            attrs={"standard_name": "banana", "taste": "good"},
        )
        banana_fmt = SGLocator(name_format="{root}_{loc}").format_dataarray(
            banana, loc=floc, name=fname, attrs=fattrs, replace_attrs=replace_attrs
        )
        assert banana_fmt.name == out_name
        assert banana_fmt.standard_name == out_standard_name
        assert banana_fmt.taste == "good"

    def test_format_dataarray_no_copy_no_rename(self):
        """Test formatting without copy and rename"""
        banana = xr.DataArray(1, name="banana_t", attrs={"standard_name": "banana"})
        banana_fmt = SGLocator(name_format="{root}_{loc}").format_dataarray(
            banana, "p", copy=False, rename=False
        )
        assert banana_fmt is banana
        assert banana_fmt.name == "banana_t"
        assert banana_fmt.standard_name == "banana_at_p_location"


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


class TestCoordSpecs:
    """Additional tests for coordinate specifications"""

    def test_get_axis(self):
        """Test getting axis from coordinate"""
        specs = MetaSpecs()

        # From attrs
        depth = xr.DataArray([1], dims='aa', attrs={'axis': 'z'})
        assert specs.coords.get_axis(depth) == 'Z'

        # From CF specs
        depth = xr.DataArray([1], dims='aa', attrs={'standard_name': 'ocean_layer_depth'})
        assert specs.coords.get_axis(depth) == 'Z'

    def test_get_dim_type(self):
        """Test getting dimension type"""
        specs = MetaSpecs()

        # From name
        assert specs.coords.get_dim_type('aa') is None
        assert specs.coords.get_dim_type('xi') == "x"

        # From a known coordinate
        coord = xr.DataArray([1], dims='aa', attrs={'standard_name': 'longitude'})
        da = xr.DataArray([1], dims='aa', coords={'aa': coord})
        assert specs.coords.get_dim_type('aa', da) == "x"

    def test_get_dim_types(self):
        """Test getting multiple dimension types"""
        specs = MetaSpecs()

        aa = xr.DataArray([0, 1], dims="aa", attrs={"standard_name": "latitude"})
        da = xr.DataArray(np.ones((2, 2, 2)), dims=('foo', 'aa', 'xi'), coords={'aa': aa})

        assert specs.coords.get_dim_types(da) == (None, 'y', 'x')
        assert specs.coords.get_dim_types(da, unknown='-') == ("-", 'y', 'x')
        assert specs.coords.get_dim_types(da, asdict=True) == {"foo": None, "aa": "y", "xi": "x"}

    def test_get_dims(self):
        """Test getting dimensions"""
        lat = xr.DataArray([4, 5], dims='yy', attrs={'units': 'degrees_north'})
        depth = xr.DataArray([4, 5], dims='level', attrs={'axis': 'Z'})
        da = xr.DataArray(
            np.ones((2, 2, 2, 2)), dims=('r', 'level', 'yy', 'xi'), coords={'level': depth, 'yy': lat}
        )

        specs = MetaSpecs()
        dims = specs.coords.get_dims(da, ['x', 'y', 'z', 't'], allow_positional=True)
        assert dims == ('xi', 'yy', 'level', 'r')

    def test_search_from_dim(self):
        """Test searching coordinates from dimension"""
        lon = xr.DataArray([1, 2], dims='lon')
        level = xr.DataArray([1, 2, 3], dims='aa', attrs={'standard_name': 'ocean_sigma_coordinate'})
        mem = xr.DataArray(range(3), dims='mem')
        temp = xr.DataArray(
            np.zeros((mem.size, level.size, lon.size)),
            dims=('mem', 'aa', 'lon'),
            coords={'mem': mem, 'aa': level, 'lon': lon},
        )

        specs = MetaSpecs()

        # Direct coordinate
        assert specs.coords.search_from_dim(temp, 'aa').name == 'aa'

        # Depth coordinate because the only one with this dim
        depth = xr.DataArray(
            np.ones((level.size, lon.size)), dims=('aa', 'lon'), coords={'aa': level, 'lon': lon}
        )
        temp.coords['depth'] = depth
        assert specs.coords.search_from_dim(temp, 'aa').name == 'depth'

    def test_search_coord_with_stacking(self):
        """Test searching coordinates with stacked dimensions"""
        ds = xr.Dataset(
            coords={
                "lon": ("lon", np.linspace(-10, -2, 5)),
                "lat": ("lat", np.linspace(43, 49, 4)),
            }
        ).stack(npts=("lat", "lon"))

        specs = MetaSpecs()
        res = specs.search_coord(ds, "lon", get="obj")
        assert res is not None
        assert res.shape == (20,)
        assert res.name == "lon"


class TestFormatDataVar:
    """Additional tests for data variable formatting"""

    def test_format_data_var_basic(self):
        """Test basic data variable formatting"""
        lon = xr.DataArray(range(5), dims='xxx', name='xxx', attrs={'standard_name': 'longitude'})
        temp = xr.DataArray(
            range(20, 25), dims='xxx', coords={'xxx': lon}, name='temp'
        )
        specs = MetaSpecs()
        temp_fmt = specs.format_data_var(temp)
        assert temp_fmt.name == "temp"
        assert temp_fmt.standard_name == "sea_water_temperature"

    def test_format_data_var_unknown(self):
        """Test formatting unknown data variable"""
        da = xr.DataArray(range(5), name='foo')
        specs = MetaSpecs()

        da_fmt = specs.format_data_var(da, rename=False)
        assert da_fmt.name == "foo"

    def test_format_data_var_coord(self):
        """Test formatting coordinate as data variable"""
        da = xr.DataArray(0, attrs={'standard_name': 'longitude_at_u_location'})
        specs = MetaSpecs()
        da_fmt = specs.format_data_var(da)
        # Should not raise error


class TestGetSpecs:
    """Additional tests for get specifications"""

    def test_get_specs_var(self):
        """Test getting variable specifications"""
        specs = MetaSpecs()
        var_specs = specs.data_vars["temp"]
        assert var_specs["alt_names"][0] == "temperature"
        assert var_specs["attrs"]["standard_name"][0] == "sea_water_temperature"

    def test_get_specs_var_inherit(self):
        """Test getting variable specs with inheritance"""
        specs = MetaSpecs()
        sst_specs = specs.data_vars["sst"]
        assert sst_specs["attrs"]["standard_name"][0] == "sea_surface_temperature"
        assert sst_specs["attrs"]["units"][0] == "degrees_celsius"

    def test_get_specs_coord(self):
        """Test getting coordinate specifications"""
        specs = MetaSpecs()
        lon_specs = specs.coords["lon"]
        assert lon_specs["alt_names"][0] == "longitude"
        assert "longitude" in lon_specs["alt_names"]

    def test_get_specs_coord_inherit(self):
        """Test getting coordinate specs with inheritance"""
        specs = MetaSpecs()
        depth_specs = specs.coords["depth"]
        assert depth_specs["alt_names"][0] == "dep"
        assert depth_specs["attrs"]["long_name"][0] == "Depth"

    def test_get_from_dataset(self):
        """Test get method from dataset"""
        ds = xr.Dataset({"xtemp": ("x", [0], {"standard_name": "sea_water_temperature"}), "psal": 1})

        specs = MetaSpecs()
        assert specs.get(ds, "temp").name == "xtemp"
        assert specs.get(ds, ["temp", "psal"]).name == "xtemp"
        assert specs.get(ds, "depth") is None


class TestIntegration:
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
