#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.cf` module
"""

# from unittest.mock import Mock
import pytest

from xoa import cf


@pytest.mark.parametrize(
    "attr,value,expected",
    [
        ("standard_name", "my_var_at_t_location", ("my_var", "t")),
        ("standard_name", "my_var", ("my_var", None)),
        ("long_name", "My var at T location", ("My var", "t")),
        ("name", "myvar_t", ("myvar", "t")),
    ],
)
def test_cf_sglocator_parse_attr(attr, value, expected):
    assert cf.SGLocator().parse_attr(attr, value) == expected


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
def test_cf_sglocator_match_attr(attr, root, loc, expected):
    value = dict(
        standard_name="my_var_at_t_location",
        long_name="My var at T location",
        name="myvar_t",
    )[attr]
    assert cf.SGLocator().match_attr(attr, value, root, loc) is expected


@pytest.mark.parametrize(
    "attr,root,loc,expected",
    [
        ("standard_name", "my_var", "t", "my_var_at_t_location"),
        ("standard_name", "my_var", "", "my_var"),
        ("long_name", "My var", "t", "My var at T location"),
        ("name", "myvar", "t", "myvar_t"),
    ],
)
def test_cf_sglocator_format_attr(attr, root, loc, expected):
    assert cf.SGLocator().format_attr(attr, root, loc) == expected


def test_cf_sglocator_format_attr_valid_locations():
    with pytest.raises(cf.XoaCFError) as excinfo:
        cf.SGLocator(valid_locations="x").format_attr("name", "banana", "y")
    assert str(excinfo.value) == "Invalid location: y. Please one use of: x."


def test_cf_sglocator_format_attrs_no_loc():
    attrs = {
        "name": "u_u",
        "standard_name": "banana_at_t_location",
        "long_name": "Banana",
        "int_attr": 10,
        "str_attr": "good",
    }

    fmt_attrs = cf.SGLocator().format_attrs(attrs)
    assert fmt_attrs["name"] == "u"
    assert fmt_attrs["standard_name"] == "banana"
    assert fmt_attrs["long_name"] == "Banana"
    for attr in ("int_attr", "str_attr"):
        assert fmt_attrs[attr] == attrs[attr]


def test_cf_sglocator_format_attrs_with_loc():
    attrs = {
        "name": "u_u",
        "standard_name": "banana_at_t_location",
        "long_name": "Banana",
        "int_attr": 10,
        "str_attr": "good",
    }

    fmt_attrs = cf.SGLocator().format_attrs(attrs, loc="f")
    assert fmt_attrs["name"] == "u_f"
    assert fmt_attrs["standard_name"] == "banana_at_f_location"
    assert fmt_attrs["long_name"] == "Banana at F location"
    for attr in ("int_attr", "str_attr"):
        assert fmt_attrs[attr] == attrs[attr]


def test_cf_sglocator_format_dataarray():
    import xarray as xr

    lon = xr.DataArray(range(5), dims="lon")
    banana = xr.DataArray(
        lon + 20,
        dims="lon",
        coords=[lon],
        name="banana",
        attrs={"standard_name": "banana", "taste": "good"},
    )
    print(banana)
    banana_fmt = cf.SGLocator().format_dataarray(banana, "p")
    print(banana_fmt)
    assert banana_fmt.name == "banana_p"
    assert banana_fmt.standard_name == "banana_at_p_location"
    assert banana_fmt.taste == "good"


@pytest.mark.parametrize(
    "cache", ["ignore", "write", "rw", "read", "ignore", "clean", "rw"]
)
def test_cf_get_cfg_specs(cache):
    assert isinstance(cf.get_cf_specs(cache=cache), cf.CFSpecs)


def test_cf_get_cfg_specs_var():
    specs = cf.get_cf_specs("temp", "data_vars")
    assert specs["name"][0] == "temp"
    assert specs["standard_name"][0] == "sea_water_temperature"
    assert specs["cmap"] == "cmo.thermal"
    new_specs = cf.get_cf_specs("temp")
    assert new_specs is specs


def test_cf_get_cfg_specs_var_inherit():
    specs = cf.get_cf_specs("sst", "data_vars")
    assert specs["standard_name"][0] == "sea_surface_temperature"
    assert specs["units"][0] == "degrees_celsius"


def test_cf_get_cfg_specs_coord():
    specs = cf.get_cf_specs("lon", "coords")
    assert specs["name"][0] == "lon"
    assert "longitude" in specs["name"]
    new_specs = cf.get_cf_specs("lon")
    assert new_specs is specs


def test_cf_get_cfg_specs_coord_inherit():
    specs = cf.get_cf_specs("depth", "coords")
    assert specs["name"][0] == "depth"
    assert specs["long_name"][0] == "Depth"


@pytest.mark.parametrize(
    "cfg,key,name",
    [
        ({"data_vars": {"temp": {"name": "mytemp"}}}, "temp", "mytemp"),
        ("[data_vars]\n[[sal]]\nname=mysal", "sal", "mysal"),
    ],
)
def test_cf_cfspecs_load_cfg(cfg, key, name):
    cfspecs = cf.get_cf_specs()
    cfspecs.load_cfg(cfg)
    assert name in cfspecs["data_vars"][key]["name"]


def test_cf_cfspecs_copy():
    cfspecs0 = cf.get_cf_specs()
    cfspecs1 = cfspecs0.copy()
    assert id(cfspecs0._dict) != id(cfspecs1._dict)
    assert sorted(list(cfspecs0._dict["data_vars"])) == sorted(
        list(cfspecs1._dict["data_vars"])
    )
    assert cfspecs0._dict["coords"] == cfspecs1._dict["coords"]
    assert (
        cfspecs0._dict["data_vars"]["temp"]
        == cfspecs1._dict["data_vars"]["temp"]
    )
    assert "temp" in cfspecs1["data_vars"]
    assert "temperature" in cfspecs1["data_vars"]["temp"]["name"]


def test_cf_set_cf_specs():
    cf._CACHE.clear()
    cfspecs = cf.get_cf_specs()
    cf.set_cf_specs(cfspecs)
    assert "specs" in cf._CACHE
    assert cf._CACHE["specs"] is cfspecs
    assert cf.get_cf_specs() is cfspecs


def test_cf_set_cf_specs_context():
    cfspecs0 = cf.get_cf_specs()
    cfspecs1 = cf.CFSpecs({"data_vars": {"temp": {"name": "tempouille"}}})
    assert cf.get_cf_specs() is cfspecs0
    with cf.set_cf_specs(cfspecs1) as cfspecs:
        assert cfspecs is cfspecs1
        assert cf.get_cf_specs() is cfspecs1
    assert cf.get_cf_specs() is cfspecs0


@pytest.mark.parametrize("ename", [None, "lon"])
@pytest.mark.parametrize(
    "name,attrs",
    [
        ("lon", None),
        ("xxx", {"standard_name": "longitude"}),
        ("xxx", {"units": "degree_east"}),
    ],
)
def test_cf_cfspecs_format_coord(ename, name, attrs):
    import xarray as xr

    lon = xr.DataArray(range(5), dims=name, name=name, attrs=attrs)
    lon = cf.get_cf_specs().format_coord(lon, ename)
    assert lon.name == "lon"
    assert lon.standard_name == "longitude"
    assert lon.long_name == "Longitude"
    assert lon.units == "degrees_east"
