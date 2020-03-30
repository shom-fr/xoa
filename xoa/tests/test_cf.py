#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.cf` module
"""

# from unittest.mock import Mock
import warnings
import pytest

import numpy as np
import xarray as xr

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
    "attr,value,expected",
    [
        ("standard_name", "my_var_at_t_location",
         ("my_var_at_t_location", None)),
        ("standard_name", "my_var_at_u_location", ("my_var", "u")),
        ("long_name", "My var at RHO location", ("My var", "rho")),
        ("long_name", "My var at rho location", ("My var", "rho")),
        ("name", "myvarrho", ("myvar", "rho")),
    ],
)
def test_cf_sglocator_parse_attr_with_valid_locations(attr, value, expected):
    assert cf.SGLocator(valid_locations=['u', 'rho'],
                        name_format="{root}{loc}",
                        ).parse_attr(attr, value) == expected


@pytest.mark.parametrize(
    "name,standard_name,long_name,loc",
    [
     ("u_t", None, None, "t"),
     (None, "u_at_t_location", None, "t"),
     (None, None, "U at T location", "t"),
     ("u_t", "_at_t_location", "U at T location", "t"),
     ("u", "u_at_t_location", None, "t"),
     ("u", "u", "U", None)
     ],
    )
def test_cf_sglocator_get_location(name, standard_name, long_name, loc):

    da = xr.DataArray(0)
    if name:
        da.name = name
    if standard_name:
        da.attrs["standard_name"] = standard_name
    if long_name:
        da.attrs["long_name"] = long_name

    parsed_loc = cf.SGLocator().get_location(da)
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
def test_cf_sglocator_get_location_error(name, standard_name, long_name):

    da = xr.DataArray(0)
    if name:
        da.name = name
    if standard_name:
        da.attrs["standard_name"] = standard_name
    if long_name:
        da.attrs["long_name"] = long_name

    with pytest.raises(cf.XoaCFError) as excinfo:
        cf.SGLocator().get_location(da)


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
        "long_name": "Banana at T location",
        "int_attr": 10,
        "str_attr": "good",
    }

    fmt_attrs = cf.SGLocator().format_attrs(attrs, loc='')
    assert fmt_attrs["name"] == "u_u"
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
    assert fmt_attrs["name"] == "u_u"
    assert fmt_attrs["standard_name"] == "banana_at_f_location"
    assert fmt_attrs["long_name"] == "Banana at F location"
    for attr in ("int_attr", "str_attr"):
        assert fmt_attrs[attr] == attrs[attr]


def test_cf_sglocator_format_dataarray():

    lon = xr.DataArray(range(5), dims="lon")
    banana = xr.DataArray(
        lon + 20,
        dims="lon",
        coords=[lon],
        name="banana",
        attrs={"standard_name": "banana", "taste": "good"},
    )
    banana_fmt = cf.SGLocator().format_dataarray(banana, "p")
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


@pytest.mark.parametrize("cf_name", [None, "lon"])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("lon", None),
        ("xxx", {"standard_name": "longitude"}),
        ("xxx", {"standard_name": "longitude_at_t_location"}),
        ("xxx", {"units": "degree_east"}),
    ],
)
def test_cf_cfspecs_match_coord(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims=in_name, name=in_name, attrs=in_attrs)
    res = cf.get_cf_specs().match_coord(lon, cf_name)
    if cf_name is None:
        assert res == 'lon'
    else:
        assert res is True


@pytest.mark.parametrize("cf_name", ["lon", None])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("lon", None),
        ("xxx", {"standard_name": "longitude"}),
        ("xxx", {"standard_name": "longitude_at_t_location"}),
        ("xxx", {"units": "degree_east"}),
    ],
)
def test_cf_cfspecs_search_coord(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims=in_name, name=in_name, attrs=in_attrs)
    temp = xr.DataArray(range(20, 25), dims=in_name,
                        coords={in_name: lon}, name='temp')
    res = cf.get_cf_specs().search_coord(temp, cf_name, get="name")
    assert res == 'lon'


@pytest.mark.parametrize("cf_name", ["temp", None])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("temp", None),
        ("xxx", {"standard_name": "sea_water_temperature"}),
    ],
)
def test_cf_cfspecs_match_data_var(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims='lon', name='lon')
    temp = xr.DataArray(range(20, 25), dims='lon', coords={'lon': lon},
                        name=in_name, attrs=in_attrs)
    res = cf.get_cf_specs().match_data_var(temp, cf_name)
    if cf_name is None:
        assert res == 'temp'
    else:
        assert res is True


@pytest.mark.parametrize("cf_name", ["temp", None])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("temp", None),
        ("xxx", {"standard_name": "sea_water_temperature"}),
    ],
)
def test_cf_cfspecs_search_data_var(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims='lon', name='lon')
    temp = xr.DataArray(range(20, 25), dims='lon', coords={'lon': lon},
                        name=in_name, attrs=in_attrs)
    ds = temp.to_dataset()
    assert cf.get_cf_specs().search_data_var(
        ds, cf_name, get="name") == 'temp'


@pytest.mark.parametrize("cf_name", [None, "lon"])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("lon", None),
        ("xxx", {"standard_name": "longitude"}),
        ("xxx", {"units": "degree_east"}),
    ],
)
def test_cf_cfspecs_format_coord(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims=in_name, name=in_name, attrs=in_attrs)
    lon = cf.get_cf_specs().format_coord(lon, cf_name)
    assert lon.name == "lon"
    assert lon.standard_name == "longitude"
    assert lon.long_name == "Longitude"
    assert lon.units == "degrees_east"


@pytest.mark.parametrize("cf_name", [None, "temp"])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("temp", None),
        ("xxx", {"standard_name": "sea_water_temperature"}),
    ],
)
def test_cf_cfspecs_format_data_var(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims='xxx', name='xxx',
                       attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(range(20, 25), dims='xxx', coords={'xxx': lon},
                        name=in_name, attrs=in_attrs)
    temp = cf.get_cf_specs().format_data_var(temp, cf_name)
    assert temp.name == "temp"
    assert temp.standard_name == "sea_water_temperature"
    assert temp.long_name == "Temperature"
    assert temp.units == "degrees_celsius"

    assert temp.lon.standard_name == "longitude"


def test_cf_cfspecs_coords_get_dim_type():
    cfspecs = cf.get_cf_specs().coords

    # from attrs
    depth = xr.DataArray([1], dims='aa', attrs={'axis': 'z'})
    assert cfspecs.get_dim_type(depth) == 'z'

    # from CF specs
    depth = xr.DataArray([1], dims='aa',
                         attrs={'standard_name': 'ocean_layer_depth'})
    assert cfspecs.get_dim_type(depth) == 'z'


def test_cf_cfspecs_coords_search_dim():
    cfspecs = cf.get_cf_specs().coords

    # from name
    temp = xr.DataArray(np.arange(2*3).reshape(1, 2, 3),
                        dims=('aa', 'ny', 'x'))
    assert cfspecs.search_dim(temp, 'y') == 'ny'
    assert cfspecs.search_dim(temp) is None

    # from explicit axis attribute
    depth = xr.DataArray([1], dims='aa', attrs={'axis': 'z'})
    temp.coords['aa'] = depth
    assert cfspecs.search_dim(temp, 'z') == 'aa'
    assert cfspecs.search_dim(temp) is None

    # from known coordinate
    del temp.coords['aa'].attrs['axis']
    temp.coords['aa'].attrs['standard_name'] = 'ocean_layer_depth'
    assert cfspecs.search_dim(temp, 'z') == 'aa'
    assert cfspecs.search_dim(temp) is None

    # subcoords
    level = xr.DataArray(np.arange(2), dims='level')
    depth = xr.DataArray(np.arange(2*3).reshape(2, 3),
                         dims=('level', 'x'), name='depth',
                         coords={'level': level})
    assert cfspecs.search_dim(depth) == ('level', 'z')
    assert cfspecs.search_dim(depth.level) == ('level', 'z')
    depth = depth.rename(level='aa')
    depth.aa.attrs['axis'] = 'Z'
    assert cfspecs.search_dim(depth) == ('aa', 'z')

    assert cfspecs.search_dim(xr.DataArray([5], dims='bb')) == ('bb', None)


def test_cf_cfspecs_coords_search_from_dim():

    lon = xr.DataArray([1, 2], dims='lon')
    level = xr.DataArray([1, 2, 3], dims='aa',
                         attrs={'standard_name': 'ocean_sigma_coordinate'})
    mem = xr.DataArray(range(3), dims='mem')
    temp = xr.DataArray(np.zeros((mem.size, level.size, lon.size)),
                        dims=('mem', 'aa', 'lon'),
                        coords={'mem': mem, 'aa': level, 'lon': lon})

    cfspecs = cf.get_cf_specs().coords

    # Direct coordinate
    assert cfspecs.search_from_dim(temp, 'aa').name == 'aa'

    # Depth coordinate from thanks to dim_type of 1D coordinate
    depth = xr.DataArray(np.ones((level.size, lon.size)),
                         dims=('aa', 'lon'),
                         coords={'aa': level, 'lon': lon})
    temp.coords['depth'] = depth
    assert cfspecs.search_from_dim(temp, 'aa').name == 'depth'

    # Cannot get dim_type
    del temp.coords['aa']
    assert cfspecs.search_from_dim(temp, 'aa') is None

    # Cen get dim_type back from name
    temp = temp.rename(aa='level')
    assert cfspecs.search_from_dim(temp, 'level').name == 'depth'

    # Nothing identifiable
    temp = xr.DataArray([3], dims='banana')
    assert cfspecs.search_from_dim(temp, "banana") is None


def test_cf_dataarraycfaccessor():
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            xr.core.extensions.AccessorRegistrationWarning)
        xr.register_dataarray_accessor('cf')(cf.DataArrayCFAccessor)

    lon = xr.DataArray(range(5), dims='xxx', name='xxx',
                       attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(range(20, 25), dims='xxx',
                        coords={'xxx': lon}, name='temp')

    assert temp.cf.lon.name == 'xxx'
    assert temp.cf.lat is None


def test_cf_datasetcfaccessor():
    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            xr.core.extensions.AccessorRegistrationWarning)
        xr.register_dataset_accessor('cf')(cf.DatasetCFAccessor)

    lon = xr.DataArray(range(5), dims='xxx', name='xxx',
                       attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(range(20, 25), dims='xxx',
                        coords={'xxx': lon}, name='yoyo',
                        attrs={'standard_name': 'sea_water_temperature'})

    ds = temp.to_dataset()
    assert ds.cf.temp.name == 'yoyo'
    assert ds.cf.sal is None
