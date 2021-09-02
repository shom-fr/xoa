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

import xoa
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
        ("standard_name", "my_var_at_t_location", ("my_var_at_t_location", None)),
        ("standard_name", "my_var_at_u_location", ("my_var", "u")),
        ("long_name", "My var at RHO location", ("My var", "rho")),
        ("long_name", "My var at rho location", ("My var", "rho")),
        ("name", "myvarrho", ("myvar", "rho")),
    ],
)
def test_cf_sglocator_parse_attr_with_valid_locations(attr, value, expected):
    assert (
        cf.SGLocator(
            valid_locations=['u', 'rho'],
            name_format="{root}{loc}",
        ).parse_attr(attr, value)
        == expected
    )


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
def test_cf_sglocator_get_loc_from_da(name, standard_name, long_name, loc):

    da = xr.DataArray(0)
    if name:
        da.name = name
    if standard_name:
        da.attrs["standard_name"] = standard_name
    if long_name:
        da.attrs["long_name"] = long_name

    parsed_loc = cf.SGLocator().get_loc_from_da(da)
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
def test_cf_sglocator_get_loc_from_da_error(name, standard_name, long_name):

    da = xr.DataArray(0)
    if name:
        da.name = name
    if standard_name:
        da.attrs["standard_name"] = standard_name
    if long_name:
        da.attrs["long_name"] = long_name

    with pytest.raises(cf.XoaCFError):
        cf.SGLocator().get_loc_from_da(da, errors="raise")

    cf.SGLocator().get_loc_from_da(da, errors="ignore")


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
    assert str(excinfo.value) == (
        'Location "y" is not recognised by the currents '
        'specifications. Registered locations are: x'
    )


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
def test_cf_sglocator_merge_attr(value0, value1, loc, value):
    out = cf.SGLocator().merge_attr("name", value0, value1, loc)
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
def test_cf_sglocator_patch_attrs(isn, psn, osn, loc, replace):

    iattrs = {"units": "m", "color": "blue"}
    patch = {"cmap": "viridis", "mylist": [1, 2], "units": "cm"}
    if isn:
        iattrs["standard_name"] = isn
    if psn:
        patch["standard_name"] = psn

    oattrs = cf.SGLocator().patch_attrs(iattrs, patch, loc=loc, replace=replace)

    assert oattrs["units"] == ("cm" if replace else "m")
    assert oattrs["color"] == "blue"
    assert oattrs["cmap"] == "viridis"
    assert oattrs["mylist"] == [1, 2]

    assert oattrs.get("standard_name") == osn


@pytest.mark.parametrize(
    "floc,fname,fattrs,out_name,out_standard_name,replace_attrs",
    [
        # ("p", None, None, "banana_p", "banana_at_p_location", False),
        # (None, None, None, "banana_t", "banana", False),

        # ("p", "sst", {"standard_name": "potatoe"}, "sst_p", "banana_at_p_location", False),
        # ("p", "sst", {"standard_name": "potatoe"}, "sst_p", "potatoe_at_p_location", True),
        # (
        #     'x',
        #     "sst",
        #     {"standard_name": ["potatoe", "banana"]},
        #     "sst_x",
        #     "banana_at_x_location",
        #     True,
        # ),
        (None, "sst_q", {"standard_name": ["potatoe"]}, "sst_q", "potatoe_at_q_location", True),
        # (None, "sst", {"standard_name": ["potatoe"]}, "sst_t", "potatoe_at_t_location", True),
    ],
)
def test_cf_sglocator_format_dataarray(
    floc, fname, fattrs, out_name, out_standard_name, replace_attrs
):

    lon = xr.DataArray(range(5), dims="lon")
    banana = xr.DataArray(
        lon + 20,
        dims="lon",
        coords=[lon],
        name="banana_t",
        attrs={"standard_name": "banana", "taste": "good"},
    )
    banana_fmt = cf.SGLocator().format_dataarray(
        banana, loc=floc, name=fname, attrs=fattrs, replace_attrs=replace_attrs
    )
    assert banana_fmt.name == out_name
    assert banana_fmt.standard_name == out_standard_name
    assert banana_fmt.taste == "good"


def test_cf_sglocator_format_dataarray_no_copy_no_rename():
    banana = xr.DataArray(1, name="banana_t", attrs={"standard_name": "banana"})
    banana_fmt = cf.SGLocator().format_dataarray(banana, "p", copy=False, rename=False)
    assert banana_fmt is banana
    assert banana_fmt.name == "banana_t"
    assert banana_fmt.standard_name == "banana_at_p_location"


@pytest.mark.parametrize("cache", ["ignore", "write", "rw", "read", "ignore", "clean", "rw"])
def test_cf_get_cfg_specs(cache):
    assert isinstance(cf.get_cf_specs(cache=cache), cf.CFSpecs)


def test_cf_get_cfg_specs_var():
    specs = cf.get_cf_specs().data_vars["temp"]
    assert specs["alt_names"][0] == "temperature"
    assert specs["attrs"]["standard_name"][0] == "sea_water_temperature"
    assert specs["cmap"] == "cmo.thermal"
    new_specs = cf.get_cf_specs()["temp"]
    assert new_specs is specs


def test_cf_get_cfg_specs_var_inherit():
    specs = cf.get_cf_specs().data_vars["sst"]
    assert specs["attrs"]["standard_name"][0] == "sea_surface_temperature"
    assert specs["attrs"]["units"][0] == "degrees_celsius"


def test_cf_get_cfg_specs_coord():
    specs = cf.get_cf_specs().coords["lon"]
    assert specs["alt_names"][0] == "longitude"
    assert "longitude" in specs["alt_names"]
    new_specs = cf.get_cf_specs()["lon"]
    assert new_specs is specs


def test_cf_get_cfg_specs_coord_inherit():
    specs = cf.get_cf_specs().coords["depth"]
    assert specs["alt_names"][0] == "dep"
    assert specs["attrs"]["long_name"][0] == "Depth"


@pytest.mark.parametrize(
    "cfg,key,name",
    [
        ({"data_vars": {"temp": {"alt_names": "mytemp"}}}, "temp", "mytemp"),
        ("[data_vars]\n[[sal]]\nalt_names=mysal", "sal", "mysal"),
    ],
)
def test_cf_cfspecs_load_cfg(cfg, key, name):
    cfspecs = cf.get_cf_specs()
    cfspecs.load_cfg(cfg)
    assert name in cfspecs["data_vars"][key]["alt_names"]


def test_cf_cfspecs_copy():
    cfspecs0 = cf.get_cf_specs()
    cfspecs1 = cfspecs0.copy()
    assert id(cfspecs0._dict) != id(cfspecs1._dict)
    assert sorted(list(cfspecs0._dict["data_vars"])) == sorted(list(cfspecs1._dict["data_vars"]))
    assert cfspecs0._dict["coords"] == cfspecs1._dict["coords"]
    assert cfspecs0._dict["data_vars"]["temp"] == cfspecs1._dict["data_vars"]["temp"]
    assert "temp" in cfspecs1["data_vars"]
    assert "temperature" in cfspecs1["data_vars"]["temp"]["alt_names"]


def test_cf_set_cf_specs():
    cf.reset_cache(disk=False)
    cfspecs = cf.get_cf_specs()
    cf.set_cf_specs(cfspecs)
    cf_cache = cf._get_cache_()
    assert cf_cache["current"] is cfspecs
    assert cf.get_cf_specs() is cfspecs


def test_cf_set_cf_specs_context():
    cfspecs0 = cf.get_cf_specs()
    cfspecs1 = cf.CFSpecs({"data_vars": {"temp": {"alt_names": "tempouille"}}})
    assert cf.get_cf_specs() is cfspecs0
    with cf.set_cf_specs(cfspecs1) as cfspecs:
        assert cfspecs is cfspecs1
        assert cf.get_cf_specs() is cfspecs1
    assert cf.get_cf_specs() is cfspecs0


@pytest.mark.parametrize("specialize,expected", [(False, "temp"), (True, "temperature")])
def test_cf_cfspecs_get_name(specialize, expected):
    cfspecs = cf.CFSpecs({"data_vars": {"temp": {"name": "temperature"}}})
    assert cfspecs.data_vars.get_name("temp", specialize=specialize) == expected


def test_cf_cfspecs_get_attrs():
    cfspecs = cf.get_cf_specs()
    attrs = cfspecs.data_vars.get_attrs("temp", other="ok")
    assert attrs["long_name"] == "Temperature"
    assert attrs["other"] == "ok"


def test_cf_cfspecs_get_loc_mapping():

    cf_dict0 = {
        "sglocator": {
            "valid_locations": ["u", "v"],
        },
        "data_vars": {
            "u": {
                "loc": "u",
                "add_loc": False,
                "add_coords_loc": {"lon": True, "x": True},
            },
            "bathy": {"add_loc": True},
        },
    }
    cf_specs0 = cf.CFSpecs(cf_dict0)

    ds0 = xr.Dataset(
        {"u": (("time", "y", "x"), np.ones((1, 2, 3))), "bathy": (("y", "x"), np.ones((2, 3)))},
        coords={
            "lon": (("y", "x"), np.ones((2, 3))),
            "lat": (("y", "x"), np.ones((2, 3))),
            "time": ("time", [1]),
        },
    )

    locations = cf_specs0.get_loc_mapping(ds0)
    assert locations == {'u': False, 'lon': 'u', 'x': 'u', 'bathy': 'u', 'lat': None}


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
    temp = xr.DataArray(range(20, 25), dims=in_name, coords={in_name: lon}, name='temp')
    res = cf.get_cf_specs().search_coord(temp, cf_name, get="cf_name")
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
    temp = xr.DataArray(
        range(20, 25), dims='lon', coords={'lon': lon}, name=in_name, attrs=in_attrs
    )
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
    temp = xr.DataArray(
        range(20, 25), dims='lon', coords={'lon': lon}, name=in_name, attrs=in_attrs
    )
    ds = temp.to_dataset()
    assert cf.get_cf_specs().search_data_var(ds, cf_name, get="cf_name") == 'temp'


def test_cf_cfspecs_cats_get_loc_arg():

    cf_dict0 = {
        "sglocator": {
            "valid_locations": ["u", "v"],
        },
        "data_vars": {
            "u": {
                "loc": "u",
                "add_loc": False,
                "add_coords_loc": {"lon": True},
            },
        },
    }
    cf_specs0 = cf.CFSpecs(cf_dict0)

    ds0 = xr.Dataset(
        {"u": (("time", "y", "x"), np.ones((1, 2, 3))), "bathy": (("y", "x"), np.ones((2, 3)))},
        coords={
            "lon": (("y", "x"), np.ones((2, 3))),
            "lat": (("y", "x"), np.ones((2, 3))),
            "time": ("time", [1]),
        },
    )

    assert cf_specs0.coords.get_loc_arg(ds0["u"]) is None
    assert cf_specs0.coords.get_loc_arg(ds0["bathy"]) is None
    assert cf_specs0.coords.get_loc_arg(ds0["lon"]) is None

    locations0 = cf_specs0.get_loc_mapping(ds0)
    assert cf_specs0.coords.get_loc_arg(ds0["lon"], locations=locations0) == "u"

    cf_dict1 = cf_dict0.copy()
    cf_dict1["data_vars"]["u"]["add_coords_loc"]["lon"] = "v"
    cf_specs1 = cf.CFSpecs(cf_dict1)
    locations1 = cf_specs1.get_loc_mapping(ds0)
    assert cf_specs1.coords.get_loc_arg(ds0["lon"], locations=locations1) == "v"


@pytest.mark.parametrize("cf_name", [None, "lon"])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("lon", None),
        ("xxx", {"standard_name": "longitude"}),
        ("xxx", {"units": "degrees_east"}),
    ],
)
def test_cf_cfspecs_cats_format_dataarray(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims=in_name, name=in_name, attrs=in_attrs)
    lon = cf.get_cf_specs().coords.format_dataarray(lon, cf_name)
    assert lon.name == "lon"
    assert lon.standard_name == "longitude"
    assert lon.long_name == "Longitude"
    assert lon.units == "degrees_east"


def test_cf_cfspecs_cats_format_dataarray_unknown():
    coord = xr.DataArray(range(5), name='foo')
    cfspecs = cf.get_cf_specs()

    coord_fmt = cfspecs.coords.format_dataarray(coord, rename=False)
    assert coord_fmt is None

    coord_fmt = cfspecs.coords.format_dataarray(coord, rename=True)
    assert coord_fmt.name == "foo"


def test_cf_cfspecs_cats_get_allowed_names():
    cfg = {"data_vars": {"banana": {"name": "bonono", "alt_names": ["binini", "bununu"]}}}
    cfspecs = cf.CFSpecs(cfg)
    assert cfspecs.data_vars.get_allowed_names("banana") == ['banana', 'bonono', 'binini', 'bununu']


def test_cf_cfspecs_format_obj_with_loc():

    cf_dict0 = {
        "sglocator": {
            "valid_locations": ["u", "v"],
        },
        "data_vars": {
            "u": {
                "loc": "u",
                "add_loc": False,
                "add_coords_loc": {"lon": True, "x": True},
            },
            "bathy": {"add_loc": True},
        },
    }
    cf_specs0 = cf.CFSpecs(cf_dict0)

    ds0 = xr.Dataset(
        {"u": (("time", "y", "x"), np.ones((1, 2, 3))), "bathy": (("y", "x"), np.ones((2, 3)))},
        coords={
            "lon": (("y", "x"), np.ones((2, 3))),
            "lat": (("y", "x"), np.ones((2, 3))),
            "time": ("time", [1]),
        },
    )
    ds = cf_specs0.format_dataset(ds0)
    assert "x_u" in ds.dims
    assert "y" in ds.dims
    assert "u" in ds
    assert "bathy_u" in ds


@pytest.mark.parametrize("cf_name", [None, "temp"])
@pytest.mark.parametrize(
    "in_name,in_attrs",
    [
        ("temp", None),
        ("yyy", {"standard_name": "sea_water_temperature"}),
    ],
)
def test_cf_cfspecs_format_data_var(cf_name, in_name, in_attrs):

    lon = xr.DataArray(range(5), dims='xxx', name='xxx', attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(
        range(20, 25), dims='xxx', coords={'xxx': lon}, name=in_name, attrs=in_attrs
    )
    temp = cf.get_cf_specs().format_data_var(temp, cf_name)
    assert temp.name == "temp"
    assert temp.standard_name == "sea_water_temperature"
    assert temp.long_name == "Temperature"
    assert temp.units == "degrees_celsius"
    assert temp.lon.standard_name == "longitude"


def test_cf_cfspecs_format_data_var_coord():
    da = xr.DataArray(0, attrs={'standard_name': 'longitude_at_u_location'})
    da = cf.get_cf_specs().format_data_var(da)


#     assert da.name == "lon_u"


def test_cf_cfspecs_format_data_var_specialize():

    da = xr.DataArray(1, name="salinity")
    cfspecs = cf.CFSpecs({'data_vars': {'sal': {'name': 'supersal'}}})
    da = cfspecs.format_data_var(da, specialize=True)
    assert da.name == "supersal"
    assert da.standard_name == "sea_water_salinity"


def test_cf_cfspecs_format_data_var_loc():
    temp = xr.DataArray(0, name='xtemp', attrs={'standard_name': 'banana_at_x_location'})
    cfspecs = cf.get_cf_specs()

    temp_fmt = cfspecs.format_data_var(temp, "temp", format_coords=False, replace_attrs=True)
    assert temp_fmt.name == "temp"
    assert temp_fmt.standard_name == "sea_water_temperature"  #_at_x_location"

    cfspecs = cf.CFSpecs({"data_vars": {"temp": {"add_loc": True}}})
    temp_fmt = cfspecs.format_data_var(temp, "temp", format_coords=False, replace_attrs=True)
    assert temp_fmt.name == "temp_x"


def test_cf_cfspecs_format_data_var_unkown():
    da = xr.DataArray(range(5), name='foo')
    cfspecs = cf.get_cf_specs()

    da_fmt = cfspecs.format_data_var(da, rename=False)
    assert da_fmt.name == "foo"

    da_fmt = cfspecs.format_data_var(da, rename=True)
    assert da_fmt.name == "foo"


def test_cf_cfspecs_to_loc():
    ds = xr.Dataset(
        {"u_t": (("time", "y", "x_u"), np.ones((1, 2, 3)))},
        coords={
            "lon_u": (("y", "x_u"), np.ones((2, 3))),
            "lat": (("y", "x_u"), np.ones((2, 3))),
            "time": ("time", [1]),
        },
    )
    cfspecs = cf.get_cf_specs()
    dso = cfspecs.to_loc(ds, x=False, y='v', u=None)
    assert "x" in dso.dims
    assert "y_v" in dso.dims
    assert "u_t" in dso
    assert "lon_u" in dso

def test_cf_cfspecs_reloc():
    ds = xr.Dataset(
        {"u_t": (("time", "y", "x_u"), np.ones((1, 2, 3)))},
        coords={
            "lon_u": (("y", "x_u"), np.ones((2, 3))),
            "lat": (("y", "x_u"), np.ones((2, 3))),
            "time": ("time", [1]),
        },
    )
    cfspecs = cf.get_cf_specs()
    dso = cfspecs.reloc(ds, u=False, t='u')
    assert "x" in dso.dims
    assert "u_u" in dso
    assert "lon" in dso

def test_cf_cfspecs_coords_get_axis():
    cfspecs = cf.get_cf_specs().coords

    # from attrs
    depth = xr.DataArray([1], dims='aa', attrs={'axis': 'z'})
    assert cfspecs.get_axis(depth) == 'Z'

    # from CF specs
    depth = xr.DataArray([1], dims='aa', attrs={'standard_name': 'ocean_layer_depth'})
    assert cfspecs.get_axis(depth) == 'Z'


def test_cf_cfspecs_coords_get_dim_type():
    cfspecs = cf.get_cf_specs().coords

    # from name
    assert cfspecs.get_dim_type('aa') is None
    assert cfspecs.get_dim_type('xi') == "x"

    # from a known coordinate
    coord = xr.DataArray([1], dims='aa', attrs={'standard_name': 'longitude'})
    da = xr.DataArray([1], dims='aa', coords={'aa': coord})
    assert cfspecs.get_dim_type('aa', da) == "x"


def test_cf_cfspecs_coords_get_dim_types():
    cfspecs = cf.get_cf_specs().coords

    aa = xr.DataArray([0, 1], dims="aa", attrs={"standard_name": "latitude"})
    da = xr.DataArray(np.ones((2, 2, 2)), dims=('foo', 'aa', 'xi'), coords={'aa': aa})

    assert cfspecs.get_dim_types(da) == (None, 'y', 'x')
    assert cfspecs.get_dim_types(da, unknown='-') == ("-", 'y', 'x')
    assert cfspecs.get_dim_types(da, asdict=True) == {"foo": None, "aa": "y", "xi": "x"}


def test_cf_cfspecs_coords_search_dim():
    cfspecs = cf.get_cf_specs().coords

    # from name
    temp = xr.DataArray(np.arange(2 * 3).reshape(1, 2, 3), dims=('aa', 'ny', 'x'))
    assert cfspecs.search_dim(temp, 'y') == 'ny'
    assert cfspecs.search_dim(temp) is None

    # from explicit axis attribute
    depth = xr.DataArray([1], dims='aa', attrs={'axis': 'z'})
    temp.coords['aa'] = depth
    assert cfspecs.search_dim(temp, 'z') == 'aa'
    assert cfspecs.search_dim(temp) is None
    assert cfspecs.search_dim(depth) == {"dim": "aa", "type": "z", "cf_name": None}

    # from known coordinate
    del temp.coords['aa'].attrs['axis']
    temp.coords['aa'].attrs['standard_name'] = 'ocean_layer_depth'
    assert cfspecs.search_dim(temp, 'z') == 'aa'
    assert cfspecs.search_dim(temp, 'depth') == 'aa'  # by generic name
    assert cfspecs.search_dim(temp) is None

    # subcoords
    level = xr.DataArray(np.arange(2), dims='level')
    depth = xr.DataArray(
        np.arange(2 * 3).reshape(2, 3),
        dims=('level', 'lon'),
        name='depth',
        coords={'level': level, 'lon': [3, 4, 5]},
    )
    assert cfspecs.search_dim(depth) == {'dim': 'level', 'cf_name': 'level', 'type': 'z'}
    assert cfspecs.search_dim(depth.level) == {'dim': 'level', 'type': 'z', 'cf_name': 'level'}
    depth = depth.rename(level='aa')
    depth.aa.attrs['axis'] = 'Z'
    assert cfspecs.search_dim(depth) == {'dim': 'aa', 'type': 'z', 'cf_name': None}

    # not found but only 1d and no dim_type specified
    assert cfspecs.search_dim(xr.DataArray([5], dims='bb')) == {
        'dim': 'bb',
        'type': None,
        'cf_name': None,
    }


def test_cf_cfspecs_coords_search_from_dim():

    lon = xr.DataArray([1, 2], dims='lon')
    level = xr.DataArray([1, 2, 3], dims='aa', attrs={'standard_name': 'ocean_sigma_coordinate'})
    mem = xr.DataArray(range(3), dims='mem')
    temp = xr.DataArray(
        np.zeros((mem.size, level.size, lon.size)),
        dims=('mem', 'aa', 'lon'),
        coords={'mem': mem, 'aa': level, 'lon': lon},
    )

    cfspecs = cf.get_cf_specs().coords

    # Direct coordinate
    assert cfspecs.search_from_dim(temp, 'aa').name == 'aa'

    # Depth coordinate because the only one with this dim
    depth = xr.DataArray(
        np.ones((level.size, lon.size)), dims=('aa', 'lon'), coords={'aa': level, 'lon': lon}
    )
    temp.coords['depth'] = depth
    assert cfspecs.search_from_dim(temp, 'aa').name == 'depth'

    # Cannot get dim_type and we have multiple coords with this dim
    del temp.coords['aa']
    fake = xr.DataArray(np.ones((level.size, lon.size)), dims=('aa', 'lon'), coords={'lon': lon})
    temp.coords['fake'] = fake
    assert cfspecs.search_from_dim(temp, 'aa') is None

    # Can get dim_type back from name
    temp = temp.rename(aa='level')
    assert cfspecs.search_from_dim(temp, 'level').name == 'depth'

    # Nothing identifiable
    temp = xr.DataArray([3], dims='banana')
    assert cfspecs.search_from_dim(temp, "banana") is None


def test_cf_cfspecs_coords_get_dims():
    lat = xr.DataArray([4, 5], dims='yy', attrs={'units': 'degrees_north'})
    depth = xr.DataArray([4, 5], dims='level', attrs={'axis': 'Z'})
    da = xr.DataArray(
        np.ones((2, 2, 2, 2)), dims=('r', 'level', 'yy', 'xi'), coords={'level': depth, 'yy': lat}
    )

    cfspecs = cf.get_cf_specs().coords
    dims = cfspecs.get_dims(da, 'xyzt', allow_positional=True)
    assert dims == ('xi', 'yy', 'level', 'r')
    dims = cfspecs.get_dims(da, 'f', errors="ignore")
    assert dims == (None,)


def test_cf_cfspecs_infer_coords():
    ds = xr.Dataset({"temp": ("nx", [1, 2]), "lon": ("nx", [4, 5])})
    ds = cf.get_cf_specs().infer_coords(ds)
    assert "lon" in ds.coords


def test_cf_cfspecs_decode_encode():
    ds = xoa.open_data_sample("croco.south-africa.meridional.nc")
    cfspecs = cf.CFSpecs(xoa.get_data_sample("croco.cfg"))

    dsc = cfspecs.decode(ds)
    assert list(dsc) == [
        'akt',
        'cs_r',
        'cs_w',
        'Vtransform',
        'angle',
        'el',
        'corio',
        'bathy',
        'hbl',
        'hc',
        'mask_rho',
        'ex',
        'ey',
        'sal',
        'sc_r',
        'sc_w',
        'ptemp',
        'time_step',
        'u',
        'v',
        'w',
        'xl',
        'ssh',
    ]
    assert list(dsc.coords) == [
        'y_rho',
        'y_v',
        'lat_rho',
        'lat_u',
        'lat_v',
        'lon_rho',
        'lon_u',
        'lon_v',
        'sig_rho',
        'sig_w',
        'time',
        'x_rho',
        'x_u',
    ]
    assert set(dsc.dims) == {'auxil', 'sig_rho', 'sig_w', 'time', 'x_rho', 'x_u', 'y_rho', 'y_v'}

    dse = cfspecs.encode(dsc)
    assert list(dse) == list(ds)
    assert list(dse.coords) == list(ds.coords)
    assert list(dse.dims) == list(ds.dims)
    ds.close()


def test_cf_dataarraycfaccessor():
    from xoa.accessors import CFDataArrayAccessor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xr.core.extensions.AccessorRegistrationWarning)
        xr.register_dataarray_accessor('xcf')(CFDataArrayAccessor)

    lon = xr.DataArray(range(5), dims='xxx', name='xxx', attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(range(20, 25), dims='xxx', coords={'xxx': lon}, name='temp')

    assert temp.xcf.lon.name == 'xxx'
    assert temp.xcf.lat is None
    assert temp.xcf.lon.xcf.name == "lon"


def test_cf_datasetcfaccessor():
    from xoa.accessors import CFDatasetAccessor

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", xr.core.extensions.AccessorRegistrationWarning)
        xr.register_dataset_accessor('xcf')(CFDatasetAccessor)

    lon = xr.DataArray(range(5), dims='xxx', name='xxx', attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(
        range(20, 25),
        dims='xxx',
        coords={'xxx': lon},
        name='yoyo',
        attrs={'standard_name': 'sea_water_temperature'},
    )

    ds = temp.to_dataset()
    assert ds.xcf.temp.name == 'yoyo'
    assert ds.xcf.sal is None


def test_cf_register_cf_specs():

    cf_cache = cf._get_cache_()
    cf_cache["registered"].clear()

    content = """
        [register]
        name=myname

        [data_vars]
            [[temp]]
            name=mytemp
        """

    cf_specs = cf.CFSpecs(content)
    assert cf_specs.name == "myname"

    cf.register_cf_specs(cf_specs)
    assert cf_specs in cf_cache["registered"]
    assert cf_specs.name == "myname"

    cf.register_cf_specs(myothername=cf_specs)
    assert cf_specs in cf_cache["registered"]
    assert cf_specs.name == "myothername"


def test_cf_get_cf_specs_registered():

    cf_cache = cf._get_cache_()
    cf_cache["registered"].clear()
    content = """
        [register]
        name=myname

        [data_vars]
            [[temp]]
            name=mytemp
        """
    cf_specs_in = cf.CFSpecs(content)
    cf.register_cf_specs(cf_specs_in)

    cf_specs_out = cf.get_cf_specs(name='myname')
    assert cf_specs_out is cf_specs_in


def test_cf_get_cf_specs_from_encoding():

    cf_cache = cf._get_cache_()
    cf_cache["registered"].clear()
    content = """
        [register]
        name=mynam234

        [data_vars]
            [[temp]]
            name=mytemp
        """
    cf_specs_in = cf.CFSpecs(content)
    cf.register_cf_specs(cf_specs_in)

    ds = xr.Dataset(
        {
            "mytemp": (["mylat", "mylon"], np.ones((2, 2))),
            "mysal": (["mylat", "mylon"], np.ones((2, 2))),
        },
        coords={"mylon": np.arange(2), "mylat": np.arange(2)},
    )

    ds.encoding.update(cf_specs="mynam234")
    assert cf.get_cf_specs_from_encoding(ds) is cf_specs_in

    ds.mytemp.encoding.update(cf_specs="mynam234")
    assert cf.get_cf_specs_from_encoding(ds.mytemp) is cf_specs_in

    ds.mylon.encoding.update(cf_specs="mynam234")
    assert cf.get_cf_specs_from_encoding(ds.mylon) is cf_specs_in

    assert cf.get_cf_specs_from_encoding(ds.mylat) is None


def test_cf_set_cf_specs_registered():

    cf_cache = cf._get_cache_()
    cf_cache["registered"].clear()
    content = """
        [register]
        name=myname2

        [data_vars]
            [[temp]]
            name=mytemp
        """
    cf_specs_in = cf.CFSpecs(content)
    cf.register_cf_specs(cf_specs_in)

    with cf.set_cf_specs("myname2") as cfspecs:
        assert cfspecs is cf_specs_in


def test_cf_get_cf_specs_matching_score():

    cf_content0 = """
        [data_vars]
            [[temp]]
            name=mytemp
        """
    cf_specs0 = cf.CFSpecs(cf_content0)
    cf_content1 = """
        [data_vars]
            [[temp]]
            name=mytemp
            [[sal]]
            name=mysal
        [coords]
            [[lon]]
            name=mylon
        """
    cf_specs1 = cf.CFSpecs(cf_content1)
    cf_content2 = """
        [data_vars]
            [[temp]]
            name=mytemp
            [[sal]]
            name=mysal
        """
    cf_specs2 = cf.CFSpecs(cf_content2)

    ds = xr.Dataset(
        {
            "mytemp": (["mylat", "mylon"], np.ones((2, 2))),
            "mysal": (["mylat", "mylon"], np.ones((2, 2))),
        },
        coords={"mylon": np.arange(2), "mylat": np.arange(2)},
    )

    for cf_specs, score in [(cf_specs0, 25), (cf_specs1, 75), (cf_specs2, 50)]:
        assert cf.get_cf_specs_matching_score(ds, cf_specs) == score


def test_cf_infer_cf_specs():

    cf_content0 = """
        [register]
            [[attrs]]
            source="*hycom3d*"

        [data_vars]
            [[temp]]
            name=mytemp
        """
    cf_specs0 = cf.CFSpecs(cf_content0)
    cf_content1 = """
        [data_vars]
            [[temp]]
            name=mytemp
            [[sal]]
            name=mysal
        [coords]
            [[lon]]
            name=mylon
        """
    cf_specs1 = cf.CFSpecs(cf_content1)
    cf_content2 = """
        [register]
        name=hycom3d

        [data_vars]
            [[temp]]
            name=mytemp
            [[sal]]
            name=mysal
        """
    cf_specs2 = cf.CFSpecs(cf_content2)

    cf_cache = cf._get_cache_()
    cf_cache["registered"].clear()
    cf.register_cf_specs(cf_specs0, cf_specs1, cf_specs2)

    temp = xr.DataArray([1], dims="mylon")
    sal = xr.DataArray([1], dims="mylon")
    lon = xr.DataArray([1], dims="mylon")

    ds = xr.Dataset({"mytemp": temp, "mysal": sal}, coords={"mylon": lon})
    assert cf.infer_cf_specs(ds) is cf_specs1

    ds.attrs.update(source="my hycom3d!")
    assert cf.infer_cf_specs(ds) is cf_specs0

    ds.attrs.update(cf_specs="hycom3d")
    assert cf.infer_cf_specs(ds) is cf_specs2


# test_cf_cfspecs_decode_encode()
# test_cf_cfspecs_format_data_var_loc()
# test_cf_cfspecs_coords_get_loc_arg()
# test_cf_cfspecs_format_obj_with_loc()
# test_cf_cfspecs_get_loc_mapping()
# test_cf_cfspecs_coords_search_dim()

# lon = xr.DataArray(range(5), dims="lon")
# banana = xr.DataArray(
#     lon + 20,
#     dims="lon",
#     coords=[lon],
#     name="banana_t",
#     attrs={"standard_name": "banana", "taste": "good"},
# )
# floc, fname, fattrs, out_name, out_standard_name, replace_attrs=None, "sst_q", {"standard_name": ["potatoe"]}, "sst_q", "potatoe_at_q_location", True
# print(cf.SGLocator().format_dataarray(
#     banana, loc=floc, name=fname, attrs=fattrs, replace_attrs=replace_attrs
# ))

# sg = cf.SGLocator()
# print(sg.get_loc(name="u_t", attrs=dict(standard_name = None, long_name = None)))

# test_cf_cfspecs_decode_encode()