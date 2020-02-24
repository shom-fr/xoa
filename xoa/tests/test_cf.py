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
    [('standard_name', 'my_var_at_t_location', ('my_var', 't')),
     ('standard_name', 'my_var', ('my_var', None)),
     ('long_name', 'My var at T location', ('My var', 't')),
     ('name', 'myvar_t', ('myvar', 't')),
     ])
def test_cf_sglocator_parse_attr(attr, value, expected):
    assert cf.SGLocator().parse_attr(attr, value) == expected


@pytest.mark.parametrize(
    "attr,root,loc,expected",
    [('standard_name', 'my_var', 't', True),
     ('standard_name', 'my_var2', 't', False),
     ('standard_name', 'my_var', 'x', False),
     ('standard_name', 'my_var', None, True),
     ('standard_name', 'my_var', 'xtu', True),
     ('long_name', 'My var', 't', True),
     ('long_name', 'My var', 'x', False),
     ('name', 'myvar', 't', True),
     ('name', 'myvar', 'x', False)
     ])
def test_cf_sglocator_match_attr(attr, root, loc, expected):
    value = dict(standard_name='my_var_at_t_location',
                 long_name='My var at T location',
                 name='myvar_t')[attr]
    assert cf.SGLocator().match_attr(attr, value, root, loc) is expected


@pytest.mark.parametrize(
    "attr,root,loc,expected",
    [('standard_name', 'my_var', 't', 'my_var_at_t_location'),
     ('standard_name', 'my_var', '', 'my_var'),
     ('long_name', 'My var', 't', 'My var at T location'),
     ('name', 'myvar', 't', 'myvar_t'),
     ])
def test_cf_sglocator_format_attr(attr, root, loc, expected):
    assert cf.SGLocator().format_attr(attr, root, loc) == expected


@pytest.mark.parametrize(
    "cache", ['ignore', 'rw', 'rw', 'ignore', 'clean', 'rw'])
def test_cf_get_cfg_specs(cache):
    assert isinstance(cf.get_cf_specs(cache=cache), cf.CFSpecs)


def test_cf_get_cfg_specs_var():
    specs = cf.get_cf_specs('temp', 'variables')
    assert specs['name'][0] == 'temp'
    assert specs['standard_name'][0] == 'sea_water_temperature'
    assert specs['cmap'] == 'cmo.thermal'
    new_specs = cf.get_cf_specs('temp')
    assert new_specs is specs


def test_cf_get_cfg_specs_var_inherit():
    specs = cf.get_cf_specs('sst', 'variables')
    assert specs['standard_name'][0] == 'sea_surface_temperature'
    assert specs['units'][0] == 'degrees_celsius'
