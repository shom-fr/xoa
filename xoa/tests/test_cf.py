#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.cf` module
"""

import pytest

from xoa import cf


class TestObj(object):
    pass


@pytest.mark.parametrize(
    "attr,root,loc,expected",
    [('standard_name', 'my_var', 't', True),
     ('standard_name', 'my_var2', 't', False),
     ('standard_name', 'my_var', 'x', False),
     ('long_name', 'My var', 't', True),
     ('long_name', 'My var', 'x', False),
     ('name', 'myvar', 't', True),
     ('name', 'myvar', 'x', False)
     ])
def test_cf_sglocator_match(attr, root, loc, expected):

    obj = TestObj()
    obj.standard_name = 'my_var_at_t_location'
    obj.long_name = 'My var at T location'
    obj.name = 'myvar_t'

    return cf.SGLocator().match(obj, attr, root, loc)
