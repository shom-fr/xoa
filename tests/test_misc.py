# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.misc` module
"""
import re

import pytest
import numpy as np

from xoa import misc


class TestArrayUtilities:
    """Test array manipulation utilities"""

    @pytest.mark.parametrize(
        "ndim", [np.arange(2*3*4).reshape(2, 3, 4), 3])
    @pytest.mark.parametrize(
        "axis", [0, 1, 2])
    def test_get_axis_slices(self, ndim, axis):
        ss = misc.get_axis_slices(ndim, axis, top=slice(-1, None))
        assert ss['mid'][(axis+1) % 3] == slice(None)
        assert ss['mid'][axis] == slice(1, -1)
        assert ss['firsts'][axis] == slice(0, -1)
        assert ss['lastm2'][axis] == -3
        assert ss['top'][axis] == slice(-1, None)


class TestTypeChecking:
    """Test type checking utilities"""

    @pytest.mark.parametrize(
        "obj,expected",
        [([], True), ((), True), ("", False), ({}, True), ({"d": 1}, True)],
    )
    def test_is_iterable(self, obj, expected):
        assert misc.is_iterable(obj) is expected


class TestStringMatching:
    """Test string matching utilities"""

    @pytest.mark.parametrize(
        "ss,checks,expected",
        [
            ("sst", "sst", True),
            ("sst", ["xxx", "sst"], True),
            ("sst", ["xxx", "yyy"], False),
            ("sst", [re.compile(r"ss.$").match], True),
            ("xst", [re.compile(r"ss.$").match], False),
            ("sst", "sss", False),
        ],
    )
    def test_match_string(self, ss, checks, expected):
        assert misc.match_string(ss, checks) is expected


class TestChoices:
    """Test Choices class"""

    def test_choices(self):
        choices = misc.Choices(['a', 'bb', 'c'])
        assert choices['a'] == 'a'
        assert choices['C'] == 'c'


class TestDictOperations:
    """Test dictionary manipulation utilities"""

    def test_dict_merge(self):
        dict0 = {'inherit': 'ptemp', 'name': ['temperature'], 'domain': 'generic',
                 'cmap': None, 'squeeze': None, 'search_order': 'sn',
                 'attrs': {'standard_name': ['sea_water_temperature'],
                           'long_name': ['Temperature'],
                           'units': []},
                 'select': {}, "mytup": ('aa',)}
        dict1 = {'cmap': 'cmo.thermal', 'name': [], 'domain': 'generic',
                 'inherit': None, 'squeeze': None, 'search_order': 'sn',
                 'attrs': {'standard_name': ['sea_water_potential_temperature'],
                           'long_name': ['Potential temperature'],
                           'units': ['degrees_celsius']},
                 'select': {}, 'processed': True, "mytup": ('bb',)}

        expected = {'inherit': 'ptemp', 'name': ['temperature'],
                    'domain': 'generic', 'cmap': 'cmo.thermal',
                    'squeeze': None, 'search_order': 'sn',
                    'attrs': {'standard_name': ['sea_water_temperature',
                                                'sea_water_potential_temperature'
                                                ],
                              'long_name': ['Temperature',
                                            'Potential temperature'
                                            ],
                              'units': ['degrees_celsius']},
                    'select': {}, 'mytup': ('aa', 'bb'), 'processed': True}

        dict01 = misc.dict_merge(
            dict0, dict1,
            mergesubdicts=True,
            mergelists=True,
            mergetuples=True,
            skipnones=False,
            overwriteempty=True,
            uniquify=False)

        assert dict01 == expected


class TestEnums:
    """Test enum utilities"""

    def test_intenum_defaultenumeta(self):
        class regrid_methods(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
            linear = 1
            bilinear = 1
            nearest = 0
            cellave = -1

        assert regrid_methods().name == "linear"  # default method
        assert regrid_methods(None).name == "linear"  # default method
        assert regrid_methods(1).name == "linear"
        assert regrid_methods[None].name == "linear"  # default method
        assert regrid_methods['linear'].name == "linear"
        assert regrid_methods['cellave'].name == "cellave"
