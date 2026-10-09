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

    @pytest.mark.parametrize("ndim", [np.arange(2 * 3 * 4).reshape(2, 3, 4), 3])
    @pytest.mark.parametrize("axis", [0, 1, 2])
    def test_get_axis_slices(self, ndim, axis):
        ss = misc.get_axis_slices(ndim, axis, top=slice(-1, None))
        assert ss['mid'][(axis + 1) % 3] == slice(None)
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


def test_choices():
    """Test Choices class"""
    choices = misc.Choices(['a', 'bb', 'c'])
    assert choices['a'] == 'a'
    assert choices['C'] == 'c'


class TestDictOperations:
    """Test dictionary manipulation utilities"""

    def test_dict_merge(self):
        dict0 = {
            'inherit': 'ptemp',
            'name': ['temperature'],
            'domain': 'generic',
            'cmap': None,
            'squeeze': None,
            'search_order': 'sn',
            'attrs': {
                'standard_name': ['sea_water_temperature'],
                'long_name': ['Temperature'],
                'units': [],
            },
            'select': {},
            "mytup": ('aa',),
        }
        dict1 = {
            'cmap': 'cmo.thermal',
            'name': [],
            'domain': 'generic',
            'inherit': None,
            'squeeze': None,
            'search_order': 'sn',
            'attrs': {
                'standard_name': ['sea_water_potential_temperature'],
                'long_name': ['Potential temperature'],
                'units': ['degrees_celsius'],
            },
            'select': {},
            'processed': True,
            "mytup": ('bb',),
        }

        expected = {
            'inherit': 'ptemp',
            'name': ['temperature'],
            'domain': 'generic',
            'cmap': 'cmo.thermal',
            'squeeze': None,
            'search_order': 'sn',
            'attrs': {
                'standard_name': ['sea_water_temperature', 'sea_water_potential_temperature'],
                'long_name': ['Temperature', 'Potential temperature'],
                'units': ['degrees_celsius'],
            },
            'select': {},
            'mytup': ('aa', 'bb'),
            'processed': True,
        }

        dict01 = misc.dict_merge(
            dict0,
            dict1,
            mergesubdicts=True,
            mergelists=True,
            mergetuples=True,
            skipnones=False,
            overwriteempty=True,
            uniquify=False,
        )

        assert dict01 == expected


def test_intenum_defaultenumeta():
    """Test enum utilities"""

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


class TestArrayFingerprint:
    def test_same_content_same_fingerprint(self):
        a = np.arange(12.0).reshape(3, 4)
        assert misc.get_array_fingerprint(a) == misc.get_array_fingerprint(a.copy())
        assert misc.get_array_fingerprint(a, None) == misc.get_array_fingerprint(a.copy(), None)
        # Non contiguous arrays are handled
        assert misc.get_array_fingerprint(a.T) == misc.get_array_fingerprint(a.T.copy())

    def test_different_arrays(self):
        a = np.arange(12.0).reshape(3, 4)
        b = a.copy()
        b[1, 1] += 1e-9
        assert misc.get_array_fingerprint(a) != misc.get_array_fingerprint(b)
        assert misc.get_array_fingerprint(a) != misc.get_array_fingerprint(a.reshape(4, 3))
        assert misc.get_array_fingerprint(a) != misc.get_array_fingerprint(a.astype("f4"))
        assert misc.get_array_fingerprint(a, None) != misc.get_array_fingerprint(None, a)
        assert misc.get_array_fingerprint(a, a) != misc.get_array_fingerprint(a)

    def test_empty_and_bool_arrays(self):
        assert isinstance(misc.get_array_fingerprint(np.empty((0, 3)), np.ones(3, bool)), str)


class TestSmallCache:
    def test_get_or_create(self):
        cache = misc.SmallCache(maxsize=2)
        calls = []

        def factory(value):
            calls.append(value)
            return value

        assert cache.get_or_create("a", lambda: factory(1)) == 1
        assert cache.get_or_create("a", lambda: factory(2)) == 1
        assert calls == [1]
        assert "a" in cache and len(cache) == 1

    def test_least_recently_used_is_forgotten(self):
        cache = misc.SmallCache(maxsize=2)
        cache.get_or_create("a", lambda: 1)
        cache.get_or_create("b", lambda: 2)
        cache.get_or_create("a", lambda: 0)  # a is now the most recent
        cache.get_or_create("c", lambda: 3)
        assert "a" in cache and "c" in cache and "b" not in cache

    def test_clear(self):
        cache = misc.SmallCache()
        cache.get_or_create("a", lambda: 1)
        cache.clear()
        assert len(cache) == 0


def test_combine_fingerprints():
    a = misc.get_array_fingerprint(np.arange(3.0))
    b = misc.get_array_fingerprint(np.arange(4.0))
    combined = misc.combine_fingerprints(a, b)
    assert isinstance(combined, str)
    assert combined == misc.combine_fingerprints(a, b)
    assert combined != misc.combine_fingerprints(b, a)
    assert combined != misc.combine_fingerprints(a, a)
    assert combined != a
