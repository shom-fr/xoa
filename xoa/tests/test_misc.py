# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.misc` module
"""
import re

import pytest
import numpy as np

from xoa import misc


@pytest.mark.parametrize(
    "ndim", [np.arange(2*3*4).reshape(2, 3, 4), 3])
@pytest.mark.parametrize(
    "axis", [0, 1, 2])
def test_misc_get_axis_slices(ndim, axis):
    ss = misc.get_axis_slices(ndim, axis, top=slice(-1, None))
    assert ss['mid'][(axis+1) % 3] == slice(None)
    assert ss['mid'][axis] == slice(1, -1)
    assert ss['firsts'][axis] == slice(0, -1)
    assert ss['lastm2'][axis] == -3
    assert ss['top'][axis] == slice(-1, None)


@pytest.mark.parametrize(
    "obj,expected",
    [([], True), ((), True), ("", False), ({}, True), ({"d": 1}, True)],
)
def test_misc_is_iterable(obj, expected):
    assert misc.is_iterable(obj) is expected


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
def test_misc_match_string(ss, checks, expected):
    assert misc.match_string(ss, checks) is expected
