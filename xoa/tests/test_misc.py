#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.misc` module
"""
import re

import pytest

from xoa import misc


@pytest.mark.parametrize(
    "obj,expected",
    [([], True),
     ((), True),
     ('', False),
     ({}, True),
     ({'d': 1}, True)])
def test_misc_is_iterable(obj, expected):
    assert misc.is_iterable(obj) is expected


@pytest.mark.parametrize(
    "ss,checks,expected",
    [("sst", "sst", True),
     ("sst", [re.compile(r'ss.$').match], True),
     ("sst", "sss", False)])
def test_misc_match_string(ss, checks, expected):
    assert misc.match_string(ss, checks) is expected
