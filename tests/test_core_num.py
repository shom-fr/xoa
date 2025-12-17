# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.num` module
"""

import numpy as np

from xoa.core import num


class TestArrayIndexing:
    """Test array indexing utilities"""

    def test_get_iminmax(self):
        """Test finding first and last valid values in 1D array"""
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, np.nan])
        imin, imax = num.get_iminmax(data)
        assert imin == 2
        assert imax == 4

    def test_get_iminmax_all_nan(self):
        """Test with all NaN values"""
        data = np.array([np.nan, np.nan, np.nan])
        imin, imax = num.get_iminmax(data)
        assert imin == -1
        assert imax == -1

    def test_get_iminmax_no_nan(self):
        """Test with no NaN values"""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        imin, imax = num.get_iminmax(data)
        assert imin == 0
        assert imax == 3


class TestIndexConversion:
    """Test multi-dimensional index conversion utilities"""

    def test_unravel_index(self):
        """Test converting flat index to multi-dimensional"""
        shape = np.array([3, 4, 5], dtype=np.int64)
        i = 27
        ii = num.unravel_index(i, shape)
        expected = np.array([1, 1, 2], dtype=np.int64)
        np.testing.assert_array_equal(ii, expected)

    def test_ravel_index(self):
        """Test converting multi-dimensional index to flat"""
        shape = np.array([3, 4, 5], dtype=np.int64)
        ii = np.array([1, 1, 2], dtype=np.int64)
        i = num.ravel_index(ii, shape)
        assert i == 27

    def test_ravel_unravel_roundtrip(self):
        """Test that ravel and unravel are inverse operations"""
        shape = np.array([3, 4, 5], dtype=np.int64)
        for i in [0, 10, 27, 59]:
            ii = num.unravel_index(i, shape)
            i_back = num.ravel_index(ii, shape)
            assert i == i_back
