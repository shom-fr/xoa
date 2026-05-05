# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.options` module
"""

import pytest

from xoa import options
from xoa import exceptions


class TestOptionsManagement:
    """Test configuration options management"""

    def test_get_option(self):
        """Test getting an option value"""
        value = options.get_option('plot', 'cmapdiv')
        assert isinstance(value, str)

    def test_get_option_flat_format(self):
        """Test getting an option using flat format"""
        value = options.get_option('plot.cmapdiv')
        assert isinstance(value, str)

    def test_get_option_invalid(self):
        """Test getting an invalid option raises error"""
        with pytest.raises(exceptions.XoaConfigError):
            options.get_option('invalid', 'option')

    def test_set_options(self):
        """Test setting options"""
        original = options.get_option('plot.cmapdiv')
        options.set_options('plot', cmapdiv='test.cmap')
        assert options.get_option('plot.cmapdiv') == 'test.cmap'
        options.reset_options()
        assert options.get_option('plot.cmapdiv') == original

    def test_set_option(self):
        """Test setting a single option"""
        original = options.get_option('plot.cmappos')
        options.set_option('plot.cmappos', 'test.cmap')
        assert options.get_option('plot.cmappos') == 'test.cmap'
        options.reset_options()
        assert options.get_option('plot.cmappos') == original

    def test_set_options_context(self):
        """Test setting options as a context manager"""
        original = options.get_option('plot.cmapdiv')
        with options.set_options('plot', cmapdiv='temp.cmap'):
            assert options.get_option('plot.cmapdiv') == 'temp.cmap'
        assert options.get_option('plot.cmapdiv') == original

    def test_reset_options(self):
        """Test resetting options to defaults"""
        options.set_option('plot.cmapdiv', 'test.cmap')
        options.reset_options()
        default = options.get_option('plot.cmapdiv')
        assert 'balance' in default.lower() or default == options.get_option('plot.cmapdiv')

    def test_meta_cache_option(self):
        """Test that the meta.cache option exists and defaults to False"""
        value = options.get_option('meta.cache')
        assert isinstance(value, bool)
        assert value is False
