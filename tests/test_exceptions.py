# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.exceptions` module
"""

import pytest
import warnings

from xoa import exceptions


class TestExceptions:
    """Test custom exception classes"""

    def test_xoa_error(self):
        with pytest.raises(exceptions.XoaError):
            raise exceptions.XoaError("test error")

    def test_xoa_config_error(self):
        with pytest.raises(exceptions.XoaConfigError):
            raise exceptions.XoaConfigError("config error")

    def test_exception_hierarchy(self):
        """Test that specific errors inherit from XoaError"""
        assert issubclass(exceptions.XoaConfigError, exceptions.XoaError)
        assert issubclass(exceptions.XoaCoordsError, exceptions.XoaError)
        assert issubclass(exceptions.XoaGridError, exceptions.XoaError)


class TestWarnings:
    """Test custom warning classes"""

    def test_xoa_warn(self):
        with pytest.warns(exceptions.XoaWarning):
            exceptions.xoa_warn("test warning")

    def test_xoa_warn_deprecation(self):
        with pytest.warns(exceptions.XoaDeprecationWarning):
            exceptions.xoa_warn("deprecation warning", category="deprecation")

    def test_warning_hierarchy(self):
        """Test that XoaWarning inherits from UserWarning"""
        assert issubclass(exceptions.XoaWarning, UserWarning)
        assert issubclass(exceptions.XoaDeprecationWarning, exceptions.XoaWarning)
        assert issubclass(exceptions.XoaDeprecationWarning, DeprecationWarning)
