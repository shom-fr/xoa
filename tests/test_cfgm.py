# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.cfgm` module
"""
import pytest

from xoa import cfgm
from xoa.color import CmapAdapter


class TestConfigManager:
    specs = [
        "[logger]",
        "level=option(debug,info,error,default=info)",
        "[domain]",
        "xlim=floats(n=2, default=list(-20,0))",
        "[plot]",
        "cmap=cmap(default=viridis)",
    ]

    def test_defaults_and_load(self):
        cm = cfgm.ConfigManager(self.specs)
        defaults = cm.defaults
        assert defaults["logger"]["level"] == "info"
        assert defaults["domain"]["xlim"] == [-20.0, 0.0]
        cfg = cm.load(["[logger]", "level=error", "[domain]", "xlim=-10,0"])
        assert cfg["logger"]["level"] == "error" and cfg["domain"]["xlim"] == [-10.0, 0.0]

    def test_cmap(self):
        cm = cfgm.ConfigManager(self.specs)
        assert isinstance(cm.defaults["plot"]["cmap"], CmapAdapter)
        assert cm.defaults["plot"]["cmap"].cmap.name == "viridis"

    def test_invalid_value(self):
        cm = cfgm.ConfigManager(self.specs)
        with pytest.raises(cfgm.XoaValidateError, match="level"):
            cm.load(["[logger]", "level=verbose"])

    def test_register_validation_function(self):
        cfgm.register_validation_functions(double=lambda value, default=None: 2 * float(value))
        assert cfgm.get_validator().check("double", "3") == 6.0
