# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.cfgm` module
"""
from argparse import ArgumentParser

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


class TestArgParse:
    specs = [
        "[logger]",
        "level=option(debug,info,error,default=info) # logging level",
    ]

    def _get_help_(self, option, capsys):
        parser = ArgumentParser(prog="demo")
        parser.add_argument("-v", "--verbose", action="store_true", help="be verbose")
        with pytest.raises(SystemExit):
            cfgm.ConfigManager(self.specs).arg_parse(parser, args=[option])
        return capsys.readouterr().out

    def test_parse(self):
        cfg = cfgm.ConfigManager(self.specs).arg_parse(args=["--logger-level", "error"])
        assert cfg["logger"]["level"] == "error"

    @pytest.mark.parametrize("option", ["-h", "--short-help"])
    def test_short_help(self, option, capsys):
        out = self._get_help_(option, capsys)
        assert "--verbose" in out and "--long-help" in out
        assert "logger:" not in out

    def test_long_help(self, capsys):
        out = self._get_help_("--long-help", capsys)
        assert "--verbose" in out and "logger:" in out and "--logger-level" in out

    def test_arg_long_help(self):
        cm = cfgm.ConfigManager(self.specs)
        assert ".. cmdoption:: --logger-level LOGGER_LEVEL" in cm.arg_long_help()
        assert "--logger-level" in cm.arg_long_help(rst=False)
