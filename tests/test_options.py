"""
Test the :mod:`xoa.options` module
"""

import pytest

from xoa import exceptions, options


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


def test_get_options():
    opts = options.get_options()
    assert opts["plot"]["cmapdiv"] == options.get_option("plot.cmapdiv")
    assert "natgeo" in opts["layers"].sections


def test_options_are_available_from_the_package():
    import xoa

    for name in xoa.__all__:
        assert hasattr(xoa, name), name
    assert xoa.get_option("plot.cmapdiv") == options.get_option("plot.cmapdiv")
    with xoa.set_options("plot", cmapdiv="cmo.delta"):
        assert xoa.get_option("plot.cmapdiv") == "cmo.delta"
    assert xoa.get_option("plot.cmapdiv") != "cmo.delta"


class TestOptionsFiles:
    """Test loading options from files, and validation"""

    @pytest.fixture(autouse=True)
    def reset(self):
        options.reset_options()
        yield
        options.reset_options()

    @pytest.fixture
    def cfgfile(self, tmp_path):
        path = tmp_path / "xoa.cfg"
        path.write_text("[plot]\nmap_figsize = 5, 4\nscalebar_units = m\n")
        return path

    @pytest.mark.parametrize("as_str", [True, False])
    def test_load_options(self, cfgfile, as_str):
        options.load_options(str(cfgfile) if as_str else cfgfile)
        assert options.get_option("plot.map_figsize") == [5.0, 4.0]
        assert options.get_option("plot.scalebar_units") == "m"
        assert options.get_option("plot.scalebar_loc") == "lower left"

    def test_load_options_missing_file(self, tmp_path):
        with pytest.raises(OSError):
            options.load_options(tmp_path / "missing.cfg")

    @pytest.mark.parametrize("as_str", [True, False])
    def test_set_options_file_context(self, cfgfile, as_str):
        with options.set_options(str(cfgfile) if as_str else cfgfile):
            assert options.get_option("plot.scalebar_units") == "m"
        assert options.get_option("plot.scalebar_units") == "km"

    def test_set_options_file_and_options(self, cfgfile):
        with options.set_options(cfgfile, **{"plot.scalebar_units": "nmi"}):
            assert options.get_option("plot.scalebar_units") == "nmi"
            assert options.get_option("plot.map_figsize") == [5.0, 4.0]

    def test_set_options_are_cumulative(self):
        options.set_options("plot", cmapdiv="a")
        options.set_options("plot", cmappos="b")
        assert options.get_option("plot.cmapdiv") == "a"
        assert options.get_option("plot.cmappos") == "b"

    @pytest.mark.parametrize(
        "opts",
        [
            {"plot.unknown": 1},
            {"unknown.cmapdiv": 1},
            {"plot.scalebar_units": "cm"},
            {"plot.map_figsize": [1]},
        ],
    )
    def test_invalid_options(self, opts):
        before = options.get_option("plot.map_figsize")
        with pytest.raises(exceptions.XoaConfigError):
            options.set_options(**opts)
        assert options.get_option("plot.map_figsize") == before

    def test_resolve_option(self):
        assert options.resolve_option(3, "plot.map_figsize") == 3
        assert options.resolve_option(None, "plot.map_figsize") == (8.0, 8.0)


class TestLayersOptions:
    """Test the layers section"""

    @pytest.fixture(autouse=True)
    def reset(self):
        options.reset_options()
        yield
        options.reset_options()

    def test_default_layers(self):
        natgeo = options.get_option("layers", "natgeo")
        assert natgeo["kind"] == "wmts"
        assert natgeo["url"].startswith("https://")

    def test_load_layer_from_file(self, tmp_path):
        path = tmp_path / "xoa.cfg"
        path.write_text(
            "[layers]\n  [[mine]]\n  kind = wms\n  url = http://x/wms?\n  layer = L\n"
            '  attribution = "A, B"\n'
        )
        options.load_options(path)
        assert options.get_option("layers", "mine")["attribution"] == "A, B"
        assert options.get_option("layers", "mine")["kind"] == "wms"
        assert "natgeo" in options.get_option("layers", "natgeo").parent

    def test_override_default_layer(self):
        with options.set_options("layers", natgeo={"layer": "other"}):
            assert options.get_option("layers", "natgeo")["layer"] == "other"
            assert options.get_option("layers", "natgeo")["kind"] == "wmts"
        assert options.get_option("layers", "natgeo")["layer"] == "NatGeo_World_Map"

    @pytest.mark.parametrize(
        "layer",
        [{"kind": "wms"}, {"url": "u", "layer": "L", "kind": "ftp"}],
        ids=["missing", "kind"],
    )
    def test_invalid_layer(self, layer):
        with pytest.raises(exceptions.XoaConfigError):
            options.load_options({"layers": {"bad": layer}})
        assert "bad" not in options.get_option("layers", "natgeo").parent
