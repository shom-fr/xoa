# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.color` module
"""
import matplotlib.colors as mcolors
import pytest

from xoa import color as xcolor


class TestCmapAdapter:
    def test_no_specs(self):
        cma = xcolor.CmapAdapter("viridis")
        assert cma.specs is None
        assert cma.get_cmap() is cma.cmap
        assert cma.set_vlim(-1, 3) == (-1, 3)
        assert cma.get_dict()["vmin"] == -1 and cma.get_dict()["vmax"] == 3

    def test_pivot(self):
        cma = xcolor.CmapAdapter("RdBu", "piv0")
        assert cma.specs == ("pivot", 0.0)
        cma.set_vlim(-1, 3)
        assert isinstance(cma.get_cmap(), mcolors.Colormap)
        assert cma.get_cmap() is not cma.cmap

    def test_pivot_without_vlim(self):
        cma = xcolor.CmapAdapter("RdBu", "piv0")
        with pytest.warns(UserWarning):
            assert cma.get_cmap() is cma.cmap

    def test_cycle(self):
        cma = xcolor.CmapAdapter("twilight", "cyc360")
        assert cma.specs == ("cycle", 360.0) and (cma.vmin, cma.vmax) == (0, 360.0)
        assert cma.set_vlim(10, 20) == (0, 360.0)
        assert cma.get_cmap() is cma.cmap

    def test_invalid_specs(self):
        with pytest.raises(ValueError, match="Invalid cmap specs"):
            xcolor.CmapAdapter("viridis", "foo")
