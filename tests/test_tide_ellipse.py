"""
Test the :mod:`xoa.tide.ellipse` module
"""

import numpy as np

import xoa.tide as xtide


def test_tidal_ellipses(tide_uv, tide_coef_uv):
    *_, smaj, smin = tide_uv
    ell = xtide.tidal_ellipses(tide_coef_uv)
    np.testing.assert_allclose(ell["ellipticity"], smin / smaj, atol=1e-8)
    assert set(np.unique(ell["rotation"])) <= {-1.0, 0.0, 1.0}
    np.testing.assert_allclose(ell["wplus_amp"] + ell["wminus_amp"], smaj, atol=1e-8)


def test_ellipse_outline(tide_uv, tide_coef_uv):
    *_, smaj, smin = tide_uv
    uo, vo = xtide.ellipse_outline(tide_coef_uv, n=50)
    assert uo.dims == ("constituent", "y", "x", "ellipse_point") and uo.shape[-1] == 50
    np.testing.assert_allclose(np.hypot(uo, vo).max("ellipse_point"), smaj, rtol=1e-2)
