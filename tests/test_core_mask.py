# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.mask` module
"""

import numpy as np

from xoa.core import mask as cmask


def _grid(lon0=0.0):
    lon, lat = np.meshgrid(np.arange(10.0) + 0.5, np.arange(8.0) + 0.5)
    lon_bounds = np.stack([lon - 0.5, lon + 0.5, lon + 0.5, lon - 0.5], axis=-1)
    lat_bounds = np.stack([lat - 0.5, lat - 0.5, lat + 0.5, lat + 0.5], axis=-1)
    wrap = lambda x: (x + lon0 + 180) % 360 - 180  # noqa
    return wrap(lon), lat, wrap(lon_bounds), lat_bounds, wrap


RECT = np.array([[1.5, 1.5], [5.5, 1.5], [5.5, 4.5], [1.5, 4.5]])


def test_polygon_cell_fraction():
    lon, lat, lonb, latb, _ = _grid()
    frac = cmask.polygon_cell_fraction(RECT, lon, lat, lonb, latb)
    assert frac.shape == lon.shape
    np.testing.assert_allclose(frac[2:4, 2:5], 1.0)  # Fully covered
    np.testing.assert_allclose(frac[2:4, 1], 0.5)  # Half covered edge cells
    np.testing.assert_allclose(frac[1, 2:5], 0.5)
    np.testing.assert_allclose(frac[1, 1], 0.25)  # Corner
    assert frac[0].sum() == 0
    # Total covered area
    np.testing.assert_allclose(frac.sum(), 4 * 3)


def test_polygon_mask_threshold():
    lon, lat, lonb, latb, _ = _grid()
    assert cmask.polygon_mask(RECT, lon, lat, lonb, latb, 1.0).sum() == 2 * 3
    assert cmask.polygon_mask(RECT, lon, lat, lonb, latb, 0.5).sum() == 4 * 5 - 4
    assert cmask.polygon_mask(RECT, lon, lat, lonb, latb, 0.3).sum() == 4 * 5 - 4
    assert cmask.polygon_mask(RECT, lon, lat, lonb, latb, 0.2).sum() == 4 * 5
    assert cmask.polygon_mask(RECT, lon, lat, lonb, latb, 0.0).sum() == 4 * 5
    # Centers only
    mask = cmask.polygon_mask(RECT, lon, lat)
    assert mask.sum() == 4 * 3
    assert mask.dtype == bool


def test_polygon_mask_concave():
    lon, lat, lonb, latb, _ = _grid()
    concave = np.array([[0, 0], [8, 0], [8, 6], [4, 2], [0, 6]], dtype="d")
    frac = cmask.polygon_cell_fraction(concave, lon, lat, lonb, latb)
    # Area of the polygon: rectangle minus the notch triangle
    np.testing.assert_allclose(frac.sum(), 8 * 6 - 0.5 * 8 * 4)
    assert frac[5, 4] == 0.0  # In the notch


def test_polygon_mask_across_the_dateline():
    ref = cmask.polygon_cell_fraction(RECT, *_grid()[:4])
    lon, lat, lonb, latb, wrap = _grid(175.0)
    assert (lon < 0).any() and (lon > 0).any()
    shifted = RECT.copy()
    shifted[:, 0] = wrap(shifted[:, 0])
    frac = cmask.polygon_cell_fraction(shifted, lon, lat, lonb, latb)
    np.testing.assert_allclose(frac, ref)


def test_polygon_mask_missing_centers():
    lon, lat, lonb, latb, _ = _grid()
    lon[2, 2] = np.nan
    mask = cmask.polygon_mask(RECT, lon, lat, lonb, latb, 0.5)
    assert not mask[2, 2]
