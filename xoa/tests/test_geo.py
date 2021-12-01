# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.geo` module
"""

import numpy as np
import xarray as xr

from xoa import geo


def test_geo_haversine():

    assert geo.haversine(0.0, 0.0, 180.0, 0.0, 1.0) == np.pi
    assert geo.haversine(0, -90.0, 0, 90.0, 1.0) == np.pi
    assert geo.haversine(0, -90.0, 0, 90.0, 2.0) == 2 * np.pi
    assert int(geo.haversine(0, 0, 10.0, 10.0)) == 1568520


def test_geo_cdist():

    xy0 = np.array([[0, 1], [0.0, 1.0]]).T
    xy1 = np.array([[0, 0.5, 180.0], [0.0, 0.5, 0]]).T
    dd = geo.cdist(xy0, xy1, radius=1)
    assert dd.shape == (2, 3)
    assert dd[0, 0] == 0.0
    assert dd[0, -1] == np.pi


def test_geo_pdist():

    xy0 = np.array([[0, 1], [0.0, 1.0]]).T
    dd = geo.pdist(xy0)
    assert dd.shape == (2, 2)
    ddc = geo.pdist(xy0, compact=True)
    assert ddc.shape == (1,)
    assert ddc[0] == dd[1, 0]


def test_geo_clusterize():

    x = [0, 1, 3, 4, 8, 9, 10.0]
    y = x
    ds = xr.Dataset(coords={"lon": ("npts", x), "lat": ("npts", y)})
    clusters = geo.clusterize(ds, npmax=3, split=True)
    assert len(clusters) == 3
    assert set([c.sizes["npts"] for c in clusters]) == {2, 2, 3}
