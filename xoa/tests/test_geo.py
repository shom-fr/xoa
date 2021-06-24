# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.geo` module
"""

import numpy as np
import xarray as xr

from xoa import geo


def test_geo_haversine():

    assert geo.haversine(0., 0., 180., 0., 1.) == np.pi
    assert geo.haversine(0, -90., 0, 90., 1.) == np.pi
    assert geo.haversine(0, -90., 0, 90., 2.) == 2*np.pi
    assert int(geo.haversine(0, 0, 10., 10.)) == 1568520


def test_geo_cdist():

    dd = geo.cdist([0, 1], [0., 1.], [0, 0.5, 180.], [0., 0.5, 0], radius=1)
    assert dd.shape == (3, 2)
    assert dd[0, 0] == 0.
    assert dd[0, -1] == np.pi


def test_geo_pdist():

    dd = geo.pdist([0, 1], [0., 1.])
    assert dd.shape == (2, 2)
    ddc = geo.pdist([0, 1], [0., 1.], compact=True)
    assert ddc.shape == (1, )
    assert ddc[0] == dd[1, 0]
