# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.dyn` module
"""

import numpy as np
import xarray as xr

from xoa import dyn


def test_dyn_flow2d():

    lon = xr.DataArray(np.arange(4), dims='lon')
    lat = xr.DataArray(np.arange(3), dims='lat')
    u = xr.DataArray(lat*lon/3, dims=('lat', 'lon'), coords={'lon': lon, 'lat': lat})
    v = xr.DataArray((lat+lon)/2, dims=('lat', 'lon'), coords={'lon': lon, 'lat': lat})

    ff = dyn.flow2d(
        u, v, ([1., 2.], [1., 1.5]), np.timedelta64(3, "h"), np.timedelta64(2, "h"),
        date="2000-01-01")
    assert list(ff.coords) == ['lon', 'lat', 'time']
    assert ff.lon.dims == ('time', 'particles')
    assert ff.lon.shape == (3, 2)
    assert 'lat' in ff.coords
    np.testing.assert_allclose(
        ff.lon.values.ravel(),
        [1., 2., 1.0148674, 2.04497533, 1.02261437, 2.06865695])
    assert ff.time.values[0] == np.datetime64("2000-01-01")
