# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.coords` module
"""

import pytest
import numpy as np
import xarray as xr

from xoa import coords


@pytest.mark.parametrize(
    "inshape,indims,tdims,mode,outshape,outdims",
    [
     ((1, 2), ('y', 'x'), ("x", "y"), "classic",
      (2, 1), ("x", "y")),
     ((3, 2), ('y', 'x'), ("t", "y"), "insert",
      (2, 1, 3), ("x", "t", "y")),
     ((3, 2), ('y', 'x'), ("x", "t", "y"), "compat",
      (2, 3), ("x", "y")),
     ((3, 4, 2), ('y', 't', 'x'), (Ellipsis, "x", "e", "y"), "compat",
      (4, 2, 3), ("t", "x", "y")),
    ]
)
def test_coords_transpose(inshape, indims, tdims, mode, outshape, outdims):
    da = xr.DataArray(np.ones(inshape), dims=indims)
    dao = coords.transpose(da, tdims, mode)
    assert dao.dims == outdims
    assert dao.shape == outshape


def test_coords_is_lon():

    x = xr.DataArray([5], dims="lon", name="lon")
    y = xr.DataArray([5], dims="lat")
    temp = xr.DataArray(np.ones((1, 1)), dims=('lat', 'lon'), coords={'lon': x, 'lat': y})

    assert coords.is_lon(x)
    assert coords.is_lon(temp.lon)
    assert not coords.is_lon(temp.lat)
