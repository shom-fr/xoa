# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.coords` module
"""

import pytest
import numpy as np
import xarray as xr

import xoa
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


def test_coords_get_depth_from_variable():

    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("depth", "lon"),
        coords={"depth": ("depth", [0, 1]), "lon": [1, 2, 3]})
    depth = coords.get_depth(da)
    assert depth is not None
    np.testing.assert_allclose(depth.values, [0, 1])


def test_coords_get_depth_from_sigma():

    ds = xoa.open_data_sample("croco.south-africa.meridional.nc")
    depth = coords.get_depth(ds)
    assert depth is not None
    assert depth.name == "depth"


def test_coords_get_depth_from_dz():

    ds = xoa.open_data_sample("hycom.gdp.h.nc")
    ds = ds.rename(h="dz")
    depth = coords.get_depth(ds)
    assert depth is not None
    assert depth.name == "depth"
