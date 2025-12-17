# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.coords` module
"""

import pytest
import numpy as np
import xarray as xr

import xoa
from xoa import coords


class TestTranspose:
    """Test array transposition functions"""

    @pytest.mark.parametrize(
        "inshape,indims,tdims,mode,outshape,outdims",
        [
            ((1, 2), ('y', 'x'), ("x", "y"), "classic", (2, 1), ("x", "y")),
            ((3, 2), ('y', 'x'), ("t", "y"), "insert", (2, 1, 3), ("x", "t", "y")),
            ((3, 2), ('y', 'x'), ("x", "t", "y"), "compat", (2, 3), ("x", "y")),
            (
                (3, 4, 2),
                ('y', 't', 'x'),
                (Ellipsis, "x", "e", "y"),
                "compat",
                (4, 2, 3),
                ("t", "x", "y"),
            ),
        ],
    )
    def test_transpose(self, inshape, indims, tdims, mode, outshape, outdims):
        da = xr.DataArray(np.ones(inshape), dims=indims)
        dao = coords.transpose(da, tdims, mode)
        assert dao.dims == outdims
        assert dao.shape == outshape


class TestCoordinateIdentification:
    """Test coordinate identification functions"""

    def test_is_lon(self):
        x = xr.DataArray([5], dims="lon", name="lon")
        y = xr.DataArray([5], dims="lat")
        temp = xr.DataArray(np.ones((1, 1)), dims=('lat', 'lon'), coords={'lon': x, 'lat': y})

        assert coords.is_lon(x)
        assert coords.is_lon(temp.lon)
        assert not coords.is_lon(temp.lat)


class TestDepth:
    """Test depth coordinate functions"""

    def test_get_depth_from_variable(self):
        da = xr.DataArray(
            np.ones((2, 3)),
            dims=("depth", "lon"),
            coords={"depth": ("depth", [0, 1]), "lon": [1, 2, 3]},
        )
        depth = coords.get_depth(da)
        assert depth is not None
        np.testing.assert_allclose(depth.values, [0, 1])


class TestStacking:
    """Test coordinate stacking functions"""

    def test_geo_stack(self):
        ds = xr.Dataset(
            {"temp": (("depth", "lat", "lon"), np.ones((2, 3, 4)))},
            coords={"depth": [0, 1], "lat": [40, 41, 42], "lon": [-10, -9, -8, -7]},
        )
        dss = coords.geo_stack(ds, "npts")
        assert dss.temp.dims == ("depth", "npts")
        assert dss.temp.lon.shape == dss.temp.shape[1:]

        tempc = coords.geo_stack(ds.temp, "npts")
        xr.testing.assert_equal(tempc, dss.temp)
