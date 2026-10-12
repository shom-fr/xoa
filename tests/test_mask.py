# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.mask` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import mask as xmask
from xoa.exceptions import XoaError


def _lonlat_ds():
    lon = xr.DataArray(
        np.linspace(0, 4, 5),
        dims="lon",
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    lat = xr.DataArray(
        np.linspace(0, 3, 4),
        dims="lat",
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )
    return xr.Dataset(coords={"lon": lon, "lat": lat})


class TestGetPolygonMask:
    poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    def get_da(self):
        ds = _lonlat_ds()
        return xr.DataArray(np.zeros((4, 5)), dims=("lat", "lon"), coords=ds.coords)

    def test_threshold(self):
        da = self.get_da()
        mask = xmask.get_polygon_mask(da, self.poly)
        assert mask.dtype == bool
        assert mask.dims == ("lat", "lon")
        assert mask.shape == da.shape
        # The edges of the polygon are at the center of the cells
        assert int(mask.sum()) == 5
        assert int(xmask.get_polygon_mask(da, self.poly, 1.0).sum()) == 1
        assert int(xmask.get_polygon_mask(da, self.poly, 0.1).sum()) == 9
        shifted = [(x + 0.2, y + 0.2) for x, y in self.poly]
        assert int(xmask.get_polygon_mask(da, shifted, None).sum()) == 4

    def test_several_polygons(self):
        da = self.get_da()
        other = [(3.5, 2.5), (4.5, 2.5), (4.5, 3.5), (3.5, 3.5)]  # Around a single cell
        one = xmask.get_polygon_mask(da, self.poly, 1.0)
        both = xmask.get_polygon_mask(da, [self.poly, other], 1.0)
        assert int(both.sum()) == int(one.sum()) + 1

    def test_shapely(self):
        shapely = pytest.importorskip("shapely.geometry")
        da = self.get_da()
        expected = xmask.get_polygon_mask(da, self.poly)
        mask = xmask.get_polygon_mask(da, shapely.Polygon(self.poly))
        assert mask.equals(expected)
        mask = xmask.get_polygon_mask(da, shapely.MultiPolygon([shapely.Polygon(self.poly)]))
        assert mask.equals(expected)

    def test_curvilinear(self):
        da = self.get_da()
        lon2d, lat2d = xr.broadcast(da.lon, da.lat)
        da2d = xr.DataArray(da.values.T, dims=lon2d.dims, coords={"lon": lon2d, "lat": lat2d})
        mask = xmask.get_polygon_mask(da2d, self.poly)
        assert mask.dims == ("lat", "lon")  # Y and X order whatever the input order
        np.testing.assert_array_equal(mask, xmask.get_polygon_mask(da, self.poly))

    def test_bad_polygon(self):
        with pytest.raises(XoaError):
            xmask.get_polygon_mask(self.get_da(), [(0, 0), (1, 1)])


class TestGetCommonMask:
    def get_da(self):
        ds = _lonlat_ds()
        return xr.DataArray(np.ones((4, 5)), dims=("lat", "lon"), coords=ds.coords)

    def test_finite_everywhere(self):
        a, b = self.get_da(), self.get_da()
        a[0, 1] = np.nan
        b[2, 3] = np.inf
        mask = xmask.get_common_mask(a, b)
        assert mask.dtype == bool and mask.dims == ("lat", "lon")
        assert int(mask.sum()) == 18
        assert not mask[0, 1] and not mask[2, 3]

    def test_edge(self):
        mask = xmask.get_common_mask(self.get_da(), edge=1)
        assert int(mask.sum()) == 2 * 3
        assert not mask[0].any() and not mask[:, -1].any()

    def test_no_array(self):
        with pytest.raises(XoaError):
            xmask.get_common_mask()
