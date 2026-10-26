"""
Test the :mod:`xoa.poly` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import mask as xmask
from xoa import poly as xpoly
from xoa.exceptions import XoaError


def get_field():
    """Two bumps on a regular grid"""
    lon = xr.DataArray(
        np.linspace(0, 10, 41),
        dims="lon",
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    lat = xr.DataArray(
        np.linspace(0, 6, 25),
        dims="lat",
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )
    field = -np.exp(-((lon - 2.5) ** 2 + (lat - 3) ** 2)) - 0.5 * np.exp(
        -((lon - 7.5) ** 2 + (lat - 3) ** 2)
    )
    return field.transpose("lat", "lon").assign_coords(lon=lon, lat=lat)


def test_all_polygons():
    polygons = xpoly.get_contour_polygons(get_field(), -0.3)
    assert len(polygons) == 2
    for poly in polygons:
        assert poly.shape[1] == 2
        np.testing.assert_allclose(poly[0], poly[-1])


def test_containing_point_and_extrema():
    field = get_field()
    poly = xpoly.get_contour_polygons(field, -0.3, containing="min")
    assert 1.0 < poly[:, 0].min() and poly[:, 0].max() < 4.0
    poly = xpoly.get_contour_polygons(field, -0.3, containing=(7.5, 3.0))
    assert 6.0 < poly[:, 0].min() and poly[:, 0].max() < 9.0
    with pytest.raises(XoaError, match="No contour"):
        xpoly.get_contour_polygons(field, -0.3, containing=(0.0, 0.0))
    with pytest.raises(XoaError):
        xpoly.get_contour_polygons(field, -0.3, containing="mean")


def test_transposed_nan_and_mask():
    field = get_field()
    field[0, 0] = np.nan
    poly = xpoly.get_contour_polygons(field.T, -0.3, containing="min")
    mask = xmask.get_polygon_mask(field, poly, threshold=None)
    assert mask.sum() > 10
    assert mask.sel(lon=2.5, lat=3.0)


def test_not_2d():
    with pytest.raises(XoaError):
        xpoly.get_contour_polygons(get_field().isel(lat=0), 0.0)
