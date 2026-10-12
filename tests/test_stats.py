"""
Test the :mod:`xoa.stats` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import stats as xstats
from xoa.exceptions import XoaError


def get_data():
    ref = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("y", "x"), name="ref")
    da = ref + xr.DataArray([1.0, -1.0, 2.0], dims="y")
    return da, ref


def test_rms():
    da, ref = get_data()
    assert float(xstats.rms(da, ref)) == pytest.approx(np.sqrt((1 + 1 + 4) / 3))
    np.testing.assert_allclose(xstats.rms(da, ref, dim="x"), [1, 1, 2])
    assert float(xstats.rms(da - ref)) == float(xstats.rms(da, ref))


def test_rms_mask_and_nan():
    da, ref = get_data()
    mask = xr.DataArray([True, False, True], dims="y")
    assert float(xstats.rms(da, ref, mask=mask)) == pytest.approx(np.sqrt(5 / 2))
    da[0, 0] = np.nan
    assert np.isfinite(xstats.rms(da, ref))


def test_rms_by_region():
    da, ref = get_data()
    regions = {
        "all": xr.ones_like(da, dtype=bool),
        "first": xr.DataArray([True, False, False], dims="y"),
    }
    res = xstats.rms(da, ref, by=regions)
    assert res.dims == ("region",)
    assert list(res.region.values) == ["all", "first"]
    np.testing.assert_allclose(res, [np.sqrt(2), 1.0])
    with pytest.raises(XoaError):
        xstats.rms(da, by={})
