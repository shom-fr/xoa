# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.filter` module
"""

import pytest
import numpy as np
import xarray as xr

import xoa.filter as xfilter

np.random.seed(0)
da1d = xr.DataArray(np.random.normal(size=10), dims=('nt',))
da2d = xr.DataArray(np.random.normal(size=(50, 50)), dims=('ny', 'nx'))
da3d = xr.DataArray(np.random.normal(size=(30, 50, 50)), dims=('nt', 'ny', 'nx'))


def test_filter_generate_orthogonal_kernel():
    kernel = xfilter.generate_orthogonal_kernel((3, 5), "bartlett")
    assert kernel.shape == (3, 5)
    np.testing.assert_allclose(kernel[1], [0.0, 0.5, 1.0, 0.5, 0.0])

    kernel = xfilter.generate_orthogonal_kernel([2.8], "ones", fill_value=-1)
    np.testing.assert_allclose(kernel, [-1, 1, 1, -1])


@pytest.mark.parametrize(
    "kernel, data, oshape, odims, osum",
    [
        (3, da1d, (3,), ('nt',), 3),
        ((4, 3), da2d, (4, 3), ('ny', 'nx'), 12),
        ([1, 2, 1], da1d, (3,), ('nt',), 4),
        ([1, 2, 1], da2d, (3, 3), ('ny', 'nx'), 16),
        ({'nx': [1, 2, 1], 'ny': [1, 1, 2, 1, 1]}, da2d, (5, 3), ('ny', 'nx'), 24),
        (np.ones((3, 5)), da2d, (3, 5), ('ny', 'nx'), 15),
        (xr.DataArray(np.ones((3, 5)), dims=('nx', 'ny')), da2d, (5, 3), ('ny', 'nx'), 15),
        (xr.DataArray(np.ones((3, 5)), dims=('ny', 'nx')), da3d, (1, 3, 5), ('nt', 'ny', 'nx'), 15),
        (xr.DataArray(np.ones(3), dims='nx'), da2d, (3,), ('nx',), 3),
    ],
)
def test_filter_generate_kernel(kernel, data, oshape, odims, osum):
    okernel = xfilter.generate_kernel(kernel, data)
    assert okernel.shape == oshape
    assert okernel.dims == odims
    np.testing.assert_allclose(okernel.data.sum(), osum)


def test_filter_convolve():

    da = xr.DataArray(np.ones((5, 5)), dims=('ny', 'nx'))
    da[2, 2] = np.nan

    dac = xfilter.convolve(da, 3, normalize=True, na_thres=1)
    np.testing.assert_allclose(dac.data, np.ones(da.shape))

    dac = xfilter.convolve(da, 3, normalize=True, na_thres=0)
    assert dac.isnull().all()

    dac = xfilter.convolve(da, 3, normalize=False, na_thres=1)
    assert dac.sum() == 160


def test_filter_erode_mask():

    da_in = xr.DataArray(np.random.normal(size=(2, 9, 11)), dims=('nt', 'ny', 'nx'))
    nt, ny, nx = da_in.shape
    i0, n = 2, 5
    da_in[:, i0 : i0 + n, i0 : i0 + n] = np.nan

    # iteration
    da_out_iter = xfilter.erode_mask(da_in, 2)
    assert int(da_out_iter[1].count()) == (da_in[1].size - (n - 4) ** 2)

    # until mask
    mask = np.isnan(da_out_iter)
    da_out_until = xfilter.erode_mask(da_in, mask)
    np.testing.assert_allclose(da_out_iter, da_out_until)


def test_filter_erode_coast():

    da_in = xr.DataArray(np.random.normal(size=(3, 3, 3)), dims=('nt', 'ny', 'nx'))
    da_in[:, 1, 1] = np.nan

    np_out = xfilter.erode_coast(da_in, 1).values
    np_in = da_in.values
    np.testing.assert_allclose(
        np_out[1, 1, 1], 0.25 * (np_in[1, 0, 1] + np_in[1, 2, 1] + np_in[1, 1, 0] + np_in[1, 1, 2])
    )


def test_filter_dermerliac():

    times = np.arange("2000-01-01", "2000-01-10", dtype="M8[h]")
    nt = len(times)
    data = np.cos(np.arange(nt) * np.pi / 12.2)
    data = np.resize(data, (3, 4, nt)).T
    da_in = xr.DataArray(data, dims=('time', 'ny', 'nx'), coords={"time": times})

    da_out = xfilter.demerliac(da_in, na_thres=0)
    assert da_out.dims == da_in.dims
    assert da_out.shape == da_in.shape
    assert da_out.min() > -1e-4
    assert da_out.max() < 1e-4

    da_out = xfilter.demerliac(da_in, na_thres=1)
    assert abs(da_out.mean()) < 0.1


@pytest.mark.parametrize(
    "radius, expected",
    [
        (0.0, [True, True, True, True, True]),
        (0.025, [True, False, True, True, False]),
        (np.inf, [True, False, False, False, False]),
    ],
)
def test_filter_get_decimate_arg(radius, expected):

    x = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    y = x.copy()
    keep = xfilter._get_decimate_arg_(x, y, radius)
    np.testing.assert_allclose(keep, expected)


def test_filter_decimate():

    ds = xr.Dataset(
        {"temp": ("npts", 20 + np.arange(5))},
        coords={
            "lon": ("npts", [0.0, 1.0, 2.0, 5.0, 6.0]),
            "lat": ("npts", [0.0, 1.0, 2.0, 5.0, 6.0]),
        },
    )

    dsc0 = xfilter.decimate(ds, 160e3, method="pick")
    assert dsc0.lon.dims == ("npts",)
    np.testing.assert_allclose(dsc0.lon.values, [0.0, 2.0, 5.0])

    dac = xfilter.decimate(ds.temp, 160e3, method="pick")
    np.testing.assert_allclose(dac.lon.values, [0.0, 2.0, 5.0])

    dsc1 = xfilter.decimate(ds.drop_vars("temp"), 160e3, method="pick")
    np.testing.assert_allclose(dsc1.lon.values, [0.0, 2.0, 5.0])

    dsc0a = xfilter.decimate(ds, 160e3, method="average", smooth_factor=0)
    xr.testing.assert_equal(dsc0.temp, dsc0a.temp)
    dsc0b = xfilter.decimate(ds, 160e3, method="average", smooth_factor=np.inf)
    assert np.isclose(ds.temp.values.mean(), dsc0b.temp.values.mean())
