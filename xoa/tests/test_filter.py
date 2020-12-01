# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.filter` module
"""

import pytest
import numpy as np
import xarray as xr

from xoa import filter as xfilter

np.random.seed(0)
da1d = xr.DataArray(np.random.normal(size=10), dims=('nt', ))
da2d = xr.DataArray(np.random.normal(size=(50, 50)), dims=('ny', 'nx'))
da3d = xr.DataArray(np.random.normal(size=(30, 50, 50)),
                    dims=('nt', 'ny', 'nx'))


@pytest.mark.parametrize(
    "kernel, data, oshape, odims, osum",
    [
     (3, da1d, (3, ), ('nt', ), 3),
     ((4, 3), da2d, (4, 3), ('ny', 'nx'), 12),
     ([1, 2, 1], da1d, (3,), ('nt', ), 4),
     ([1, 2, 1], da2d, (3, 3), ('ny', 'nx'), 16),
     ({'nx': [1, 2, 1], 'ny': [1, 1, 2, 1, 1]}, da2d,
      (5, 3), ('ny', 'nx'), 24),
     (np.ones((3, 5)), da2d, (3, 5), ('ny', 'nx'), 15),
     (xr.DataArray(np.ones((3, 5)), dims=('nx', 'ny')), da2d,
      (5, 3), ('ny', 'nx'), 15),
     (xr.DataArray(np.ones(3), dims='nx'), da2d, (1, 3), ('ny', 'nx'), 3),
    ]
)
def test_filter_generate_kernel(kernel, data, oshape, odims, osum):
    okernel = xfilter.generate_kernel(kernel, data)
    assert okernel.shape == oshape
    assert okernel.dims == odims
    np.testing.assert_allclose(okernel.data.sum(), osum)


def test_convolve():

    da = xr.DataArray(np.ones((5, 5)), dims=('ny', 'nx'))
    da[2, 2] = np.nan

    dac = xfilter.convolve(da, 3, normalize=True)
    np.testing.assert_allclose(dac.data, np.ones(da.shape))

    dac = xfilter.convolve(da, 3, normalize=False)
    assert dac.sum() == 100


def test_erode_mask():

    da_in = da3d.copy()
    nt, ny, nx = da_in.shape
    da_in[:, int(ny/3):int(2*ny/3), int(nx/3):int(2*nx/3)] = np.nan

    # iteration
    da_out = xfilter.erode_mask(da_in, 2)
    assert int(da_out.sum()) == -25
    assert int(da_out.count()) == 69930

    # check average
    da_out = xfilter.erode_mask(da_in*0+1, 2)
    assert da_out.mean() == 1

    # until mask
    mask = np.zeros((ny, nx), dtype='?')
    ny2 = int(ny/2)
    nx2 = int(nx/2)
    xpad = int(nx/10)
    ypad = int(ny/10)
    mask[ny2-ypad:ny2+ypad+1, nx2-xpad:nx2+xpad+1] = True
    da_out = xfilter.erode_mask(da_in, mask)
    assert int(da_out.sum()) == -21
    assert int(da_out.count()) == 72570
