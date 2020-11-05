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


def test_coords_dimflusher1d():

    nz0 = 5
    nz1 = 7
    nlon = 4
    nmem = 2
    dep0 = xr.DataArray(np.linspace(-100., 0., nz0), dims='nz', name='nz')
    dep1 = xr.DataArray(np.linspace(-1000., 0., nz1), dims='nk', name='nk')
    lon = xr.DataArray(range(nlon), dims='lon')
    mem = xr.DataArray(range(nmem), dims='mem')

    da = xr.DataArray(np.ones((mem.size, dep0.size, lon.size)),
                      dims=('mem', 'nz', 'lon'),
                      coords=(mem, dep0, lon),
                      attrs={'long_name': 'Big banana'})
    coord_out = xr.DataArray(np.ones((dep1.size, lon.size)),
                             dims=('nk', 'lon'), name='mydepth',
                             attrs={'standard_name': 'ocean_layer_depth'})

    # 1d -> 1d
    coord = dep1
    dfl = coords.DimFlusher1D(da, coord)
    assert dfl.da_in_data.shape == (nmem*nlon, nz0)
    assert dfl.coord_in_data.shape == (1, nz0)
    assert dfl.coord_out_data.shape == (1, nz1)
    assert dfl.work_dims == ("mem", "lon", "nk")
    assert dfl.work_shape == (nmem, nlon, nz1)
    da_out_data = np.ones((nmem*nlon, nz1))
    da_out = dfl.get_back(da_out_data)
    assert da_out.coords["nk"].shape == (nz1,)
    assert da_out.long_name == "Big banana"

    # 1d -> nd
    dfl = coords.DimFlusher1D(da, coord_out)
    assert dfl.da_in_data.shape == (nmem*nlon, nz0)
    assert dfl.coord_in_data.shape == (nlon, nz0)
    assert dfl.coord_out_data.shape == (nlon, nz1)
    assert dfl.work_dims == ("mem", "lon", "nk")
    assert dfl.work_shape == (nmem, nlon, nz1)
    da_out_data = np.ones((nmem*nlon, nz1))
    da_out = dfl.get_back(da_out_data)
    assert da_out.coords["mydepth"].dims == ("nk", "lon")

    # nd -> nd
    coord_in = xr.DataArray(np.ones((nmem, dep0.size)),
                            dims=('mem', 'nz'))
    da.coords['dep'] = coord_in
    dfl = coords.DimFlusher1D(da, coord_out)
    assert dfl.da_in_data.shape == (nmem*nlon, nz0)
    assert dfl.coord_in_data.shape == (nmem*nlon, nz0)
    assert dfl.coord_out_data.shape == (nmem*nlon, nz1)
    assert dfl.work_dims == ("mem", "lon", "nk")
    assert dfl.work_shape == (nmem, nlon, nz1)
    da_out_data = np.ones((nmem*nlon, nz1))
    da_out = dfl.get_back(da_out_data)
    assert da_out.coords["mydepth"].dims == ("nk", "lon")
