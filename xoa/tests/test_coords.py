# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.coords` module
"""
import re

import pytest
import numpy as np
import xarray as xr

from xoa import coords


def test_coords_flush_work_dim_right():


    dep0 = xr.DataArray([-100., 0.], dims='nz', name='nz')
    dep1 = xr.DataArray([-1000., -50, 0.], dims='nk', name='nk')
    lon = xr.DataArray(range(4), dims='lon')
    mem = xr.DataArray(range(2), dims='mem')
    time = xr.DataArray(range(1), dims='time')

    da = xr.DataArray(np.ones((mem.size, dep0.size, lon.size)),
                      dims=('mem', 'nz', 'lon'),
                      coords=(mem, dep0, lon))
    # coord = dep1
    # fda, fcoord = coords.flush_work_dim_right(da, coord)
    # assert fda.dims == ('mem', 'lon', 'nz')
    # assert fcoord.dims == ('nk', )

    coord = xr.DataArray(np.ones((dep1.size, lon.size)),
                         dims=('nk', 'lon'),
                         attrs={'standard_name': 'ocean_layer_depth'})
    # fda, fcoord = coords.flush_work_dim_right(da, coord)
    # assert fda.dims == ('mem', 'lon', 'nz')
    # assert fcoord.dims == ('lon', 'nk')
    print(da[0, :, 0])
    da1d = da[0, :, 0]
    del da1d.coords['lon'], da1d.coords['mem']
    print(da1d)
    fda, fcoord = coords.flush_work_dim_right(da1d, coord)
    assert fda.dims == ('lon', 'nz')
    assert fcoord.dims == ('lon', 'nk')

