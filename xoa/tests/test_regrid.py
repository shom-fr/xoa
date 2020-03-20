# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.regrid` module
"""

import numpy as np
import xarray as xr

from xoa import regrid


def test_regrid_regrid1d():

    nz0 = 6
    nz1 = 7
    nx = 4
    nt = 5
    dep0 = xr.DataArray(np.linspace(-100., 0., nz0), dims='nz', name='nz')
    dep1 = xr.DataArray(np.linspace(-80., 10., nz1), dims='nk', name='nk')
    lon = xr.DataArray(np.arange(nx, dtype='d'), dims='lon')
    time = xr.DataArray(np.arange(nt, dtype='d'), dims='time')

    da_in = xr.DataArray(np.arange(nt*nz0*nx, dtype='d').reshape(nt, nz0, nx),
                         dims=('time', 'nz', 'lon'),
                         coords=(time, dep0, lon),
                         name="banana",
                         attrs={'long_name': 'Big banana'})

    # data:3d, coord_in:1d, coord_out:1d
    da_out = regrid.regrid1d(da_in, dep1)
    assert not np.isnan(da_out).all()
    assert da_out.min() >= da_in.min()
    assert da_out.max() <= da_in.max()

    # coord_out = xr.DataArray(np.ones((nz1, nx)),
    #                          dims=('nk', 'lon'), name='mydepth',
    #                          attrs={'standard_name': 'ocean_layer_depth'})
