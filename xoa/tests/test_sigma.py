# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.sigma` module
"""

import xoa
from xoa import sigma


def test_sigma_decode_cf_sigma():

    ds = xoa.open_data_sample("croco.south-africa.meridional.nc")
    dsd = sigma.decode_cf_sigma(ds)

    assert "depth" in dsd
    assert dsd.depth.dims == ('time', 's_rho', 'eta_rho', 'xi_rho')
    assert dsd.depth.shape == (1, 32, 56, 1)
