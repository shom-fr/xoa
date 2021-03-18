# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.grid` module
"""
import numpy as np
import xarray as xr
import pytest

from xoa import grid


def test_grid_get_edges_1d():

    x = np.arange(3) * 2.
    xe = grid.get_edges_1d(x)
    xet = np.arange(4) * 2. - 1
    np.testing.assert_allclose(xe, xet)

    xx = x[np.newaxis, :, np.newaxis]
    xxe = grid.get_edges_1d(xx, axis=1)
    assert xxe.shape == (1, 4, 1)
    np.testing.assert_allclose(xxe[0, :, 0], xet)

    x_ = xr.DataArray(x, dims='x', name='x')
    xe_ = grid.get_edges_1d(x_)
    np.testing.assert_allclose(xe_, xet)
    assert xe_.dims[0] == 'x_edges'
    assert xe_.name == 'x_edges'

    for axis in (1, 'x'):
        xx_ = xr.DataArray(xx, dims=('y', 'x', 't'), name='lon')
        xxe_ = grid.get_edges_1d(xx_, axis=axis)
        np.testing.assert_allclose(xxe_[0, :, 0], xet)
        assert xxe_.dims[1] == 'x_edges'
        assert xxe_.dims[0] == 'y'
        assert xxe_.name == 'lon_edges'


@pytest.mark.parametrize(
    "positive, expected", [
        ["down", [100., 600., 1600.]],
        ["up", [-1600, -1500, -1000]]
        ])
def test_dz2depth(positive, expected):

    dz = xr.DataArray(
        np.resize([100, 500, 1000.], (2, 3)).T,
        dims=("nz", "nx"))

    depth = grid.dz2depth(dz, positive)

    np.testing.assert_allclose(depth.isel(nx=1), expected)
