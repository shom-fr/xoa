# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.interp` module
"""

import functools
import numpy as np
import pytest

from xoa.core import interp


def vfunc(t=0, z=0, y=0, x=0):
    """A function that returns a linear combination of coordinates"""
    return 1.13 * x + 12.35 * y + 3.24 * z - 0.65 * t


def round_as_time(arr, units="us", origin="1950-01-01"):
    arr = arr.astype(f"m8[{units}]")
    origin = np.datetime64(origin, units)
    arr = arr + origin
    return (arr - origin) / np.timedelta64(1, units)


@pytest.mark.parametrize(
    "x,y,pt,qt",
    [
        (0.0, 0, 0, 0),
        (3, 1, 0, 1),
        (2, 3, 1, 1),
        (-1, 2, 1, 0),
        (1.0, 1.5, 0.5, 0.5),
        (-1, -1, -1, -1),
    ],
)
def test_interp_cell2relloc(x, y, pt, qt):
    x1, y1 = 0.0, 0.0
    x2, y2 = 3.0, 1.0
    x3, y3 = 2.0, 3.0
    x4, y4 = -1.0, 2.0
    p, q = interp.cell2relloc(x1, x2, x3, x4, y1, y2, y3, y4, x, y)
    assert p == pt
    assert q == qt


@functools.lru_cache()
def get_grid2locs_coords(nex=4, nexz=2, nxi=7, nyi=6, nzi=5, nti=4, no=10):
    np.random.seed(0)

    tti, zzi, yyi, xxi = np.mgrid[
        0 : nti - 1 : nti * 1j,
        0 : nzi - 1 : nzi * 1j,
        0 : nyi - 1 : nyi * 1j,
        0 : nxi - 1 : nxi * 1j,
    ]

    zzi = zzi[None]
    zzi = np.repeat(zzi, nexz, axis=0)

    xyztomin = -0.5
    xo = np.random.uniform(xyztomin, nxi - 1.5, no)
    yo = np.random.uniform(xyztomin, nyi - 1.5, no)
    zo = np.random.uniform(xyztomin, nzi - 1.5, no)
    to = np.random.uniform(xyztomin, nti - 1.5, no)

    tti = round_as_time(tti)
    to = round_as_time(to)

    return xxi, yyi, zzi, tti, xo, yo, to, zo


def test_interp_grid2locs():
    # Multi-dimensional generic coordinates
    nex = 4
    nexz = 2
    nxi = 7
    nyi = 6
    nzi = 5
    nti = 4
    no = 10
    xxi, yyi, zzi, tti, xo, yo, to, zo = get_grid2locs_coords(
        nex=nex, nexz=nexz, nxi=nxi, nyi=nyi, nzi=nzi, nti=nti, no=no
    )

    # Pure 1D axes
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)
    vo_truth = np.array(vfunc(to, zo, yo, xo))
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    assert vo_interp[0].shape == vo_truth.shape
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # Single point in space
    xi = xxi[0, 0, 0:1, :1]  # (nyix=1,nxi)
    yi = yyi[0, 0, :1, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :, :, :1, :1]
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,1,1)
    vo_truth = np.array(vfunc(to, zo, yi[0], xi[0]))
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # Constant time
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:1, 0, 0, 0]  # (1)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :1, :, :, :]
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,1,nzi,nyi,nxi)
    vo_truth = vfunc(ti, zo, yo, xo)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # Variable depth with 1D X/Y + T
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[:, :, :, :, :]  # (nexz,ntiz=nti,nzi,nyiz=nyi,nxiz=nxi)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)
    vo_truth = vfunc(to, zo, yo, xo)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # 2D X/Y with no other axes (pure curvilinear)
    xi = xxi[0, 0]  # (nyix=nyi,nxi)
    yi = yyi[0, 0]  # (nyi,nxiy=nxi)
    zi = zzi[0:1, 0:1, 0:1, 0:1, 0:1]  # (nexz=1,ntiz=1,1,nyiz=1,nxiz=1)
    ti = tti[:1, 0, 0, 0]  # (1)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :1, :1, :, :]
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,1,1,nyi,nxi)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_interp_rect = interp.grid2locs(xi[:1], yi[:, :1], zi, ti, vi, xo, yo, zo, to)
    vo_truth = vfunc(ti, zi.ravel()[0], yo, xo)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)
    vo_truth = vfunc(ti, zi.ravel()[0], yo, xo)
    vo_truth[np.isnan(vo_interp_rect[0])] = np.nan
    np.testing.assert_allclose(vo_interp_rect[0], vo_truth)

    # Same coordinates
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)
    tzyxo = np.meshgrid(ti, zi, yi, xi, indexing='ij')
    xo = tzyxo[3].ravel()
    yo = tzyxo[2].ravel()
    zo = tzyxo[1].ravel()
    to = tzyxo[0].ravel()
    vo_truth = vfunc(to, zo, yo, xo)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)


def test_interp_isoslice():
    depth = np.linspace(-50, 0.0, 6)
    values = np.linspace(10, 20.0, 6)
    isoval = 15.0

    isodepth = interp.isoslice(depth, values, isoval, False)
    assert isodepth == -25.0
    isodepth = interp.isoslice(depth, values, isoval, True)
    assert isodepth == -25.0

    depth = np.resize(depth, (2,) + depth.shape)
    isodepth = interp.isoslice(depth, values, isoval, False)
    np.testing.assert_allclose(isodepth, [-25.0, -25.0])
