# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa._interp` module
"""

import numpy as np

from xoa import _interp


def vfunc(t=0, z=0, y=0, x=0):
    return 1*x + 2.35*y + 3.24*z - 0.65*t


def test_interp_linear4dto1dxx():
    nex = 4
    nexz = 2
    nxi = 7
    nyi = 6
    nzi = 5
    nti = 4
    no = 1
    # no = 1

    tti, zzi, yyi, xxi = np.mgrid[0:nti-1:nti*1j, 0:nzi-1:nzi*1j,
                                  0:nyi-1:nyi*1j, 0:nxi-1:nxi*1j]

    zzi = zzi[None]
    zzi = np.repeat(zzi, nexz, axis=0)

    # Pure 1D axes

    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex, )+vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)

    np.random.seed(0)
    xyztomin = -0.5
    xo = np.random.uniform(xyztomin, nxi-1.5, no)
    yo = np.random.uniform(xyztomin, nyi-1.5, no)
    zo = np.random.uniform(xyztomin, nzi-1.5, no)
    to = np.random.uniform(xyztomin, nti-1.5, no)
    vo_truth = np.ma.array(vfunc(to, zo, yo, xo))

    vo_interp = _interp.linear4dto1dxx(xi, yi, zi, ti, vi, xo, yo, zo, to)

    np.testing.assert_almost_equal(vo_interp[0], vo_truth)

    # Single point in space

    xi = xxi[0, 0, 0:1, :1]  # (nyix=1,nxi)
    yi = yyi[0, 0, :1, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :, :, :1, :1]
    vi = np.resize(vi, (nex, )+vi.shape[1:])  # (nex,nti,nzi,1,1)

    np.random.seed(0)
    xyztomin = -0.5
    xo = np.random.uniform(xyztomin, nxi-1.5, no)
    yo = np.random.uniform(xyztomin, nyi-1.5, no)
    zo = np.random.uniform(xyztomin, nzi-1.5, no)
    to = np.random.uniform(xyztomin, nti-1.5, no)
    vo_truth = np.ma.array(vfunc(to, zo, yi[0], xi[0]))

    vo_interp = _interp.linear4dto1dxx(xi, yi, zi, ti, vi, xo, yo, zo, to)

    np.testing.assert_almost_equal(vo_interp[0], vo_truth)

    # Constant time

    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:1, 0, 0, 0]  # (1)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :1, :, :, :]
    vi = np.resize(vi, (nex, )+vi.shape[1:])  # (nex,1,nzi,nyi,nxi)

    np.random.seed(0)
    xyztomin = -0.5
    xo = np.random.uniform(xyztomin, nxi-1.5, no)
    yo = np.random.uniform(xyztomin, nyi-1.5, no)
    zo = np.random.uniform(xyztomin, nzi-1.5, no)
    to = np.random.uniform(xyztomin, nti-1.5, no)
    vo_truth = vfunc(ti, zo, yo, xo)

    vo_interp = _interp.linear4dto1dxx(xi, yi, zi, ti, vi, xo, yo, zo, to)

    np.testing.assert_almost_equal(vo_interp[0], vo_truth)

    # Variable depth with 1D X/Y + T

    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[:, :, :, :, :]  # (nexz,ntiz=nti,nzi,nyiz=nyi,nxiz=nxi)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex, )+vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)

    np.random.seed(0)
    xyztomin = -0.5
    xo = np.random.uniform(xyztomin, nxi-1.5, no)
    yo = np.random.uniform(xyztomin, nyi-1.5, no)
    zo = np.random.uniform(xyztomin, nzi-1.5, no)
    to = np.random.uniform(xyztomin, nti-1.5, no)
    vo_truth = vfunc(to, zo, yo, xo)

    vo_interp = _interp.linear4dto1dxx(xi, yi, zi, ti, vi, xo, yo, zo, to)

    np.testing.assert_almost_equal(vo_interp[0], vo_truth)

    # 2D X/Y with no other axes (pure curvilinear)

    xi = xxi[0, 0]  # (nyix=nyi,nxi)
    yi = yyi[0, 0]  # (nyi,nxiy=nxi)
    zi = zzi[0:1, 0:1, 0:1, 0:1, 0:1]  # (nexz=1,ntiz=1,1,nyiz=1,nxiz=1)
    ti = tti[:1, 0, 0, 0]  # (1)

    vi = vfunc(tti, zzi, yyi, xxi)[:, :1, :1, :, :]
    vi = np.resize(vi, (nex, )+vi.shape[1:])  # (nex,1,1,nyi,nxi)

    np.random.seed(0)
    xyztomin = -0.5
    xo = np.random.uniform(xyztomin, nxi-1.5, no)
    yo = np.random.uniform(xyztomin, nyi-1.5, no)
    zo = np.random.uniform(xyztomin, nzi-1.5, no)
    to = np.random.uniform(xyztomin, nti-1.5, no)
    vo_truth = vfunc(ti, zi.ravel()[0], yo, xo)

    vo_interp = _interp.linear4dto1dxx(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_interp_rect = _interp.linear4dto1dxx(xi[:1], yi[:, :1],
                                            zi, ti, vi, xo, yo, zo, to)

    np.testing.assert_almost_equal(vo_interp[0], vo_truth)
    np.testing.assert_almost_equal(vo_interp_rect[0], vo_truth)

    # Same coordinates

    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex, )+vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)

    tzyxo = np.meshgrid(ti, zi, yi, xi, indexing='ij')
    xo = tzyxo[3].ravel()
    yo = tzyxo[2].ravel()
    zo = tzyxo[1].ravel()
    to = tzyxo[0].ravel()
    vo_truth = vfunc(to, zo, yo, xo)

    vo_interp = _interp.linear4dto1dxx(xi, yi, zi, ti, vi, xo, yo, zo, to)

    np.testing.assert_almost_equal(vo_interp[0], vo_truth)


def test_interp_linear1d():

    # coords
    nx = 17
    nyi = 10
    nyo = 30
    yi = np.linspace(-1000., 0., nyi)
    yo = np.linspace(-1200, 100, nyo)
    x = np.arange(nx)

    # input
    yyi, xxi = np.meshgrid(yi, x)
    vari = vfunc(y=yyi, x=xxi)
    vari[int(nx/3):int(2*nx/3), int(nyi/3):int(2*nyi/3)] = np.nan

    # nearest
    varon = _interp.linear1d(vari, yi, yo, 0, extrap=0)
    assert not np.isnan(varon).all()
    yyon = _interp.linear1d(yyi, yi, yo, 0, extrap=0)
    xxon = _interp.linear1d(xxi, yi, yo, 0, extrap=0)
    varon_true = vfunc(y=yyon, x=xxon)
    varon_true[np.isnan(varon)] = np.nan
    np.testing.assert_allclose(varon_true, varon)

    # linear
    yyo, xxo = np.meshgrid(yo, x)
    varol = _interp.linear1d(vari, yi, yo, 1, extrap=0)
    assert not np.isnan(varol).all()
    varol_true = vfunc(y=yyo, x=xxo)
    varol_true[np.isnan(varol)] = np.nan
    np.testing.assert_allclose(varol_true, varol)

    # cubic
    varoh = _interp.linear1d(vari, yi, yo, 3, extrap=0)
    assert not np.isnan(varoh).all()
    assert np.nanmax(varoh) <= np.nanmax(vari)
    assert np.nanmin(varoh) >= np.nanmin(vari)

    # extrap
    varole = _interp.linear1d(vari, yi, yo, 1, extrap=2)
    vv = vari[-1, ~np.isnan(vari[-1])]
    np.testing.assert_allclose(varole[-1, [0, -1]], vv[[0, -1]])


def test_interp_linear1dx():

    # coords
    nx = 17
    nyi = 10
    nyo = 30
    yi = np.linspace(-1000., 0., nyi)
    yo = np.linspace(-1200, 100, nyo)
    x = np.arange(nx)
    dyi = (yi[1]-yi[0])*0.49
    yyi = np.resize(yi, (nx, nyi))
    yyi += np.random.uniform(-dyi, dyi, (nx, nyi))

    # input
    xxi = np.resize(x, (nyi, nx)).T
    vari = vfunc(y=yyi, x=xxi)
    vari[int(nx/3):int(2*nx/3), int(nyi/3):int(2*nyi/3)] = np.nan
    # vari[:] = yyi

    # nearest
    varon = _interp.linear1dx(vari, yyi, yo, 0, extrap=0)
    assert not np.isnan(varon).all()
    yyon = _interp.linear1dx(yyi, yyi, yo, 0, extrap=0)
    xxon = _interp.linear1dx(xxi, yyi, yo, 0, extrap=0)
    varon_true = vfunc(y=yyon, x=xxon)
    varon_true[np.isnan(varon)] = np.nan
    np.testing.assert_allclose(varon_true, varon)

    # linear
    yyo, xxo = np.meshgrid(yo, x)
    varol = _interp.linear1dx(vari, yyi, yo, 1, extrap=0)
    assert not np.isnan(varol).all()
    varol_true = vfunc(y=yyo, x=xxo)
    varol_true[np.isnan(varol)] = np.nan
    np.testing.assert_allclose(varol_true, varol)

    # cubic
    varoh = _interp.linear1dx(vari, yyi, yo, 3, extrap=0)
    assert not np.isnan(varoh).all()
    assert np.nanmax(varoh) <= np.nanmax(vari)
    assert np.nanmin(varoh) >= np.nanmin(vari)

    # extrap
    varole = _interp.linear1dx(vari, yyi, yo, 1, extrap=2)
    vv = vari[-1, ~np.isnan(vari[-1])]
    np.testing.assert_allclose(varole[-1, [0, -1]], vv[[0, -1]])


def test_interp_linear1dxx():

    # coords
    nx = 17
    nyi = 10
    nyo = 30
    yi = np.linspace(-1000., 0., nyi)
    yo = np.linspace(-1200, 100, nyo)
    x = np.arange(nx)
    dyi = (yi[1]-yi[0])*0.49
    yyi = np.resize(yi, (nx, nyi))
    yyi += np.random.uniform(-dyi, dyi, (nx, nyi))
    dyo = (yo[1]-yo[0])*0.49
    yyo = np.resize(yo, (nx, nyo))
    yyo += +np.random.uniform(-dyo, dyo, (nx, nyo))

    # input
    xxi = np.resize(x, (nyi, nx)).T
    vari = vfunc(y=yyi, x=xxi)
    vari[int(nx/3):int(2*nx/3), int(nyi/3):int(2*nyi/3)] = np.nan
    # vari[:] = yyi

    # nearest
    varon = _interp.linear1dxx(vari, yyi, yyo, 0, extrap=0)
    assert not np.isnan(varon).all()
    yyon = _interp.linear1dxx(yyi, yyi, yyo, 0, extrap=0)
    xxon = _interp.linear1dxx(xxi, yyi, yyo, 0, extrap=0)
    varon_true = vfunc(y=yyon, x=xxon)
    varon_true[np.isnan(varon)] = np.nan
    np.testing.assert_allclose(varon_true, varon)

    # linear
    xxo = np.resize(x, (nyo, nx)).T
    varol = _interp.linear1dxx(vari, yyi, yyo, 1, extrap=0)
    assert not np.isnan(varol).all()
    varol_true = vfunc(y=yyo, x=xxo)
    varol_true[np.isnan(varol)] = np.nan
    np.testing.assert_allclose(varol_true, varol)

    # cubic
    varoh = _interp.linear1dxx(vari, yyi, yyo, 3, extrap=0)
    assert not np.isnan(varoh).all()
    assert np.nanmax(varoh) <= np.nanmax(vari)
    assert np.nanmin(varoh) >= np.nanmin(vari)

    # extrap
    varole = _interp.linear1dxx(vari, yyi, yyo, 1, extrap=2)
    vv = vari[-1, ~np.isnan(vari[-1])]
    np.testing.assert_allclose(varole[-1, [0, -1]], vv[[0, -1]])


def test_interp_cellave1d():

    # coords
    nx = 17
    nyi = 20
    nyo = 12
    yib = np.linspace(-1000., 0., nyi+1)
    yob = np.linspace(-1200, 200, nyo+1)
    xb = np.arange(nx+1)
    xxib, yyib = np.meshgrid(xb, yib)
    xxob, yyob = np.meshgrid(xb, yob)

    # input
    u, v = np.mgrid[-3:3:nx*1j, -3:3:nyi*1j]-2
    vari = np.asarray(u**2+v**2)
    vari[int(nx/3):int(2*nx/3), int(nyi/3):int(2*nyi/3)] = np.nan

    # conserv, no extrap
    varoc = _interp.cellave1d(vari, yib, yob, conserv=1, extrap=0)
    sumi = np.nansum(vari*np.resize(np.diff(yib), vari.shape), axis=1)
    sumo = np.nansum(varoc*np.resize(np.diff(yob), varoc.shape), axis=1)
    np.testing.assert_allclose(sumi, sumo)

    # average, no extrap
    varoa = _interp.cellave1d(vari, yib, yob, conserv=0, extrap=0)
    np.testing.assert_allclose([np.nanmin(vari), np.nanmax(vari)],
                               [np.nanmin(vari), np.nanmax(vari)])

    # average, extrap
    varoe = _interp.cellave1d(vari, yib, yob, conserv=0, extrap=2)
    assert not np.isnan(varoe[0]).any()
    vv = varoa[0, ~np.isnan(varoa[0])]
    np.testing.assert_allclose(vv[[0, -1]], varoe[0, [0, -1]])


def test_interp_cellave1dx():

    # coords
    nx = 17
    nyi = 20
    nyo = 12
    yib = np.linspace(-1000., 0., nyi+1)
    yob = np.linspace(-1200, 200, nyo+1)
    yyib = np.resize(yib, (nx, nyi+1))
    dyi = (yib[1]-yib[0])*0.49
    np.random.seed(0)
    yyib += np.random.uniform(-dyi, dyi, yyib.shape)

    # input
    u, v = np.mgrid[-3:3:nx*1j, -3:3:nyi*1j]-2
    vari = np.asarray(u**2+v**2)
    vari[int(nx/3):int(2*nx/3), int(nyi/3):int(2*nyi/3)] = np.nan

    # conserv, no extrap
    varoc = _interp.cellave1dx(vari, yyib, yob, conserv=1, extrap=0)
    sumi = np.nansum(vari*np.diff(yyib, axis=1), axis=1)
    sumo = np.nansum(varoc*np.resize(np.diff(yob), varoc.shape), axis=1)
    np.testing.assert_allclose(sumi, sumo)

    # average, no extrap
    _interp.cellave1dx(vari, yyib, yob, conserv=0, extrap=0)

    # average, extrap
    varoe = _interp.cellave1dx(vari, yyib, yob, conserv=0, extrap=2)
    assert not np.isnan(varoe[0]).any()


def test_interp_cellave1dxx():

    # coords
    nx = 17
    nyi = 20
    nyo = 12
    yib = np.linspace(-1000., 0., nyi+1)
    yob = np.linspace(-1200, 200, nyo+1)
    yyib = np.resize(yib, (nx, nyi+1))
    dyi = (yib[1]-yib[0])*0.49
    np.random.seed(0)
    yyib += np.random.uniform(-dyi, dyi, yyib.shape)
    yyob = np.resize(yob, (nx, nyo+1))
    dyo = (yob[1]-yob[0])*0.49
    yyob += np.random.uniform(-dyo, dyo, yyob.shape)

    # input
    u, v = np.mgrid[-3:3:nx*1j, -3:3:nyi*1j]-2
    vari = np.asarray(u**2+v**2)
    vari[int(nx/3):int(2*nx/3), int(nyi/3):int(2*nyi/3)] = np.nan

    # conserv, no extrap
    varoc = _interp.cellave1dxx(vari, yyib, yyob, conserv=1, extrap=0)
    sumi = np.nansum(vari*np.diff(yyib, axis=1), axis=1)
    sumo = np.nansum(varoc*np.diff(yyob, axis=1), axis=1)
    np.testing.assert_allclose(sumi, sumo)

    # average, no extrap
    _interp.cellave1dxx(vari, yyib, yyob, conserv=0, extrap=0)

    # average, extrap
    varoe = _interp.cellave1dxx(vari, yyib, yyob, conserv=0, extrap=2)
    assert not np.isnan(varoe[0]).any()
