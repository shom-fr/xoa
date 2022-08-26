# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.interp` module
"""

import functools
import numpy as np
import pytest

from xoa import interp


def vfunc(t=0, z=0, y=0, x=0):
    """A function that returns a linear combination of coordinates"""
    return 1.13 * x + 2.35 * y + 3.24 * z - 0.65 * t


def round_as_time(arr, units="us", origin="1950-01-01"):
    arr = arr.astype(f"m8[{units}]")
    origin = np.datetime64(origin, units)
    arr = arr + origin
    return (arr - origin) / np.timedelta64(1, units)


@functools.lru_cache()
def get_interp1d_data(
    yimin=-100.0,
    yimax=0.0,
    yomin=-90.0,
    yomax=10.0,
    irregular=True,
    nx=17,
    nyi=15,
    nyo=25,
    mask=True,
):

    np.random.seed(0)

    # coords
    yi = np.linspace(yimin, yimax, nyi)
    yo = np.linspace(yomin, yomax, nyo)
    x = np.linspace(0, 700, nx)
    yyi = np.resize(yi, (nx, nyi))
    yyo = np.resize(yo, (nx, nyo))
    if irregular:
        dyi = (yi[1] - yi[0]) * 0.49
        yyi += np.random.uniform(-dyi, dyi, (nx, nyi))
        dyo = (yo[1] - yo[0]) * 0.49
        yyo += +np.random.uniform(-dyo, dyo, (nx, nyo))
    xxi = np.resize(x, (nyi, nx)).T
    xxo = np.resize(x, (nyo, nx)).T

    # input
    xxi = np.resize(x, (nyi, nx)).T
    vari = vfunc(y=yyi, x=xxi)
    if mask:
        vari[int(nx / 3) : int(2 * nx / 3), int(nyi / 3) : int(2 * nyi / 3)] = np.nan

    # shapes of extra dims
    eshapes = np.vstack((vari.shape[:-1], yyi.shape[:-1], yyo.shape[:-1]))

    return xxi, yyi, vari, xxo, yyo, eshapes


def test_interp_nearest1d():

    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_interp1d_data()

    # Interpolation
    varon = interp.nearest1d(vari, yyi, yyo, eshapes)
    yyon = interp.nearest1d(yyi, yyi, yyo, eshapes)
    xxon = interp.nearest1d(xxi, yyi, yyo, eshapes)
    varon_true = vfunc(y=yyon, x=xxon)
    varon_true[np.isnan(varon)] = np.nan
    np.testing.assert_allclose(varon_true, varon)


def test_interp_linear1d():

    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_interp1d_data()

    # Interpolation
    varol = interp.linear1d(vari, yyi, yyo, eshapes)
    assert not np.isnan(varol).all()
    varol_true = vfunc(y=yyo, x=xxo)
    varol_true[np.isnan(varol)] = np.nan
    np.testing.assert_allclose(varol_true, varol)


def test_interp_cubic1d():

    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_interp1d_data()

    # Interpolation
    varoh = interp.cubic1d(vari, yyi, yyo, eshapes)
    assert not np.isnan(varoh).all()
    assert np.nanmax(varoh) <= np.nanmax(vari)
    assert np.nanmin(varoh) >= np.nanmin(vari)


def test_interp_hermit1d():

    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_interp1d_data()

    # Interpolation
    varoh = interp.hermit1d(vari, yyi, yyo, eshapes)
    assert not np.isnan(varoh).all()
    assert np.nanmax(varoh) <= np.nanmax(vari)
    assert np.nanmin(varoh) >= np.nanmin(vari)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "hermit"])
def test_interp1d_nans_in_coords(method):

    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_interp1d_data()

    # Add nans to coords
    yyin = yyi.copy()
    yyin[:, :3] = np.nan
    yyin[:, -3:] = np.nan
    yyon = yyo.copy()
    yyon[:, :3] = np.nan
    yyon[:, -3:] = np.nan

    # Interpolations
    func = getattr(interp, method + "1d")
    varol = func(vari[:, 3:-3], yyi[:, 3:-3], yyo[:, 3:-3], eshapes)
    varoln = func(vari, yyin, yyon, eshapes)
    np.testing.assert_allclose(varol, varoln[:, 3:-3])


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "hermit"])
def test_interp_interp1d_eshapes(method):

    # Get data and func
    xxi, yyi, vari, xxo, yyo, eshapes = get_interp1d_data(nx=18, mask=False, irregular=False)
    eshapes = np.repeat([[3, 6]], 3, axis=0)
    func = getattr(interp, method + "1d")

    # Reference
    varol_ref = func(vari, yyi, yyo, eshapes).reshape(3, 6, -1)

    # missing dim 0 for vari
    vari0 = vari.reshape(3, 6, -1)[0]
    eshapes0 = eshapes.copy()
    eshapes0[0, 0] = 1
    varol0 = func(vari0, yyi, yyo, eshapes0).reshape(3, 6, -1)
    np.testing.assert_allclose(varol_ref[0], varol0[0])
    np.testing.assert_allclose(varol_ref[0], varol0[1])

    # missing dim 1 for vari
    vari1 = vari.reshape(3, 6, -1)[:, 0]
    eshapes1 = eshapes.copy()
    eshapes1[0, 1] = 1
    varol1 = func(vari1, yyi, yyo, eshapes1).reshape(3, 6, -1)
    np.testing.assert_allclose(varol_ref[:, 0], varol1[:, 0])
    np.testing.assert_allclose(varol_ref[:, 0], varol1[:, 1])

    # missing dim 0 for yyi
    yyi0 = yyi.reshape(3, 6, -1)[0]
    eshapes0 = eshapes.copy()
    eshapes0[1, 0] = 1
    varol0 = func(vari, yyi0, yyo, eshapes0).reshape(3, 6, -1)
    np.testing.assert_allclose(varol_ref, varol0)

    # missing dim 1 for yyi
    yyi1 = yyi.reshape(3, 6, -1)[:, 0]
    eshapes1 = eshapes.copy()
    eshapes1[1, 1] = 1
    varol1 = func(vari, yyi1, yyo, eshapes1).reshape(3, 6, -1)
    np.testing.assert_allclose(varol_ref, varol1)

    # missing dim 0 for yyo
    yyo0 = yyo.reshape(3, 6, -1)[0]
    eshapes0 = eshapes.copy()
    eshapes0[2, 0] = 1
    varol0 = func(vari, yyi, yyo0, eshapes0).reshape(3, 6, -1)
    np.testing.assert_allclose(varol_ref, varol0)

    # missing dim 1 for yyo
    yyo1 = yyo.reshape(3, 6, -1)[:, 0]
    eshapes1 = eshapes.copy()
    eshapes1[2, 1] = 1
    varol1 = func(vari, yyi, yyo1, eshapes1).reshape(3, 6, -1)
    np.testing.assert_allclose(varol_ref, varol1)


def test_interp_cellave1d():

    np.random.seed(0)

    # coords
    nx = 17
    nyi = 20
    nyo = 12
    yib = np.linspace(-1000.0, 0.0, nyi + 1)
    yob = np.linspace(-1200, 200, nyo + 1)
    yyib = np.resize(yib, (nx, nyi + 1))
    dyi = (yib[1] - yib[0]) * 0.49
    yyib += np.random.uniform(-dyi, dyi, yyib.shape)
    yyob = np.resize(yob, (nx, nyo + 1))
    dyo = (yob[1] - yob[0]) * 0.49
    yyob += np.random.uniform(-dyo, dyo, yyob.shape)
    eshapes = np.full((3, 1), nx)

    # input
    u, v = np.mgrid[-3 : 3 : nx * 1j, -3 : 3 : nyi * 1j] - 2
    vari = np.asarray(u**2 + v**2)
    vari[int(nx / 3) : int(2 * nx / 3), int(nyi / 3) : int(2 * nyi / 3)] = np.nan

    # conserv, no extrap
    varoc = interp.cellave1d(vari, yyib, yyob, eshapes, conserv=True, extrap="no")
    sumi = np.nansum(vari * np.diff(yyib, axis=1), axis=1)
    sumo = np.nansum(varoc * np.diff(yyob, axis=1), axis=1)
    np.testing.assert_allclose(sumi, sumo)

    # average, no extrap
    interp.cellave1d(vari, yyib, yyob, eshapes, conserv=0, extrap="no")

    # average, extrap
    varoe = interp.cellave1d(vari, yyib, yyob, eshapes, conserv=False, extrap="both")
    assert not np.isnan(varoe[0]).any()


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
