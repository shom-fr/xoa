# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.interp` module
"""

import functools
import numpy as np
import pytest

from xoa.core import regrid


def vfunc(t=0, z=0, y=0, x=0):
    """A function that returns a linear combination of coordinates"""
    return 1.13 * x + 12.35 * y + 3.24 * z - 0.65 * t


@functools.lru_cache()
def get_regrid1d_data(
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


def test_nearest1d():
    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()

    # Interpolation
    varon = regrid.nearest1d(vari, yyi, yyo, eshapes)
    yyon = regrid.nearest1d(yyi, yyi, yyo, eshapes)
    xxon = regrid.nearest1d(xxi, yyi, yyo, eshapes)
    varon_true = vfunc(y=yyon, x=xxon)
    varon_true[np.isnan(varon)] = np.nan
    np.testing.assert_allclose(varon_true, varon)


def test_linear1d():
    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()

    # Interpolation
    varol = regrid.linear1d(vari, yyi, yyo, eshapes)
    assert not np.isnan(varol).all()
    varol_true = vfunc(y=yyo, x=xxo)
    varol_true[np.isnan(varol)] = np.nan
    np.testing.assert_allclose(varol_true, varol)


def test_cubic1d():
    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()

    # Interpolation
    varoh = regrid.cubic1d(vari, yyi, yyo, eshapes)
    assert not np.isnan(varoh).all()
    assert np.nanmax(varoh) <= np.nanmax(vari)
    assert np.nanmin(varoh) >= np.nanmin(vari)


def test_hermit1d():
    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()

    # Interpolation
    varoh = regrid.hermit1d(vari, yyi, yyo, eshapes)
    assert not np.isnan(varoh).all()
    assert np.nanmax(varoh) <= np.nanmax(vari)
    assert np.nanmin(varoh) >= np.nanmin(vari)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "hermit"])
def test_regrid1d_nans_in_coords(method):
    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data()

    # Add nans to coords
    yyin = yyi.copy()
    yyin[:, :3] = np.nan
    yyin[:, -3:] = np.nan
    yyon = yyo.copy()
    yyon[:, :3] = np.nan
    yyon[:, -3:] = np.nan

    # Interpolations
    func = getattr(regrid, method + "1d")
    varol = func(vari[:, 3:-3], yyi[:, 3:-3], yyo[:, 3:-3], eshapes)
    varoln = func(vari, yyin, yyon, eshapes)
    np.testing.assert_allclose(varol, varoln[:, 3:-3])


def test_linear1d_drop_na():
    # Get data
    xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data(yomax=-20)

    varo = regrid.linear1d(vari, yyi, yyo, eshapes, drop_na=True)
    assert not np.isnan(varo).any()
    varo_true = vfunc(y=yyo, x=xxo)
    np.testing.assert_allclose(varo_true, varo)

    varo0 = regrid.linear1d(vari, yyi, yyo, eshapes, drop_na=False)
    varo1 = regrid.linear1d(vari, yyi, yyo, eshapes, drop_na=True, maxgap=1)
    np.testing.assert_allclose(varo0, varo1)


@pytest.mark.parametrize("method", ["nearest", "linear", "cubic", "hermit"])
def test_interp1d_eshapes(method):
    # Get data and func
    xxi, yyi, vari, xxo, yyo, eshapes = get_regrid1d_data(nx=18, mask=False, irregular=False)
    eshapes = np.repeat([[3, 6]], 3, axis=0)
    func = getattr(regrid, method + "1d")

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


def test_cellave1d():
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
    varoc = regrid.cellave1d(vari, yyib, yyob, eshapes, conserv=True, extrap="no")
    sumi = np.nansum(vari * np.diff(yyib, axis=1), axis=1)
    sumo = np.nansum(varoc * np.diff(yyob, axis=1), axis=1)
    np.testing.assert_allclose(sumi, sumo)

    # average, no extrap
    regrid.cellave1d(vari, yyib, yyob, eshapes, conserv=0, extrap="no")

    # average, extrap
    varoe = regrid.cellave1d(vari, yyib, yyob, eshapes, conserv=False, extrap="both")
    assert not np.isnan(varoe[0]).any()
