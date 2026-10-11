# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.dyn` module
"""

import numpy as np
import pytest

from xoa.core import dyn


def _grid(ny=20, nx=30, d=1000.0):
    y, x = np.mgrid[:ny, :nx] * d
    return x, y


def _reference(u, v, dx, dy):
    ux = np.gradient(u, axis=-1) / dx
    uy = np.gradient(u, axis=-2) / dy
    vx = np.gradient(v, axis=-1) / dx
    vy = np.gradient(v, axis=-2) / dy
    return ux + vy, ux - vy, vx + uy, vx - uy


class TestKinematics:
    def test_solid_body_rotation(self):
        x, y = _grid()
        w = 1e-4
        u, v = -w * y, w * x
        assert np.allclose(dyn.relvort(u, v, 1000.0, 1000.0), 2 * w)
        assert np.allclose(dyn.div(u, v, 1000.0, 1000.0), 0, atol=1e-12)
        assert np.allclose(dyn.okuboweiss(u, v, 1000.0, 1000.0), -4 * w**2)

    def test_pure_strain(self):
        x, y = _grid()
        s = 1e-4
        u, v = s * x, -s * y
        assert np.allclose(dyn.div(u, v, 1000.0, 1000.0), 0, atol=1e-12)
        assert np.allclose(dyn.relvort(u, v, 1000.0, 1000.0), 0, atol=1e-12)
        assert np.allclose(dyn.okuboweiss(u, v, 1000.0, 1000.0), 4 * s**2)
        sn, ss = dyn.strain(u, v, 1000.0, 1000.0)
        assert np.allclose(sn, 2 * s)
        assert np.allclose(ss, 0, atol=1e-12)

    def test_pure_divergence(self):
        x, y = _grid()
        d = 1e-5
        assert np.allclose(dyn.div(d * x, d * y, 1000.0, 1000.0), 2 * d)

    def test_uniform_flow(self):
        u = np.full((10, 12), 0.5)
        v = np.full((10, 12), -0.2)
        for res in dyn.kinematics(u, v, 500.0, 500.0):
            assert np.allclose(res, 0)

    def test_matches_numpy_gradient_with_variable_resolution(self):
        rng = np.random.default_rng(0)
        u = rng.normal(size=(15, 25))
        v = rng.normal(size=(15, 25))
        dx = rng.uniform(800, 1200, size=u.shape)
        dy = rng.uniform(800, 1200, size=u.shape)
        for res, ref in zip(dyn.kinematics(u, v, dx, dy), _reference(u, v, dx, dy)):
            assert np.allclose(res, ref)

    def test_nan_propagation(self):
        rng = np.random.default_rng(1)
        u = rng.normal(size=(8, 9))
        v = rng.normal(size=(8, 9))
        u[3, 4] = np.nan
        v[5, 2] = np.nan
        for res in dyn.kinematics(u, v, 1.0, 1.0):
            assert np.isnan(res[3, 4]) and np.isnan(res[5, 2])
            assert not np.isnan(res[0, 0])
        # A neighbor is used only by the gradients that need it
        assert np.isnan(dyn.div(u, v, 1.0, 1.0)[3, 3])
        assert not np.isnan(dyn.relvort(u, v, 1.0, 1.0)[3, 3])

    def test_leading_dims(self):
        rng = np.random.default_rng(2)
        u = rng.normal(size=(3, 2, 10, 12))
        v = rng.normal(size=(3, 2, 10, 12))
        res = dyn.relvort(u, v, 1000.0, 2000.0)
        assert res.shape == u.shape
        assert np.allclose(res[1, 0], dyn.relvort(u[1, 0], v[1, 0], 1000.0, 2000.0))

    def test_integer_inputs(self):
        u = np.arange(20).reshape(4, 5)
        v = np.arange(20).reshape(4, 5)
        assert dyn.div(u, v, 1, 1).dtype == np.float64


class TestGeostrophy:
    def test_coriolis(self):
        assert dyn.coriolis(0.0) == 0
        assert np.isclose(dyn.coriolis(90.0), 2 * dyn.OMEGA)
        assert np.isclose(dyn.coriolis(-30.0), -dyn.OMEGA)

    def test_geos_linear_ssh(self):
        x, y = _grid()
        f = 1e-4
        dhdy, dhdx = 1e-6, 2e-6
        ssh = dhdx * x + dhdy * y
        ug, vg = dyn.geos(ssh, 1000.0, 1000.0, f)
        assert np.allclose(ug, -dyn.GRAVITY / f * dhdy)
        assert np.allclose(vg, dyn.GRAVITY / f * dhdx)

    @pytest.mark.parametrize("shape", [(20,), (20, 30)])
    def test_geos_coriolis_shapes(self, shape):
        _, y = _grid()
        corio = np.full(shape, 1e-4)
        ug, _ = dyn.geos(1e-6 * y, 1000.0, 1000.0, corio)
        assert ug.shape == y.shape
        assert np.allclose(ug, -dyn.GRAVITY / 1e-4 * 1e-6)

    def test_geos_null_coriolis_and_nan(self):
        ssh = np.random.default_rng(3).normal(size=(6, 7))
        ssh[2, 2] = np.nan
        corio = np.full(6, 1e-4)
        corio[0] = 0.0
        ug, vg = dyn.geos(ssh, 1.0, 1.0, corio)
        assert np.isnan(ug[0]).all() and np.isnan(vg[0]).all()
        assert np.isnan(ug[2, 2])
        assert not np.isnan(ug[4, 4])
