# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.tide` module
"""

import numpy as np

from xoa.core import tide as ctide


def test_doodson_multipliers():
    np.testing.assert_allclose(ctide.doodson_multipliers("255.555"), [2, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(ctide.doodson_multipliers("165.555"), [1, 1, 0, 0, 0, 0])


def test_astronomical_args_ranges():
    for arg in ctide.astronomical_args(1.7e9):
        assert 0 <= arg < 360


def test_phase_conventions_roundtrip():
    rng = np.random.default_rng(0)
    g = rng.uniform(0, 2 * np.pi, (3, 5))
    v0, u = rng.uniform(0, 2 * np.pi, 3), rng.uniform(-0.3, 0.3, 3)
    local = ctide.greenwich_to_local(g, v0, u)
    assert local.min() > -np.pi and local.max() <= np.pi
    np.testing.assert_allclose(ctide.local_to_greenwich(local, v0, u), g % (2 * np.pi))


def test_ssh_solve_predict_roundtrip():
    t = np.arange(500) * 3600.0
    omega = 2 * np.pi / (np.array([12.42, 24.0]) * 3600)
    amp, phase = np.array([[1.0, 2.0], [0.3, 0.1]]), np.array([[0.5, 1.0], [2.0, -1.0]])
    h0 = np.array([0.2, -0.1])
    h = ctide.predict_ssh(t, omega, h0, amp, phase)
    Ai = ctide.design_matrix_pinv(ctide.design_matrix(t, omega))
    h0_, amp_, phase_ = ctide.solve_ssh(Ai, h)
    np.testing.assert_allclose(h0_, h0, atol=1e-8)
    np.testing.assert_allclose(amp_, amp, atol=1e-8)
    np.testing.assert_allclose(np.cos(phase_), np.cos(phase), atol=1e-8)


def test_rotary_ellipse_roundtrip():
    smaj, smin, incl, phase = 0.8, -0.3, 0.4, 1.1
    wp, wm = ctide.rotary_components(smaj, smin, incl, phase)
    out = ctide.ellipse_from_rotary(wp, wm)
    np.testing.assert_allclose(out, [smaj, smin, incl, phase])
    ellipticity, rotation = ctide.ellipse_properties(smaj, smin)
    assert ellipticity == smin / smaj and rotation == -1
    assert np.isnan(ctide.ellipse_properties(0.0, 0.0)[0])
    u, v = ctide.ellipse_vector(smaj, smin, incl, phase, np.array([0.0, np.pi]))
    np.testing.assert_allclose([u[0], v[0]], -np.array([u[1], v[1]]), atol=1e-12)


def test_high_waters_skew_surge_and_springs():
    h = np.cos(2 * np.pi * np.arange(24 * 30) / 12.42) + 0.3 * np.cos(
        2 * np.pi * np.arange(720) / 354
    )
    hw = ctide.high_water_mask(h)
    assert abs(hw.sum() - 720 / 12.42) <= 2
    surge = ctide.skew_surge_series(h + 0.1, h, hw)
    np.testing.assert_allclose(surge[hw & ~np.isnan(surge)], 0.1)
    spring = ctide.spring_mask(h, hw)
    assert 0 < spring.sum() < hw.sum() and not (spring & ~hw).any()
