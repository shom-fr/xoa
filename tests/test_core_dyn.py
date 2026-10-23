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


def _eddy(amp, radius=50e3, f=1e-4, res=5e3, n=121):
    """Gaussian sea surface height eddy, whose gradient wind solution is known

    A positive amplitude is an anticyclone in the northern hemisphere.
    """
    x = (np.arange(n) - n // 2) * res
    xx, yy = np.meshgrid(x, x)
    rr = np.hypot(xx, yy)
    eta = amp * np.exp(-(rr**2) / (2 * radius**2))
    vg = -dyn.GRAVITY / f * amp * rr / radius**2 * np.exp(-(rr**2) / (2 * radius**2))
    with np.errstate(invalid="ignore"):
        # Azimuthal gradient wind: V^2 / r + f V = f Vg
        vgw = 0.5 * (-f * rr + np.sqrt(f**2 * rr**2 + 4 * rr * f * vg))
    return eta, xx, yy, rr, vgw, res, f


def _azimuthal(u, v, xx, yy, rr):
    rr = np.where(rr == 0, np.inf, rr)
    return (-yy * u + xx * v) / rr


class TestCyclogeos:
    @pytest.mark.parametrize("amp", [0.25, 0.1, -0.1, -0.25])
    def test_gradient_wind(self, amp):
        eta, xx, yy, rr, vgw, res, f = _eddy(amp)
        ug, vg = dyn.geos(eta, res, res, f)
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full")
        assert not np.isnan(uc).any()
        err = np.abs(_azimuthal(uc, vc, xx, yy, rr) - vgw)[1:-1, 1:-1]
        assert err.max() < 0.01 * np.abs(vgw).max()
        # Far from the eddy, the geostrophic currents are not modified
        assert abs(_azimuthal(uc, vc, xx, yy, rr)[60, 5] - vgw[60, 5]) < 1e-3

    def test_cyclones_are_slower_and_anticyclones_faster(self):
        speeds = {}
        for amp in (0.25, -0.25):
            eta, *_, res, f = _eddy(amp)
            ug, vg = dyn.geos(eta, res, res, f)
            uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full")
            speeds[amp] = np.hypot(uc, vc).max() / np.hypot(ug, vg).max()
        assert speeds[0.25] > 1.03  # Anticyclone: the amplitude is positive in the north
        assert speeds[-0.25] < 0.97  # Cyclone

    def test_converged_solution_is_a_balance(self):
        eta, *_, res, f = _eddy(0.25)
        ug, vg = dyn.geos(eta, res, res, f)
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full", smooth=0, tol=1e-9)
        ux, uy = np.gradient(uc, res, axis=1), np.gradient(uc, res, axis=0)
        vx, vy = np.gradient(vc, res, axis=1), np.gradient(vc, res, axis=0)
        ax, ay = uc * ux + vc * uy, uc * vx + vc * vy
        resx = -f * (vc - vg) + ax
        resy = f * (uc - ug) + ay
        assert np.abs(np.hypot(resx, resy))[2:-2, 2:-2].max() < 1e-3 * f * np.hypot(ug, vg).max()

    def test_small_rossby_number_is_geostrophic(self):
        eta, *_, res, f = _eddy(0.001)
        ug, vg = dyn.geos(eta, res, res, f)
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full")
        np.testing.assert_allclose(uc, ug, atol=1e-3 * np.abs(ug).max())

    def test_uniform_currents_are_unchanged(self):
        ug = np.full((10, 12), 0.3)
        vg = np.full((10, 12), -0.2)
        uc, vc = dyn.cyclogeos(ug, vg, 5e3, 5e3, 1e-4)
        np.testing.assert_allclose(uc, ug)
        np.testing.assert_allclose(vc, vg)

    @pytest.mark.parametrize("relax", [1.0, 0.5, 0.25])
    def test_relaxation_gives_the_same_solution(self, relax):
        eta, *_, res, f = _eddy(-0.25)
        ug, vg = dyn.geos(eta, res, res, f)
        ref = dyn.cyclogeos(ug, vg, res, res, f, terms="full", tol=1e-9, relax=1.0)
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full", tol=1e-9, relax=relax, niter=1000)
        np.testing.assert_allclose(uc, ref[0], atol=1e-6)
        np.testing.assert_allclose(vc, ref[1], atol=1e-6)

    def test_strong_anticyclones_have_no_solution(self):
        eta, *_, res, f = _eddy(1.0)
        ug, vg = dyn.geos(eta, res, res, f)
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full")
        bad = np.isnan(uc)
        assert bad.any() and not bad.all()
        assert np.isfinite(uc[0, 0])  # Far from the eddy
        assert (np.hypot(uc, vc)[~bad] < 3 * np.hypot(ug, vg).max()).all()

    def test_not_enough_iterations(self):
        eta, *_, res, f = _eddy(0.25)
        ug, vg = dyn.geos(eta, res, res, f)
        uc, _ = dyn.cyclogeos(ug, vg, res, res, f, terms="full", niter=1, tol=1e-12)
        assert np.isnan(uc[60, 60 - 10])

    def test_missing_values_and_equator(self):
        eta, *_, res, f = _eddy(0.1)
        ug, vg = dyn.geos(eta, res, res, f)
        ug[10, 10] = np.nan
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full")
        # The neighbors of a missing value are missing, but the others are not
        assert np.isnan(uc[10, 10]) and np.isnan(uc[10, 11]) and np.isnan(uc[9, 10])
        assert np.isnan(uc).sum() < 10
        assert not np.isnan(uc[60, 60])
        # Null Coriolis parameter
        corio = np.full(ug.shape[0], f)
        corio[:3] = 1e-7
        uc, _ = dyn.cyclogeos(ug, vg, res, res, corio, terms="full")
        assert np.isnan(uc[:3]).all() and not np.isnan(uc[60, 60])

    def test_leading_dims_and_variable_resolution(self):
        eta, *_, res, f = _eddy(0.1, n=41)
        eta = np.stack([eta, 0.5 * eta])
        dx = np.full(eta.shape[-2:], res)
        ug, vg = dyn.geos(eta, dx, res, f)
        uc, vc = dyn.cyclogeos(ug, vg, dx, res, f)
        assert uc.shape == eta.shape
        np.testing.assert_allclose(uc[0], dyn.cyclogeos(ug[0], vg[0], res, res, f)[0])

    @pytest.mark.parametrize("amp", [0.25, -0.25])
    def test_penven_terms_follow_the_published_iterations(self, amp):
        """The default correction has the two terms u dudx and v dvdy, starting from zero"""
        eta, *_, res, f = _eddy(amp)
        ug, vg = dyn.geos(eta, res, res, f)
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="penven", relax=1.0, tol=1e-10, niter=500)
        u = np.zeros_like(ug)
        v = np.zeros_like(vg)
        for _ in range(200):
            u, v = (
                ug - v * np.gradient(v, res, axis=0) / f,
                vg + u * np.gradient(u, res, axis=1) / f,
            )
        np.testing.assert_allclose(uc[2:-2, 2:-2], u[2:-2, 2:-2], atol=1e-8)
        np.testing.assert_allclose(vc[2:-2, 2:-2], v[2:-2, 2:-2], atol=1e-8)

    def test_penven_terms_are_a_partial_correction(self):
        """They go the same way as the full advection, but not as far"""
        eta, *_, res, f = _eddy(0.25)
        ug, vg = dyn.geos(eta, res, res, f)
        up, vp = dyn.cyclogeos(ug, vg, res, res, f, terms="penven")
        uf, vf = dyn.cyclogeos(ug, vg, res, res, f)
        dp = np.hypot(up - ug, vp - vg).max()
        df = np.hypot(uf - ug, vf - vg).max()
        assert 0.1 * df < dp < 0.7 * df
        assert np.hypot(up, vp).max() > np.hypot(ug, vg).max()

    def test_smoothing_stabilizes_the_iterations(self):
        """The iterations are ill-posed at the grid scale: they diverge when the resolution
        gets fine, even for a cyclone that always has a solution"""
        eta, xx, yy, rr, vgw, res, f = _eddy(-0.3, radius=30e3, res=1e3)
        ug, vg = dyn.geos(eta, res, res, f)
        raw = dyn.cyclogeos(ug, vg, res, res, f, terms="full", smooth=0)[0]
        assert np.isnan(raw).mean() > 0.3
        uc, vc = dyn.cyclogeos(ug, vg, res, res, f, terms="full")
        assert not np.isnan(uc).any()
        err = np.abs(_azimuthal(uc, vc, xx, yy, rr) - vgw)[1:-1, 1:-1]
        assert err.max() < 0.02 * np.abs(vgw).max()
        # The smoothing is the default with the full terms, which are the default, only
        np.testing.assert_allclose(uc, dyn.cyclogeos(ug, vg, res, res, f, smooth=1)[0])
        eta, *_, res, f = _eddy(0.25)
        ug, vg = dyn.geos(eta, res, res, f)
        np.testing.assert_allclose(
            dyn.cyclogeos(ug, vg, res, res, f, terms="penven")[0],
            dyn.cyclogeos(ug, vg, res, res, f, terms="penven", smooth=0)[0],
        )

    def test_invalid_terms(self):
        with pytest.raises(ValueError, match="terms"):
            dyn.cyclogeos(np.zeros((3, 3)), np.zeros((3, 3)), 1.0, 1.0, 1e-4, terms="other")


def _sloping_levels(nz=8, ny=18, nx=22, res=2000.0, omega=1e-5, shear=3e-4):
    """Solid body rotation with a vertical shear, on levels that follow a sloping bottom

    The fields are linear in x, y and z, so that the horizontal derivatives at constant z
    are known: they are not the ones along the levels, since the levels are sloping.
    """
    x = (np.arange(nx) - nx / 2) * res
    y = (np.arange(ny) - ny / 2) * res
    xx, yy = np.meshgrid(x, y)
    depth = 200.0 + 60.0 * xx / x.max() + 40.0 * yy / y.max()
    sigma = np.linspace(-1, 0, nz)[:, None, None]
    z = sigma * depth
    u = -omega * yy + shear * z
    v = omega * xx - 0.5 * shear * z
    return u, v, z, res, omega, shear


class TestKinematicsZ:
    def test_constant_z_derivatives(self):
        u, v, z, res, omega, shear = _sloping_levels()
        div, sn, ss, vort = dyn.kinematics(u, v, res, res, z)
        assert vort.shape == u.shape
        np.testing.assert_allclose(vort, 2 * omega, rtol=1e-9)
        np.testing.assert_allclose(div, 0, atol=1e-15)
        np.testing.assert_allclose(sn, 0, atol=1e-15)
        np.testing.assert_allclose(ss, 0, atol=1e-15)
        np.testing.assert_allclose(dyn.relvort(u, v, res, res, z), vort)
        np.testing.assert_allclose(dyn.okuboweiss(u, v, res, res, z), -4 * omega**2, rtol=1e-9)
        np.testing.assert_allclose(dyn.div(u, v, res, res, z), div)
        np.testing.assert_allclose(dyn.strain(u, v, res, res, z)[0], sn)

    def test_along_the_levels_is_wrong(self):
        u, v, z, res, omega, shear = _sloping_levels()
        naive = dyn.relvort(u, v, res, res)
        assert np.abs(naive - 2 * omega).max() > 0.1 * omega

    def test_flat_levels_give_the_2d_result(self):
        u, v, z, res, omega, shear = _sloping_levels()
        z = np.broadcast_to(np.linspace(-100, 0, 8)[:, None, None], u.shape)
        np.testing.assert_allclose(
            dyn.relvort(u, v, res, res, z), dyn.relvort(u, v, res, res), atol=1e-15
        )

    def test_sign_and_order_of_z(self):
        u, v, z, res, omega, shear = _sloping_levels()
        ref = dyn.relvort(u, v, res, res, z)
        # Depths, which are positive down
        np.testing.assert_allclose(dyn.relvort(u, v, res, res, -z), ref, atol=1e-15)
        # Levels from the surface to the bottom
        np.testing.assert_allclose(
            dyn.relvort(u[::-1], v[::-1], res, res, z[::-1]), ref[::-1], atol=1e-15
        )

    def test_missing_values(self):
        u, v, z, res, omega, shear = _sloping_levels()
        u[3, 5, 6] = np.nan
        vort = dyn.relvort(u, v, res, res, z)
        assert np.isnan(vort[3, 5, 6])
        assert not np.isnan(vort[0, 0, 0])
        z = z.copy()
        z[2, 4, 4] = np.nan
        assert np.isnan(dyn.relvort(u, v, res, res, z)[2, 4, 4])

    def test_leading_dims_and_variable_resolution(self):
        u, v, z, res, omega, shear = _sloping_levels()
        uu = np.stack([u, 2 * u])
        vv = np.stack([v, 2 * v])
        dx = np.full(u.shape[-2:], res)
        vort = dyn.relvort(uu, vv, dx, res, z)
        assert vort.shape == uu.shape
        np.testing.assert_allclose(vort[0], 2 * omega, rtol=1e-9)
        # The same z for all the times, or one per time
        np.testing.assert_allclose(dyn.relvort(uu, vv, dx, res, np.stack([z, z])), vort)

    def test_bad_shapes(self):
        u, v, z, res, omega, shear = _sloping_levels()
        with pytest.raises(ValueError, match="same last three dimensions"):
            dyn.relvort(u, v, res, res, z[:-1])
        with pytest.raises(ValueError, match="same last three dimensions"):
            dyn.relvort(u[0], v[0], res, res, z[0])
