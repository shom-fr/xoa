"""
Test the :mod:`xoa.core.modes` module
"""

import numpy as np
import pytest

from xoa.core import modes

G = 9.81


def _column(nz=200, depth=4000.0, n2=1e-5, stretch=0.0):
    """Levels from the bottom to the surface, with a constant stratification

    z is positive up. A positive stretch refines the levels close to the surface.
    """
    s = (np.arange(nz) + 0.5) / nz
    if stretch:
        s = 1 - (np.exp(stretch * (1 - s)) - 1) / (np.exp(stretch) - 1)
    return -depth * (1 - s), np.full(nz, n2)


class TestVerticalModes:
    def test_constant_stratification(self):
        z, n2 = _column(nz=400)
        depth = 4000.0
        F, c, dz = modes.vertical_modes(n2, z, nmodes=5)
        assert F.shape == (5, 400) and c.shape == (5,) and dz.shape == (400,)
        # c_n = N H / (n pi), where the barotropic mode of the rigid lid is infinitely fast
        assert np.isnan(c[0])
        n = np.arange(1, 5)
        np.testing.assert_allclose(c[1:], np.sqrt(n2[0]) * depth / (n * np.pi), rtol=2e-3)
        # F_n = sqrt(2) cos(n pi z / H), which is positive at the surface
        for k in range(1, 5):
            np.testing.assert_allclose(F[k], np.sqrt(2) * np.cos(k * np.pi * z / depth), atol=0.02)
        np.testing.assert_allclose(F[0], 1.0)
        np.testing.assert_allclose(dz.sum(), depth)

    def test_orthonormality(self):
        # With a stretched grid and a realistic stratification
        z, _ = _column(nz=60, stretch=3.0)
        n2 = 1e-6 + 2e-4 * np.exp(z / 300.0)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=60)
        gram = (F * dz) @ F.T / dz.sum()
        np.testing.assert_allclose(gram, np.eye(60), atol=1e-8)
        # Speeds decrease with the mode number, and the modes oscillate more and more
        assert (np.diff(c[1:]) < 0).all()
        nodes = [(np.diff(np.sign(F[k])) != 0).sum() for k in range(1, 6)]
        assert nodes == [1, 2, 3, 4, 5]
        # The modes are positive at the surface, where the first ones are far from zero
        assert (F[:10, -1] > 0).all()

    def test_free_surface_has_a_barotropic_mode(self):
        z, n2 = _column(nz=100, n2=1e-7)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=3, free_surface=True)
        np.testing.assert_allclose(c[0], np.sqrt(G * 4000.0), rtol=0.02)
        np.testing.assert_allclose(F[0], 1.0, atol=0.01)
        # The baroclinic modes are hardly modified by the free surface
        rigid = modes.vertical_modes(n2, z, nmodes=3)
        np.testing.assert_allclose(c[1:], rigid[1][1:], rtol=0.01)

    def test_two_layers(self):
        # Two homogeneous layers separated by a thin interface, with a reduced gravity
        h1, h2, delta, gprime = 500.0, 3500.0, 20.0, 0.02
        z = np.linspace(-4000.0, 0.0, 1601)
        z = 0.5 * (z[1:] + z[:-1])
        n2 = np.full(z.size, 1e-10)
        n2[np.abs(z + h1) < 0.5 * delta] = gprime / delta
        F, c, dz = modes.vertical_modes(n2, z, nmodes=2, n2min=1e-10)
        expected = np.sqrt(gprime * h1 * h2 / (h1 + h2))
        np.testing.assert_allclose(c[1], expected, rtol=0.05)
        # The first mode is barotropic-like below the interface
        assert abs(F[1][z < -h1 - delta].std()) < 0.2 * abs(F[1][-1])

    def test_stretched_grid_is_consistent(self):
        z, n2 = _column(nz=200, stretch=4.0)
        c = modes.vertical_modes(n2, z, nmodes=3)[1]
        np.testing.assert_allclose(c[1:], 1e-5**0.5 * 4000.0 / (np.arange(1, 3) * np.pi), rtol=0.01)

    def test_level_order_does_not_matter(self):
        z, _ = _column(nz=50)
        n2 = 1e-6 + 2e-4 * np.exp(z / 300.0)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=4)
        Fr, cr, dzr = modes.vertical_modes(n2[::-1], z[::-1], nmodes=4)
        np.testing.assert_allclose(cr, c, equal_nan=True)
        np.testing.assert_allclose(Fr[:, ::-1], F, atol=1e-10)
        np.testing.assert_allclose(dzr[::-1], dz)

    def test_missing_levels_below_the_bottom(self):
        z, _ = _column(nz=50)
        n2 = 1e-6 + 2e-4 * np.exp(z / 300.0)
        # Shallower column: the deepest levels are missing
        zm, n2m = z.copy(), n2.copy()
        zm[:10] = np.nan
        n2m[:10] = np.nan
        F, c, dz = modes.vertical_modes(n2m, zm, nmodes=3)
        Fref, cref, dzref = modes.vertical_modes(n2[10:], z[10:], nmodes=3)
        np.testing.assert_allclose(c, cref, equal_nan=True)
        np.testing.assert_allclose(F[:, 10:], Fref, atol=1e-10)
        assert np.isnan(F[:, :10]).all() and np.isnan(dz[:10]).all()
        np.testing.assert_allclose(dz[10:], dzref)
        # Only the stratification is missing
        n2m = n2.copy()
        n2m[:10] = np.nan
        np.testing.assert_allclose(modes.vertical_modes(n2m, z, nmodes=3)[1], cref, equal_nan=True)

    def test_empty_and_too_short_columns(self):
        z, n2 = _column(nz=20)
        F, c, dz = modes.vertical_modes(np.full(20, np.nan), np.full(20, np.nan), nmodes=3)
        assert np.isnan(F).all() and np.isnan(c).all() and np.isnan(dz).all()
        z2, n22 = z.copy(), n2.copy()
        z2[1:] = np.nan
        assert np.isnan(modes.vertical_modes(n22, z2, nmodes=3)[0]).all()

    def test_vertical_zoom(self):
        z, _ = _column(nz=100)
        n2 = 1e-6 + 2e-4 * np.exp(z / 300.0)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=4, zmin=-1000.0, zmax=-100.0)
        inside = (z >= -1000.0) & (z <= -100.0)
        Fref, cref, dzref = modes.vertical_modes(n2[inside], z[inside], nmodes=4)
        np.testing.assert_allclose(c, cref, equal_nan=True)
        np.testing.assert_allclose(F[:, inside], Fref, atol=1e-10)
        assert np.isnan(F[:, ~inside]).all() and np.isnan(dz[~inside]).all()

    def test_few_levels_and_many_modes(self):
        z, n2 = _column(nz=6)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=9)
        assert F.shape == (9, 6)
        assert np.isfinite(F[:6]).all() and np.isnan(F[6:]).all() and np.isnan(c[6:]).all()

    def test_bad_stratification_is_clipped(self):
        z, _ = _column(nz=40)
        n2 = np.full(40, 1e-5)
        n2[10] = -1e-6  # Unstable
        n2[20] = 0.0
        F, c, dz = modes.vertical_modes(n2, z, nmodes=3)
        assert np.isfinite(F).all() and np.isfinite(c[1:]).all()
        assert (c[1:] > 0).all()

    def test_leading_dimensions_and_broadcast(self):
        z, _ = _column(nz=30)
        n2 = np.stack([1e-6 + 2e-4 * np.exp(z / s) for s in (200.0, 300.0, 500.0)])
        n2 = np.stack([n2, 2 * n2])  # (2, 3, nz)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=4)  # one z for all the columns
        assert F.shape == (2, 3, 4, 30) and c.shape == (2, 3, 4) and dz.shape == (2, 3, 30)
        Fi, ci, _ = modes.vertical_modes(n2[1, 2], z, nmodes=4)
        np.testing.assert_allclose(F[1, 2], Fi)
        np.testing.assert_allclose(c[1, 2], ci, equal_nan=True)
        # One z per column
        zz = np.broadcast_to(z, n2.shape).copy()
        zz[0, 1, :5] = np.nan
        n2z = n2.copy()
        F2, c2, dz2 = modes.vertical_modes(n2z, zz, nmodes=4)
        assert dz2.shape == n2.shape
        np.testing.assert_allclose(F2[1, 2], Fi)
        assert np.isnan(F2[0, 1, :, :5]).all() and not np.isnan(F2[0, 0]).any()

    def test_invalid_arguments(self):
        z, n2 = _column(nz=10)
        with pytest.raises(ValueError, match="nmodes"):
            modes.vertical_modes(n2, z, nmodes=0)
        with pytest.raises(ValueError, match="shape"):
            modes.vertical_modes(n2[:-1], z, nmodes=2)


class TestProjection:
    @staticmethod
    def get_modes(nz=50, **kwargs):
        z, _ = _column(nz=nz, stretch=2.0)
        n2 = 1e-6 + 2e-4 * np.exp(z / 300.0)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=nz, **kwargs)
        return z, F, c, dz

    def test_exact_reconstruction_with_all_the_modes(self):
        z, F, c, dz = self.get_modes()
        field = np.random.default_rng(0).normal(size=z.size)
        amps = modes.project(field, F, dz)
        assert amps.shape == (50,)
        np.testing.assert_allclose(modes.reconstruct(amps, F), field, atol=1e-10)

    def test_barotropic_amplitude_is_the_vertical_mean(self):
        z, F, c, dz = self.get_modes()
        field = np.exp(z / 500.0)
        amps = modes.project(field, F, dz)
        np.testing.assert_allclose(amps[0], (field * dz).sum() / dz.sum())

    def test_a_mode_projects_on_itself(self):
        z, F, c, dz = self.get_modes()
        amps = modes.project(2.5 * F[3], F, dz)
        expected = np.zeros(50)
        expected[3] = 2.5
        np.testing.assert_allclose(amps, expected, atol=1e-10)

    def test_truncation_error_decreases(self):
        z, F, c, dz = self.get_modes()
        field = np.tanh((z + 300.0) / 100.0)
        amps = modes.project(field, F, dz)
        errors = []
        for n in (1, 3, 6, 12, 50):
            amps_n = amps.copy()
            amps_n[n:] = 0
            errors.append(np.sqrt((((modes.reconstruct(amps_n, F) - field) ** 2) * dz).sum()))
        assert (np.diff(errors) < 0).all() and errors[-1] < 1e-9

    def test_missing_values(self):
        z, F, c, dz = self.get_modes()
        field = np.exp(z / 500.0)
        field[5] = np.nan
        assert np.isnan(modes.project(field, F, dz)).all()

    def test_vertical_zoom(self):
        z, F, c, dz = self.get_modes(zmin=-1500.0, zmax=-200.0)
        inside = np.isfinite(dz)
        assert 0 < inside.sum() < z.size
        field = np.exp(z / 500.0)
        amps = modes.project(field, F, dz)
        rec = modes.reconstruct(amps, F)
        np.testing.assert_allclose(rec[inside], field[inside], atol=1e-8)
        assert np.isnan(rec[~inside]).all()
        # The field outside of the layer does not matter
        field2 = field.copy()
        field2[~inside] = np.nan
        np.testing.assert_allclose(modes.project(field2, F, dz), amps)

    def test_leading_dimensions(self):
        z, F, c, dz = self.get_modes(nz=20)
        fields = np.random.default_rng(1).normal(size=(2, 3, 20))
        amps = modes.project(fields, F, dz)
        assert amps.shape == (2, 3, 20)
        np.testing.assert_allclose(modes.reconstruct(amps, F), fields, atol=1e-10)
        np.testing.assert_allclose(amps[1, 2], modes.project(fields[1, 2], F, dz))


class TestSolvers:
    """The few lowest modes use a tridiagonal solver, and all of them a dense one"""

    @pytest.mark.parametrize("free_surface", [False, True])
    @pytest.mark.parametrize("nz", [40, 120])
    def test_tridiagonal_matches_dense(self, nz, free_surface):
        rng = np.random.default_rng(nz)
        z, _ = _column(nz=nz, stretch=3.0)
        scale = rng.uniform(100.0, 800.0, size=(6, 1))
        n2 = 1e-6 + 2e-4 * np.exp(z / scale) * rng.uniform(0.5, 2.0, size=(6, 1))
        few = modes.vertical_modes(n2, z, nmodes=6, free_surface=free_surface)  # Tridiagonal
        full = modes.vertical_modes(n2, z, nmodes=nz, free_surface=free_surface)  # Dense
        np.testing.assert_allclose(few[1], full[1][:, :6], rtol=1e-8, equal_nan=True)
        np.testing.assert_allclose(few[0], full[0][:, :6], atol=1e-6)
        np.testing.assert_allclose(few[2], full[2])
        # The modes are orthonormal
        F, dz = few[0][3], few[2][3]
        np.testing.assert_allclose((F * dz) @ F.T / dz.sum(), np.eye(6), atol=1e-8)

    def test_sharp_stratification(self):
        # Quasi-discontinuous N2 over a thin layer, which makes the spectrum hard
        z = np.linspace(-4000.0, 0.0, 801)
        z = 0.5 * (z[1:] + z[:-1])
        n2 = np.full(z.size, 1e-8)
        n2[np.abs(z + 100.0) < 10.0] = 1e-3
        few = modes.vertical_modes(n2, z, nmodes=4, n2min=1e-9)
        full = modes.vertical_modes(n2, z, nmodes=z.size, n2min=1e-9)
        np.testing.assert_allclose(few[1], full[1][:4], rtol=1e-7, equal_nan=True)
        np.testing.assert_allclose(few[0], full[0][:4], atol=1e-5)

    def test_many_columns_are_independent(self):
        z, _ = _column(nz=50, stretch=2.0)
        scale = np.linspace(150.0, 700.0, 500)[:, None]
        n2 = 1e-6 + 2e-4 * np.exp(z / scale)
        F, c, dz = modes.vertical_modes(n2, z, nmodes=4)
        assert F.shape == (500, 4, 50)
        for k in (0, 137, 499):
            Fk, ck, _ = modes.vertical_modes(n2[k], z, nmodes=4)
            np.testing.assert_allclose(F[k], Fk)
            np.testing.assert_allclose(c[k], ck, equal_nan=True)
        # The speeds vary smoothly with the stratification
        assert (np.diff(c[:, 1]) > 0).all()
