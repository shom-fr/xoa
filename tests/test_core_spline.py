# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.spline` kernels
"""

import numpy as np
import pytest

from xoa.core.geo import relative_cell_coords as bilinear_cell_coords
from xoa.core.spline import (
    bicubic_frac,
    bilinear_frac,
    compute_cubic_weights_1d,
    compute_frac_indices,
    compute_tangent,
    hermite_basis,
    interpolate_bicubic,
    interpolate_cubic,
)


class TestHermiteBasis:
    """Test Hermite basis function computation"""

    def test_at_zero(self):
        """Test Hermite basis at mu=0"""
        h = hermite_basis(0.0)
        # At mu=0: h00=1, h10=0, h01=0, h11=0
        assert h[0] == pytest.approx(1.0)
        assert h[1] == pytest.approx(0.0)
        assert h[2] == pytest.approx(0.0)
        assert h[3] == pytest.approx(0.0)

    def test_at_one(self):
        """Test Hermite basis at mu=1"""
        h = hermite_basis(1.0)
        # At mu=1: h00=0, h10=0, h01=1, h11=0
        assert h[0] == pytest.approx(0.0)
        assert h[1] == pytest.approx(0.0)
        assert h[2] == pytest.approx(1.0)
        assert h[3] == pytest.approx(0.0)

    def test_at_half(self):
        """Test Hermite basis at mu=0.5"""
        h = hermite_basis(0.5)
        # At mu=0.5: h00=0.5, h10=0.125, h01=0.5, h11=-0.125
        mu = 0.5
        mu2 = mu * mu
        mu3 = mu2 * mu
        expected_h00 = 2 * mu3 - 3 * mu2 + 1
        expected_h10 = mu3 - 2 * mu2 + mu
        expected_h01 = -2 * mu3 + 3 * mu2
        expected_h11 = mu3 - mu2

        assert h[0] == pytest.approx(expected_h00)
        assert h[1] == pytest.approx(expected_h10)
        assert h[2] == pytest.approx(expected_h01)
        assert h[3] == pytest.approx(expected_h11)

    def test_partition_of_unity(self):
        """Test that h00 + h01 equals 1 (partition of unity for positions)"""
        for mu in np.linspace(0, 1, 11):
            h = hermite_basis(mu)
            # The position basis functions should sum to 1
            assert h[0] + h[2] == pytest.approx(1.0, abs=1e-10)


class TestComputeTangent:
    """Test tangent computation for cubic interpolation"""

    def test_zero_bias_zero_tension(self):
        """Test tangent with bias=0, tension=0"""
        p0, p1, p2 = 0.0, 1.0, 2.0
        tangent = compute_tangent(p0, p1, p2, bias=0.0, tension=0.0)
        # Should be: 0.5 * (p1 - p0) + 0.5 * (p2 - p1) = 0.5 * (p2 - p0)
        expected = 0.5 * (p2 - p0)
        assert tangent == pytest.approx(expected)

    def test_positive_bias(self):
        """Test tangent with positive bias"""
        p0, p1, p2 = 0.0, 1.0, 2.0
        tangent = compute_tangent(p0, p1, p2, bias=0.5, tension=0.0)
        # With positive bias, weight toward first interval
        a = 0.5 * 1.5 * (p1 - p0)  # (1-0)*(1+0.5)*0.5
        b = 0.5 * 0.5 * (p2 - p1)  # (1-0)*(1-0.5)*0.5
        expected = a + b
        assert tangent == pytest.approx(expected)

    def test_negative_bias(self):
        """Test tangent with negative bias"""
        p0, p1, p2 = 0.0, 1.0, 2.0
        tangent = compute_tangent(p0, p1, p2, bias=-0.5, tension=0.0)
        # With negative bias, weight toward second interval
        a = 0.5 * 0.5 * (p1 - p0)  # (1-0)*(1-0.5)*0.5
        b = 0.5 * 1.5 * (p2 - p1)  # (1-0)*(1+0.5)*0.5
        expected = a + b
        assert tangent == pytest.approx(expected)

    def test_high_tension(self):
        """Test tangent with high tension (approaches linear)"""
        p0, p1, p2 = 0.0, 1.0, 2.0
        tangent = compute_tangent(p0, p1, p2, bias=0.0, tension=1.0)
        # With tension=1, tangent should be zero
        assert tangent == pytest.approx(0.0)

    def test_linear_data(self):
        """Test tangent on linear data"""
        # For linear data, tangent should match slope
        for slope in [1.0, 2.0, -1.0]:
            p0, p1, p2 = 0.0, slope, 2 * slope
            tangent = compute_tangent(p0, p1, p2, bias=0.0, tension=0.0)
            assert tangent == pytest.approx(slope)


class TestComputeCubicWeights1D:
    """Test 1D cubic weight decomposition.

    compute_cubic_weights_1d is implemented by evaluating compute_tangent on
    unit-basis vectors to extract per-point coefficients, so these tests also
    indirectly validate that relationship.
    """

    def test_weights_sum_to_one(self):
        """Weights must form a partition of unity."""
        for mu in [0.0, 0.25, 0.5, 0.75, 1.0]:
            hh = hermite_basis(mu)
            for bias in [-0.5, 0.0, 0.5]:
                for tension in [0.0, 0.5, 1.0]:
                    weights = compute_cubic_weights_1d(hh, bias, tension)
                    assert np.sum(weights) == pytest.approx(
                        1.0, abs=1e-12
                    ), f"Weights don't sum to 1 at mu={mu}, bias={bias}, tension={tension}"

    def test_consistency_with_interpolate_cubic(self):
        """Weights must reproduce the same result as interpolate_cubic point-evaluation."""
        p = np.array([1.0, 2.0, 3.0, 4.0])

        for mu in [0.0, 0.25, 0.5, 0.75, 1.0]:
            hh = hermite_basis(mu)
            for bias in [-0.5, 0.0, 0.5]:
                for tension in [0.0, 0.5]:
                    direct = interpolate_cubic(hh, p[0], p[1], p[2], p[3], bias, tension)
                    weights = compute_cubic_weights_1d(hh, bias, tension)
                    from_weights = np.dot(weights, p)
                    assert from_weights == pytest.approx(
                        direct, abs=1e-12
                    ), f'Weight decomposition inconsistent at mu={mu}, bias={bias}, tension={tension}'

    def test_weights_match_compute_tangent_unit_basis(self):
        """Weights must equal what compute_tangent on unit-basis vectors predicts.

        This directly tests the implementation strategy: because compute_tangent
        is linear in (p0,p1,p2), evaluating it on (1,0,0), (0,1,0), (0,0,1)
        gives the per-point coefficients c0, c1, c2 for any tangent call.
        """
        for mu in [0.0, 0.3, 0.5, 0.7, 1.0]:
            hh = hermite_basis(mu)
            for bias in [-0.5, 0.0, 0.5]:
                for tension in [0.0, 0.5, 1.0]:
                    c0 = compute_tangent(1.0, 0.0, 0.0, bias, tension)
                    c1 = compute_tangent(0.0, 1.0, 0.0, bias, tension)
                    c2 = compute_tangent(0.0, 0.0, 1.0, bias, tension)

                    expected = np.array(
                        [
                            hh[1] * c0,
                            hh[0] + hh[1] * c1 + hh[3] * c0,
                            hh[2] + hh[1] * c2 + hh[3] * c1,
                            hh[3] * c2,
                        ]
                    )
                    weights = compute_cubic_weights_1d(hh, bias, tension)
                    np.testing.assert_allclose(weights, expected, atol=1e-15)

    def test_linear_polynomial_exact_via_weights(self):
        """Weights must reproduce linear polynomials exactly."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        p = 2 * x + 3

        for mu in [0.1, 0.333, 0.5, 0.667, 0.9]:
            hh = hermite_basis(mu)
            weights = compute_cubic_weights_1d(hh, bias=0.0, tension=0.0)
            result = np.dot(weights, p)
            expected = 2 * (1.0 + mu) + 3
            assert result == pytest.approx(
                expected, abs=1e-12
            ), f'Linear polynomial not exact via weights at mu={mu}'


class TestInterpolateCubic:
    """Test 1D cubic Hermite interpolation"""

    def test_at_endpoints(self):
        """Test that interpolation passes through endpoints"""
        p0, p1, p2, p3 = 0.0, 1.0, 2.0, 3.0

        # At mu=0, should return p1
        hh = hermite_basis(0.0)
        result = interpolate_cubic(hh, p0, p1, p2, p3, bias=0.0, tension=0.0)
        assert result == pytest.approx(p1)

        # At mu=1, should return p2
        hh = hermite_basis(1.0)
        result = interpolate_cubic(hh, p0, p1, p2, p3, bias=0.0, tension=0.0)
        assert result == pytest.approx(p2)

    def test_linear_data(self):
        """Test interpolation on linear data"""
        # For linear data, cubic interpolation should be exact
        x_values = np.array([0.0, 1.0, 2.0, 3.0])
        y_values = 2.0 * x_values + 3.0  # Linear function

        # Interpolate at midpoint
        hh = hermite_basis(0.5)
        result = interpolate_cubic(
            hh, y_values[0], y_values[1], y_values[2], y_values[3], bias=0.0, tension=0.0
        )
        expected = 2.0 * 1.5 + 3.0  # y = 2x + 3 at x=1.5
        assert result == pytest.approx(expected, abs=1e-10)

    def test_quadratic_data(self):
        """Test interpolation on quadratic data"""
        # For quadratic data, cubic should be exact
        x_values = np.array([0.0, 1.0, 2.0, 3.0])
        y_values = x_values**2  # Quadratic function

        # Interpolate at various points
        for mu in [0.25, 0.5, 0.75]:
            hh = hermite_basis(mu)
            result = interpolate_cubic(
                hh, y_values[0], y_values[1], y_values[2], y_values[3], bias=0.0, tension=0.0
            )
            x_interp = 1.0 + mu  # x position for interpolation
            expected = x_interp**2
            assert result == pytest.approx(expected, abs=1e-10)

    def test_cubic_data_approximation_quality(self):
        """Test that cubic Hermite interpolation provides good approximation for cubics"""
        # NOTE: Hermite interpolation does NOT exactly reproduce general cubic polynomials!
        # It matches values and derivatives at endpoints, but has interpolation error for cubics.
        # Test with f(x) = x^3 - 2x^2 + 3x + 1
        x_values = np.array([0.0, 1.0, 2.0, 3.0])
        y_values = x_values**3 - 2 * x_values**2 + 3 * x_values + 1

        # Interpolate at various points - expect good but not exact approximation
        test_points = [0.25, 0.5, 0.75]
        max_error = 0.0
        for mu in test_points:
            hh = hermite_basis(mu)
            result = interpolate_cubic(
                hh, y_values[0], y_values[1], y_values[2], y_values[3], bias=0.0, tension=0.0
            )
            x_interp = 1.0 + mu  # x position for interpolation between x=1 and x=2
            expected = x_interp**3 - 2 * x_interp**2 + 3 * x_interp + 1
            error = abs(result - expected)
            max_error = max(max_error, error)

        # Error should be small but not machine precision
        # Hermite gives good approximation, just not exact for general cubics
        assert max_error < 0.1, f'Cubic approximation error too large: {max_error}'
        assert max_error > 1e-12, 'Error suspiciously small - check if test is meaningful'

    def test_constant_data(self):
        """Test interpolation on constant data"""
        p = 5.0
        hh = hermite_basis(0.3)
        result = interpolate_cubic(hh, p, p, p, p, bias=0.0, tension=0.0)
        assert result == pytest.approx(p)

    def test_with_tension(self):
        """Test that tension reduces overshoot"""
        # Use data that could overshoot
        p0, p1, p2, p3 = 0.0, 1.0, 1.0, 0.0

        hh = hermite_basis(0.5)

        # Without tension
        result_no_tension = interpolate_cubic(hh, p0, p1, p2, p3, bias=0.0, tension=0.0)

        # With high tension
        result_tension = interpolate_cubic(hh, p0, p1, p2, p3, bias=0.0, tension=0.8)

        # High tension should be closer to linear interpolation
        linear_value = 0.5 * p1 + 0.5 * p2
        assert abs(result_tension - linear_value) < abs(result_no_tension - linear_value)


class TestInterpolateBicubic:
    """Test 2D bicubic Hermite interpolation.

    interpolate_bicubic is the scalar reference implementation.  The vectorised
    production path (bicubic_frac) reaches the same result via the outer product
    of two compute_cubic_weights_1d weight vectors.  test_equivalent_to_weight_outer_product
    verifies both paths are identical.
    """

    def test_at_corners(self):
        """Test that interpolation passes through corners"""
        # Create a 4x4 grid of values
        pp = np.array(
            [[0.0, 1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 4.0, 5.0, 6.0]]
        )

        # Test all four corners of the central cell
        corners = [
            (0.0, 0.0, pp[1, 1]),  # Bottom-left
            (1.0, 0.0, pp[1, 2]),  # Bottom-right
            (0.0, 1.0, pp[2, 1]),  # Top-left
            (1.0, 1.0, pp[2, 2]),  # Top-right
        ]

        for mu_x, mu_y, expected in corners:
            xhh = hermite_basis(mu_x)
            yhh = hermite_basis(mu_y)
            result = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)
            assert result == pytest.approx(expected, abs=1e-10)

    def test_linear_data(self):
        """Test bicubic on linear data"""
        # Create linear surface: f(x,y) = 2x + 3y + 1
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        pp = 2 * xx + 3 * yy + 1

        # Interpolate at center of central cell
        xhh = hermite_basis(0.5)
        yhh = hermite_basis(0.5)
        result = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)

        # Expected value at (1.5, 1.5)
        expected = 2 * 1.5 + 3 * 1.5 + 1
        assert result == pytest.approx(expected, abs=1e-10)

    def test_constant_data(self):
        """Test bicubic on constant data"""
        pp = np.ones((4, 4)) * 5.0

        xhh = hermite_basis(0.3)
        yhh = hermite_basis(0.7)
        result = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)
        assert result == pytest.approx(5.0)

    def test_bilinear_data(self):
        """Test bicubic on bilinear surface"""
        # Create bilinear surface: f(x,y) = x*y
        x = np.arange(4)
        y = np.arange(4)
        xx, yy = np.meshgrid(x, y)
        pp = xx * yy

        # Interpolate at various points
        test_points = [(0.3, 0.3), (0.5, 0.5), (0.7, 0.7)]
        for mu_x, mu_y in test_points:
            xhh = hermite_basis(mu_x)
            yhh = hermite_basis(mu_y)
            result = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)

            # Expected at position (1+mu_x, 1+mu_y)
            x_pos = 1.0 + mu_x
            y_pos = 1.0 + mu_y
            expected = x_pos * y_pos
            assert result == pytest.approx(expected, abs=1e-10)

    def test_quadratic_polynomial_exact(self):
        """Test that bicubic exactly reproduces quadratic/bilinear polynomials"""
        # Bicubic Hermite can exactly reproduce up to degree 2 polynomials (quadratic/bilinear)
        # Test with f(x,y) = x^2 + y^2 + x*y + 2x + 3y + 1
        x = np.arange(4, dtype=float)
        y = np.arange(4, dtype=float)
        xx, yy = np.meshgrid(x, y)
        pp = xx**2 + yy**2 + xx * yy + 2 * xx + 3 * yy + 1

        # Interpolate at various points - should all be exact
        test_points = [
            (0.2, 0.3),
            (0.25, 0.25),
            (0.5, 0.5),
            (0.333, 0.667),
            (0.75, 0.75),
            (0.8, 0.9),
        ]
        for mu_x, mu_y in test_points:
            xhh = hermite_basis(mu_x)
            yhh = hermite_basis(mu_y)
            result = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)

            # Expected at position (1+mu_x, 1+mu_y)
            x_pos = 1.0 + mu_x
            y_pos = 1.0 + mu_y
            expected = x_pos**2 + y_pos**2 + x_pos * y_pos + 2 * x_pos + 3 * y_pos + 1
            assert result == pytest.approx(
                expected, abs=1e-10
            ), f'Bicubic not exact for quadratic at ({mu_x},{mu_y}): error={abs(result - expected)}'

    def test_smooth_function(self):
        """Test bicubic on smooth function"""
        # Use a smooth function like sin
        x = np.linspace(0, 1, 4)
        y = np.linspace(0, 1, 4)
        xx, yy = np.meshgrid(x, y)
        pp = np.sin(np.pi * xx) * np.cos(np.pi * yy)

        # Bicubic should interpolate smoothly
        xhh = hermite_basis(0.5)
        yhh = hermite_basis(0.5)
        result = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)

        # Result should be finite and reasonable
        assert np.isfinite(result)
        assert abs(result) <= np.max(np.abs(pp))

    def test_with_bias_and_tension(self):
        """Test bicubic with different bias and tension parameters"""
        pp = np.random.rand(4, 4)

        xhh = hermite_basis(0.5)
        yhh = hermite_basis(0.5)

        # Different parameter combinations should give different results
        result_default = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)
        result_biased = interpolate_bicubic(xhh, yhh, pp, bias=0.5, tension=0.0)
        result_tense = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.5)

        # Results should be different (unless data is very special)
        assert result_default != result_biased or result_default != result_tense

    def test_symmetry(self):
        """Test that bicubic respects symmetry in data"""
        # Create symmetric data
        pp = np.array(
            [[1.0, 2.0, 2.0, 1.0], [2.0, 3.0, 3.0, 2.0], [2.0, 3.0, 3.0, 2.0], [1.0, 2.0, 2.0, 1.0]]
        )

        xhh = hermite_basis(0.5)
        yhh = hermite_basis(0.5)
        result = interpolate_bicubic(xhh, yhh, pp, bias=0.0, tension=0.0)

        # Result should be finite and reasonable (bicubic can overshoot slightly)
        assert np.isfinite(result)
        assert 1.0 <= result <= 4.0  # Within reasonable bounds

    def test_equivalent_to_weight_outer_product(self):
        """interpolate_bicubic must match the compute_cubic_weights_1d outer-product path.

        bicubic_frac computes w_bic[r,c] = y_w[r] * x_w[c] then sums over the 4x4
        stencil.  This test verifies both routes give identical results, validating
        interpolate_bicubic as a correct scalar reference for that kernel.
        """
        rng = np.random.default_rng(0)
        for bias, tension in [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (-0.3, 0.3)]:
            pp = rng.random((4, 4))
            for mu_x, mu_y in [(0.2, 0.7), (0.5, 0.5), (0.0, 1.0), (0.9, 0.1)]:
                xhh = hermite_basis(mu_x)
                yhh = hermite_basis(mu_y)

                ref = interpolate_bicubic(xhh, yhh, pp, bias, tension)

                x_w = compute_cubic_weights_1d(xhh, bias, tension)
                y_w = compute_cubic_weights_1d(yhh, bias, tension)
                via_weights = sum(y_w[r] * x_w[c] * pp[r, c] for r in range(4) for c in range(4))

                assert via_weights == pytest.approx(
                    ref, abs=1e-12
                ), f'Mismatch at mu_x={mu_x}, mu_y={mu_y}, bias={bias}, tension={tension}'


class TestInterpolationConsistency:
    """Test consistency between 1D and 2D interpolation"""

    def test_1d_vs_2d_consistency(self):
        """Test that 2D reduces to 1D when one dimension is constant"""
        # Create data that varies only in x
        p_1d = np.array([1.0, 2.0, 3.0, 4.0])
        pp_2d = np.repeat(p_1d.reshape(1, 4), 4, axis=0)  # Same values in all rows

        mu_x = 0.4
        xhh = hermite_basis(mu_x)
        yhh = hermite_basis(0.5)  # Doesn't matter since y is constant

        # 1D interpolation
        result_1d = interpolate_cubic(xhh, p_1d[0], p_1d[1], p_1d[2], p_1d[3], 0.0, 0.0)

        # 2D interpolation should give same result
        result_2d = interpolate_bicubic(xhh, yhh, pp_2d, 0.0, 0.0)

        assert result_1d == pytest.approx(result_2d, abs=1e-10)

    def test_interpolation_bounds(self):
        """Test that interpolated values stay within reasonable bounds"""
        # For monotonic data without overshoot
        pp = np.arange(16).reshape(4, 4).astype(float)

        for mu_x in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for mu_y in [0.0, 0.25, 0.5, 0.75, 1.0]:
                xhh = hermite_basis(mu_x)
                yhh = hermite_basis(mu_y)
                result = interpolate_bicubic(xhh, yhh, pp, 0.0, 0.5)  # Use some tension

                # With tension, result should be in reasonable range
                assert np.min(pp) - 1 <= result <= np.max(pp) + 1


# Integration tests
class TestInterpolationIntegration:
    """Integration tests for interpolation functions"""

    @pytest.mark.parametrize('bias', [0.0, 0.5, -0.5])
    @pytest.mark.parametrize('tension', [0.0, 0.5, 1.0])
    def test_parameter_combinations(self, bias, tension):
        """Test various bias and tension combinations"""
        pp = np.random.rand(4, 4)

        xhh = hermite_basis(0.5)
        yhh = hermite_basis(0.5)

        result = interpolate_bicubic(xhh, yhh, pp, bias, tension)

        # Result should be finite
        assert np.isfinite(result)

    def test_grid_interpolation(self):
        """Test interpolation over a grid of points"""
        # Create source data
        x_src = np.linspace(0, 3, 4)
        y_src = np.linspace(0, 3, 4)
        xx_src, yy_src = np.meshgrid(x_src, y_src)
        pp = np.sin(xx_src) * np.cos(yy_src)

        # Interpolate at many points
        mu_values = np.linspace(0, 1, 11)
        results = np.zeros((len(mu_values), len(mu_values)))

        for i, mu_x in enumerate(mu_values):
            for j, mu_y in enumerate(mu_values):
                xhh = hermite_basis(mu_x)
                yhh = hermite_basis(mu_y)
                results[j, i] = interpolate_bicubic(xhh, yhh, pp, 0.0, 0.0)

        # All results should be finite
        assert np.all(np.isfinite(results))

        # Results should be smooth (no sudden jumps)
        dx = np.diff(results, axis=1)
        dy = np.diff(results, axis=0)
        assert np.max(np.abs(dx)) < 1.0  # Reasonable gradient
        assert np.max(np.abs(dy)) < 1.0


class TestBilinearCellCoords:
    """Test bilinear coordinate computation"""

    def test_unit_square_center(self):
        """Test point at center of unit square"""
        # Unit square: (0,0), (1,0), (1,1), (0,1)
        p, q = bilinear_cell_coords(
            0.0,
            1.0,
            1.0,
            0.0,  # x coordinates (counter-clockwise)
            0.0,
            0.0,
            1.0,
            1.0,  # y coordinates
            0.5,
            0.5,  # target point
        )

        assert p == pytest.approx(0.5, abs=1e-10)
        assert q == pytest.approx(0.5, abs=1e-10)

    def test_unit_square_corners(self):
        """Test points at corners of unit square"""
        # Bottom-left corner (0,0)
        p, q = bilinear_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
        assert p == pytest.approx(0.0, abs=1e-10)
        assert q == pytest.approx(0.0, abs=1e-10)

        # Bottom-right corner (1,0)
        p, q = bilinear_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0)
        assert p == pytest.approx(1.0, abs=1e-10)
        assert q == pytest.approx(0.0, abs=1e-10)

        # Top-right corner (1,1)
        p, q = bilinear_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
        assert p == pytest.approx(1.0, abs=1e-10)
        assert q == pytest.approx(1.0, abs=1e-10)

        # Top-left corner (0,1)
        p, q = bilinear_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0)
        assert p == pytest.approx(0.0, abs=1e-10)
        assert q == pytest.approx(1.0, abs=1e-10)

    def test_outside_cell(self):
        """Test point outside cell returns invalid coordinates"""
        # Point outside unit square
        p, q = bilinear_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0)  # Outside

        assert p == -1.0 and q == -1.0

    def test_irregular_quadrilateral(self):
        """Test with irregular quadrilateral"""
        # Tilted quadrilateral
        p, q = bilinear_cell_coords(
            0.0,
            2.0,
            1.0,
            -1.0,  # x coordinates
            0.0,
            1.0,
            3.0,
            2.0,  # y coordinates
            0.5,
            1.5,  # point inside
        )

        # Should find valid coordinates
        assert 0.0 <= p <= 1.0
        assert 0.0 <= q <= 1.0

    def test_degenerate_cases(self):
        """Test degenerate quadrilaterals"""
        # Collapsed to a line (all y coordinates same)
        p, q = bilinear_cell_coords(0.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

        # Should handle gracefully (may return valid or invalid coords)
        assert isinstance(p, float) and isinstance(q, float)


class TestComputeFracIndices:
    """Unit tests for the compute_frac_indices Numba kernel."""

    @staticmethod
    def _regular_grid(ny, nx, lon0=0.0, lat0=0.0, dlon=1.0, dlat=1.0):
        lon = np.linspace(lon0, lon0 + dlon * (nx - 1), nx)
        lat = np.linspace(lat0, lat0 + dlat * (ny - 1), ny)
        lon2d, lat2d = np.meshgrid(lon, lat)
        return lon2d, lat2d

    def test_regular_center_point(self):
        """A point exactly at cell center should give (j=2, i=2, a=0.5, b=0.5)."""
        src_lon, src_lat = self._regular_grid(5, 5)  # 0..4 x 0..4 spacing 1
        dst_lon = np.array([[2.5]])
        dst_lat = np.array([[2.5]])
        j_base, i_base, frac_a, frac_b = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0, stencil_margin=0
        )
        assert j_base[0] == 2
        assert i_base[0] == 2
        assert frac_a[0] == pytest.approx(0.5, abs=1e-12)
        assert frac_b[0] == pytest.approx(0.5, abs=1e-12)

    def test_regular_out_of_domain(self):
        """Points outside the source grid must return j_base=-1."""
        src_lon, src_lat = self._regular_grid(5, 5)
        dst_lon = np.array([[-10.0]])
        dst_lat = np.array([[-10.0]])
        j_base, i_base, frac_a, frac_b = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0, stencil_margin=0
        )
        assert j_base[0] == -1

    def test_stencil_margin_bilinear_allows_boundary(self):
        """stencil_margin=0 (bilinear) allows the edge cell (j=0, i=0)."""
        src_lon, src_lat = self._regular_grid(5, 5)
        # Point in the first cell (j=0, i=0, a≈0.5, b≈0.5)
        dst_lon = np.array([[0.5]])
        dst_lat = np.array([[0.5]])
        j_base, _, _, _ = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0, stencil_margin=0
        )
        assert j_base[0] == 0

    def test_stencil_margin_bicubic_forbids_boundary(self):
        """stencil_margin=1 (bicubic) must mark edge cells as invalid."""
        src_lon, src_lat = self._regular_grid(5, 5)
        # Same point as above — in the edge cell
        dst_lon = np.array([[0.5]])
        dst_lat = np.array([[0.5]])
        j_base, _, _, _ = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0, stencil_margin=1
        )
        assert j_base[0] == -1

    def test_bicubic_interior_point_valid(self):
        """stencil_margin=1 allows interior cells."""
        src_lon, src_lat = self._regular_grid(5, 5)
        dst_lon = np.array([[2.5]])
        dst_lat = np.array([[2.5]])
        j_base, i_base, frac_a, frac_b = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0, stencil_margin=1
        )
        assert j_base[0] == 2
        assert i_base[0] == 2

    def test_multiple_dst_points(self):
        """Multiple destination points: mix of valid and invalid."""
        src_lon, src_lat = self._regular_grid(5, 5)
        dst_lon = np.array([[0.5, 2.5]])
        dst_lat = np.array([[0.5, 2.5]])
        j_base, _, _, _ = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0, stencil_margin=1
        )
        assert j_base[0] == -1  # boundary cell: invalid for bicubic
        assert j_base[1] == 2  # interior cell: valid

    def test_rectangular_grid(self):
        """grid_type=1 (rectangular) gives same result as regular for axis-aligned grids."""
        src_lon, src_lat = self._regular_grid(5, 5)
        dst_lon = np.array([[2.5]])
        dst_lat = np.array([[2.5]])
        j_reg, i_reg, a_reg, b_reg = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0
        )
        j_rec, i_rec, a_rec, b_rec = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=1
        )
        assert j_reg[0] == j_rec[0]
        assert i_reg[0] == i_rec[0]
        assert a_reg[0] == pytest.approx(a_rec[0], abs=1e-10)
        assert b_reg[0] == pytest.approx(b_rec[0], abs=1e-10)

    def test_frac_values_clamped_to_01(self):
        """Fractional values must be in [0, 1]."""
        src_lon, src_lat = self._regular_grid(5, 5)
        dst_lon = np.array([[1.7, 3.9, 0.1]])
        dst_lat = np.array([[1.3, 2.8, 0.4]])
        j_base, i_base, frac_a, frac_b = compute_frac_indices(
            src_lon, src_lat, dst_lon, dst_lat, grid_type=0, stencil_margin=0
        )
        valid = j_base >= 0
        assert np.all(frac_a[valid] >= 0.0) and np.all(frac_a[valid] <= 1.0)
        assert np.all(frac_b[valid] >= 0.0) and np.all(frac_b[valid] <= 1.0)


class TestBilinearFracKernel:
    """Unit tests for the bilinear_frac Numba kernel."""

    @staticmethod
    def _make_inputs(ny, nx, K=1):
        """Return X (n_src, K) and a pre-allocated out."""
        n_src = ny * nx
        return np.empty((n_src, K)), np.empty((1, K))

    def test_center_of_cell_bilinear_exact(self):
        """At cell center (a=0.5, b=0.5) bilinear gives the average of 4 corners."""
        ny, nx = 3, 4
        n_src = ny * nx
        # Linear field f[j, i] = i + 2*j
        f = np.array([[float(i + 2 * j) for i in range(nx)] for j in range(ny)])
        X = f.reshape(n_src, 1)
        j_base = np.array([1], dtype=np.int64)
        i_base = np.array([1], dtype=np.int64)
        frac_a = np.array([0.5])
        frac_b = np.array([0.5])
        out = np.empty((1, 1))
        bilinear_frac(j_base, i_base, frac_a, frac_b, X, nx, out, False, 1.0)
        # Linear field: exact result = 1.5 + 2*1.5 = 4.5
        assert out[0, 0] == pytest.approx(1.5 + 2 * 1.5, abs=1e-12)

    def test_invalid_j_base_gives_nan(self):
        """j_base=-1 must produce NaN regardless of X."""
        ny, nx = 3, 4
        X = np.ones((ny * nx, 2))
        j_base = np.array([-1], dtype=np.int64)
        i_base = np.array([0], dtype=np.int64)
        frac_a = np.array([0.5])
        frac_b = np.array([0.5])
        out = np.empty((1, 2))
        bilinear_frac(j_base, i_base, frac_a, frac_b, X, nx, out, False, 1.0)
        assert np.all(np.isnan(out[0]))

    def test_skipna_false_propagates_nan(self):
        """skipna=False: a NaN corner makes the output NaN."""
        ny, nx = 3, 4
        X = np.ones((ny * nx, 1))
        X[0, 0] = np.nan  # SW corner
        j_base = np.array([0], dtype=np.int64)
        i_base = np.array([0], dtype=np.int64)
        frac_a = np.array([0.5])
        frac_b = np.array([0.5])
        out = np.empty((1, 1))
        bilinear_frac(j_base, i_base, frac_a, frac_b, X, nx, out, False, 1.0)
        assert np.isnan(out[0, 0])

    def test_skipna_true_ignores_nan(self):
        """skipna=True: NaN corner renorms over remaining valid corners."""
        ny, nx = 3, 4
        X = np.ones((ny * nx, 1))
        X[0, 0] = np.nan  # SW corner NaN; other 3 corners = 1.0
        j_base = np.array([0], dtype=np.int64)
        i_base = np.array([0], dtype=np.int64)
        frac_a = np.array([0.5])
        frac_b = np.array([0.5])
        out = np.empty((1, 1))
        bilinear_frac(j_base, i_base, frac_a, frac_b, X, nx, out, True, 1.0)
        # Constant 1 field: result should still be 1 (renorm over valid)
        assert out[0, 0] == pytest.approx(1.0, abs=1e-12)

    def test_multicolumn_K_greater_than_1(self):
        """K>1 columns: each column interpolated independently."""
        ny, nx = 4, 5
        K = 6
        n_src = ny * nx
        rng = np.random.default_rng(7)
        X = rng.random((n_src, K))
        j_base = np.array([1], dtype=np.int64)
        i_base = np.array([2], dtype=np.int64)
        frac_a = np.array([0.3])
        frac_b = np.array([0.7])
        out = np.empty((1, K))
        bilinear_frac(j_base, i_base, frac_a, frac_b, X, nx, out, False, 1.0)

        # Compare to manual computation
        a, b = 0.3, 0.7
        j0, i0 = 1, 2
        w00 = (1 - a) * (1 - b)
        w10 = a * (1 - b)
        w01 = (1 - a) * b
        w11 = a * b
        c00 = j0 * nx + i0
        c10 = j0 * nx + i0 + 1
        c01 = (j0 + 1) * nx + i0
        c11 = (j0 + 1) * nx + i0 + 1
        expected = w00 * X[c00] + w10 * X[c10] + w01 * X[c01] + w11 * X[c11]
        np.testing.assert_allclose(out[0], expected, rtol=1e-12)

    def test_linear_field_exact(self):
        """Bilinear is exact for affine fields (a*x + b*y + c)."""
        ny, nx = 6, 8
        n_src = ny * nx
        lon = np.linspace(0, 7, nx)
        lat = np.linspace(0, 5, ny)
        lon2d, lat2d = np.meshgrid(lon, lat)
        field = 2.0 * lon2d + 3.0 * lat2d + 1.0
        X = field.reshape(n_src, 1)

        j_base = np.array([2], dtype=np.int64)
        i_base = np.array([3], dtype=np.int64)
        frac_a = np.array([0.6])
        frac_b = np.array([0.4])
        out = np.empty((1, 1))
        bilinear_frac(j_base, i_base, frac_a, frac_b, X, nx, out, False, 1.0)

        dst_lon = lon[3] + 0.6 * (lon[1] - lon[0])
        dst_lat = lat[2] + 0.4 * (lat[1] - lat[0])
        expected = 2.0 * dst_lon + 3.0 * dst_lat + 1.0
        assert out[0, 0] == pytest.approx(expected, abs=1e-10)


class TestBicubicFracKernel:
    """Unit tests for the bicubic_frac Numba kernel."""

    def test_invalid_j_base_gives_nan(self):
        """j_base=-1 must produce NaN."""
        ny, nx = 6, 8
        X = np.ones((ny * nx, 1))
        j_base = np.array([-1], dtype=np.int64)
        i_base = np.array([2], dtype=np.int64)
        frac_a = np.array([0.5])
        frac_b = np.array([0.5])
        out = np.empty((1, 1))
        bicubic_frac(j_base, i_base, frac_a, frac_b, X, ny, nx, out, False, 1.0)
        assert np.isnan(out[0, 0])

    def test_linear_field_exact_no_nan(self):
        """Bicubic is exact for linear fields when all 16 pts are valid."""
        ny, nx = 8, 10
        n_src = ny * nx
        lon = np.linspace(0, 9, nx)
        lat = np.linspace(0, 7, ny)
        lon2d, lat2d = np.meshgrid(lon, lat)
        field = 2.0 * lon2d + 3.0 * lat2d + 1.0
        X = field.reshape(n_src, 1)

        # j_base=2, i_base=3 means 4x4 stencil rows 1..4, cols 2..5 (all valid)
        j_base = np.array([2], dtype=np.int64)
        i_base = np.array([3], dtype=np.int64)
        frac_a = np.array([0.4])
        frac_b = np.array([0.6])
        out = np.empty((1, 1))
        bicubic_frac(j_base, i_base, frac_a, frac_b, X, ny, nx, out, False, 1.0)

        dst_lon = lon[3] + 0.4 * (lon[1] - lon[0])
        dst_lat = lat[2] + 0.6 * (lat[1] - lat[0])
        expected = 2.0 * dst_lon + 3.0 * dst_lat + 1.0
        assert out[0, 0] == pytest.approx(expected, abs=1e-8)

    def test_nan_in_stencil_falls_back_to_bilinear(self):
        """Any NaN in the 16-pt stencil triggers bilinear fallback on center 4."""
        ny, nx = 8, 10
        n_src = ny * nx
        # Constant field = 5.0: bilinear fallback must still give 5.0
        X = np.full((n_src, 1), 5.0)
        j_base = np.array([2], dtype=np.int64)
        i_base = np.array([3], dtype=np.int64)
        frac_a = np.array([0.4])
        frac_b = np.array([0.6])

        # Poison a corner of the 4x4 stencil (ri=0, ci=0) → row j0-1, col i0-1
        j0, i0 = 2, 3
        poison_idx = (j0 - 1) * nx + (i0 - 1)
        X[poison_idx, 0] = np.nan

        out = np.empty((1, 1))
        bicubic_frac(j_base, i_base, frac_a, frac_b, X, ny, nx, out, False, 1.0)
        # Bilinear fallback on constant 5.0 field → still 5.0
        assert out[0, 0] == pytest.approx(5.0, abs=1e-12)

    def test_nan_in_center_skipna_true(self):
        """NaN in one center stencil point + skipna=True → renorm, not NaN."""
        ny, nx = 8, 10
        n_src = ny * nx
        X = np.full((n_src, 1), 3.0)
        j_base = np.array([2], dtype=np.int64)
        i_base = np.array([3], dtype=np.int64)
        frac_a = np.array([0.5])
        frac_b = np.array([0.5])

        # Make center SW corner NaN (ri=1,ci=1) → stencil_flat[5]
        j0, i0 = 2, 3
        sw_idx = (j0) * nx + (i0)
        X[sw_idx, 0] = np.nan

        out = np.empty((1, 1))
        bicubic_frac(j_base, i_base, frac_a, frac_b, X, ny, nx, out, True, 1.0)
        # skipna renorms over the 3 valid center points; constant 3.0 → result 3.0
        assert out[0, 0] == pytest.approx(3.0, abs=1e-12)

    def test_multicolumn_K(self):
        """K>1: each column interpolated independently."""
        ny, nx = 8, 10
        K = 5
        n_src = ny * nx
        rng = np.random.default_rng(11)
        X = rng.random((n_src, K))

        j_base = np.array([2], dtype=np.int64)
        i_base = np.array([3], dtype=np.int64)
        frac_a = np.array([0.4])
        frac_b = np.array([0.6])

        out = np.empty((1, K))
        bicubic_frac(j_base, i_base, frac_a, frac_b, X, ny, nx, out, False, 1.0)

        # Run single-column version for each column and compare
        for k in range(K):
            out_k = np.empty((1, 1))
            bicubic_frac(
                j_base,
                i_base,
                frac_a,
                frac_b,
                X[:, k : k + 1].copy(),
                ny,
                nx,
                out_k,
                False,
                1.0,
            )
            assert out[0, k] == pytest.approx(out_k[0, 0], abs=1e-12)


if __name__ == '__main__':
    pytest.main([__file__])
