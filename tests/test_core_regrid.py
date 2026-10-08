# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.interp` module
"""

import functools
import numpy as np
import pytest

from xoa.core import regrid
import os
import tempfile
from xoa.core.grid import create_regular_grid, create_rotated_grid
from xoa.core.regrid import XYRegridder
from xoa.core.num import CsrMatrix
from xoa.core.poly import spherical_area


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


def get_valid_mask(result, min_fraction=0.5):
    """Mask of the valid points, that must be numerous for a test to mean something"""
    valid = ~np.isnan(result)
    assert valid.mean() >= min_fraction, f"Only {valid.mean():.0%} of valid points"
    return valid


class TestXYRegridder:
    """Test main XYRegridder class"""

    def setup_method(self):
        """Set up test grids and data"""
        self.src_grid = create_regular_grid(5, 4, (-2.0, 2.0), (-1.5, 1.5))
        self.dst_grid = create_regular_grid(3, 3, (-1.0, 1.0), (-1.0, 1.0))

        # Create test data
        self.test_data_2d = np.sin(self.src_grid['lat']) * np.cos(self.src_grid['lon'])
        self.test_data_3d = np.stack(
            [self.test_data_2d, self.test_data_2d * 2, self.test_data_2d * 3]
        )
        self.test_data_4d = np.stack([self.test_data_3d, self.test_data_3d * 1.5])

    def test_regridder_initialization_bilinear(self):
        """Test regridder initialization for bilinear method"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')

        assert regridder.method == 'bilinear'
        assert regridder.src_grid is not None
        assert regridder.dst_grid is not None
        assert regridder.weights is None  # Not computed yet

    def test_regridder_initialization_conservative(self):
        """Test regridder initialization for conservative method"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='conservative')

        assert regridder.method == 'conservative'
        assert regridder.weights is None  # Not computed yet

    def test_regridder_invalid_method(self):
        """Test regridder with invalid method"""
        with pytest.raises(ValueError, match="Method 'invalid' not supported"):
            XYRegridder(self.src_grid, self.dst_grid, method='invalid')

    def test_regridder_invalid_grids(self):
        """Test regridder with invalid grid inputs"""
        # Missing required keys
        invalid_grid = {'lon': self.src_grid['lon']}  # Missing 'lat'

        with pytest.raises(ValueError, match='missing required key'):
            XYRegridder(invalid_grid, self.dst_grid, method='bilinear')

        # Wrong array shapes
        invalid_grid = {
            'lon': self.src_grid['lon'],
            'lat': self.src_grid['lat'][:-1, :],  # Different shape
        }

        with pytest.raises(ValueError, match='must have the same shape'):
            XYRegridder(invalid_grid, self.dst_grid, method='bilinear')

    def test_compute_weights_bilinear(self):
        """Test weight computation for bilinear method"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
        regridder.compute_weights()

        # Bilinear uses fractional-index representation
        assert regridder._j_base is not None
        assert regridder._i_base is not None
        assert regridder._frac_a is not None
        assert regridder._frac_b is not None
        assert regridder._valid_dst_mask is not None

        dst_size = self.dst_grid['lon'].size
        assert regridder._j_base.shape == (dst_size,)
        assert np.sum(regridder._valid_dst_mask) > 0

    def test_compute_weights_conservative(self):
        """Test weight computation for conservative method"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='conservative')
        weights = regridder.compute_weights()

        assert weights is not None
        assert isinstance(weights, CsrMatrix)

        # Check matrix dimensions
        dst_size = self.dst_grid['lon'].size
        src_size = self.src_grid['lon'].size
        assert weights.shape == (dst_size, src_size)

        # Conservative method typically has more weights per destination cell
        assert weights.nnz > 0

    def test_regrid_2d_data(self):
        """Test regridding 2D data"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
        result = regridder.regrid(self.test_data_2d)

        # Check output shape
        assert result.shape == self.dst_grid['lon'].shape

        # Should produce reasonable values (not all NaN)
        assert not np.all(np.isnan(result))

        # Should preserve general structure (check that result has similar range)
        valid_result = result[get_valid_mask(result)]
        if len(valid_result) > 0:
            assert np.min(valid_result) >= np.min(self.test_data_2d) - 1.0
            assert np.max(valid_result) <= np.max(self.test_data_2d) + 1.0

    def test_regrid_3d_data(self):
        """Test regridding 3D data"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
        result = regridder.regrid(self.test_data_3d)

        # Check output shape
        expected_shape = (3,) + self.dst_grid['lon'].shape
        assert result.shape == expected_shape

        # Check that each time slice is regridded
        for t in range(3):
            slice_result = result[t]
            assert not np.all(np.isnan(slice_result))

    def test_regrid_4d_data(self):
        """Test regridding 4D data"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
        result = regridder.regrid(self.test_data_4d)

        # Check output shape
        expected_shape = (2, 3) + self.dst_grid['lon'].shape
        assert result.shape == expected_shape

        # Check that data is regridded for all dimensions
        for t_dim in range(2):
            for l_dim in range(3):
                slice_result = result[t_dim, l_dim]
                assert not np.all(np.isnan(slice_result))

    def test_regrid_invalid_data_shape(self):
        """Test regridding with invalid data shape"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')

        # Wrong spatial dimensions
        invalid_data = np.random.rand(3, 3)  # Should be (4, 5) for source grid

        with pytest.raises(ValueError, match="doesn't match source grid"):
            regridder.regrid(invalid_data)

    def test_regrid_with_nans(self):
        """Test regridding data containing NaN values"""
        data_with_nans = self.test_data_2d.copy()
        data_with_nans[0, 0] = np.nan
        data_with_nans[1, 1] = np.nan

        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
        result = regridder.regrid(data_with_nans, skipna=True)

        # Should handle NaNs gracefully
        assert result.shape == self.dst_grid['lon'].shape

        # Some valid results should exist
        valid_results = get_valid_mask(result)
        assert np.sum(valid_results) > 0

    def test_regrid_conservation(self):
        """Test conservation properties for conservative method"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='conservative')

        # Create uniform field
        uniform_field = np.ones_like(self.src_grid['lon'])
        result = regridder.regrid(uniform_field)

        # Conservative method should preserve integral
        # For uniform field, result should be close to 1 where valid
        valid_mask = get_valid_mask(result)
        if np.sum(valid_mask) > 0:
            valid_results = result[valid_mask]
            # Should be close to 1.0
            assert np.all(np.abs(valid_results - 1.0) < 0.1)

    def test_save_and_load_weights(self):
        """Test saving and loading weights"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
        regridder.compute_weights()

        # Save weights to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
            tmp_filename = tmp.name

        try:
            regridder.save_weights(tmp_filename)

            # Create new regridder and load weights (validates consistency)
            loaded_regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
            loaded_regridder.load_weights(tmp_filename)

            assert loaded_regridder.method == 'bilinear'
            assert loaded_regridder._j_base is not None

            # Test that loaded regridder works
            result_original = regridder.regrid(self.test_data_2d)
            result_loaded = loaded_regridder.regrid(self.test_data_2d)

            # Results should be identical
            np.testing.assert_allclose(result_original, result_loaded, equal_nan=True)

        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

    def test_save_weights_without_computation(self):
        """Test error when saving weights before computation"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
            tmp_filename = tmp.name

        try:
            with pytest.raises(ValueError, match='No weights computed yet'):
                regridder.save_weights(tmp_filename)
        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)

    def test_load_weights_grid_mismatch(self):
        """Test that load_weights validates grid consistency"""
        regridder = XYRegridder(self.src_grid, self.dst_grid, method='bilinear')
        regridder.compute_weights()

        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as tmp:
            tmp_filename = tmp.name

        try:
            regridder.save_weights(tmp_filename)

            # Create grids with wrong shapes (different from src and dst)
            wrong_src_grid = create_regular_grid(3, 3, (-1.0, 1.0), (-1.0, 1.0))  # 3x3 != src 5x4
            wrong_dst_grid = create_regular_grid(4, 4, (-1.0, 1.0), (-1.0, 1.0))  # 4x4 != dst 3x3

            # Should raise ValueError for source grid shape mismatch
            wrong_regridder_src = XYRegridder(wrong_src_grid, self.dst_grid, method='bilinear')
            with pytest.raises(
                ValueError, match="Source grid shape.*doesn't match saved weights shape"
            ):
                wrong_regridder_src.load_weights(tmp_filename)

            # Should raise ValueError for destination grid shape mismatch
            wrong_regridder_dst = XYRegridder(self.src_grid, wrong_dst_grid, method='bilinear')
            with pytest.raises(ValueError, match="Destination.*doesn't match saved weights shape"):
                wrong_regridder_dst.load_weights(tmp_filename)

        finally:
            if os.path.exists(tmp_filename):
                os.unlink(tmp_filename)


class TestBilinearIsExact:
    """Verify that bilinear regridding reproduces bilinear functions exactly (no islands)."""

    def test_bilinear_function_exact(self):
        """
        For a strictly bilinear function f(x,y) = a + b*x + c*y + d*x*y,
        bilinear interpolation must reproduce the exact value at every destination
        point when all source values are finite (no masked/NaN cells).
        """
        src_grid = create_regular_grid(10, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = create_regular_grid(7, 9, (-3.0, 3.0), (-2.5, 2.5))

        a, b, c, d = 1.5, 0.3, -0.7, 0.05
        src_data = (
            a + b * src_grid['lon'] + c * src_grid['lat'] + d * src_grid['lon'] * src_grid['lat']
        )
        expected = (
            a + b * dst_grid['lon'] + c * dst_grid['lat'] + d * dst_grid['lon'] * dst_grid['lat']
        )

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(src_data)

        # All destination points are interior to the source grid, so none should be NaN
        assert not np.any(np.isnan(result)), 'Unexpected NaN in result for fully covered grids'

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=1e-10,
            err_msg='Bilinear regrid is not exact for a bilinear function',
        )


class TestRegridderEdgeCases:
    """Test edge cases and error conditions for regridder"""

    def test_regridder_with_masked_grids(self):
        """Test regridder with masked grids"""
        src_grid = create_regular_grid(4, 3, (-1.0, 1.0), (-1.0, 1.0))
        dst_grid = create_regular_grid(2, 2, (-0.5, 0.5), (-0.5, 0.5))

        # Add masks
        src_mask = np.ones_like(src_grid['lon'], dtype=bool)
        src_mask[0, 0] = False  # Mask corner
        src_grid['mask'] = src_mask

        dst_mask = np.ones_like(dst_grid['lon'], dtype=bool)
        dst_mask[0, 0] = False  # Mask corner
        dst_grid['mask'] = dst_mask

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        regridder.compute_weights()

        assert regridder._j_base is not None
        assert regridder._valid_dst_mask is not None

    def test_regridder_non_overlapping_grids(self):
        """Test regridder with non-overlapping grids"""
        src_grid = create_regular_grid(2, 2, (-2.0, -1.0), (-2.0, -1.0))
        dst_grid = create_regular_grid(2, 2, (1.0, 2.0), (1.0, 2.0))

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(np.ones((2, 2)))

        # Result should be all NaN (no valid interpolation)
        assert np.all(np.isnan(result))

    def test_regridder_curvilinear_to_regular(self):
        """Test regridding from curvilinear to regular grid"""
        src_grid = create_rotated_grid(4, 4, 0.0, 0.0, 45.0, 3.0, 3.0)
        dst_grid = create_regular_grid(3, 3, (-1.0, 1.0), (-1.0, 1.0))

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')

        # Create test data
        test_data = np.sin(src_grid['lat']) * np.cos(src_grid['lon'])
        result = regridder.regrid(test_data)

        assert result.shape == dst_grid['lon'].shape
        # Should have some valid interpolations
        assert not np.all(np.isnan(result))


class TestNumericalAccuracy:
    """Test numerical accuracy and consistency"""

    def test_bilinear_interpolation_accuracy(self):
        """Test bilinear interpolation accuracy for known functions"""
        # Create a fine source grid
        src_grid = create_regular_grid(11, 11, (-5.0, 5.0), (-5.0, 5.0))
        dst_grid = create_regular_grid(5, 5, (-2.0, 2.0), (-2.0, 2.0))

        # Create linear function (should be interpolated exactly)
        linear_field = 2.0 * src_grid['lon'] + 3.0 * src_grid['lat'] + 1.0

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(linear_field)

        # Compute expected values at destination points
        expected = 2.0 * dst_grid['lon'] + 3.0 * dst_grid['lat'] + 1.0

        # Should be very accurate for linear function
        valid_mask = get_valid_mask(result)
        if np.sum(valid_mask) > 0:
            np.testing.assert_allclose(
                result[valid_mask], expected[valid_mask], rtol=1e-10, atol=1e-10
            )

    def test_bilinear_interpolation_accuracy_curvilinear(self):
        """Test bilinear interpolation accuracy for known functions on curvilinear grids"""
        # Create a fine source grid
        # The destination grid is smaller than the source one, with another rotation,
        # so that all of its points are inside the source grid
        src_grid = create_rotated_grid(11, 11, 0.0, 45.0, 15.0, lon_span=10, lat_span=10)
        dst_grid = create_rotated_grid(9, 9, 0.0, 45.0, 40.0, lon_span=5, lat_span=5)

        # Create linear function (should be interpolated exactly)
        linear_field = 2.0 * src_grid['lon'] + 3.0 * src_grid['lat'] + 1.0

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(linear_field)

        # Compute expected values at destination points
        expected = 2.0 * dst_grid['lon'] + 3.0 * dst_grid['lat'] + 1.0

        # All the points are inside, and the result is very accurate for a linear function
        assert not np.isnan(result).any()
        valid_mask = get_valid_mask(result, min_fraction=1.0)
        if np.sum(valid_mask) > 0:
            np.testing.assert_allclose(
                result[valid_mask], expected[valid_mask], rtol=1e-10, atol=1e-10
            )

    # def test_conservative_mass_conservation(self):
    #     """Test mass conservation for conservative method"""
    #     src_grid = create_regular_grid(6, 6, (-3.0, 3.0), (-3.0, 3.0))
    #     dst_grid = create_regular_grid(3, 3, (-2.0, 2.0), (-2.0, 2.0))

    #     # Create non-uniform field
    #     field = np.exp(-(src_grid['lon']**2 + src_grid['lat']**2))

    #     regridder = XYRegridder(src_grid, dst_grid, method='conservative')
    #     result = regridder.regrid(field)

    #     # Compute approximate integrals (sum * cell_area for regular grids)
    #     # This is a simplified test - actual conservation test would need cell areas
    #     src_sum = np.sum(field)
    #     dst_sum = np.sum(result[get_valid_mask(result)])

    #     # Should conserve mass approximately (within grid resolution limits)
    #     # Note: This is approximate due to boundary effects
    #     if dst_sum > 0:
    #         conservation_error = abs(dst_sum - src_sum) / src_sum
    #         assert conservation_error < 0.5  # Allow for significant error due to grid mismatch

    def test_regridding_consistency(self):
        """Test that regridding is consistent between calls"""
        src_grid = create_regular_grid(4, 4, (-2.0, 2.0), (-2.0, 2.0))
        dst_grid = create_regular_grid(3, 3, (-1.0, 1.0), (-1.0, 1.0))

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')

        test_data = np.random.rand(*src_grid['lon'].shape)

        # Regrid multiple times
        result1 = regridder.regrid(test_data)
        result2 = regridder.regrid(test_data)
        result3 = regridder.regrid(test_data)

        # Results should be identical
        np.testing.assert_allclose(result1, result2, equal_nan=True)
        np.testing.assert_allclose(result2, result3, equal_nan=True)


class TestBicubicRegridder:
    """Test XYRegridder with bicubic method"""

    def test_bicubic_initialization(self):
        """Test creating regridder with bicubic method"""
        src_grid = create_regular_grid(20, 15, (-10.0, 10.0), (-5.0, 5.0))
        dst_grid = create_regular_grid(15, 12, (-8.0, 8.0), (-4.0, 4.0))

        regridder = XYRegridder(src_grid, dst_grid, method='bicubic', bias=0.0, tension=0.0)

        assert regridder.method == 'bicubic'
        assert regridder.bias == 0.0
        assert regridder.tension == 0.0

    def test_bicubic_smooth_function(self):
        """Test bicubic interpolation on smooth function"""
        src_grid = create_regular_grid(20, 15, (-10.0, 10.0), (-5.0, 5.0))
        dst_grid = create_regular_grid(15, 12, (-8.0, 8.0), (-4.0, 4.0))

        # Create smooth test data
        def smooth_func(lon, lat):
            return np.sin(lon * np.pi / 10) * np.cos(lat * np.pi / 5)

        src_data = smooth_func(src_grid['lon'], src_grid['lat'])

        # Bicubic regridding
        regridder_bicubic = XYRegridder(src_grid, dst_grid, method='bicubic')
        result_bicubic = regridder_bicubic.regrid(src_data)

        # Bilinear for comparison
        regridder_bilinear = XYRegridder(src_grid, dst_grid, method='bilinear')
        result_bilinear = regridder_bilinear.regrid(src_data)

        # Both should have valid results
        assert np.sum(~np.isnan(result_bicubic)) > 0
        assert np.sum(~np.isnan(result_bilinear)) > 0

        # Results should be similar but not identical
        valid_mask = get_valid_mask(result_bicubic) & get_valid_mask(result_bilinear)
        if np.sum(valid_mask) > 0:
            corr = np.corrcoef(
                result_bicubic[valid_mask].flatten(), result_bilinear[valid_mask].flatten()
            )[0, 1]
            assert corr > 0.95  # High correlation for smooth data

    def test_bicubic_smoother_than_bilinear(self):
        """Test that bicubic produces smoother results than bilinear"""
        src_grid = create_regular_grid(30, 20, (-10.0, 10.0), (-5.0, 5.0))
        dst_grid = create_regular_grid(20, 15, (-8.0, 8.0), (-4.0, 4.0))

        # Smooth test function
        src_data = np.sin(src_grid['lon'] * np.pi / 5) * np.exp(-(src_grid['lat'] ** 2) / 10)

        regridder_bicubic = XYRegridder(src_grid, dst_grid, method='bicubic')
        result_bicubic = regridder_bicubic.regrid(src_data)

        regridder_bilinear = XYRegridder(src_grid, dst_grid, method='bilinear')
        result_bilinear = regridder_bilinear.regrid(src_data)

        # Compute smoothness metric (smaller gradients = smoother)
        def compute_gradient_magnitude(data):
            valid_data = np.nan_to_num(data, nan=0.0)
            gy, gx = np.gradient(valid_data)
            return np.sqrt(gx**2 + gy**2)

        grad_bicubic = compute_gradient_magnitude(result_bicubic)
        grad_bilinear = compute_gradient_magnitude(result_bilinear)

        # Bicubic should generally have similar or lower average gradients for smooth data
        # (This is a weak test - mainly checking that bicubic doesn't make things worse)
        assert np.median(grad_bicubic) < np.median(grad_bilinear) * 1.5

    def test_bicubic_linear_preservation(self):
        """Test that bicubic preserves linear fields exactly"""
        src_grid = create_regular_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = create_regular_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        # Linear field: f(lon, lat) = 2*lon + 3*lat + 1
        src_data = 2.0 * src_grid['lon'] + 3.0 * src_grid['lat'] + 1.0

        regridder = XYRegridder(src_grid, dst_grid, method='bicubic')
        result = regridder.regrid(src_data)

        # Expected values on destination grid
        expected = 2.0 * dst_grid['lon'] + 3.0 * dst_grid['lat'] + 1.0

        # Should match closely (bicubic should be exact for linear, within numerical precision)
        valid_mask = get_valid_mask(result)
        np.testing.assert_allclose(result[valid_mask], expected[valid_mask], rtol=1e-8, atol=1e-10)

    def test_bicubic_with_parameters(self):
        """Test bicubic with different bias and tension parameters"""
        src_grid = create_regular_grid(20, 15, (-10.0, 10.0), (-5.0, 5.0))
        dst_grid = create_regular_grid(15, 12, (-8.0, 8.0), (-4.0, 4.0))

        src_data = np.sin(src_grid['lon'] * np.pi / 10) * np.cos(src_grid['lat'] * np.pi / 5)

        # Test with different parameters
        regridder1 = XYRegridder(src_grid, dst_grid, method='bicubic', bias=0.0, tension=0.0)
        regridder2 = XYRegridder(src_grid, dst_grid, method='bicubic', bias=0.5, tension=0.0)
        regridder3 = XYRegridder(src_grid, dst_grid, method='bicubic', bias=0.0, tension=0.5)

        result1 = regridder1.regrid(src_data)
        result2 = regridder2.regrid(src_data)
        result3 = regridder3.regrid(src_data)

        # All should have valid results
        assert np.sum(~np.isnan(result1)) > 0
        assert np.sum(~np.isnan(result2)) > 0
        assert np.sum(~np.isnan(result3)) > 0

        # Results should be different with different parameters
        valid_mask = get_valid_mask(result1) & get_valid_mask(result2)
        if np.sum(valid_mask) > 10:
            assert not np.allclose(result1[valid_mask], result2[valid_mask])

    def test_bicubic_3d_data(self):
        """Test bicubic regridding with 3D data"""
        src_grid = create_regular_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = create_regular_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        # 3D data (time, lat, lon)
        n_time = 5
        src_data_3d = np.random.rand(n_time, *src_grid['lon'].shape)

        regridder = XYRegridder(src_grid, dst_grid, method='bicubic')
        result = regridder.regrid(src_data_3d)

        assert result.shape == (n_time, *dst_grid['lon'].shape)
        assert np.sum(~np.isnan(result)) > 0


class TestBilinearLinearPreservation:
    """Test that bilinear interpolation exactly preserves linear fields (critical property)"""

    @staticmethod
    def create_simple_grid(nx, ny, lon_range, lat_range):
        """Create a simple regular grid without using edges2bounds"""
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        return {'lon': lon_2d, 'lat': lat_2d, 'type': 'regular'}

    def test_constant_field_exact(self):
        """Constant field should be exactly preserved"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        src_data = np.ones_like(src_grid['lon']) * 42.0

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(src_data, skipna=False)

        valid_mask = get_valid_mask(result)
        assert np.sum(valid_mask) > 0

        max_error = np.max(np.abs(result[valid_mask] - 42.0))
        assert max_error < 1e-13, f'Constant field error {max_error} too large'

    def test_linear_x_exact(self):
        """Linear in x: f(x,y) = a*x + c should be exactly preserved"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        a, c = 2.5, 10.0
        src_data = a * src_grid['lon'] + c

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(src_data, skipna=False)

        expected = a * dst_grid['lon'] + c
        valid_mask = get_valid_mask(result)

        max_error = np.max(np.abs(result[valid_mask] - expected[valid_mask]))
        assert max_error < 1e-12, f'Linear-x error {max_error} too large'

    def test_linear_y_exact(self):
        """Linear in y: f(x,y) = b*y + c should be exactly preserved"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        b, c = 3.7, -5.0
        src_data = b * src_grid['lat'] + c

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(src_data, skipna=False)

        expected = b * dst_grid['lat'] + c
        valid_mask = get_valid_mask(result)

        max_error = np.max(np.abs(result[valid_mask] - expected[valid_mask]))
        assert max_error < 1e-12, f'Linear-y error {max_error} too large'

    def test_bilinear_field_exact(self):
        """General bilinear: f(x,y) = a*x + b*y + c should be exactly preserved"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        a, b, c = 2.0, 3.0, 1.0
        src_data = a * src_grid['lon'] + b * src_grid['lat'] + c

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(src_data, skipna=False)

        expected = a * dst_grid['lon'] + b * dst_grid['lat'] + c
        valid_mask = get_valid_mask(result)

        max_error = np.max(np.abs(result[valid_mask] - expected[valid_mask]))
        assert max_error < 1e-12, f'Bilinear field error {max_error} too large'

    def test_skipna_false_machine_precision(self):
        """Verify skipna=False achieves machine precision for linear fields"""
        src_grid = self.create_simple_grid(20, 15, (-10.0, 10.0), (-5.0, 5.0))
        dst_grid = self.create_simple_grid(15, 12, (-8.0, 8.0), (-4.0, 4.0))

        # Powers of 2 for exact floating point representation
        a, b, c = 2.0, 4.0, 8.0
        src_data = a * src_grid['lon'] + b * src_grid['lat'] + c

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        result = regridder.regrid(src_data, skipna=False)

        expected = a * dst_grid['lon'] + b * dst_grid['lat'] + c
        valid_mask = get_valid_mask(result)

        max_error = np.max(np.abs(result[valid_mask] - expected[valid_mask]))
        assert max_error < 1e-13, f'skipna=False error {max_error} exceeds machine precision'


class TestBicubicLinearPreservation:
    """Test that bicubic interpolation exactly preserves linear fields"""

    @staticmethod
    def create_simple_grid(nx, ny, lon_range, lat_range):
        """Create a simple regular grid"""
        lon = np.linspace(lon_range[0], lon_range[1], nx)
        lat = np.linspace(lat_range[0], lat_range[1], ny)
        lon_2d, lat_2d = np.meshgrid(lon, lat)
        return {'lon': lon_2d, 'lat': lat_2d, 'type': 'regular'}

    def test_constant_field_exact(self):
        """Constant field should be exactly preserved"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        src_data = np.ones_like(src_grid['lon']) * 42.0

        regridder = XYRegridder(src_grid, dst_grid, method='bicubic')
        result = regridder.regrid(src_data, skipna=False)

        valid_mask = get_valid_mask(result)
        assert np.sum(valid_mask) > 0

        max_error = np.max(np.abs(result[valid_mask] - 42.0))
        assert max_error < 1e-13, f'Bicubic constant field error {max_error} too large'

    def test_linear_field_exact(self):
        """Linear field should be exactly preserved with bicubic"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        a, b, c = 2.0, 3.0, 1.0
        src_data = a * src_grid['lon'] + b * src_grid['lat'] + c

        regridder = XYRegridder(src_grid, dst_grid, method='bicubic')
        result = regridder.regrid(src_data, skipna=False)

        expected = a * dst_grid['lon'] + b * dst_grid['lat'] + c
        valid_mask = get_valid_mask(result)

        max_error = np.max(np.abs(result[valid_mask] - expected[valid_mask]))
        assert max_error < 1e-10, f'Bicubic linear field error {max_error} too large'

    @pytest.mark.parametrize('bias', [0.0, 0.5, -0.5])
    def test_linear_exact_varying_bias(self, bias):
        """Linear fields should be exact with varying bias (when tension=0)"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        a, b, c = 2.0, 3.0, 1.0
        src_data = a * src_grid['lon'] + b * src_grid['lat'] + c

        regridder = XYRegridder(src_grid, dst_grid, method='bicubic', bias=bias, tension=0.0)
        result = regridder.regrid(src_data, skipna=False)

        expected = a * dst_grid['lon'] + b * dst_grid['lat'] + c
        valid_mask = get_valid_mask(result)

        max_error = np.max(np.abs(result[valid_mask] - expected[valid_mask]))
        assert max_error < 1e-10, f'Linear not exact with bias={bias}: error={max_error}'

    def test_tension_behavior(self):
        """Verify tension=0 is exact, tension>0 may deviate (intentional)"""
        src_grid = self.create_simple_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        dst_grid = self.create_simple_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))

        a, b, c = 2.0, 3.0, 1.0
        src_data = a * src_grid['lon'] + b * src_grid['lat'] + c

        # With tension=0, should be exact
        regridder_no_tension = XYRegridder(src_grid, dst_grid, method='bicubic', tension=0.0)
        result_no_tension = regridder_no_tension.regrid(src_data, skipna=False)

        expected = a * dst_grid['lon'] + b * dst_grid['lat'] + c
        valid_mask = get_valid_mask(result_no_tension)

        # No tension should be exact
        error_no_tension = np.max(np.abs(result_no_tension[valid_mask] - expected[valid_mask]))
        assert error_no_tension < 1e-10, 'Should be exact with tension=0'


class TestPerformance:
    """Test performance-related aspects"""

    def test_large_grid_handling(self):
        """Test regridder with moderately large grids"""
        # Note: Keep sizes reasonable for unit tests
        src_grid = create_regular_grid(50, 40, (-10.0, 10.0), (-8.0, 8.0))
        dst_grid = create_regular_grid(25, 20, (-5.0, 5.0), (-4.0, 4.0))

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')

        # Should complete without errors
        regridder.compute_weights()
        assert regridder._j_base is not None

        # Test regridding
        test_data = np.random.rand(*src_grid['lon'].shape)
        result = regridder.regrid(test_data)

        assert result.shape == dst_grid['lon'].shape

    def test_sparse_matrix_efficiency(self):
        """Test that bilinear uses compact fractional-index storage"""
        src_grid = create_regular_grid(10, 10, (-5.0, 5.0), (-5.0, 5.0))
        dst_grid = create_regular_grid(5, 5, (-2.0, 2.0), (-2.0, 2.0))

        regridder = XYRegridder(src_grid, dst_grid, method='bilinear')
        regridder.compute_weights()

        # Frac-idx: 4 arrays of n_dst elements (very compact)
        n_dst = dst_grid['lon'].size
        assert regridder._j_base.shape == (n_dst,)
        assert regridder._i_base.shape == (n_dst,)
        assert regridder._frac_a.shape == (n_dst,)
        assert regridder._frac_b.shape == (n_dst,)


class TestSaveLoadWeightsBicubic:
    """Pickle save/load round-trip for the bicubic method."""

    def setup_method(self):
        self.src_grid = create_regular_grid(15, 12, (-5.0, 5.0), (-4.0, 4.0))
        self.dst_grid = create_regular_grid(10, 8, (-4.0, 4.0), (-3.0, 3.0))
        self.test_data = np.sin(self.src_grid['lat']) * np.cos(self.src_grid['lon'])

    def test_save_load_bicubic(self):
        r1 = XYRegridder(self.src_grid, self.dst_grid, method='bicubic')
        r1.compute_weights()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        try:
            r1.save_weights(fname)
            r2 = XYRegridder(self.src_grid, self.dst_grid, method='bicubic')
            r2.load_weights(fname)
            assert r2._j_base is not None
            assert r2._frac_a is not None
            result1 = r1.regrid(self.test_data)
            result2 = r2.regrid(self.test_data)
            np.testing.assert_allclose(result1, result2, equal_nan=True)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_grid_mismatch_raises(self):
        r1 = XYRegridder(self.src_grid, self.dst_grid, method='bicubic')
        r1.compute_weights()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        try:
            r1.save_weights(fname)
            wrong_src = create_regular_grid(3, 3, (-1.0, 1.0), (-1.0, 1.0))
            r_bad = XYRegridder(wrong_src, self.dst_grid, method='bicubic')
            with pytest.raises(ValueError, match="doesn't match saved weights shape"):
                r_bad.load_weights(fname)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)


class TestSaveLoadWeightsConservative:
    """Pickle save/load round-trip for the conservative method."""

    def setup_method(self):
        self.src_grid = create_regular_grid(8, 6, (-4.0, 4.0), (-3.0, 3.0))
        self.dst_grid = create_regular_grid(5, 4, (-2.0, 2.0), (-2.0, 2.0))
        self.test_data = np.ones(self.src_grid['lon'].shape)

    def test_save_load_conservative(self):
        r1 = XYRegridder(self.src_grid, self.dst_grid, method='conservative')
        r1.compute_weights()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        try:
            r1.save_weights(fname)
            r2 = XYRegridder(self.src_grid, self.dst_grid, method='conservative')
            r2.load_weights(fname)
            assert r2.weights is not None
            assert r2._nb_indptr is not None
            result1 = r1.regrid(self.test_data)
            result2 = r2.regrid(self.test_data)
            np.testing.assert_allclose(result1, result2, equal_nan=True)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)

    def test_uniform_field_after_load(self):
        """Conservative: uniform field should give ~1.0 after load."""
        r1 = XYRegridder(self.src_grid, self.dst_grid, method='conservative')
        r1.compute_weights()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.npz') as f:
            fname = f.name
        try:
            r1.save_weights(fname)
            r2 = XYRegridder(self.src_grid, self.dst_grid, method='conservative')
            r2.load_weights(fname)
            result = r2.regrid(self.test_data)
            valid = get_valid_mask(result)
            if np.sum(valid) > 0:
                np.testing.assert_allclose(result[valid], 1.0, atol=0.1)
        finally:
            if os.path.exists(fname):
                os.unlink(fname)


if __name__ == '__main__':
    pytest.main([__file__])


def test_conservative_with_mask_does_not_modify_the_input_nor_copy_twice():
    import tracemalloc

    ny = nx = 100
    lon, lat = np.meshgrid(np.linspace(0, 10, nx), np.linspace(0, 10, ny))
    dst_lon, dst_lat = np.meshgrid(np.linspace(1, 9, 11), np.linspace(1, 9, 11))
    mask = np.ones((ny, nx), bool)
    mask[:10] = False
    regridder = XYRegridder(
        {"lon": lon, "lat": lat, "mask": mask}, {"lon": dst_lon, "lat": dst_lat}, "conservative"
    )
    regridder.compute_weights()
    data = np.random.rand(30, ny, nx)
    ref = data.copy()
    regridder.regrid(data)
    tracemalloc.start()
    regridder.regrid(data)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    np.testing.assert_array_equal(data, ref)
    assert peak < 1.2 * data.nbytes


def test_conservative_has_weights():
    lon, lat = np.meshgrid(np.linspace(0, 10, 6), np.linspace(0, 10, 5))
    dst_lon, dst_lat = np.meshgrid(np.linspace(2, 8, 3), np.linspace(2, 8, 3))
    for method in "bilinear", "conservative":
        regridder = XYRegridder({"lon": lon, "lat": lat}, {"lon": dst_lon, "lat": dst_lat}, method)
        assert not regridder.has_weights
        regridder.compute_weights()
        assert regridder.has_weights


def get_cells(grid):
    return grid["lat_bounds"].reshape(-1, 4), grid["lon_bounds"].reshape(-1, 4)


def get_areas(grid):
    lats, lons = get_cells(grid)
    return np.array([spherical_area(la, lo) for la, lo in zip(lats, lons)]).reshape(
        grid["lon"].shape
    )


class TestConservative:
    # Same outer edges (0-12 and 0-9.6), different resolutions
    FINE = (24, 16, (0.25, 11.75), (0.3, 9.3))
    COARSE = (6, 8, (1.0, 11.0), (0.6, 9.0))

    @pytest.mark.parametrize("fine_to_coarse", [True, False])
    def test_integral_is_conserved(self, fine_to_coarse):
        fine = create_regular_grid(*self.FINE)
        coarse = create_regular_grid(*self.COARSE)
        np.testing.assert_allclose(get_areas(fine).sum(), get_areas(coarse).sum(), rtol=1e-9)
        src, dst = (fine, coarse) if fine_to_coarse else (coarse, fine)
        field = np.random.default_rng(1).uniform(1.0, 5.0, src["lon"].shape)
        out = XYRegridder(src, dst, "conservative").regrid(field)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(
            (out * get_areas(dst)).sum(), (field * get_areas(src)).sum(), rtol=1e-8
        )

    @pytest.mark.parametrize("curvilinear_source", [True, False])
    def test_curvilinear_grids_give_a_convex_combination_of_the_cells(self, curvilinear_source):
        rotated = create_rotated_grid(18, 14, 5.0, 5.0, 20.0, lon_span=8.0, lat_span=6.0)
        regular = create_regular_grid(12, 10, (2.0, 8.0), (3.0, 7.0))  # inside the rotated one
        src, dst = (rotated, regular) if curvilinear_source else (regular, rotated)
        regridder = XYRegridder(src, dst, "conservative")
        regridder.compute_weights()
        weights = regridder.weights
        # Weights are positive and sum to one on the covered cells
        assert np.all(weights.data > 0)
        sums = np.add.reduceat(weights.data, weights.indptr[:-1][np.diff(weights.indptr) > 0])
        np.testing.assert_allclose(sums, 1.0, atol=1e-12)
        # So that a constant field is kept, and the data bounds are respected
        field = np.random.default_rng(5).uniform(2.0, 7.0, src["lon"].shape)
        out = regridder.regrid(field)
        covered = ~np.isnan(out)
        assert covered.sum() == (np.diff(weights.indptr) > 0).sum()
        assert out[covered].min() >= field.min() - 1e-12
        assert out[covered].max() <= field.max() + 1e-12
        np.testing.assert_allclose(
            regridder.regrid(np.full(src["lon"].shape, 3.3))[covered], 3.3, atol=1e-12
        )
        if curvilinear_source:
            assert covered.all()  # the regular destination is inside the source

    def test_result_stays_within_the_bounds_of_the_data(self):
        fine = create_regular_grid(*self.FINE)
        coarse = create_regular_grid(*self.COARSE)
        field = np.random.default_rng(2).uniform(-3.0, 8.0, fine["lon"].shape)
        out = XYRegridder(fine, coarse, "conservative").regrid(field)
        assert out.min() >= field.min() - 1e-12 and out.max() <= field.max() + 1e-12
        np.testing.assert_allclose(
            XYRegridder(fine, coarse, "conservative").regrid(np.full(fine["lon"].shape, 4.2)),
            4.2,
            atol=1e-12,
        )

    def test_aligned_coarser_grid_gives_the_mean_of_the_cells(self):
        # Each coarse cell is made of 2 x 2 fine cells: the mean of a field that is linear
        # in longitude is the value at the center
        fine = create_regular_grid(8, 6, (0.5, 7.5), (0.5, 5.5))
        coarse = create_regular_grid(4, 3, (1.0, 7.0), (1.0, 5.0))
        out = XYRegridder(fine, coarse, "conservative").regrid(2.0 * fine["lon"] + 1.0)
        np.testing.assert_allclose(out, 2.0 * coarse["lon"] + 1.0, rtol=0, atol=1e-9)
        # And the plain mean of the 4 fine cells for any field, since they have the same area
        # at this precision in the longitude direction
        field = fine["lon"] ** 2
        out = XYRegridder(fine, coarse, "conservative").regrid(field)
        blocks = field.reshape(3, 2, 4, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(out, blocks, rtol=1e-3)


class TestConservativeDateline:
    """Conservative weights of cells that cross the dateline or are across it"""

    @staticmethod
    def get_grids(offset):
        """Source and destination grids, whose longitudes are shifted by an offset"""
        slon, slat = np.meshgrid(np.arange(170.0, 191.1, 2.0), np.arange(0.0, 8.1, 2.0))
        dlon, dlat = np.meshgrid(np.arange(173.0, 188.1, 3.0), np.arange(1.0, 7.1, 3.0))
        out = []
        for lon, lat in ((slon, slat), (dlon, dlat)):
            lon = lon + offset
            out.append({"lon": ((lon + 180.0) % 360.0) - 180.0, "lat": lat})
        return out

    @pytest.mark.parametrize("offset", [-30.0, 0.0, 17.0])
    def test_same_weights_as_far_from_the_dateline(self, offset):
        # The offset 0 puts the cells on both sides of the dateline, the others do not
        ref_src, ref_dst = self.get_grids(-100.0)
        src, dst = self.get_grids(offset)
        rng = np.random.default_rng(3)
        data = rng.uniform(size=src["lon"].shape)
        ref = XYRegridder(ref_src, ref_dst, "conservative").regrid(data)
        out = XYRegridder(src, dst, "conservative").regrid(data)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out, ref, rtol=0, atol=1e-10)

    def test_constant_field_and_destination_cells_across_the_dateline(self):
        src, _ = self.get_grids(0.0)
        dlon, dlat = np.meshgrid(np.array([178.0, -178.0]), np.array([1.0, 3.0]))
        out = XYRegridder(src, {"lon": dlon, "lat": dlat}, "conservative").regrid(
            np.ones(src["lon"].shape)
        )
        np.testing.assert_allclose(out, 1.0)
