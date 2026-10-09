# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.conserv` kernels
"""

import numpy as np
import pytest

from xoa.core.grid import create_regular_grid
from xoa.core.conserv import compute_conservative_weights
from xoa.core.regrid import XYRegridder


class TestComputeConservativeWeights:
    """Test conservative weights computation"""

    def setup_method(self):
        """Set up test grids"""
        self.src_grid = create_regular_grid(4, 4, (-2.0, 2.0), (-2.0, 2.0))
        self.dst_grid = create_regular_grid(2, 2, (-1.0, 1.0), (-1.0, 1.0))

    def test_compute_conservative_weights_basic(self):
        """Test basic conservative weights computation"""
        # Flatten bounds for the function
        src_bounds_lat = self.src_grid['lat_bounds'].reshape(-1, 4)
        src_bounds_lon = self.src_grid['lon_bounds'].reshape(-1, 4)
        dst_bounds_lat = self.dst_grid['lat_bounds'].reshape(-1, 4)
        dst_bounds_lon = self.dst_grid['lon_bounds'].reshape(-1, 4)

        row_indices, col_indices, weights, valid_dst = compute_conservative_weights(
            src_bounds_lat, src_bounds_lon, dst_bounds_lat, dst_bounds_lon
        )

        # Should find overlaps between source and destination cells
        assert len(row_indices) > 0
        assert len(col_indices) > 0
        assert len(weights) > 0
        assert len(row_indices) == len(col_indices) == len(weights)

        # All weights should be positive
        assert np.all(weights > 0)

        # Check conservation: weights for each destination cell should sum to 1
        n_dst = self.dst_grid['lon'].size
        for dst_idx in range(n_dst):
            if valid_dst[dst_idx]:
                mask = row_indices == dst_idx
                weights_for_dst = weights[mask]
                weight_sum = np.sum(weights_for_dst)
                # Should be close to 1.0 for conservative method
                assert weight_sum == pytest.approx(1.0, rel=1e-6)

    def test_compute_conservative_weights_identical_grids(self):
        """Test conservative weights with identical source and destination grids"""
        # Same grid for source and destination
        grid = create_regular_grid(3, 3, (-1.0, 1.0), (-1.0, 1.0))

        src_bounds_lat = grid['lat_bounds'].reshape(-1, 4)
        src_bounds_lon = grid['lon_bounds'].reshape(-1, 4)
        dst_bounds_lat = grid['lat_bounds'].reshape(-1, 4)
        dst_bounds_lon = grid['lon_bounds'].reshape(-1, 4)

        row_indices, col_indices, weights, valid_dst = compute_conservative_weights(
            src_bounds_lat, src_bounds_lon, dst_bounds_lat, dst_bounds_lon
        )

        # Should create identity mapping
        n_cells = grid['lon'].size
        assert len(weights) == n_cells

        # Each destination cell should map to exactly one source cell with weight 1
        for i in range(n_cells):
            assert i in row_indices
            assert i in col_indices

            # Find weight for diagonal mapping
            mask = (row_indices == i) & (col_indices == i)
            diagonal_weights = weights[mask]
            assert len(diagonal_weights) == 1
            assert diagonal_weights[0] == pytest.approx(1.0, rel=1e-10)

    def test_compute_conservative_weights_no_overlap(self):
        """Test conservative weights with non-overlapping grids"""
        src_grid = create_regular_grid(2, 2, (-2.0, -1.0), (-2.0, -1.0))
        dst_grid = create_regular_grid(2, 2, (1.0, 2.0), (1.0, 2.0))

        src_bounds_lat = src_grid['lat_bounds'].reshape(-1, 4)
        src_bounds_lon = src_grid['lon_bounds'].reshape(-1, 4)
        dst_bounds_lat = dst_grid['lat_bounds'].reshape(-1, 4)
        dst_bounds_lon = dst_grid['lon_bounds'].reshape(-1, 4)

        row_indices, col_indices, weights, valid_dst = compute_conservative_weights(
            src_bounds_lat, src_bounds_lon, dst_bounds_lat, dst_bounds_lon
        )

        # Should find no overlaps
        assert len(weights) == 0 or np.all(weights == 0)
        assert np.sum(valid_dst) == 0


class TestConservativeBoundaryRenorm:
    """Regression tests: boundary cells must have weight sum == 1 when coarsening."""

    # Fine-to-coarse over the same domain — dst boundary cells exceed src extent before fix.
    src = create_regular_grid(40, 30, (-5.0, 5.0), (-3.0, 3.0))
    dst = create_regular_grid(8, 6, (-5.0, 5.0), (-3.0, 3.0))

    def test_weight_sums_are_one(self):
        r = XYRegridder(self.src, self.dst, method='conservative')
        r.compute_weights()
        w = r.weights
        n_dst = self.dst['lon'].size
        sums = np.array([w.data[w.indptr[j] : w.indptr[j + 1]].sum() for j in range(n_dst)])
        np.testing.assert_allclose(sums, 1.0, atol=1e-10)

    def test_constant_field_preserved(self):
        r = XYRegridder(self.src, self.dst, method='conservative')
        result = r.regrid(np.ones(self.src['lon'].shape))
        np.testing.assert_allclose(result, 1.0, atol=1e-10)


if __name__ == '__main__':
    pytest.main([__file__])


def brute_force_weights(src_lat, src_lon, dst_lat, dst_lon, dst_mask=None):
    """Reference weights: all the source cells are intersected with all the destination ones"""
    from xoa.core.num import EPSILON
    from xoa.core.poly import compute_overlap_area, spherical_area

    links = {}
    for j in range(dst_lat.shape[0]):
        if dst_mask is not None and not dst_mask[j]:
            continue
        area = spherical_area(dst_lat[j], dst_lon[j])
        if area < EPSILON:
            continue
        row = {}
        for i in range(src_lat.shape[0]):
            overlap = max(0.0, compute_overlap_area(src_lat[i], src_lon[i], dst_lat[j], dst_lon[j]))
            if overlap > EPSILON and overlap / area > EPSILON:
                row[i] = overlap / area
        total = sum(row.values())
        for i, weight in row.items():
            links[(j, i)] = weight / total if total > EPSILON else weight
    return links


def get_cells(grid):
    return grid["lat_bounds"].reshape(-1, 4), grid["lon_bounds"].reshape(-1, 4)


def as_links(rows, cols, weights):
    return {(int(j), int(i)): w for j, i, w in zip(rows, cols, weights)}


class TestSpatialIndex:
    """The source cells are indexed, and the result must be the one of the full search"""

    @pytest.mark.parametrize(
        "src,dst",
        [
            ((12, 10, (0, 10), (0, 8)), (7, 6, (1, 9), (1, 7))),  # coarser destination
            ((6, 5, (0, 10), (0, 8)), (21, 17, (1, 9), (1, 7))),  # finer destination
            ((9, 9, (0, 10), (0, 10)), (9, 9, (5, 15), (5, 15))),  # partial overlap
            ((9, 9, (0, 10), (0, 10)), (9, 9, (20, 30), (20, 30))),  # no overlap
            ((10, 8, (-179, 179), (-60, 60)), (6, 5, (-170, 170), (-50, 50))),  # wide
        ],
    )
    def test_same_links_as_the_full_search(self, src, dst):
        src_grid = create_regular_grid(src[0], src[1], src[2], src[3])
        dst_grid = create_regular_grid(dst[0], dst[1], dst[2], dst[3])
        args = get_cells(src_grid) + get_cells(dst_grid)
        rows, cols, weights, valid = compute_conservative_weights(*args)
        expected = brute_force_weights(*args)
        got = as_links(rows, cols, weights)
        assert set(got) == set(expected)
        for key, weight in expected.items():
            np.testing.assert_allclose(got[key], weight, rtol=1e-10)
        assert valid.sum() == len({j for j, _ in expected})

    def test_curvilinear_grids(self):
        from xoa.core.grid import create_rotated_grid

        src_grid = create_rotated_grid(14, 11, 5.0, 5.0, 20.0, 8.0, 6.0)
        dst_grid = create_regular_grid(9, 8, (2.0, 7.0), (2.0, 7.0))
        for src, dst in ((src_grid, dst_grid), (dst_grid, src_grid)):
            args = get_cells(src) + get_cells(dst)
            expected = brute_force_weights(*args)
            rows, cols, weights, _ = compute_conservative_weights(*args)
            got = as_links(rows, cols, weights)
            assert set(got) == set(expected)
            np.testing.assert_allclose(
                [got[k] for k in sorted(got)], [expected[k] for k in sorted(got)], rtol=1e-10
            )

    def test_destination_mask(self):
        src_grid = create_regular_grid(8, 7, (0, 10), (0, 8))
        dst_grid = create_regular_grid(6, 5, (1, 9), (1, 7))
        mask = np.ones(30, bool)
        mask[::3] = False
        args = get_cells(src_grid) + get_cells(dst_grid)
        rows, cols, weights, valid = compute_conservative_weights(*args, dst_mask=mask)
        expected = brute_force_weights(*args, dst_mask=mask)
        assert set(as_links(rows, cols, weights)) == set(expected)
        assert not valid[::3].any()

    def test_links_are_sorted_by_destination_then_source(self):
        src_grid = create_regular_grid(9, 8, (0, 10), (0, 8))
        dst_grid = create_regular_grid(5, 4, (1, 9), (1, 7))
        rows, cols, _, _ = compute_conservative_weights(*get_cells(src_grid), *get_cells(dst_grid))
        order = np.lexsort((cols, rows))
        np.testing.assert_array_equal(order, np.arange(len(rows)))


class TestNoSilentTruncation:
    """The number of links is not limited by the number of source cells"""

    def test_destination_much_finer_than_the_source(self):
        # 36 source cells and 1600 destination ones: more links than 10 per source cell
        lon_c, lat_c = np.meshgrid(np.linspace(0, 10, 6), np.linspace(0, 10, 6))
        lon_f, lat_f = np.meshgrid(np.linspace(1, 9, 40), np.linspace(1, 9, 40))
        regridder = XYRegridder(
            {"lon": lon_c, "lat": lat_c}, {"lon": lon_f, "lat": lat_f}, "conservative"
        )
        regridder.compute_weights()
        assert regridder.weights.nnz > 36 * 10
        out = regridder.regrid(np.ones_like(lon_c))
        np.testing.assert_allclose(out, 1.0)
        # And a field that is linear is reproduced up to the averaging over cells
        out = regridder.regrid(lon_c)
        assert np.abs(out - lon_f).max() < 1.0
        assert np.all(np.diff(out, axis=1) >= -1e-12)  # piecewise constant

    def test_many_links_per_destination_cell(self):
        # A few big destination cells over many source cells: the storage must grow
        src_grid = create_regular_grid(90, 80, (0, 10), (0, 8))
        dst_grid = create_regular_grid(3, 3, (1, 9), (1, 7))
        args = get_cells(src_grid) + get_cells(dst_grid)
        rows, cols, weights, valid = compute_conservative_weights(*args)
        assert len(rows) > 1024  # initial capacity
        assert valid.all()
        sums = np.bincount(rows, weights=weights, minlength=9)
        np.testing.assert_allclose(sums, 1.0)
        assert len(rows) == len(brute_force_weights(*args))


def test_weights_computation_is_fast_on_large_grids():
    import time

    lon, lat = np.meshgrid(np.linspace(0, 10, 120), np.linspace(0, 10, 120))
    dlon, dlat = np.meshgrid(np.linspace(1, 9, 60), np.linspace(1, 9, 60))
    regridder = XYRegridder({"lon": lon, "lat": lat}, {"lon": dlon, "lat": dlat}, "conservative")
    t0 = time.perf_counter()
    regridder.compute_weights()
    elapsed = time.perf_counter() - t0
    assert regridder.weights.nnz == 24336
    assert elapsed < 5.0  # 16 s with the full search
