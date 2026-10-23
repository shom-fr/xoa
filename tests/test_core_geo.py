"""
Test the :mod:`xoa.core.geo` module
"""

import numpy as np
import pytest

from xoa.core import geo


class TestHaversine:
    """Test haversine distance calculation on unit sphere"""

    def test_haversine_scalar(self):
        """Test haversine with scalar inputs"""
        # Distance from (0,0) to (180,0) should be pi on unit sphere
        dist = geo.haversine(0.0, 0.0, 180.0, 0.0)
        np.testing.assert_allclose(dist, np.pi)

    def test_haversine_poles(self):
        """Test haversine between poles"""
        dist = geo.haversine(0.0, -90.0, 0.0, 90.0)
        np.testing.assert_allclose(dist, np.pi)

    def test_haversine_same_point(self):
        """Test haversine for same point"""
        dist = geo.haversine(10.0, 20.0, 10.0, 20.0)
        np.testing.assert_allclose(dist, 0.0)

    def test_haversine_array(self):
        """Test haversine with array inputs"""
        lon0 = np.array([0.0, 0.0, 10.0])
        lat0 = np.array([0.0, -90.0, 20.0])
        lon1 = np.array([180.0, 0.0, 10.0])
        lat1 = np.array([0.0, 90.0, 20.0])

        dists = geo.haversine(lon0, lat0, lon1, lat1)
        assert dists.shape == (3,)
        np.testing.assert_allclose(dists[0], np.pi)
        np.testing.assert_allclose(dists[1], np.pi)
        np.testing.assert_allclose(dists[2], 0.0)


class TestBearing:
    """Test bearing angle calculation"""

    def test_bearing_north(self):
        """Test bearing pointing north (90° in math convention)"""
        angle = geo.bearing(0.0, 0.0, 0.0, 90.0)
        np.testing.assert_allclose(angle, 90.0, atol=1e-10)

    def test_bearing_east(self):
        """Test bearing pointing east (0° in math convention)"""
        angle = geo.bearing(0.0, 0.0, 90.0, 0.0)
        np.testing.assert_allclose(angle, 0.0, atol=1e-10)

    def test_bearing_south(self):
        """Test bearing pointing south (-90° in math convention)"""
        angle = geo.bearing(0.0, 90.0, 0.0, 0.0)
        np.testing.assert_allclose(angle, -90.0, atol=1e-10)

    def test_bearing_west(self):
        """Test bearing pointing west (180° in math convention)"""
        angle = geo.bearing(0.0, 0.0, -90.0, 0.0)
        np.testing.assert_allclose(angle, 180.0, atol=1e-10)

    def test_bearing_array(self):
        """Test bearing with array inputs"""
        lon0 = np.array([0.0, 0.0, 0.0, 0.0])
        lat0 = np.array([0.0, 0.0, 90.0, 0.0])
        lon1 = np.array([0.0, 90.0, 0.0, -90.0])
        lat1 = np.array([90.0, 0.0, 0.0, 0.0])

        angles = geo.bearing(lon0, lat0, lon1, lat1)
        assert angles.shape == (4,)
        np.testing.assert_allclose(angles[0], 90.0, atol=1e-10)  # North
        np.testing.assert_allclose(angles[1], 0.0, atol=1e-10)  # East
        np.testing.assert_allclose(angles[2], -90.0, atol=1e-10)  # South
        np.testing.assert_allclose(angles[3], 180.0, atol=1e-10)  # West

    def test_bearing_same_point(self):
        """Test bearing for same point (undefined but should not error)"""
        angle = geo.bearing(10.0, 20.0, 10.0, 20.0)
        # Just check it returns a value without error
        assert isinstance(angle, (float, np.floating))

    def test_bearing_diagonal(self):
        """Test bearing for diagonal direction"""
        # From (0,0) to (45,45) should give a specific angle
        angle = geo.bearing(0.0, 0.0, 45.0, 45.0)
        # Just verify it's in reasonable range and consistent
        assert -180 <= angle <= 180
        assert isinstance(angle, (float, np.floating))


class TestGridHelpers:
    """Test the grid search and cell location helpers"""

    def test_diff_lon(self):
        assert geo.diff_lon(170.0, -170.0) == 20.0
        assert geo.diff_lon(-170.0, 170.0) == -20.0

    def test_normalize_longitude(self):
        assert geo.normalize_longitude(190.0) == -170.0
        assert geo.normalize_longitude(-190.0) == 170.0

    def test_closest_point_fast(self):
        for n in (5, 120):
            lons, lats = np.meshgrid(np.arange(n, dtype="d"), np.arange(n, dtype="d"))
            assert geo.closest_point_fast(lons, lats, 2.1, 3.2) == (2, 3)

    def test_relative_cell_coords(self):
        p, q = geo.relative_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.25, 0.75)
        np.testing.assert_allclose((p, q), (0.25, 0.75))
        assert geo.relative_cell_coords(0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0) == (
            -1.0,
            -1.0,
        )


class TestClosestPointFastOnLargeGrids:
    """The search for large grids must find the closest node of the grid"""

    @staticmethod
    def get_grid(n):
        x, y = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 8, n))
        return x + 0.5 * np.sin(0.7 * y) + 0.05 * y, y + 0.4 * np.sin(0.6 * x) - 0.03 * x

    @staticmethod
    def get_exact(lon, lat, x, y):
        dist = geo.haversine(x, y, lon, lat)
        j, i = np.unravel_index(np.argmin(dist), dist.shape)
        return i, j

    @pytest.mark.parametrize("n", [60, 300, 600])
    def test_same_node_as_the_full_search(self, n):
        lon, lat = self.get_grid(n)
        rng = np.random.default_rng(1)
        jj, ii = rng.integers(2, n - 3, 150), rng.integers(2, n - 3, 150)
        # Points inside the cells, which are not on the nodes
        x = lon[jj, ii] + rng.uniform(-0.5, 0.5, 150) * (lon[jj, ii + 1] - lon[jj, ii])
        y = lat[jj, ii] + rng.uniform(-0.5, 0.5, 150) * (lat[jj + 1, ii] - lat[jj, ii])
        for xi, yi in zip(x, y):
            assert geo.closest_point_fast(lon, lat, xi, yi) == self.get_exact(lon, lat, xi, yi)

    def test_points_that_are_far_from_the_grid(self):
        lon, lat = self.get_grid(200)
        i, j = geo.closest_point_fast(lon, lat, -5.0, -5.0)
        assert (i, j) == self.get_exact(lon, lat, -5.0, -5.0) == (0, 0)
        i, j = geo.closest_point_fast(lon, lat, 20.0, 20.0)
        assert (i, j) == self.get_exact(lon, lat, 20.0, 20.0)

    def test_nan_nodes_are_ignored(self):
        lon, lat = self.get_grid(200)
        x, y = lon[100, 100], lat[100, 100]
        lon, lat = lon.copy(), lat.copy()
        lon[95:106, 95:106] = np.nan
        lat[95:106, 95:106] = np.nan
        i, j = geo.closest_point_fast(lon, lat, x, y)
        assert np.isfinite(lon[j, i])
        dist = geo.haversine(x, y, lon, lat)
        jx, ix = np.unravel_index(np.nanargmin(dist), dist.shape)
        assert (i, j) == (ix, jx)

    def test_all_the_subsampled_nodes_are_nan(self):
        lon, lat = self.get_grid(200)
        mask = np.ones(lon.shape, bool)
        mask[::10, ::10] = (
            False  # the subsampled nodes (step 10) are the only valid ones... inverted
        )
        lon = np.where(mask, lon, np.nan)
        lat = np.where(mask, lat, np.nan)
        x, y = lon[51, 52], lat[51, 52]
        assert geo.closest_point_fast(lon, lat, x, y) == (52, 51)


class TestSphericalKDTree:
    """Test the spherical KD-tree against a brute force search"""

    @pytest.fixture(scope="class")
    def points(self):
        rng = np.random.default_rng(0)
        n = 2000
        lon = rng.uniform(-180, 180, n)
        lat = np.degrees(np.arcsin(rng.uniform(-1, 1, n)))
        lon[3] = np.nan
        # Close to the poles and the dateline
        lon[:3], lat[:3] = [179.9, -179.9, 0.0], [0.0, 0.0, 89.99]
        return lon, lat

    @staticmethod
    def brute_force(lon, lat, tlon, tlat):
        dist = geo.haversine(tlon[:, None], tlat[:, None], lon[None], lat[None])
        return np.where(np.isnan(dist), np.inf, dist * geo.EARTH_RADIUS)

    def test_chord_dist(self):
        dist = np.array([0.0, 1e3, 1e6, np.pi * geo.EARTH_RADIUS])
        chord = geo.dist2chord(dist, geo.EARTH_RADIUS)
        np.testing.assert_allclose(geo.chord2dist(chord, geo.EARTH_RADIUS), dist, atol=1e-6)
        assert np.isinf(geo.chord2dist(np.inf, 1.0))

    def test_query(self, points):
        lon, lat = points
        tlon, tlat = np.array([180.0, 0.0, 12.0, -170.0]), np.array([0.0, 90.0, -45.0, 30.0])
        tree = geo.SphericalKDTree(lon, lat)
        dist, idx = tree.query(tlon, tlat, k=4)
        ref = self.brute_force(lon, lat, tlon, tlat)
        assert dist.shape == idx.shape == (4, 4)
        np.testing.assert_allclose(dist, np.sort(ref, axis=1)[:, :4], atol=1e-3)
        np.testing.assert_array_equal(idx[:, 0], ref.argmin(axis=1))
        dist, idx = tree.query(tlon, tlat)
        assert dist.shape == idx.shape == (4,)

    def test_query_missing(self, points):
        lon, lat = points
        tree = geo.SphericalKDTree(lon, lat)
        dist, idx = tree.query([0.0, np.nan], [0.0, 0.0], max_dist=1.0)
        assert np.isinf(dist).all() and (idx == -1).all()
        small = geo.SphericalKDTree([0.0, 1.0], [0.0, 0.0])
        dist, idx = small.query([0.0], [0.0], k=3)
        assert idx[0, 0] == 0 and idx[0, 1] == 1 and idx[0, 2] == -1 and np.isinf(dist[0, 2])

    def test_query_ball_point(self, points):
        lon, lat = points
        tlon, tlat = np.array([180.0, 0.0, np.nan]), np.array([0.0, 90.0, 0.0])
        tree = geo.SphericalKDTree(lon, lat)
        found = tree.query_ball_point(tlon, tlat, 2e6, workers=-1)
        ref = self.brute_force(lon, lat, tlon[:2], tlat[:2])
        for ff, rr in zip(found[:2], ref):
            np.testing.assert_array_equal(ff, np.flatnonzero(rr <= 2e6))
        assert found[2].size == 0
        found2, dists = tree.query_ball_point(tlon, tlat, 2e6, return_dist=True)
        for f0, f1, dd, rr in zip(found, found2, dists, ref):
            np.testing.assert_array_equal(f0, f1)
            np.testing.assert_allclose(dd, rr[f1], atol=1e-3)
        assert dists[2].size == 0
        assert 3 not in np.concatenate(found)  # nan point

    def test_radius(self):
        tree = geo.SphericalKDTree([0.0, 90.0], [0.0, 0.0], radius=1.0)
        dist, idx = tree.query([0.0], [0.0], k=2)
        np.testing.assert_allclose(dist, [[0.0, np.pi / 2]])

    def test_no_valid_point(self):
        with pytest.raises(ValueError):
            geo.SphericalKDTree([np.nan], [0.0])


class TestLonInRange:
    def test_plain(self):
        lon = np.array([-10.0, 0.0, 5.0, 10.0, 11.0, np.nan])
        np.testing.assert_array_equal(
            geo.lon_in_range(lon, 0.0, 10.0), [False, True, True, True, False, False]
        )

    def test_dateline(self):
        lon = np.array([170.0, 179.0, -179.0, -170.0, 0.0, 350.0])
        np.testing.assert_array_equal(
            geo.lon_in_range(lon, 175.0, 190.0), [False, True, True, True, False, False]
        )

    def test_global(self):
        assert geo.lon_in_range(123.0, -180.0, 180.0)
        assert not geo.lon_in_range(np.nan, 0.0, 360.0)


class TestWindows:
    def test_bbox_window(self):
        lon, lat = np.meshgrid(np.arange(10.0), np.arange(6.0))
        assert geo.bbox_window(lon, lat, 2.0, 4.0, 1.0, 3.0) == (1, 4, 2, 5)
        assert geo.bbox_window(lon, lat, 20.0, 40.0, 1.0, 3.0)[1] <= 0

    def test_max_true_rectangle(self):
        assert geo.max_true_rectangle(np.zeros((3, 4), dtype=bool)) == (0, 0, 0, 0)
        mask = np.ones((4, 5), dtype=bool)
        assert geo.max_true_rectangle(mask) == (0, 4, 0, 5)
        mask[0, 0] = mask[3, 4] = False
        j0, j1, i0, i1 = geo.max_true_rectangle(mask)
        assert (j1 - j0) * (i1 - i0) == 12
        assert mask[j0:j1, i0:i1].all()

    def test_max_true_rectangle_brute_force(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            mask = rng.random((7, 8)) > 0.2
            best = 0
            for j0 in range(7):
                for j1 in range(j0 + 1, 8):
                    for i0 in range(8):
                        for i1 in range(i0 + 1, 9):
                            if mask[j0:j1, i0:i1].all():
                                best = max(best, (j1 - j0) * (i1 - i0))
            j0, j1, i0, i1 = geo.max_true_rectangle(mask)
            assert (j1 - j0) * (i1 - i0) == best
            assert mask[j0:j1, i0:i1].all()


class TestShapes:
    def test_box(self):
        box = geo.Box((170.0, 190.0), (0.0, 10.0))
        assert box.bounds() == (170.0, 190.0, 0.0, 10.0)
        np.testing.assert_array_equal(
            box.contains(np.array([175.0, -175.0, 0.0]), np.array([5.0, 5.0, 5.0])),
            [True, True, False],
        )
        assert geo.Box().bounds() == (0.0, 360.0, -90.0, 90.0)

    def test_circle(self):
        circ = geo.Circle(0.0, 0.0, 1e6)
        lonmin, lonmax, latmin, latmax = circ.bounds()
        dd = 1e6 / geo.EARTH_RADIUS * 180 / np.pi
        np.testing.assert_allclose([lonmax, latmax], dd)
        assert circ.contains(dd * 0.99, 0.0) and not circ.contains(dd * 1.01, 0.0)

    def test_circle_pole(self):
        assert geo.Circle(0.0, 89.0, 500e3).bounds()[:2] == (0.0, 360.0)

    def test_polygon(self):
        poly = geo.Polygon([[179, 0], [-179, 0], [-179, 2], [179, 2]])
        assert poly.bounds() == (179.0, 181.0, 0.0, 2.0)
        np.testing.assert_array_equal(
            poly.contains(np.array([180.0, 170.0]), np.array([1.0, 1.0])), [True, False]
        )


class TestGeoSelector:
    @pytest.fixture
    def rect(self):
        return geo.GeoSelector(np.arange(0.0, 360.0, 10.0), np.arange(-60.0, 61.0, 10.0))

    @pytest.fixture
    def curv(self):
        # A rotated grid
        x, y = np.meshgrid(np.arange(20.0), np.arange(15.0))
        lon = 10 + 0.5 * x - 0.2 * y
        lat = 40 + 0.5 * y + 0.2 * x
        return lon, lat

    @staticmethod
    def brute_nearest(lon, lat, tlon, tlat):
        d = geo.haversine(tlon, tlat, lon, lat)
        return np.unravel_index(np.nanargmin(d), d.shape)

    def test_kinds(self, rect, curv):
        assert rect.kind == "rect"
        assert geo.GeoSelector(*curv).kind == "curv"
        assert geo.GeoSelector([0.0, 1.0], [0.0, 1.0]).kind == "cloud"
        assert geo.GeoSelector([0.0, 1.0], [0.0, 1.0], kind="rect").kind == "rect"
        with pytest.raises(ValueError):
            geo.GeoSelector([0.0, 1.0], [0.0, 1.0], kind="bad")
        with pytest.raises(ValueError):
            geo.GeoSelector(*curv, kind="cloud")

    def test_nearest_rect(self, rect):
        (i, j), dist = rect.nearest([358.0, 14.0], [11.0, -47.0])
        np.testing.assert_array_equal(i, [0, 1])
        np.testing.assert_array_equal(j, [7, 1])
        assert dist[0] > 0
        (i, j), dist = rect.nearest(14.0, -47.0, max_dist=1.0)
        assert i[0] == -1 and j[0] == -1 and np.isinf(dist[0])

    @pytest.mark.parametrize("ntarget", [1, 5])
    def test_nearest_curv(self, curv, ntarget):
        lon, lat = curv
        sel = geo.GeoSelector(lon, lat)
        tlon = np.linspace(11.0, 15.0, ntarget)
        tlat = np.linspace(41.0, 45.0, ntarget)
        (j, i), dist = sel.nearest(tlon, tlat)
        for k in range(ntarget):
            jr, ir = self.brute_nearest(lon, lat, tlon[k], tlat[k])
            assert (j[k], i[k]) == (jr, ir)
        (j, i), dist = sel.nearest(np.nan, 40.0)
        assert j[0] == -1 and np.isinf(dist[0])

    def test_nearest_cloud(self):
        sel = geo.GeoSelector([0.0, 10.0, 179.0], [0.0, 10.0, 0.0])
        (idx,), dist = sel.nearest([-179.0, 9.0], [0.0, 9.0])
        np.testing.assert_array_equal(idx, [2, 1])
        (idx,), dist = sel.nearest([-179.0, 9.0], [0.0, 9.0], max_dist=2e5)
        np.testing.assert_array_equal(idx, [-1, 1])

    def test_box_rect(self, rect):
        (ix, iy), mask = rect.select(geo.Box((15.0, 45.0), (-25.0, 5.0)))
        assert ix == slice(2, 5) and iy == slice(4, 7) and mask is None
        assert rect.lon[ix][0] >= 15 and rect.lat[iy][-1] <= 5

    def test_box_rect_dateline(self, rect):
        (ix, iy), mask = rect.select(geo.Box((345.0, 25.0), (0.0, 10.0)))
        np.testing.assert_array_equal(rect.lon[ix], [350.0, 0.0, 10.0, 20.0])

    def test_box_rect_empty(self, rect):
        (ix, iy), mask = rect.select(geo.Box((1.0, 2.0), (0.0, 10.0)))
        assert ix == slice(0, 0) and mask is None

    @pytest.mark.parametrize("fit", ["outer", "inner"])
    def test_box_curv(self, curv, fit):
        lon, lat = curv
        sel = geo.GeoSelector(lon, lat)
        box = geo.Box((12.0, 16.0), (43.0, 46.0))
        (js, is_), mask = sel.select(box, fit=fit)
        inside = box.contains(lon, lat)
        if fit == "outer":
            assert mask is not None and not mask.all()
            assert mask.sum() == inside.sum()
            np.testing.assert_array_equal(mask, inside[js, is_])
        else:
            assert mask is None
            assert inside[js, is_].all()
            assert inside[js, is_].size > 0

    def test_circle_rect_and_cloud(self, rect):
        circ = geo.Circle(100.0, 20.0, 1500e3)
        (ix, iy), mask = rect.select(circ)
        inside = circ.contains(rect.lon[:, None], rect.lat[None, :])
        np.testing.assert_array_equal(mask, inside[ix, iy])
        (ix, iy), mask = rect.select(circ, fit="inner")
        assert mask is None and inside[ix, iy].all()
        lon, lat = np.meshgrid(rect.lon, rect.lat)
        cloud = geo.GeoSelector(lon.ravel(), lat.ravel())
        (idx,), mask = cloud.select(circ)
        np.testing.assert_array_equal(idx, np.flatnonzero(circ.contains(lon.ravel(), lat.ravel())))
        assert mask is None

    def test_polygon_curv(self, curv):
        lon, lat = curv
        sel = geo.GeoSelector(lon, lat)
        poly = geo.Polygon([[11, 42], [16, 42], [14, 46]])
        (js, is_), mask = sel.select(poly)
        np.testing.assert_array_equal(mask, poly.contains(lon, lat)[js, is_])
        # Tight window
        assert mask[0].any() and mask[:, 0].any() and mask[-1].any() and mask[:, -1].any()

    def test_bad_fit(self, rect):
        with pytest.raises(ValueError):
            rect.select(geo.Box(), fit="bad")


class TestPad:
    @pytest.fixture
    def rect(self):
        return geo.GeoSelector(np.arange(0.0, 360.0, 10.0), np.arange(-60.0, 61.0, 10.0))

    @pytest.fixture
    def curv(self):
        x, y = np.meshgrid(np.arange(20.0), np.arange(15.0))
        return 10 + 0.5 * x - 0.2 * y, 40 + 0.5 * y + 0.2 * x

    def test_half_extents_rect(self, rect):
        hlon, hlat = rect.half_extents
        np.testing.assert_allclose(hlon, 5.0)
        np.testing.assert_allclose(hlat, 5.0)
        sel = geo.GeoSelector([0.0, 1.0, 3.0, 4.0], [0.0, 2.0, 10.0])
        hlon, hlat = sel.half_extents
        np.testing.assert_allclose(hlon, [0.5, 1.0, 1.0, 0.5])
        np.testing.assert_allclose(hlat, [1.0, 4.0, 4.0])

    def test_half_extents_dateline(self):
        hlon, _ = geo.GeoSelector([178.0, 180.0, -178.0], [0.0, 1.0]).half_extents
        np.testing.assert_allclose(hlon, 1.0)

    def test_half_extents_curv(self, curv):
        hlon, hlat = geo.GeoSelector(*curv).half_extents
        assert hlon.shape == curv[0].shape
        # Rotated cells: envelope of corners is wider than half a step
        np.testing.assert_allclose(hlon[5, 5], 0.5 * (0.5 + 0.2))
        np.testing.assert_allclose(hlat[5, 5], 0.5 * (0.5 + 0.2))

    def test_box_rect_points(self, rect):
        box = geo.Box((15.0, 45.0), (-25.0, 5.0))
        (ix, iy), _ = rect.select(box, pad=1)
        assert ix == slice(1, 6) and iy == slice(3, 8)
        # 0.5 grid point is edges, which reach points whose cell touches the box
        (ix, iy), _ = rect.select(box, pad=0.5)
        assert ix == slice(1, 6) and iy == slice(3, 8)
        point = geo.Box((10.0, 10.0), (0.0, 0.0))
        assert rect.select(point, pad=0.1)[0][0] == slice(1, 2)  # margin of 1 degree
        assert rect.select(point, pad=0.5)[0][0] == slice(1, 2)  # 5 degrees
        assert rect.select(point, pad=1)[0][0] == slice(0, 3)  # 10 degrees
        assert rect.select(box, pad="edges") == rect.select(box, pad=0.5)

    def test_box_rect_local_step(self):
        sel = geo.GeoSelector([0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 1.0])
        box = geo.Box((0.0, 2.0), (0.0, 1.0))
        # Cells of lon=4 and lon=8 are 2 wide: margins are 2 * pad * 2 and 2 * pad * 2
        assert sel.select(box)[0][0] == slice(0, 3)
        assert sel.select(box, pad=0.1)[0][0] == slice(0, 3)
        assert sel.select(box, pad=0.5)[0][0] == slice(0, 4)  # edges: cell of 4 touches 2
        assert sel.select(box, pad=1)[0][0] == slice(0, 4)  # 8 is 6 away, margin is 4

    def test_box_rect_dateline_pad(self, rect):
        (ix, _), _ = rect.select(geo.Box((355.0, 5.0), (0.0, 0.0)), pad=1)
        np.testing.assert_array_equal(rect.lon[ix], [350.0, 0.0, 10.0])

    def test_box_curv_pad(self, curv):
        lon, lat = curv
        sel = geo.GeoSelector(lon, lat)
        box = geo.Box((12.0, 16.0), (43.0, 46.0))
        hlon, hlat = sel.half_extents
        ref = box.contains(lon, lat, (2 * 0.5 * hlon, 2 * 0.5 * hlat))
        (js, is_), mask = sel.select(box, pad="edges")
        np.testing.assert_array_equal(mask, ref[js, is_])
        assert ref.sum() > box.contains(lon, lat).sum()

    @pytest.mark.parametrize("units", ["dist", "deg"])
    def test_circle_pad_exact(self, rect, units):
        circ = geo.Circle(100.0, 20.0, 1000e3)
        pad = 500e3 if units == "dist" else 500e3 / geo.EARTH_RADIUS * 180 / np.pi
        (ix, iy), mask = rect.select(circ, pad=pad, pad_units=units)
        big = geo.Circle(100.0, 20.0, 1500e3)
        ref = big.contains(rect.lon[:, None], rect.lat[None, :])
        full = np.zeros_like(ref)
        full[ix, iy] = True if mask is None else mask
        np.testing.assert_array_equal(full, ref)

    def test_circle_pad_points_curv(self, curv):
        lon, lat = curv
        sel = geo.GeoSelector(lon, lat)
        circ = geo.Circle(13.0, 44.0, 150e3)
        (js, is_), mask = sel.select(circ, pad=1)
        hlon, hlat = sel.half_extents
        ref = circ.contains(lon, lat, (2 * hlon, 2 * hlat))
        full = np.zeros_like(ref)
        full[js, is_] = True if mask is None else mask
        np.testing.assert_array_equal(full, ref)

    def test_circle_pad_pole(self):
        lon, lat = np.meshgrid(np.arange(0.0, 360.0, 10.0), np.arange(70.0, 91.0, 5.0))
        sel = geo.GeoSelector(lon, lat)
        circ = geo.Circle(0.0, 80.0, 100e3)
        (js, is_), mask = sel.select(circ, pad=1500e3, pad_units="dist")
        ref = geo.Circle(0.0, 80.0, 1600e3).contains(lon, lat)
        full = np.zeros_like(ref)
        full[js, is_] = True if mask is None else mask
        np.testing.assert_array_equal(full, ref)

    def test_polygon_pad(self, curv):
        lon, lat = curv
        sel = geo.GeoSelector(lon, lat)
        poly = geo.Polygon([[11, 42], [16, 42], [14, 46]])
        base = poly.contains(lon, lat)
        (js, is_), mask = sel.select(poly, pad=60e3, pad_units="dist")
        full = np.zeros_like(base)
        full[js, is_] = True if mask is None else mask
        assert full[base].all() and full.sum() > base.sum()
        # Every added point is close to the polygon, and every close point is added
        d = geo.polygon_distance(poly.poly, lon[~base], lat[~base]) * geo.EARTH_RADIUS
        np.testing.assert_array_equal(full[~base], d <= 60e3)

    def test_inner_pad(self, curv):
        lon, lat = curv
        sel = geo.GeoSelector(lon, lat)
        box = geo.Box((12.0, 16.0), (43.0, 46.0))
        (j0, i0), m0 = sel.select(box, fit="inner")
        (j1, i1), m1 = sel.select(box, fit="inner", pad=1)
        assert m1 is None
        assert (j1.stop - j1.start) * (i1.stop - i1.start) > (j0.stop - j0.start) * (
            i0.stop - i0.start
        )

    def test_cloud(self):
        sel = geo.GeoSelector([0.0, 1.0, 5.0], [0.0, 0.0, 0.0])
        with pytest.raises(ValueError):
            sel.select(geo.Box((0.0, 0.5), (-1.0, 1.0)), pad=1)
        (idx,), _ = sel.select(geo.Box((0.0, 0.5), (-1.0, 1.0)))
        assert idx == slice(0, 1)
        (idx,), _ = sel.select(geo.Box((0.0, 0.5), (-1.0, 1.0)), pad=1.0, pad_units="deg")
        assert idx == slice(0, 2)
        (idx,), _ = sel.select(geo.Circle(0.0, 0.0, 10e3), pad=120e3, pad_units="dist")
        assert idx == slice(0, 2)

    def test_bad_args(self, rect):
        with pytest.raises(ValueError):
            rect.select(geo.Box(), pad="bad")
        with pytest.raises(ValueError):
            rect.select(geo.Box(), pad=1, pad_units="bad")
