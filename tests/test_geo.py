# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.geo` and :mod:`xoa.core.geo` modules
"""

import pytest
import numpy as np
import xarray as xr

from xoa.core import geo as core_geo
from xoa import geo


class TestDistance:
    """Test distance calculation functions"""

    def test_haversine(self):
        assert geo.haversine(0.0, 0.0, 180.0, 0.0, 1.0) == np.pi
        assert geo.haversine(0, -90.0, 0, 90.0, 1.0) == np.pi
        assert geo.haversine(0, -90.0, 0, 90.0, 2.0) == 2 * np.pi
        assert int(geo.haversine(0, 0, 10.0, 10.0)) == 1568520

    def test_cdist(self):
        xy0 = np.array([[0, 1], [0.0, 1.0]]).T
        xy1 = np.array([[0, 0.5, 180.0], [0.0, 0.5, 0]]).T
        dd = geo.cdist(xy0, xy1, radius=1)
        assert dd.shape == (2, 3)
        assert dd[0, 0] == 0.0
        assert dd[0, -1] == np.pi

    def test_pdist(self):
        xy0 = np.array([[0, 1], [0.0, 1.0]]).T
        dd = geo.pdist(xy0)
        assert dd.shape == (2, 2)
        ddc = geo.pdist(xy0, compact=True)
        assert ddc.shape == (1,)
        assert ddc[0] == dd[1, 0]


class TestGetDistances:
    @staticmethod
    def get_ds(lons, lats):
        return xr.Dataset(
            coords={
                "lon": ("p", lons, {"standard_name": "longitude"}),
                "lat": ("p", lats, {"standard_name": "latitude"}),
            }
        )

    def test_self_distances_have_distinct_dims(self):
        ds = self.get_ds([0.0, 10.0, 20.0], [0.0, 0.0, 0.0])
        dd = geo.get_distances(ds, units="km")
        assert dd.dims == ("npts0", "npts1") and dd.shape == (3, 3)
        assert dd.attrs["units"] == "km"
        np.testing.assert_allclose(np.diag(dd), 0)
        np.testing.assert_allclose(dd, dd.T)
        np.testing.assert_allclose(dd[0, 1], geo.haversine(0, 0, 10, 0) * 1e-3)

    def test_between_two_sets(self):
        dd = geo.get_distances(self.get_ds([0.0, 10.0], [0.0, 0.0]), self.get_ds([0.0], [10.0]))
        assert dd.dims == ("npts0", "npts1") and dd.shape == (2, 1)
        np.testing.assert_allclose(dd[0, 0], geo.haversine(0, 0, 0, 10))


def test_clusterize():
    """Test clustering functions"""
    x = [0, 1, 3, 4, 8, 9, 10.0]
    y = x
    ds = xr.Dataset(coords={"lon": ("npts", x), "lat": ("npts", y)})
    clusters = geo.clusterize(ds, npmax=3, split=True)
    assert len(clusters) == 3
    assert set([c.sizes["npts"] for c in clusters]) == {2, 2, 3}


class TestGeoKDTree:
    """Test the high level KD-tree"""

    @pytest.fixture(scope="class")
    def grid(self):
        lon = xr.DataArray(np.arange(-10.0, 10.1, 1.0), dims="x", name="lon")
        lat = xr.DataArray(np.arange(40.0, 50.1, 1.0), dims="y", name="lat")
        return xr.DataArray(
            np.arange(lat.size * lon.size, dtype="d").reshape(lat.size, lon.size),
            dims=("y", "x"),
            coords={"lon": lon, "lat": lat},
        )

    @pytest.fixture(scope="class")
    def curv(self, grid):
        lat, lon = xr.broadcast(grid.lat, grid.lon)
        return xr.DataArray(
            grid.values,
            dims=("y", "x"),
            coords={"lon": lon.rename("lon"), "lat": lat.rename("lat")},
        )

    @pytest.fixture(scope="class")
    def points(self):
        return xr.Dataset(coords={"lon": ("p", [0.2, -9.7, 30.0]), "lat": ("p", [44.9, 40.1, 0.0])})

    def test_query_rectilinear_and_curvilinear(self, grid, curv, points):
        for obj in grid, curv:
            dist, idx = geo.GeoKDTree(obj).query(points, k=1)
            assert dist.dims == ("p",)
            assert set(idx) == {"y", "x"}
            sel = obj.isel(idx)
            assert sel.dims == ("p",)
            np.testing.assert_array_equal(sel.lon.values[:2], [0.0, -10.0])
            np.testing.assert_array_equal(sel.lat.values[:2], [45.0, 40.0])
            assert dist.values[0] == pytest.approx(geo.haversine(0.2, 44.9, 0.0, 45.0), rel=1e-6)

    def test_query_k_units_max_dist(self, grid, points):
        tree = geo.GeoKDTree(grid)
        dist, idx = tree.query(points, k=3)
        assert dist.dims == ("p", "neighbor") and idx["x"].shape == (3, 3)
        distkm, _ = tree.query(points, k=3, units="km")
        np.testing.assert_allclose(distkm * 1e3, dist)
        dist, idx = tree.query(points, max_dist=50, units="km")
        assert np.isinf(dist.values[2]) and idx["x"].values[2] == -1
        assert np.isfinite(dist.values[:2]).all()

    def test_query_tuple(self, grid):
        dist, idx = geo.GeoKDTree(grid).query((0.1, 45.1))
        assert dist.dims == () and idx["y"].item() == 5
        dist, idx = geo.GeoKDTree(grid).query(([0.1, 3.1], [45.1, 41.9]))
        assert dist.dims == ("target",) and idx["x"].values.tolist() == [10, 13]

    def test_query_radius(self, grid, points):
        tree = geo.GeoKDTree(grid)
        found = tree.query_radius(points, 100, units="km")
        assert len(found) == 3 and found[2]["x"].size == 0
        sel = grid.isel(found[0])
        d = geo.haversine(0.2, 44.9, sel.lon.values, sel.lat.values)
        assert (d <= 100e3).all() and sel.size > 1
        mask = tree.query_radius(points, 100e3, output="mask")
        assert mask.dims == grid.dims and mask.dtype == bool
        union = set(zip(*[np.concatenate([f[d] for f in found]) for d in ("y", "x")]))
        assert mask.sum().item() == len(union)
        with pytest.raises(Exception):
            tree.query_radius(points, 1, output="bad")

    def test_grid_outputs(self, grid, points):
        tree = geo.GeoKDTree(grid)
        # Radius: closest target distance, nan elsewhere
        dist = tree.query_radius(points, 100, units="km", output="dist")
        mask = tree.query_radius(points, 100, units="km", output="mask")
        assert dist.dims == mask.dims == grid.dims
        assert dist.attrs["units"] == "km"
        np.testing.assert_array_equal(mask.values, dist.notnull().values)
        assert float(dist.max()) <= 100
        lon2, lat2 = np.meshgrid(grid.lon.values, grid.lat.values)
        ref = geo.haversine(
            lon2, lat2, points.lon.values[:, None, None], points.lat.values[:, None, None]
        ).min(axis=0)
        sel = ref <= 100e3
        np.testing.assert_array_equal(mask.values, sel)
        np.testing.assert_allclose(dist.values[sel] * 1e3, ref[sel], rtol=1e-6)
        # k nearest: the closest neighbor of each target is selected
        dk = tree.query(points, k=1, output="dist")
        mk = tree.query(points, k=2, output="mask")
        assert dk.dims == grid.dims and int(dk.notnull().sum()) == 3
        assert float(dk.min()) < 100e3 and int(mk.sum()) >= 5
        dkm = tree.query(points, k=1, max_dist=50, units="km", output="dist")
        assert dkm.attrs["units"] == "km" and int(dkm.notnull().sum()) == 2

    def test_wrapper_and_accessor(self, grid, points):
        dist0, idx0 = geo.GeoKDTree(grid).query(points, k=2)
        dist1, idx1 = geo.get_neighbors(grid, points, k=2)
        xr.testing.assert_equal(dist0, dist1)
        assert idx0["x"].equals(idx1["x"])
        mask = geo.get_neighbors(grid, points, radius=100, units="km", output="mask")
        assert mask.sum() > 0
        # Accessor cache
        import xoa  # noqa: F401

        tree = grid.xoa.get_kdtree()
        assert grid.xoa.get_kdtree() is tree
        assert grid.xoa.get_kdtree(leafsize=8) is not tree
        dist2, _ = grid.xoa.get_neighbors(points, k=2)
        xr.testing.assert_equal(dist0, dist2)
        assert grid.xoa.get_neighbors(points, radius=100, units="km", output="mask").equals(mask)
        with pytest.raises(Exception):
            geo.get_neighbors(grid, points)


def test_clusterize_seed():
    rng = np.random.default_rng(0)
    ds = xr.Dataset(
        {"v": ("npts", rng.normal(size=30))},
        coords={"lon": ("npts", rng.uniform(0, 1, 30)), "lat": ("npts", rng.uniform(0, 1, 30))},
    )
    sizes = [
        [c.sizes["npts"] for c in geo.clusterize(ds, npmax=8, split=True, seed=3)] for _ in range(2)
    ]
    assert sizes[0] == sizes[1]
    first = geo.clusterize(ds, npmax=8, split=True, seed=3)
    again = geo.clusterize(ds, npmax=8, split=True, seed=3)
    for c0, c1 in zip(first, again):
        np.testing.assert_array_equal(c0.indices, c1.indices)
