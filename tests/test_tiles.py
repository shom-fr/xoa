"""
Test the :mod:`xoa.tiles` module
"""

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
import xarray as xr

from xoa import exceptions
from xoa.tiles import TileMerger, merge_tiles


@pytest.fixture
def full():
    lon = -5.0 + 0.1 * np.arange(20)
    lat = 45.0 + 0.05 * np.arange(12)
    data = np.random.default_rng(0).random((3, 12, 20)).astype("f")
    return xr.Dataset(
        {"u": (("time", "lat", "lon"), data), "t": ("time", [1.0, 2.0, 3.0])},
        coords={"lon": lon, "lat": lat, "time": [0, 1, 2]},
    )


def cut(full, xs, ys):
    return full.isel(lon=slice(*xs), lat=slice(*ys))


def test_roundtrip_inferred(full):
    tiles = [cut(full, (0, 12), (0, 7)), cut(full, (10, 20), (0, 7)), cut(full, (0, 20), (7, 12))]
    out = merge_tiles(tiles)
    xr.testing.assert_allclose(out.u.transpose(*full.u.dims), full.u)
    np.testing.assert_allclose(out.lon, full.lon)
    np.testing.assert_allclose(out.lat, full.lat)
    xr.testing.assert_equal(out.t, full.t)


def test_gap_and_explicit_grid(full):
    tiles = [cut(full, (0, 5), (0, 4)), cut(full, (10, 20), (6, 12))]
    tm = TileMerger(dx=0.1, dy=0.05, lon0=-5.0, lat0=45.0, tiles=tiles)
    assert (tm.grid["nx"], tm.grid["ny"]) == (20, 12)
    out = tm.merge()
    assert out.u.isnull().isel(lon=10, lat=0).all()
    assert out.u.notnull().isel(lon=2, lat=0).all()
    assert out.u.shape == (3, 12, 20)


def test_descending_lat_and_dataarray(full):
    tile = cut(full, (0, 20), (0, 12)).u.isel(lat=slice(None, None, -1))
    out = merge_tiles(tile)
    np.testing.assert_allclose(out.u.transpose(*full.u.dims), full.u)


def test_2d_coords(full):
    lon2d, lat2d = np.meshgrid(full.lon, full.lat)
    ds = xr.Dataset(
        {"u": (("time", "y", "x"), full.u.values)},
        coords={"lon": (("y", "x"), lon2d), "lat": (("y", "x"), lat2d)},
    )
    out = merge_tiles([ds.isel(x=slice(0, 12)), ds.isel(x=slice(10, 20))])
    assert out.u.shape == (3, 12, 20)
    np.testing.assert_allclose(out.u.values, full.u.values)


def test_overlap_modes(full):
    a, b = cut(full, (0, 12), (0, 12)), cut(full, (10, 20), (0, 12)).copy(deep=True)
    b["u"] = b.u + 1
    last = merge_tiles([a, b], overlap="last").u.values
    first = merge_tiles([a, b], overlap="first").u.values
    mean = merge_tiles([a, b], overlap="mean").u.values
    np.testing.assert_allclose(last[..., 10:12], b.u.values[..., :2])
    np.testing.assert_allclose(first[..., 10:12], a.u.values[..., 10:12])
    np.testing.assert_allclose(mean[..., 10:12], a.u.values[..., 10:12] + 0.5, rtol=1e-6)
    with pytest.raises(exceptions.XoaError, match="differs"):
        merge_tiles([a, b], overlap="raise")
    merge_tiles([a, cut(full, (10, 20), (0, 12))], overlap="raise")


def test_errors(full):
    with pytest.raises(exceptions.XoaError, match="not on the grid"):
        merge_tiles(
            [
                cut(full, (0, 10), (0, 12)),
                cut(full, (10, 20), (0, 12)).assign_coords(lon=lambda ds: ds.lon + 0.05),
            ],
            dx=0.1,
            dy=0.05,
        )
    with pytest.raises(exceptions.XoaError, match="No tile"):
        TileMerger(dx=1, dy=1, lon0=0, lat0=0, nx=2, ny=2).merge()
    with pytest.raises(exceptions.XoaError, match="Invalid overlap"):
        TileMerger(overlap="bad")
    with pytest.raises(exceptions.XoaError, match="exceeds"):
        merge_tiles(full, nx=5)


def test_files_and_glob(full, tmp_path):
    cut(full, (0, 12), (0, 12)).to_netcdf(tmp_path / "t1.nc")
    cut(full, (10, 20), (0, 12)).to_netcdf(tmp_path / "t2.nc")
    out = merge_tiles(str(tmp_path / "t*.nc"))
    np.testing.assert_allclose(out.u.values, full.u.values)
    tm = TileMerger(tiles=[tmp_path / "t1.nc"])
    assert tm._tiles[0].name == "t1"
    tm.close()


def test_memory_repr_and_warning(full):
    tm = TileMerger(tiles=full, max_memory=100)
    assert tm.nbytes == full.u.size * 4
    assert "memory~" in repr(tm) and "x[0:20]" in str(tm)
    with pytest.warns(exceptions.XoaWarning, match="more than"):
        tm.merge()
    assert TileMerger(overlap="mean", tiles=full).nbytes == full.u.size * 8


def test_plot(full):
    tm = TileMerger(tiles=[cut(full, (0, 12), (0, 7)), cut(full, (10, 20), (0, 12))])
    ax = tm.plot()
    assert "memory" in ax.get_legend().get_title().get_text().lower()
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    tm.plot(ax=ax, labels=False, cells=False)
    lon, lat, counts = tm.get_coverage()
    assert counts.max() == 2 and counts.min() == 0


def test_plot_single_color_many_tiles(full):
    tiles = [full.isel(lon=slice(i, i + 3)) for i in range(0, 18, 2)]
    tm = TileMerger(tiles=tiles)
    colors = tm._get_tile_colors_()
    assert (colors[:-1] != colors[1:]).all()
    ax = tm.plot(color="C0")
    assert "Tiles" in [t.get_text() for t in ax.get_legend().get_texts()]


def lon_tile(lon, name):
    lon = np.asarray(lon, "d")
    return xr.Dataset(
        {"u": (("lat", "lon"), np.tile(lon, (3, 1)))},
        coords={"lon": lon, "lat": np.arange(3.0)},
        attrs={"name": name},
    )


@pytest.mark.parametrize(
    "tiles, lon_range, lon_min, lon_max",
    [
        # 0/360 seam, convention of the first tile
        ([np.arange(340.0, 360), np.arange(0.0, 20)], None, 0, 359),
        ([np.arange(340.0, 360), np.arange(0.0, 20)], (-180, 180), -20, 19),
        # Mixed conventions of the same place
        ([np.arange(-20.0, 0), np.arange(0.0, 20), np.arange(340.0, 360)], None, -20, 19),
        # Dateline, with a doubt on the convention
        ([np.arange(160.0, 180), np.arange(-180.0, -160)], None, -180, 179),
        ([np.arange(160.0, 180), np.arange(-180.0, -160)], (0, 360), 160, 199),
        # A tile that crosses the dateline
        ([np.r_[170:180, -180:-170]], None, -180, 179),
    ],
)
def test_lon_wrapping(tiles, lon_range, lon_min, lon_max):
    n = sum(len(lon) for lon in tiles)
    out = merge_tiles([lon_tile(lon, str(i)) for i, lon in enumerate(tiles)], lon_range=lon_range)
    lon = out.lon.values
    assert lon.min() == lon_min and lon.max() == lon_max
    assert (np.diff(lon) > 0).all()
    # Compact grid, fully filled, with the values at the right longitudes
    assert out.lon.size <= n and out.u.notnull().all()
    np.testing.assert_allclose((out.u.isel(lat=0).values - lon) % 360, 0, atol=1e-9)


def test_lon_range():
    assert TileMerger(tiles=lon_tile(np.arange(0.0, 90), "a")).lon_range == (-180.0, 180.0)
    assert TileMerger(tiles=lon_tile(np.arange(170.0, 190), "a")).lon_range == (0.0, 360.0)
    assert TileMerger(lon_range=(-10, 350)).lon_range == (-10.0, 350.0)
    with pytest.raises(exceptions.XoaError, match="360"):
        TileMerger(lon_range=(0, 180))


def test_lon0_wrapping():
    tiles = [lon_tile(np.arange(350.0, 360), "a"), lon_tile(np.arange(0.0, 10), "b")]
    tm = TileMerger(lon0=-10.0, dx=1.0, tiles=tiles)
    assert tm.grid["lon0"] == -10.0 and tm.grid["nx"] == 20


def test_plot_across_dateline():
    tm = TileMerger(
        tiles=[lon_tile(np.arange(170.0, 180), "a"), lon_tile(np.arange(-180.0, -170), "b")]
    )
    ax = tm.plot()
    assert ax.projection.proj4_params["lon_0"] == pytest.approx(180.0, abs=1)
