# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.grid` module
"""
import functools
import warnings
import numpy as np
import xarray as xr
import pytest

import xoa
from xoa import grid


@functools.lru_cache()
def get_da():
    x = xr.DataArray(np.arange(4), dims='x')
    y = xr.DataArray(np.arange(3), dims='y')
    lon = xr.DataArray(np.resize(x * 2.0, (3, 4)), dims=('y', 'x'))
    lat = xr.DataArray(np.resize(y * 3.0, (4, 3)).T, dims=('y', 'x'))
    z = xr.DataArray(np.arange(2), dims='z')
    dep = z * (lat + lon) * 100
    da = xr.DataArray(
        np.resize(lat - lon, (2, 3, 4)),
        dims=('z', 'y', 'x'),
        coords={'dep': dep, 'lat': lat, "z": z, 'lon': lon, 'y': y, 'x': x},
        attrs={"long_name": "SST"},
        name="sst",
    )
    da.encoding.update(cfspecs='croco')
    return da


def test_get_centers():
    """Test grid center calculation"""
    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        dac = grid.get_centers(da, dim=("y", "x"))
    assert dac.shape == (da.shape[0], da.shape[1] - 1, da.shape[2] - 1)
    assert dac.x[0] == 0.5
    assert dac.y[0] == 0.5
    assert dac.lon[0, 0] == 1.0
    assert dac.lat[0, 0] == 1.5
    assert dac.dep[-1, 0, 0] == 250
    assert dac[0, 0, 0] == 0.5
    assert dac.name == 'sst'
    assert dac.long_name == "SST"
    assert dac.encoding["cfspecs"] == "croco"


def test_pad():
    """Test grid padding operations"""
    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        warnings.filterwarnings("error", ".*ist-or-tuple.*")
        dap = grid.pad(da, {"y": 1, "x": 1}, name_kwargs={'dep': {"mode": 'edge'}})
    assert dap.shape == (da.shape[0], da.shape[1] + 2, da.shape[2] + 2)
    assert dap.x[0] == -1
    assert dap.x[-1] == da.sizes['x']
    assert dap.y[0] == -1
    assert dap.y[-1] == da.sizes['y']
    assert dap.lon[0, 0] == -2
    assert dap.lat[0, 0] == -3
    assert dap.dep[-1, 0, 0] == da.dep[-1, 0, 0]
    assert dap[-1, 0, 0] == da[-1, 0, 0]
    assert dap.name == 'sst'
    assert dap.long_name == "SST"
    assert dap.encoding["cfspecs"] == "croco"


def test_get_edges():
    """Test grid edge operations"""
    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        dae = grid.get_edges(da, "y")
    assert dae.shape == (da.shape[0], da.shape[1] + 1, da.shape[2])
    np.testing.assert_allclose(dae.y[:2].values, da.y[:2] - 0.5)
    np.testing.assert_allclose(dae.lat[:2, 0], da.lat[:2, 0] - 1.5)
    # Outer edges are extrapolated
    np.testing.assert_allclose(dae[1, 0, 0], da[1, 0, 0] - 0.5 * (da[1, 1, 0] - da[1, 0, 0]))
    # Unless the end values are asked to be replicated
    dae = grid.get_edges(da, "y", mode="edge")
    assert dae[1, 0, 0] == da[1, 0, 0]


def test_shift():
    """Test grid shifting operations"""
    da = get_da()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        dae = grid.shift(da, {"x": "left", "y": 1})
    assert dae.shape == da.shape
    np.testing.assert_allclose(dae.x.values, da.x - 0.5)
    np.testing.assert_allclose(dae.y.values, da.y + 0.5)
    assert dae[1, 0, 1] == float(da[1, :2, :2].mean())


class TestDepthConversion:
    """Test depth conversion utilities"""

    @pytest.mark.parametrize(
        "positive, expected, ref, ref_type",
        [
            ["down", [0, 100.0, 600.0, 1600.0], None, None],
            ["down", [10, 110.0, 610.0, 1610.0], 10, "top"],
            ["down", [15, 115.0, 615.0, 1615.0], 1615, "bottom"],
            ["up", [-1600, -1500, -1000, 0], None, None],
            ["up", [-1610, -1510, -1010, -10], 1610, "bottom"],
            ["up", [-1595, -1495, -995, 5], 5, "top"],
            ["up", [-1595, -1495, -995, 5], xr.DataArray(5, name="ssh"), "top"],
        ],
    )
    def test_dz2depth(self, positive, expected, ref, ref_type):
        dz = xr.DataArray(
            np.resize([100, 500, 1000.0], (2, 3)).T,
            dims=("z", "x"),
            coords={"z": ("z", np.arange(3, dtype="d"))},
        )

        depth = grid.dz2depth(dz, positive, ref=ref, ref_type=ref_type)
        np.testing.assert_allclose(depth.isel(x=1), expected)
        assert depth.z[0] == -0.5

        depth = grid.dz2depth(dz, positive, ref=ref, ref_type=ref_type, centered=True)
        assert depth[0, 0] == 0.5 * sum(expected[:2])
        assert depth.z[0] == 0


def test_to_rect():
    """Test grid coordinate conversion"""
    x = xr.DataArray(np.arange(4), dims='x')
    y = xr.DataArray(np.arange(3), dims='y')
    lon = xr.DataArray(np.ones((3, 4)), dims=('y', 'x'), coords={"y": y, "x": x})
    lat = xr.DataArray(np.ones((3, 4)), dims=('y', 'x'), coords={"y": y, "x": x})
    temp = xr.DataArray(
        np.ones((2, 3, 4)), dims=('time', 'y', 'x'), coords={'lon': lon, 'lat': lat, "y": y, "x": x}
    )

    tempr = grid.to_rect(temp)
    assert tempr.dims == ('time', 'lat', 'lon')
    assert tempr.lon.ndim == 1
    np.testing.assert_allclose(tempr.lon.values, temp.lon[0].values)

    ds = xr.Dataset({"temp": temp})
    dsr = grid.to_rect(ds)
    assert dsr.temp.dims == tempr.dims


def test_to_rect_curvilinear():
    """A curvilinear grid is left unchanged"""
    x = xr.DataArray(np.arange(4), dims='x')
    y = xr.DataArray(np.arange(3), dims='y')
    xx, yy = np.meshgrid(np.arange(4.0), np.arange(3.0))
    lon = xr.DataArray(xx + 0.1 * yy, dims=('y', 'x'), attrs={"standard_name": "longitude"})
    lat = xr.DataArray(yy + 0.1 * xx, dims=('y', 'x'), attrs={"standard_name": "latitude"})
    temp = xr.DataArray(
        np.ones((3, 4)), dims=('y', 'x'), coords={'lon': lon, 'lat': lat, "y": y, "x": x}
    )
    with pytest.warns(xoa.XoaWarning):
        out = grid.to_rect(temp)
    assert out.lon.ndim == 2
    assert out.dims == temp.dims
    with pytest.raises(xoa.XoaError):
        grid.to_rect(temp, errors="raise")
    assert grid.to_rect(temp, errors="ignore").lon.ndim == 2


def test_grid_ds2grid_dict():
    lon = xr.DataArray(
        np.arange(5.0), dims="lon", attrs={"standard_name": "longitude", "units": "degrees_east"}
    )
    lat = xr.DataArray(
        np.arange(4.0), dims="lat", attrs={"standard_name": "latitude", "units": "degrees_north"}
    )
    ds = xr.Dataset(coords={"lon": lon, "lat": lat})
    ds["mask"] = xr.DataArray(np.ones((4, 5), dtype=bool), dims=("lat", "lon"))
    gd = grid.ds2grid_dict(ds, mask="mask")
    assert gd["dims"] == ("lat", "lon")
    assert gd["lon"].shape == gd["lat"].shape == (4, 5)
    assert gd["type"] == "regular"
    # Edges and bounds are large and only computed on demand
    assert not {"lon_edges", "lat_edges", "lon_bounds", "lat_bounds"} & set(gd)
    gd = grid.ds2grid_dict(ds, mask="mask", bounds=True)
    assert gd["lon_edges"].shape == (5, 6)
    assert gd["lon_bounds"].shape == (4, 5, 4)
    assert gd["mask"].shape == (4, 5)
    assert "time_name" not in gd


def _lonlat_ds():
    lon = xr.DataArray(
        np.linspace(0, 4, 5),
        dims="lon",
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    lat = xr.DataArray(
        np.linspace(0, 3, 4),
        dims="lat",
        attrs={"standard_name": "latitude", "units": "degrees_north"},
    )
    return xr.Dataset(coords={"lon": lon, "lat": lat})


def test_get_resolution():
    ds = _lonlat_ds()
    dx, dy = grid.get_resolution(ds)
    assert dx.dims == dy.dims == ("lat", "lon")
    assert dx.shape == (4, 4)
    assert dy.shape == (3, 5)
    assert dx.attrs["units"] == "m"
    ref = np.deg2rad(1.0) * 6371e3
    np.testing.assert_allclose(dy, ref)
    np.testing.assert_allclose(dx[0], ref)
    # From a data array with 2D coordinates and transposed dimensions
    lon2d, lat2d = xr.broadcast(ds.lon, ds.lat)
    da = xr.DataArray(
        np.zeros(lon2d.shape), dims=lon2d.dims, coords={"lon": lon2d, "lat": lat2d}
    ).transpose("lon", "lat")
    dx2, dy2 = grid.get_resolution(da)
    np.testing.assert_allclose(dx2, dx)


def test_get_median_resolution():
    np.testing.assert_allclose(grid.get_median_resolution(_lonlat_ds()), 1.0)


def test_get_edge_extents():
    ds = _lonlat_ds()
    ext = grid.get_edge_extents(ds, "all", n_cells=2)
    assert list(ext) == ["north", "south", "east", "west"]
    assert ext["north"] == [0.0, 4.0, 2.0, 3.0]
    assert ext["south"] == [0.0, 4.0, 0.0, 1.0]
    assert ext["east"] == [3.0, 4.0, 0.0, 3.0]
    assert ext["west"] == [0.0, 1.0, 0.0, 3.0]
    ext = grid.get_edge_extents(ds, ["west", "north", "west"], n_cells=1)
    assert list(ext) == ["west", "north"]
    assert ext["north"] == [0.0, 4.0, 3.0, 3.0]
    with pytest.raises(xoa.XoaError):
        grid.get_edge_extents(ds, "up")


class TestGetFingerprint:
    @staticmethod
    def get_ds(lon0=0.0, nlat=4):
        lon = xr.DataArray(
            lon0 + np.arange(5.0),
            dims="lon",
            attrs={"standard_name": "longitude", "units": "degrees_east"},
        )
        lat = xr.DataArray(
            np.arange(float(nlat)),
            dims="lat",
            attrs={"standard_name": "latitude", "units": "degrees_north"},
        )
        return xr.Dataset(coords={"lon": lon, "lat": lat})

    def test_equal_grids_have_the_same_fingerprint(self):
        assert grid.get_fingerprint(self.get_ds()) == grid.get_fingerprint(self.get_ds())
        assert isinstance(grid.get_fingerprint(self.get_ds()), str)

    def test_whatever_the_dimensions_and_the_kind_of_object(self):
        ds = self.get_ds()
        ref = grid.get_fingerprint(ds)
        # 1D axes or the same 2D coordinates, with other dimension names
        lon2d, lat2d = xr.broadcast(ds.lon, ds.lat)
        da = xr.DataArray(
            np.zeros((4, 5)),
            dims=("a", "b"),
            coords={"lon": (("a", "b"), lon2d.transpose("lat", "lon").values, ds.lon.attrs)},
        )
        da = da.assign_coords(lat=(("a", "b"), lat2d.transpose("lat", "lon").values, ds.lat.attrs))
        assert grid.get_fingerprint(da) == ref
        # A data array that has the coordinates of the dataset
        assert (
            grid.get_fingerprint(
                xr.DataArray(np.zeros((4, 5)), dims=("lat", "lon"), coords=ds.coords)
            )
            == ref
        )
        # A grid dictionary
        assert grid.get_fingerprint(grid.ds2grid_dict(ds)) == ref

    def test_what_changes_the_fingerprint(self):
        ref = grid.get_fingerprint(self.get_ds())
        assert grid.get_fingerprint(self.get_ds(lon0=1e-6)) != ref
        assert grid.get_fingerprint(self.get_ds(nlat=5)) != ref
        mask = np.ones((4, 5), bool)
        with_mask = grid.get_fingerprint(self.get_ds(), mask=mask)
        assert with_mask != ref
        mask[0, 0] = False
        assert grid.get_fingerprint(self.get_ds(), mask=mask) != with_mask

    def test_mask_by_name(self):
        ds = self.get_ds()
        mask = np.ones((4, 5), bool)
        ds["valid"] = (("lat", "lon"), mask)
        assert grid.get_fingerprint(ds, mask="valid") == grid.get_fingerprint(ds, mask=mask)
