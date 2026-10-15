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


@pytest.mark.parametrize("chunked", [False, True])
def test_float_operators_vs_xarray(chunked):
    """Float arrays use the numba kernels, which must agree with xarray"""
    rng = np.random.default_rng(0)
    da = xr.DataArray(rng.random((3, 4, 5), dtype="f4"), dims=("z", "y", "x"))
    if chunked:
        da = da.chunk({"y": 2})
    pad_width = {"y": (2, 1), "x": 1}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        dap = grid.pad(da, pad_width)
        dapl = grid.pad(da, {"y": (2, 1)}, mode="linear_extrap")
        dac = grid.get_centers(da, ("y", "x"))
        dae = grid.get_edges(da, "x")
        daed = grid.get_edges(da, "x", mode="edge")
        dad = grid.diff(da, "y")
    assert dap.dtype == da.dtype
    ref = da.pad({"y": (2, 1)}, mode="edge").pad({"x": 1}, mode="edge")
    np.testing.assert_array_equal(dap, ref)
    np.testing.assert_allclose(dapl[:, 1], da[:, 0] + (da[:, 0] - da[:, 1]), rtol=1e-6)
    np.testing.assert_allclose(dac, 0.25 * (da[:, :-1, :-1] + da[:, 1:, :-1] + da[:, :-1, 1:] + da[:, 1:, 1:]), rtol=1e-6)
    np.testing.assert_allclose(dae[..., 1:-1], 0.5 * (da[..., :-1] + da[..., 1:]), rtol=1e-6)
    np.testing.assert_allclose(dae[..., 0], da[..., 0] + 0.5 * (da[..., 0] - da[..., 1]), rtol=1e-6)
    np.testing.assert_array_equal(daed[..., 0], da[..., 0])
    np.testing.assert_allclose(dad, da.diff("y"), rtol=1e-6)


def test_int_operators():
    """Other types use xarray"""
    da = xr.DataArray(np.arange(4), dims="x")
    assert grid.pad(da, {"x": 1}, coord_mode="edge").dtype == da.dtype
    np.testing.assert_allclose(grid.get_centers(da, "x"), [0.5, 1.5, 2.5])
    np.testing.assert_array_equal(grid.diff(da, "x"), [1, 1, 1])


def test_pad_modes():
    da = xr.DataArray(np.arange(4.0), dims="x")
    for mode in "constant", "wrap":
        np.testing.assert_array_equal(
            grid.pad(da, {"x": (1, 2)}, mode=mode, coord_mode="edge"),
            da.pad(x=(1, 2), mode=mode),
        )
    np.testing.assert_array_equal(
        grid.pad(da, {"x": 1}, mode="constant", constant_values=7, coord_mode="edge"),
        [7, 0, 1, 2, 3, 7],
    )


def test_dateline():
    """Longitudes stay continuous across the dateline"""
    lon = xr.DataArray(
        [170.0, 178.0, -174.0, -166.0],
        dims="x",
        name="lon",
        attrs={"standard_name": "longitude", "units": "degrees_east"},
    )
    da = xr.DataArray(np.arange(4.0), dims="x", coords={"lon": lon})
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        np.testing.assert_allclose(grid.get_centers(da, "x").lon, [174, -178, -170])
        np.testing.assert_allclose(grid.get_edges(da, "x").lon, [166, 174, -178, -170, -162])
        np.testing.assert_allclose(grid.pad(da, {"x": 1}).lon, [162, 170, 178, -174, -166, -158])
        np.testing.assert_allclose(grid.shift(da, {"x": 1}).lon, [174, -178, -170, -162])
        np.testing.assert_allclose(grid.diff(da, "x").lon, [174, -178, -170])
        np.testing.assert_allclose(grid.diff(lon, "x"), [8, 8, 8])
        np.testing.assert_allclose(grid.get_centers(lon, "x"), [174, -178, -170])
    # Left untouched when continuous
    lon0 = xr.DataArray(lon.values % 360, dims="x", name="lon", attrs=lon.attrs)
    da0 = da.assign_coords(lon=lon0)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*cf.*")
        np.testing.assert_allclose(grid.get_centers(da0, "x").lon, [174, 182, 190])


def test_shift_vs_pad_centers():
    da = xr.DataArray(np.random.default_rng(0).random((3, 4)), dims=("y", "x"))
    for mode in "edge", "linear_extrap":
        for dirs, width in [({"x": -1}, (1, 0)), ({"x": 1}, (0, 1))]:
            ref = grid.get_centers(grid.pad(da, {"x": width}, mode=mode), "x")
            np.testing.assert_allclose(grid.shift(da, dirs, mode=mode), ref)


def test_gradient():
    da = xr.DataArray(
        np.arange(12.0).reshape(3, 4) ** 2,
        dims=("y", "x"),
        coords={"x": [0.0, 1.0, 3.0, 7.0], "y": [0, 1, 2]},
        attrs={"long_name": "Sq"},
    )
    dg = grid.gradient(da, "x")
    assert dg.shape == (3, 3)
    np.testing.assert_allclose(dg, da.diff("x") / da.x.diff("x"))
    np.testing.assert_allclose(dg.x, [0.5, 2.0, 5.0])
    assert dg.long_name == "Sq"
    # 3D coordinate and integer data
    depth = da.x * xr.ones_like(da) + da.y
    di = grid.gradient(da.astype(int), "x", coord=depth)
    np.testing.assert_allclose(di, da.diff("x") / depth.diff("x"))


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
        assert depth.name == ("z" if positive == "up" else "depth")
        assert depth.attrs["positive"] == positive

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


def test_get_dx_dy():
    ds = _lonlat_ds()
    dx, dy = grid.get_dx_dy(ds)
    assert dx.dims == dy.dims == ("lat", "lon")
    assert dx.shape == dy.shape == (4, 5)
    assert dx.attrs["units"] == "m"
    ref = np.deg2rad(1.0) * 6371e3
    np.testing.assert_allclose(dy, ref)
    np.testing.assert_allclose(dx[0], ref)
    # Smaller x resolution at high latitudes
    assert dx[-1, 0] < dx[0, 0]


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

    def test_decode_dz2depth(self):
        """Test the output name of the decoding, and its conflict with a dimension"""
        dz = xr.DataArray(
            np.resize([100, 500, 1000.0], (2, 3)).T,
            dims=("lev", "x"),
            coords={"lev": ("lev", np.arange(3, dtype="d"), {"positive": "down"})},
            name="dz",
        )
        ds = grid.decode_dz2depth(dz.to_dataset())
        assert "depth" in ds.coords
        dz = dz.rename(lev="z")
        dz["z"].attrs["positive"] = "up"
        ds = grid.decode_dz2depth(dz.to_dataset(), errors="ignore")
        assert ds.z.dims == ("z",)
        with pytest.raises(xoa.XoaError):
            grid.decode_dz2depth(dz.to_dataset())


class TestDz2z:
    """Test z and thickness conversions"""

    dz = xr.DataArray([1.0, 3.0, 4.0], dims="nz")

    @pytest.mark.parametrize(
        "positive, kwargs, expected",
        [
            ["down", {}, [0, -1, -4, -8]],
            ["up", {}, [-8, -7, -4, 0]],
            ["up", {"top": 0.5}, [-7.5, -6.5, -3.5, 0.5]],
            ["up", {"bottom": -10}, [-10, -9, -6, -2]],
            ["down", {"top": 2}, [2, 1, -2, -6]],
            ["down", {"bottom": -10}, [-2, -3, -6, -10]],
        ],
    )
    def test_dz2z(self, positive, kwargs, expected):
        z = grid.dz2z(self.dz, positive, zdim="nz", **kwargs)
        np.testing.assert_allclose(z, expected)
        assert z.name == "z" and z.attrs["positive"] == "up"
        zc = grid.dz2z(self.dz, positive, zdim="nz", centered=True, **kwargs)
        np.testing.assert_allclose(zc, 0.5 * (np.array(expected)[1:] + np.array(expected)[:-1]))

    def test_dz2z_errors(self):
        with pytest.raises(xoa.XoaError):
            grid.dz2z(self.dz, "up", zdim="nz", top=0, bottom=-8)
        with pytest.raises(xoa.XoaError):
            grid.dz2z(self.dz, zdim="nz")

    def test_dz2depth_is_minus_dz2z(self):
        z = grid.dz2z(self.dz, "down", zdim="nz", top=2)
        np.testing.assert_allclose(grid.dz2depth(self.dz, "down", zdim="nz", ref=-2), -z)

    @pytest.mark.parametrize("positive", ["up", "down"])
    def test_z2dz_round_trip_with_varying_z(self, positive):
        # Sigma-like layers whose thickness varies in space
        h = xr.DataArray([10.0, 40.0, 90.0], dims="x")
        dz = xr.DataArray(np.full(5, 0.2), dims="nz") * h
        zc = grid.dz2z(dz, "up", zdim="nz", centered=True)
        if positive == "down":
            zc = zc.isel(nz=slice(None, None, -1))
        out = grid.z2dz(zc, zdim="nz")
        assert out.name == "dz" and out.dims == zc.dims
        np.testing.assert_allclose(
            out, dz.transpose(*out.dims).isel(nz=slice(None, None, 1 if positive == "up" else -1))
        )

    def test_z2dz_depth_and_outer_edges(self):
        # Stretched layers: the inner interfaces are the middle of the centers, hence approximate,
        # but the total thickness is exact when the outer ones are provided
        dz = xr.DataArray([1.0, 3.0, 4.0], dims="nz")
        z = grid.dz2z(dz, "up", zdim="nz", centered=True)
        assert not np.isclose(grid.z2dz(z, zdim="nz").sum(), 8.0)
        out = grid.z2dz(z, zdim="nz", top=0, bottom=-8)
        assert np.isclose(out.sum(), 8.0)
        # Depth, which is positive down
        depth = (0 - z).assign_attrs(positive="down").rename("depth")
        np.testing.assert_allclose(grid.z2dz(depth, zdim="nz", top=0, bottom=-8), out)

    def test_get_dz(self):
        dz = xr.DataArray(np.resize([100.0, 300.0], (3, 2)), dims=("lev", "x"), name="dz")
        dz["lev"] = ("lev", np.arange(3.0), {"positive": "down"})
        # From a thickness variable
        np.testing.assert_allclose(grid.get_dz(dz.to_dataset()), dz)
        # From z, which is exact for evenly spaced levels
        z = grid.dz2z(dz, "down", zdim="lev", centered=True)
        da = xr.DataArray(np.zeros(z.shape), dims=z.dims, coords={"z": z})
        np.testing.assert_allclose(grid.get_dz(da).transpose(*dz.dims), dz)
        # Nothing
        assert grid.get_dz(xr.DataArray([1.0], dims="foo"), errors="ignore") is None
        with pytest.raises(xoa.XoaError):
            grid.get_dz(xr.DataArray([1.0], dims="foo"))

    def test_get_dz_terms_and_decode_dz2z(self):
        dz = xr.DataArray(
            np.resize([100, 500, 1000.0], (2, 3)).T,
            dims=("lev", "x"),
            coords={"lev": ("lev", np.arange(3, dtype="d"), {"positive": "down"})},
            name="dz",
        )
        assert grid.get_dz_terms(dz.to_dataset())[1] == "lev"
        assert grid.get_dz_terms(xr.Dataset(), errors="ignore") is None
        ds = grid.decode_dz2z(dz.to_dataset())
        np.testing.assert_allclose(ds.z.isel(x=0), [-50, -350, -1100])


def test_apply_along_dim_name_kwargs():
    """The keywords of a coordinate do not leak to the next ones"""
    da = xr.DataArray(
        np.arange(3.0),
        dims="x",
        coords={"y": ("x", np.arange(3.0)), "z": ("x", np.arange(3.0))},
    )
    calls = []

    def func(da, dim, **kwargs):
        calls.append(kwargs)
        return da

    grid.apply_along_dim(da, "x", func, name_kwargs={"y": {"a": 1}}, b=0)
    assert calls == [{"b": 0}, {"b": 0, "a": 1}, {"b": 0}]
