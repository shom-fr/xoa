# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.coords` module
"""

import pytest
import numpy as np
import xarray as xr

import xoa
from xoa import coords


class TestTranspose:
    """Test array transposition functions"""

    @pytest.mark.parametrize(
        "inshape,indims,tdims,mode,outshape,outdims",
        [
            ((1, 2), ('y', 'x'), ("x", "y"), "classic", (2, 1), ("x", "y")),
            ((3, 2), ('y', 'x'), ("t", "y"), "insert", (2, 1, 3), ("x", "t", "y")),
            ((3, 2), ('y', 'x'), ("x", "t", "y"), "compat", (2, 3), ("x", "y")),
            (
                (3, 4, 2),
                ('y', 't', 'x'),
                (Ellipsis, "x", "e", "y"),
                "compat",
                (4, 2, 3),
                ("t", "x", "y"),
            ),
        ],
    )
    def test_transpose(self, inshape, indims, tdims, mode, outshape, outdims):
        da = xr.DataArray(np.ones(inshape), dims=indims)
        dao = coords.transpose(da, tdims, mode)
        assert dao.dims == outdims
        assert dao.shape == outshape


def test_is_lon():
    """Test coordinate identification functions"""
    x = xr.DataArray([5], dims="lon", name="lon")
    y = xr.DataArray([5], dims="lat")
    temp = xr.DataArray(np.ones((1, 1)), dims=('lat', 'lon'), coords={'lon': x, 'lat': y})

    assert coords.is_lon(x)
    assert coords.is_lon(temp.lon)
    assert not coords.is_lon(temp.lat)


def test_copy_coord_attrs():
    ref = xr.DataArray(
        np.zeros(3),
        dims="x",
        coords={"lon": ("x", np.arange(3.0), {"standard_name": "longitude"})},
    )
    res = (ref + 1).copy(deep=True)
    res.lon.attrs.clear()
    assert coords.copy_coord_attrs(res, ref) is res
    assert res.lon.attrs == {"standard_name": "longitude"}
    assert coords.get_lon(res).name == "lon"
    # Only the shared coordinates are concerned
    other = xr.DataArray(np.zeros(3), dims="x")
    assert coords.copy_coord_attrs(other, ref).coords.keys() == set()


def test_as_z():
    heights = xr.DataArray([-30.0, -10.0, 0.0], dims="k", attrs={"units": "m"})
    depths = xr.DataArray([0.0, 10.0, 30.0], dims="k", attrs={"units": "m"})
    # Guessed from the values
    assert coords.as_z(heights) is heights
    np.testing.assert_allclose(coords.as_z(depths), -depths)
    assert coords.as_z(depths).attrs["positive"] == "up"
    # Given by the attribute, even when the values say otherwise
    assert coords.as_z(heights.assign_attrs(positive="up")).equals(
        heights.assign_attrs(positive="up")
    )
    flat = xr.DataArray([0.0, 0.0], dims="k", attrs={"positive": "down"})
    np.testing.assert_allclose(coords.as_z(flat), 0.0)
    np.testing.assert_allclose(coords.as_z(heights.assign_attrs(positive="down")), -heights)


def test_get_depth_from_variable():
    """Test depth coordinate functions"""
    da = xr.DataArray(
        np.ones((2, 3)),
        dims=("depth", "lon"),
        coords={"depth": ("depth", [0, 1]), "lon": [1, 2, 3]},
    )
    depth = coords.get_depth(da)
    assert depth is not None
    np.testing.assert_allclose(depth.values, [0, 1])


def test_geo_stack():
    """Test coordinate stacking functions"""
    ds = xr.Dataset(
        {"temp": (("depth", "lat", "lon"), np.ones((2, 3, 4)))},
        coords={"depth": [0, 1], "lat": [40, 41, 42], "lon": [-10, -9, -8, -7]},
    )
    dss = coords.geo_stack(ds, "npts")
    assert dss.temp.dims == ("depth", "npts")
    assert dss.temp.lon.shape == dss.temp.shape[1:]

    tempc = coords.geo_stack(ds.temp, "npts")
    xr.testing.assert_equal(tempc, dss.temp)


def test_geo_stack_singleton():
    ds = xr.Dataset(coords={"lon": -5.0, "lat": 48.0})
    assert coords.geo_stack(ds).encoding["geo_stack"][0] is None
    dss = coords.geo_stack(ds, "npts", singleton=True)
    assert dss.lon.dims == dss.lat.dims == ("npts",)


@pytest.mark.parametrize("lat", [[40.0, 41.0, 42.0], [42.0, 41.0, 40.0]])
def test_geo_unstack_grid_keeps_coordinates_and_order(lat):
    attrs = {"units": "degrees_north"}
    ref = xr.Dataset(
        coords={
            "lon": ("lon", [-7.0, -9.0, -8.0, -10.0], {"units": "degrees_east"}),
            "lat": ("lat", lat, attrs),
        }
    )
    stacked = coords.geo_stack(ref, "npts", drop=True)
    data = xr.DataArray(
        np.arange(2 * stacked.sizes["npts"]).reshape(2, -1),
        dims=("time", "npts"),
        coords={"lon": stacked.lon, "lat": stacked.lat},
        name="v",
        attrs={"units": "K"},
    )
    out = coords.geo_unstack(data, ref, "npts")
    assert out.dims == ("time", "lat", "lon")
    np.testing.assert_array_equal(out.lon, ref.lon)
    np.testing.assert_array_equal(out.lat, ref.lat)
    assert out.lat.attrs == attrs and out.lon.attrs == ref.lon.attrs
    assert out.attrs == {"units": "K"} and out.name == "v"
    # Values at the right place
    np.testing.assert_array_equal(
        out.isel(time=0, lat=1, lon=2), data.isel(time=0).values[1 * 4 + 2]
    )
    np.testing.assert_array_equal(out.lon.values[2], data.lon.values[1 * 4 + 2])


def test_geo_unstack_curvilinear_points_and_singleton():
    lon, lat = np.meshgrid(np.arange(5.0), np.arange(4.0))
    cur = xr.Dataset(coords={"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)})
    stacked = coords.geo_stack(cur, "npts", drop=True)
    out = coords.geo_unstack(
        xr.Dataset({"v": ("npts", stacked.lon.values)}, coords=stacked.coords), cur
    )
    assert out.v.dims == ("y", "x")
    np.testing.assert_array_equal(out.v, lon)
    np.testing.assert_array_equal(out.lat, lat)

    pts = xr.Dataset(coords={"lon": ("pts", [1.0, 2.0]), "lat": ("pts", [3.0, 4.0])})
    da = xr.DataArray([5.0, 6.0], dims="pts")
    assert coords.geo_unstack(da, pts, "pts") is da

    single = xr.Dataset(coords={"lon": -5.0, "lat": 48.0})
    stacked = coords.geo_stack(single, "npts", singleton=True)
    out = coords.geo_unstack(xr.DataArray([1.0], dims="npts", coords=stacked.coords), single)
    assert out.ndim == 0 and float(out.lon) == -5.0 and float(out.lat) == 48.0


def test_coords_geo_merge():
    # Numpy: same shape
    lon, lat = coords.geo_merge(np.arange(3.0), np.arange(3.0))
    assert lon.dims == lat.dims == ("pts",)
    assert (lon.name, lat.name) == ("lon", "lat")
    # Numpy: rectangular
    lon, lat = coords.geo_merge(np.arange(5.0), np.arange(4.0))
    assert lon.dims == ("pts_y", "pts_x")
    assert lon.shape == (4, 5)
    # DataArrays with different dims
    lon = xr.DataArray(np.arange(5.0), dims="x", name="xx")
    lat = xr.DataArray(np.arange(4.0), dims="y", name="yy")
    lon, lat = coords.geo_merge(lon, lat)
    assert lon.dims == lat.dims == ("y", "x")
    assert (lon.name, lat.name) == ("xx", "yy")
    # Incompatible
    with pytest.raises(xoa.XoaError):
        coords.geo_merge(np.zeros((2, 3)), np.zeros(4))


class TestZDepth:
    """Test the z (positive up) and depth (positive down) conventions"""

    @staticmethod
    def get_depth(name="depth"):
        return xr.DataArray(
            [0.0, 10.0, 50.0], dims="k", name=name, attrs={"units": "m", "long_name": "foo"}
        )

    def test_to_z_and_to_depth(self):
        depth = self.get_depth()
        z = coords.to_z(depth)
        assert z.name == "z"
        np.testing.assert_array_equal(z, [0, -10, -50])
        assert z.attrs == {"units": "m", "positive": "up"}
        assert not np.signbit(z.values[0])
        back = coords.to_depth(z)
        assert back.name == "depth"
        np.testing.assert_array_equal(back, depth)
        assert back.attrs == {"units": "m", "positive": "down"}
        assert coords.to_z(depth, name="zz").name == "zz"

    def test_to_z_reverse(self):
        z = coords.to_z(self.get_depth(), reverse=True)
        np.testing.assert_array_equal(z, [-50, -10, 0])
        da = xr.DataArray(np.arange(6.0).reshape(2, 3), dims=("x", "k"))
        np.testing.assert_array_equal(coords.to_depth(da, reverse="k")[0], [-2, -1, 0])
        np.testing.assert_array_equal(coords.reverse_dim(da)[:, 0], [3, 0])

    def test_to_z_drops_stale_coord(self):
        depth = self.get_depth()
        da = xr.DataArray(np.ones(3), dims="k", coords={"depth": depth})
        z = coords.to_z(da.depth)
        assert "depth" not in z.coords
        assert z.attrs["positive"] == "up"

    def test_get_z_and_depth_from_variable(self):
        z = coords.to_z(self.get_depth())
        da = xr.DataArray(np.ones(3), dims="k", coords={"z": z})
        assert coords.is_z(da.z)
        assert not coords.is_depth(da.z)
        assert coords.get_z(da).name == "z"
        depth = coords.get_depth(da)
        assert depth.name == "depth"
        np.testing.assert_array_equal(depth, [0, 10, 50])
        assert depth.attrs["positive"] == "down"

        da = xr.DataArray(np.ones(3), dims="k", coords={"depth": self.get_depth()})
        zz = coords.get_z(da)
        assert zz.name == "z"
        np.testing.assert_array_equal(zz, [0, -10, -50])

    @pytest.mark.parametrize("name", ["z", "depth"])
    @pytest.mark.parametrize("positive", ["up", "down"])
    def test_positive_attribute_prevails(self, name, positive):
        """The positive attribute decides the sign, whatever the name of the coordinate"""
        vert = self.get_depth(name)
        vert.attrs["positive"] = positive
        da = xr.DataArray(np.ones(3), dims="k", coords={name: vert})
        for func, target in ((coords.get_z, "up"), (coords.get_depth, "down")):
            out = func(da)
            sign = 1 if positive == target else -1
            np.testing.assert_array_equal(out, sign * vert.values)
            assert out.attrs["positive"] == target

    def test_name_decides_without_positive_attribute(self):
        da = xr.DataArray(np.ones(3), dims="k", coords={"depth": self.get_depth()})
        np.testing.assert_array_equal(coords.get_depth(da), [0, 10, 50])
        np.testing.assert_array_equal(coords.get_z(da), [0, -10, -50])

    def test_get_z_errors(self):
        da = xr.DataArray(np.ones(3), dims="k")
        with pytest.raises(xoa.XoaError):
            coords.get_z(da)
        assert coords.get_z(da, errors="ignore") is None
        assert coords.get_depth(da, errors="ignore") is None

    def test_get_z_from_sigma(self):
        ds = xr.Dataset(
            {"temp": (("sig", "nx"), np.ones((5, 3)))},
            coords={
                "sig": (
                    "sig",
                    np.linspace(-1, 0, 5),
                    {
                        "standard_name": "ocean_sigma_coordinate",
                        "formula_terms": "sigma: sig eta: ssh depth: bathy",
                    },
                ),
                "ssh": ("nx", np.zeros(3)),
                "bathy": ("nx", 100.0 * np.ones(3)),
            },
        )
        z = coords.get_z(ds)
        assert z.name == "z"
        np.testing.assert_allclose(z.isel(nx=0), np.linspace(-100, 0, 5))
        depth = coords.get_depth(ds)
        np.testing.assert_allclose(depth.isel(nx=0), np.linspace(100, 0, 5))
        assert depth.attrs["positive"] == "down"

    def test_get_vertical_order(self):
        z = coords.to_z(self.get_depth())
        da = xr.DataArray(np.ones(3), dims="k", coords={"z": z})
        assert coords.get_vertical(da).name == "z"
        da = da.assign_coords(depth=self.get_depth())
        assert coords.get_vertical(da).name == "depth"


def test_to_z_to_depth_keep_vertical_ref():
    z = xr.DataArray(
        [-2.0, 0.0], dims="k", attrs={"units": "m", "positive": "up", "xoa_vertical_ref": "surface"}
    )
    assert coords.to_depth(z).attrs["xoa_vertical_ref"] == "surface"
    assert coords.to_z(coords.to_depth(z)).attrs["xoa_vertical_ref"] == "surface"
