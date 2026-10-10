# -*- coding: utf-8 -*-
"""
Test the map, grid, section and stick helpers of the :mod:`xoa.plot` module
"""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import xoa
from xoa import coords as xcoords
from xoa import plot as xplot

pytest.importorskip("cartopy")

LON_ATTRS = {"standard_name": "longitude", "units": "degrees_east"}
LAT_ATTRS = {"standard_name": "latitude", "units": "degrees_north"}


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def get_temp(nx=11, ny=9):
    """A temperature field on a regular lon/lat grid"""
    lon = xr.DataArray(np.linspace(10.0, 15.0, nx), dims="lon", attrs=LON_ATTRS)
    lat = xr.DataArray(np.linspace(40.0, 44.0, ny), dims="lat", attrs=LAT_ATTRS)
    temp = (lon + lat).transpose("lat", "lon").rename("temp")
    return temp.assign_coords(lon=lon, lat=lat)


def get_ds():
    """A dataset with a temperature, a bathymetry and a mask"""
    temp = get_temp()
    bathy = (100.0 + 50.0 * (temp.lon - 10.0)).rename("bathy")
    bathy = bathy.broadcast_like(temp)
    mask = xr.where(temp.lon < 14.0, 1, 0).broadcast_like(temp).rename("mask")
    return xr.Dataset({"temp": temp, "bathy": bathy, "mask": mask})


def get_section(positive="down", depth2d=False):
    """A section with a vertical and a horizontal dimension"""
    lon = xr.DataArray(np.linspace(10.0, 12.0, 6), dims="x", attrs=LON_ATTRS)
    lat = xr.DataArray(np.linspace(40.0, 41.0, 6), dims="x", attrs=LAT_ATTRS)
    z = np.array([0.0, 10.0, 30.0, 60.0])
    if depth2d:
        depth = xr.DataArray(
            z[:, None] * (1 + 0.1 * np.arange(6))[None], dims=("z", "x"), name="depth"
        )
    else:
        depth = xr.DataArray(z, dims="z", name="depth")
    depth.attrs.update(standard_name="depth", units="m", positive=positive)
    data = xr.DataArray(
        np.arange(24.0).reshape(4, 6),
        dims=("z", "x"),
        name="temp",
        attrs={"long_name": "temperature", "units": "degC"},
    )
    return data.assign_coords(lon=lon, lat=lat, depth=depth)


class TestLazyCartopy:
    def test_import_does_not_load_cartopy(self):
        code = "import sys, xoa.plot; assert 'cartopy' not in sys.modules"
        subprocess.run([sys.executable, "-W", "ignore", "-c", code], check=True)

    def test_missing_cartopy(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "cartopy", None)
        monkeypatch.setitem(sys.modules, "cartopy.crs", None)
        with pytest.raises(ImportError, match="cartopy is required"):
            xplot._import_cartopy_()


class TestGetLabel:
    def test_attrs(self):
        da = xr.DataArray([1.0], dims="x", attrs={"long_name": "my field", "units": "m"})
        assert xplot.get_label(da) == "My field [m]"
        assert xplot.get_label(da, units=False) == "My field"

    def test_fill_from_meta(self):
        label = xplot.get_label(xr.DataArray([1.0], dims="x", name="temp"))
        assert label.startswith("Sea water")
        assert "[" in label

    def test_fallback_to_name(self):
        assert xplot.get_label(xr.DataArray([1.0], dims="x", name="unknown_xyz")) == "Unknown_xyz"


class TestMapAxes:
    def test_create_base_map(self):
        fig, ax = xplot.create_base_map([10.0, 15.0, 40.0, 44.0], title="title", figsize=(4, 3))
        assert ax.get_title() == "title"
        assert tuple(fig.get_size_inches()) == (4, 3)
        extent = ax.get_extent(crs=xplot._import_cartopy_()[0].PlateCarree())
        np.testing.assert_allclose(extent, [10.0, 15.0, 40.0, 44.0], atol=1e-6)

    def test_setup_map_axes_options(self):
        ccrs, _ = xplot._import_cartopy_()

        def feature_artists(ax):
            return [c for c in ax.get_children() if type(c).__name__ == "FeatureArtist"]

        ax = plt.figure().add_subplot(111, projection=ccrs.PlateCarree())
        xplot.setup_map_axes(ax, [0, 10, 0, 10])
        assert len(ax.artists) == 1
        assert not feature_artists(ax)
        assert ax.spines["geo"].get_visible()

        ax = plt.figure().add_subplot(111, projection=ccrs.PlateCarree())
        xplot.setup_map_axes(ax, [0, 10, 0, 10], gridlines=False, land=True, bbox=False)
        assert len(ax.artists) == 0
        assert len(feature_artists(ax)) == 1
        assert not ax.spines["geo"].get_visible()

    def test_add_land(self):
        ccrs, _ = xplot._import_cartopy_()
        ax = plt.figure().add_subplot(111, projection=ccrs.PlateCarree())
        n0 = len(ax.get_children())
        xplot.add_land(ax, scale="110m", color="red")
        assert len(ax.get_children()) == n0 + 1


class TestPlotField:
    def test_basic(self):
        temp = get_temp()
        mappable = xplot.plot_field(temp, title="title")
        assert mappable.axes.get_title() == "title"
        assert mappable.colorbar.ax.get_ylabel().startswith("Sea water")

    @pytest.mark.parametrize("method", ["pcolormesh", "contourf", "contour"])
    def test_methods(self, method):
        assert xplot.plot_field(get_temp(), method=method, add_colorbar=False) is not None

    def test_colorbar_is_shrunk_by_default(self):
        cbar_height = xplot.plot_field(get_temp()).colorbar.ax.get_position().height
        full = xplot.plot_field(get_temp(), cbar_kwargs={"shrink": 1.0})
        np.testing.assert_allclose(cbar_height / full.colorbar.ax.get_position().height, 0.7)

    def test_colorbar_label_override(self):
        mappable = xplot.plot_field(get_temp(), cbar_kwargs={"label": "custom"})
        assert mappable.colorbar.ax.get_ylabel() == "custom"

    def test_margin(self):
        temp = get_temp()
        ccrs, _ = xplot._import_cartopy_()
        pcar = ccrs.PlateCarree()
        ext0 = xplot.plot_field(temp, add_colorbar=False).axes.get_extent(pcar)
        ext1 = xplot.plot_field(temp, add_colorbar=False, margin=0.2).axes.get_extent(pcar)
        assert ext1[0] < ext0[0] and ext1[1] > ext0[1]

    def test_own_axes(self):
        ccrs, _ = xplot._import_cartopy_()
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": ccrs.Mercator()})
        for ax in axes:
            mappable = xplot.plot_field(
                get_temp(), ax=ax, add_colorbar=False, map_kw={"gridlines": False}
            )
            assert mappable.axes is ax

    def test_overlay_contours(self):
        ds = get_ds()
        ax = xplot.plot_field(ds.temp, add_colorbar=False).axes
        n0 = len(ax.collections)
        xplot.plot_field(ds.temp, ax=ax, add_colorbar=False, overlay_contours=True)
        assert len(ax.collections) > n0
        # Data array, meta names and style
        ax = xplot.plot_field(
            ds.temp,
            add_colorbar=False,
            overlay_contours=[
                {"field": ds.bathy, "levels": [150.0]},
                {"field": "bathy", "levels": [200.0], "colors": "0.5"},
                {"field": "mask", "levels": [0.5]},
            ],
            ds=ds,
        ).axes
        assert len(ax.collections) >= 4

    def test_overlay_errors(self):
        ds = get_ds()
        with pytest.raises(xoa.XoaError, match="ds parameter"):
            xplot.plot_field(ds.temp, overlay_contours=[{"field": "bathy"}])
        with pytest.raises(xoa.XoaError, match="not found"):
            xplot.plot_field(ds.temp, overlay_contours=[{"field": "ssh"}], ds=ds.drop_vars("bathy"))


def get_edge_segments(ax):
    """The segments of the edges drawn by plot_grid"""
    lines = [c for c in ax.collections if isinstance(c, mcollections.LineCollection)]
    return lines[0].get_segments() if lines else []


def get_center_points(ax):
    """The points of the centers drawn by plot_grid"""
    for line in ax.lines:
        if line.get_marker() == ".":
            return np.column_stack(line.get_data())
    return np.empty((0, 2))


class TestPlotGrid:
    def test_edges_and_centers_by_default(self):
        temp = get_temp(nx=11, ny=9)
        ax = xplot.plot_grid(temp)
        segments = get_edge_segments(ax)
        # One line per edge along each direction
        assert len(segments) == (9 + 1) + (11 + 1)
        centers = get_center_points(ax)
        assert len(centers) == 9 * 11
        lons, lats = np.meshgrid(temp.lon.values, temp.lat.values)
        np.testing.assert_allclose(np.sort(centers[:, 0]), np.sort(lons.ravel()))
        # Edges are in the middle of the centers and extrapolated at the ends
        np.testing.assert_allclose(min(s[:, 0].min() for s in segments), 10.0 - 0.25)

    def test_explicit_stride_keeps_last_edges(self):
        ax = xplot.plot_grid(get_temp(nx=11, ny=9), stride=4)
        # Edge indices 0, 4, 8 and 9 along y, and 0, 4, 8 and 11 along x
        assert len(get_edge_segments(ax)) == 4 + 4
        # Centers of the 3 x 3 coarse cells
        centers = get_center_points(ax)
        assert len(centers) == 3 * 3

    def test_coarse_centers_are_inside_coarse_cells(self):
        temp = get_temp(nx=11, ny=9)
        centers = get_center_points(xplot.plot_grid(temp, stride=4))
        assert np.all((centers[:, 0] > 9.75) & (centers[:, 0] < 15.25))
        assert np.all((centers[:, 1] > 39.75) & (centers[:, 1] < 44.25))

    def test_stride_tuple(self):
        ax = xplot.plot_grid(get_temp(nx=11, ny=9), stride=(2, 3))
        # y edges 0, 2, 4, 6, 8, 9 and x edges 0, 3, 6, 9, 11
        assert len(get_edge_segments(ax)) == 6 + 5

    @pytest.mark.parametrize("stride", [0, -1, "x", (1,), (1, 0), 1.5j])
    def test_invalid_stride(self, stride):
        with pytest.raises(xoa.XoaError, match="stride"):
            xplot.plot_grid(get_temp(), stride=stride)

    def test_edges_and_centers_can_be_disabled(self):
        ax = xplot.plot_grid(get_temp(), centers=False)
        assert len(get_edge_segments(ax)) > 0
        assert len(get_center_points(ax)) == 0
        ax = xplot.plot_grid(get_temp(), edges=False)
        assert len(get_edge_segments(ax)) == 0
        assert len(get_center_points(ax)) > 0

    def test_style(self):
        ax = xplot.plot_grid(
            get_temp(), edge_kw={"colors": "red", "linewidths": 2}, center_kw={"markersize": 6}
        )
        lines = [c for c in ax.collections if isinstance(c, mcollections.LineCollection)]
        np.testing.assert_allclose(lines[0].get_linewidths(), 2)
        assert [line for line in ax.lines if line.get_marker() == "."][0].get_markersize() == 6

    def test_auto_stride_adapts_to_the_grid_and_axes(self):
        temp = get_temp(nx=300, ny=200)
        nseg = len(get_edge_segments(xplot.plot_grid(temp)))
        assert nseg < 200 + 300  # under-sampled
        assert nseg > 10
        # Smaller axes need a larger stride
        small = len(get_edge_segments(xplot.plot_grid(temp, map_kw={"figsize": (3, 3)})))
        large = len(get_edge_segments(xplot.plot_grid(temp, map_kw={"figsize": (14, 14)})))
        assert small < nseg < large
        # And so does a larger minimal spacing
        assert len(get_edge_segments(xplot.plot_grid(temp, min_spacing=40))) < nseg
        # A small grid is fully drawn
        assert len(get_edge_segments(xplot.plot_grid(get_temp(nx=6, ny=5)))) == 6 + 7

    def test_auto_stride_with_rotated_grid(self):
        from xoa.core.grid import create_rotated_grid

        grid = create_rotated_grid(200, 100, 18.5, -36.0, 25.0, 4.0, 2.5)
        ds = xr.Dataset(
            {
                "lon": (("y", "x"), grid["lon"], LON_ATTRS),
                "lat": (("y", "x"), grid["lat"], LAT_ATTRS),
            }
        )
        ax = xplot.plot_grid(ds)
        nseg = len(get_edge_segments(ax))
        assert 10 < nseg < 101 + 201
        assert len(get_center_points(ax)) > 0

    def test_bathy(self):
        ds = get_ds()
        ax = xplot.plot_grid(ds, kind="bathy", add_colorbar=False)
        assert len(get_edge_segments(ax)) == 0
        assert len(get_center_points(ax)) == 0
        assert len(ax.collections) >= 1

    def test_bathy_with_edges(self):
        ax = xplot.plot_grid(get_ds(), kind="bathy", edges=True, add_colorbar=False)
        assert len(get_edge_segments(ax)) > 0

    def test_bathy_fallback_to_mesh(self):
        with pytest.warns(xoa.XoaWarning, match="No bathymetry"):
            ax = xplot.plot_grid(get_temp(), kind="bathy")
        assert len(get_edge_segments(ax)) > 0

    def test_resolution(self):
        ax = xplot.plot_grid(get_temp(), kind="resolution")
        values = np.asarray(ax.collections[0].get_array())
        assert np.all(values > 0)
        assert ax.figure.axes[-1].get_ylabel() == "Grid resolution [km]"
        assert len(get_edge_segments(ax)) == 0

    def test_strips(self):
        ax = xplot.plot_grid(get_temp(), strips=["north", "west"], n_cells=2, centers=False)
        assert len(ax.lines) == 2

    def test_invalid_kind(self):
        with pytest.raises(xoa.XoaError, match="Invalid kind"):
            xplot.plot_grid(get_temp(), kind="unknown")


class TestPlotSection:
    def test_depth_down_is_inverted(self):
        fig, ax = plt.subplots()
        xplot.plot_section(get_section("down"), ax=ax)
        assert ax.yaxis_inverted()
        assert ax.get_xlabel().startswith("Longitude")
        assert ax.get_ylabel().startswith("Depth")
        assert fig.axes[-1].get_ylabel() == "Temperature [degC]"

    def test_colorbar_is_shrunk_by_default(self):
        fig, ax = plt.subplots()
        xplot.plot_section(get_section(), ax=ax)
        cbar_height = fig.axes[-1].get_position().height
        fig, ax = plt.subplots()
        xplot.plot_section(get_section(), ax=ax, cbar_kwargs={"shrink": 1.0})
        np.testing.assert_allclose(cbar_height / fig.axes[-1].get_position().height, 0.7)

    def test_depth_up_is_not_inverted(self):
        fig, ax = plt.subplots()
        xplot.plot_section(get_section("up"), ax=ax)
        assert not ax.yaxis_inverted()

    def test_z_up_is_not_inverted(self):
        da = get_section("up")
        z = xcoords.to_z(da.depth)
        z.attrs["standard_name"] = "altitude"
        da = da.drop_vars("depth").assign_coords(z=z)
        fig, ax = plt.subplots()
        xplot.plot_section(da, ax=ax, add_colorbar=False)
        assert not ax.yaxis_inverted()
        assert ax.get_ylabel().startswith("Height")

    @pytest.mark.parametrize("method", ["pcolormesh", "contourf", "contour"])
    def test_methods_with_2d_depth(self, method):
        fig, ax = plt.subplots()
        xplot.plot_section(get_section(depth2d=True), ax=ax, method=method, add_colorbar=False)
        assert len(ax.collections) >= 1

    def test_horizontal_axis(self):
        da = get_section()
        for x, label in (("lon", "Longitude"), ("lat", "Latitude"), ("distance", "Distance")):
            fig, ax = plt.subplots()
            xplot.plot_section(da, x=x, ax=ax, add_colorbar=False)
            assert ax.get_xlabel().startswith(label)
        # Largest extent by default: here longitude, as 2 degrees against 1
        fig, ax = plt.subplots()
        xplot.plot_section(da, ax=ax, add_colorbar=False)
        assert ax.get_xlabel().startswith("Longitude")

    def test_distance_values(self):
        fig, ax = plt.subplots()
        mappable = xplot.plot_section(get_section(), x="distance", ax=ax, add_colorbar=False)
        coords = mappable.get_coordinates()
        assert coords[0, 0, 0] < coords[0, -1, 0]
        assert coords[0, -1, 0] > 100.0  # km

    def test_array_axis(self):
        da = get_section()
        xaxis = xr.DataArray(np.arange(6.0), dims="x", name="index")
        fig, ax = plt.subplots()
        xplot.plot_section(da, x=xaxis, ax=ax, add_colorbar=False)
        assert ax.get_xlabel() == "Index"

    def test_title_and_no_colorbar(self):
        fig, ax = plt.subplots()
        xplot.plot_section(get_section(), ax=ax, title="title", add_colorbar=False)
        assert ax.get_title() == "title"
        assert len(fig.axes) == 1

    def test_errors(self):
        da = get_section()
        with pytest.raises(xoa.XoaError, match="single horizontal"):
            xplot.plot_section(da.expand_dims(y=2))
        with pytest.raises(xoa.XoaError, match="x must be"):
            xplot.plot_section(da, x="unknown")


class TestPlotStick:
    @staticmethod
    def get_uv(n=5):
        time = np.arange("2020-01-01", n, dtype="datetime64[h]").astype("datetime64[ns]")
        coords = {"time": ("time", time, {"standard_name": "time"})}
        u = xr.DataArray(
            np.linspace(-0.5, 0.5, n),
            dims="time",
            coords=coords,
            attrs={"standard_name": "eastward_sea_water_velocity"},
            name="uu",
        )
        v = xr.DataArray(
            np.linspace(0.5, -0.5, n),
            dims="time",
            coords=coords,
            attrs={"standard_name": "northward_sea_water_velocity"},
            name="vv",
        )
        return u, v

    def test_arrays(self):
        u, v = self.get_uv()
        quiver = xplot.plot_stick(u, v, scale=2.0)
        assert len(quiver.U) == 5
        assert quiver.axes.get_xlabel() == "time"
        assert not quiver.axes.get_yaxis().get_visible()

    def test_dataset_found_with_meta(self):
        u, v = self.get_uv()
        quiver = xplot.plot_stick(xr.Dataset({"uu": u, "vv": v}))
        np.testing.assert_allclose(quiver.U, u.values)
        np.testing.assert_allclose(quiver.V, v.values)

    def test_without_time(self):
        u, v = self.get_uv()
        quiver = xplot.plot_stick(u.drop_vars("time"), v.drop_vars("time"))
        assert len(quiver.U) == 5

    def test_own_axes(self):
        u, v = self.get_uv()
        fig, ax = plt.subplots()
        assert xplot.plot_stick(u, v, ax=ax).axes is ax

    def test_errors(self):
        u, v = self.get_uv()
        with pytest.raises(xoa.XoaError, match="Cannot find"):
            xplot.plot_stick(xr.Dataset({"x": u}))
        with pytest.raises(xoa.XoaError, match="single dimension"):
            xplot.plot_stick(u.expand_dims(y=2), v.expand_dims(y=2))


class TestPlotTs:
    @staticmethod
    def get_ts():
        temp = xr.DataArray(
            np.linspace(10, 15, 20),
            dims="x",
            name="temp",
            attrs={
                "long_name": "temperature",
                "units": "degC",
                "standard_name": "sea_water_potential_temperature",
            },
        )
        sal = xr.DataArray(
            np.linspace(35, 36, 20),
            dims="x",
            name="sal",
            attrs={"long_name": "salinity", "standard_name": "sea_water_practical_salinity"},
        )
        depth = xr.DataArray(
            np.linspace(0, 100, 20),
            dims="x",
            name="depth",
            attrs={"long_name": "depth", "units": "m"},
        )
        return temp, sal, depth

    def test_labels_and_colorbar(self):
        temp, sal, depth = self.get_ts()
        out = xplot.plot_ts(temp, sal, dens=False, scatter_c=depth)
        assert out["axes"].get_xlabel() == "Salinity"
        assert out["axes"].get_ylabel() == "Temperature [degC]"
        assert out["colorbar"].ax.get_ylabel() == "Depth [m]"

    def test_colorbar_is_shrunk_by_default(self):
        temp, sal, depth = self.get_ts()
        shrunk = xplot.plot_ts(temp, sal, dens=False, scatter_c=depth)["colorbar"]
        full = xplot.plot_ts(temp, sal, dens=False, scatter_c=depth, colorbar_shrink=1.0)[
            "colorbar"
        ]
        np.testing.assert_allclose(
            shrunk.ax.get_position().height / full.ax.get_position().height, 0.7
        )

    def test_colorbar_label_override(self):
        temp, sal, depth = self.get_ts()
        out = xplot.plot_ts(temp, sal, dens=False, scatter_c=depth, colorbar_label="custom")
        assert out["colorbar"].ax.get_ylabel() == "custom"
        out = xplot.plot_ts(
            temp, sal, dens=False, scatter_c=depth, colorbar_kwargs={"label": "other"}
        )
        assert out["colorbar"].ax.get_ylabel() == "other"

    def test_no_colorbar_without_color_array(self):
        temp, sal, depth = self.get_ts()
        assert "colorbar" not in xplot.plot_ts(temp, sal, dens=False)
        out = xplot.plot_ts(temp, sal, dens=False, scatter_c=depth.values)
        assert "colorbar" not in out
        out = xplot.plot_ts(temp, sal, dens=False, scatter_c=depth.values, colorbar=True)
        assert out["colorbar"].ax.get_ylabel() == ""


class TestAddColorbar:
    def test_label_from_the_array(self):
        temp = get_temp()
        mappable = xplot.plot_field(temp, add_colorbar=False)
        cbar = xplot.add_colorbar(mappable, da=temp)
        assert cbar.ax.get_ylabel() == xplot.get_label(temp)

    def test_label_override_and_no_array(self):
        temp = get_temp()
        mappable = xplot.plot_field(temp, add_colorbar=False)
        assert xplot.add_colorbar(mappable, da=temp, label="custom").ax.get_ylabel() == "custom"
        assert xplot.add_colorbar(mappable).ax.get_ylabel() == ""

    def test_shared_colorbar_on_several_axes(self):
        ccrs, _ = xplot._import_cartopy_()
        fig, axes = plt.subplots(1, 2, subplot_kw={"projection": ccrs.Mercator()})
        for ax in axes:
            mappable = xplot.plot_field(get_temp(), ax=ax, add_colorbar=False)
        n = len(fig.axes)
        cbar = xplot.add_colorbar(mappable, axes, da=get_temp())
        assert len(fig.axes) == n + 1
        assert cbar.ax.get_ylabel().startswith("Sea water")

    def test_same_defaults_as_the_plot_functions(self):
        temp = get_temp()
        via_field = xplot.plot_field(temp).colorbar
        mappable = xplot.plot_field(temp, add_colorbar=False)
        via_helper = xplot.add_colorbar(mappable, da=temp)
        assert via_field.ax.get_ylabel() == via_helper.ax.get_ylabel()
        np.testing.assert_allclose(
            via_field.ax.get_position().height, via_helper.ax.get_position().height
        )


class TestPlotTaylor:
    @staticmethod
    def get_data(nm=3):
        rng = np.random.default_rng(1)
        ref = xr.DataArray(rng.normal(size=(40, 5)), dims=("time", "x"))
        mod = xr.concat(
            [0.7 * ref + 0.3 * rng.normal(size=ref.shape), -ref, 2 * ref][:nm],
            dim=xr.DataArray(list("abc")[:nm], dims="model", name="model"),
        )
        return mod, ref

    def test_single_point_by_default(self):
        mod, ref = self.get_data()
        diagram = xplot.plot_taylor(mod.isel(model=0), ref)
        assert len(diagram.markers) == 1
        assert diagram.legend is None
        assert np.isclose(diagram.ref_std, float(ref.std()))

    def test_points_from_remaining_dims(self):
        mod, ref = self.get_data()
        diagram = xplot.plot_taylor(mod, ref, dim=("time", "x"), normalize=True)
        assert diagram.negative
        assert len(diagram.markers) == 3
        assert [t.get_text() for t in diagram.legend.get_texts()][1:] == ["a", "b", "c"]
        theta, std = diagram.markers[2].get_data()
        assert np.isclose(std[0], 2.0) and np.isclose(theta[0], 0)
        assert diagram.ref_std == 1.0

    def test_not_normalized_needs_common_reference(self):
        mod, ref = self.get_data()
        ref = ref * xr.DataArray([1.0, 2.0], dims="model", coords={"model": ["a", "b"]})
        mod = mod.sel(model=["a", "b"])
        with pytest.raises(xoa.exceptions.XoaError):
            xplot.plot_taylor(mod, ref, dim=("time", "x"))
        xplot.plot_taylor(mod, ref, dim=("time", "x"), normalize=True)

    def test_dataset_and_values(self):
        mod, ref = self.get_data(2)
        ds = xr.Dataset({"u": mod.isel(model=0, drop=True), "v": mod.isel(model=1, drop=True)})
        diagram = xplot.plot_taylor(
            ds, ref, values=xr.DataArray([1.0, 2.0], attrs={"long_name": "Depth"})
        )
        assert len(diagram.markers) == 1
        assert diagram.colorbar.ax.get_ylabel() == "Depth"

    def test_dataset_reference_by_name(self):
        mod, ref = self.get_data(1)
        ds = xr.Dataset({"obs": ref, "mod": mod.isel(model=0, drop=True)})
        diagram = xplot.plot_taylor(ds, "obs")
        assert len(diagram.markers) == 1

    def test_errors(self):
        mod, ref = self.get_data()
        with pytest.raises(xoa.exceptions.XoaError):
            xplot.plot_taylor(mod, ref, dim="lon")
