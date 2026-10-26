"""
Test the map, grid, section and stick helpers of the :mod:`xoa.plot` module
"""

import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.quiver as mquiver
import numpy as np
import pytest
import xarray as xr

import xoa
from xoa import coords as xcoords
from xoa import plot as xplot
from xoa.core import plot as cplot

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
            xplot.import_cartopy()


class TestGetLabel:
    def test_attrs(self):
        da = xr.DataArray([1.0], dims="x", attrs={"long_name": "my field", "units": "m"})
        assert xplot.get_label(da) == "My field [m]"
        assert xplot.get_label(da, units=False) == "My field"

    @pytest.mark.parametrize(
        "units, expected",
        [
            ("degrees_celsius", "°C"),
            ("Celsius", "°C"),
            ("degC", "°C"),
            ("1e-3", "psu"),
            ("m s-1", "m s⁻¹"),
            ("m2 s-2", "m² s⁻²"),
            ("kg m-3", "kg m⁻³"),
            ("m", "m"),
        ],
    )
    def test_pretty_units(self, units, expected):
        da = xr.DataArray([1.0], dims="x", name="a", attrs={"long_name": "a", "units": units})
        assert xplot.get_label(da) == f"A [{expected}]"
        assert xplot.get_label(da, pretty=False) == f"A [{units}]"

    def test_dimensionless_units_are_skipped(self):
        da = xr.DataArray([1.0], dims="x", name="a", attrs={"long_name": "Rossby", "units": "1"})
        assert xplot.get_label(da) == "Rossby"
        assert xplot.get_label(da, pretty=False) == "Rossby [1]"

    def test_fill_from_meta(self):
        label = xplot.get_label(xr.DataArray([1.0], dims="x", name="temp"))
        assert label.startswith("Sea water")
        assert "[" in label

    def test_fallback_to_name(self):
        assert xplot.get_label(xr.DataArray([1.0], dims="x", name="unknown_xyz")) == "Unknown_xyz"


class TestMapAxes:
    def test_create_map(self):
        fig, ax = xplot.create_map([10.0, 15.0, 40.0, 44.0], title="title", figsize=(4, 3))
        assert ax.get_title() == "title"
        assert tuple(fig.get_size_inches()) == (4, 3)
        extent = ax.get_extent(crs=xplot.import_cartopy()[0].PlateCarree())
        np.testing.assert_allclose(extent, [10.0, 15.0, 40.0, 44.0], atol=1e-6)

    def test_decorate_map_options(self):
        ccrs, _ = xplot.import_cartopy()

        def feature_artists(ax):
            return [c for c in ax.get_children() if type(c).__name__ == "FeatureArtist"]

        ax = plt.figure().add_subplot(111, projection=ccrs.PlateCarree())
        xplot.decorate_map(ax, [0, 10, 0, 10])
        assert len(ax.artists) == 1
        assert not feature_artists(ax)
        assert ax.spines["geo"].get_visible()

        ax = plt.figure().add_subplot(111, projection=ccrs.PlateCarree())
        xplot.decorate_map(ax, [0, 10, 0, 10], gridlines=False, land=True, bbox=False)
        assert len(ax.artists) == 0
        assert len(feature_artists(ax)) == 1
        assert not ax.spines["geo"].get_visible()

    def test_add_land(self):
        ccrs, _ = xplot.import_cartopy()
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

    def test_scalebar_and_compass(self):
        mappable = xplot.plot_field(get_temp(), scalebar=True, compass={"style": "fancy"})
        texts = [t.get_text() for t in mappable.axes.texts]
        assert "N" in texts and any(t.endswith(" km") for t in texts)
        texts = [t.get_text() for t in xplot.plot_field(get_temp()).axes.texts]
        assert "N" not in texts

    def test_colorbar_label_override(self):
        mappable = xplot.plot_field(get_temp(), cbar_kwargs={"label": "custom"})
        assert mappable.colorbar.ax.get_ylabel() == "custom"

    def test_margin(self):
        temp = get_temp()
        ccrs, _ = xplot.import_cartopy()
        pcar = ccrs.PlateCarree()
        ext0 = xplot.plot_field(temp, add_colorbar=False).axes.get_extent(pcar)
        ext1 = xplot.plot_field(temp, add_colorbar=False, margin=0.2).axes.get_extent(pcar)
        assert ext1[0] < ext0[0] and ext1[1] > ext0[1]

    def test_own_axes(self):
        ccrs, _ = xplot.import_cartopy()
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
    def test_projection(self):
        assert type(xplot.plot_grid(get_temp()).projection).__name__ == "Orthographic"
        ax = xplot.plot_grid(get_temp(), projection="merc", map_kw={"projection": "pc"})
        assert type(ax.projection).__name__ == "Mercator"
        ax = xplot.plot_grid(get_temp(), map_kw={"projection": "pc"})
        assert type(ax.projection).__name__ == "PlateCarree"

    def test_figsize_follows_the_aspect_ratio(self):
        def get_size(dlon, dlat):
            lon = xr.DataArray(
                np.linspace(0, dlon, 5), dims="lon", attrs={"standard_name": "longitude"}
            )
            lat = xr.DataArray(
                np.linspace(0, dlat, 5), dims="lat", attrs={"standard_name": "latitude"}
            )
            return xplot.plot_grid(
                xr.Dataset(coords={"lon": lon, "lat": lat})
            ).figure.get_size_inches()

        assert get_size(20, 5)[1] < get_size(5, 20)[1]
        ax = xplot.plot_grid(get_temp(), map_kw={"figsize": (5, 5)})
        assert tuple(ax.figure.get_size_inches()) == (5, 5)

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
        assert fig.axes[-1].get_ylabel() == "Temperature [°C]"

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

    def test_key(self):
        u, v = self.get_uv()
        quiver = xplot.plot_stick(u, v)
        keys = [c for c in quiver.axes.get_children() if isinstance(c, mquiver.QuiverKey)]
        assert len(keys) == 1
        quiver = xplot.plot_stick(u, v, key=False)
        assert not [c for c in quiver.axes.get_children() if isinstance(c, mquiver.QuiverKey)]

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


class TestPlotVectors:
    @staticmethod
    def get_uv(nx=11, ny=9):
        temp = get_temp(nx, ny)
        u = (0 * temp + 0.5).rename("u")
        v = (0.1 * (temp.lon - 12.5)).broadcast_like(temp).transpose("lat", "lon").rename("v")
        u.attrs["units"] = v.attrs["units"] = "m s-1"
        return u, v

    def test_default(self):
        u, v = self.get_uv()
        quiver = xplot.plot_vectors(u, v, stride=1)
        assert quiver.U.size == u.size
        # Reference arrow
        keys = [c for c in quiver.axes.get_children() if isinstance(c, mquiver.QuiverKey)]
        assert len(keys) == 1
        assert "m s-1" in keys[0].text.get_text()

    def test_stride_and_own_axes(self):
        u, v = self.get_uv()
        mappable = xplot.plot_field(u)
        quiver = xplot.plot_vectors(u, v, ax=mappable.axes, stride=(2, 5))
        assert quiver.axes is mappable.axes
        assert quiver.U.size == len(range(0, 9, 2)) * len(range(0, 11, 5))
        auto = xplot.plot_vectors(u, v, ax=mappable.axes)
        assert 0 < auto.U.size <= u.size

    def test_transposed_dims(self):
        u, v = self.get_uv()
        quiver = xplot.plot_vectors(u.transpose("lon", "lat"), v.transpose("lon", "lat"), stride=1)
        assert quiver.U.size == u.size

    def test_color_by_speed(self):
        u, v = self.get_uv()
        quiver = xplot.plot_vectors(u, v, color="speed", stride=1, cmap="magma")
        np.testing.assert_allclose(quiver.get_array(), np.hypot(u, v).values.ravel())
        assert quiver.cmap.name == "magma"
        fig = quiver.axes.figure
        assert len(fig.axes) == 2  # With a colorbar
        quiver = xplot.plot_vectors(u, v, color="speed", add_colorbar=False, key=False)
        assert len(quiver.axes.figure.axes) == 1

    def test_ellipses(self):
        u, v = self.get_uv()
        std = (0 * u + 0.1).rename("ustd")
        quiver = xplot.plot_vectors(u, v, stride=1, ustd=std, vstd=std / 2, uvcorr=0 * std)
        assert quiver.ellipses.axes is quiver.axes
        assert len(quiver.ellipses.get_widths()) == u.size
        quiver.axes.figure.canvas.draw()
        assert xplot.plot_vectors(u, v, stride=1).__dict__.get("ellipses") is None

    def test_ellipses_take_the_color_of_the_arrows(self):
        u, v = self.get_uv()
        std = 0 * u + 0.1
        ellipses = xplot.plot_vectors(u, v, color="tab:red", ustd=std).ellipses
        np.testing.assert_allclose(ellipses.get_facecolor()[0][:3], mcolors.to_rgb("tab:red"))
        assert ellipses.get_facecolor()[0][3] < 1
        np.testing.assert_allclose(ellipses.get_edgecolor()[0], mcolors.to_rgba("tab:red"))
        ellipses = xplot.plot_vectors(
            u, v, color="tab:red", ustd=std, ellipse_kwargs={"facecolor": "none"}
        ).ellipses
        assert ellipses.get_facecolor().shape[0] == 0 or ellipses.get_facecolor()[0][3] == 0
        ellipses = xplot.plot_vectors(u, v, color="speed", ustd=std).ellipses
        np.testing.assert_allclose(ellipses.get_edgecolor()[0][:3], 0)

    def test_ellipses_with_stride_and_transposed_dims(self):
        u, v = self.get_uv()
        std = (0 * u + 0.1).rename("ustd")
        quiver = xplot.plot_vectors(
            u,
            v,
            stride=(2, 3),
            ustd=std.transpose("lon", "lat"),
            nsigma=1,
            ellipse_kwargs={"label": "1 sigma"},
        )
        assert len(quiver.ellipses.get_widths()) == quiver.U.size
        assert quiver.ellipses.get_label() == "1 sigma"

    def test_ellipses_rotated_by_the_projection(self):
        ccrs = pytest.importorskip("cartopy.crs")
        u, v = self.get_uv()
        std = 0 * u + 0.1
        ax = xplot.get_map_axes(None, (10, 20, 40, 60), ccrs.PlateCarree(), None)
        quiver = xplot.plot_vectors(u, v, ax=ax, stride=1, ustd=std, vstd=std / 2)
        np.testing.assert_allclose(quiver.ellipses.get_angles(), 0, atol=1e-6)
        ax = xplot.get_map_axes(None, (10, 20, 40, 60), ccrs.Mercator(), None)
        quiver = xplot.plot_vectors(u, v, ax=ax, stride=1, ustd=std, vstd=std / 2)
        assert quiver.ellipses.get_angles().shape == (u.size,)

    def test_no_key_and_title(self):
        u, v = self.get_uv()
        quiver = xplot.plot_vectors(u, v, key=False, title="Currents", color="tab:red")
        keys = [c for c in quiver.axes.get_children() if isinstance(c, mquiver.QuiverKey)]
        assert not keys
        assert quiver.axes.get_title() == "Currents"

    def test_scale_can_be_forced(self):
        u, v = self.get_uv()
        assert xplot.plot_vectors(u, v, scale=3.0).scale == 3.0
        assert xplot.plot_vectors(u, v).scale > 0

    def test_key_value(self):
        assert cplot.get_quiverkey_value(np.array([0.9, 0.2])) == 1.0
        assert cplot.get_quiverkey_value(np.array([0.31, np.nan])) == 0.2
        assert cplot.get_quiverkey_value(np.array([4.0, 0.0])) == 5.0
        assert cplot.get_quiverkey_value(np.arange(1, 101), percentile=50) == 5e1
        assert cplot.get_quiverkey_value(np.zeros(3)) == 1.0

    def test_key_above_left(self):
        u, v = self.get_uv()
        quiver = xplot.plot_vectors(u, v)
        key = [c for c in quiver.axes.get_children() if isinstance(c, mquiver.QuiverKey)][0]
        assert key.Y == 1.03 and 0 < key.X < 0.5  # The arrow starts at the left of the axes


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
        assert out["axes"].get_ylabel() == "Temperature [°C]"
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
        ccrs, _ = xplot.import_cartopy()
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


class TestAutoCmap:
    @pytest.mark.parametrize(
        "offset, option, symmetric",
        [
            (-55.0, "plot.cmapdiv", True),
            (0.0, "plot.cmappos", False),
            (-70.0, "plot.cmapneg", False),
        ],
    )
    def test_cmap_follows_the_sign(self, offset, option, symmetric):
        temp = get_temp() + offset  # From 50 to 59 without offset
        mappable = xplot.plot_field(temp, cmap="auto")
        assert mappable.get_cmap().name == xoa.get_option(option)
        vmin, vmax = mappable.get_clim()
        assert (vmin == -vmax) == symmetric

    def test_option_is_used(self):
        with xoa.set_options("plot", cmapdiv="cmo.delta"):
            mappable = xplot.plot_field(get_temp() - 55.0, cmap="auto")
        assert mappable.get_cmap().name == "cmo.delta"

    def test_explicit_limits_and_levels_are_kept(self):
        temp = get_temp() - 55.0
        mappable = xplot.plot_field(temp, cmap="auto", vmin=-100, vmax=10)
        assert mappable.get_clim() == (-100, 10)
        mappable = xplot.plot_field(temp, cmap="auto", method="contourf", levels=[-20, 0, 20])
        assert list(mappable.levels) == [-20, 0, 20]

    def test_missing_values_only(self):
        temp = get_temp() * np.nan
        assert xplot.plot_field(temp, cmap="auto").get_cmap().name == "cmo.amp"


class TestPlotFields:
    def get_fields(self, n=3):
        temp = get_temp()
        return {f"t{i}": temp + i for i in range(n)}

    def test_dict(self):
        fig, axes = xplot.plot_fields(self.get_fields(4), ncols=2)
        assert len(axes) == 4
        assert [ax.get_title() for ax in axes] == ["t0", "t1", "t2", "t3"]
        assert len(fig.axes) == 8  # One colorbar per map

    def test_shared_colorbar(self):
        fig, _ = xplot.plot_fields(self.get_fields(), colorbar="shared")
        assert len(fig.axes) == 4

    def test_no_colorbar_and_unused_axes(self):
        fig, axes = xplot.plot_fields(list(self.get_fields(3).values()), ncols=2, colorbar=None)
        assert len(axes) == 3 and len(fig.axes) == 3

    def test_per_field_kwargs(self):
        fields = [(f, {"vmax": 20 + i}) for i, f in enumerate(self.get_fields(2).values())]
        _, axes = xplot.plot_fields(fields, titles=["a", "b"], vmin=0)
        assert axes[1].get_title() == "b"

    def test_bad_colorbar(self):
        with pytest.raises(xoa.exceptions.XoaError):
            xplot.plot_fields(self.get_fields(), colorbar="bad")

    def test_panel_labels(self):
        _, axes = xplot.plot_fields(self.get_fields(3), colorbar=None, panel_labels=True)
        assert [ax.texts[0].get_text() for ax in axes] == ["a)", "b)", "c)"]
        _, axes = xplot.plot_fields(self.get_fields(2), colorbar=None, panel_labels="({A})")
        assert axes[1].texts[0].get_text() == "(B)"
        _, axes = xplot.plot_fields(
            self.get_fields(2), colorbar=None, panel_labels={"loc": "lower right"}
        )
        assert axes[0].texts[0].get_position() == (0.97, 0.03)


class TestPanelLabeler:
    def test_labels(self):
        labeler = xplot.PanelLabeler("{A}.")
        assert [labeler.get_label(i) for i in (0, 25, 26, 27, 52)] == [
            "A.",
            "Z.",
            "AA.",
            "AB.",
            "BA.",
        ]
        assert xplot.PanelLabeler("{n}").get_label(2) == "3"

    def test_counter_goes_on(self):
        _, axes = plt.subplots(1, 3)
        labeler = xplot.PanelLabeler(start=1, fontsize=8)
        texts = labeler(axes[:2]) + labeler(axes[2])
        assert [t.get_text() for t in texts] == ["b)", "c)", "d)"]
        assert texts[0].get_fontsize() == 8

    def test_add_panel_labels(self):
        _, ax = plt.subplots()
        assert xplot.add_panel_labels(ax, loc="upper right")[0].get_ha() == "right"

    def test_bad_loc(self):
        with pytest.raises(xoa.exceptions.XoaError):
            xplot.PanelLabeler(loc="middle")


class TestMinimap:
    @staticmethod
    def get_ds():
        return xr.Dataset(
            coords={"lon": ("st", [-7.0, -5.0, -5.0]), "lat": ("st", [44.0, 44.0, 46.0])}
        )

    @pytest.mark.parametrize("loc", ["upper right", "upper left", "lower right", "lower left"])
    def test_inset_is_inside_the_parent(self, loc):
        fig, parent = plt.subplots(figsize=(6, 4))
        ax = xplot.plot_minimap(self.get_ds(), parent=parent, loc=loc, size=0.25)
        pbox, box = parent.get_position(), ax.get_position(original=True)
        assert ax.figure is fig and len(fig.axes) == 2
        assert pbox.x0 <= box.x0 and box.x1 <= pbox.x1 and pbox.y0 <= box.y0 and box.y1 <= pbox.y1
        assert ("right" in loc) == (box.x0 > pbox.x0 + 0.5 * pbox.width)
        assert ("upper" in loc) == (box.y0 > pbox.y0 + 0.5 * pbox.height)
        # Square in pixels
        np.testing.assert_allclose(box.width * 6, box.height * 4)
        np.testing.assert_allclose(box.width * 6, 0.25 * pbox.width * 6)

    def test_default_position_and_explicit_ax(self):
        plt.figure()
        ax = xplot.plot_minimap(self.get_ds())
        assert ax.get_position(original=True).x0 == pytest.approx(0.9)
        ax2 = xplot.plot_minimap(self.get_ds(), ax=[0.1, 0.1, 0.2])
        assert ax2.get_position(original=True).x0 == pytest.approx(0.1)

    def test_2d_grid_is_drawn_as_its_outline(self):
        lon, lat = np.meshgrid(np.arange(-8.0, -4.0), np.arange(44.0, 47.0))
        ds = xr.Dataset(coords={"lon": (("y", "x"), lon), "lat": (("y", "x"), lat)})
        ax = xplot.plot_minimap(ds)
        (line,) = ax.lines
        x, y = line.get_data()
        assert len(x) == 2 * (3 + 4) - 4 + 1  # Each edge point once, and the closing one
        assert (x[0], y[0]) == (x[-1], y[-1]) == (-8.0, 44.0)
        assert set(zip(x, y)) == {
            (a, b) for a, b in zip(lon.ravel(), lat.ravel()) if a in (-8, -5) or b in (44, 46)
        }
        assert line.get_linewidth() == 1.5

    def test_invalid_loc(self):
        fig, parent = plt.subplots()
        with pytest.raises(xoa.XoaError, match="Invalid loc"):
            xplot.plot_minimap(self.get_ds(), parent=parent, loc="middle")

    def test_double_minimap_with_parent(self):
        fig, parent = plt.subplots(figsize=(6, 4))
        glob, regional = xplot.plot_double_minimap(
            self.get_ds(), parent=parent, loc="lower right", size=0.2, regional_ax="left"
        )
        gbox, rbox, pbox = (a.get_position(original=True) for a in (glob, regional, parent))
        assert gbox.x1 <= pbox.x1 and rbox.x1 <= gbox.x0 + 1e-9 and rbox.x0 >= pbox.x0


class TestRose:
    @staticmethod
    def get_ds(n=100):
        rng = np.random.default_rng(0)
        time = np.arange(n)
        return xr.Dataset(
            {
                "hs": ("time", rng.uniform(0, 6, n), {"units": "m"}),
                "direction": ("time", rng.uniform(0, 360, n), {"units": "degree"}),
            },
            coords={"time": time},
        )

    def test_dataset_variables_are_searched(self):
        rose = xplot.plot_rose(self.get_ds(), nsector=8)
        assert len(rose.bars[0]) == 8
        assert "Significant wave height" in rose.legend.get_title().get_text()

    def test_accessor_and_arrays(self):
        ds = self.get_ds()
        rose = ds.hs.xoa.plot.rose(ds.direction, edges=[0, 3])
        assert len(rose.bars) == 2

    def test_components_go_to(self):
        n = 400
        u = xr.DataArray(np.full(n, 3.0), dims="time", name="u", attrs={"units": "m s-1"})
        v = xr.DataArray(np.full(n, 4.0), dims="time", name="v", attrs={"units": "m s-1"})
        # Speed 5 m/s going to the north-east (atan2(3, 4) = 36.87 degrees)
        for rose in (
            xplot.plot_rose(u, v, nsector=36),
            xplot.plot_rose(u.rename("a"), v.rename("b"), cartesian=True, nsector=36),
            xplot.plot_rose(xr.Dataset({"uo": u.rename("uo"), "vo": v.rename("vo")}), nsector=36),
        ):
            assert "Speed" in rose.legend.get_title().get_text()
            heights = sum(np.array([p.get_height() for p in bar]) for bar in rose.bars)
            assert np.argmax(heights) == 4  # sector of 40 degrees
        rose = xplot.plot_rose(u, v, from_=True, nsector=36)
        heights = sum(np.array([p.get_height() for p in bar]) for bar in rose.bars)
        assert np.argmax(heights) == 22  # sector of 220 degrees

    def test_components_are_inferred(self):
        ds = self.get_ds()
        # a direction in degrees is not mistaken for a component, and an unknown array is a direction
        assert len(xplot.plot_rose(ds.hs, ds.direction, nsector=8).bars[0]) == 8
        assert xplot._is_cartesian_(ds.hs.rename("v")) and not xplot._is_cartesian_(ds.direction)
        assert not xplot._is_cartesian_(ds.hs.rename("foo"))

    def test_missing_variables(self):
        ds = self.get_ds()
        with pytest.raises(xoa.XoaError, match="direction is required"):
            xplot.plot_rose(ds.hs)
        with pytest.raises(xoa.XoaError, match="Cannot find the direction"):
            xplot.plot_rose(ds[["hs"]])
        with pytest.raises(xoa.XoaError, match="wind speed or the wave height"):
            xplot.plot_rose(xr.Dataset({"foo": ("time", [1.0])}))


class TestShapes:
    """Shapes with the arguments of the geographical selections"""

    def test_circle_units(self):
        ax = plt.figure().add_subplot(111)
        km = xplot.plot_circle((-3.5, 48, 70), units="km", ax=ax).get_xy()
        m = xplot.plot_circle((-3.5, 48, 70000), ax=ax).get_xy()
        np.testing.assert_allclose(km, m)

    def test_box_and_current_axes(self):
        ax = plt.figure().add_subplot(111)
        patch = xplot.plot_box(lon=xr.DataArray([-6, -4]), lat=(46, 48))
        assert patch.axes is ax

    def test_polygon(self):
        shapely = pytest.importorskip("shapely.geometry")
        ax = plt.figure().add_subplot(111)
        verts = [[-6, 46], [-4, 45.5], [-3.5, 47.5]]
        patch = xplot.plot_polygon(shapely.Polygon(verts), ax=ax)
        np.testing.assert_allclose(patch.get_xy()[:3], verts)
        with pytest.raises(xoa.XoaError, match="single polygon"):
            xplot.plot_polygon(shapely.MultiPolygon([shapely.Polygon(verts)] * 2), ax=ax)
