# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.plot` module
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.collections as mcollections
import matplotlib.pyplot as plt
import numpy as np
import pytest

import xoa
from xoa import plot as xplot
from xoa.core import plot as cplot
from xoa.core import grid as cgrid


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def get_axes(size=5.0, xlim=(0, 100), ylim=(0, 100)):
    """A plain axes that fills a square figure of ``size`` inches at 100 dpi"""
    fig = plt.figure(figsize=(size, size), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    return ax


def get_lonlat(nx, ny, x1=100.0, y1=100.0):
    return np.meshgrid(np.linspace(0, x1, nx), np.linspace(0, y1, ny))


def test_moved_helpers_are_still_available_from_xoa_plot():
    for name in "add_land", "create_base_map", "setup_map_axes", "_import_cartopy_":
        assert getattr(xplot, name) is getattr(cplot, name)


class TestStrides:
    def test_sampled_indices_keep_last(self):
        assert cplot.get_sampled_indices(10, 4) == [0, 4, 8, 9]
        assert cplot.get_sampled_indices(9, 4) == [0, 4, 8]
        assert cplot.get_sampled_indices(1, 3) == [0]

    def test_explicit(self):
        lon, lat = get_lonlat(5, 4)
        assert cplot.get_strides(3, lon, lat) == (3, 3)
        assert cplot.get_strides((2, 5), lon, lat) == (2, 5)
        assert cplot.get_strides([1, 4], lon, lat) == (1, 4)

    @pytest.mark.parametrize("stride", [0, -2, "x", (1,), (1, 0), (1, 2, 3)])
    def test_invalid(self, stride):
        lon, lat = get_lonlat(5, 4)
        with pytest.raises(xoa.XoaError, match="stride"):
            cplot.get_strides(stride, lon, lat)

    def test_auto_needs_axes(self):
        lon, lat = get_lonlat(5, 4)
        with pytest.raises(xoa.XoaError, match="axes"):
            cplot.get_strides("auto", lon, lat)

    def test_auto_follows_the_pixel_spacing(self):
        # 101 points over 100 data units drawn on 500 pixels: 5 pixels between points
        lon, lat = get_lonlat(101, 101)
        ax = get_axes()
        assert cplot.get_strides("auto", lon, lat, ax, min_spacing=12) == (3, 3)
        assert cplot.get_strides("auto", lon, lat, ax, min_spacing=5) == (1, 1)
        assert cplot.get_strides("auto", lon, lat, ax, min_spacing=26) == (6, 6)
        # Twice as many points means twice as small spacing
        lon, lat = get_lonlat(201, 201)
        assert cplot.get_strides("auto", lon, lat, ax, min_spacing=12) == (5, 5)
        # A zoom shows less cells
        ax = get_axes(xlim=(0, 50), ylim=(0, 50))
        assert cplot.get_strides("auto", *get_lonlat(101, 101), ax, min_spacing=12) == (2, 2)

    def test_auto_is_independent_for_each_direction(self):
        lon, lat = get_lonlat(101, 26)  # 5 pixels along x and 20 along y
        sy, sx = cplot.get_strides("auto", lon, lat, get_axes(), min_spacing=12)
        assert (sy, sx) == (1, 3)

    def test_auto_with_one_row(self):
        lon, lat = get_lonlat(101, 1)
        sy, sx = cplot.get_strides("auto", lon, lat, get_axes(), min_spacing=12)
        assert sy == 1


class TestGetMesh:
    def test_no_stride(self):
        lon, lat = get_lonlat(4, 3, 3.0, 2.0)
        segments, clon, clat = cplot.get_mesh(lon, lat)
        assert len(segments) == (3 + 1) + (4 + 1)
        np.testing.assert_array_equal(clon, lon)
        np.testing.assert_array_equal(clat, lat)
        # Edges are in the middle, and extrapolated at the ends
        np.testing.assert_allclose(segments[0][:, 0], np.arange(-0.5, 4.0))
        np.testing.assert_allclose(segments[0][:, 1], -0.5)
        assert all(seg.shape[1] == 2 for seg in segments)

    def test_stride(self):
        lon, lat = get_lonlat(4, 3, 3.0, 2.0)
        segments, clon, clat = cplot.get_mesh(lon, lat, (2, 2))
        # Edge indices 0, 2 and 3 along y, and 0, 2, 4 along x
        assert len(segments) == 3 + 3
        assert clon.shape == clat.shape == (2, 2)
        # First coarse cell goes from edge -0.5 to 1.5 along x and -0.5 to 1.5 along y
        np.testing.assert_allclose(clon[0, 0], 0.5)
        np.testing.assert_allclose(clat[0, 0], 0.5)

    def test_scalar_stride(self):
        lon, lat = get_lonlat(4, 3, 3.0, 2.0)
        assert len(cplot.get_mesh(lon, lat, 2)[0]) == len(cplot.get_mesh(lon, lat, (2, 2))[0])

    def test_curvilinear(self):
        lon, lat = get_lonlat(6, 5, 5.0, 4.0)
        lon, lat = lon + 0.1 * lat, lat + 0.1 * lon
        segments, clon, clat = cplot.get_mesh(lon, lat)
        assert len(segments) == 6 + 7
        assert clon.shape == (5, 6)


class TestPlotMesh:
    def test_edges_and_centers(self):
        ax = get_axes(xlim=(-1, 6), ylim=(-1, 6))
        lon, lat = get_lonlat(6, 5, 5.0, 4.0)
        collection, line = cplot.plot_mesh(ax, lon, lat)
        assert isinstance(collection, mcollections.LineCollection)
        assert len(collection.get_segments()) == (5 + 1) + (6 + 1)
        assert len(line.get_xdata()) == 30
        assert collection in ax.collections
        assert line in ax.lines

    def test_disable(self):
        ax = get_axes()
        lon, lat = get_lonlat(6, 5)
        collection, line = cplot.plot_mesh(ax, lon, lat, edges=False)
        assert collection is None and line is not None
        collection, line = cplot.plot_mesh(ax, lon, lat, centers=False)
        assert collection is not None and line is None

    def test_style(self):
        ax = get_axes()
        lon, lat = get_lonlat(6, 5)
        collection, line = cplot.plot_mesh(
            ax,
            lon,
            lat,
            edge_kw={"linewidths": 3},
            center_kw={"markersize": 7, "color": "k"},
        )
        np.testing.assert_allclose(collection.get_linewidths(), 3)
        assert line.get_markersize() == 7
        assert line.get_color() == "k"

    def test_auto_stride_reduces_the_number_of_lines(self):
        lon, lat = get_lonlat(101, 101)
        collection, line = cplot.plot_mesh(get_axes(), lon, lat)
        assert len(collection.get_segments()) == 2 * (len(range(0, 102, 3)) + 1)
        collection, line = cplot.plot_mesh(get_axes(), lon, lat, min_spacing=1)
        assert len(collection.get_segments()) == 2 * 102


class TestPlotDepthSection:
    @staticmethod
    def get_arrays():
        x, z = np.meshgrid(np.arange(6.0), np.array([0.0, 10.0, 30.0, 60.0]))
        return x, z, x + z

    @pytest.mark.parametrize("method", ["pcolormesh", "contourf", "contour"])
    def test_methods(self, method):
        fig, ax = plt.subplots()
        mappable = cplot.plot_depth_section(ax, *self.get_arrays(), method=method)
        assert mappable is not None
        assert len(ax.collections) >= 1

    def test_invert_yaxis(self):
        fig, ax = plt.subplots()
        cplot.plot_depth_section(ax, *self.get_arrays(), invert_yaxis=True)
        assert ax.yaxis_inverted()
        fig, ax = plt.subplots()
        cplot.plot_depth_section(ax, *self.get_arrays())
        assert not ax.yaxis_inverted()

    def test_pcolormesh_flat_shading(self):
        fig, ax = plt.subplots()
        x, z, data = self.get_arrays()
        mappable = cplot.plot_depth_section(ax, x, z, data[:-1, :-1], shading="flat")
        assert mappable.get_array().size == 3 * 5

    def test_shape_errors(self):
        fig, ax = plt.subplots()
        x, z, data = self.get_arrays()
        with pytest.raises(xoa.XoaError, match="same shape"):
            cplot.plot_depth_section(ax, x, z[:-1], data)
        with pytest.raises(xoa.XoaError, match="2D"):
            cplot.plot_depth_section(ax, x[0], z[0], data[0])


class TestPlotSticks:
    def test_values_and_axes(self):
        fig, ax = plt.subplots()
        x = np.arange(5.0)
        u = np.linspace(-1, 1, 5)
        v = np.linspace(1, -1, 5)
        quiver = cplot.plot_sticks(ax, x, u, v, scale=2.0, color="red")
        np.testing.assert_allclose(quiver.U, u)
        np.testing.assert_allclose(quiver.V, v)
        np.testing.assert_allclose(quiver.X, x)
        np.testing.assert_allclose(quiver.Y, 0)
        assert not ax.get_yaxis().get_visible()
        assert len(ax.lines) == 1  # the zero line

    def test_head_less_by_default_and_overridable(self):
        fig, ax = plt.subplots()
        quiver = cplot.plot_sticks(ax, [0.0, 1.0], [1.0, 1.0], [0.0, 0.0])
        assert quiver.headlength == 0
        quiver = cplot.plot_sticks(ax, [0.0, 1.0], [1.0, 1.0], [0.0, 0.0], headlength=5)
        assert quiver.headlength == 5


class TestCenterResolution:
    def test_regular_grid_on_the_equator(self):
        lon, lat = np.meshgrid(np.arange(5.0), np.arange(4.0) * 0)
        res = cgrid.compute_center_resolution(lon, lat)
        assert res.shape == (4, 5)
        # dy is zero on this degenerate grid, so use a real grid below
        lon, lat = np.meshgrid(np.arange(5.0), np.arange(4.0))
        res = cgrid.compute_center_resolution(lon, lat)
        ref = np.deg2rad(1.0) * 6371e3
        np.testing.assert_allclose(res[0], np.sqrt(ref * ref), rtol=5e-3)

    def test_matches_resolution_in_the_interior(self):
        lon, lat = np.meshgrid(np.linspace(0, 4, 5), np.linspace(40, 44, 5))
        dx, dy = cgrid.compute_resolution(lon, lat)
        res = cgrid.compute_center_resolution(lon, lat)
        expected = np.sqrt(0.5 * (dx[2, 1] + dx[2, 2]) * 0.5 * (dy[1, 2] + dy[2, 2]))
        np.testing.assert_allclose(res[2, 2], expected)

    def test_smallest_grid(self):
        lon, lat = np.meshgrid(np.arange(2.0), np.arange(2.0))
        res = cgrid.compute_center_resolution(lon, lat)
        assert res.shape == (2, 2)
        assert np.all(res > 0)


class TestAddColorbar:
    @staticmethod
    def get_mappable(ax):
        return ax.pcolormesh(np.arange(12.0).reshape(3, 4))

    def test_shrunk_by_default(self):
        fig, ax = plt.subplots()
        cbar = cplot.add_colorbar(self.get_mappable(ax), ax)
        fig2, ax2 = plt.subplots()
        full = cplot.add_colorbar(self.get_mappable(ax2), ax2, shrink=1.0)
        ratio = cbar.ax.get_position().height / full.ax.get_position().height
        np.testing.assert_allclose(ratio, cplot.CBAR_SHRINK)
        assert cplot.CBAR_SHRINK == 0.7

    def test_axes_default_to_the_artist_axes(self):
        fig, ax = plt.subplots()
        cbar = cplot.add_colorbar(self.get_mappable(ax), label="my label")
        assert len(fig.axes) == 2
        assert cbar.ax.get_ylabel() == "my label"

    def test_shared_colorbar(self):
        fig, axes = plt.subplots(1, 3)
        mappable = None
        for ax in axes:
            mappable = self.get_mappable(ax)
        cbar = cplot.add_colorbar(mappable, axes)
        assert len(fig.axes) == 4
        assert cbar.ax is fig.axes[-1]
        # All axes made room for the colorbar
        assert all(ax.get_position().x1 < cbar.ax.get_position().x0 for ax in axes)


def test_auto_stride_uses_a_bounded_amount_of_memory():
    import tracemalloc

    lon, lat = get_lonlat(2000, 2000)
    ax = get_axes()
    tracemalloc.start()
    strides = cplot.get_strides("auto", lon, lat, ax, min_spacing=12)
    peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    assert strides == (48, 48)  # 0.25 pixel spacing, so at least 48 cells for 12 pixels
    assert peak < 0.5 * lon.nbytes  # and not a full (ny, nx, 2) array


def test_get_projection():
    ccrs = pytest.importorskip("cartopy.crs")
    assert isinstance(cplot.get_projection(), ccrs.Mercator)
    assert isinstance(cplot.get_projection("MERC"), ccrs.Mercator)
    assert isinstance(cplot.get_projection("pc"), ccrs.PlateCarree)
    assert isinstance(cplot.get_projection("platecarree"), ccrs.PlateCarree)
    proj = cplot.get_projection("ortho", [10, 20, -40, -30])
    assert isinstance(proj, ccrs.Orthographic)
    assert proj.proj4_params["lon_0"] == 15 and proj.proj4_params["lat_0"] == -35
    assert isinstance(cplot.get_projection("robinson"), ccrs.Robinson)
    proj = cplot.get_projection("stereographic", [10, 20, -40, -30])
    assert isinstance(proj, ccrs.Stereographic) and proj.proj4_params["lat_0"] == -35
    crs = ccrs.PlateCarree()
    assert cplot.get_projection(crs) is crs
    with pytest.raises(cplot.exceptions.XoaError, match="Invalid projection name"):
        cplot.get_projection("unknown")
    with pytest.raises(cplot.exceptions.XoaError, match="mandatory parameters"):
        cplot.get_projection("utm")


class TestTaylor:
    def test_artists_attributes(self):
        diagram = cplot.plot_taylor([0.5, 1.0], [0.0, 1.0], labels=["a", "b"])
        assert diagram.reference.get_marker() == "*"
        assert diagram.ref_arc is not None
        assert diagram.contours is not None and len(diagram.contour_labels)
        assert diagram.corr_label.get_text() == "Correlation"
        assert diagram.std_label.get_text() == "Standard deviation"
        assert len(diagram.markers) == 2
        assert diagram.labels == []
        diagram = cplot.plot_taylor([0.5, 1.0], [0.0, 1.0], labels=["a", "b"], values=[1, 2])
        assert [t.get_text() for t in diagram.labels] == ["a", "b"]

    def test_positions(self):
        diagram = cplot.plot_taylor([0.5, 1.0], [0.0, 1.0])
        x, y = diagram.markers[0].get_data()
        assert np.isclose(x[0], np.pi / 2) and np.isclose(y[0], 0.5)
        assert not diagram.negative
        assert np.isclose(diagram.ax.get_thetamax(), 90)

    def test_negative_correlations(self):
        diagram = cplot.plot_taylor([0.5, 1.0], [-0.5, 1.0])
        assert diagram.negative
        assert np.isclose(diagram.ax.get_thetamax(), 180)
        x, _ = diagram.markers[0].get_data()
        assert np.isclose(x[0], np.arccos(-0.5))
        with pytest.raises(xoa.exceptions.XoaError):
            cplot.plot_taylor([1.0], [-0.5], diagram=cplot.TaylorDiagram())

    def test_default_reference_is_normalized(self):
        assert cplot.TaylorDiagram().ref_std == 1.0
        diagram = cplot.plot_taylor([3.0], [0.9], ref_std=2.0)
        assert diagram.ref_std == 2.0
        assert diagram.rmax >= 3.0

    def test_labels_markers_legend(self):
        diagram = cplot.plot_taylor(
            [1, 1, 1], [0.9, 0.8, 0.7], labels=["a", "b", "c"], markers=["o", "s"]
        )
        texts = [t.get_text() for t in diagram.legend.get_texts()]
        assert texts == ["Reference", "a", "b", "c"]
        assert [a.get_marker() for a in diagram.markers] == ["o", "s", "o"]

    def test_values_make_a_colorbar(self):
        diagram = cplot.plot_taylor(
            [1, 1, 1], [0.9, 0.8, 0.7], values=[1, 2, 3], markers=["o", "s"]
        )
        assert diagram.colorbar is not None
        assert diagram.legend is None
        assert len(diagram.markers) == 2

    def test_errors(self):
        with pytest.raises(xoa.exceptions.XoaError):
            cplot.plot_taylor([1.0, 1.0], [0.5])
        with pytest.raises(xoa.exceptions.XoaError):
            cplot.plot_taylor([1.0], [1.5])
        with pytest.raises(xoa.exceptions.XoaError):
            cplot.plot_taylor([1.0], [0.5], labels=["a", "b"])

    def test_tuning(self):
        diagram = cplot.plot_taylor(
            [1.0],
            [0.5],
            rmax=2,
            rms_levels=False,
            corr_ticks=[0, 0.5, 1],
            ref_kwargs={"arc": False},
        )
        assert diagram.ax.get_ylim()[1] == 2
        assert len(diagram.ax.get_xticks()) == 3
