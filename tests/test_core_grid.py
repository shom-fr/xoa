# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.grid` module
"""

import numpy as np
import pytest

from xoa.core import grid


def test_centers2edges_1d():
    np.testing.assert_allclose(grid.centers2edges(np.array([0.0, 1.0, 2.0])), [-0.5, 0.5, 1.5, 2.5])


def test_centers2edges_2d():
    xx, yy = np.meshgrid(np.arange(3.0), np.arange(2.0))
    assert grid.centers2edges(xx).shape == (3, 4)
    assert grid.centers2edges(xx, axis=1).shape == (2, 4)


def test_edges2bounds():
    assert grid.edges2bounds(np.arange(4.0)).shape == (3, 2)
    xx, yy = np.meshgrid(np.arange(4.0), np.arange(3.0))
    bounds = grid.edges2bounds(xx)
    assert bounds.shape == (2, 3, 4)
    np.testing.assert_array_equal(bounds[0, 0], [0.0, 1.0, 1.0, 0.0])


def test_check_grid_type():
    xx, yy = np.meshgrid(np.arange(4.0), np.arange(3.0))
    assert grid.check_grid_type({"lon": xx, "lat": yy}) == "regular"
    xx, yy = np.meshgrid(np.array([0.0, 1.0, 3.0, 4.0]), np.arange(3.0))
    assert grid.check_grid_type({"lon": xx, "lat": yy}) == "rectangular"
    assert grid.check_grid_type({"lon": xx, "lat": yy + 0.1 * xx}) == "curvilinear"
    assert grid.check_grid_type({"lon": xx, "lat": yy, "type": "regular"}) == "regular"


def test_compute_resolution():
    lon, lat = np.meshgrid(np.linspace(0, 2, 3), np.linspace(0, 3, 4))
    dx, dy = grid.compute_resolution(lon, lat)
    assert dx.shape == (4, 2)
    assert dy.shape == (3, 3)
    # One degree along the equator and a meridian
    ref = np.deg2rad(1.0) * 6371e3
    np.testing.assert_allclose(dx[0], ref)
    np.testing.assert_allclose(dy, ref)
    # Smaller x resolution at high latitudes
    assert dx[-1, 0] < dx[0, 0]
    np.testing.assert_allclose(dx[-1, 0], ref * np.cos(np.deg2rad(3.0)), rtol=1e-3)
    # Radius
    dx2, dy2 = grid.compute_resolution(lon, lat, radius=1.0)
    np.testing.assert_allclose(dx2, dx / 6371e3)


def test_median_resolution_deg():
    lon, lat = np.meshgrid(np.arange(0, 5, 0.5), np.arange(0, 4, 0.25))
    np.testing.assert_allclose(grid.median_resolution_deg(lon, lat), 0.5)
    # A rotated grid spans more than its step along each axis
    rot = np.deg2rad(30.0)
    rlon = lon * np.cos(rot) - lat * np.sin(rot)
    rlat = lon * np.sin(rot) + lat * np.cos(rot)
    assert grid.median_resolution_deg(rlon, rlat) > 0.25


def test_check_grid_type_with_a_single_row_or_column():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        xx, yy = np.meshgrid(np.arange(5.0), np.zeros(1))
        assert grid.check_grid_type({"lon": xx, "lat": yy}) == "regular"
        xx, yy = np.meshgrid(np.zeros(1), np.arange(4.0))
        assert grid.check_grid_type({"lon": xx, "lat": yy}) == "regular"
        xx, yy = np.meshgrid(np.array([0.0, 1.0, 3.0]), np.zeros(1))
        assert grid.check_grid_type({"lon": xx, "lat": yy}) == "rectangular"


def test_unwrap_grid_longitudes():
    lon = np.array([[178.0, 180.0, -178.0, -176.0]] * 2)
    out = grid.unwrap_grid_longitudes(lon)
    np.testing.assert_allclose(out[0], [178.0, 180.0, 182.0, 184.0])
    assert out.shape == lon.shape
    # Already continuous: unchanged
    lon = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(grid.unwrap_grid_longitudes(lon), lon)


def test_compute_centered_resolution():
    lon, lat = np.meshgrid(np.linspace(0, 2, 3), np.linspace(0, 3, 4))
    dx, dy = grid.compute_centered_resolution(lon, lat)
    assert dx.shape == dy.shape == lon.shape
    ref = np.deg2rad(1.0) * 6371e3
    np.testing.assert_allclose(dy, ref)
    np.testing.assert_allclose(dx[0], ref)
    # Resolution between the adjacent points on the edges, and their mean elsewhere
    dxe, _ = grid.compute_resolution(lon, lat)
    np.testing.assert_allclose(dx[:, 0], dxe[:, 0])
    np.testing.assert_allclose(dx[:, 1], dxe[:, 0])
    np.testing.assert_allclose(dx[:, -1], dxe[:, -1])


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_axis_operators(axis, dtype):
    rng = np.random.default_rng(0)
    a = rng.random((3, 4, 5)).astype(dtype)
    a[0, 0, 0] = np.nan
    first = np.take(a, [0], axis)
    last = np.take(a, [-1], axis)
    lo = np.take(a, range(a.shape[axis] - 1), axis)
    hi = np.take(a, range(1, a.shape[axis]), axis)

    for out, ref in [
        (grid.diff(a, axis), hi - lo),
        (grid.centers(a, axis), lo + 0.5 * (hi - lo)),
        (
            grid.pad(a, axis, (2, 1)),
            np.concatenate([first, first, a, last], axis),
        ),
        (
            grid.pad(a, axis, 1, "linear_extrap"),
            np.concatenate([2 * first - np.take(a, [1], axis), a, 2 * last - np.take(a, [-2], axis)], axis),
        ),
        (
            grid.edges(a, axis),
            np.concatenate(
                [
                    first + 0.5 * (first - np.take(a, [1], axis)),
                    lo + 0.5 * (hi - lo),
                    last + 0.5 * (last - np.take(a, [-2], axis)),
                ],
                axis,
            ),
        ),
    ]:
        assert out.dtype == a.dtype
        np.testing.assert_allclose(out, ref, rtol=1e-6, equal_nan=True)


def test_axis_operators_errors():
    with pytest.raises(ValueError):
        grid.pad(np.arange(3.0), mode="reflect")
    with pytest.raises(ValueError):
        grid.edges(np.ones(1))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_pad_modes_shift_gradient(axis):
    rng = np.random.default_rng(1)
    a = rng.random((3, 4, 5))
    n = a.shape[axis]
    width = [(0, 0)] * 3
    width[axis] = (2, 6)
    for mode, kw in [("wrap", {}), ("constant", {"constant_values": 7.0})]:
        np.testing.assert_array_equal(grid.pad(a, axis, (2, 6), mode, **kw), np.pad(a, width, mode, **kw))
    assert np.isnan(grid.pad(a, axis, 1, "constant")).sum() == 2 * a.size // n

    padded = np.pad(a, _width(axis, (1, 0)), "edge")
    np.testing.assert_allclose(grid.shift(a, axis, -1), grid.centers(padded, axis))
    padded = np.pad(a, _width(axis, (0, 1)), "edge")
    np.testing.assert_allclose(grid.shift(a, axis, 1), grid.centers(padded, axis))
    padded = grid.pad(a, axis, (0, 1), "linear_extrap")
    np.testing.assert_allclose(grid.shift(a, axis, 1, "linear_extrap"), grid.centers(padded, axis))

    x = np.sort(rng.random(n))
    shape = [1, 1, 1]
    shape[axis] = n - 1
    ref = np.diff(a, axis=axis) / np.diff(x).reshape(shape)
    np.testing.assert_allclose(grid.gradient(a, x, axis), ref)
    shape[axis] = n
    np.testing.assert_allclose(grid.gradient(a, np.broadcast_to(x.reshape(shape), a.shape), axis), ref)
    with pytest.raises(ValueError):
        grid.gradient(a, x[:-1], axis)


def _width(axis, pad):
    width = [(0, 0)] * 3
    width[axis] = pad
    return width


def test_centers2edges_axes():
    xx = np.random.default_rng(0).random((3, 4))
    np.testing.assert_allclose(grid.centers2edges(xx, axis=0), grid.edges(xx, 0))
    np.testing.assert_allclose(grid.centers2edges(xx), grid.edges(grid.edges(xx, 0), 1))


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_lon_across_dateline(dtype):
    lon = np.array([[170.0, 178.0, -174.0, -166.0]] * 2, dtype)
    row = lambda x: np.array([x] * 2, dtype)  # noqa
    np.testing.assert_allclose(grid.diff(lon, 1, lon=True), row([8, 8, 8]))
    np.testing.assert_allclose(grid.centers(lon, 1, lon=True), row([174, -178, -170]))
    np.testing.assert_allclose(grid.edges(lon, 1, lon=True), row([166, 174, -178, -170, -162]))
    np.testing.assert_allclose(
        grid.pad(lon, 1, 2, "linear_extrap", lon=True),
        row([154, 162, 170, 178, -174, -166, -158, -150]),
    )
    np.testing.assert_allclose(
        grid.shift(lon, 1, 1, "linear_extrap", lon=True), row([174, -178, -170, -162])
    )
    # Positive longitudes keep their range, and no wrapping occurs without a jump
    np.testing.assert_allclose(grid.centers(lon % 360, 1, lon=True), row([174, 182, 190]))
    np.testing.assert_allclose(grid.pad(np.arange(3.0), 0, 2, "linear_extrap", lon=True), np.arange(-2, 5))


def test_is_periodic_lon():
    for lons in [np.arange(-180.0, 180.0, 2.0), np.arange(0.0, 360.0, 3.0)[::-1], np.roll(np.arange(-180.0, 180.0, 2.0), 7)]:
        assert grid.is_periodic_lon(np.tile(lons, (2, 1)))
    for lons in [np.arange(-120.0, 121.0, 2.0), np.arange(0.0, 300.0, 2.0), np.arange(-180.0, 181.0, 2.0)[:5]]:
        assert not grid.is_periodic_lon(np.tile(lons, (2, 1)))
