# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.interp` module
"""

import functools
import numpy as np
import pytest

from xoa.core import geo, interp
from xoa.exceptions import XoaDeprecationWarning
import os
import tempfile
from xoa.core.interp import XYInterpolator, interp_transect
from xoa.core.regrid import XYRegridder
from xoa.core.time import compute_time_frac_indices
from xoa.core.grid import create_rotated_grid


def vfunc(t=0, z=0, y=0, x=0):
    """A function that returns a linear combination of coordinates"""
    return 1.13 * x + 12.35 * y + 3.24 * z - 0.65 * t


def round_as_time(arr, units="us", origin="1950-01-01"):
    arr = arr.astype(f"m8[{units}]")
    origin = np.datetime64(origin, units)
    arr = arr + origin
    return (arr - origin) / np.timedelta64(1, units)


@pytest.mark.parametrize(
    "x,y,pt,qt",
    [
        (0.0, 0, 0, 0),
        (3, 1, 0, 1),
        (2, 3, 1, 1),
        (-1, 2, 1, 0),
        (1.0, 1.5, 0.5, 0.5),
        (-1, -1, -1, -1),
    ],
)
def test_interp_cell2relloc(x, y, pt, qt):
    x1, y1 = 0.0, 0.0
    x2, y2 = 3.0, 1.0
    x3, y3 = 2.0, 3.0
    x4, y4 = -1.0, 2.0
    with pytest.warns(XoaDeprecationWarning):
        p, q = interp.cell2relloc(x1, x2, x3, x4, y1, y2, y3, y4, x, y)
    assert p == pt
    assert q == qt
    p, q = geo.relative_cell_coords(x1, x4, x3, x2, y1, y4, y3, y2, x, y)
    assert p == pt
    assert q == qt


def test_interp_closest2d_deprecated():
    xxi, yyi = np.meshgrid(np.arange(5.0), np.arange(4.0))
    with pytest.warns(XoaDeprecationWarning):
        assert interp.closest2d(xxi, yyi, 2.1, 1.2) == (2, 1)


@functools.lru_cache()
def get_grid2locs_coords(nex=4, nexz=2, nxi=7, nyi=6, nzi=5, nti=4, no=10):
    np.random.seed(0)

    tti, zzi, yyi, xxi = np.mgrid[
        0 : nti - 1 : nti * 1j,
        0 : nzi - 1 : nzi * 1j,
        0 : nyi - 1 : nyi * 1j,
        0 : nxi - 1 : nxi * 1j,
    ]

    zzi = zzi[None]
    zzi = np.repeat(zzi, nexz, axis=0)

    xyztomin = -0.5
    xo = np.random.uniform(xyztomin, nxi - 1.5, no)
    yo = np.random.uniform(xyztomin, nyi - 1.5, no)
    zo = np.random.uniform(xyztomin, nzi - 1.5, no)
    to = np.random.uniform(xyztomin, nti - 1.5, no)

    tti = round_as_time(tti)
    to = round_as_time(to)

    return xxi, yyi, zzi, tti, xo, yo, to, zo


def test_interp_grid2locs():
    # Multi-dimensional generic coordinates
    nex = 4
    nexz = 2
    nxi = 7
    nyi = 6
    nzi = 5
    nti = 4
    no = 10
    xxi, yyi, zzi, tti, xo, yo, to, zo = get_grid2locs_coords(
        nex=nex, nexz=nexz, nxi=nxi, nyi=nyi, nzi=nzi, nti=nti, no=no
    )

    # Pure 1D axes
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)
    vo_truth = np.array(vfunc(to, zo, yo, xo))
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    assert vo_interp[0].shape == vo_truth.shape
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # Single point in space
    xi = xxi[0, 0, 0:1, :1]  # (nyix=1,nxi)
    yi = yyi[0, 0, :1, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :, :, :1, :1]
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,1,1)
    vo_truth = np.array(vfunc(to, zo, yi[0], xi[0]))
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # Constant time
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:1, 0, 0, 0]  # (1)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :1, :, :, :]
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,1,nzi,nyi,nxi)
    vo_truth = vfunc(ti, zo, yo, xo)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # Variable depth with 1D X/Y + T
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[:, :, :, :, :]  # (nexz,ntiz=nti,nzi,nyiz=nyi,nxiz=nxi)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)
    vo_truth = vfunc(to, zo, yo, xo)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)

    # 2D X/Y with no other axes (pure curvilinear)
    xi = xxi[0, 0]  # (nyix=nyi,nxi)
    yi = yyi[0, 0]  # (nyi,nxiy=nxi)
    zi = zzi[0:1, 0:1, 0:1, 0:1, 0:1]  # (nexz=1,ntiz=1,1,nyiz=1,nxiz=1)
    ti = tti[:1, 0, 0, 0]  # (1)
    vi = vfunc(tti, zzi, yyi, xxi)[:, :1, :1, :, :]
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,1,1,nyi,nxi)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_interp_rect = interp.grid2locs(xi[:1], yi[:, :1], zi, ti, vi, xo, yo, zo, to)
    vo_truth = vfunc(ti, zi.ravel()[0], yo, xo)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)
    vo_truth = vfunc(ti, zi.ravel()[0], yo, xo)
    vo_truth[np.isnan(vo_interp_rect[0])] = np.nan
    np.testing.assert_allclose(vo_interp_rect[0], vo_truth)

    # Same coordinates
    xi = xxi[0, 0, 0:1, :]  # (nyix=1,nxi)
    yi = yyi[0, 0, :, 0:1]  # (nyi,nxiy=1)
    zi = zzi[0:1, 0:1, :, 0:1, 0:1]  # (nexz=1,ntiz=1,nzi,nyiz=1,nxiz=1)
    ti = tti[:, 0, 0, 0]  # (nti)
    vi = vfunc(tti, zzi, yyi, xxi)
    vi = np.resize(vi, (nex,) + vi.shape[1:])  # (nex,nti,nzi,nyi,nxi)
    tzyxo = np.meshgrid(ti, zi, yi, xi, indexing='ij')
    xo = tzyxo[3].ravel()
    yo = tzyxo[2].ravel()
    zo = tzyxo[1].ravel()
    to = tzyxo[0].ravel()
    vo_truth = vfunc(to, zo, yo, xo)
    vo_interp = interp.grid2locs(xi, yi, zi, ti, vi, xo, yo, zo, to)
    vo_truth[np.isnan(vo_interp[0])] = np.nan
    np.testing.assert_allclose(vo_interp[0], vo_truth)


def get_sheared_curvilinear_grid(nxi=6, nyi=5):
    """A genuinely non-separable (sheared) curvilinear lon/lat grid

    Unlike a curvilinear-shaped grid built from ``np.meshgrid`` of two
    independent 1D axes, longitude and latitude here both depend on
    *both* grid indices, so the grid cannot be reduced to a rectilinear
    one. The mapping is affine so the inverse (index -> position) is
    known exactly, which is used to check the interpolation results.
    """
    ii, jj = np.meshgrid(np.arange(nxi, dtype="d"), np.arange(nyi, dtype="d"))
    xxi = ii + 0.3 * jj
    yyi = 0.1 * ii + jj
    return xxi, yyi


def relloc_sheared_curvilinear_grid(xo, yo):
    """Exact analytic inverse of :func:`get_sheared_curvilinear_grid`"""
    j = (yo - 0.1 * xo) / 0.97
    i = xo - 0.3 * j
    return i, j


def test_geo_closest_point_fast_curvilinear():
    """Regression test for a data race in the closest2d parallel scan

    On a genuinely non-separable curvilinear grid, ``closest2d`` used to
    parallelize its row scan with ``numba.prange`` while accumulating
    the result in shared scalars (``mindist``, ``i``, ``j``). This is a
    classic data race that silently returned wrong (and non
    deterministic) indices -- typically ``(0, 0)`` -- instead of the
    actual closest grid point.
    """
    xxi, yyi = get_sheared_curvilinear_grid()
    for _ in range(20):  # repeat: a race condition may not fail every time
        i, j = geo.closest_point_fast(xxi, yyi, 2.3, 2.6)
        assert (i, j) == (2, 2)


def test_geo_closest_point_fast_ignores_nan_corners():
    """Regression test for closest2d silently misbehaving on NaN corners

    ``closest2d`` used to be decorated with ``fastmath=True``, which
    enables LLVM's ``nnan`` flag and can break the ``dist <= mindist``
    NaN-skip comparison it relies on to ignore invalid (e.g.
    land-masked) grid corners -- silently returning the wrong point
    instead of skipping the NaN ones.
    """
    xxi, yyi = get_sheared_curvilinear_grid(nxi=6, nyi=5)
    # Mask a block of corners as NaN, away from the query point
    xxi = xxi.copy()
    yyi = yyi.copy()
    xxi[0:2, 0:2] = np.nan
    yyi[0:2, 0:2] = np.nan

    i, j = geo.closest_point_fast(xxi, yyi, 2.3, 2.6)
    assert (i, j) == (2, 2)


def test_interp_grid2relloc_curvilinear():
    """grid2relloc on a genuinely non-separable curvilinear grid"""
    xxi, yyi = get_sheared_curvilinear_grid()
    xo, yo = 2.3, 2.6

    p, q = interp.grid2relloc(xxi, yyi, xo, yo)

    i_expected, j_expected = relloc_sheared_curvilinear_grid(xo, yo)
    np.testing.assert_allclose([p, q], [i_expected, j_expected])


def test_interp_grid2locs_curvilinear():
    """grid2locs value interpolation on a genuinely curvilinear grid"""
    nxi, nyi = 6, 5
    xxi, yyi = get_sheared_curvilinear_grid(nxi=nxi, nyi=nyi)
    ii, jj = np.meshgrid(np.arange(nxi, dtype="d"), np.arange(nyi, dtype="d"))
    vi = (ii + 10.0 * jj).reshape(1, 1, 1, nyi, nxi)

    xo = np.array([2.3])
    yo = np.array([2.6])
    zo = np.zeros(1)
    to = np.zeros(1)
    ti = np.zeros(1)
    zi = np.zeros((1, 1, 1, 1, 1))

    vo = interp.grid2locs(xxi, yyi, zi, ti, vi, xo, yo, zo, to)

    i_expected, j_expected = relloc_sheared_curvilinear_grid(xo[0], yo[0])
    expected = i_expected + 10.0 * j_expected
    np.testing.assert_allclose(vo[0], [expected])


def test_interp_isoslice():
    depth = np.linspace(-50, 0.0, 6)
    values = np.linspace(10, 20.0, 6)
    isoval = 15.0

    isodepth = interp.isoslice(depth, values, isoval, False)
    assert isodepth == -25.0
    isodepth = interp.isoslice(depth, values, isoval, True)
    assert isodepth == -25.0

    depth = np.resize(depth, (2,) + depth.shape)
    isodepth = interp.isoslice(depth, values, isoval, False)
    np.testing.assert_allclose(isodepth, [-25.0, -25.0])


def _make_src_grid(nx=40, ny=30, lon0=-10.0, lat0=40.0, dlon=0.25, dlat=0.25):
    lon = np.linspace(lon0, lon0 + dlon * nx, nx, endpoint=False)
    lat = np.linspace(lat0, lat0 + dlat * ny, ny, endpoint=False)
    lon2d, lat2d = np.meshgrid(lon, lat)
    return {'lon': lon2d, 'lat': lat2d, 'type': 'regular'}


def _make_data(src_grid, K=None):
    base = (src_grid['lon'] + src_grid['lat']).astype(np.float64)
    if K is None:
        return base
    return np.stack([base + k for k in range(K)])


class TestXYInterpolatorScalar:
    def test_single_point_bilinear(self):
        src = _make_src_grid()
        data = _make_data(src)
        interp = XYInterpolator(src, np.array([-5.0]), np.array([43.0]), method='bilinear')
        result = interp.interp(data)
        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert result[0] == pytest.approx(-5.0 + 43.0, abs=0.05)

    def test_single_point_bicubic(self):
        src = _make_src_grid()
        data = _make_data(src)
        interp = XYInterpolator(src, np.array([-5.0]), np.array([43.0]), method='bicubic')
        result = interp.interp(data)
        assert result.shape == (1,)
        assert np.isfinite(result[0])
        assert result[0] == pytest.approx(-5.0 + 43.0, abs=0.05)


class TestXYInterpolatorTransect:
    def test_shape(self):
        src = _make_src_grid()
        data = _make_data(src)
        dst_lon = np.linspace(-8.0, -2.0, 20)
        dst_lat = np.linspace(41.0, 45.0, 20)
        result = XYInterpolator(src, dst_lon, dst_lat, method='bilinear').interp(data)
        assert result.shape == (20,)
        assert np.all(np.isfinite(result))

    def test_values_lon_plus_lat(self):
        src = _make_src_grid()
        data = _make_data(src)
        dst_lon = np.linspace(-8.0, -3.0, 10)
        dst_lat = np.linspace(41.5, 44.5, 10)
        result = XYInterpolator(src, dst_lon, dst_lat, method='bilinear').interp(data)
        np.testing.assert_allclose(result, dst_lon + dst_lat, atol=0.05)

    def test_outside_domain_is_nan(self):
        src = _make_src_grid()
        data = _make_data(src)
        result = XYInterpolator(src, np.array([100.0]), np.array([43.0])).interp(data)
        assert np.isnan(result[0])


class TestXYInterpolator2D:
    def test_2d_dst_shape_preserved(self):
        src = _make_src_grid()
        data = _make_data(src)
        dst_lon = np.tile(np.linspace(-8.0, -2.0, 5), (4, 1))
        dst_lat = np.tile(np.linspace(41.0, 44.0, 4)[:, None], (1, 5))
        result = XYInterpolator(src, dst_lon, dst_lat).interp(data)
        assert result.shape == (4, 5)

    def test_matches_regridder_for_grid_dst(self):
        src = _make_src_grid()
        data = _make_data(src)
        dst_lon2d, dst_lat2d = np.meshgrid(np.linspace(-8.0, -2.0, 8), np.linspace(41.0, 44.0, 6))
        dst_grid = {'lon': dst_lon2d, 'lat': dst_lat2d, 'type': 'regular'}
        result_interp = XYInterpolator(src, dst_lon2d, dst_lat2d, method='bilinear').interp(data)
        result_regrid = XYRegridder(src, dst_grid, method='bilinear').regrid(data)
        np.testing.assert_array_equal(result_interp, result_regrid)


class TestXYInterpolatorExtraDims:
    def test_3d_data(self):
        src = _make_src_grid()
        data = _make_data(src, K=5)
        dst_lon = np.linspace(-8.0, -2.0, 10)
        dst_lat = np.linspace(41.0, 44.0, 10)
        result = XYInterpolator(src, dst_lon, dst_lat).interp(data)
        assert result.shape == (5, 10)

    def test_4d_data(self):
        src = _make_src_grid()
        ny, nx = src['lon'].shape
        data = np.random.rand(3, 5, ny, nx)
        dst_lon = np.linspace(-8.0, -2.0, 7)
        dst_lat = np.linspace(41.0, 44.0, 7)
        result = XYInterpolator(src, dst_lon, dst_lat).interp(data)
        assert result.shape == (3, 5, 7)


class TestXYInterpolatorSaveLoad:
    def test_weights_roundtrip(self):
        src = _make_src_grid()
        data = _make_data(src)
        dst_lon = np.linspace(-8.0, -2.0, 15)
        dst_lat = np.linspace(41.0, 44.0, 15)
        interp = XYInterpolator(src, dst_lon, dst_lat, method='bilinear')
        interp.compute_weights()
        expected = interp.interp(data)

        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            interp.save_weights(fname)
            interp2 = XYInterpolator(src, dst_lon, dst_lat, method='bilinear')
            interp2.load_weights(fname)
            np.testing.assert_array_equal(expected, interp2.interp(data))
        finally:
            os.unlink(fname)

    def test_save_before_compute_raises(self):
        src = _make_src_grid()
        interp = XYInterpolator(src, np.array([-5.0]), np.array([43.0]))
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as f:
            fname = f.name
        try:
            with pytest.raises(ValueError, match='No weights computed yet'):
                interp.save_weights(fname)
        finally:
            os.unlink(fname)


def _make_timed_data(src_grid, nt=8):
    """Build f(lon, lat, t) = lon + lat + t: linear in all vars, reproduced exactly."""
    t = np.arange(nt, dtype=np.float64)
    return (
        src_grid['lon'][np.newaxis, :, :]
        + src_grid['lat'][np.newaxis, :, :]
        + t[:, np.newaxis, np.newaxis]
    ).astype(np.float64)


class TestInterpTransect:
    """Direct tests of the interp_transect numba kernel."""

    def _weights(self, src, dst_lon, dst_lat, method):
        xi = XYInterpolator(src, dst_lon, dst_lat, method=method)
        xi.compute_weights()
        return xi

    def test_linear_field_bilinear_exact(self):
        """f = lon + lat + t must be reproduced exactly (bilinear XY + linear T)."""
        src = _make_src_grid()
        nt = 6
        data = _make_timed_data(src, nt)
        ny, nx = src['lon'].shape
        src_times = np.arange(nt, dtype=np.float64)

        n_pts = 15
        dst_lon = np.linspace(-8.0, -3.0, n_pts)
        dst_lat = np.linspace(41.5, 44.5, n_pts)
        dst_times = np.linspace(0.5, nt - 1.5, n_pts)

        xi = self._weights(src, dst_lon, dst_lat, 'bilinear')
        it_base, frac_t = compute_time_frac_indices(src_times, dst_times)

        out = interp_transect(
            data.reshape(nt, 1, ny * nx),
            xi._j_base,
            xi._i_base,
            xi._frac_a,
            xi._frac_b,
            it_base,
            frac_t,
            nx,
            ny,
            False,
            1.0,
            1,
            1,
            0.0,
            0.0,
        )
        np.testing.assert_allclose(out[0], dst_lon + dst_lat + dst_times, atol=1e-10)

    def test_linear_field_bicubic_exact(self):
        """f = lon + lat + t must also be exact under bicubic XY + linear T."""
        src = _make_src_grid()
        nt = 6
        data = _make_timed_data(src, nt)
        ny, nx = src['lon'].shape
        src_times = np.arange(nt, dtype=np.float64)

        n_pts = 15
        dst_lon = np.linspace(-8.0, -3.0, n_pts)
        dst_lat = np.linspace(41.5, 44.5, n_pts)
        dst_times = np.linspace(0.5, nt - 1.5, n_pts)

        xi = self._weights(src, dst_lon, dst_lat, 'bicubic')
        it_base, frac_t = compute_time_frac_indices(src_times, dst_times)

        out = interp_transect(
            data.reshape(nt, 1, ny * nx),
            xi._j_base,
            xi._i_base,
            xi._frac_a,
            xi._frac_b,
            it_base,
            frac_t,
            nx,
            ny,
            False,
            1.0,
            2,
            1,
            0.0,
            0.0,
        )
        np.testing.assert_allclose(out[0], dst_lon + dst_lat + dst_times, atol=1e-10)

    def test_out_of_time_range_is_nan(self):
        src = _make_src_grid()
        nt, ny, nx = 4, *src['lon'].shape
        data = _make_timed_data(src, nt).reshape(nt, 1, ny * nx)
        src_times = np.arange(nt, dtype=np.float64)
        dst_times = np.array([99.0])  # far outside range

        xi = self._weights(src, np.array([-5.0]), np.array([43.0]), 'bilinear')
        it_base, frac_t = compute_time_frac_indices(src_times, dst_times)

        out = interp_transect(
            data,
            xi._j_base,
            xi._i_base,
            xi._frac_a,
            xi._frac_b,
            it_base,
            frac_t,
            nx,
            ny,
            False,
            1.0,
            1,
            1,
            0.0,
            0.0,
        )
        assert np.isnan(out[0, 0])

    def test_out_of_xy_domain_is_nan(self):
        src = _make_src_grid()
        nt, ny, nx = 4, *src['lon'].shape
        data = _make_timed_data(src, nt).reshape(nt, 1, ny * nx)
        src_times = np.arange(nt, dtype=np.float64)
        dst_times = np.array([1.5])

        xi = self._weights(src, np.array([100.0]), np.array([43.0]), 'bilinear')
        it_base, frac_t = compute_time_frac_indices(src_times, dst_times)

        out = interp_transect(
            data,
            xi._j_base,
            xi._i_base,
            xi._frac_a,
            xi._frac_b,
            it_base,
            frac_t,
            nx,
            ny,
            False,
            1.0,
            1,
            1,
            0.0,
            0.0,
        )
        assert np.isnan(out[0, 0])


class TestXYInterpolatorInterpWithTime:
    def test_linear_field_bilinear_exact(self):
        """f = lon + lat + t must be reproduced exactly (bilinear XY + linear T)."""
        src = _make_src_grid()
        nt = 8
        data = _make_timed_data(src, nt)
        src_times = np.arange(nt, dtype=np.float64)

        n_pts = 20
        dst_lon = np.linspace(-8.0, -3.0, n_pts)
        dst_lat = np.linspace(41.5, 44.5, n_pts)
        dst_times = np.linspace(0.5, nt - 1.5, n_pts)

        result = XYInterpolator(src, dst_lon, dst_lat, method='bilinear').interp_with_time(
            data, src_times, dst_times
        )
        np.testing.assert_allclose(result, dst_lon + dst_lat + dst_times, atol=1e-10)

    def test_linear_field_bicubic_exact(self):
        """f = lon + lat + t must also be exact under bicubic XY + linear T."""
        src = _make_src_grid()
        nt = 8
        data = _make_timed_data(src, nt)
        src_times = np.arange(nt, dtype=np.float64)

        n_pts = 20
        dst_lon = np.linspace(-8.0, -3.0, n_pts)
        dst_lat = np.linspace(41.5, 44.5, n_pts)
        dst_times = np.linspace(0.5, nt - 1.5, n_pts)

        result = XYInterpolator(src, dst_lon, dst_lat, method='bicubic').interp_with_time(
            data, src_times, dst_times
        )
        np.testing.assert_allclose(result, dst_lon + dst_lat + dst_times, atol=1e-10)

    def test_extra_dims_preserved(self):
        """Extra dimensions (depth) are present in output with correct shape."""
        src = _make_src_grid()
        ny, nx = src['lon'].shape
        nt, nz, n_pts = 6, 5, 12
        data = np.random.rand(nt, nz, ny, nx)
        src_times = np.arange(nt, dtype=np.float64)
        dst_times = np.linspace(0.5, nt - 1.5, n_pts)

        result = XYInterpolator(
            src,
            np.linspace(-8.0, -3.0, n_pts),
            np.linspace(41.5, 44.5, n_pts),
        ).interp_with_time(data, src_times, dst_times)
        assert result.shape == (nz, n_pts)

    def test_out_of_time_range_is_nan(self):
        src = _make_src_grid()
        nt = 4
        data = _make_timed_data(src, nt)
        result = XYInterpolator(src, np.array([-5.0]), np.array([43.0])).interp_with_time(
            data, np.arange(nt, dtype=float), np.array([99.0])
        )
        assert np.isnan(result[0])

    def test_mismatched_src_times_raises(self):
        src = _make_src_grid()
        data = _make_timed_data(src, nt=4)
        with pytest.raises(ValueError):
            XYInterpolator(src, np.array([-5.0]), np.array([43.0])).interp_with_time(
                data, np.arange(10, dtype=float), np.array([1.5])
            )


if __name__ == '__main__':
    pytest.main([__file__])


class TestSourceMatrixCopies:
    """The data are copied only when needed, and never modified"""

    def test_single_field_is_not_copied(self):
        from xoa.core.interp import _get_source_matrix_

        data = np.random.rand(6, 8)
        matrix = _get_source_matrix_(data)
        assert matrix.shape == (48, 1)
        assert np.shares_memory(matrix, data)
        np.testing.assert_array_equal(matrix[:, 0], data.ravel())

    def test_several_fields_are_copied_once_in_the_layout_of_the_kernels(self):
        from xoa.core.interp import _get_source_matrix_

        data = np.random.rand(3, 2, 6, 8)
        matrix = _get_source_matrix_(data)
        assert matrix.shape == (48, 6)
        assert matrix.flags.c_contiguous
        assert not np.shares_memory(matrix, data)
        np.testing.assert_array_equal(matrix[:, 4], data[2, 0].ravel())

    @pytest.mark.parametrize("shape", [(6, 8), (3, 6, 8)])
    def test_mask_does_not_modify_the_input(self, shape):
        from xoa.core.interp import _get_source_matrix_

        data = np.random.rand(*shape)
        ref = data.copy()
        mask = np.ones((6, 8), bool)
        mask[2, 3] = False
        matrix = _get_source_matrix_(data, mask)
        np.testing.assert_array_equal(data, ref)
        assert np.all(np.isnan(matrix[2 * 8 + 3]))
        assert np.isnan(matrix).sum() == matrix.shape[1]

    def test_peak_memory_with_a_mask(self):
        import tracemalloc

        ny = nx = 150
        lon, lat = np.meshgrid(np.linspace(0, 10, nx), np.linspace(0, 10, ny))
        mask = np.ones((ny, nx), bool)
        mask[:20] = False
        data = np.random.rand(20, ny, nx)
        dst_lon, dst_lat = np.meshgrid(np.linspace(1, 9, 10), np.linspace(1, 9, 10))
        interp = XYInterpolator({"lon": lon, "lat": lat, "mask": mask}, dst_lon, dst_lat)
        interp.compute_weights()
        ref = data.copy()
        interp.interp(data, skipna=True)  # compilation
        tracemalloc.start()
        out = interp.interp(data, skipna=True)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        # One copy of the data for the kernels and no other one
        assert peak < 1.2 * data.nbytes
        np.testing.assert_array_equal(data, ref)
        assert out.shape == (20, 10, 10)

    def test_peak_memory_of_a_single_field(self):
        import tracemalloc

        ny = nx = 300
        lon, lat = np.meshgrid(np.linspace(0, 10, nx), np.linspace(0, 10, ny))
        data = np.random.rand(ny, nx)
        dst_lon, dst_lat = np.meshgrid(np.linspace(1, 9, 10), np.linspace(1, 9, 10))
        interp = XYInterpolator({"lon": lon, "lat": lat}, dst_lon, dst_lat)
        interp.compute_weights()
        interp.interp(data)
        tracemalloc.start()
        interp.interp(data)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        assert peak < 0.1 * data.nbytes


class TestPointsOnTheEdgesOfTheGrid:
    """Cells are closed, so the first and last lines of the grid are inside, and what is
    outside is outside on both sides"""

    @staticmethod
    def get_grids():
        lon, lat = np.meshgrid(np.linspace(0, 5, 11), np.linspace(0, 4, 9))
        # Not regular, so that the other search algorithms are tested too
        rect_lon, rect_lat = np.meshgrid(
            np.array([0, 0.4, 1.0, 1.5, 2.2, 2.6, 3.0, 3.8, 4.1, 4.6, 5.0]),
            np.array([0, 0.3, 1.0, 1.4, 2.0, 2.2, 3.0, 3.5, 4.0]),
        )
        curv_lon = lon + 0.05 * lat
        curv_lat = lat + 0.05 * lon
        return {
            "regular": (lon, lat),
            "rectangular": (rect_lon, rect_lat),
            "curvilinear": (curv_lon, curv_lat),
        }

    @pytest.mark.parametrize("kind", ["regular", "rectangular", "curvilinear"])
    def test_bilinear_grid_lines_are_inside(self, kind):
        lon, lat = self.get_grids()[kind]
        field = 2 * lon + lat
        interp = XYInterpolator({"lon": lon, "lat": lat}, lon, lat, "bilinear")
        assert interp.src_grid["type"] == (None if kind == "regular" and False else kind)
        out = interp.interp(field)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out, field, atol=1e-9)

    @pytest.mark.parametrize("kind", ["regular", "rectangular", "curvilinear"])
    def test_points_outside_are_nan_on_all_sides(self, kind):
        lon, lat = self.get_grids()[kind]
        field = 2 * lon + lat
        x0, x1 = lon[:, 0].mean(), lon[:, -1].mean()
        y0, y1 = lat[0].mean(), lat[-1].mean()
        outside_lon = np.array([x0 - 0.4, x0 - 0.05, x1 + 0.05, x1 + 0.4, 2.0, 2.0, 2.0, 2.0])
        outside_lat = np.array([2.0, 2.0, 2.0, 2.0, y0 - 0.4, y0 - 0.05, y1 + 0.05, y1 + 0.4])
        interp = XYInterpolator({"lon": lon, "lat": lat}, outside_lon, outside_lat, "bilinear")
        assert np.isnan(interp.interp(field)).all()

    def test_regular_grid_in_between_values(self):
        lon, lat = self.get_grids()["regular"]
        points_lon = np.array([0.0, -1e-12, 5.0, 5.0 + 1e-12, 4.9, 0.25, 2.0, 2.0, 2.0, 2.0])
        points_lat = np.array([1.0, 1.0, 1.0, 1.0, 4.0, 0.0, 0.0, -1e-12, 4.0, 4.0 + 1e-12])
        interp = XYInterpolator({"lon": lon, "lat": lat}, points_lon, points_lat, "bilinear")
        out = interp.interp(2 * lon + lat)
        np.testing.assert_allclose(out, 2 * points_lon + points_lat, atol=1e-6)
        # Slightly more than the tolerance is outside
        for x, y in (5.0 + 1e-6, 1.0), (2.0, 4.0 + 1e-6), (-1e-6, 1.0), (2.0, -1e-6):
            interp = XYInterpolator({"lon": lon, "lat": lat}, np.array([x]), np.array([y]))
            assert np.isnan(interp.interp(2 * lon + lat)).all()

    def test_bicubic_needs_one_more_cell_around(self):
        lon, lat = self.get_grids()["regular"]
        field = 2 * lon + lat
        interp = XYInterpolator({"lon": lon, "lat": lat}, lon, lat, "bicubic")
        out = interp.interp(field)
        # The outer lines have no 4 x 4 stencil, but the next ones are inside
        assert np.isnan(out[0]).all() and np.isnan(out[-1]).all()
        assert np.isnan(out[:, 0]).all() and np.isnan(out[:, -1]).all()
        np.testing.assert_allclose(out[1:-1, 1:-1], field[1:-1, 1:-1], atol=1e-9)

    def test_dateline_and_longitude_wrapping_are_unchanged(self):
        lon, lat = np.meshgrid(np.linspace(170, 180, 6), np.linspace(0, 4, 5))
        interp = XYInterpolator(
            {"lon": lon, "lat": lat}, np.array([175.0, 180.0, 169.0]), np.array([1.0, 1.0, 1.0])
        )
        out = interp.interp(lon)
        np.testing.assert_allclose(out[:2], [175.0, 180.0])
        assert np.isnan(out[2])


def linear(lon, lat):
    return 2.0 * lon - 3.0 * lat + 1.5


def quadratic(lon, lat):
    return 0.3 * lon**2 + 0.2 * lon * lat - 0.1 * lat**2 + lon + 2.0


def smooth(lon, lat):
    return np.sin(0.7 * lon) * np.cos(0.5 * lat) + 0.1 * lon


def get_grids():
    """Source grids of the three types, and points that are far enough from their edges"""
    rng = np.random.default_rng(42)
    regular = np.meshgrid(np.linspace(0, 9, 19), np.linspace(0, 7, 15))
    rect = np.meshgrid(
        np.cumsum(np.r_[0, rng.uniform(0.3, 0.7, 18)]),
        np.cumsum(np.r_[0, rng.uniform(0.3, 0.6, 14)]),
    )
    rotated = create_rotated_grid(21, 17, 5.0, 45.0, 25.0, lon_span=10.0, lat_span=8.0)
    curv = (rotated["lon"], rotated["lat"])
    grids = {}
    for name, (lon, lat) in {"regular": regular, "rectangular": rect, "curvilinear": curv}.items():
        # Random points in a disc that is well inside the grid, so that bicubic stencils are valid
        lon0, lat0 = lon.mean(), lat.mean()
        # The bounding box of a rotated grid is larger than the grid itself
        radius = 0.3 * 8.0 if name == "curvilinear" else 0.35 * min(np.ptp(lon), np.ptp(lat))
        r = radius * np.sqrt(rng.uniform(0, 1, 300))
        angle = rng.uniform(0, 2 * np.pi, 300)
        grids[name] = (lon, lat, lon0 + r * np.cos(angle), lat0 + r * np.sin(angle))
    return grids


GRIDS = get_grids()


class TestExactnessOnPolynomials:
    @pytest.mark.parametrize(
        "name,method",
        [
            ("regular", "bilinear"),
            ("regular", "bicubic"),
            ("rectangular", "bilinear"),
            pytest.param(
                "rectangular",
                "bicubic",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bicubic tangents ignore the irregular spacing of the grid",
                ),
            ),
            ("curvilinear", "bilinear"),
            ("curvilinear", "bicubic"),
        ],
    )
    def test_linear_field_is_exact_at_random_points(self, name, method):
        lon, lat, plon, plat = GRIDS[name]
        interp = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, method)
        assert interp.src_grid["type"] == name or (name == "regular" and interp.src_grid["type"])
        out = interp.interp(linear(lon, lat))
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out, linear(plon, plat), rtol=0, atol=1e-9)

    @pytest.mark.parametrize("name", ["regular", "rectangular"])
    def test_bilinear_function_is_exact(self, name):
        # a + b x + c y + d x y is bilinear in a cell
        lon, lat, plon, plat = GRIDS[name]

        def func(x, y):
            return 1.0 + 2.0 * x - y + 0.5 * x * y

        out = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, "bilinear").interp(
            func(lon, lat)
        )
        np.testing.assert_allclose(out, func(plon, plat), rtol=0, atol=1e-9)

    def test_bicubic_is_exact_for_quadratics_on_a_regular_grid_and_bilinear_is_not(self):
        lon, lat, plon, plat = GRIDS["regular"]
        expected = quadratic(plon, plat)
        out = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, "bicubic").interp(
            quadratic(lon, lat)
        )
        np.testing.assert_allclose(out, expected, rtol=0, atol=1e-9)
        out = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, "bilinear").interp(
            quadratic(lon, lat)
        )
        assert np.abs(out - expected).max() > 1e-3  # the test can tell them apart

    @pytest.mark.parametrize("name", list(GRIDS))
    @pytest.mark.parametrize("method", ["bilinear", "bicubic"])
    def test_values_at_the_nodes_are_the_data(self, name, method):
        lon, lat, _, _ = GRIDS[name]
        field = smooth(lon, lat)
        interp = XYInterpolator({"lon": lon, "lat": lat}, lon, lat, method)
        out = interp.interp(field)
        valid = ~np.isnan(out)
        if method == "bilinear":
            assert valid.all()
        else:  # outer ring of cells
            assert valid[1:-1, 1:-1].all()
        np.testing.assert_allclose(out[valid], field[valid], rtol=0, atol=1e-9)

    @pytest.mark.parametrize("name", list(GRIDS))
    @pytest.mark.parametrize("method", ["bilinear", "bicubic"])
    def test_constant_field_and_partition_of_unity(self, name, method):
        lon, lat, plon, plat = GRIDS[name]
        interp = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, method)
        for value in 0.0, 3.7, -12.0:
            np.testing.assert_allclose(interp.interp(np.full(lon.shape, value)), value, atol=1e-12)
        # The same with missing values that are skipped: weights are renormalized
        field = np.full(lon.shape, 3.7)
        field[::4, ::3] = np.nan
        out = interp.interp(field, skipna=True)
        np.testing.assert_allclose(out[~np.isnan(out)], 3.7, atol=1e-12)
        assert (~np.isnan(out)).mean() > 0.95


class TestConvergence:
    @staticmethod
    def get_errors(method, sizes=(11, 21, 41, 81)):
        rng = np.random.default_rng(0)
        plon, plat = rng.uniform(1.0, 8.0, 500), rng.uniform(1.0, 8.0, 500)
        errors = []
        for n in sizes:
            lon, lat = np.meshgrid(np.linspace(0, 9, n), np.linspace(0, 9, n))
            out = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, method).interp(
                smooth(lon, lat)
            )
            assert not np.isnan(out).any()
            errors.append(np.abs(out - smooth(plon, plat)).max())
        return np.array(errors), 9.0 / (np.array(sizes) - 1)

    @staticmethod
    def get_order(errors, steps):
        return np.polyfit(np.log(steps), np.log(errors), 1)[0]

    def test_bilinear_is_second_order(self):
        errors, steps = self.get_errors("bilinear")
        assert np.all(np.diff(errors) < 0)
        assert 1.8 < self.get_order(errors, steps) < 2.2

    def test_bicubic_is_third_order(self):
        errors, steps = self.get_errors("bicubic")
        assert np.all(np.diff(errors) < 0)
        assert 2.7 < self.get_order(errors, steps) < 3.4

    def test_bicubic_is_more_accurate_than_bilinear(self):
        bilinear, _ = self.get_errors("bilinear")
        bicubic, _ = self.get_errors("bicubic")
        assert np.all(bicubic < bilinear / 4)


class TestSkipnaRenormalization:
    def test_weighted_mean_of_the_valid_corners(self):
        lon, lat = np.meshgrid([0.0, 1.0, 2.0], [0.0, 1.0, 2.0])
        field = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]])
        x, y = 0.3, 0.8  # in the cell with the missing point as its upper right corner
        interp = XYInterpolator({"lon": lon, "lat": lat}, np.array([x]), np.array([y]))
        weights = {(0, 0): (1 - x) * (1 - y), (0, 1): x * (1 - y), (1, 0): (1 - x) * y}
        manual = sum(w * field[ij] for ij, w in weights.items()) / sum(weights.values())
        np.testing.assert_allclose(interp.interp(field, skipna=True), manual, rtol=1e-12)
        assert np.isnan(interp.interp(field, skipna=False)).all()
        # Weight that is missing, to compare with na_thres
        missing = x * y
        assert missing == pytest.approx(0.24)
        assert not np.isnan(interp.interp(field, skipna=True, na_thres=missing + 0.01)).any()
        assert np.isnan(interp.interp(field, skipna=True, na_thres=missing - 0.01)).all()

    @pytest.mark.parametrize("method", ["bilinear", "bicubic"])
    def test_na_thres_zero_keeps_the_cells_with_all_valid_corners(self, method):
        """The sum of the weights of valid corners is not lost by rounding errors"""
        n = 30
        lon, lat = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float))
        field = np.ones((n, n))
        field[10:15, 12:20] = np.nan
        rng = np.random.default_rng(0)
        plon = rng.uniform(1, n - 2.001, 2000)
        plat = rng.uniform(1, n - 2.001, 2000)
        j, i = plat.astype(int), plon.astype(int)
        valid = ~np.isnan(field)
        corners_ok = valid[j, i] & valid[j, i + 1] & valid[j + 1, i] & valid[j + 1, i + 1]
        out = XYInterpolator(
            {"lon": lon, "lat": lat}, plon, plat, method=method
        ).interp(field, skipna=True, na_thres=0)
        assert corners_ok.any() and (~corners_ok).any()
        np.testing.assert_array_equal(~np.isnan(out), corners_ok)

    def test_mask_is_equivalent_to_nan(self):
        lon, lat, plon, plat = GRIDS["regular"]
        mask = np.ones(lon.shape, bool)
        mask[5:8, 6:9] = False
        field = smooth(lon, lat)
        with_nan = np.where(mask, field, np.nan)
        by_mask = XYInterpolator({"lon": lon, "lat": lat, "mask": mask}, plon, plat).interp(
            field, skipna=True
        )
        by_nan = XYInterpolator({"lon": lon, "lat": lat}, plon, plat).interp(with_nan, skipna=True)
        np.testing.assert_allclose(by_mask, by_nan, equal_nan=True)


def test_no_hole_in_large_curvilinear_grids():
    """Points inside a large curvilinear grid are always found"""
    n = 400
    x, y = np.meshgrid(np.linspace(0, 10, n), np.linspace(0, 8, n))
    lon = x + 0.5 * np.sin(0.7 * y) + 0.05 * y
    lat = y + 0.4 * np.sin(0.6 * x) - 0.03 * x
    rng = np.random.default_rng(0)
    k = 20000
    ii, jj = rng.integers(2, n - 3, k), rng.integers(2, n - 3, k)
    a, b = rng.uniform(size=k), rng.uniform(size=k)
    plon = (
        (1 - a) * (1 - b) * lon[jj, ii]
        + a * (1 - b) * lon[jj, ii + 1]
        + a * b * lon[jj + 1, ii + 1]
        + (1 - a) * b * lon[jj + 1, ii]
    )
    plat = (
        (1 - a) * (1 - b) * lat[jj, ii]
        + a * (1 - b) * lat[jj, ii + 1]
        + a * b * lat[jj + 1, ii + 1]
        + (1 - a) * b * lat[jj + 1, ii]
    )
    interp = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, "bilinear")
    out = interp.interp(2.0 * lon - lat)
    assert not np.isnan(out).any()
    weights = interp.get_weights()
    np.testing.assert_array_equal(weights["j_base"], jj)
    np.testing.assert_array_equal(weights["i_base"], ii)
    np.testing.assert_allclose(weights["frac_a"], a, atol=1e-6)
    np.testing.assert_allclose(weights["frac_b"], b, atol=1e-6)


class TestDateline:
    """Grids and points on both sides of the dateline"""

    @staticmethod
    def unwrap(lon):
        return np.where(lon < 0, lon + 360.0, lon)

    @staticmethod
    def get_grid(kind):
        lons = np.array([174.0, 176.0, 178.0, 180.0, -178.0, -176.0, -174.0, -172.0])
        if kind == "rectangular":  # irregular steps
            lons = np.array([174.0, 175.5, 178.0, 180.0, -178.5, -176.0, -174.0, -172.5])
        lon, lat = np.meshgrid(lons, np.linspace(0.0, 6.0, 7))
        if kind == "curvilinear":
            lon = lon + 0.1 * lat
            lon = np.where(lon > 180.0, lon - 360.0, lon)
        return lon, lat

    @pytest.mark.parametrize(
        "kind,method",
        [
            ("regular", "bilinear"),
            ("regular", "bicubic"),
            ("rectangular", "bilinear"),
            pytest.param(
                "rectangular",
                "bicubic",
                marks=pytest.mark.xfail(
                    strict=True,
                    reason="Bicubic tangents ignore the irregular spacing of the grid",
                ),
            ),
            ("curvilinear", "bilinear"),
            ("curvilinear", "bicubic"),
        ],
    )
    def test_linear_field_is_exact_across_the_dateline(self, kind, method):
        lon, lat = self.get_grid(kind)
        points_lon = np.array([177.3, 179.0, 180.0, -179.0, -177.2, -175.4])
        points_lat = np.array([2.0, 2.5, 1.5, 3.0, 4.4, 3.3])
        interp = XYInterpolator({"lon": lon, "lat": lat}, points_lon, points_lat, method)
        out = interp.interp(2.0 * self.unwrap(lon) + lat)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(
            out, 2.0 * self.unwrap(points_lon) + points_lat, rtol=0, atol=1e-8
        )

    @pytest.mark.parametrize("kind", ["regular", "rectangular", "curvilinear"])
    def test_points_far_from_the_grid_are_nan(self, kind):
        lon, lat = self.get_grid(kind)
        interp = XYInterpolator(
            {"lon": lon, "lat": lat}, np.array([170.0, -160.0, 0.0]), np.array([2.0, 2.0, 2.0])
        )
        assert np.isnan(interp.interp(lat)).all()


class TestAxisOrientation:
    """Axes may increase or decrease, like latitudes that go from north to south"""

    @staticmethod
    def get_axes(regular):
        if regular:
            return np.linspace(0.0, 6.0, 13), np.linspace(0.0, 4.0, 9)
        return (
            np.array([0.0, 0.7, 1.5, 2.6, 3.0, 4.2, 5.0, 5.5, 6.0]),
            np.array([0.0, 0.5, 1.4, 2.0, 3.1, 3.4, 4.0]),
        )

    @pytest.mark.parametrize("regular", [True, False])
    @pytest.mark.parametrize("reverse_lon", [False, True])
    @pytest.mark.parametrize("reverse_lat", [False, True])
    def test_linear_field_is_exact_whatever_the_orientation(
        self, regular, reverse_lon, reverse_lat
    ):
        lons, lats = self.get_axes(regular)
        lons = lons[::-1] if reverse_lon else lons
        lats = lats[::-1] if reverse_lat else lats
        lon, lat = np.meshgrid(lons, lats)
        rng = np.random.default_rng(7)
        plon, plat = rng.uniform(0.2, 5.8, 200), rng.uniform(0.2, 3.8, 200)
        interp = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, "bilinear")
        assert interp.src_grid["type"] == ("regular" if regular else "rectangular")
        out = interp.interp(2.0 * lon - 3.0 * lat + 1.0)
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out, 2.0 * plon - 3.0 * plat + 1.0, rtol=0, atol=1e-9)

    @pytest.mark.parametrize("regular", [True, False])
    def test_points_on_the_edges_and_outside(self, regular):
        lons, lats = self.get_axes(regular)
        for lat_axis in lats, lats[::-1]:
            lon, lat = np.meshgrid(lons, lat_axis)
            points_lon = np.array(
                [lons[0], lons[-1], 2.0, 2.0, lons[0] - 0.1, lons[-1] + 0.1, 2.0, 2.0]
            )
            points_lat = np.array(
                [1.0, 1.0, lats[0], lats[-1], 1.0, 1.0, lats[0] - 0.1, lats[-1] + 0.1]
            )
            out = XYInterpolator({"lon": lon, "lat": lat}, points_lon, points_lat).interp(lon + lat)
            np.testing.assert_allclose(out[:4], points_lon[:4] + points_lat[:4], atol=1e-9)
            assert np.isnan(out[4:]).all()

    def test_bicubic_with_descending_axes(self):
        lons, lats = self.get_axes(True)
        lon, lat = np.meshgrid(lons[::-1], lats[::-1])
        rng = np.random.default_rng(8)
        plon, plat = rng.uniform(0.6, 5.4, 100), rng.uniform(0.6, 3.4, 100)
        out = XYInterpolator({"lon": lon, "lat": lat}, plon, plat, "bicubic").interp(
            2.0 * lon - lat
        )
        assert not np.isnan(out).any()
        np.testing.assert_allclose(out, 2.0 * plon - plat, atol=1e-9)
