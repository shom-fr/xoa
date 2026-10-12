"""
Test the :mod:`xoa.stats` module
"""

import numpy as np
import pytest
import xarray as xr

from xoa import grid as xgrid
from xoa import stats as xstats
from xoa.exceptions import XoaError
from xoa.stats import StatAccum


def get_data():
    ref = xr.DataArray(np.arange(12.0).reshape(3, 4), dims=("y", "x"), name="ref")
    da = ref + xr.DataArray([1.0, -1.0, 2.0], dims="y")
    return da, ref


def test_rms():
    da, ref = get_data()
    assert float(xstats.rms(da, ref)) == pytest.approx(np.sqrt((1 + 1 + 4) / 3))
    np.testing.assert_allclose(xstats.rms(da, ref, dim="x"), [1, 1, 2])
    assert float(xstats.rms(da - ref)) == float(xstats.rms(da, ref))


def test_rms_mask_and_nan():
    da, ref = get_data()
    mask = xr.DataArray([True, False, True], dims="y")
    assert float(xstats.rms(da, ref, mask=mask)) == pytest.approx(np.sqrt(5 / 2))
    da[0, 0] = np.nan
    assert np.isfinite(xstats.rms(da, ref))


def test_rms_by_region():
    da, ref = get_data()
    regions = {
        "all": xr.ones_like(da, dtype=bool),
        "first": xr.DataArray([True, False, False], dims="y"),
    }
    res = xstats.rms(da, ref, by=regions)
    assert res.dims == ("region",)
    assert list(res.region.values) == ["all", "first"]
    np.testing.assert_allclose(res, [np.sqrt(2), 1.0])
    with pytest.raises(XoaError):
        xstats.rms(da, by={})


# Statistics accumulator
# ----------------------
# Sigma-like layers whose total thickness h varies in space and time, with
# z = -h * (k + 0.5) / nz and the linear field a * z, hence analytical statistics.

NT, NZ, NX, A = 6, 8, 5, 3.0


def get_sigma_data():
    time = np.array("2020-01-01", dtype="M8[D]") + np.arange(NT)
    h0 = xr.DataArray(10.0 + 7.0 * np.arange(NX) ** 2, dims="x")
    h = h0 * xr.DataArray(1 + 0.1 * np.arange(NT), dims="time", coords={"time": time})
    k = xr.DataArray(np.arange(NZ) + 0.5, dims="z")
    z = -h * k / NZ
    dz = (h / NZ + 0 * k).transpose("time", "z", "x")
    return h0, h.transpose("time", "x"), z.transpose("time", "z", "x"), dz


def test_stataccum_vertical_weighted_analytic():
    _, h, z, dz = get_sigma_data()
    sa = StatAccum(temporal=False, spatial=True, sdims="z", weights=dz).append(A * z)
    stats = sa.get_stats()
    assert stats.smean.dims == ("time", "x")
    assert np.allclose(stats.smean, -A * h / 2)
    assert np.allclose(stats.svar, A**2 * h**2 * (NZ**2 - 1) / (12 * NZ**2))
    assert np.allclose(stats.smin, A * z.isel(z=-1) * 1.0)
    assert np.all(stats.scount == NZ) and np.all(stats.savail == 1)


def test_stataccum_volume_weighted_analytic():
    _, h, z, dz = get_sigma_data()
    stats = StatAccum(temporal=False, spatial=True, weights=dz).append(A * z).get_stats()
    h2, h3, h1 = (h**2).sum("x"), (h**3).sum("x"), h.sum("x")
    mean = -A * h2 / (2 * h1)
    var = A**2 * h3 * (4 * NZ**2 - 1) / (12 * NZ**2 * h1) - mean**2
    assert stats.smean.dims == ("time",)
    assert np.allclose(stats.smean, mean)
    assert np.allclose(stats.svar, var)
    # Without weights, thin columns count as much as thick ones
    unweighted = StatAccum(temporal=False, spatial=True, weights=False).append(A * z).get_stats()
    assert not np.allclose(unweighted.smean, mean)


def test_stataccum_temporal_analytic():
    h0, _, z, dz = get_sigma_data()
    f = 1 + 0.1 * np.arange(NT)
    zk = (-h0 * (xr.DataArray(np.arange(NZ) + 0.5, dims="z") / NZ)).transpose("z", "x")
    stats = StatAccum(weights=dz).append(A * z).get_stats()
    assert stats.tmean.dims == ("z", "x")
    assert np.allclose(stats.tmean, A * zk * f.mean())
    assert np.allclose(stats.tstd, abs(A * zk) * f.std())
    assert np.allclose(stats.tmin, A * zk * f.max())
    assert np.allclose(stats.tmax, A * zk * f.min())
    assert np.all(stats.tavail == 1)
    assert "smean" not in stats


def test_stataccum_comparison_analytic():
    _, h, z, dz = get_sigma_data()
    da0 = A * z
    da1 = 2 * da0 + 3
    stats = StatAccum(spatial=True, sdims="z", weights=dz).append(da0, da1).get_stats()
    mean0 = -A * h / 2
    assert np.allclose(stats.sbias, mean0 + 3)
    assert np.allclose(stats.scorr, 1)
    assert np.allclose(stats.scov, 2 * stats.svar0)
    assert np.allclose(stats.scrms, stats.sstd0)
    assert np.allclose(stats.srms, np.sqrt(stats.scrms**2 + stats.sbias**2))
    assert np.allclose(stats.tcorr, 1)
    assert np.allclose(stats.tbias, (da1 - da0).mean("time"))
    assert "smean1" in stats and "smean" not in stats


def test_stataccum_select_and_ddof():
    _, _, z, dz = get_sigma_data()
    sa = StatAccum(spatial=True, sdims="z", weights=dz).append(A * z)
    assert set(sa.get_stats("tmean")) == {"tmean"}
    assert set(sa.get_stats(["smean", "count"])) == {"smean", "tcount", "scount"}
    assert np.allclose(sa.get_stats("tvar", ddof=1).tvar, (A * z).var("time", ddof=1) * 1.0)
    with pytest.raises(XoaError):
        sa.get_stats("bad")


def test_stataccum_missing_values():
    _, _, z, dz = get_sigma_data()
    da = (A * z).copy()
    da[:, -2:, :] = np.nan
    stats = StatAccum(spatial=True, sdims="z", weights=dz).append(da).get_stats()
    assert np.all(stats.scount == NZ - 2)
    assert np.allclose(stats.savail, (NZ - 2) / NZ)
    assert np.allclose(stats.smin, da.min("z"))


def test_stataccum_incremental_and_restart(tmp_path):
    _, _, z, dz = get_sigma_data()
    da, da1 = A * z, A * z + 1
    kw = {"spatial": True, "sdims": ("z", "x"), "weights": dz}
    full = StatAccum(**kw).append(da, da1).get_stats()

    # Step by step, with a stop and a restart in the middle
    path = tmp_path / "accum.nc"
    sa = StatAccum(**kw)
    for it in range(3):
        sa += da.isel(time=[it]), da1.isel(time=[it])
    sa.dump(path)
    sa = StatAccum.from_file(path, weights=dz)
    assert sa.nt == 3 and sa.tdim == "time" and sa.dual
    sa.append(da.isel(time=slice(3, None)), da1.isel(time=slice(3, None)))
    xr.testing.assert_allclose(sa.get_stats(), full)

    # Replayed steps
    with pytest.raises(XoaError):
        sa.append(da.isel(time=[-1]), da1.isel(time=[-1]))
    sa.on_overlap = "skip"
    sa.append(da, da1)
    assert sa.nt == NT
    xr.testing.assert_allclose(sa.get_stats(), full)


def test_stataccum_errors(tmp_path):
    _, _, z, dz = get_sigma_data()
    with pytest.raises(XoaError):
        StatAccum().dump(tmp_path / "x.nc")
    with pytest.raises(XoaError):
        StatAccum(spatial=True, sdims="y").append(z)
    with pytest.raises(XoaError):
        StatAccum(on_overlap="bad")
    sa = StatAccum().append(z)
    with pytest.raises(XoaError):
        sa.append(z.isel(time=[0]), z.isel(time=[0]))
    with pytest.raises(XoaError):
        StatAccum(spatial=True, weights=dz.isel(z=slice(2))).append(z)


def test_stataccum_attributes(tmp_path):
    _, _, z, dz = get_sigma_data()
    temp = (A * z).rename("temp")
    temp.attrs.update(
        long_name="Sea temperature", units="degC", standard_name="sea_water_temperature"
    )
    ref = (temp + 1).rename("temp_ref")
    ref.attrs.update(long_name="Reference temperature", units="degC")
    sa = StatAccum(spatial=True, sdims="z", weights=dz).append(temp, ref)

    def check(stats):
        assert stats.tmean0.attrs["long_name"] == "Temporal mean of Sea temperature"
        assert stats.tmean0.attrs["units"] == "degC"
        assert stats.tmean0.attrs["standard_name"] == "sea_water_temperature"
        assert stats.tmean0.attrs["cell_methods"] == "time: mean"
        assert stats.smean1.attrs["long_name"] == "Spatial mean of Reference temperature"
        assert stats.smean1.attrs["cell_methods"] == "z: mean"
        assert stats.svar0.attrs["units"] == "(degC)2"
        assert stats.sbias.attrs["units"] == "degC"
        assert stats.sbias.attrs["long_name"] == (
            "Spatial bias of Sea temperature and Reference temperature"
        )
        assert "units" not in stats.scorr.attrs and "units" not in stats.tcount.attrs

    check(sa.get_stats())
    sa.dump(tmp_path / "accum.nc")
    restored = StatAccum.from_file(tmp_path / "accum.nc", weights=dz)
    assert restored.names == ["temp", "temp_ref"]
    check(restored.get_stats())

    # Single variable and no attributes
    stats = StatAccum().append(z.rename("zz")).get_stats()
    assert stats.tmean.attrs == {"long_name": "Temporal mean of zz", "cell_methods": "time: mean"}


def test_stataccum_automatic_vertical_weights():
    # Same as the explicit weights, but only z is known, and it varies in space and time
    _, h, z, _ = get_sigma_data()
    da = (A * z).rename(z="nz").assign_coords(z=z.rename(z="nz").rename("z"))
    da.coords["z"].attrs["positive"] = "up"
    stats = StatAccum(temporal=False, spatial=True, sdims="nz").append(da).get_stats()
    assert np.allclose(stats.smean, -A * h / 2)
    assert np.allclose(stats.svar, A**2 * h**2 * (NZ**2 - 1) / (12 * NZ**2))
    with pytest.warns(UserWarning, match="not weighted"):
        stats = StatAccum(temporal=False, spatial=True, sdims=("nz", "x")).append(da).get_stats()
    h2, h3, h1 = (h**2).sum("x"), (h**3).sum("x"), h.sum("x")
    mean = -A * h2 / (2 * h1)
    assert np.allclose(stats.smean, mean)
    assert np.allclose(stats.svar, A**2 * h3 * (4 * NZ**2 - 1) / (12 * NZ**2 * h1) - mean**2)

    # No z
    with pytest.raises(XoaError):
        StatAccum(spatial=True, sdims="z").append(A * z)
    # Disabled
    stats = StatAccum(temporal=False, spatial=True, sdims="z", weights=False).append(A * z)
    assert np.allclose(stats.get_stats().smean, (A * z).mean("z"))


def test_stataccum_automatic_horizontal_weights():
    lon = xr.DataArray(np.linspace(0, 10, 11), dims="x", attrs={"units": "degrees_east"})
    lat = xr.DataArray(np.linspace(0, 60, 13), dims="y", attrs={"units": "degrees_north"})
    time = np.array("2020-01-01", dtype="M8[D]") + np.arange(3)
    da = xr.DataArray(
        np.ones((3, 13, 11)) * lat.values[:, None],
        dims=("time", "y", "x"),
        coords={"time": time, "lon": lon, "lat": lat},
        name="lat",
    )
    stats = StatAccum(temporal=False, spatial=True).append(da).get_stats()
    dx, dy = xgrid.get_dx_dy(da)
    weights = (dx * dy).values
    expected = np.average(da.values[0], weights=weights)
    assert np.allclose(stats.smean, expected)
    assert not np.allclose(expected, lat.mean())

    # Only the reduced dimensions are weighted: the zonal mean is not
    stats = StatAccum(temporal=False, spatial=True, sdims="x").append(da).get_stats()
    assert np.allclose(stats.smean, da.mean("x"))

    # Without coordinates, with a warning
    with pytest.warns(UserWarning):
        stats = StatAccum(temporal=False, spatial=True).append(da.drop_vars(["lon", "lat"]))
    assert np.allclose(stats.get_stats().smean, da.mean(("y", "x")))
