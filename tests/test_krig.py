"""
Test the :mod:`xoa.krig` module
"""

import time

import numpy as np
import pytest
import xarray as xr

from xoa import krig as xkrig


def get_obs(n=30):
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(-5, -4, n), rng.uniform(47, 48, n)
    values = np.sin(5 * (lon + 5)) + np.cos(5 * (lat - 47))
    return xr.DataArray(
        values, dims="npts", coords={"lon": ("npts", lon), "lat": ("npts", lat)}, name="temp"
    )


@pytest.mark.parametrize("dist_units", ["m", "km"])
def test_krig_kriger_is_exact_at_observations(dist_units):
    """Same distance units for the input and output points, whatever they are"""
    obs = get_obs()
    scale = 1.0 if dist_units == "km" else 1e3
    func = lambda d: 1.0 - np.exp(-3 * d / (50 * scale))  # noqa: E731
    kriger = xkrig.Kriger(obs, "ordinary", func, dist_units=dist_units)
    target = xr.Dataset(coords={"lon": obs.lon, "lat": obs.lat})
    out = kriger.interp(target)
    np.testing.assert_allclose(out.temp.values, obs.values, atol=1e-6)


def test_krig_clustered_kriger_matches_kriger_without_split():
    obs = get_obs()
    func = lambda d: 1.0 - np.exp(-3 * d / 5e4)  # noqa: E731
    target = xr.Dataset(coords={"lon": np.linspace(-4.9, -4.1, 6), "lat": np.full(6, 47.5)})
    ref = xkrig.Kriger(obs, "ordinary", func).interp(target)
    out = xkrig.ClusteredKriger(obs, "ordinary", func, npmax=100, nproc=1).interp(target)
    np.testing.assert_allclose(out.temp.values, ref.temp.values)
    np.testing.assert_allclose(out.temp_error.values, ref.temp_error.values)


@pytest.mark.parametrize("nproc", [1, 2])
def test_krig_clustered_kriger_with_clusters(nproc):
    obs = get_obs()
    func = lambda d: 1.0 - np.exp(-3 * d / 5e4)  # noqa: E731
    kriger = xkrig.ClusteredKriger(obs, "ordinary", func, npmax=12, nproc=nproc)
    assert kriger.nclust > 1
    target = xr.Dataset(coords={"lon": obs.lon, "lat": obs.lat})
    out = kriger.interp(target)
    np.testing.assert_allclose(out.temp.values, obs.values, atol=1e-5)


def test_krig_krig_selects_kriger_class():
    obs = get_obs()
    func = lambda d: 1.0 - np.exp(-3 * d / 5e4)  # noqa: E731
    target = xr.Dataset(coords={"lon": obs.lon, "lat": obs.lat})
    out = xkrig.krig(obs, target, variogram_func=func, npmax=12)
    np.testing.assert_allclose(out.temp.values, obs.values, atol=1e-5)


def test_krig_block_kriging():
    """A block that covers all the targets gives the same value everywhere"""
    obs = get_obs()
    func = lambda d: 1.0 - np.exp(-3 * d / 5e4)  # noqa: E731
    target = xr.Dataset(coords={"lon": np.linspace(-4.9, -4.1, 6), "lat": np.full(6, 47.5)})
    out = xkrig.Kriger(obs, "ordinary", func).interp(target, block=10)
    assert out.temp.size == 36
    np.testing.assert_allclose(out.temp.values, out.temp.values.flat[0])


def test_krig_empirical_variogram_and_fit():
    obs = get_obs(60)
    ev = xkrig.empirical_variogram(obs, nbin=10, dist_units="km")
    assert ev.name == "semivariogram"
    assert ev.sizes["dist"] > 1
    model = xkrig.VariogramModel("spherical", dist_units="km")
    model.fit(ev)
    assert model["s"] > 0
    assert model["r"] > 0


def _time_(func, repeat=2):
    """Best time of `repeat` calls in seconds"""
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        func()
        best = min(best, time.perf_counter() - start)
    return best


def _get_perf_case_(n):
    obs = get_obs(n)
    func = lambda d: 1.0 - np.exp(-3 * d / 5e4)  # noqa: E731
    target = xr.Dataset(
        coords={"lon": np.linspace(-4.9, -4.1, 20), "lat": np.linspace(47.1, 47.9, 20)}
    )
    return obs, func, target


def test_krig_perf_clustered_kriger_is_faster_on_big_clouds(capsys):
    obs, func, target = _get_perf_case_(1600)
    t_full = _time_(lambda: xkrig.Kriger(obs, "ordinary", func).interp(target))
    t_clust = _time_(
        lambda: xkrig.ClusteredKriger(obs, "ordinary", func, npmax=400, nproc=1).interp(target)
    )
    with capsys.disabled():
        print(f"\nkriger={t_full:.3f}s clustered={t_clust:.3f}s")
    assert t_clust < t_full


def test_krig_perf_kriger_scaling(capsys):
    """Doubling the number of points must cost less than a cubic increase"""
    times = {}
    for n in (400, 800):
        obs, func, target = _get_perf_case_(n)
        times[n] = _time_(lambda: xkrig.Kriger(obs, "ordinary", func).interp(target))
    with capsys.disabled():
        print("\n" + " ".join(f"n={n}: {t:.3f}s" for n, t in times.items()))
    assert times[800] < 12 * times[400]


def get_model(**kwargs):
    return xkrig.VariogramModel("exponential", dist_units="km", n=0.0, s=1.0, r=50.0, **kwargs)


def get_grid_target():
    return xr.Dataset(coords={"lon": np.linspace(-4.9, -4.1, 8), "lat": np.linspace(47.1, 47.9, 8)})


@pytest.mark.parametrize("krigtype", ["ordinary", "simple"])
def test_krig_simple_and_ordinary_are_exact_at_observations(krigtype):
    obs = get_obs()
    target = xr.Dataset(coords={"lon": obs.lon, "lat": obs.lat})
    out = xkrig.Kriger(obs, krigtype, get_model(), dist_units="km").interp(target)
    np.testing.assert_allclose(out.temp.values, obs.values, atol=1e-5)
    np.testing.assert_allclose(out.temp_error.values, 0, atol=1e-5)


def test_krig_simple_kriging_goes_to_the_mean_far_from_data():
    obs = get_obs()
    target = xr.Dataset(coords={"lon": [20.0], "lat": [-40.0]})
    out = xkrig.Kriger(obs, "simple", get_model(), dist_units="km").interp(target)
    np.testing.assert_allclose(out.temp.values, float(obs.mean()))
    np.testing.assert_allclose(out.temp_error.values, 1.0)


def test_krig_kriger_with_a_dataset_input():
    obs = get_obs()
    out = xkrig.Kriger(obs.to_dataset(), "simple", get_model(), dist_units="km")
    assert np.isclose(out.mean, float(obs.mean()))


def test_krig_fitted_model_is_not_fitted_again():
    obs = get_obs(60)
    model = xkrig.VariogramModel("spherical", dist_units="km")
    model.fit(obs)
    params = model.params
    xkrig.Kriger(obs, "ordinary", model, dist_units="km")
    assert model.params == params


def test_krig_model_with_frozen_params_needs_no_fit():
    model = get_model()
    assert model.is_fitted
    model.fit(get_obs())
    assert model.params == {"n": 0.0, "s": 1.0, "r": 50.0}


def test_krig_empirical_variogram_big_sample_and_attrs():
    obs = get_obs(60)
    obs.attrs.update(long_name="Temperature", units="degC")
    ev = xkrig.empirical_variogram(obs, nbin=10, nmax=30)
    assert ev.attrs["units"] == "degC^2"
    assert "Temperature" in ev.attrs["long_name"]


def test_krig_empirical_variogram_keeps_pairs_at_same_distance():
    """Regular grid with a lot of pairs at the same distance: a single bin has all of them"""
    lon, lat = np.meshgrid(np.arange(6.0), np.arange(6.0))
    values = lon.ravel() ** 2
    obs = xr.DataArray(
        values, dims="npts", coords={"lon": ("npts", lon.ravel()), "lat": ("npts", lat.ravel())}
    )
    ev = xkrig.empirical_variogram(obs, nbin=1, nbin0=1)
    iu = np.triu_indices(obs.size, 1)
    expected = (0.5 * (values[:, None] - values[None]) ** 2)[iu].mean()
    np.testing.assert_allclose(ev.values, [expected])


def test_krig_variogram_model_fit_dataset_and_plot():
    import matplotlib

    matplotlib.use("Agg")
    obs = get_obs(60)
    model = xkrig.VariogramModel("spherical", dist_units="km")
    model.fit(obs.to_dataset())
    model.plot(rmax=30)
    assert model.is_fitted


@pytest.mark.parametrize("krigtype", ["ordinary", "simple"])
def test_krig_local_kriger_matches_kriger_with_all_neighbors(krigtype):
    obs = get_obs()
    target = get_grid_target()
    ref = xkrig.Kriger(obs, krigtype, get_model(), dist_units="km").interp(target)
    out = xkrig.LocalKriger(obs, krigtype, get_model(), nneigh=obs.size, dist_units="km").interp(
        target
    )
    np.testing.assert_allclose(out.temp.values, ref.temp.values, atol=1e-8)
    np.testing.assert_allclose(out.temp_error.values, ref.temp_error.values, atol=1e-8)


def test_krig_local_kriger_is_exact_at_observations_with_few_neighbors():
    obs = get_obs()
    target = xr.Dataset(coords={"lon": obs.lon, "lat": obs.lat})
    out = xkrig.LocalKriger(obs, "ordinary", get_model(), nneigh=8, dist_units="km").interp(target)
    np.testing.assert_allclose(out.temp.values, obs.values, atol=1e-5)


def test_krig_local_kriger_is_close_to_kriger_with_enough_neighbors():
    obs = get_obs()
    target = get_grid_target()
    ref = xkrig.Kriger(obs, "ordinary", get_model(), dist_units="km").interp(target)
    out = xkrig.LocalKriger(obs, "ordinary", get_model(), nneigh=30, dist_units="km").interp(target)
    assert float(abs(out.temp - ref.temp).max()) < 0.1


def test_krig_local_kriger_nan_without_enough_neighbors():
    obs = get_obs()
    target = xr.Dataset(
        coords={
            "lon": ("npts", [obs.lon.values[0], 20.0]),
            "lat": ("npts", [obs.lat.values[0], -40.0]),
        }
    )
    out = xkrig.LocalKriger(
        obs, "ordinary", get_model(), nneigh=8, maxdist=60.0, nmin=3, dist_units="km"
    ).interp(target)
    values = out.temp.values
    assert np.isfinite(values[0])
    assert np.isnan(values[1])


def test_krig_local_kriger_ignores_missing_data():
    obs = get_obs()
    bad = obs.copy()
    bad[:3] = np.nan
    target = get_grid_target()
    ref = xkrig.LocalKriger(obs[3:], "ordinary", get_model(), nneigh=10, dist_units="km")
    out = xkrig.LocalKriger(bad, "ordinary", get_model(), nneigh=10, dist_units="km")
    np.testing.assert_allclose(out.interp(target).temp.values, ref.interp(target).temp.values)


def test_krig_local_kriger_across_the_dateline():
    rng = np.random.default_rng(2)
    n = 40
    lon = rng.uniform(178, 182, n) % 360
    lat = rng.uniform(0, 4, n)
    obs = xr.DataArray(
        np.cos(np.deg2rad(lon)) + lat,
        dims="npts",
        coords={"lon": ("npts", lon), "lat": ("npts", lat)},
        name="temp",
    )
    target = xr.Dataset(coords={"lon": np.array([179.5, 180.5]), "lat": np.array([1.0, 2.0])})
    model = xkrig.VariogramModel("exponential", dist_units="km", n=0.0, s=1.0, r=300.0)
    ref = xkrig.Kriger(obs, "ordinary", model, dist_units="km").interp(target)
    out = xkrig.LocalKriger(obs, "ordinary", model, nneigh=n, dist_units="km").interp(target)
    np.testing.assert_allclose(out.temp.values, ref.temp.values, atol=1e-8)


def test_krig_local_kriger_with_several_slices():
    obs = get_obs()
    obs2 = xr.concat([obs, 2 * obs], dim="time")
    target = get_grid_target()
    out = xkrig.LocalKriger(obs2, "ordinary", get_model(), nneigh=10, dist_units="km").interp(
        target
    )
    ref = xkrig.LocalKriger(obs, "ordinary", get_model(), nneigh=10, dist_units="km").interp(target)
    assert out.temp.dims[0] == "time"
    np.testing.assert_allclose(out.temp.isel(time=1).values, 2 * ref.temp.values)
    np.testing.assert_allclose(out.temp.isel(time=0).values, ref.temp.values)


def test_krig_local_kriger_same_result_with_one_thread():
    import numba

    obs = get_obs()
    target = get_grid_target()
    kriger = xkrig.LocalKriger(obs, "ordinary", get_model(), nneigh=10, dist_units="km")
    ref = kriger.interp(target)
    nthreads = numba.get_num_threads()
    numba.set_num_threads(1)
    try:
        out = kriger.interp(target)
    finally:
        numba.set_num_threads(nthreads)
    np.testing.assert_array_equal(out.temp.values, ref.temp.values)


def test_krig_local_kriger_needs_a_variogram_model():
    with pytest.raises(xkrig.exceptions.XoaKrigingError):
        xkrig.LocalKriger(get_obs(), "ordinary", lambda d: d)


def test_krig_krig_selects_local_kriger():
    obs = get_obs()
    out = xkrig.krig(obs, get_grid_target(), variogram_func=get_model(), nneigh=10, dist_units="km")
    assert np.isfinite(out.temp.values).all()


def test_krig_perf_local_kriger_is_faster_than_kriger(capsys):
    obs, func, target = _get_perf_case_(1600)
    model = xkrig.VariogramModel("exponential", dist_units="m", n=0.0, s=1.0, r=5e4)
    xkrig.LocalKriger(obs, "ordinary", model, nneigh=16).interp(target)  # compile
    t_full = _time_(lambda: xkrig.Kriger(obs, "ordinary", model).interp(target))
    t_local = _time_(lambda: xkrig.LocalKriger(obs, "ordinary", model, nneigh=16).interp(target))
    with capsys.disabled():
        print(f"\nkriger={t_full:.3f}s local={t_local:.3f}s")
    assert t_local < t_full


def test_krig_solver_uses_cholesky_for_covariance_matrices():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(30, 30))
    a = m @ m.T + 30 * np.eye(30)
    b = rng.normal(size=(30, 4))
    solver = xkrig.ckrig.MatrixSolver(a, spd=True)
    assert solver._cho is not None
    np.testing.assert_allclose(a @ solver.solve(b), b, atol=1e-10)


def test_krig_solver_uses_lu_for_indefinite_matrices():
    a = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])  # not positive
    solver = xkrig.ckrig.MatrixSolver(a, spd=True)  # cholesky fails
    assert solver._cho is None and solver._lu is not None
    b = np.arange(3.0)
    np.testing.assert_allclose(a @ solver.solve(b), b, atol=1e-12)


def test_krig_solver_falls_back_to_pinv_for_singular_matrices():
    a = np.array([[1.0, 1.0], [1.0, 1.0]])
    solver = xkrig.ckrig.MatrixSolver(a)
    assert solver._lu is None and solver._pinv is not None
    np.testing.assert_allclose(solver.solve(np.array([2.0, 2.0])), [1.0, 1.0])


def test_krig_solver_falls_back_to_pinv_for_ill_conditioned_matrices():
    """A gaussian variogram without nugget"""
    lon = np.linspace(0, 1, 60)
    d = np.abs(lon[:, None] - lon[None])
    a = 1 - xkrig.ckrig.variogram(d, xkrig.variogram_model_types.gaussian.value, 0.0, 1.0, 0.5)
    assert xkrig.ckrig.MatrixSolver(a, spd=True)._pinv is not None


def test_krig_kriger_with_duplicate_points_and_no_nugget():
    obs = get_obs(20)
    dup = xr.concat([obs, obs[:3]], dim="npts")
    target = xr.Dataset(coords={"lon": obs.lon, "lat": obs.lat})
    out = xkrig.Kriger(dup, "ordinary", get_model(), dist_units="km", exact=True).interp(target)
    assert np.isfinite(out.temp.values).all()
    np.testing.assert_allclose(out.temp.values, obs.values, atol=1e-5)


def test_krig_variogram_fit_has_a_nugget_floor():
    obs = get_obs(60)
    model = xkrig.VariogramModel("gaussian", dist_units="km")
    model.fit(obs)
    assert model["n"] >= xkrig.NUGGET_MIN * float(model["s"]) * 0.5


KRIGERS = {
    "kriger": (xkrig.Kriger, {}),
    "clustered": (xkrig.ClusteredKriger, {"npmax": 1000, "nproc": 1}),  # no kmeans, which is random
    "local": (xkrig.LocalKriger, {"nneigh": 10}),
}
LON_ATTRS = {"units": "degrees_east", "long_name": "Longitude"}
LAT_ATTRS = {"units": "degrees_north", "long_name": "Latitude"}


def get_attrs_obs(n=40):
    obs = get_obs(n)
    obs = obs.assign_coords(
        lon=("npts", obs.lon.values, LON_ATTRS), lat=("npts", obs.lat.values, LAT_ATTRS)
    )
    obs = xr.concat([obs, 2 * obs], dim="time").assign_coords(
        time=("time", [0, 1], {"long_name": "Time axis"}), depth=5.0
    )
    obs.attrs.update(units="degC", long_name="Temperature", source="test")
    return obs.rename("temp")


@pytest.mark.parametrize("name", list(KRIGERS))
def test_krig_output_is_on_the_target_grid_with_input_attributes(name):
    cls, kw = KRIGERS[name]
    obs = get_attrs_obs()
    lon = xr.DataArray([-4.5, -4.9, -4.2], dims="lon", attrs=LON_ATTRS)  # not sorted
    lat = xr.DataArray([47.9, 47.1, 47.5, 47.3], dims="lat", attrs=LAT_ATTRS)
    out = cls(obs, "ordinary", get_model(), dist_units="km", **kw).interp(
        xr.Dataset(coords={"lon": lon, "lat": lat})
    )
    assert isinstance(out, xr.Dataset)
    assert out.temp.dims == ("time", "lat", "lon") and out.temp_error.dims == ("lat", "lon")
    # Target coordinates are kept as they are, with their attributes
    np.testing.assert_array_equal(out.lon, lon)
    np.testing.assert_array_equal(out.lat, lat)
    assert out.lon.attrs == LON_ATTRS and out.lat.attrs == LAT_ATTRS
    # Input attributes and non geographic coordinates are kept
    assert out.temp.attrs == obs.attrs
    assert out.temp_error.attrs["units"] == "degC^2"
    assert out.time.attrs == {"long_name": "Time axis"} and float(out.depth) == 5.0
    # Values are at the right place
    point = xr.Dataset(coords={"lon": ("npts", [-4.2]), "lat": ("npts", [47.3])})
    ref = cls(obs, "ordinary", get_model(), dist_units="km", **kw).interp(point)
    np.testing.assert_allclose(
        out.temp.sel(lon=-4.2, lat=47.3).values, ref.temp.isel(npts=0).values, atol=1e-8
    )


@pytest.mark.parametrize("name", list(KRIGERS))
def test_krig_output_on_points_and_curvilinear_and_scalar_targets(name):
    cls, kw = KRIGERS[name]
    obs = get_attrs_obs()
    kriger = cls(obs, "ordinary", get_model(), dist_units="km", **kw)

    pts = xr.Dataset(
        coords={
            "lon": ("pts", [-4.5, -4.4], LON_ATTRS),
            "lat": ("pts", [47.5, 47.6], LAT_ATTRS),
            "id": ("pts", ["a", "b"]),
        }
    )
    out = kriger.interp(pts)
    assert out.temp.dims == ("time", "pts") and list(out.id.values) == ["a", "b"]
    assert out.lon.attrs == LON_ATTRS

    lon, lat = np.meshgrid([-4.8, -4.6, -4.4], [47.2, 47.4])
    cur = xr.Dataset(
        coords={
            "lon_rho": (("eta", "xi"), lon, LON_ATTRS),
            "lat_rho": (("eta", "xi"), lat, LAT_ATTRS),
        }
    )
    out = kriger.interp(cur)
    assert out.temp.dims == ("time", "eta", "xi")
    np.testing.assert_array_equal(out.lon_rho, lon)

    out = kriger.interp(xr.Dataset(coords={"lon": -4.5, "lat": 47.5}))
    assert out.temp.dims == ("time",) and out.temp_error.dims == ()
    assert float(out.lon) == -4.5 and float(out.lat) == 47.5


@pytest.mark.parametrize("name", list(KRIGERS))
def test_krig_input_forms(name):
    """Datasets, unnamed arrays, other coordinate names and grids as inputs"""
    cls, kw = KRIGERS[name]
    obs = get_obs(40)
    target = get_grid_target()
    ref = cls(obs, "ordinary", get_model(), dist_units="km", **kw).interp(target).temp.values

    def interp(da, **kws):
        return cls(da, "ordinary", get_model(), dist_units="km", **kw).interp(target, **kws)

    np.testing.assert_allclose(interp(obs.to_dataset()).temp.values, ref)
    out = interp(obs.rename(None))
    assert "data" in out and np.allclose(out.data.values, ref)
    renamed = obs.rename(lon="longitude", lat="latitude")
    assert np.allclose(interp(renamed).temp.values, ref)
    assert "other" in interp(obs, name="other")
    # Target as a data array
    da_target = xr.DataArray(
        np.zeros((8, 8)), dims=("lat", "lon"), coords={"lon": target.lon, "lat": target.lat}
    )
    assert np.allclose(interp(obs).temp.values, interp(obs).temp.values)
    assert np.allclose(
        cls(obs, "ordinary", get_model(), dist_units="km", **kw).interp(da_target).temp.values, ref
    )


def test_krig_input_on_a_grid_equals_the_same_input_as_points():
    lon, lat = np.meshgrid(np.linspace(-4.9, -4.1, 6), np.linspace(47.1, 47.9, 5))
    values = np.sin(5 * (lon + 5)) + np.cos(5 * (lat - 47))
    grid = xr.DataArray(
        values,
        dims=("lat", "lon"),
        coords={"lon": ("lon", lon[0]), "lat": ("lat", lat[:, 0])},
        name="temp",
    )
    points = xr.DataArray(
        values.ravel(),
        dims="npts",
        coords={"lon": ("npts", lon.ravel()), "lat": ("npts", lat.ravel())},
        name="temp",
    )
    target = xr.Dataset(coords={"lon": [-4.55, -4.35], "lat": [47.45, 47.65]})
    out = xkrig.LocalKriger(grid, "ordinary", get_model(), nneigh=10, dist_units="km").interp(
        target
    )
    ref = xkrig.LocalKriger(points, "ordinary", get_model(), nneigh=10, dist_units="km").interp(
        target
    )
    np.testing.assert_allclose(out.temp.values, ref.temp.values)


def get_random_fields(nobs=100, ntgt=60, nreal=40, sill=1.0, rng_km=40.0):
    """Gaussian random fields with an exponential covariance, at observations and targets"""
    rng = np.random.default_rng(0)
    lon = rng.uniform(-5, -4, nobs + ntgt)
    lat = rng.uniform(47, 48, nobs + ntgt)
    d = (
        1e-3
        * xkrig.xgeo.EARTH_RADIUS
        * xkrig.ckrig.haversine(lon[:, None], lat[:, None], lon[None], lat[None])
    )
    chol = np.linalg.cholesky(sill * np.exp(-3 * d / rng_km) + 1e-9 * np.eye(nobs + ntgt))
    fields = chol @ rng.normal(size=(nobs + ntgt, nreal))
    pos = lambda sl: {"lon": ("npts", lon[sl]), "lat": ("npts", lat[sl])}  # noqa: E731
    return pos(slice(nobs)), xr.Dataset(coords=pos(slice(nobs, None))), fields[:nobs], fields[nobs:]


@pytest.mark.parametrize(
    "name,krigtype,low,high",
    [
        ("kriger", "ordinary", 0.85, 1.15),
        ("kriger", "simple", 0.85, 1.15),
        ("local", "ordinary", 0.85, 1.15),
        ("local", "simple", 0.85, 1.15),
        ("clustered", "ordinary", 0.6, 1.0),  # conservative
    ],
)
def test_krig_error_is_calibrated(name, krigtype, low, high):
    """The squared errors are, on average, close to the kriging variance"""
    cls, kw = KRIGERS[name]
    if name == "clustered":
        kw = {"npmax": 50, "nproc": 1}
    pos, target, zobs, ztgt = get_random_fields()
    model = xkrig.VariogramModel("exponential", dist_units="km", n=0.0, s=1.0, r=40.0)
    ratios = []
    for i in range(zobs.shape[1]):
        obs = xr.DataArray(zobs[:, i], dims="npts", coords=pos, name="t")
        kriger = cls(
            obs,
            krigtype,
            model,
            dist_units="km",
            **{**kw, **({"mean": 0.0} if krigtype == "simple" else {})},
        )
        out = kriger.interp(target)
        ratios.append(np.mean((out.t.values - ztgt[:, i]) ** 2 / out.t_error.values))
    assert low < np.mean(ratios) < high


def test_krig_clustered_kriger_is_reproducible_with_a_seed():
    obs = get_obs(60)
    target = get_grid_target()

    def run(seed):
        kriger = xkrig.ClusteredKriger(
            obs, "ordinary", get_model(), dist_units="km", npmax=15, nproc=1, seed=seed
        )
        return kriger.interp(target).temp.values

    np.testing.assert_array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))
