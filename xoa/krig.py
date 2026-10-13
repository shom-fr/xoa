# -*- coding: utf8 -*-
"""
Kriging adapted from vacumm's module
https://github.com/VACUMM/vacumm/blob/master/lib/python/vacumm/misc/grid/kriging.py

"""

from functools import partial
from multiprocessing import Pool, cpu_count
import warnings

import numpy as np
import xarray as xr

from . import exceptions
from . import misc
from . import geo as xgeo
from . import coords as xcoords
from .core import krig as ckrig
from .core.geo import SphericalKDTree

#: Minimal nugget, relative to the sill, when fitting a variogram model
NUGGET_MIN = 1e-6


def get_xyz(obj):
    """Get lon/lat coordinates and data values from a data array or dataset

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        If a data array, it must have valid longitude and latitude coordinates.
        If a dataset, it must have a single variable as in the data array case.

    Return
    ------
    numpy.array
        Longitudes as 1D array
    numpy.array
        Latitudes as 1D array
    numpy.array
        Values as a 1D or 2D. None if `obj` is a dataset.
    """
    # Xarray stuff
    obj = xcoords.geo_stack(obj, "npts")
    lon = xcoords.get_lon(obj)
    lat = xcoords.get_lat(obj)

    # Numpy
    x = obj.coords[lon.name].values
    y = obj.coords[lat.name].values
    if isinstance(obj, xr.DataArray):
        z = obj.values.reshape(-1, x.size)
    else:
        z = None

    return x, y, z


def empirical_variogram(
    da: xr.DataArray,
    nbin=30,
    nbin0=10,
    nmax=1500,
    dist_units="m",
    distmax=None,
    errfunc=None,
):
    """Compute the semi-variogram from data

    Parameters
    ----------

    da: xarray.DataArray
        Data array with lon and lat coordinates.
    nbin: optional
        Number of bins of pairs of points sorted by distance.
    nbin0: optional
        If greater than 1, the first bin is split into `nbin0` sub-bins.
    nmax: optional
        Above this number, size of the sample is reduced by a crude undersampling.
    dist_units: str, int, xoa.geo.distance_units
        Distance units as one of: {xgeo.distance_units.rst_with_links}
    distmax: optional
        Max distance to consider.
    errfunc: optional
        Callable function to compute "errors" like square
        root difference between to z values. It take two arguments and
        defaults to :math:`(z1-z0)^2/2`.

    Return
    ------
    xarray.DataArray
        Values as 1D array with name "semivariogram" and with "dist" as distance coordinate in km

    """
    da = xcoords.geo_stack(da, "npts", rename=True, reset_index=True)
    npts = da.sizes["npts"]

    # Undepsample?
    if npts > nmax:
        samp = int(np.ceil(npts / nmax))
        da = da.isel(npts=slice(None, None, samp))
        npts = da.sizes["npts"]

    # Distances
    dist_units = xgeo.distance_units[dist_units]
    dd = xgeo.get_distances(da).values
    if dist_units == xgeo.distance_units.km:
        dd *= 1e-3
    iitriu = np.triu_indices(dd.shape[0], 1)
    d = dd[iitriu]
    del dd

    # Max distance
    if distmax:
        iiclose = d <= distmax
        d = d[iiclose]
        # v = v[valid]
        # del valid
    else:
        iiclose = ...

    # Variogram
    if errfunc is None:

        def errfunc(a0, a1):
            return 0.5 * (a1 - a0) ** 2

    z = np.atleast_2d(da.values)
    v = np.asarray([errfunc(*np.meshgrid(z[i], z[i]))[iitriu][iiclose] for i in range(z.shape[0])])

    # Sort
    iisort = np.argsort(d, kind="stable")
    d = d[iisort]
    v = v[:, iisort]

    # Compute edges
    # - classic bins
    nbin = min(d.size, nbin)
    iiedges = np.linspace(0, d.size - 1, nbin + 1).astype('l').tolist()
    # - more details in the first bin
    if nbin0 > 1 and nbin0 < iiedges[1]:  # split first bin
        iiedges = np.linspace(0.0, iiedges[1], nbin0 + 1).astype('l')[:-1].tolist() + iiedges[1:]
        nbin = nbin - 1 + nbin0  # len(iiedges)-1

    # Compute histogram
    db = np.empty(nbin)
    vb = np.empty(nbin)
    for ib in range(nbin):
        iib = slice(iiedges[ib], iiedges[ib + 1] + 1)
        db[ib] = d[iib].mean()
        vb[ib] = v[:, iib].mean()

    # Dataarray
    # dist = xr.DataArray(db, dims="dist", attrs={'long_name': "Distance", "units": "km"})
    attrs = {}
    if "long_name" in da.attrs:
        attrs = {"long_name": "Semi-variogram of " + da.attrs["long_name"]}
    else:
        attrs = {"long_name": "Semi-variogram"}
    if "units" in da.attrs:  # TODO: pint support
        attrs["units"] = da.attrs["units"] + "^2"
    return xr.DataArray(
        vb,
        dims="dist",
        coords={"dist": ("dist", db, {"long_name": "Distance", "units": str(dist_units)})},
        attrs=attrs,
        name="semivariogram",
    )


empirical_variogram.__doc__ = empirical_variogram.__doc__.format(**locals())


class variogram_model_types(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported types of variograms"""

    #: Exponential (default)
    exponential = 1
    #: Linear
    linear = 0
    #: Gaussian
    gaussian = 2
    #: Spherical
    spherical = 3


def get_variogram_model_func(mtype, n, s, r, nrelmax=0.2):
    """Get the variogram model function from its name"""
    code = variogram_model_types[mtype].value
    n, s, r = ckrig.clamp_variogram_params(n, s, r, nrelmax)
    return lambda h: ckrig.variogram(h, code, n, s, r)


class VariogramModel(object):
    """Class used when fitting a variogram model to data to better control params

    Parameters
    ----------
    mtype: int, str, variogram_model_types
    dist_units: int, str, xoa.geo.distance_units
    **frozen_params:
        Variogram parameters that must be frozen.

    """

    param_names = ['n', 's', 'r']

    def __init__(self, mtype, dist_units="m", **frozen_params):
        self._dist_units = xgeo.distance_units[dist_units]
        self.mtype = variogram_model_types[mtype]
        self._frozen_params = {}
        self._estimated_params = {}
        self._fit = None
        self._fit_err = None
        self.set_params(**frozen_params)
        self._ev = None

    def __str__(self):
        clsname = self.__class__.__name__
        mtype = self.mtype.name
        dist_units = self._dist_units
        sp = []
        for name in self.param_names:
            sp.append("{}={}".format(name, self[name]))
        sp = ', '.join(sp)
        return f"<{clsname}('{mtype}', dist_units='{dist_units}', {sp})>"

    def __repr__(self):
        return str(self)

    @property
    def dist_units(self):
        """Distance units of type :class:`~xoa.geo.distance_units`"""
        return self._dist_units

    @property
    def frozen_params(self):
        """Frozen parameters"""
        return dict(
            (name, self._frozen_params[name])
            for name in self.param_names
            if name in self._frozen_params
        )

    def get_estimated_params(self):
        """Get parameters that were estimated (not frozen)"""
        return dict(
            (name, self._estimated_params.get(name))
            for name in self.param_names
            if name not in self._frozen_params
        )

    def set_estimated_params(self, overwrite=True, **params):
        """Set the value of non-frozen parameters"""
        params_update = dict(
            (name, params[name])
            for name in self.param_names
            if name not in self._frozen_params
            and name in params
            and (not overwrite or name not in self._estimated_params)
        )
        self._estimated_params.update(params_update)

    estimated_params = property(
        get_estimated_params, set_estimated_params, doc='Estimated parameters as :class:`dict`'
    )

    def set_params(self, **params):
        """Freeze some parameters"""
        params = dict(
            [(p, v) for (p, v) in params.items() if p in self.param_names and v is not None]
        )
        self._frozen_params.update(params)

    def get_params(self, **params):
        """Get current parameters with optional update

        Parameters
        ----------
        params:
            Extra parameters to alter currents values

        Return
        ------
        dict, numpy.array
        """
        these_params = dict(**self.frozen_params, **self.estimated_params)
        if params:
            these_params.update(
                dict(
                    [(p, v) for (p, v) in params.items() if p in self.param_names and v is not None]
                )
            )
        return dict((name, these_params[name]) for name in self.param_names)

    def get_param(self, name):
        """Get a single parameter

        Parameters
        ----------
        name: str
            A valid parameter name

        Return
        ------
        float, None
            Returns None if the parameter is not frozen and has not been estimated yet.
        """
        if name not in self.param_names:
            raise exceptions.KrigingError(
                f"Invalid param name: {name}. Please use one of: " + ", ".join(self.param_names)
            )
        if name in self._frozen_params:
            return self._frozen_params[name]
        return self._estimated_params.get(name)

    __getitem__ = get_param

    def get_params_array(self):
        """Get the :attr:`estimated_params` as an array

        Return
        ------
        numpy.array
        """
        pp = list(self.estimated_params.values())
        if None in pp:
            raise exceptions.XoaKrigingError(
                "Not all parameters are estimated: {}".format(self.estimated_params)
            )
        return np.array(pp)

    def set_params_array(self, pp):
        """Set the :attr:`estimated_param` with an array

        Parameters
        ----------
        pp: numpy.array
            Array of estimated parameters
        """
        for i, name in enumerate(self.estimated_params):
            self._estimated_params[name] = pp[i]
        return self.params

    @property
    def params(self):
        """Current variogram model parameters"""
        return self.get_params()

    @property
    def is_fitted(self):
        """Are all the parameters known, either frozen or estimated?"""
        return all(self.get_param(name) is not None for name in self.param_names)

    def apply(self, d, pp=None):
        """Call the variogram model function

        Parameters
        ----------
        d: array
            Distances
        """
        return self.get_func(pp)(d)

    __call__ = apply

    def get_func(self, pp=None):
        """Get the variogram model function using `pp` variable arguments"""
        if pp is not None:
            params = self.set_params_array(pp)
        else:
            params = self.get_params()
            if None in list(params.values()):
                raise exceptions.XoaKrigingError(
                    "Not all parameters are estimated: {}".format(self.estimated_params)
                )
        return get_variogram_model_func(self.mtype, **params)

    def fit(self, da: xr.DataArray, **kwargs):
        """Estimate parameters from data"""
        # We need a data array
        if isinstance(da, xr.Dataset):
            if len(da) == 0:
                raise exceptions.XoaKrigingError(
                    "No variable found in the dataset for estimating the variogram parameters."
                )
            if len(da) > 1:
                exceptions.xoa_warn(
                    "Multiple candidate variables found in the dataset for estimating "
                    "the variogram parameters. Keeping only the first one."
                )
            da = da[list(da)[0]]

        # Empirical variogram
        if da.name in ("semivariogram", "variogram") and "dist" in da.coords:
            if (
                "units" in da.dist.attrs
                and da.dist.attrs["units"] in xgeo.distance_units
                and xgeo.distance_units[da.dist.attrs["units"]] != self._dist_units
            ):
                exceptions.xoa_warn(
                    "Incompatible distance units: {} vs {}".format(
                        self._dist_units, da.dist.attrs["units"]
                    )
                )
            ev = da
        else:
            kwargs["dist_units"] = self._dist_units
            ev = empirical_variogram(da, **kwargs)
        dist = ev.dist.values
        values = ev.values
        self._ev = ev

        # Nothing to estimate
        if not self.estimated_params:
            return

        # First guess of paramaters
        imax = np.ma.argmax(values)
        self.set_estimated_params(n=0.0, s=values[imax], r=dist[imax], overwrite=False)
        pp0 = self.get_params_array()

        # Fitting
        from scipy.optimize import minimize

        def func(pp):
            return ((values - self(dist, pp)) ** 2).sum()

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'divide by zero encountered in divide')
            nugget_min = NUGGET_MIN * values.max()
            bounds = [
                (nugget_min if name == "n" else np.finfo('d').eps, None)
                for name in self.estimated_params
            ]
            self._fit = minimize(func, pp0, bounds=bounds, method='L-BFGS-B')
            pp = self._fit['x']
            self._fit_err = np.sqrt(func(pp) / values.size)

        self.set_params_array(pp)

    def plot(self, rmax=None, nr=100, show_params=True, **kwargs):
        """Plot the semivariogram

        Parameters
        ----------
        rmax: float
            Max range in meters
        nr: int
            Number of points to plot the curve
        show_params: bool, dict
            Show a text box that contains the variogram parameters in the lower right corner.
        kwargs: dict
            Extra keyword are passed to the `xarray.DataArray.plot` callable accessor
        """
        # Distances
        if rmax is None:
            if self._ev is not None:
                rmax = float(self._ev.dist.max())
            else:
                rmax = self["r"] * 1.2
        dist = np.linspace(0, rmax, nr)

        # Array and plot
        du = str(self._dist_units)
        mv = xr.DataArray(
            self.get_func()(dist),
            dims='dist',
            coords=[("dist", dist, {"long_name": "Distance", "units": du})],
            attrs={"long_name": self.mtype.name.title() + " fit"},
        )
        kwargs.setdefault("label", mv.long_name)
        p = mv.plot(**kwargs)

        # Text box for params
        if show_params:
            params = self.params.copy()
            text = [
                "r[ange] = {:<g} {}".format(params["r"], du),
                "n[ugget] = {:<g}".format(params["n"]),
                "s[ill] = {:<g}".format(params["s"]),
            ]
            maxlen = max([len(t) for t in text])
            text = "\n".join(t.ljust(maxlen) for t in text)
            axes = p[0].axes
            axes.text(
                0.98,
                0.04,
                text,
                transform=axes.transAxes,
                family="monospace",
                bbox=dict(facecolor=(1, 1, 1, 0.5)),
                ha="right",
            )
        return p


class kriging_types(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported kriging types"""

    #: Ordinary kriging (default)
    ordinary = 1
    #: Simple kriging
    simple = 0


class BaseKriger(object):
    """Base class of the krigers

    It manages the kriging type, the variogram, the input points and the formatting
    of the output as a dataset that has the coordinates of the target positions,
    and the attributes of the input data.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        Input positions and data.
        The dimensions that are not geographic are kept in the output.
    krigtype: optional
        Kriging type: {kriging_types.rst_with_links}.
    variogram_func: callable, VariogramModel, str
        A callable, an instance of :class:`VariogramModel` or a model name.
        A model that is not fitted yet is fitted to the data.
    exact: optional
        If True, variogram is exactly zero when distance is zero.
    dist_units: optional
        Distance units as one of: {xgeo.distance_units.rst_with_links}.
    mean: optional
        Mean for the simple kriging, which defaults to the mean of the data.
    """

    def __init__(
        self, da, krigtype, variogram_func, exact=False, dist_units=None, mean=None, **kwargs
    ):
        # Kriging type
        self.krigtype = kriging_types[krigtype]

        # Variogram function
        if isinstance(variogram_func, str):
            variogram_func = VariogramModel(variogram_func, dist_units=dist_units)
        if isinstance(variogram_func, VariogramModel):
            if dist_units is None:
                dist_units = variogram_func.dist_units
            elif xgeo.distance_units[dist_units] != variogram_func.dist_units:
                exceptions.xoa_warn(
                    f"Incompatible distance units: {xgeo.distance_units[dist_units]} "
                    f"vs {variogram_func.dist_units}"
                )
            if not variogram_func.is_fitted:
                variogram_func.fit(da)
        self._variogram_func = variogram_func
        self._dist_units = xgeo.distance_units[dist_units]

        # Input points
        cloud = xgeo.clusterize(da, npmax=np.inf)
        self._vname = cloud.encoding["clust_var_names"][0]
        cloud[self._vname] = cloud[self._vname].transpose(..., "npts")
        self._cloud = cloud
        self._attrs = cloud[self._vname].attrs
        self._dims = cloud[self._vname].dims[:-1]
        self._unstacked_coords = {
            cname: cdat for cname, cdat in cloud.coords.items() if "npts" not in cdat.dims
        }

        # Other parameters
        self.exact = exact
        if self.krigtype != kriging_types.simple:
            mean = 0.0
        elif mean is None:
            mean = float(da.to_array().mean() if isinstance(da, xr.Dataset) else da.mean())
        self.mean = mean

    @property
    def dist_units(self):
        """Distance units"""
        return self._dist_units

    @property
    def variogram_func(self):
        """Variogram function or callable, like a :class:`VariogramModel` instance"""
        return self._variogram_func

    @property
    def sill(self):
        """Sill of the variogram, used to convert it to a covariance for the simple kriging"""
        vgf = self.variogram_func
        if isinstance(vgf, VariogramModel):
            return vgf.get_params()["s"]
        return float(vgf(np.float64(1e12)))

    def _interp_points(self, dso, xname, yname, **kwargs):
        """Interpolate to stacked positions and return the values and the errors"""
        raise NotImplementedError

    def interp(self, dso, name=None, **kwargs):
        """Interpolate to the `dso` positions

        Parameters
        ----------
        dso: xarray.Dataset, xarray.DataArray
            Object that contains lon and lat coordinates, which are not necessarily
            along the same dimension, like a regular grid.
        name: str, None:
            Name of the output data array, which defaults to the name of the input one.

        Return
        ------
        xarray.Dataset
            A dataset that contains the interpolated values and the associated kriging variance
            with the suffix ``_error``.
            It has the coordinates of `dso` with their attributes, and the non geographic
            coordinates of the input data.
            The attributes of the data are those of the input data.
        """
        # Positions
        stacked = xcoords.geo_stack(dso, rename=False, drop=True, singleton=True)
        sname, xname, yname = stacked.encoding["geo_stack"]
        stacked_coords = {
            cname: cdat for cname, cdat in stacked.coords.items() if sname in cdat.dims
        }

        # Interpolate
        zo, eo = self._interp_points(stacked, xname, yname, **kwargs)

        # Format
        name = name or self._vname
        coords = {**self._unstacked_coords, **stacked_coords}
        attrs_err = {"long_name": "Kriging variance"}
        if "units" in self._attrs:
            attrs_err["units"] = self._attrs["units"] + "^2"
        out = xr.Dataset(
            {
                name: xr.DataArray(
                    zo, dims=self._dims + (sname,), coords=coords, attrs=self._attrs
                ),
                name
                + "_error": xr.DataArray(eo, dims=(sname,), coords=stacked_coords, attrs=attrs_err),
            }
        )
        return xcoords.geo_unstack(out, dso, sname)

    __call__ = interp


BaseKriger.__doc__ = BaseKriger.__doc__.format(**locals())


class Kriger(BaseKriger):
    """Kriger that works with all the input points at once

    The variogram matrix of the input points is factorized once, and the values
    are computed at the output positions using this factorization.
    The memory and time costs grow quickly with the number of input points:
    use :class:`LocalKriger` or :class:`ClusteredKriger` for big clouds of points.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        Input positions and data.
    krigtype: optional
        Kriging type: {kriging_types.rst_with_links}.
    variogram_func: callable, VariogramModel, optional
        Callable to be used as a variogram function.
        It is either a function or an instance of :class:`VariogramModel`.
    exact: optional
        If True, variogram is exactly zero when distance is zero.

    See also
    --------
    BaseKriger
    LocalKriger
    ClusteredKriger
    """

    def __init__(self, da, krigtype, variogram_func, **kwargs):
        super().__init__(da, krigtype, variogram_func, **kwargs)
        self._clouds = [self._cloud]

    @property
    def _spd(self):
        """Is the kriging matrix symmetric positive definite?"""
        return self.krigtype == kriging_types.simple

    def _get_matrix(self, dd, diag=False):
        """Convert distances to the variogram, or the covariance for the simple kriging"""
        g = np.asarray(self.variogram_func(dd), dtype="d")
        if self.exact:
            if diag:
                np.fill_diagonal(g, 0)
            g[np.isclose(g, 0.0)] = 0.0
        if self.krigtype == kriging_types.simple:
            g = self.sill - g
        return g

    def get_A(self, cloud):
        """Get the variogram matrix of a cloud of points"""
        plus1 = int(self.krigtype == kriging_types.ordinary)
        dd = xgeo.get_distances(cloud, units=self._dist_units).values
        npts = cloud.sizes["npts"]
        A = np.empty((npts + plus1, npts + plus1))
        A[:npts, :npts] = self._get_matrix(dd, diag=True)
        if plus1:
            A[-1] = 1
            A[:, -1] = 1
            A[-1, -1] = 0
        return A

    @property
    def solvers(self):
        """Get the solver of the A matrix of each cloud of points"""
        if not hasattr(self, '_solvers'):
            self._solvers = [
                ckrig.MatrixSolver(self.get_A(cloud), spd=self._spd) for cloud in self._clouds
            ]
        return self._solvers

    def _interp_cloud(self, cloud, solver, dso, neighbors):
        """Interpolate with a single cloud of points

        Return
        ------
        numpy.ndarray
            Values
        numpy.ndarray
            Errors
        """
        plus1 = int(self.krigtype == kriging_types.ordinary)

        # Form B
        dd = xgeo.get_distances(cloud, dso, units=self._dist_units).values
        npts = cloud.sizes["npts"]
        B = np.empty((npts + plus1, dd.shape[-1]))
        B[:npts] = self._get_matrix(dd)
        if plus1:
            B[-1] = 1
        del dd

        # Block kriging
        if neighbors is not None:
            Bb = B.copy()
            for i, iineigh in enumerate(neighbors):
                Bb[:, i] = B[:, iineigh].mean(axis=1)
            B = Bb

        # Compute weights
        W = solver.solve(B)

        # Interpolate
        zc = cloud[self._vname].values
        if self.krigtype == kriging_types.simple:
            zc = zc - self.mean
        z = zc @ W[:npts]
        if self.krigtype == kriging_types.simple:
            z += self.mean

        # Get error
        e = (W * B).sum(axis=0)
        if self.krigtype == kriging_types.simple:
            e = self.sill - e
        return z, e

    def _interp_points(self, dso, xname, yname, block=None):
        # Block kriging
        neighbors = None
        if block:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.dstack([dso[xname].values, dso[yname].values])[0])
            neighbors = tree.query_ball_tree(tree, block)

        return self._interp_cloud(self._cloud, self.solvers[0], dso, neighbors)

    def interp(self, dso, block=None, name=None):
        """Interpolate to the `dso` positions

        Parameters
        ----------
        dso: xarray.Dataset, xarray.DataArray
            Object that contains lon and lat coordinates
        block: None, float
            Radius of the blocks for block kriging, in the units of the coordinates.
        name: str, None:
            Name of the output data array.

        Return
        ------
        xarray.Dataset
            See :meth:`BaseKriger.interp`.
        """
        return super().interp(dso, name=name, block=block)

    __call__ = interp


Kriger.__doc__ = Kriger.__doc__.format(**locals())


class ClusteredKriger(Kriger):
    """Kriger that supports clusterization to limit memory

    Big input cloud of points (size > ``npmax``)
    are split into smaller clusters using cluster analysis of distance with
    function :func:`~xoa.geo.clusterize`.

    The problem is solved in this way:

        #. Input points are split in clusters if necessary.
        #. The input variogram matrix is factorized
           for each cluster, possibly using
           :mod:`multiprocessing` if ``nproc>1``.
        #. Values are computed at output positions
           each using the factorized matrix of cluster.
        #. Final value is a weighted average of
           the values estimated using each cluster.
           Weights are inversely proportional to the squared error.
        #. The final error is the same weighted average of the errors, which is
           conservative since the errors of the clusters are correlated.

    Parameters
    ----------
    npmax: optional
        Maximal number of points to be used simultaneously for kriging.
        When the number of input points is greater than this value,
        clusterization is applied.
    nproc: optional
        Number of processes to use to factorize matrices.
        Set it to a number <2 to switch off parallelisation.
    seed: int, None
        Seed of the random initialisation of the clusters, to get reproducible results.
        The clusters, hence the result, change slightly from one call to another without it.

    See also
    --------
    Kriger
    """

    def __init__(self, da, krigtype, variogram_func, npmax=None, nproc=None, seed=None, **kwargs):
        super().__init__(da, krigtype, variogram_func, **kwargs)

        # Clusters
        if npmax is None:
            npmax = np.inf
        self._clouds = xgeo.clusterize(da, npmax=npmax, split=True, seed=seed)

        # Number of cores for parallel computing
        if nproc is None:
            nproc = cpu_count()
        else:
            nproc = max(1, min(cpu_count(), nproc))
        self.nproc = min(nproc, self.nclust)

    @property
    def clusters(self):
        return self._clouds

    @property
    def nclust(self):
        """Number of clusters"""
        return len(self.clusters)

    @property
    def npmax(self):
        """Max number of points per cluster"""
        return self.clusters[0].npmax

    @property
    def solvers(self):
        """Get the solver of the A matrix of each cluster"""
        if not hasattr(self, '_solvers'):
            AA = [self.get_A(clust) for clust in self.clusters]
            if self.nproc > 1:
                with Pool(self.nproc) as pool:
                    self._solvers = pool.map(
                        partial(ckrig.MatrixSolver, spd=self._spd), AA, chunksize=1
                    )
            else:
                self._solvers = [ckrig.MatrixSolver(A, spd=self._spd) for A in AA]
        return self._solvers

    def _interp_points(self, dso, xname, yname, block=None):
        neighbors = None
        if block:
            from scipy.spatial import cKDTree

            tree = cKDTree(np.dstack([dso[xname].values, dso[yname].values])[0])
            neighbors = tree.query_ball_tree(tree, block)

        zo = eo = wo = 0.0
        for clust, solver in zip(self.clusters, self.solvers):
            z, e = self._interp_cloud(clust, solver, dso, neighbors)

            # Weighted contribution based on errors, which can be zero at an input point
            w = 1 / np.maximum(e, 1e-12 * self.sill) ** 2
            zo = zo + z * w
            eo = eo + e * w
            wo = wo + w
        return zo / wo, eo / wo


class LocalKriger(BaseKriger):
    """Kriger that uses only the nearest input points of each output point

    The neighbors are found with a :class:`~xoa.core.geo.SphericalKDTree`, and a small
    kriging system is solved for each output point with numba, in parallel.
    The cost grows linearly with the number of output points and hardly depends on
    the number of input points, which makes it suited to big clouds of points.

    Parameters
    ----------
    da: xarray.DataArray, xarray.Dataset
        Input positions and data, which must not contain missing values:
        the points with a missing value are ignored.
    krigtype: optional
        Kriging type: {kriging_types.rst_with_links}.
    variogram_func: VariogramModel, str
        A :class:`VariogramModel` instance or a model name, which is then fitted.
        A callable is not supported.
    nneigh: int
        Number of nearest neighbors.
    maxdist: float, None
        Maximal distance to the neighbors, in `dist_units`.
    nmin: int
        Minimal number of neighbors, below which the result is nan.
    exact: optional
        If True, variogram is exactly zero when distance is zero.

    See also
    --------
    BaseKriger
    Kriger
    ClusteredKriger
    """

    def __init__(self, da, krigtype, variogram_func, nneigh=16, maxdist=None, nmin=3, **kwargs):
        super().__init__(da, krigtype, variogram_func, **kwargs)
        if not isinstance(self.variogram_func, VariogramModel):
            raise exceptions.XoaKrigingError(
                "The local kriging needs a variogram model name or a VariogramModel instance"
            )
        self.nneigh = nneigh
        self.maxdist = maxdist
        self.nmin = max(nmin, 1)

        # Valid input points
        cloud = self._cloud
        data = cloud[self._vname]
        values = data.values.reshape(-1, data.shape[-1])
        valid = np.isfinite(values).all(axis=0) & np.isfinite(cloud.lon) & np.isfinite(cloud.lat)
        self._shape = data.shape[:-1]
        self._lon = np.ascontiguousarray(cloud.lon.values[valid], "d")
        self._lat = np.ascontiguousarray(cloud.lat.values[valid], "d")
        self._values = np.ascontiguousarray(values[:, valid], "d")

        # Tree, whose radius sets the units of distances
        self._radius = xgeo.EARTH_RADIUS * (
            1e-3 if self._dist_units == xgeo.distance_units.km else 1.0
        )
        self.tree = SphericalKDTree(self._lon, self._lat, radius=self._radius)

    def _interp_points(self, dso, xname, yname):
        lono = np.ascontiguousarray(dso[xname].values, "d")
        lato = np.ascontiguousarray(dso[yname].values, "d")
        k = min(self.nneigh, self._lon.size)
        dist, idx = self.tree.query(lono, lato, k=k, max_dist=self.maxdist, workers=-1)
        vgf = self.variogram_func
        n, s, r = ckrig.clamp_variogram_params(**vgf.get_params())
        zo, eo = ckrig.krig_local(
            self._lon,
            self._lat,
            self._values,
            lono,
            lato,
            np.ascontiguousarray(idx.reshape(-1, k), "i8"),
            np.ascontiguousarray(dist.reshape(-1, k), "d"),
            vgf.mtype.value,
            n,
            s,
            r,
            min(self.nmin, k),
            self.krigtype == kriging_types.ordinary,
            bool(self.exact),
            float(self.mean),
            self._radius,
        )
        return zo.reshape(self._shape + (-1,)), eo


LocalKriger.__doc__ = LocalKriger.__doc__.format(**locals())


def krig(dsi, dso, krigtype="ordinary", **kwargs):
    """Quickly krig data

    The :class:`LocalKriger` is used when `nneigh` is given,
    and the :class:`ClusteredKriger` when `npmax` or `nproc` is given.
    """
    kwinterp = {}
    for key in "block", "name":
        if key in kwargs:
            kwinterp[key] = kwargs.pop(key)
    if "nneigh" in kwargs:
        kwinterp.pop("block", None)
        cls = LocalKriger
    elif "npmax" in kwargs or "nproc" in kwargs:
        cls = ClusteredKriger
    else:
        cls = Kriger
    return cls(dsi, krigtype, **kwargs).interp(dso, **kwinterp)
