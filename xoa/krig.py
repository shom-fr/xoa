# -*- coding: utf8 -*-
"""
Kriging adapted from vacumm's module
https://github.com/VACUMM/vacumm/blob/master/lib/python/vacumm/misc/grid/kriging.py

"""
from __future__ import absolute_import
from multiprocessing import Pool, cpu_count
import warnings

import numpy as np
import xarray as xr

from .__init__ import XoaError, xoa_warn
from . import misc
from . import geo as xgeo
from . import coords as xcoords


def _get_blas_func_(name):
    import scipy.linalg.blas

    return scipy.linalg.blas.get_blas_funcs(name)


def _get_lapack_func_(name):
    import scipy.linalg.lapack

    return scipy.linalg.lapack.get_lapack_funcs(name)


def _dgemv_(a, x):
    blas_dgemv = _get_blas_func_('gemv')
    return blas_dgemv(1.0, a, x)


def _symm_(a, b):
    blas_dgemm = _get_blas_func_('gemm')
    return blas_dgemm(1.0, a, b)


def _syminv_(a):
    """Invert a real symmetric definite matrix"""
    return np.linalg.pinv(a)
    n = a.shape[0]
    jj, ii = np.triu_indices(n)
    up = a[jj, ii]
    pptri = _get_lapack_func_("pptri")
    res = pptri(n, up)
    if isinstance(res, tuple):
        info = res[1]
        if info:
            raise KrigingError(f'Error during call to Lapack DPPTRI (info={info})')
        return res[0]
    else:
        return res


class KrigingError(XoaError):
    pass


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

    da: xarray.dataArray
        Data array with lon and lat coordinates.
    nmax: optional
        Above this number, size of the sample is reduced by a crude undersampling.
    binned: optional
        If set to a number,
        data are arranged in bins to estimate
        variogram. If set to ``None``, data are
        arranged in bins if the number of pairs
        of points is greater than ``nbindef*nbmin``.
    nbindef: optional
        Default number of bins (not used if ``binned`` is a number).
    nbin0: optional
        If set to a number > 1,
        the first bin is split into nbin0 sub-bins.
        If set to ``None``, it is evaluated with
        ``min(bins[1]/nbmin, nbin)``.
    nbmin: optional
        Minimal number of points in a bin.
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
        samp = npts / nmax
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

    # Unique
    d, iiuni = np.unique(d, return_index=True)
    v = v[:, iiuni]

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
        attrs = {"units": da.attrs["units"] + "^2"}
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
    #: Gausian
    gaussian = 2
    #: Spherical
    spherical = 3


def get_variogram_model_func(mtype, n, s, r, nrelmax=0.2):
    """Get the variogram model function from its name"""

    mtype = variogram_model_types[mtype]

    n = max(n, 0)
    n = min(n, nrelmax * s)
    r = max(r, 0)
    s = max(s, 0)

    if mtype.name == 'linear':
        return lambda h: n + (s - n) * ((h / r) * (h <= r) + 1 * (h > r))

    if mtype.name == 'exponential':
        return lambda h: n + (s - n) * (1 - np.exp(-3 * h / r))

    if mtype.name == 'gaussian':
        return lambda h: n + (s - n) * (1 - np.exp(-3 * h ** 2 / r ** 2))

    if mtype.name == 'spherical':
        return lambda h: n + (s - n) * ((1.5 * h / r - 0.5 * (h / r) ** 3) * (h <= r) + 1 * (h > r))


class VariogramModel(object):
    """Class used when fitting a variogram model to data to better control params

    Parameters
    ----------
    mtype: int, str, variogram_model_types
    dist_units: int, str, xoa.geo.distance_units
    **frozen_params:
        Varigram paramaters that must be frozen.

    """

    param_names = list(get_variogram_model_func.__code__.co_varnames[1:])
    param_names.remove('nrelmax')

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
        """Set the value to no frozen paramters"""
        params_update = dict(
            (name, params[name])
            for name in self.param_names
            if name not in self._frozen_params
            and name in params
            and (not overwrite or name not in self._estimated_params)
        )
        self._estimated_params.update(params_update)

    estimated_params = property(
        get_estimated_params, set_estimated_params, doc='Estimated paramaters as :class:`dict`'
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
            raise KrigingError(
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
            raise VariogramModelError(
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
                raise VariogramModelError(
                    "Not all parameters are estimated: {}".format(self.estimated_params)
                )
        return get_variogram_model_func(self.mtype, **params)

    def fit(self, da: xr.DataArray, **kwargs):
        """Estimate parameters from data"""
        # We need a data array
        if isinstance(da, xr.Dataset):
            if len(da) == 1:
                da = da[list(da)[0]]
            elif len(da) > 1:
                xoa_warn(
                    "Multiple candidate variables found in the dataset for estimating "
                    "the variogram parameters. Keeping only the first one."
                )
            elif len(da) == 0:
                raise KrigingError(
                    "No variable found in the dataset for estimating the variogram parameters."
                )

        # Empirical variogram
        if da.name == "semivariogram" or da.name == "variogram" and "dist" in da.coords:
            if (
                "units" in da.dist.attrs
                and da.dist.attrs["units"] in xgeo.distance_units
                and xgeo.distance_units[da.dist.attrs["units"]] != self._dist_units
            ):
                xoa_warn(
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
            self._fit = minimize(
                func, pp0, bounds=[(np.finfo('d').eps, None)] * len(pp0), method='L-BFGS-B'
            )
            pp = self._fit['x']
            self._fit_err = np.sqrt(func(pp)) / values.size

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
        if rmax is None and self._ev is not None:
            rmax = self._ev.dist.max()
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


class VariogramModelError(KrigingError):
    pass


class kriging_types(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported kriging of variograms"""

    #: Exponential (default)
    ordinary = 1
    #: Linear
    simple = 0


class Kriger(object):
    """Kriger that supports clusterization to limit memory

    Big input cloud of points (size > ``npmax``)
    are split into smaller clusters using cluster analysis of distance with
    function :func:`~xoa.geo.clusterize`.

    The problem is solved in this way:

        #. Input points are split in clusters if necessary.
        #. The input variogram matrix is inverted
           for each cluster, possibly using
           :mod:`multiprocessing` if ``nproc>1``.
        #. Values are computed at output positions
           each using the inverted matrix of cluster.
        #. Final value is a weighted average of
           the values estimated using each cluster.
           Weights are inversely proportional to the inverse
           of the squared error.

    Parameters
    ----------

    da: xarray.DataArray, xarray.Dataset
        Input positions and optionaly data.
    krigtype: optional
        Kriging type: {kriging_types.rst_with_links}.
    variogram_func: callable, VariogramModel, optional
        Callable to be used as a variogram function.
        It is either a function or an instance of :class:`VariogramModel`.
    npmax: optional
        Maximal number of points to be used simultanously for kriging.
        When the number of input points is greater than this value,
        clusterization is applied.
    nproc: optional
        Number of processes to use to invert matrices.
        Set it to a number <2 to switch off parallelisation.
    exact: optional
        If True, variogram is exactly zero when distance is zero.


    """

    def __init__(
        self,
        da: xr.DataArray,
        krigtype,
        variogram_func,
        npmax=None,
        nproc=None,
        exact=False,
        dist_units=None,
        mean=None,
        farvalue=None,
        **kwargs,
    ):
        # Kriging type
        self.krigtype = kriging_types[krigtype]

        # Variogram function
        if isinstance(variogram_func, str):
            variogram_func = VariogramModel(variogram_func, dist_units=dist_units)
        if isinstance(variogram_func, VariogramModel):
            if dist_units is None:
                dist_units = variogram_func.dist_units
            else:
                dist_units = xgeo.distance_units[dist_units]
            if dist_units != variogram_func.dist_units:
                xoa_warn(
                    f"Incompatible distance units: {dist_units} vs {variogram_func.dist_units}"
                )
            variogram_func.fit(da)
            self._variogram_func = variogram_func
        else:
            self._variogram_func = variogram_func
        self._dist_units = xgeo.distance_units[dist_units]

        # Clusters
        if npmax is None:
            npmax = np.inf
        self._clusters = xgeo.clusterize(da, npmax=npmax, split=True)
        self._unstacked_coords = {}
        for cname, cdat in self.clusters[0].coords.items():
            if "npts" not in cdat.dims:
                self._unstacked_coords[cname] = cdat

        # Number of cores for parallel computing
        if nproc is None:
            nproc = cpu_count()
        else:
            nproc = max(1, min(cpu_count(), nproc))
        self.nproc = min(nproc, self.nclust)

        # Other parameters
        self.exact = exact
        if self.krigtype != kriging_types.simple:
            mean = 0.0
        elif mean is None:
            mean = float(da.mean())
        self.mean = mean

    @property
    def clusters(self):
        return self._clusters

    @property
    def nclust(self):
        """Number of clusters"""
        return len(self.clusters)

    @property
    def npmax(self):
        """Max number of points per cluster"""
        return self.clusters[0].npmax

    @property
    def dist_units(self):
        """Distance units"""
        return self._dist_units

    @property
    def variogram_func(self):
        """Variogram function or callable, like a :class:`VariogramModel` instance"""
        return self._variogram_func

    @property
    def Ainv(self):
        """Get the inverse of the A matrix"""

        # Already computed
        if hasattr(self, '_Ainv'):
            return self._Ainv

        # Variogram function
        vgf = self.variogram_func

        # Loop on clusters
        Ainv = []
        AA = []
        plus1 = int(self.krigtype == kriging_types.ordinary)
        for ic, clust in enumerate(self.clusters):

            # Get distance between input points
            dd = xgeo.get_distances(clust).values

            # Form A
            npts = clust.sizes["npts"]
            A = np.empty((npts + plus1, npts + plus1))
            A[:npts, :npts] = vgf(dd)
            if self.exact:
                np.fill_diagonal(A, 0)
                A[:npts, :npts][np.isclose(A[:npts, :npts], 0.0)] = 0.0
            if self.krigtype == kriging_types.ordinary:
                A[-1] = 1
                A[:, -1] = 1
                A[-1, -1] = 0

            # Invert for single processor
            if self.nproc == 1:
                Ainv.append(_syminv_(A))
            else:
                AA.append(A)

        # Multiprocessing inversion
        if self.nproc > 1:
            pool = Pool(self.nproc)
            Ainv = pool.map(_syminv_, AA, chunksize=1)
            pool.close()

        # Fortran arrays
        Ainv = [np.asfortranarray(ainv, 'd') for ainv in Ainv]
        self._Ainv = Ainv
        return Ainv

    def interp(self, dso: xr.Dataset, block=None, name=None):
        """Interpolate to the `dso` positions

        Parameters
        ----------
        dso: xr.Dataset
            Dataset that contains lon and lat coordinates
        block: None, int
            Number of nearest neighbours for block kriging
        name: str, None:
            Default name of the output data array.

        Return
        ------
        xarray.Dataset
            A dataset that contains the interpolated values and associated erreors

        """

        # Inits
        dso = xcoords.geo_stack(dso, rename=False, drop=True)
        sname, xname, yname = dso.encoding["geo_stack"]
        stacked_coords = {}
        for cname, cdat in dso.coords.items():
            if sname in cdat.dims:
                stacked_coords[cname] = cdat
        vgf = self.variogram_func
        nptso = dso.sizes[sname]
        vname = self.clusters[0].encoding["clust_var_names"][0]
        so = self.clusters[0][vname].shape[:-1] + (nptso,)
        dimso = self.clusters[0][vname].dims[:-1] + (sname,)
        dimsoe = (sname,)
        zo = np.zeros(so, 'd')
        eo = np.zeros(nptso, 'd')
        wo = np.zeros(nptso, 'd')
        if block:
            xyo = np.dstack([dso[xname].values, dso[yname].values])[0]

        # Loop on clusters
        Ainv = self.Ainv
        plus1 = int(self.krigtype == kriging_types.ordinary)
        for ic in range(self.nclust):  # TODO: multiproc here?

            # Distances to output points
            dd = xgeo.get_distances(self.clusters[ic], dso, units=self._dist_units).values

            # Form B
            npts = self.clusters[ic].sizes["npts"]
            B = np.empty((npts + plus1, nptso))
            B[:npts] = vgf(dd)
            if self.krigtype == kriging_types.ordinary:
                B[-1] = 1
            if self.exact:
                B[:npts][np.isclose(B[:npts], 0.0)] = 0.0
            del dd

            # Block kriging
            if block:
                from scipy.spatial import cKDTree

                tree = cKDTree(xyo)
                Bb = B.copy()
                for i, iineigh in enumerate(tree.query_ball_tree(tree, block)):
                    Bb[:, i] = B[:, iineigh].mean()
                B = Bb

            # Compute weights
            W = np.ascontiguousarray(_symm_(Ainv[ic], np.asfortranarray(B, 'd')))

            # Interpolate
            zc = self.clusters[ic][vname].values
            if self.krigtype == kriging_types.simple:
                zc = zc - self.mean
            z = zc @ W[:npts]
            if self.krigtype == kriging_types.simple:
                z += self.mean

            # Get error
            #            e = (W[:-1]*B[:-1]).sum(axis=0)
            e = (W * B).sum(axis=0)
            del W, B

            # Weigthed contribution based on errors
            w = 1 / e ** 2
            if self.nclust > 1:
                z[:] *= w
            wo += w
            del w
            zo += z
            del z

        # Error
        eo = 1 / np.sqrt(wo)

        # Normalization
        if self.nclust > 1:
            zo /= wo

        # Format
        coords = self._unstacked_coords.copy()
        coords.update(stacked_coords)
        dao = xr.DataArray(zo, dims=dimso, coords=coords, attrs=self.clusters[0].attrs)
        daoe = xr.DataArray(
            eo, dims=dimsoe, coords=stacked_coords, attrs={"long_name": "Squared error"}
        )
        if name is None:
            name = self.clusters[0].encoding["clust_var_names"][0]
        if name is None:
            name = "data"
        return xr.Dataset({name: dao, name + "_error": daoe}).unstack()

        # gc.collect()
        # if geterr:
        #     return zo, eo
        # return zo

    __call__ = interp


Kriger.__doc__ = Kriger.__doc__.format(**locals())


def krig(dsi, dso, krigtype="ordinary", **kwargs):
    """Quickly krig data"""
    kwinterp = {}
    for key in "block", "name":
        if key in kwargs:
            kwinterp[key] = kwargs.pop(key)
    return Kriger(dsi, krigtype, **kwargs).interp(dso, **kwinterp)
