"""
Routines related to ocean dynamics.

Provides :func:`get_sea_level` for identifying sea level variables,
:func:`flow2d` for 2D Lagrangian particle advection, and the kinematic diagnostics
computed from velocities on collocated grids, that may have levels with a variable vertical
coordinate (see also :class:`VerticalModes` for the vertical modes): :func:`get_relvort`, :func:`get_div`,
:func:`get_strain`, :func:`get_okuboweiss`, as well as :func:`get_coriolis` and
:func:`get_geos`.
"""

# Copyright 2020-2026 Shom
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import textwrap

import numpy as np
import pandas as pd
import xarray as xr

from . import exceptions
from . import coords as xcoords
from . import grid as xgrid
from . import meta as xmeta
from . import misc as xmisc
from . import stats as xstats
from . import interp
from . import geo
from .core import dyn as cdyn
from .core import modes as cmodes

SEA_LEVEL_VARIANTS = xmisc.Choices(
    {
        None: "No restriction",
        "ssh": "sea surface height",
        "adt": "absolute dynamic topography",
        "sla": "sea level anomaly",
        "mdt": "mean dynamic topography",
        "mss": "mean sea surface",
    },
    parameter="variant",
    description="Restrict checking to a given variant(s)",
    multi=True,
)


def _get_sea_level_variant_(variant):
    variant = SEA_LEVEL_VARIANTS[variant]
    if variant is None:
        variant = ["ssh", "adt", "sla", "mdt", "mss"]
    return variant


@SEA_LEVEL_VARIANTS.format_function_docstring
def get_sea_level(ds, variant=None, errors="ignore"):
    """Search for a sea level variable in a dataset

    Looks for SSH, ADT, SLA, MDT or MSS variables using the
    :mod:`xoa.meta` specifications.

    See: https://help.marine.copernicus.eu/en/articles/6025269-what-are-the-differences-between-the-ssh-and-sla-variables

    Parameters
    ----------
    ds: xarray.Dataset
    {variant}
    errors: str
        Error handling: ``"ignore"``, ``"warn"`` or ``"raise"``.

    Return
    ------
    xarray.DataArray, None
        The sea level array, or None if not found.

    Example
    -------
    .. code-block:: python

        >>> ds = xr.Dataset(dict(ssh=(("lat", "lon"), data)))
        >>> get_sea_level(ds)
        <xarray.DataArray 'ssh' ...>

    """
    variant = _get_sea_level_variant_(variant)
    return xmeta.get_meta_specs(ds).get(ds, variant, errors=errors)
    # errors = xmisc.ERRORS[errors]
    # meta_specs = xmeta.get_meta_specs(ds)
    # try:
    #     sea_level = meta_specs.get(ds, candidates, errors=errors)
    # except Exception as e:
    #     raise exceptions.XoaError("Can't find a single sea level-like variable: " + e.message)
    # return sea_level


def _get_uv2d_(t, txy, gx, gy, gz, gt, guv):
    """Interpolate gridded u and v on track positions"""
    # Shape
    npts = int(txy.size / 2)
    tx = txy[:npts]
    ty = txy[npts:]
    tt = np.full(npts, t)
    tz = np.zeros(npts)

    # Interpolate
    tuv = interp.grid2locs(gx, gy, gz, gt, guv, tx, ty, tz, tt)

    # Scale the speed for degrees
    tuv[0] *= 180.0 / (np.pi * geo.EARTH_RADIUS)
    tuv[1] *= 180.0 / (np.pi * geo.EARTH_RADIUS)
    tuv[1] *= np.cos(ty * np.pi / 180)

    # Pack velocity
    return tuv.ravel()


def _rk4_(xy, f, t, dt, **kwargs):
    """Integrate one time step with RK4"""
    k1 = dt * f(t, xy, **kwargs)
    k2 = dt * f(t + 0.5 * dt, xy + 0.5 * k1, **kwargs)
    k3 = dt * f(t + 0.5 * dt, xy + 0.5 * k2, **kwargs)
    k4 = dt * f(t + 0.5 * dt, xy + 0.5 * k3, **kwargs)
    return xy + (k1 + k2 + k3 + k4) / 6.0


def _integrate_(xy, f, t0, t1, dt, **kwargs):
    """Low level integration with packed, pure-numeric data"""

    # Fit the time steps to the time range
    dt *= 1.0
    if t1 < 0:
        dts = [dt] * int(-t1)
    else:
        nt = int((t1 - t0) // dt)
        dts = [dt] * nt
        dtf = (t1 - t0) % dt
        if dtf:
            dts.append(dtf)

    # Iterative integration
    t = t0
    tt = [t0]
    xxyy = [xy]
    for dt in dts:
        xy2 = _rk4_(xy, f, t, dt, **kwargs)
        t2 = t + dt
        tt.append(t2)
        xxyy.append(xy2)
        t = t2
        xy = xy2
    return np.array(tt), np.asarray(xxyy)


def flow2d(u, v, xy0, duration, step, date=None):
    """Integrate gridded 2D velocities from initial positions

    Uses a 4th-order Runge-Kutta scheme to advect particles
    in a 2D velocity field.

    Parameters
    ----------
    u: xarray.DataArray
        Gridded zonal velocity (must be 2D after squeezing).
    v: xarray.DataArray
        Gridded meridional velocity (must be 2D after squeezing).
    xy0: tuple, int, xarray.Dataset
        Initial positions. Either:

        - a ``(x_array, y_array)`` tuple of longitudes and latitudes,
        - an ``int`` for randomly placed particles,
        - a :class:`xarray.Dataset` with longitude and latitude coordinates.
    duration: int, numpy.timedelta64
        Total integration time in seconds.
    step: int, numpy.timedelta64
        Integration time step in seconds.
    date: None, numpy.datetime64
        A reference date for the output time coordinate.

    Return
    ------
    xarray.Dataset
        Output positions with ``lon`` and ``lat`` coordinates
        varying along ``time`` and ``particles`` dimensions.

    Example
    -------
    Advect two particles for 3 hours with a 2-hour time step::

        ff = flow2d(u, v, ([1., 2.], [1., 1.5]),
                    np.timedelta64(3, "h"), np.timedelta64(2, "h"),
                    date="2000-01-01")

    See Also
    --------
    xoa.interp.grid2loc
    """
    # Gridded field
    time0 = xcoords.get_time(u, errors="ignore")
    u = u.squeeze(drop=True)
    v = v.squeeze(drop=True)
    if u.ndim != 2 or v.ndim != 2:
        raise exceptions.XoaError("The velocity field must be 2D")
    u = xgrid.to_rect(u)
    gx = xcoords.get_lon(u).values
    gy = xcoords.get_lat(u).values
    if gx.ndim == 1:
        gx = gx.reshape(1, -1)
    if gy.ndim == 1:
        gy = gy.reshape(-1, 1)
    gz = np.zeros((1,) * 5)
    gt = np.zeros(1)
    gu = u.values.reshape((1,) * 2 + u.shape[-2:])
    gv = v.values.reshape(gu.shape)
    guv = np.array([gu, gv])

    # Initial positions
    tt0 = None
    if isinstance(xy0, xr.Dataset):
        tx0 = xcoords.get_lon(xy0).values.ravel()
        ty0 = xcoords.get_lon(xy0).values.ravel()
        tt0 = xcoords.get_time(xy0, errors='ignore')
    elif isinstance(xy0, int):
        tx0 = np.random.uniform(gx.min(), gx.max(), xy0)
        ty0 = np.random.uniform(gy.min(), gy.max(), xy0)
    else:
        tx0, ty0 = xy0
        tx0 = np.asarray(tx0)
        ty0 = np.asarray(ty0)
    txy0 = np.concatenate((tx0, ty0))

    # Integration
    t0 = 0
    if isinstance(duration, np.timedelta64):
        duration /= np.timedelta64(1, "s")
    if isinstance(step, np.timedelta64):
        step /= np.timedelta64(1, "s")
    t1 = duration
    tt, txy = _integrate_(txy0, _get_uv2d_, t0, t1, step, gx=gx, gy=gy, gz=gz, gt=gt, guv=guv)
    tx, ty = np.split(txy, 2, axis=1)

    # As a dataset
    if date is not None:
        date = pd.to_datetime(date).to_datetime64()
    elif tt0 is not None and len(tt0) <= 1:
        date = tt0.values
    elif time0 is not None:
        date = time0.values
    else:
        date = pd.Timestamp.now().to_datetime64()
    time = xr.DataArray(tt * np.timedelta64(1, 's') + date, dims='time')

    return xr.Dataset(
        coords={
            'lon': (('time', 'particles'), tx),
            'lat': (('time', 'particles'), ty),
            'time': time,
        }
    )


def _get_dx_dy_(u, dx, dy):
    """Get the resolutions, that are computed from coordinates when not provided"""
    if dx is None or dy is None:
        dx0, dy0 = xgrid.get_dx_dy(u)
        dx = dx0 if dx is None else dx
        dy = dy0 if dy is None else dy
    return dx, dy


def _name_(da, ref, name, long_name, units):
    """Name an output and give it clean attributes, while its coordinates keep theirs"""
    da = xcoords.copy_coord_attrs(da, ref)
    da.name = name
    da.attrs = {"long_name": long_name, "units": units}
    return da


def _normalize_(da, u, divisor, name, long_name):
    """Divide a diagnostic by a scale to make it dimensionless"""
    return _name_((da / divisor).transpose(*u.dims), u, name, long_name, "1")


def _apply_kinematics_(func, nout, u, v, dx, dy, z, names, long_names, units):
    """Apply a core function to velocities and set the attributes of its outputs"""
    xdim = xcoords.get_xdim(u, errors="raise")
    ydim = xcoords.get_ydim(u, errors="raise")
    dx, dy = _get_dx_dy_(u, dx, dy)
    dx = xr.DataArray(dx) if np.ndim(dx) == 0 else dx
    dy = xr.DataArray(dy) if np.ndim(dy) == 0 else dy
    hdims = [ydim, xdim]
    args = [u, v, dx, dy]
    in_dims = [hdims, hdims] + [hdims if np.ndim(d) else [] for d in (dx, dy)]
    out_dims = hdims
    if z is not None:
        zdim = xcoords.get_zdim(u, errors="raise")
        out_dims = [zdim] + hdims
        in_dims = [out_dims] * 2 + in_dims[2:] + [out_dims]
        args.append(z)
    res = xr.apply_ufunc(
        func,
        *args,
        join="override",
        input_core_dims=in_dims,
        output_core_dims=[out_dims] * nout,
        dask="parallelized",
        output_dtypes=[float] * nout,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    if nout == 1:
        res = (res,)
    out = []
    for da, name, long_name, unit in zip(res, names, long_names, units):
        out.append(_name_(da.transpose(*u.dims), u, name, long_name, unit))
    return out[0] if nout == 1 else tuple(out)


_DOC_UV_ = """
    Parameters
    ----------
    u, v: xarray.DataArray
        Zonal and meridional velocities in m s-1 on the same collocated grid,
        with the X and Y dimensions found with :mod:`xoa.coords`.
    dx, dy: None, float, xarray.DataArray
        Resolutions along X and Y in meters, which default to the ones computed
        from the coordinates with :func:`xoa.grid.get_dx_dy`.
    z: None, xarray.DataArray
        Vertical coordinate of the levels, like the depths or heights of the sigma levels.
        The horizontal derivatives are then computed at constant z instead of along the
        levels, which are sloping. The velocities must then have a vertical dimension,
        which is the one of `z`, and the sign convention of `z` does not matter.
        See :ref:`indepth.dyn`.
{normalize}
"""

_DOC_NORMALIZE_ = {
    "div": "the Coriolis parameter",
    "strain": "the absolute value of the Coriolis parameter",
    "okuboweiss": "its standard deviation",
}


def _format_uv_doc_(func):
    # The indentation of docstrings is already removed
    scale = _DOC_NORMALIZE_.get(func.__name__[4:])
    normalize = (
        ""
        if scale is None
        else f"    normalize: bool\n        Divide the result by {scale} to make it dimensionless.\n"
    )
    doc = _DOC_UV_.replace("{normalize}", normalize)
    func.__doc__ = func.__doc__.replace("{parameters}", textwrap.dedent(doc))
    return func


@_format_uv_doc_
def get_relvort(u, v, dx=None, dy=None, z=None):
    """Compute the relative vorticity, in s-1

    It is :math:`\\partial_x v - \\partial_y u`, which is positive for a counter-clockwise
    rotation, i.e. a cyclonic one in the northern hemisphere.
    {parameters}
    Return
    ------
    xarray.DataArray

    See also
    --------
    xoa.core.dyn.relvort
    """
    return _apply_kinematics_(
        cdyn.relvort, 1, u, v, dx, dy, z, ["relvort"], ["Relative vorticity"], ["s-1"]
    )


@_format_uv_doc_
def get_div(u, v, dx=None, dy=None, z=None, normalize=False):
    """Compute the horizontal divergence, in s-1

    It is :math:`\\partial_x u + \\partial_y v`.
    {parameters}
    Return
    ------
    xarray.DataArray

    See also
    --------
    xoa.core.dyn.div
    """
    div = _apply_kinematics_(
        cdyn.div, 1, u, v, dx, dy, z, ["div"], ["Horizontal divergence"], ["s-1"]
    )
    if normalize:
        corio = get_coriolis(xgrid._get_lonlat_yx_(u)[1])
        div = _normalize_(div, u, corio, "div", "Divergence / f")
    return div


@_format_uv_doc_
def get_strain(u, v, dx=None, dy=None, z=None, normalize=False):
    """Compute the normal and shear strains, in s-1

    They are :math:`\\partial_x u - \\partial_y v` and :math:`\\partial_x v + \\partial_y u`.
    {parameters}
    Return
    ------
    xarray.DataArray
        Normal strain
    xarray.DataArray
        Shear strain

    See also
    --------
    xoa.core.dyn.strain
    """
    strains = _apply_kinematics_(
        cdyn.strain,
        2,
        u,
        v,
        dx,
        dy,
        z,
        ["strain_normal", "strain_shear"],
        ["Normal strain", "Shear strain"],
        ["s-1", "s-1"],
    )
    if normalize:
        corio = abs(get_coriolis(xgrid._get_lonlat_yx_(u)[1]))
        strains = tuple(
            _normalize_(da, u, corio, da.name, da.long_name + " / |f|") for da in strains
        )
    return strains


@_format_uv_doc_
def get_okuboweiss(u, v, dx=None, dy=None, z=None, normalize=False):
    """Compute the Okubo-Weiss parameter, in s-2

    It is the squared strain minus the squared vorticity: it is negative where
    rotation dominates, like in the core of eddies, and positive where strain dominates.
    {parameters}
    Return
    ------
    xarray.DataArray

    See also
    --------
    xoa.core.dyn.okuboweiss
    """
    ow = _apply_kinematics_(
        cdyn.okuboweiss,
        1,
        u,
        v,
        dx,
        dy,
        z,
        ["okuboweiss"],
        ["Okubo-Weiss parameter"],
        ["s-2"],
    )
    if normalize:
        ow = _normalize_(ow, u, ow.std(), "okuboweiss", "Okubo-Weiss / std")
    return ow


def get_coriolis(lat):
    """Compute the Coriolis parameter, in s-1

    Parameters
    ----------
    lat: xarray.DataArray, float, array_like
        Latitudes in degrees

    Return
    ------
    xarray.DataArray, float, numpy.ndarray
        Same type as the input
    """
    corio = cdyn.coriolis(lat)
    if isinstance(corio, xr.DataArray):
        corio.name = "coriolis"
        corio.attrs.update(long_name="Coriolis parameter", units="s-1")
    return corio


def get_geos(ssh, dx=None, dy=None):
    """Compute the geostrophic currents from the sea surface height, in m s-1

    Parameters
    ----------
    ssh: xarray.DataArray
        Sea surface height in meters
    dx, dy: None, float, xarray.DataArray
        Resolutions along X and Y in meters, which default to the ones computed
        from the coordinates with :func:`xoa.grid.get_dx_dy`.

    Return
    ------
    xarray.DataArray
        Zonal geostrophic current
    xarray.DataArray
        Meridional geostrophic current

    See also
    --------
    xoa.core.dyn.geos
    """
    xdim = xcoords.get_xdim(ssh, errors="raise")
    ydim = xcoords.get_ydim(ssh, errors="raise")
    dx, dy = _get_dx_dy_(ssh, dx, dy)
    dx = xr.DataArray(dx) if np.ndim(dx) == 0 else dx
    dy = xr.DataArray(dy) if np.ndim(dy) == 0 else dy
    lon, lat = xgrid._get_lonlat_yx_(ssh)
    corio = get_coriolis(lat)
    core_dims = [ydim, xdim]
    ug, vg = xr.apply_ufunc(
        cdyn.geos,
        ssh,
        dx,
        dy,
        corio,
        join="override",
        input_core_dims=[core_dims]
        + [core_dims if np.ndim(d) else [] for d in (dx, dy)]
        + [core_dims],
        output_core_dims=[[ydim, xdim]] * 2,
        dask="parallelized",
        output_dtypes=[float] * 2,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    out = []
    for da, name, long_name in zip(
        (ug, vg),
        ("ugeos", "vgeos"),
        ("Zonal geostrophic current", "Meridional geostrophic current"),
    ):
        out.append(_name_(da.transpose(*ssh.dims), ssh, name, long_name, "m s-1"))
    return tuple(out)


def get_cyclogeos(
    ssh, dx=None, dy=None, terms="full", smooth=None, niter=300, tol=1e-4, relax=0.5, fmin=1e-5
):
    """Compute the cyclogeostrophic currents from the sea surface height, in m s-1

    They include the advection terms of the momentum equations on top of the geostrophic
    balance, which is needed where the currents are curved like in eddies and meanders.
    They are found with an iterative method like the one of Penven et al. (2014), see
    :func:`xoa.core.dyn.cyclogeos` for the details.

    Parameters
    ----------
    ssh: xarray.DataArray
        Sea surface height in meters
    dx, dy: None, float, xarray.DataArray
        Resolutions along X and Y in meters, which default to the ones computed
        from the coordinates with :func:`xoa.grid.get_dx_dy`.
    terms: {"full", "penven"}
        Advection terms of the correction: the four terms of the complete advection,
        or the two terms that are taught as the ones of Penven et al. (2014).
        See :func:`xoa.core.dyn.cyclogeos`.
    smooth: None, int
        Number of smoothings of the advection terms at each iteration, which prevents
        the iterations from diverging. See :func:`xoa.core.dyn.cyclogeos`.
    niter: int
        Maximal number of iterations
    tol: float
        The iterations stop when the currents change by less than this value, in m s-1
    relax: float
        Relaxation factor between 0 and 1
    fmin: float
        Minimal absolute value of the Coriolis parameter, in s-1, below which the currents
        are missing, which is close to the equator

    Return
    ------
    xarray.DataArray
        Zonal cyclogeostrophic current
    xarray.DataArray
        Meridional cyclogeostrophic current

    Currents are missing where the iterations did not converge, like in very strong
    anticyclones that have no cyclogeostrophic solution.

    References
    ----------
    Penven, P., Halo, I., Pous, S., Marié, L. (2014), Cyclogeostrophic balance in the
    Mozambique Channel, J. Geophys. Res. Oceans, 119, 1054-1067, doi:10.1002/2013JC009528.

    See also
    --------
    get_geos
    xoa.core.dyn.cyclogeos
    """
    xdim = xcoords.get_xdim(ssh, errors="raise")
    ydim = xcoords.get_ydim(ssh, errors="raise")
    dx, dy = _get_dx_dy_(ssh, dx, dy)
    if np.ndim(dx) == 0:
        dx = xr.DataArray(dx)
    if np.ndim(dy) == 0:
        dy = xr.DataArray(dy)
    ug, vg = get_geos(ssh, dx=dx, dy=dy)
    corio = get_coriolis(xgrid._get_lonlat_yx_(ssh)[1])
    core_dims = [ydim, xdim]
    uc, vc = xr.apply_ufunc(
        cdyn.cyclogeos,
        ug,
        vg,
        dx,
        dy,
        corio,
        join="override",
        input_core_dims=[core_dims, core_dims]
        + [core_dims if np.ndim(d) else [] for d in (dx, dy)]
        + [core_dims],
        output_core_dims=[core_dims] * 2,
        dask="parallelized",
        output_dtypes=[float] * 2,
        dask_gufunc_kwargs={"allow_rechunk": True},
        kwargs={
            "terms": terms,
            "smooth": smooth,
            "niter": niter,
            "tol": tol,
            "relax": relax,
            "fmin": fmin,
        },
    )
    out = []
    for da, name, long_name in zip(
        (uc, vc),
        ("ucgeos", "vcgeos"),
        ("Zonal cyclogeostrophic current", "Meridional cyclogeostrophic current"),
    ):
        out.append(_name_(da.transpose(*ssh.dims), ssh, name, long_name, "m s-1"))
    return tuple(out)


def _mode_dims_(res, ref, *drop):
    """Order the dimensions of a result: the mode, the ones of the reference and the others"""
    dims = [d for d in res.dims if d not in drop]
    first = ["mode"] if "mode" in res.dims else []
    ordered = [d for d in ref.dims if d in dims and d != "mode"]
    return res.transpose(*first, *ordered, *[d for d in dims if d not in ordered and d != "mode"])


class VerticalModes:
    """Vertical modes of water columns, to decompose and recompose profiles

    The quasi-geostrophic vertical modes :math:`F_n(z)` are computed from the stratification
    of each water column, with :mod:`xoa.core.modes`: the first one is the barotropic mode,
    and the next ones are the baroclinic modes, that have more and more nodes and are slower
    and slower. Profiles of the currents, of the pressure or of the temperature
    are then split into the amplitudes of the modes with :meth:`decomp`
    and put back together with :meth:`recomp`.

    Everything works with extra dimensions, like time or horizontal ones, and with dask,
    since the computations are done with :func:`xarray.apply_ufunc`.
    The levels may vary in space, like with sigma coordinates, the levels that are below
    the bottom are missing, and the columns that have different depths have
    different modes.

    Parameters
    ----------
    n2: xarray.DataArray
        Squared Brunt-Vaisala frequency in s-2 at the levels, with a vertical dimension.
        See :func:`xoa.thermdyn.get_n2`.
    z: None, xarray.DataArray
        Vertical coordinate of the levels, which defaults to the one of `n2`.
        It may vary in space, and its sign convention is given by its ``positive``
        attribute, see :func:`xoa.coords.as_z`.
    nmodes: int
        Number of modes, including the barotropic one
    bc: {"rigid", "free"}
        Boundary condition at the surface: either a rigid lid, which has no
        speed for the barotropic mode, or a free surface.
    lat: None, float, xarray.DataArray
        Latitudes, to compute the deformation radii. They default to the ones
        of the coordinates, if any.
    zrange: None, tuple(float)
        Focus on a layer: only the levels between these two values, in the convention
        of `z`, are used, and the modes have no flux through the limits of the layer.
        Everything is missing outside of it.
    n2min: float
        Minimal stratification in s-2, which is used in the unstable or mixed layers
    fmin: float
        The deformation radii are missing where the absolute value of
        the Coriolis parameter is smaller than this value in s-1, like close to the equator.

    Attributes
    ----------
    modes: xarray.DataArray
        The modes, whose mean square over the water column is one,
        and which are positive at the surface, with the ``mode`` dimension first.
    c: xarray.DataArray
        Speeds of the modes in m s-1, which are missing for the barotropic mode of
        the rigid lid.
    rd: None, xarray.DataArray
        Deformation radii in meters, which are the speeds divided by the Coriolis parameter,
        if the latitudes are known.
    dz: xarray.DataArray
        Thickness of the layers, that are the weights of the integrals.

    Example
    -------
    .. code-block:: python

        n2 = xoa.thermdyn.get_n2(ds.temp, ds.sal)
        vm = xoa.dyn.VerticalModes(n2, nmodes=5)
        amps = vm.decomp(ds.u)  # Amplitudes of the modes
        baroclinic = vm.recomp(amps, modes=[1, 2])  # Part of the first baroclinic modes
        vm.rd.isel(mode=1)  # First deformation radius

    See also
    --------
    xoa.core.modes.vertical_modes
    xoa.thermdyn.get_n2
    """

    def __init__(
        self, n2, z=None, nmodes=5, bc="rigid", lat=None, zrange=None, n2min=1e-9, fmin=1e-5
    ):
        nmodes = int(nmodes)
        if nmodes < 1:
            raise ValueError("nmodes must be a positive integer")
        if bc not in ("rigid", "free"):
            raise ValueError(f"bc must be 'rigid' or 'free', not {bc!r}")
        self.nmodes = nmodes
        self.bc = bc
        self.zrange = None if zrange is None else tuple(float(v) for v in zrange)
        self.zdim = zdim = xcoords.get_zdim(n2, errors="raise")
        self.n2 = n2

        # Vertical coordinate, which is positive up
        if z is None:
            z = xcoords.get_z(n2)
            positive = xcoords.get_positive_attr(n2, zdim)
        else:
            positive = z.attrs.get("positive")
            if positive not in ("up", "down"):
                positive = "up" if not bool((z > 0).any()) else "down"
            z = xcoords.as_z(z)
        self.z = z

        # Layer, in the convention of the vertical coordinate
        zmin, zmax = -np.inf, np.inf
        if self.zrange is not None:
            bounds = sorted(self.zrange)
            zmin, zmax = bounds if positive == "up" else (-bounds[1], -bounds[0])

        modes, c, dz = xr.apply_ufunc(
            cmodes.vertical_modes,
            n2,
            z,
            input_core_dims=[[zdim], [zdim]],
            output_core_dims=[["mode", zdim], ["mode"], [zdim]],
            kwargs={
                "nmodes": nmodes,
                "free_surface": bc == "free",
                "zmin": zmin,
                "zmax": zmax,
                "n2min": n2min,
            },
            join="override",
            dask="parallelized",
            output_dtypes=[float] * 3,
            dask_gufunc_kwargs={"output_sizes": {"mode": nmodes}, "allow_rechunk": True},
        )
        specs = (
            (modes, "vmodes", "Vertical modes", "1"),
            (c, "c", "Speed of the vertical modes", "m s-1"),
            (dz, "dz", "Thickness of the layers", "m"),
        )
        out = []
        for da, name, long_name, units in specs:
            da = _mode_dims_(da, n2)
            if "mode" in da.dims:
                da = da.assign_coords(mode=np.arange(nmodes))
            out.append(_name_(da, n2, name, long_name, units))
        self.modes, self.c, self.dz = out

        # Deformation radii
        self.rd = None
        if lat is None:
            lat = xcoords.get_lat(n2, errors="ignore")
        if lat is not None:
            corio = abs(get_coriolis(float(lat) if np.ndim(lat) == 0 else lat))
            with np.errstate(divide="ignore", invalid="ignore"):
                rd = xr.where(corio >= fmin, self.c / corio, np.nan)
            self.rd = _name_(rd, n2, "rd", "Deformation radius of the vertical modes", "m")
            self.rd = _mode_dims_(self.rd, self.c)

    def __repr__(self):
        zrange = "" if self.zrange is None else f", zrange={self.zrange}"
        return f"<VerticalModes: {self.nmodes} modes, bc={self.bc!r}, zdim={self.zdim!r}{zrange}>"

    def decomp(self, da, nmodes=None):
        """Decompose profiles into the amplitudes of the vertical modes

        The amplitude of the barotropic mode is the vertical mean of the profile.

        Parameters
        ----------
        da: xarray.DataArray
            Profiles, with the vertical dimension of the modes and possibly other dimensions,
            that are broadcast with the ones of the modes. It must be defined at all
            the levels of the water columns, or the amplitudes are missing.
        nmodes: None, int
            Number of modes to keep, which defaults to all of them.

        Return
        ------
        xarray.DataArray
            The amplitudes, with the ``mode`` dimension first and the other dimensions
            of the profiles, without the vertical one.

        See also
        --------
        recomp
        """
        modes = self.modes if nmodes is None else self.modes.isel(mode=slice(0, nmodes))
        amps = xr.apply_ufunc(
            cmodes.project,
            da,
            modes,
            self.dz,
            input_core_dims=[[self.zdim], ["mode", self.zdim], [self.zdim]],
            output_core_dims=[["mode"]],
            join="override",
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={
                "output_sizes": {"mode": modes.sizes["mode"]},
                "allow_rechunk": True,
            },
        )
        amps = amps.assign_coords(mode=modes.mode.values)
        amps = _mode_dims_(amps, da)
        amps = xcoords.copy_coord_attrs(amps, da)
        amps.name = "amps"
        amps.attrs = {
            "long_name": "Amplitude of the vertical modes",
            "units": da.attrs.get("units", ""),
        }
        return amps

    def recomp(self, amps, modes=None):
        """Recompose profiles from the amplitudes of the vertical modes

        Parameters
        ----------
        amps: xarray.DataArray
            Amplitudes as returned by :meth:`decomp`, that may have less modes
            than the object.
        modes: None, int, list(int)
            Modes to use: the first ones if it is an integer, else the ones of the list,
            the other amplitudes being null. All the modes of `amps` by default.

        Return
        ------
        xarray.DataArray
            Profiles, with the vertical dimension at the same place as in the
            stratification, and the other dimensions after them. They are missing
            outside of the layer of the zoom.

        See also
        --------
        decomp
        """
        available = amps.mode.values
        if modes is not None:
            wanted = np.arange(modes) if np.ndim(modes) == 0 else np.asarray(modes)
            missing = [int(m) for m in wanted if m not in available]
            if missing:
                raise ValueError(f"The modes {missing} are not available in the amplitudes")
            amps = amps.where(amps.mode.isin(wanted), 0.0)
        basis = self.modes.sel(mode=available)
        rec = xr.apply_ufunc(
            cmodes.reconstruct,
            amps,
            basis,
            input_core_dims=[["mode"], ["mode", self.zdim]],
            output_core_dims=[[self.zdim]],
            join="override",
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        rec = _mode_dims_(rec, self.n2)
        rec = xcoords.copy_coord_attrs(rec, amps)
        rec.name = "recomp"
        rec.attrs = {
            "long_name": "Profile recomposed from its vertical modes",
            "units": amps.attrs.get("units", ""),
        }
        return rec

    def energy_share(self, u, v=None):
        """Get the share of the kinetic energy of each mode

        The mean kinetic energy over the water column is the sum of the squared amplitudes
        of the modes, by orthogonality. This is the ratio of each of them to the total.

        Parameters
        ----------
        u: xarray.DataArray
            Profiles of the currents, or of a single component
        v: None, xarray.DataArray
            Profiles of the other component

        Return
        ------
        xarray.DataArray
            Shares between 0 and 1, with the ``mode`` dimension first

        See also
        --------
        decomp
        """
        comps = [u] if v is None else [u, v]
        energy = sum(self.decomp(c) ** 2 for c in comps)
        total = sum((c**2 * self.dz).sum(self.zdim) / self.dz.sum(self.zdim) for c in comps)
        share = energy / total
        return _name_(share, u, "energy_share", "Share of the kinetic energy", "1")

    def truncation_error(self, da, nmodes):
        """Get the error of recomposing profiles with a limited number of modes

        Parameters
        ----------
        da: xarray.DataArray
            Profiles
        nmodes: list(int)
            Numbers of modes to test, which must not exceed the ones of the object

        Return
        ------
        xarray.DataArray
            Root mean square error over all the points, with an ``nmodes`` dimension

        See also
        --------
        recomp
        """
        amps = self.decomp(da)
        errors = [float(xstats.rms(self.recomp(amps, modes=n), ref=da)) for n in nmodes]
        errors = xr.DataArray(errors, dims="nmodes", coords={"nmodes": list(nmodes)}, name="rms")
        errors.attrs = {
            "long_name": "Truncation error of the vertical modes",
            "units": da.attrs.get("units", ""),
        }
        return errors
