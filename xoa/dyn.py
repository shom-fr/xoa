"""
Routines related to ocean dynamics.

Provides :func:`get_sea_level` for identifying sea level variables,
:func:`flow2d` for 2D Lagrangian particle advection, and the kinematic diagnostics
computed from velocities on collocated grids: :func:`get_relvort`, :func:`get_div`,
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


import numpy as np
import pandas as pd
import xarray as xr

from . import exceptions
from . import coords as xcoords
from . import grid as xgrid
from . import meta as xmeta
from . import misc as xmisc
from . import interp
from . import geo
from .core import dyn as cdyn


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


def _set_attrs_(da, ref, name, long_name, units):
    """Set the attributes of an output and restore the ones of the coordinates

    :func:`xarray.apply_ufunc` drops the attributes of the coordinates, which are needed
    to find the longitudes and latitudes.
    """
    for cname, coord in ref.coords.items():
        if cname in da.coords:
            da.coords[cname].attrs.update(coord.attrs)
    da.name = name
    da.attrs = {"long_name": long_name, "units": units}
    return da


def _apply_kinematics_(func, nout, u, v, dx, dy, names, long_names, units, dtypes=None):
    """Apply a core function to velocities and set the attributes of its outputs"""
    xdim = xcoords.get_xdim(u, errors="raise")
    ydim = xcoords.get_ydim(u, errors="raise")
    dx, dy = _get_dx_dy_(u, dx, dy)
    dx = xr.DataArray(dx) if np.ndim(dx) == 0 else dx
    dy = xr.DataArray(dy) if np.ndim(dy) == 0 else dy
    core_dims = [[ydim, xdim]]
    res = xr.apply_ufunc(
        func,
        u,
        v,
        dx,
        dy,
        join="override",
        input_core_dims=[core_dims[0], core_dims[0]]
        + [core_dims[0] if np.ndim(d) else [] for d in (dx, dy)],
        output_core_dims=core_dims * nout,
        dask="parallelized",
        output_dtypes=[float] * nout,
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    if nout == 1:
        res = (res,)
    out = []
    for da, name, long_name, unit in zip(res, names, long_names, units):
        out.append(_set_attrs_(da.transpose(*u.dims), u, name, long_name, unit))
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
"""


def _format_uv_doc_(func):
    func.__doc__ = func.__doc__.replace("{parameters}", _DOC_UV_)
    return func


@_format_uv_doc_
def get_relvort(u, v, dx=None, dy=None):
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
        cdyn.relvort, 1, u, v, dx, dy, ["relvort"], ["Relative vorticity"], ["s-1"]
    )


@_format_uv_doc_
def get_div(u, v, dx=None, dy=None):
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
    return _apply_kinematics_(
        cdyn.div, 1, u, v, dx, dy, ["div"], ["Horizontal divergence"], ["s-1"]
    )


@_format_uv_doc_
def get_strain(u, v, dx=None, dy=None):
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
    return _apply_kinematics_(
        cdyn.strain,
        2,
        u,
        v,
        dx,
        dy,
        ["strain_normal", "strain_shear"],
        ["Normal strain", "Shear strain"],
        ["s-1", "s-1"],
    )


@_format_uv_doc_
def get_okuboweiss(u, v, dx=None, dy=None):
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
    return _apply_kinematics_(
        cdyn.okuboweiss,
        1,
        u,
        v,
        dx,
        dy,
        ["okuboweiss"],
        ["Okubo-Weiss parameter"],
        ["s-2"],
    )


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
    lon, lat = xgrid._get_lonlat_yx_(ssh)
    corio = get_coriolis(lat)
    ug, vg = xr.apply_ufunc(
        cdyn.geos,
        ssh,
        dx,
        dy,
        corio,
        join="override",
        input_core_dims=[[ydim, xdim]] * 4,
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
        out.append(_set_attrs_(da.transpose(*ssh.dims), ssh, name, long_name, "m s-1"))
    return tuple(out)
