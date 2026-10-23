"""
Dimensionless numbers of ocean dynamics

Provides :func:`get_rossby`, :func:`get_burger`, :func:`get_froude` and :func:`get_ekman`.
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
import xarray as xr

from . import coords as xcoords
from . import grid as xgrid
from .dyn import _format_uv_doc_, _name_, get_coriolis, get_relvort


def _result_(da, name, long_name, *refs):
    """Name a dimensionless result after the first data array of the inputs"""
    ref = next((r for r in refs if isinstance(r, xr.DataArray)), None)
    if ref is not None:
        return _name_(da, ref, name, long_name, "1")
    return xr.DataArray(da, name=name, attrs={"long_name": long_name, "units": "1"})


def _get_abs_coriolis_(lat, *refs):
    """Get the absolute Coriolis parameter, that is missing where it is zero"""
    if lat is None:
        ref = next((r for r in refs if isinstance(r, xr.DataArray)), None)
        if ref is None:
            raise ValueError("lat is required when no input is a data array")
        lat = xcoords.get_lat(ref, errors="raise")
    corio = abs(get_coriolis(xr.DataArray(lat) if np.ndim(lat) == 0 else lat))
    return corio.where(corio > 0)


@_format_uv_doc_
def get_rossby(u, v, dx=None, dy=None, z=None):
    """Compute the Rossby number, which is dimensionless

    It is the relative vorticity divided by the Coriolis parameter, that is positive for
    a cyclonic rotation, and negative for an anticyclonic one, in both hemispheres.
    {parameters}
    Return
    ------
    xarray.DataArray

    See also
    --------
    xoa.dyn.get_relvort
    xoa.dyn.get_coriolis
    """
    relvort = get_relvort(u, v, dx=dx, dy=dy, z=z)
    corio = get_coriolis(xgrid.get_lonlat_2d(u)[1])
    rossby = relvort / corio
    return _name_(rossby.transpose(*u.dims), u, "rossby", "Rossby number", "1")


def get_burger(length, rd=None, n2=None, h=None, lat=None):
    """Compute the Burger number, which is dimensionless

    It compares the deformation radius to the horizontal scale of the flow:
    :math:`Bu = (R_d / L)^2`. It is of order one for the mesoscale, and small for
    flows that are larger than the deformation radius, like the basin scale.

    The deformation radius is either given with `rd`, like the one of
    :class:`xoa.dyn.VerticalModes`, which accounts for the real stratification, or
    approximated by :math:`N H / |f|` from `n2` and `h`.

    Parameters
    ----------
    length: float, xarray.DataArray
        Horizontal scale of the flow in meters
    rd: None, float, xarray.DataArray
        Deformation radius in meters
    n2: None, xarray.DataArray
        Squared Brunt-Vaisala frequency in s-2, which is used with `h` when `rd` is not given.
        See :func:`xoa.thermdyn.get_n2`.
    h: None, float, xarray.DataArray
        Vertical scale in meters, like the depth of the layer
    lat: None, float, xarray.DataArray
        Latitudes in degrees, that default to the ones of `n2` or `length`.
        It is not used when `rd` is given.

    Return
    ------
    xarray.DataArray
    """
    if rd is None:
        if n2 is None or h is None:
            raise ValueError("Either rd or both n2 and h must be provided")
        n = np.sqrt(n2.where(n2 > 0) if isinstance(n2, xr.DataArray) else n2)
        rd = n * h / _get_abs_coriolis_(lat, n2, length)
    return _result_((rd / length) ** 2, "burger", "Burger number", rd, length)


def get_froude(speed, n2, h):
    """Compute the internal Froude number, which is dimensionless

    It is the speed over the internal wave speed :math:`N H`: the flow is subcritical
    when it is smaller than one, and critical or supercritical otherwise.

    Parameters
    ----------
    speed: float, xarray.DataArray
        Speed of the flow in m s-1
    n2: xarray.DataArray
        Squared Brunt-Vaisala frequency in s-2, that is missing where it is not positive.
        See :func:`xoa.thermdyn.get_n2`.
    h: float, xarray.DataArray
        Vertical scale in meters, like the depth of the layer

    Return
    ------
    xarray.DataArray
    """
    n = np.sqrt(n2.where(n2 > 0))
    return _result_(speed / (n * h), "froude", "Froude number", speed, n2)


def get_ekman(nu, h, lat):
    """Compute the vertical Ekman number, which is dimensionless

    It is :math:`E = \\nu / (|f| H^2)`, the ratio of the vertical friction
    to the Coriolis force, that is small when the friction is confined to thin boundary layers.

    Parameters
    ----------
    nu: float, xarray.DataArray
        Vertical eddy viscosity in m2 s-1
    h: float, xarray.DataArray
        Vertical scale in meters, like the depth of the layer
    lat: float, xarray.DataArray
        Latitudes in degrees

    Return
    ------
    xarray.DataArray
    """
    ekman = nu / (_get_abs_coriolis_(lat) * h**2)
    return _result_(ekman, "ekman", "Ekman number", nu, h, lat)
