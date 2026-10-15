#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermodynamics utilities.

Provides functions to identify and retrieve temperature, salinity and
density variables from datasets, and to compute the mixed layer depth.
"""
import numpy as np
import xarray as xr

from . import exceptions
from . import meta as xmeta
from . import misc as xmisc
from . import coords as xcoords
from . import regrid as xregrid
from .core import thermdyn as cthermdyn


# def _get_array_(ds, func, variants, variant, errors, name):
#     if not variant:
#         variant = None
#     if isinstance(variant, str) or variant is None:
#         variant = [variant]

#     das = []
#     for da in ds.values():
#         for vr in variant:
#             if func(da, vr):
#                 das.append(da)

#     errors = xmisc.ERRORS[errors]
#     if not das:
#         msg = f"Found no {name} array"
#         if errors == "raise":
#             raise exceptions.XoaThermdynError(msg)
#         if errors == "warning":
#             exceptions.xoa_warn(msg)
#         return
#     if len(das) > 1:
#         msg = f"Found more than one {name} array"
#         if errors == "raise":
#             raise exceptions.XoaThermdynError(msg)
#         if errors == "warning":
#             exceptions.xoa_warn(msg, stacklevel=3)
#     return das[0]


_TEMP_KINDS = {
    "temp": "In situ",
    "ctemp": "Conservative temperature",
    "ptemp": "Potential temperature",
}
_TEMP_ALIASES = {"temp": "insitu", "ptemp": "potential", "ctemp": "conservative"}

TEMP_VARIANTS = xmisc.Choices(
    {None: "No restriction", **_TEMP_KINDS},
    parameter="variant",
    description="Restrict checking to a given variant(s)",
    aliases=_TEMP_ALIASES,
    multi=True,
)

TEMP_KINDS = xmisc.Choices(
    {None: "Inferred from the array, else potential", **_TEMP_KINDS},
    parameter="temp_kind",
    description="Kind of temperature",
    aliases=_TEMP_ALIASES,
)


def _get_temp_variant_(variant):
    variant = TEMP_VARIANTS[variant]
    if variant is None:
        variant = ["temp", "ptemp", "ctemp"]
    return variant


@TEMP_VARIANTS.format_method_docstring
def is_temp(da, variant=None):
    """Check if ``da`` is a temperature-like array

    Parameters
    ----------
    da: xarray.DataArray
    {variant}

    Return
    ------
    bool

    Example
    -------
    .. code-block:: python

        >>> da = xr.DataArray([15, 16], dims="z", name="temp")
        >>> is_temp(da)
        True

    See also
    --------
    get_temp
    is_sal
    is_dens
    """
    variant = _get_temp_variant_(variant)
    return bool(xmeta.get_meta_specs(da).match_data_var(da, variant))


@TEMP_VARIANTS.format_method_docstring
def get_temp(ds, variant=None, errors="warn"):
    """Search for a temperature variable in a dataset

    Parameters
    ----------
    ds: xarray.Dataset
    {variant}
    errors: str
        Error handling: ``"ignore"``, ``"warn"`` or ``"raise"``.

    Return
    ------
    xarray.DataArray, None
        The temperature array, or None if not found.
        If several arrays match and ``errors`` is ``"warn"``,
        the first one is returned.

    See also
    --------
    is_temp
    get_sal
    get_dens
    """
    variant = _get_temp_variant_(variant)
    return xmeta.get_meta_specs(ds).get(ds, variant, errors=errors)
    # return _get_array_(ds, is_temp, TEMP_VARIANTS, variant, errors, "temperature")


_SAL_KINDS = {
    "asal": "Absolute salinity",
    "pfsal": "Preformed salinity",
    "psal": "Practical salinity",
}
_SAL_ALIASES = {
    "psal": "practical",
    "pfsal": "preformed",
    "asal": "absolute",
}

SAL_VARIANTS = xmisc.Choices(
    {None: "No restriction", "sal": "Salinity of unknown kind", **_SAL_KINDS},
    parameter="variant",
    description="Restrict checking to a given variant(s)",
    aliases=_SAL_ALIASES,
    multi=True,
)

SAL_KINDS = xmisc.Choices(
    {None: "Inferred from the array, else practical", **_SAL_KINDS},
    parameter="sal_kind",
    description="Kind of salinity",
    aliases=_SAL_ALIASES,
)


def _get_sal_variant_(variant):
    variant = SAL_VARIANTS[variant]
    if variant is None:
        variant = ["sal", "psal", "pfsal", "asal"]
    return variant


@SAL_VARIANTS.format_method_docstring
def is_sal(da, variant=None):
    """Check if ``da`` is a salinity-like array

    Parameters
    ----------
    da: xarray.DataArray
    {variant}

    Return
    ------
    bool

    See also
    --------
    get_sal
    is_temp
    is_dens
    """
    variant = _get_sal_variant_(variant)
    return bool(xmeta.get_meta_specs(da).match_data_var(da, variant))


@SAL_VARIANTS.format_method_docstring
def get_sal(ds, variant=None, errors="warn"):
    """Search for a salinity variable in a dataset

    Parameters
    ----------
    ds: xarray.Dataset
    {variant}
    errors: str
        Error handling: ``"ignore"``, ``"warn"`` or ``"raise"``.

    Return
    ------
    xarray.DataArray, None
        The salinity array, or None if not found.

    See also
    --------
    is_sal
    get_temp
    get_dens
    """
    variant = _get_sal_variant_(variant)
    return xmeta.get_meta_specs(ds).get(ds, variant, errors=errors)
    # return _get_array_(ds, is_sal, SAL_VARIANTS, variant, errors, "salinity")


DENS_VARIANTS = xmisc.Choices(
    {
        None: "No restriction",
        "dens": "In situ density",
        "pdens": "Potential density",
        "ndens": "Neutral density",
    },
    parameter="variant",
    description="Restrict checking to a given variant(s)",
    aliases={
        "dens": ["insitu", "sigmat"],
        "pdens": ["potential", "sigma0", "sigma1", "sigma2", "sigma3", "sigma4"],
        "ndens": "neutral",
    },
    multi=True,
)


def _get_dens_variant_(variant):
    variant = DENS_VARIANTS[variant]
    if variant is None:
        variant = [
            "dens",
            "sigmat",
            "pdens",
            "sigma0",
            "sigma1",
            "sigma2",
            "sigma3",
            "sigma4",
            "ndens",
        ]
    return variant


@DENS_VARIANTS.format_method_docstring
def is_dens(da, variant=None):
    """Check if ``da`` is a density-like array

    Parameters
    ----------
    da: xarray.DataArray
    {variant}

    Return
    ------
    bool

    See also
    --------
    get_dens
    is_temp
    is_sal
    """
    meta_specs = xmeta.get_meta_specs(da)
    variant = _get_dens_variant_(variant)
    return bool(meta_specs.match_data_var(da, variant))
    # meta_names = []
    # if variant is None or variant == "insitu":
    #     meta_names.extend(["dens", "sigmat"])
    # if variant is None or variant == "potential":
    #     meta_names.extend(["pdens", "sigmatheta", "sigma0", "sigma1", "sigma2", "sigma3", "sigma4"])
    # if variant is None or variant == "neutral":
    #     meta_names.append("ndens")
    # for meta_name in meta_names:
    #     if meta_specs.match_data_var(da, meta_name):
    #         return True
    # return False


@DENS_VARIANTS.format_method_docstring
def get_dens(ds, variant=None, errors="warn"):
    """Search for a density variable in a dataset

    Parameters
    ----------
    ds: xarray.Dataset
    {variant}
    errors: str
        Error handling: ``"ignore"``, ``"warn"`` or ``"raise"``.

    Return
    ------
    xarray.DataArray, None
        The density array, or None if not found.

    See also
    --------
    is_dens
    get_temp
    get_sal
    """
    variant = _get_dens_variant_(variant)
    return xmeta.get_meta_specs(ds).get(ds, variant, errors=errors)
    # return _get_array_(ds, is_dens, DENS_VARIANTS, variant, errors, "density")


def _format_doc_(*choices):
    """Decorator that fills the choices placeholders of a docstring"""

    def decorator(func):
        func.__doc__ = func.__doc__.format(**{c._parameter: c.to_docstring(4) for c in choices})
        return func

    return decorator


def _get_temp_kind_(temp, kind):
    """Get the kind of temperature, that is potential unless found otherwise"""
    kind = TEMP_KINDS[kind]
    if kind is not None:
        return kind
    meta_specs = xmeta.get_meta_specs(temp)
    for kind in "ptemp", "ctemp":
        if meta_specs.match_data_var(temp, kind):
            return kind
    if temp.attrs.get("standard_name") == "sea_water_temperature":
        return "temp"
    return "ptemp"


def _get_sal_kind_(sal, kind):
    """Get the kind of salinity, that is practical unless found otherwise"""
    kind = SAL_KINDS[kind]
    if kind is not None:
        return kind
    meta_specs = xmeta.get_meta_specs(sal)
    for kind in "asal", "pfsal":
        if meta_specs.match_data_var(sal, kind):
            return kind
    return "psal"


def _as_celsius_(temp):
    if str(temp.attrs.get("units", "")).lower() in ("k", "kelvin"):
        temp = temp - 273.15
    return temp


def _get_zlatlon_(da, z, lat, lon):
    """Get the vertical coordinate, latitudes and longitudes, as needed for the pressure"""
    z = xcoords.get_z(da) if z is None else xcoords.as_z(z)
    if lat is None:
        lat = xcoords.get_lat(da, errors="ignore")
        if lat is None:
            raise exceptions.XoaError("The latitude is needed to compute the pressure: pass lat=")
    if lon is None:
        lon = xcoords.get_lon(da, errors="ignore")
        if lon is None:
            lon = 0.0
    return z, lat, lon


def _format_(out, da, name):
    out = xcoords.copy_coord_attrs(out, da)
    xmeta.get_meta_specs(da).format_data_var(
        out, name, format_coords=False, rename_dims=False, copy=False, replace_attrs=True
    )
    return out


@_format_doc_(SAL_KINDS)
def convert_sal(sal, to="asal", sal_kind=None, z=None, lat=None, lon=None):
    """Convert a salinity from a kind to another one

    Parameters
    ----------
    sal: xarray.DataArray
        Salinity
    to: str
        Target kind, among ``"psal"``, ``"asal"`` and ``"pfsal"``, or their aliases.
    {sal_kind}
    z: None, xarray.DataArray
        Vertical coordinate, which defaults to the z or depth of `sal`.
        See :func:`xoa.coords.as_z` for the sign convention.
    lat, lon: None, float, xarray.DataArray
        Latitudes and longitudes, that default to the coordinates of `sal`.
        The latitude is needed to compute the pressure from the depth.

    Return
    ------
    xarray.DataArray

    See also
    --------
    convert_temp
    """
    to = SAL_KINDS[to]
    sal_kind = _get_sal_kind_(sal, sal_kind)
    if to == sal_kind:
        return _format_(sal.copy(), sal, to)
    z, lat, lon = _get_zlatlon_(sal, z, lat, lon)
    out = xr.apply_ufunc(
        cthermdyn.convert_sal,
        sal,
        z,
        lat,
        lon,
        kwargs={"kind": sal_kind, "to": to},
        join="override",
        dask="parallelized",
        output_dtypes=[float],
    )
    return _format_(out, sal, to)


@_format_doc_(TEMP_KINDS)
def convert_temp(
    temp, to="ctemp", temp_kind=None, sal=None, sal_kind=None, z=None, lat=None, lon=None
):
    """Convert a temperature from a kind to another one

    Parameters
    ----------
    temp: xarray.DataArray
        Temperature in degrees Celsius or Kelvin
    to: str
        Target kind, among ``"temp"``, ``"ptemp"`` and ``"ctemp"``, or their aliases.
    {temp_kind}
    sal: xarray.DataArray, None
        Salinity, which is needed when the kinds differ.
    sal_kind: str, None
        Kind of `sal`, as in :func:`convert_sal`.
    z: None, xarray.DataArray
        Vertical coordinate, which defaults to the z or depth of `temp`.
        See :func:`xoa.coords.as_z` for the sign convention.
    lat, lon: None, float, xarray.DataArray
        Latitudes and longitudes, that default to the coordinates of `temp`.
        The latitude is needed to compute the pressure from the depth.

    Return
    ------
    xarray.DataArray
        In degrees Celsius.

    See also
    --------
    convert_sal
    """
    to = TEMP_KINDS[to]
    temp_kind = _get_temp_kind_(temp, temp_kind)
    temp = _as_celsius_(temp)
    if to == temp_kind:
        return _format_(temp.copy(), temp, to)
    if sal is None:
        raise exceptions.XoaThermdynError("The salinity is needed to convert the temperature")
    sal_kind = _get_sal_kind_(sal, sal_kind)
    z, lat, lon = _get_zlatlon_(temp, z, lat, lon)
    out = xr.apply_ufunc(
        cthermdyn.convert_temp,
        temp,
        sal,
        z,
        lat,
        lon,
        kwargs={"kind": temp_kind, "to": to, "sal_kind": sal_kind},
        join="override",
        dask="parallelized",
        output_dtypes=[float],
    )
    return _format_(out, temp, to)


MLD_METHODS = xmisc.Choices(
    {
        "deltatemp": (
            "depth at which the temperature is ``deltatemp`` "
            "lower than the surface temperature"
        ),
        "deltadens": (
            "depth at which the potential density is `deltadens` higher than the surface density"
        ),
        "kzmax": "depth at which the vertical diffusivity value reaches the `kzmax` value",
    },
    parameter="method",
    description="Method for computing the mixed layer depth",
)


@_format_doc_(TEMP_KINDS, MLD_METHODS)
def mixed_layer_depth(
    da,
    method=None,
    zref=0.0,
    temp_kind=None,
    deltatemp=0.2,
    deltadens=0.03,
    kzmax=0.0005,
    zdim=None,
    dask="parallelized",
    **kwargs,
):
    """Compute the mixed layer depth with different methods

    Parameters
    ----------
    da: xarray.DataArray
        A data array that contains either the temperature,
        the potential density or the vertical tracer diffusivity.
        This array **must contain a depth or z coordinate**, which should have
        a ``positive`` attribute.
    {method}
    zref: float
        Reference depth (in meters) from which the method criterion
        is applied. Defaults to the surface (0).
    {temp_kind}
        It is only used by the ``"deltatemp"`` method, for which the in-situ temperature
        should be avoided.
    deltatemp: float
        Temperature threshold for the ``"deltatemp"`` method (in K or degC).
    deltadens: float
        Density threshold for the ``"deltadens"`` method (in kg/m3).
    kzmax: float
        Diffusivity threshold for the ``"kzmax"`` method (in m2/s).
    zdim: str, None
        Name of the vertical dimension. Inferred by default.
    dask: str
        See :func:`xarray.apply_ufunc`.
    kwargs: dict
        Extra keywords are passed to :func:`xoa.interp.isoslice`.

    Raises
    ------
    XoaThermdynError
        When ``method`` is ``None`` and cannot be inferred from ``da``.

    Return
    ------
    xarray.DataArray
        Mixed layer depth as a positive value.

    Example
    -------
    Compute the MLD from a temperature profile::

        mld = mixed_layer_depth(temp, method="deltatemp")

    See Also
    --------
    xoa.interp.isoslice
    is_temp
    is_dens
    """
    # Vertical dimension
    if zdim is None:
        zdim = xcoords.get_zdim(da)
    else:
        assert zdim in da.dims
    assert zdim in da.dims

    # Depths, as found, since the order of levels follows their positive attribute
    depth = xcoords.get_vertical(da, errors="ignore")
    if depth is None:
        depth = xcoords.get_depth(da)
    positive = xcoords.get_positive_attr(depth, zdim=zdim) or "up"

    # Method
    meta_specs = xmeta.get_meta_specs(da)
    if method is None:
        if is_temp(da):
            method = "deltatemp"
        elif meta_specs.match_data_var(da, 'kz'):
            method = "kzmax"
        elif is_dens(da):
            method = "deltadens"
        else:
            raise exceptions.XoaThermdynError(
                "Cannot infer mixed layer depth computation method from data array. "
                "Please specify the `method` keyword."
            )
    else:
        method = MLD_METHODS[method]

    # Iso value
    if method == "kzmax":
        isoval = kzmax
    else:
        if zref == 0.0:
            surf = da.isel({zdim: 0 if positive == "down" else -1})
        else:
            dep0 = xr.DataArray([zref], dims="depth")
            surf = xregrid.regrid1d(da, dep0, method="linear").squeeze(dim="depth")
        if method == "deltatemp":
            if _get_temp_kind_(da, temp_kind) == "temp":
                exceptions.xoa_warn(
                    "The in-situ temperature is not conserved by vertical displacements: "
                    "prefer the potential or conservative temperature"
                )
            isoval = surf - deltatemp
        elif method == "deltadens":
            isoval = surf + deltadens

    # Slice
    mld = xregrid.isoslice(depth, da, isoval, dim=zdim, dask=dask, **kwargs)
    mld = np.abs(mld)
    mld = mld.drop_vars(depth.name, errors="ignore")

    # Format
    mld.attrs = {}
    meta_specs.format_data_var(
        mld, "mld", format_coords=False, rename_dims=False, copy=False, replace_attrs=True
    )

    return mld


# def get_mld(ds, methods=[None, "deltadens", "deltatemp", "kz"], **kwargs):
#     """Get or compute the mixed layer depth from a dataset"""
#     meta_specs = xmeta.get_meta_specs(ds)
#     for method in methods:
#         if method is None:
#             mld = meta_specs.search_data_var(ds, "mld", errors="ignore")
#             if mld is None:


@_format_doc_(TEMP_KINDS, SAL_KINDS)
def get_n2(temp, sal, z=None, lat=None, lon=None, temp_kind=None, sal_kind=None):
    """Compute the squared Brunt-Vaisala frequency, or buoyancy frequency, in s-2

    It is computed with :func:`gsw.Nsquared` between the levels, and averaged
    to the levels, so that it can be used with the data that are defined at the levels,
    like in :class:`xoa.dyn.VerticalModes`.

    Parameters
    ----------
    temp: xarray.DataArray
        Temperature in degrees Celsius or Kelvin. It must have a vertical dimension.
    sal: xarray.DataArray
        Salinity
    z: None, xarray.DataArray
        Vertical coordinate of the levels, that may vary in space, like with sigma
        coordinates. It defaults to the z or depth of `temp`.
        See :func:`xoa.coords.as_z` for the sign convention.
    lat, lon: None, float, xarray.DataArray
        Latitudes and longitudes, that default to the coordinates of `temp`.
        The latitude is needed to compute the pressure from the depth, and the longitude,
        that defaults to 0, only slightly modifies the absolute salinity.
    {temp_kind}
    {sal_kind}

    Return
    ------
    xarray.DataArray
        With the same dimensions as `temp`. It is missing next to the missing values.

    See also
    --------
    convert_temp
    convert_sal
    xoa.dyn.VerticalModes
    """
    zdim = xcoords.get_zdim(temp, errors="raise")
    temp_kind = _get_temp_kind_(temp, temp_kind)
    sal_kind = _get_sal_kind_(sal, sal_kind)
    temp_c = _as_celsius_(temp)
    z, lat, lon = _get_zlatlon_(temp, z, lat, lon)
    n2 = xr.apply_ufunc(
        cthermdyn.n2_at_levels,
        temp_c,
        sal,
        z,
        lat,
        lon,
        input_core_dims=[[zdim], [zdim], [zdim], [], []],
        output_core_dims=[[zdim]],
        kwargs={"temp_kind": temp_kind, "sal_kind": sal_kind},
        join="override",
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    n2 = n2.transpose(*[d for d in temp.dims if d in n2.dims], ...)
    n2 = xcoords.copy_coord_attrs(n2, temp)
    n2.name = "n2"
    n2.attrs = {"long_name": "Squared Brunt-Vaisala frequency", "units": "s-2"}
    return n2
