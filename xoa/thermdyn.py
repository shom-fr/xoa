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


TEMP_VARIANTS = xmisc.Choices(
    {
        None: "No restriction",
        "temp": "In situ",
        "ctemp": "Conservative temperature",
        "atemp": "Absolute temperature",
        "ptemp": "Potential temperature",
    },
    parameter="variant",
    description="Restrict checking to a given variant(s)",
    aliases={"temp": "insitu", "ptemp": "potential", "ctemp": "conservative", "atemp": "absolute"},
    multi=True,
)


def _get_temp_variant_(variant):
    variant = DENS_VARIANTS[variant]
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


SAL_VARIANTS = xmisc.Choices(
    {
        None: "No restriction",
        "sal": "In situ salinity",
        "asal": "Absolute salinity",
        "pfsal": "Preformed salinity",
        "psal": "Practical salinity",
    },
    parameter="variant",
    description="Restrict checking to a given variant(s)",
    aliases={
        "sal": "insitu",
        "psal": "practical",
        "pfsal": "preformed",
        "asal": "absolute salinity",
    },
    multi=True,
)


def _get_sal_variant_(variant):
    variant = DENS_VARIANTS[variant]
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


MLD_METHODS = xmisc.Choices(
    {
        "deltatemp": (
            "depth at which the potential temperature is ``deltatemp`` "
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


@MLD_METHODS.format_method_docstring
def mixed_layer_depth(
    da,
    method=None,
    zref=0.0,
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
        A data array that contains either the potential temperature,
        the potential density or the vertical tracer diffusivity.
        This array **must contain a depth or z coordinate**, which should have
        a ``positive`` attribute.
    {method}
    zref: float
        Reference depth (in meters) from which the method criterion
        is applied. Defaults to the surface (0).
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


def _n2_at_levels_(temp, sal, z, lat, lon, ptemp):
    """Squared Brunt-Vaisala frequency at the levels of arrays whose last axis is the vertical"""
    import gsw

    lat = np.asarray(lat, dtype="d")[..., None]
    lon = np.asarray(lon, dtype="d")[..., None]
    p = gsw.p_from_z(z, lat)
    sa = gsw.SA_from_SP(sal, p, lon, lat)
    ct = gsw.CT_from_pt(sa, temp) if ptemp else gsw.CT_from_t(sa, temp, p)
    mid, _ = gsw.Nsquared(sa, ct, p, lat, axis=-1)
    out = np.empty(np.shape(temp), dtype="d")
    out[..., 1:-1] = 0.5 * (mid[..., :-1] + mid[..., 1:])
    out[..., 0] = mid[..., 0]
    out[..., -1] = mid[..., -1]
    return out


def get_n2(temp, sal, z=None, lat=None, lon=None, ptemp=True):
    """Compute the squared Brunt-Vaisala frequency, or buoyancy frequency, in s-2

    It is computed with :func:`gsw.Nsquared` between the levels, and averaged
    to the levels, so that it can be used with the data that are defined at the levels,
    like in :class:`xoa.dyn.VerticalModes`.

    Parameters
    ----------
    temp: xarray.DataArray
        Potential temperature in degrees Celsius, or in-situ temperature when `ptemp`
        is False. It must have a vertical dimension.
    sal: xarray.DataArray
        Practical salinity
    z: None, xarray.DataArray
        Vertical coordinate of the levels, that may vary in space, like with sigma
        coordinates. It defaults to the z or depth of `temp`.
        See :func:`xoa.coords.as_z` for the sign convention.
    lat, lon: None, float, xarray.DataArray
        Latitudes and longitudes, that default to the coordinates of `temp`.
        The latitude is needed to compute the pressure from the depth, and the longitude,
        that defaults to 0, only slightly modifies the absolute salinity.
    ptemp: bool
        The temperature is the potential one, as in most models, and not the in-situ one.

    Return
    ------
    xarray.DataArray
        With the same dimensions as `temp`. It is missing next to the missing values.

    See also
    --------
    xoa.dyn.VerticalModes
    """
    zdim = xcoords.get_zdim(temp, errors="raise")
    z = xcoords.get_z(temp) if z is None else xcoords.as_z(z)
    if lat is None:
        lat = xcoords.get_lat(temp, errors="ignore")
        if lat is None:
            raise exceptions.XoaError("The latitude is needed to compute the pressure: pass lat=")
    if lon is None:
        lon = xcoords.get_lon(temp, errors="ignore")
        if lon is None:
            lon = 0.0
    n2 = xr.apply_ufunc(
        _n2_at_levels_,
        temp,
        sal,
        z,
        lat,
        lon,
        input_core_dims=[[zdim], [zdim], [zdim], [], []],
        output_core_dims=[[zdim]],
        kwargs={"ptemp": ptemp},
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
