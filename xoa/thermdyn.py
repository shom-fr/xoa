#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermodynamics utilities
"""
import numpy as np

import xarray as xr
from .__init__ import XoaError, xoa_warn
from . import cf as xcf
from . import misc as xmisc
from . import coords as xcoords
from . import regrid as xregrid


def _get_array_(ds, func, variants, variant, errors, name):
    if not variant:
        variant = None
    if isinstance(variant, str) or variant is None:
        variant = [variant]

    das = []
    for da in ds.values():
        for vr in variant:
            if func(da, variant):
                das.append(da)

    errors = xmisc.ERRORS[errors]
    if not das:
        msg = f"Found no {name} array"
        if errors == "raise":
            raise XoaError(msg)
        if errors == "warning":
            xoa_warn(msg)
        return
    if len(das) > 1:
        msg = f"Found more than one {name} array"
        if errors == "raise":
            raise XoaError(msg)
        if errors == "warning":
            xoa_warn(msg, stacklevel=3)
    return das[0]


TEMP_VARIANTS = xmisc.Choices(
    {
        None: "No restriction",
        "insitu": "In situ",
        "conservative": "Conservative temperature",
        "absolute": "Absolute temperature",
    },
    parameter="variant",
    description="Restrict checking to a given variant",
)


@TEMP_VARIANTS.format_method_docstring
def is_temp(da, variant=None):
    """Check if `da` is a temperature-like array

    Parameters
    ----------
    da: xarray.DataArray
    {variant}

    Return
    ------
    bool
    """
    cfspecs = xcf.get_cf_specs(da)
    variant = TEMP_VARIANTS[variant]
    cf_names = []
    if variant is None or variant == "insitu":
        cf_names.append("temp")
    if variant is None or variant == "potential":
        cf_names.append("ptemp")
    if variant is None or variant == "conservative":
        cf_names.append("ctemp")
    for cf_name in cf_names:
        if cfspecs.match_data_var(da, cf_name):
            return True
    return False


@xmisc.ERRORS.format_function_docstring
def get_temp(ds, variant=None, errors="warn"):
    """Search for temperature in a dataset

    Parameters
    ----------
    da: xarray.Dataset
    variant: None, str, list(str)
        Variant of temperature or list of them. See :func:`is_temp`.
    {errors}

    Return
    ------
    xarray.DataArray, None
        If None or several arrays are found, a warning or an error may be raised
        depending on the `errors` parameter.
        If several arrays are matching and `errors` is "warning", the first array is returned.

    """
    return _get_array_(ds, is_temp, TEMP_VARIANTS, variant, errors, "temperature")


SAL_VARIANTS = xmisc.Choices(
    {
        None: "No restriction",
        "insitu": "In situ salinity",
        "absolute": "Absolute salinity",
        "preformed": "Preformed salinity",
        "practical": "Practical salinity",
    },
    parameter="variant",
    description="Restrict checking to a given variant",
)


@SAL_VARIANTS.format_method_docstring
def is_sal(da, variant=None):
    """Check if `da` is a salinity-like array.

    Parameters
    ----------
    da: xarray.DataArray
    {variant}

    Return
    ------
    bool
    """
    cfspecs = xcf.get_cf_specs(da)
    cf_names = []
    if variant is None or variant == "insitu":
        cf_names.append("sal")
    if variant is None or variant == "practical":
        cf_names.append("psal")
    if variant is None or variant == "preformed":
        cf_names.append("pfsal")
    if variant is None or variant == "absolute":
        cf_names.append("asal")
    for cf_name in cf_names:
        if cfspecs.match_data_var(da, cf_name):
            return True
    return False


@xmisc.ERRORS.format_function_docstring
def get_sal(ds, variant=None, errors="warn"):
    """Search for salinity in a dataset.

    Parameters
    ----------
    da: xarray.Dataset
    variant: None, str, list(str)
        Variant of salinity or list of them. See :func:`is_sal`.
    {errors}

    Return
    ------
    xarray.DataArray, None
        Return None if not found
    """
    return _get_array_(ds, is_sal, SAL_VARIANTS, variant, errors, "salinity")


DENS_VARIANTS = xmisc.Choices(
    {
        None: "No restriction",
        "insitu": "In situ density",
        "potential": "Potential density",
        "neutral": "Neutral salinity",
    },
    parameter="variant",
    description="Restrict checking to a given variant",
)


@DENS_VARIANTS.format_method_docstring
def is_dens(da, variant=None):
    """Check if `da` is a density-like array.

    Parameters
    ----------
    da: xarray.DataArray
    {variant}

    Return
    ------
    bool
    """
    cfspecs = xcf.get_cf_specs(da)
    cf_names = []
    if variant is None or variant == "insitu":
        cf_names.extend(["dens", "sigmat"])
    if variant is None or variant == "potential":
        cf_names.extend(["pdens", "sigmatheta", "sigma0", "sigma1", "sigma2", "sigma3", "sigma4"])
    if variant is None or variant == "neutral":
        cf_names.append("ndens")
    for cf_name in cf_names:
        if cfspecs.match_data_var(da, cf_name):
            return True
    return False


@xmisc.ERRORS.format_function_docstring
def get_dens(ds, variant=None, errors="warn"):
    """Search for density in a dataset.

    Parameters
    ----------
    da: xarray.Dataset
    variant: None, str, list(str)
        Variant of density or list of them. See :func:`is_dens`.
    {errors}

    Return
    ------
    xarray.DataArray, None
        Return None if not found
    """
    return _get_array_(ds, is_dens, DENS_VARIANTS, variant, errors, "density")


MLD_METHODS = xmisc.Choices(
    {
        "deltatemp": (
            "depth at which the potential temperature is `deltadtemp` "
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
    deltadens=0.3,
    kzmax=0.0005,
    zdim=None,
    dask="parallelized",
    **kwargs,
):
    """Compute the mixed layer depth with different methods.

    Parameters
    ----------
    da: xarray.DataArray
        A data array that contains either the potential temperature, the potential density or
        the vertical tracer diffisivity.
        Note that this array **must contain a depth coordinate**, which optimally contains
        the `positive` attribute.
    {method}
    zref: float
        Reference depth (in meters) from which the MLD_METHODS are applied
    kwargs: dict
        Extra keywords are passed to :func:`xoa.regrid.isoslice`.

    Raise
    -----
    XoaError:
        When `method` is `None` and cannot be inferred from `da`.

    Return
    ------
    xarray.DataArray
        Mixed layer depth

    See Also
    --------
    xoa.regrid.isoslice
    """
    # Vertical dimension
    if zdim is None:
        zdim = xcoords.get_zdim(da)
    else:
        assert zdim in da.dims
    assert zdim in da.dims

    # Depths
    depth = xcoords.get_depth(da)
    positive = xcoords.get_positive_attr(depth, zdim=zdim) or "up"

    # Method
    cfspecs = xcf.get_cf_specs(da)
    if method is None:
        if is_temp(da):
            method = "deltatemp"
        elif cfspecs.match_data_var(da, 'kz'):
            method = "kzmax"
        elif is_dens(da):
            method = "deltadens"
        else:
            raise XoaError(
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
    cfspecs.format_data_var(
        mld, "mld", format_coords=False, rename_dims=False, copy=False, replace_attrs=True
    )

    return mld


# def get_mld(ds, methods=[None, "deltadens", "deltatemp", "kz"], **kwargs):
#     """Get or compute the mixed layer depth from a dataset"""
#     cfspecs = xcf.get_cf_specs(ds)
#     for method in methods:
#         if method is None:
#             mld = cfspecs.search_data_var(ds, "mld", errors="ignore")
#             if mld is None:
