#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermodynamics utilities
"""
import numpy as np

import xarray as xr
from .__init__ import XoaError
from . import cf as xcf
from . import misc as xmisc
from . import coords as xcoords
from . import regrid as xregrid


def is_temp(da, potential=None):
    """Check if `da` is a temperature-like array.

    Parameters
    ----------
    da: xarray.DataArray
    potential: None, bool
        Consider the temperature to be potential or not.

    Return
    ------
    bool
    """
    cfspecs = xcf.get_cf_specs(da)
    cf_names = []
    if potential is None or potential is False:
        cf_names.append("temp")
    if potential is None or potential is True:
        cf_names.append("ptemp")
    for cf_name in cf_names:
        if cfspecs.match_data_var(da, cf_name):
            return True
    return False


def is_dens(da, potential=None):
    """Check if `da` is a density-like array.

    Parameters
    ----------
    da: xarray.DataArray
    potential: None, bool
        Consider the density to be potential/neutral or not.

    Return
    ------
    bool
    """
    cfspecs = xcf.get_cf_specs(da)
    cf_names = []
    if potential is None or potential is False:
        cf_names.extend(["dens", "sigmat"])
    if potential is None or potential is True:
        cf_names.append(
            ["pdens", "ndens", "sigmatheta", "sigma0", "sigma1", "sigma2", "sigma3", "sigma4"]
        )
    for cf_name in cf_names:
        if cfspecs.match_data_var(da, cf_name):
            return True
    return False


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
    zref=0.,
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
        if zref == 0.:
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
