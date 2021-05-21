# -*- coding: utf-8 -*-
"""
Terrain following parametric vertical coordinates

This follows the CF conventions for
`Parametric Vertical Coordinates <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord>`_.

"""
# Copyright 2020-2021 Shom
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

import re

import numpy as np

from .__init__ import xoa_warn
from . import misc
from . import cf
from . import coords as xcoords

#: To convert from formula terms to CF names
FORMULA_TERMS_TO_CF_NAMES = {
    'c': 'cs',  # C
    's': 'sig',
    'eta': 'ssh',
    'depth': 'bathy',
    'depth_c': 'hc',
    'a': 'thetas',
    'b': 'thetab'
    }

#: Supported sigma coordinates
SIGMA_COORDINATE_TYPES = (
    "atmosphere_sigma_coordinate",
    "ocean_sigma_coordinate",
    "ocean_s_coordinate"
    "ocean_s_coordinate_g1",
    "ocean_s_coordinate_g2"
    )


class XoaSigmaError(cf.XoaCFError):
    pass


def atmosphere_sigma_coordinate(sig, ps, ptop, cache=None):
    """Convert from sigma [0, 1] to altitude in an atmopsheric model

    Source:
        `Ocean sigma coordinate  <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_atmosphere_sigma_coordinate>`_

    Formula:
        .. math::

            p = p_{top} + \\sigma * (p_{surf}-p_{top})

    Sigma standard name:
        ``atmosphere_sigma_coordinate``

    Formula terms:
        ``sigma: var1 ps: var2 ptop: var3``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`\\sigma` | ``sigma``)
    ps: xarray.DataArray
        Surface air pressure (:math:`p_{surf}` | ``ps``)
    ptop: xarray.DataArray
        Air pressure at top of model (:math:`p_{top}` | ``ptop``)

    Returns
    -------
    xarray.DataArray
        Air pressure in Pa (:math:`p`)
    """
    # Compute
    p = sig * (ps - ptop)
    p += ptop

    # Format
    return cf.get_cf_specs(sig).format_data_var(p, "plev", format_coords=False)


def ocean_sigma_coordinate(sig, ssh, bathy, cache=None):
    """Convert from sigma [-1, 0] to negative depths in an ocean model

    Source:
        `Ocean sigma coordinate <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_ocean_sigma_coordinate>`_

    Formula:
        .. math::

            z = \\eta + \\sigma * (\\eta+h)

    Sigma standard name:
        ``ocean_sigma_coordinate``

    Formula terms:
        ``sigma: var1 eta: var2 depth: var3``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`\\sigma` | ``sigma``)
    ssh: xarray.DataArray
        Surface air pressure (:math:`\\eta` | ``eta``)
    bathy: xarray.DataArray
        Positive sea floor depth (:math:`h` | ``depth``)

    Returns
    -------
    xarray.DataArray
        Negative depth below surface in m (:math:`z`)
    """
    # Compute
    z = sig * (bathy + ssh)
    z += ssh

    # Format
    return cf.get_cf_specs(sig).format_data_var(z, "depth", format_coords=False)


def get_cs(sig, thetas, thetab, cs_type=None):
    """Get a s-coordinate stretching curve

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`s` | ``s``)
    thetas: xarray.DataArray
        Surface control parameter (:math:`a` | ``a``)
    thetab: xarray.DataArray
        Bottom control parameter (:math:`b` | ``b``)
    cs_type: str, None
        Stretching type:
            ``None`` (default):

             .. math::

                 C & = (1-b)*\\frac{\\sinh(a*s)}{\\sinh(a)} +  b*\\left[\\frac{\\tanh(a*(s+0.5))}{2*\\tanh(0.5*a)} - 0.5\\right]

    Returns
    -------
    xarray.DataArray
        Stretching curve (:math:`C` | ``C``)
    """
    s, a, b = sig, thetas, thetab
    cs = np.sinh(s*a) * (1-b) / np.sinh(a)
    cs += b * (np.tanh(a*(s+0.5))/(2*np.tanh(0.5*a))-0.5)
    return cs


def ocean_s_coordinate(sig, ssh, bathy, hc, thetas, thetab, cs=None,
                       cs_type=None, cache=None):
    """Convert from s [-1, 0] to depths in an ocean model

    Source:
        `Ocean s-coordinate <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_ocean_s_coordinate>`_

    Formula:
        .. math::

            z & = \\eta*(1+s) + h_c*s + (h-h_c)*C

            C & = (1-b)*\\frac{\\sinh(a*s)}{\\sinh(a)} +  b*\\left[\\frac{\\tanh(a*(s+0.5))}{2*\\tanh(0.5*a)} - 0.5\\right]

    Sigma standard name:
        ``ocean_s_coordinate``

    Formula terms:
        ``s: var1 eta: var2 depth: var3 a: var4 b: var5 depth_c: var6``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`s` | ``s``)
    ssh: xarray.DataArray
        Surface air pressure (:math:`\\eta` | ``eta``)
    bathy: xarray.DataArray
        Positive sea floor depth (:math:`h` | ``depth``)
    hc: xarray.DataArray
        Positive critical depth (:math:`h_c` | ``depth_c``)
    thetas: xarray.DataArray
        Surface control parameter (:math:`a` | ``a``)
    thetab: xarray.DataArray
        Bottom control parameter (:math:`b` | ``b``)
    cs: xarray.DataArray, None
        Stretching curve, which defaults to the formula above
        computed by :func:`get_cs` (:math:`C` | ``C``)
    cs_type: str, None
        Stretching type (see :func:`get_cs`)
    cache: dict
        Dict variable that stores intermediate results to be used
        from call to call.

    Returns
    -------
    xarray.DataArray
        Negative depth below surface in m (:math:`z`)
    """
    # Check cache first
    if cache:
        zconst = cache['zconst']

    else:  # Compute

        # Stetching curve
        if cs is None:
            cs = get_cs(sig, thetas, thetab, cs_type)

        # Constant part of the formula
        zconst = sig * hc
        zconst += cs*(bathy-hc)

        # Store in cache
        if isinstance(cache, dict):
            cache.update(zconst=zconst)

    # Final computation
    z = (sig+1) * ssh + zconst

    # Format
    return cf.get_cf_specs(sig).format_data_var(z, "depth", format_coords=False)


def ocean_s_coordinate_g1(sig, ssh, bathy, hc, thetas=None, thetab=None,
                          cs=None, cs_type=None, cache=None):
    """Convert from s [-1, 0] generic form 1 to depths in an ocean model

    Source:
        `Ocean s-coordinate, generic form 1 <http://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_s_coordinate_generic_form_1>`_

    Formula:
        .. math::

            z & = \\eta*(1+s) + (\\eta + h * S)

            S & = \\frac{h_c s + h C}{h_c + h}

            C & = (1-b)*\\frac{\\sinh(a*s)}{\\sinh(a)} +  b*\\left[\\frac{\\tanh(a*(s+0.5))}{2*\\tanh(0.5*a)} - 0.5\\right]



    Sigma standard name:
        ``ocean_s_coordinate_g2``

    Formula terms:
        ``s: var1 C: var2 eta: var3 depth: var4 depth_c: var5``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`s` | ``s``)
    ssh: xarray.DataArray
        Surface air pressure (:math:`\\eta` | ``eta``)
    bathy: xarray.DataArray
        Positive sea floor depth (:math:`h` | ``depth``)
    hc: xarray.DataArray
        Positive critical depth (:math:`h_c` | ``depth_c``)
    thetas: xarray.DataArray
        Surface control parameter (:math:`a` | ``a``)
        Optional if `cs` is provided.
    thetab: xarray.DataArray
        Bottom control parameter (:math:`b` | ``b``)
        Optional if `cs` is provided.
    cs: xarray.DataArray, None
        Stretching curve, which defaults to the formula above
        computed by :func:`get_cs` (:math:`C` | ``C``)
    cs_type: str, None
        Stretching type (see :func:`get_cs`)
    cache: dict
        Dict variable that stores intermediate results to be used
        from call to call.

    Returns
    -------
    xarray.DataArray
        Negative depth below surface in m (:math:`z`)
    """
    s, h = sig, bathy

    # Check cache first
    if cache:
        S = cache['S']
        Sh1 = cache['Sh1']

    else:  # Compute

        # Stetching curve
        if cs is None:
            if None in (thetas, thetab):
                raise XoaSigmaError("thetas and thetas must be provided when "
                                    "cs is not")
            cs = get_cs(sig, thetas, thetab, cs_type)

        # Constant part of the formula
        S = s * hc + cs * (h - hc)
        Sh1 = S / h
        Sh1 += 1

        # Store in cache
        if isinstance(cache, dict):
            cache.update(S=S, Sh1=Sh1)

    # Final computation
    z = S + Sh1 * ssh

    # Format
    return cf.get_cf_specs(sig).format_data_var(z, "depth", format_coords=False)


def ocean_s_coordinate_g2(sig, ssh, bathy, hc, thetas=None, thetab=None,
                          cs=None, cs_type=None, cache=None):
    """Convert from s [-1, 0] generic form 2 to depths in an ocean model

    Source:
        `Ocean s-coordinate, generic form 2 <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_ocean_s_coordinate_generic_form_2>`_

    Formula:
        .. math::

            z & = \\eta*(1+s) + (\\eta + h * S)

            S & = \\frac{h_c s + h C}{h_c + h}

            C & = (1-b)*\\frac{\\sinh(a*s)}{\\sinh(a)} +  b*\\left[\\frac{\\tanh(a*(s+0.5))}{2*\\tanh(0.5*a)} - 0.5\\right]

    Sigma standard name:
        ``ocean_s_coordinate_g2``

    Formula terms:
        ``s: var1 C: var2 eta: var3 depth: var4 depth_c: var5``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`s` | ``s``)
    ssh: xarray.DataArray
        Surface air pressure (:math:`\\eta` | ``eta``)
    bathy: xarray.DataArray
        Positive sea floor depth (:math:`h` | ``depth``)
    hc: xarray.DataArray
        Positive critical depth (:math:`h_c` | ``depth_c``)
    thetas: xarray.DataArray
        Surface control parameter (:math:`a` | ``a``).
        Optional if `cs` is provided.
    thetab: xarray.DataArray
        Bottom control parameter (:math:`b` | ``b``)
        Optional if `cs` is provided.
    cs: xarray.DataArray, None
        Stretching curve, which defaults to the formula above
        computed by :func:`get_cs` (:math:`C` | ``C``)
    cs_type: str, None
        Stretching type (see :func:`get_cs`)
    cache: dict
        Dict variable that stores intermediate results to be used
        from call to call.

    Returns
    -------
    xarray.DataArray
        Negative depth below surface in m (:math:`z`)
    """
    s, h = sig, bathy

    # Check cache first
    if cache:
        S = cache['S']
        hS = cache['hS']

    else:  # Compute

        # Stetching curve
        if cs is None:
            if None in (thetas, thetab):
                raise XoaSigmaError(
                    "thetas and thetas must be provided when cs is not")
            cs = get_cs(sig, thetas, thetab, cs_type)

        # Constant part of the formula
        S = s * hc + cs * h
        S /= hc + h
        hS = S * h

        # Store in cache
        if isinstance(cache, dict):
            cache.update(S=S, hS=hS)

    # Final computation
    z = hS + (S + 1) * ssh

    # Format
    return cf.get_cf_specs(sig).format_data_var(z, "depth", format_coords=False)


def _ds_search_ci_(ds, name):
    """Case insensitive search in data_vars and coords of a dataset

    Parameters
    ----------
    ds: xarray.Dataset
    name: str
        Requested name

    Returns
    -------
    str
        Real name
    """
    lname = name.lower()
    for cat in "data_vars", "coord":
        pool = getattr(ds, cat)
        names = [nm for nm in pool.keys()]
        lnames = [nm.lower() for nm in names]
        if lname in lnames:
            return names[lnames.index(lname)]


_re_ft_split_terms = re.compile(r'\b\s+\b').split

_re_ft_split_item = re.compile(r'\s*:\s*').split


def decode_formula_terms(attr):
    """Parse the formula_terms attribute

    Parameters
    ----------
    attr: str
        Attribute value

    Returns
    -------
    dict

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.sigma import decode_formula_terms
        decode_formula_terms(
            's: sc_r C: Cs_r eta: zeta depth: h depth_c: hc')
    """
    terms = {}
    for item in _re_ft_split_terms(attr):
        item = _re_ft_split_item(item)
        if len(item) != 2:
            raise XoaSigmaError("Malformed formula_terms attribute: "+attr)
        terms[item[0]] = item[1]
    return terms


def get_sigma_terms(ds, loc=None, rename=False):
    """Get sigma terms from a dataset as another dataset

    It operates like this:

    1. Search for the sigma variables.
    2. Parse their ``formula_terms`` attribute.
    3. Create a dict for each locations from names in datasets to
       :mod:`xoa.cf` compliant names that are also used in conversion
       functions.

    Parameters
    ----------
    ds: xarray.Dataset
    loc: str, {"any", None}
        Staggered grid location.
        If any or None, results for all locations are returned.

    Returns
    -------
    dict, dict of dict
        A dict is generated for a given sigma variable,
        whose keys are array names, like ``"sc_r"``,
        and values are :mod:`~xoa.cf` names, like ``"sig"``.
        A special key is the ``"type"`` whose corresponding value
        is the ``standard_name``, stripped from its potential staggered grid
        location indicator.
        If ``loc`` is ``"any"`` or ``None``,
        each dict is embedded in a master dict
        whose keys are staggered grid location. If no location is found,
        the key is set ``None``.

    Raises
    ------
    xoa.sigma.XoaSigmaError
        In case of:

        - inconsistent staggered grid location in dataarrays
          as checked by :meth:`xoa.cf.SGLocator.get_location`
        - no standard_name in sigma/s variable
        - a malformed formula
        - a formula term variable that is not found in the dataset
        - an unknown formula term name
    """
    # Get sigma arrays
    cfspecs = cf.get_cf_specs(ds)
    single = loc not in ("any", None)
    sigs = cfspecs.search(ds, 'sig', loc=loc, single=False)
    terms = {}
    for sig in sigs:

        # Check standard_name and get loc
        if "standard_name" not in sig.attrs:
            raise XoaSigmaError("No standard_name attribute found in sigma/s "
                                "variable name: "+sig.name)
        standard_name, loc = cfspecs.sglocator.parse_attr(
            "standard_name", sig.standard_name)
        if standard_name not in SIGMA_COORDINATE_TYPES:
            raise XoaSigmaError("Sigma/s coordinate not supported: " +
                                standard_name + ". Supported coordinates: "
                                + " ".join(SIGMA_COORDINATE_TYPES))

        # Get formula terms
        if "formula_terms" not in sig.attrs:
            raise XoaSigmaError(f"Sigma/s type variable {sig.name} "
                                "has no formula_term attribute")
        formula_terms = decode_formula_terms(sig.formula_terms)

        # Check terms
        subterms = {sig.name: "sig", "type": standard_name}
        for fname, fvname in formula_terms.items():

            # Real name
            vname = _ds_search_ci_(ds, fvname)
            if vname is None:
                raise XoaSigmaError("Formula array not found: "+fvname)

            # xoa.cf name
            # TODO: handle mising cs and fallback with thetas and thetab
            if fname.lower() not in FORMULA_TERMS_TO_CF_NAMES:
                raise XoaSigmaError("Unknown formula term name: "+fname)
            cf_name = FORMULA_TERMS_TO_CF_NAMES[fname.lower()]
            subterms[vname] = cf_name

        terms[loc] = subterms

    if single:
        return subterms if sigs else None
    return terms


    # # Rename terms in dict or ds
    # ds = ds.rename({sigma.name: "sig"})
    # for term_name, da_name_ in terms.items():
    #     da_name = _ds_search_ci_(ds, da_name_)
    #     if da_name is None:
    #         xoa_warn("Formula term dataarray not found: " + da_name_)
    #     ds = ds.rename({da_name: term_name})
    # return ds

@misc.ERRORS.format_function_docstring
def decode_cf_sigma(ds, rename=False, errors="raise"):
    """Compute heights from sigma-like variable in a dataset

    This makes use of the :meth:`~xoa.cf.CFSpecs` instance that is retreived
    with :func:`xoa.cf.get_cf_specs` with ds as an argument in order to
    find needed variables.
    If the dataset is not found to have sigma-like coordinates,
    a simple copy is returned.

    When a data variable that have the same dimensions is found, the
    the computed coordinate is transposed properly and assigned
    to the variable as a a coordinate array.

    Parameters
    ----------
    ds: xarray.Dataset
        Dataset that contains everything needed to compute heights
    rename: bool
        Rename and format arrays ot make them compliant with
        :mod:`xoa.cf`
    {errors}

    Return
    ------
    xarray.Dataset
    """
    # Init
    ds = ds.copy()
    cfspecs = cf.get_cf_specs(ds)
    errors = misc.ERRORS[errors]

    # Decode formula terms
    try:
        all_terms = get_sigma_terms(ds, loc=None)
        if all_terms is None:
            return ds
    except XoaSigmaError as e:
        if errors == "raise":
            raise e
        if errors == "warn":
            xoa_warn("Error while decoding sigma coords: {}. Skipping..."
                     .format(e))
        return ds
    # print('termssss', all_terms)

    # Loop on locations
    from . import sigma as sigma_module
    for loc, terms in all_terms.items():

        # Args: keys are cf names, values ara dataarrays
        sigma_type = terms.pop("type")
        kwargs = {}
        for vname, cf_name in terms.items():
            kwargs[cf_name] = ds[vname]

        # Compute depth/altitude/pressure
        func = getattr(sigma_module, sigma_type)
        cache = {}
        kwargs['cache'] = cache
        height = func(**kwargs)

        # Format
        height = cfspecs.sglocator.format_dataarray(height, loc)

        # Transpose if approriate and set as coordinate
        transposed = False
        as_coord = False
        for da in ds.data_vars.values():
            if set(height.dims) <= set(da.dims):
                if not transposed:
                    height = xcoords.transpose(height, da, "compat")
                    transposed = True
                ds[da.name] = da.assign_coords({height.name: height})
                as_coord = True

        # Make sure it is in the dataset
        if not as_coord:
            ds.coords[height.name] = height

    return ds
