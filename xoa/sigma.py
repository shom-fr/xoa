# -*- coding: utf-8 -*-
"""
Terrain following parametric coordinates

This follows the CF conventions for
`Parametric Vertical Coordinates<http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord>`_.


"""
import re

from .__init__ import XoaError, xoa_warn
from . import misc
from . import cf

#: To convert from formula terms to CF names
FORMULA_TERMS_TO_CF_NAMES = {
    'C': 'cs',
    's': 'sig',
    'eta': 'ssh',
    'depth': 'bathy',
    'depth_c': 'hc',
    'a': 'sigmas',
    'b': 'sigmab'
    }


# def atmosphere_sigma_to_altitude(sig, oro, topheight):
#     """Convert from sigma [0, 1] to altitude in an atmopsheric model

#     Formula:

#     .. math::

#         h_{bot} + \sigma * (h_{top}-h_{bot}) / h_{top}

#     Parameters
#     ----------
#     sig: xarray.DataArray
#         Sigma coordinates range from 0 to 1 (:math:`\sigma`)
#     oro:: xarray.DataArray
#         Orographie, i.e altitude of the ground (:math:`h_{bot}`)
#     topheight:: xarray.DataArray
#         Height of the top of the model (:math:`h_{top}`)

#     Returns
#     -------
#     xarray.DataArray
#         Altitudes in meters (:math:`Z`)
#     """
#     # Compute
#     altitude = sig * (topheight - oro)
#     altitude /= topheight
#     altitude += oro

#     # Format
#     return cf.get_cf_specs().format_data_var(
#         altitude, "altitude", format_coords=False)


def atmosphere_sigma_to_pressures(sig, ps, ptop):
    """Convert from sigma [0, 1] to altitude in an atmopsheric model

    Source: `Ocean sigma coordinate  <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_atmosphere_sigma_coordinate>`_

    Formula:

    .. math::

        p = p_{top} + \sigma * (p_{surf}-p_{top})

    Sigma standard name: ``atmosphere_sigma_coordinate``

    Formula terms: ``sigma: var1 ps: var2 ptop: var3``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`\sigma` | ``sigma``)
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
    return cf.get_cf_specs().format_data_var(p, "plev", format_coords=False)


def ocean_sigma_to_depths(sig, ssh, bathy):
    """Convert from sigma [-1, 0] to negative depths in an ocean model

    Source: `Ocean sigma coordinate <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_ocean_sigma_coordinate>`_

    Formula:

    .. math::

        z = \eta + \sigma * (\eta+h)

    Sigma standard name: ``ocean_sigma_coordinate``

    Formula terms: ``sigma: var1 eta: var2 depth: var3``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`\sigma` | ``sigma``)
    ssh: xarray.DataArray
        Surface air pressure (:math:`\eta` | ``eta``)
    bathy: xarray.DataArray
        Positive sea floor depth (:math:`h` | ``depth``)

    Returns
    -------
    xarray.DataArray
        Negative depth below surface in m (:math:`z`)
    """
    # Compute
    z = sig * (bathy + ssh)
    p += eta

    # Format
    return cf.get_cf_specs().format_data_var(z, "depth", format_coords=False)


def ocean_s_to_depths(sig, ssh, bathy, hc, thetas, thetab, cs=None):
    """Convert from s [-1, 0] to depths in an ocean model

    Source: `Ocean s-coordinate <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#_ocean_s_coordinate>`_

    Formula:

    .. math::

        z & = \\eta*(1+s) + h_c*s + (h-h_c)*C

        C & = (1-b)*\\frac{\\sinh(a*s)}{\\sinh(a)} +  b*\\left[\\frac{\\tanh(a*(s+0.5))}{2*\\tanh(0.5*a)} - 0.5\\right]



    Sigma standard name: ``ocean_sigma_coordinate``

    Formula terms: ``s: var1 eta: var2 depth: var3 a: var4 b: var5 depth_c: var6``

    Parameters
    ----------
    sig: xarray.DataArray
        Sigma coordinates range from 0 to 1 (:math:`s` | ``s``)
    ssh: xarray.DataArray
        Surface air pressure (:math:`\eta` | ``eta``)
    bathy: xarray.DataArray
        Positive sea floor depth (:math:`h` | ``depth``)
    hc: xarray.DataArray
        Positive critical depth (:math:`h_c` | ``hc``)
    thetas: xarray.DataArray
        Surface control parameter (:math:`a` | ``a``)
    thetab: xarray.DataArray
        Bottom control parameter (:math:`b` | ``b``)
    cs: xarray.DataArray, None
        Stretching curve, which defaults to the formula above
        (:math:`C` | ``C``)

    Returns
    -------
    xarray.DataArray
        Negative depth below surface in m (:math:`z`)
    """
    # Compute
    #TODO: ocean_s_to_depths
    z = sig * (ssh + bathy)
    p += eta

    # Format
    return cf.get_cf_specs().format_data_var(z, "depth", format_coords=False)




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
    name = name.lower()
    for cat in "data_vars", "coord":
        pool = getattr(ds, cat)
        names = [nm.lower() for nm in pool.keys()]
        lc_names = [nm.lower() for nm in names]
        if name in lc_names:
            return names[lc_names.index(name)]


_re_ft_split_terms = re.compile(r'\b\s+\b').split

_re_ft_split_item = re.compile(r'\s*:\s*').split


def parse_formula_terms_attr(attr):
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
        from xoa.sigma import parse_formula_terms_attr
        parse_formula_terms_attr(
            's: sc_r C: Cs_r eta: zeta depth: h depth_c: hc')
    """
    terms = {}
    for item in _re_ft_split_terms(attr):
        item = _re_ft_split_item(item)
        if len(item) != 2:
            xoa_warn('Malformed formula_terms attribute: '+attr)
        else:
            terms[item[0]] = item[1]
    return terms


def get_sigma_terms(ds, loc="any"):
    """Get sigma terms from a dataset as another dataset

    It operates like this:

    1. Search for the CF level variable.
    2. Parse its ``formula_terms`` attribute.
    3. Create a new dataset with all these terms, properly renamed.

    Parameters
    ----------
    ds: xarray.Dataset
    loc: str, "any"
        Staggered grid location

    Returns
    -------
    xarray.Dataset
    """
    # Get sigma
    cfspecs = cf.get_cf_specs()
    sigma = cfspecs.search(ds, 'sigma_level', loc=loc)
    if sigma is None or "formula_terms" not in sigma.attrs:
        return

    # Get requested terms
    terms = parse_formula_terms_attr(sigma.formula_terms)

    # Rename terms
    ds = ds.rename({sigma.name: "sigma_level"})
    for term_name, da_name_ in terms.items():
        da_name = _ds_search_ci_(ds, da_name_)
        if da_name is None:
            xoa_warn("Formula term dataarray not found: " + da_name_)
        ds = ds.rename({da_name: term_name})
    return ds


class sigma_types(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported sigma/s coordinates types"""
    #: Atmosphere sigma coordinate
    atmosphere_sigma = 1
    atmosphere_sigma_coordinate = 1
    #: Ocean sigma coordinate
    ocean_sigma = -1
    ocean_sigma_coordinate = -1
    #: Ocean s coordinate
    ocean_s = -2
    ocean_s_coordinates = -2
    #: Generic ocean s coordinate of form 1
    ocean_s_g1 = -3
    ocean_s_coordinate_g1 = -3
    #: Generic ocean s coordinate of form 2
    ocean_s_g2 = -4
    ocean_s_coordinate_g2 = -4


def _sigma2coord_(sig, zremote, zref, sigma_type,
                  cs=None, hc=None, thetas=None, thetab=None,
                  zerolid=False):
    """Conversion from sigma-like coordinates to depths

    :Params:

        - **sigma**: Sigma levels (abs(sigma)<1) as an 1D array.
        - **depth**: Bottom depth.
        - **eta**, optional: Sea surface elevation (with a time axis or not).
        - **stype**, optional: Sigma coordinates type

            - ``"standard"`` or ``0``: Standard.
            - ``"ocean"`` or ``1``: Ocean standard.
            - ``"generalized"`` or ``2``: Generalized (s) coordinates.

        - **cs**, optional: Stretching function (s coords only).
          If not provided, it is computed from stretching parameters.
        - **depth_c**, optional: Surface limit depth (s coords only).
        - **a**, optional: Surface control parameter (s coords only).
        - **b**, optional: Bottom control parameter (s coords only).
        - **zerolid**, optional: The surface is put at a zero depth to simulate
          observed depths. This makes the bottom to change with time if
          the sea level is varying.
    """



    # Init depths
    if ref is None:
        eta = 0.
    if not isinstance(eta, N.ndarray):
        eta = etam = N.ma.array(eta, dtype='d')
        withtime = False
    else:
        withtime = eta.getTime() is not None
        etam = eta.asma()
    nt = eta.shape[0] if withtime else 1
    nz = sigma.shape[0]
    shape = (nt, nz) + depth.shape
    # shape = (nt, nz) + depth.shape[-2:] # cval devrait etre cela je pense
    # pour que tous les cas soient pris en compte
    depths = MV2.zeros(shape, eta.dtype)
    depths.long_name = 'Depths'
    depths.units = 'm'
    depths.id = 'depths'
#    sigman = sigma.filled() if N.ma.isMA(sigma) else sigma
    etam = N.ma.atleast_1d(etam)
#    if not withtime:
#        etam = N.ma.resize(etam, (1, )+eta.shape)

    # Compute it
    stype = _check_sigma_type_(stype)
    if stype == 2:
        if cs is None:

            if a is None or b is None or depth_c is None:
                raise SigmaError('You must prodive depth_c, and b '
                                 'parameters for sigma generalized coordinates conversions')

            cs = ((1 - b) * N.sinh(a * sigma) / math.sinh(a) +
                  b * (N.tanh(a * (sigma + .5)) - math.tanh(.5 * a)) /
                  (2 * math.tanh(.5 * a)))

        dd = depth - depth_c

    # Time loop
    for it in range(nt):
        for iz in range(nz):

            # Sigma generalized
            if stype == 2:

                depths[it, iz] = etam[it] * (1 + sigma[iz])
                depths[it, iz] += depth_c * sigma[iz]
                depths[it, iz] += dd * cs[iz]

                if zerolid:
                    depths[it, iz] -= etam[it]

            else:
                # Common base
                depths[it, iz] = sigma[iz] * (etam[it] + depth)

                # Sigma type
                if stype != 0:
                    depths[it, iz] -= depth  # ocean
                elif not zerolid:
                    depths[it, iz] += etam[it]  # standard

                altitudes[it, iz] = sigma[iz] / height * \
                    (height - orom[it]) + orom[it]


        if not withtime:
            depths = depths[0]

    # Format axes
    if copyaxes:
        axes = []
        if withtime:  # Time axis
            axes.append(eta.getTime())
        if isinstance(sigma, cdms2.axis.AbstractAxis):  # Vertical axis
            axes.append(sigma)
        elif cdms2.isVariable(sigma):
            axes.append(sigma.getAxis(0))
        else:
            zaxis = depths.getAxis(int(withtime))
            zaxis.id = 'z'
            zaxis.long_name = 'Vertical levels'
            axes.append(zaxis)
        axes.extend(depth.getAxisList())  # Horizontal axes
        depths.setAxisList(axes)  # Set axes
        grid = depth.getGrid()  # Grid
        if grid is not None:
            depths.setGrid(grid)

    return depths
