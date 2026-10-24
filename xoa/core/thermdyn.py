#!/usr/bin/env python3
"""
Numeric thermodynamics utilities based on :mod:`gsw`

Salinities are converted from and to the absolute salinity ``"asal"``,
and temperatures from and to the conservative temperature ``"ctemp"``.

Kinds of temperature are ``"temp"`` (in situ), ``"ptemp"`` (potential) and ``"ctemp"``
(conservative), and kinds of salinity are ``"psal"`` (practical), ``"asal"`` (absolute)
and ``"pfsal"`` (preformed).
"""

import numpy as np


def to_sa(sal, kind, p, lon, lat):
    """Convert a salinity of a given kind to absolute salinity in g/kg"""
    import gsw

    if kind == "psal":
        return gsw.SA_from_SP(sal, p, lon, lat)
    if kind == "pfsal":
        return gsw.SA_from_Sstar(sal, p, lon, lat)
    return np.asarray(sal, dtype="d")


def from_sa(sa, kind, p, lon, lat):
    """Convert an absolute salinity in g/kg to a salinity of a given kind"""
    import gsw

    if kind == "psal":
        return gsw.SP_from_SA(sa, p, lon, lat)
    if kind == "pfsal":
        return gsw.Sstar_from_SA(sa, p, lon, lat)
    return np.asarray(sa, dtype="d")


def to_ct(temp, kind, sa, p):
    """Convert a temperature of a given kind in degrees Celsius to conservative temperature"""
    import gsw

    if kind == "ptemp":
        return gsw.CT_from_pt(sa, temp)
    if kind == "temp":
        return gsw.CT_from_t(sa, temp, p)
    return np.asarray(temp, dtype="d")


def from_ct(ct, kind, sa, p):
    """Convert a conservative temperature to a temperature of a given kind in degrees Celsius"""
    import gsw

    if kind == "ptemp":
        return gsw.pt_from_CT(sa, ct)
    if kind == "temp":
        return gsw.t_from_CT(sa, ct, p)
    return np.asarray(ct, dtype="d")


def convert_sal(sal, z, lat, lon, kind, to):
    """Convert a salinity from a kind to another one"""
    import gsw

    if kind == to:
        return np.asarray(sal, dtype="d")
    p = gsw.p_from_z(z, lat)
    return from_sa(to_sa(sal, kind, p, lon, lat), to, p, lon, lat)


def convert_temp(temp, sal, z, lat, lon, kind, to, sal_kind):
    """Convert a temperature in degrees Celsius from a kind to another one"""
    import gsw

    if kind == to:
        return np.asarray(temp, dtype="d")
    p = gsw.p_from_z(z, lat)
    sa = to_sa(sal, sal_kind, p, lon, lat)
    return from_ct(to_ct(temp, kind, sa, p), to, sa, p)


def n2_at_levels(temp, sal, z, lat, lon, temp_kind, sal_kind, axis=-1):
    """Squared Brunt-Vaisala frequency at the levels of arrays whose vertical axis is `axis`

    The latitudes and longitudes either have no vertical axis, or one of size 1.
    """
    import gsw

    temp = np.asarray(temp, dtype="d")
    axis = axis % temp.ndim
    lat = np.asarray(lat, dtype="d")
    lon = np.asarray(lon, dtype="d")
    if lat.ndim < temp.ndim:
        lat = np.expand_dims(lat, axis if lat.ndim else 0)
    if lon.ndim < temp.ndim:
        lon = np.expand_dims(lon, axis if lon.ndim else 0)
    p = gsw.p_from_z(z, lat)
    sa = to_sa(sal, sal_kind, p, lon, lat)
    ct = to_ct(temp, temp_kind, sa, p)
    mid, _ = gsw.Nsquared(sa, ct, p, lat, axis=axis)
    out = np.empty(np.broadcast(temp, sa, ct, p).shape, dtype="d")
    mid = np.moveaxis(mid, axis, 0)  # views only, to index along the vertical axis
    outv = np.moveaxis(out, axis, 0)
    outv[1:-1] = 0.5 * (mid[:-1] + mid[1:])
    outv[0] = mid[0]
    outv[-1] = mid[-1]
    return out
