#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core terrain-folowing coordinate functions.

These functions use numba's guvectorize decorator to create
universal functions (ufuncs) that work efficiently with xarray.apply_ufunc.
"""
import numpy as np
import numba


@numba.guvectorize(
    ['void(float64[:], float64, float64, float64[:])'],
    '(k),(),()->(k)',
    nopython=True,
    cache=True
)
def atmosphere_sigma(sigma, ps, ptop, p):
    """Convert from sigma to pressure in an atmospheric model

    Parameters
    ----------
    sigma : array(k)
        Sigma coordinates
    ps : scalar
        Surface pressure
    ptop : scalar
        Pressure at top of model
    p : array(k)
        Output pressure array (modified in-place)
    """
    for k in range(sigma.shape[0]):
        p[k] = ptop + sigma[k] * (ps - ptop)


@numba.guvectorize(
    ['void(float64[:], float64, float64, float64[:])'],
    '(k),(),()->(k)',
    nopython=True,
    cache=True
)
def ocean_sigma(sigma, eta, depth, z):
    """Convert from sigma to depth in an ocean model

    Parameters
    ----------
    sigma : array(k)
        Sigma coordinates
    eta : scalar
        Sea surface height
    depth : scalar
        Bottom depth
    z : array(k)
        Output depth array (modified in-place)
    """
    for k in range(sigma.shape[0]):
        z[k] = eta + sigma[k] * (eta + depth)


@numba.guvectorize(
    ['void(float64[:], float64, float64, float64, float64[:], float64[:])'],
    '(k),(),(),(),( k)->(k)',
    nopython=True,
    cache=True
)
def ocean_s(s, eta, depth, depth_c, C, z):
    """Convert from s-coordinate to depth in an ocean model

    Parameters
    ----------
    s : array(k)
        S-coordinates
    eta : scalar
        Sea surface height
    depth : scalar
        Bottom depth
    depth_c : scalar
        Critical depth
    C : array(k)
        Stretching curve
    z : array(k)
        Output depth array (modified in-place)
    """
    # Pre-compute invariants
    depth_minus_depth_c = depth - depth_c

    for k in range(s.shape[0]):
        z[k] = eta * (1.0 + s[k]) + depth_c * s[k] + depth_minus_depth_c * C[k]


@numba.guvectorize(
    ['void(float64[:], float64, float64, float64, float64[:], float64[:])'],
    '(k),(),(),(),( k)->(k)',
    nopython=True,
    cache=True
)
def ocean_s_g1(s, eta, depth, depth_c, C, z):
    """Convert from s-coordinate (generic form 1) to depth in an ocean model

    Parameters
    ----------
    s : array(k)
        S-coordinates
    eta : scalar
        Sea surface height
    depth : scalar
        Bottom depth
    depth_c : scalar
        Critical depth
    C : array(k)
        Stretching curve
    z : array(k)
        Output depth array (modified in-place)
    """
    # Pre-compute invariants
    depth_minus_depth_c = depth - depth_c
    inv_depth = 1.0 / depth

    for k in range(s.shape[0]):
        S = depth_c * s[k] + depth_minus_depth_c * C[k]
        z[k] = S + eta * (1.0 + S * inv_depth)


@numba.guvectorize(
    ['void(float64[:], float64, float64, float64, float64[:], float64[:])'],
    '(k),(),(),(),( k)->(k)',
    nopython=True,
    cache=True
)
def ocean_s_g2(s, eta, depth, depth_c, C, z):
    """Convert from s-coordinate (generic form 2) to depth in an ocean model

    Parameters
    ----------
    s : array(k)
        S-coordinates
    eta : scalar
        Sea surface height
    depth : scalar
        Bottom depth
    depth_c : scalar
        Critical depth
    C : array(k)
        Stretching curve
    z : array(k)
        Output depth array (modified in-place)
    """
    # Pre-compute invariants
    depth_plus_depth_c = depth + depth_c
    inv_depth_plus_depth_c = 1.0 / depth_plus_depth_c
    depth_plus_eta = depth + eta

    for k in range(s.shape[0]):
        S = (depth_c * s[k] + depth * C[k]) * inv_depth_plus_depth_c
        z[k] = S * depth_plus_eta + eta
