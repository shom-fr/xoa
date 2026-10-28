"""
Core tidal harmonic analysis and prediction (pure numpy/numba, no data, no xarray).

SSH harmonic model::

    h(t) = h0 + Σ_f amp_f · sin(ω_f · t + phase_f)

Current harmonic model (rotating ellipses)::

    w(t) = u0 + i·v0 + Σ_f [A_f·exp(i·(ω_f·t + P_f)) + B_f·exp(-i·(ω_f·t - Q_f))]
    where  A = (semi_major + semi_minor) / 2,  B = (semi_major - semi_minor) / 2
           P = inclination + phase,             Q = inclination - phase

The design matrix depends only on time → compute the pseudoinverse once,
then apply to every grid point with a single matrix multiply::

    x = Ai @ h   shape (1+2*nf, ns)  from  (1+2*nf, nt) @ (nt, ns)

Also contains pure-math helpers that carry no constituent-database dependency:

    - Doodson multiplier parsing     (doodson_multipliers)
    - Slow astronomical arguments    (astronomical_args)
    - Phase convention conversions   (local_to_greenwich, greenwich_to_local)
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

from . import signal

# %% doodson multipliers


def doodson_multipliers(doodson):
    """Doodson string → 6-element array of true multipliers (removes +5 offset on n[1:]).

    Parameters
    ----------
    doodson : str
        Doodson number string with one embedded dot, e.g. ``'255.555'``.

    Returns
    -------
    n : ndarray (6,)
    """
    digits = doodson.replace(".", "")
    n = np.array([int(c) for c in digits], dtype=np.float64)
    n[1:] -= 5.0
    return n


# %% astronomical arguments

_JD_J1900_ = 2415020.5  # Julian Day of 1900-01-01 00:00 UT, the epoch of Schureman Table 1
_JD_UNIX_ = 2440587.5  # Julian Day of Unix epoch (1970-Jan-1 00:00:00 UTC)

# s, h, p, N, p1 epoch values in degrees at J1900.0 (Schureman 1958 Table 1)
_ARG_EPOCH_ = np.array([277.0247, 280.1895, 334.3853, 259.1568, 281.2208])

# s, h, p, N, p1 rates in degrees per Julian day
_ARG_RATES_ = np.array([13.1763965, 0.9856473, 0.1114042, -0.0529539, 0.0000470])


def astronomical_args(t_sec):
    """Slow Doodson variables (s, h, p, N, N', p1) in degrees at Unix timestamp.

    Parameters
    ----------
    t_sec : float
        Seconds since 1970-01-01 00:00:00 UTC.

    Returns
    -------
    s, h, p, N, N_prime, p1 : float (degrees)
        N' = -N mod 360 (positive nodal argument used in Doodson convention).

    References
    ----------
    Schureman (1958) Table 1.
    """
    T = t_sec / 86400.0 + _JD_UNIX_ - _JD_J1900_  # days since J1900.0
    vals = (_ARG_EPOCH_ + _ARG_RATES_ * T) % 360.0
    return vals[0], vals[1], vals[2], vals[3], (360.0 - vals[3]) % 360.0, vals[4]


# %% phase convention conversions


def local_to_greenwich(phase_local, v0, u):
    """Convert local sin-convention phases to Greenwich phase lags.

    The local convention stores ``phase_raw + u`` where ``phase_raw =
    arctan2(cos_coeff, sin_coeff)`` from the sin/cos design matrix.  The
    Greenwich phase lag g satisfies the standard formula::

        h(t) = f·A·cos(V₀ + ω(t−t_ref) + u − g)

    yielding ``g = V₀ + u + π/2 − phase_raw``.

    Parameters
    ----------
    phase_local : ndarray (nf, ...)
        Local phases with nodal correction u already applied (phase_raw + u).
    v0 : ndarray (nf,)
        Equilibrium argument from :func:`xoa.tide.equilibrium_argument`.
    u : ndarray (nf,)
        Nodal phase corrections from :func:`xoa.tide.nodal_corrections`.

    Returns
    -------
    g : ndarray (nf, ...) — Greenwich phase lags in radians, [0, 2π).
    """
    ndim_extra = max(np.ndim(phase_local) - 1, 0)
    bc = (-1,) + (1,) * ndim_extra
    phase_raw = phase_local - u.reshape(bc)
    return (v0.reshape(bc) + u.reshape(bc) + np.pi / 2 - phase_raw) % (2 * np.pi)


def greenwich_to_local(phase_g, v0, u):
    """Convert Greenwich phase lags to local sin-convention phases.

    Inverse of :func:`local_to_greenwich`.  Returns phases in (−π, π].

    Parameters
    ----------
    phase_g : ndarray (nf, ...)
        Greenwich phase lags in radians.
    v0 : ndarray (nf,)
        Equilibrium argument from :func:`xoa.tide.equilibrium_argument`.
    u : ndarray (nf,)
        Nodal phase corrections from :func:`xoa.tide.nodal_corrections`.

    Returns
    -------
    phase_local : ndarray (nf, ...) in (−π, π].
    """
    ndim_extra = max(np.ndim(phase_g) - 1, 0)
    bc = (-1,) + (1,) * ndim_extra
    phase_raw = v0.reshape(bc) + u.reshape(bc) + np.pi / 2 - phase_g
    return ((phase_raw + u.reshape(bc)) + np.pi) % (2 * np.pi) - np.pi


# %% design matrix


def design_matrix(t, omega):
    """Harmonic least-squares design matrix for SSH analysis.

    Parameters
    ----------
    t : ndarray (nt,)
        Time in seconds.
    omega : ndarray (nf,)
        Angular frequencies in rad/s.

    Returns
    -------
    A : ndarray (nt, 1+2*nf)
        Columns: [1, sin(ω₁t)…sin(ωₙt), cos(ω₁t)…cos(ωₙt)]
    """
    ot = t[:, None] * omega[None, :]  # (nt, nf)
    return np.hstack([np.ones((len(t), 1)), np.sin(ot), np.cos(ot)])


def design_matrix_pinv(A):
    """Moore-Penrose pseudoinverse of design matrix (the one-time inversion cost).

    Parameters
    ----------
    A : ndarray (nt, 1+2*nf)

    Returns
    -------
    Ai : ndarray (1+2*nf, nt)
    """
    return np.linalg.pinv(A)


# %% ssh analysis


def solve_ssh(Ai, h):
    """Harmonic analysis of sea surface height at all grid points simultaneously.

    Parameters
    ----------
    Ai : ndarray (1+2*nf, nt)
        From design_matrix_pinv.
    h : ndarray (nt, ns)
        SSH; NaN columns (land) propagate to NaN output.

    Returns
    -------
    h0 : ndarray (ns,)
    amp : ndarray (nf, ns)
    phase : ndarray (nf, ns)
        Radians; signal = h0 + Σ amp·sin(ω·t + phase).
    """
    nf = (Ai.shape[0] - 1) // 2
    x = Ai @ h
    h0 = x[0]
    amp = np.hypot(x[1 : nf + 1], x[nf + 1 :])
    phase = np.arctan2(x[nf + 1 :], x[1 : nf + 1])
    return h0, amp, phase


def predict_ssh(t, omega, h0, amp, phase):
    """Harmonic reconstruction of SSH at all grid points.

    Equivalent to the analysis transpose: h = A @ x, where x holds the
    (sin, cos) coefficients recovered from (amp, phase).  One BLAS DGEMM.

    Parameters
    ----------
    t : ndarray (nt,)
    omega : ndarray (nf,)
    h0 : ndarray (ns,)
    amp : ndarray (nf, ns)
    phase : ndarray (nf, ns)

    Returns
    -------
    h : ndarray (nt, ns)
    """
    a = amp * np.cos(phase)  # sin coefficients  (nf, ns)
    b = amp * np.sin(phase)  # cos coefficients  (nf, ns)
    x = np.vstack([h0[None], a, b])  # (1+2*nf, ns)
    return design_matrix(t, omega) @ x


# %% uncertainty


def solve_ssh_uncertainty(A, Ai, h, h0, amp, phase):
    """Amplitude and phase uncertainties from linearised least-squares theory.

    For each constituent f at each spatial point s::

        σ²_amp[f,s]   =  σ²_a[f,s] · cos²φ + σ²_b[f,s] · sin²φ
        σ²_phase[f,s] = (σ²_a[f,s] · sin²φ + σ²_b[f,s] · cos²φ) / amp²

    where φ = phase[f,s] and σ²_a, σ²_b are the diagonal variances of
    the coefficient covariance matrix C = σ²_res · Ai @ Ai^T.

    Assumes white residuals. For coloured noise (e.g. model output) the
    true uncertainty is larger; treat these as lower bounds.

    Parameters
    ----------
    A : ndarray (nt, 1+2*nf)
        Design matrix.
    Ai : ndarray (1+2*nf, nt)
        Pseudoinverse.
    h : ndarray (nt, ns)
        Observations (NaN land columns allowed).
    h0, amp, phase : from solve_ssh

    Returns
    -------
    amp_std : ndarray (nf, ns)   — 1-σ uncertainty on amplitude
    phase_std : ndarray (nf, ns) — 1-σ uncertainty on phase (radians)
    """
    nt, ns = h.shape
    nf = (Ai.shape[0] - 1) // 2
    dof = nt - (2 * nf + 1)

    # Residual variance per point (nt,) → (ns,) via column
    h_fit = A @ (Ai @ h)  # (nt, ns) — reconstructed
    res = h - h_fit  # (nt, ns)
    sigma2 = np.nansum(res**2, axis=0) / dof  # (ns,)

    # Diagonal of Ai @ Ai^T: (1+2*nf,) — same for all points
    diag_AiAit = np.sum(Ai**2, axis=1)  # (1+2*nf,)

    # sin coefficient variances: indices 1..nf
    # cos coefficient variances: indices nf+1..2*nf
    var_a = diag_AiAit[1 : nf + 1, None] * sigma2[None, :]  # (nf, ns)
    var_b = diag_AiAit[nf + 1 :, None] * sigma2[None, :]  # (nf, ns)

    cos_phi = np.cos(phase)
    sin_phi = np.sin(phase)
    amp_sq = np.where(amp**2 > 0, amp**2, np.nan)

    amp_std = np.sqrt(var_a * cos_phi**2 + var_b * sin_phi**2)
    phase_std = np.sqrt((var_a * sin_phi**2 + var_b * cos_phi**2) / amp_sq)

    return amp_std, phase_std


# %% current analysis


def design_matrix_currents(t, omega):
    """Complex design matrix for tidal current analysis.

    Parameters
    ----------
    t : ndarray (nt,)
    omega : ndarray (nf,)

    Returns
    -------
    M : ndarray complex128 (nt, 1+2*nf)
        Columns: [1, exp(+iω₁t)…exp(+iωₙt), exp(-iω₁t)…exp(-iωₙt)]
    """
    ot = t[:, None] * omega[None, :]
    ones = np.ones((len(t), 1), dtype=np.complex128)
    return np.hstack([ones, np.exp(1j * ot), np.exp(-1j * ot)])


def design_matrix_currents_pinv(M):
    """Moore-Penrose pseudoinverse of complex current design matrix.

    Parameters
    ----------
    M : ndarray complex128 (nt, 1+2*nf)

    Returns
    -------
    Mi : ndarray complex128 (1+2*nf, nt)
    """
    return np.linalg.pinv(M)


def solve_currents(Mi, u, v):
    """Harmonic analysis of tidal currents at all grid points.

    Parameters
    ----------
    Mi : ndarray complex128 (1+2*nf, nt)
    u : ndarray (nt, ns)  — eastward velocity
    v : ndarray (nt, ns)  — northward velocity

    Returns
    -------
    u0 : ndarray (ns,)
    v0 : ndarray (ns,)
    semi_major : ndarray (nf, ns)   — always ≥ 0
    semi_minor : ndarray (nf, ns)   — negative means clockwise rotation
    inclination : ndarray (nf, ns)  — major-axis orientation from east, radians, in (-pi/2, pi/2]
    phase : ndarray (nf, ns)        — phase advance, radians in (-pi, pi] (current maximum at ωt = -phase)
    """
    nf = (Mi.shape[0] - 1) // 2
    w = (u + 1j * v).astype(np.complex128)
    x = Mi @ w
    w0 = x[0]
    R = x[1 : nf + 1]
    S = x[nf + 1 :]
    A = np.abs(R)
    B = np.abs(S)
    P = np.angle(R)
    Q = np.angle(S)
    incl = (P + Q) / 2
    phase = (P - Q) / 2
    # (inclination, phase) and (inclination ± pi, phase ± pi) describe the same ellipse:
    # report the inclination in (-pi/2, pi/2] and the phase in (-pi, pi]
    shift = np.where(incl > np.pi / 2, -np.pi, np.where(incl <= -np.pi / 2, np.pi, 0.0))
    incl = incl + shift
    phase = np.angle(np.exp(1j * (phase + shift)))
    return (w0.real, w0.imag, A + B, A - B, incl, phase)


def predict_currents_complex(t, omega, u0, v0, semi_major, semi_minor, inclination, phase):
    """Harmonic reconstruction of tidal currents as the complex velocity ``u + 1j * v``.

    Reconstructs the complex phasor coefficients (R, S) from the ellipse
    parameters and evaluates w = M @ x — one complex BLAS DGEMM.

    Parameters
    ----------
    t : ndarray (nt,)
    omega : ndarray (nf,)
    u0, v0 : ndarray (ns,)
    semi_major, semi_minor, inclination, phase : ndarray (nf, ns)

    Returns
    -------
    w : ndarray (nt, ns) complex
    """
    Af = (semi_major + semi_minor) * 0.5  # (nf, ns)
    Bf = (semi_major - semi_minor) * 0.5  # (nf, ns)
    P = inclination + phase  # (nf, ns)
    Q = inclination - phase  # (nf, ns)
    R = Af * np.exp(1j * P)  # counter-rotating phasor  (nf, ns)
    S = Bf * np.exp(1j * Q)  # co-rotating phasor       (nf, ns)
    x = np.vstack([(u0 + 1j * v0)[None], R, S])  # (1+2*nf, ns)
    return design_matrix_currents(t, omega) @ x  # (nt, ns) complex


def predict_currents(t, omega, u0, v0, semi_major, semi_minor, inclination, phase):
    """Harmonic reconstruction of tidal currents at all grid points.

    See :func:`predict_currents_complex`.

    Returns
    -------
    u, v : ndarray (nt, ns)
    """
    w = predict_currents_complex(t, omega, u0, v0, semi_major, semi_minor, inclination, phase)
    return w.real, w.imag


# %% tidal ellipses


def rotary_components(semi_major, semi_minor, inclination, phase):
    """Counter-clockwise and clockwise circular components of tidal ellipses.

    A tidal current is the sum of two circular motions of one constituent:
    ``w(t) = Wp·exp(+iωt) + Wm·exp(-iωt)`` with ``w = u + iv``.

    Parameters
    ----------
    semi_major, semi_minor, inclination, phase : array-like
        Ellipse parameters (see :func:`solve_currents`), broadcastable together.

    Returns
    -------
    wp, wm : ndarray complex128
        Counter-clockwise and clockwise rotary phasors.
    """
    smaj, smin, incl, phas = (
        np.asarray(a, dtype=float) for a in (semi_major, semi_minor, inclination, phase)
    )
    wp = 0.5 * (smaj + smin) * np.exp(1j * (incl + phas))
    wm = 0.5 * (smaj - smin) * np.exp(1j * (incl - phas))
    return wp, wm


def ellipse_from_rotary(wp, wm):
    """Ellipse parameters from the rotary phasors (inverse of :func:`rotary_components`).

    Parameters
    ----------
    wp, wm : array-like complex
        Counter-clockwise and clockwise rotary phasors.

    Returns
    -------
    semi_major, semi_minor, inclination, phase : ndarray
        ``semi_minor`` is negative for a clockwise rotation, ``inclination``
        and ``phase`` are in radians.
    """
    wp, wm = np.asarray(wp), np.asarray(wm)
    a, b = np.abs(wp), np.abs(wm)
    return a + b, a - b, 0.5 * (np.angle(wp) + np.angle(wm)), 0.5 * (np.angle(wp) - np.angle(wm))


def ellipse_properties(semi_major, semi_minor):
    """Shape descriptors of tidal ellipses.

    Parameters
    ----------
    semi_major, semi_minor : array-like
        Semi-axes; a negative ``semi_minor`` means clockwise rotation.

    Returns
    -------
    ellipticity : ndarray
        ``semi_minor / semi_major`` in [-1, 1]: 0 is rectilinear, ±1 circular.
        NaN where the semi-major axis is zero.
    rotation : ndarray
        +1 counter-clockwise, -1 clockwise, 0 rectilinear (float, NaN where
        ``semi_minor`` is NaN).
    """
    smaj, smin = np.asarray(semi_major, dtype=float), np.asarray(semi_minor, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ellipticity = np.where(smaj > 0, smin / smaj, np.nan)
    return ellipticity, np.sign(smin)


def ellipse_vector(semi_major, semi_minor, inclination, phase, wt):
    """Current vector of one constituent along its ellipse.

    Parameters
    ----------
    semi_major, semi_minor, inclination, phase : array-like
        Ellipse parameters, shape ``S``.
    wt : array-like (n,)
        Phase angle ``ω·(t - t_ref)`` in radians.

    Returns
    -------
    u, v : ndarray (*S, n)
        Eastward and northward components along the ellipse.
    """
    wp, wm = rotary_components(semi_major, semi_minor, inclination, phase)
    wt = np.asarray(wt, dtype=float)
    w = wp[..., None] * np.exp(1j * wt) + wm[..., None] * np.exp(-1j * wt)
    return w.real, w.imag


# %% high waters and skew surge
#
# Built on the generic time-first kernels of :mod:`xoa.core.signal`: the
# arrays are ``(nt, ...)``, a 1-D series ``(nt,)`` being a field with one point.


def high_water_mask(h, min_separation=9):
    """Boolean mask of the high waters of regularly sampled series.

    Parameters
    ----------
    h : ndarray (nt, ...)
        Sea level, typically the predicted tide, **time first**; the other
        axes (points, grid) are processed independently and in parallel.
    min_separation : int
        Minimal number of samples between two high waters.

    Returns
    -------
    ndarray of bool, same shape as ``h``
    """
    return signal.peak_mask(h, min_separation)


def skew_surge_series(obs, pred, hw, half_window=3):
    """Skew surge of regularly sampled series, at the predicted high waters.

    Parameters
    ----------
    obs, pred : ndarray (nt, ...)
        Observed and predicted sea level, **time first**; ``obs`` may contain
        NaNs.
    hw : ndarray of bool, same shape
        High water mask, see :func:`high_water_mask`.
    half_window : int
        Half width in samples of the window searched for the observed maximum.

    Returns
    -------
    ndarray, same shape as ``obs``
        Highest observed level in the window minus the predicted high water, at
        each high water with a complete window and some data; NaN elsewhere.
    """
    return signal.window_max_at(obs, hw, half_window, ref=pred)


def spring_mask(h, hw, quantile=0.75, half_window=6):
    """Mask of the high waters with the largest range (spring tides).

    The range of a high water is its level minus the lowest level within
    ``half_window`` samples on each side; the quantile is taken along time, for
    each point.

    Parameters
    ----------
    h : ndarray (nt, ...)
        **Time first**.
    hw : ndarray of bool, same shape
        High water mask, see :func:`high_water_mask`.
    quantile : float
        Keep the high waters whose range is above this quantile.
    half_window : int

    Returns
    -------
    ndarray of bool, same shape as ``h``
        High waters without a full window are dropped.
    """
    return signal.quantile_mask(signal.window_min_at(h, hw, half_window, ref=h), quantile)
