"""
Tidal harmonic analysis and prediction of scalars and currents
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

from .. import dyn as xdyn
from .. import exceptions
from .. import grid as xgrid
from ..core import num as cnum
from ..core import tide as tide_core
from .data import (
    auto_constituents,
    equilibrium_argument,
    get_frequencies,
    get_time_dim,
    nodal_corrections,
)


def _to_seconds_(time_values):
    """Time values as float64 seconds since 1970-01-01"""
    t = np.asarray(time_values)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype(np.int64) / 1e9
    return t.astype(np.float64)


# %% harmonic datasets


def _coef_dataset_(names, omega, t_ref, variables, attrs=None):
    """Build a coefficient dataset from data arrays

    ``variables`` maps names to ``(data array, attrs)``.
    """
    freq_cpd = omega * 86400.0 / (2 * np.pi)
    coords = {
        "constituent": xr.DataArray(
            names, dims=["constituent"], attrs={"long_name": "Tidal constituent"}
        ),
        "frequency": xr.DataArray(
            freq_cpd,
            dims=["constituent"],
            attrs={"long_name": "Tidal frequency", "units": "cycles per day"},
        ),
    }
    data_vars = {}
    for name, (da, vattrs) in variables.items():
        da = da.rename(name)
        da.attrs = vattrs
        data_vars[name] = da
    return xr.Dataset(data_vars, coords=coords, attrs={"t_ref": float(t_ref), **(attrs or {})})


def _map_outer_(func, arrays, axis, n_out, dtype=float):
    """Call a ``(n, ninner)`` function on arrays viewed around an axis, without moving it

    The function takes 2D arrays and returns a 2D array of ``n_out`` rows.
    The result has the shape of the first array, with ``n_out`` along `axis`.
    """
    views = [cnum.view_along_axis(arr, axis) for arr in arrays]
    shape = list(arrays[0].shape)
    shape[axis] = n_out
    if len(views[0]) == 1:  # no outer axes: no copy
        return func(*(view[0] for view in views)).reshape(shape)
    out = np.empty((views[0].shape[0], n_out, views[0].shape[2]), dtype)
    for i in range(len(out)):
        out[i] = func(*(view[i] for view in views))
    return out.reshape(shape)


def _predict_(func, ref, others, t_out, t_dim, long_name, dtype=float):
    """Evaluate a prediction kernel along the constituent dimension of coefficient arrays

    The kernel takes 2D coefficients arrays ``(nf, ns)`` and returns ``(nt, ns)``.
    The time dimension replaces the constituent dimension in place, and dask arrays are
    processed by blocks.
    """
    if t_dim is None:
        t_dim = t_out.dims[0] if isinstance(t_out, xr.DataArray) else "time"
    nt = len(t_out)

    def block(*arrs, axis):
        return _map_outer_(func, arrs, axis, nt, dtype)

    da = xgrid.map_along_dim(
        block, ref, "constituent", *others, out_dim=t_dim, out_size=nt, dtype=dtype
    )
    da = da.assign_coords({t_dim: xr.DataArray(np.asarray(t_out), dims=[t_dim])})
    da.name, da.attrs = None, {"long_name": long_name}
    return da


# %% harmonic analysis


class TidalHarmonic:
    """Tidal harmonic analysis and prediction engine

    The pseudoinverse of the design matrix is computed once from the time axis,
    so that the same instance can analyse any number of data arrays that share
    this time axis, whatever their number of points.

    Parameters
    ----------
    t: array-like of datetime64 or float seconds
        Time axis of the data to be analysed.
    names: list of str, None
        Constituent names. They are selected with :func:`auto_constituents` when None.
    t_dim: str, None
        Name of the time dimension of the data arrays, searched with
        :func:`xoa.coords.get_tdim` when None.
    nodal: bool
        Apply the nodal corrections.
    rayleigh_min: float
        Rayleigh criterion multiplier, used only when ``names`` is None.

    Note
    ----
    Do not chunk along the time dimension when using dask arrays.
    Spatial chunking is fine.
    """

    def __init__(self, t, names=None, t_dim=None, nodal=True, rayleigh_min=1.0):
        t_sec = _to_seconds_(t)
        if names is None:
            names = auto_constituents((t_sec[-1] - t_sec[0]) / 86400.0, rayleigh_min)
        self.names = list(names)
        self.omega = get_frequencies(self.names)
        self.nodal = nodal
        self._t_dim = t_dim

        if nodal:
            self._t_ref = float(t_sec[len(t_sec) // 2])
            self._f, self._u = nodal_corrections(self._t_ref, self.names)
        else:
            self._t_ref = float(t_sec[0])
            self._f = np.ones(len(self.names))
            self._u = np.zeros(len(self.names))

        # Time is relative to t_ref so that phases refer to the middle of the record
        t_rel = t_sec - self._t_ref
        self._A = tide_core.design_matrix(t_rel, self.omega)
        self._Ai = tide_core.design_matrix_pinv(self._A)
        self._M = tide_core.design_matrix_currents(t_rel, self.omega)
        self._Mi = tide_core.design_matrix_currents_pinv(self._M)

    def _design_(self, kind, *das):
        """Design matrix and pseudo-inverse, restricted to the valid steps of 1D series

        Only 1D data (observations) are checked for missing values, to avoid loading
        large arrays. Return ``(M, Mi, ok)`` where ``ok`` is None when nothing is dropped.
        """
        M, Mi = (self._A, self._Ai) if kind == "ssh" else (self._M, self._Mi)
        if any(da.ndim != 1 for da in das):
            return M, Mi, None
        ok = np.logical_and.reduce([np.isfinite(da.values) for da in das])
        if ok.all():
            return M, Mi, None
        if ok.sum() < M.shape[1]:
            raise exceptions.XoaError(
                f"Not enough valid time steps ({ok.sum()}) to fit {M.shape[1]} coefficients"
            )
        dropped = 1 - ok.mean()
        if dropped > 0.1:
            exceptions.xoa_warn(f"{dropped:.0%} of the time steps are missing and ignored")
        M = M[ok]
        pinv = (
            tide_core.design_matrix_pinv if kind == "ssh" else tide_core.design_matrix_currents_pinv
        )
        return M, pinv(M), ok

    def _apply_(self, func, ncoef, *das):
        """Apply a block function along time and return the result with a ``coef`` dimension

        The function takes arrays ``(nt, ns)`` and returns an array ``(ncoef, ns)``.
        The time dimension is replaced in place, and dask arrays are processed by blocks.
        """
        t_dim = get_time_dim(das[0], self._t_dim)

        def block(*arrs, axis):
            return _map_outer_(func, arrs, axis, ncoef)

        return xgrid.map_along_dim(
            block, das[0], t_dim, *das[1:], out_dim="coef", out_size=ncoef, dtype=float
        )

    def _split_(self, res, start, stop=None):
        """Constituent-wise slice of the flat coefficients"""
        return res.isel(coef=slice(start, stop)).rename(coef="constituent")

    def analyse(self, da, uncertainty=True, greenwich=False):
        """Harmonic analysis of a scalar data array like the sea surface height

        Parameters
        ----------
        da: xarray.DataArray
            Array with the time dimension anywhere.
            Missing time steps of a 1D series are ignored.
        uncertainty: bool
            Compute the 1-sigma amplitude and phase uncertainties,
            with the white-noise assumption.
        greenwich: bool
            Convert the local phase to the Greenwich phase lag g in [0, 2π),
            so that results are comparable to tidal atlases, with::

                h(t) = f·A·cos(V₀(t_ref) + ω(t−t_ref) + u(t_ref) − g)

        Return
        ------
        xarray.Dataset
            Variables ``h0``, ``amp``, ``phase`` and optionally ``amp_std`` and ``phase_std``,
            with the ``constituent`` and ``frequency`` coordinates. The ``phase_convention``
            attribute is ``"greenwich"`` or ``"local"``.
        """
        nf = len(self.names)
        A, Ai, ok = self._design_("ssh", da)

        def block(h):
            if ok is not None:
                h = h[ok]
            h0, amp, phase = tide_core.solve_ssh(Ai, h)
            out = [h0[None], amp, phase]
            if uncertainty:
                out.extend(tide_core.solve_ssh_uncertainty(A, Ai, h, h0, amp, phase))
            return np.vstack(out)

        res = self._apply_(block, 1 + (4 if uncertainty else 2) * nf, da)
        h0 = res.isel(coef=0, drop=True)
        amp, phase = self._split_(res, 1, nf + 1), self._split_(res, nf + 1, 2 * nf + 1)

        f = xr.DataArray(self._f, dims="constituent")
        u = xr.DataArray(self._u, dims="constituent")
        if self.nodal:
            amp = amp / f
        if greenwich:
            # g = V0 + u + pi/2 - phase, see xoa.core.tide.local_to_greenwich
            v0 = xr.DataArray(equilibrium_argument(self.names, self._t_ref), dims="constituent")
            phase = (v0 + u + np.pi / 2 - phase) % (2 * np.pi)
        elif self.nodal:
            phase = phase + u

        variables = {
            "h0": (h0, {}),
            "amp": (amp, {"long_name": "Amplitude"}),
            "phase": (
                phase,
                {
                    "units": "rad",
                    "long_name": (
                        "Greenwich phase lag g"
                        if greenwich
                        else "Local phase (sin convention, ref t_ref)"
                    ),
                },
            ),
        }
        if uncertainty:
            amp_std = self._split_(res, 2 * nf + 1, 3 * nf + 1)
            variables["amp_std"] = (
                amp_std / f if self.nodal else amp_std,
                {"long_name": "1-sigma amplitude uncertainty (white-noise)"},
            )
            variables["phase_std"] = (
                self._split_(res, 3 * nf + 1),
                {"long_name": "1-sigma phase uncertainty (white-noise)", "units": "rad"},
            )
        return _coef_dataset_(
            self.names,
            self.omega,
            self._t_ref,
            variables,
            attrs={"nodal": self.nodal, "phase_convention": "greenwich" if greenwich else "local"},
        )

    def analyse_uv(self, da_u, da_v):
        """Harmonic analysis of tidal currents

        Parameters
        ----------
        da_u, da_v: xarray.DataArray
            Eastward and northward velocities.

        Return
        ------
        xarray.Dataset
            Variables ``u0``, ``v0``, ``semi_major``, ``semi_minor``,
            ``inclination`` and ``phase``, in radians for angles.
        """
        nf = len(self.names)
        _, Mi, ok = self._design_("uv", da_u, da_v)

        def block(u, v):
            if ok is not None:
                u, v = u[ok], v[ok]
            u0, v0, smaj, smin, incl, phas = tide_core.solve_currents(Mi, u, v)
            return np.vstack([u0[None], v0[None], smaj, smin, incl, phas])

        res = self._apply_(block, 2 + 4 * nf, da_u, da_v)
        variables = {
            "u0": (res.isel(coef=0, drop=True), {}),
            "v0": (res.isel(coef=1, drop=True), {}),
            "semi_major": (self._split_(res, 2, nf + 2), {"long_name": "Semi-major axis"}),
            "semi_minor": (self._split_(res, nf + 2, 2 * nf + 2), {"long_name": "Semi-minor axis"}),
            "inclination": (self._split_(res, 2 * nf + 2, 3 * nf + 2), {"units": "rad"}),
            "phase": (self._split_(res, 3 * nf + 2), {"units": "rad"}),
        }
        return _coef_dataset_(self.names, self.omega, self._t_ref, variables)

    def ensemble(self, coef, n_members=50, seed=None, inflate=1.0):
        """Ensemble of coefficient sets drawn from the analysis uncertainties

        Each member perturbs the amplitude and phase of every constituent and
        point with independent Gaussian noise of standard deviation ``amp_std``
        and ``phase_std``. Predicting the members with :meth:`predict` gives the
        spread of the tide due to the uncertainty of the analysis.

        Parameters
        ----------
        coef: xarray.Dataset
            Output of :meth:`analyse` with ``uncertainty=True``.
        n_members: int
        seed: int, None
            Seed of the random generator.
        inflate: float
            Factor applied to the standard deviations. The white-noise estimates
            are too small when the residual is correlated in time, like a storm
            surge: 3 to 5 is typical in that case.

        Return
        ------
        xarray.Dataset
            Same variables as ``coef`` with an additional first dimension ``member``.
        """
        if "amp_std" not in coef or "phase_std" not in coef:
            raise exceptions.XoaError("coef needs amp_std and phase_std: use uncertainty=True")
        rng = np.random.default_rng(seed)
        members = xr.DataArray(np.arange(n_members), dims="member", name="member")
        out = xr.Dataset(attrs=dict(coef.attrs))
        out["h0"] = coef["h0"].expand_dims(member=members, axis=0).copy()
        for var in ("amp", "phase"):
            std = coef[f"{var}_std"].expand_dims(member=members, axis=1)
            mean = coef[var].expand_dims(member=members, axis=1)
            out[var] = (mean + inflate * std * rng.standard_normal(mean.shape)).transpose(
                *mean.dims
            )
        return out.assign_coords({k: c for k, c in coef.coords.items() if k not in out.coords})

    def predict(self, coef, t_out):
        """Reconstruct the scalar signal from coefficients

        See :func:`harmonic_predict`.
        """
        return harmonic_predict(coef, t_out, self._t_dim, self.nodal)

    def predict_uv(self, coef, t_out):
        """Reconstruct the currents from coefficients

        See :func:`harmonic_predict_uv`.
        """
        return harmonic_predict_uv(coef, t_out, self._t_dim)


def harmonic_analysis(
    da, names=None, t_dim=None, nodal=True, rayleigh_min=1.0, uncertainty=True, greenwich=False
):
    """One-shot harmonic analysis of a scalar data array

    The time dimension is searched with :func:`xoa.coords.get_tdim` when ``t_dim`` is None.
    With a dataset, the sea level is searched with :func:`xoa.dyn.get_sea_level`.

    See :meth:`TidalHarmonic.analyse` and :class:`TidalHarmonic` for the arguments.
    """
    if isinstance(da, xr.Dataset):
        da = xdyn.get_sea_level(da, errors="raise")
    t_dim = get_time_dim(da, t_dim)
    return TidalHarmonic(da[t_dim].values, names, t_dim, nodal, rayleigh_min).analyse(
        da, uncertainty=uncertainty, greenwich=greenwich
    )


def harmonic_analysis_uv(da_u, da_v=None, names=None, t_dim=None, nodal=True, rayleigh_min=1.0):
    """One-shot harmonic analysis of tidal currents

    The time dimension is searched with :func:`xoa.coords.get_tdim` when ``t_dim`` is None.
    With a dataset as first argument, the velocities are searched with :func:`xoa.dyn.get_u` and :func:`~xoa.dyn.get_v`.

    See :meth:`TidalHarmonic.analyse_uv` and :class:`TidalHarmonic` for the arguments.
    """
    if isinstance(da_u, xr.Dataset):
        ds = da_u
        da_u = xdyn.get_u(ds, errors="raise")
        da_v = xdyn.get_v(ds, errors="raise")
    t_dim = get_time_dim(da_u, t_dim)
    return TidalHarmonic(da_u[t_dim].values, names, t_dim, nodal, rayleigh_min).analyse_uv(
        da_u, da_v
    )


def harmonic_predict(coef, t_out, t_dim=None, nodal=True):
    """Predict a scalar signal from harmonic coefficients

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :func:`harmonic_analysis`. Its ``nodal`` attribute prevails over ``nodal``.
    t_out: array-like of datetime64 or float seconds
    t_dim: str, None
        Name of the output time dimension: that of ``t_out`` when it is a data array, else
        the time dimension of the coefficients' source, i.e. ``"time"``.
    nodal: bool

    Return
    ------
    xarray.DataArray
        Array where time replaces the constituent dimension of the coefficients.
    """
    names = list(coef["constituent"].values)
    nf = len(names)
    t_sec = _to_seconds_(t_out)
    t_ref = float(coef.attrs.get("t_ref", t_sec[0]))
    if coef.attrs.get("nodal", nodal):
        f, u = nodal_corrections(t_ref, names)
    else:
        f, u = np.ones(nf), np.zeros(nf)
    f, u = xr.DataArray(f, dims="constituent"), xr.DataArray(u, dims="constituent")
    phase = coef["phase"]
    if coef.attrs.get("phase_convention", "local") == "greenwich":
        v0 = xr.DataArray(equilibrium_argument(names, t_ref), dims="constituent")
        phase = v0 + u + np.pi / 2 - phase
    else:
        phase = phase - u
    t_rel, omega = t_sec - t_ref, get_frequencies(names)

    def kernel(amp, phase, h0):
        return tide_core.predict_ssh(t_rel, omega, h0[0], amp, phase)

    return _predict_(kernel, coef["amp"] * f, [phase, coef["h0"]], t_out, t_dim, "Tidal prediction")


def harmonic_predict_uv(coef, t_out, t_dim=None):
    """Predict the currents from harmonic coefficients

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :func:`harmonic_analysis_uv`.
    t_out: array-like of datetime64 or float seconds
    t_dim: str, None
        See :func:`harmonic_predict`.

    Return
    ------
    xarray.DataArray, xarray.DataArray
        Eastward and northward velocities, like :func:`harmonic_predict`.
    """
    names = list(coef["constituent"].values)
    t_sec = _to_seconds_(t_out)
    t_ref = float(coef.attrs.get("t_ref", t_sec[0]))
    t_rel, omega = t_sec - t_ref, get_frequencies(names)

    def kernel(smaj, smin, incl, phase, u0, v0):
        return tide_core.predict_currents_complex(
            t_rel, omega, u0[0], v0[0], smaj, smin, incl, phase
        )

    w = _predict_(
        kernel,
        coef["semi_major"],
        [coef[k] for k in ("semi_minor", "inclination", "phase", "u0", "v0")],
        t_out,
        t_dim,
        "Tidal prediction",
        dtype=complex,
    )
    u, v = w.real, w.imag
    u.attrs, v.attrs = {"long_name": "Tidal u prediction"}, {"long_name": "Tidal v prediction"}
    return u, v
