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

from .. import exceptions
from ..core import tide as tide_core
from .data import (
    auto_constituents,
    equilibrium_argument,
    get_frequencies,
    nodal_corrections,
)


def _to_seconds_(time_values):
    """Time values as float64 seconds since 1970-01-01"""
    t = np.asarray(time_values)
    if np.issubdtype(t.dtype, np.datetime64):
        return t.astype("datetime64[ns]").astype(np.int64) / 1e9
    return t.astype(np.float64)


# %% harmonic datasets


def _coef_dataset_(names, omega, t_ref, variables, spatial_dims, spatial_shape, attrs=None):
    """Build a coefficient dataset from flat arrays

    ``variables`` maps names to ``(array, attrs, has_constituent_dim)``.
    """
    nf = len(names)
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
    sdims = list(spatial_dims)
    data_vars = {}
    for name, (arr, vattrs, lead) in variables.items():
        shape = (nf, *spatial_shape) if lead else spatial_shape
        dims = (["constituent"] + sdims) if lead else sdims
        data_vars[name] = xr.DataArray(arr.reshape(shape), dims=dims, attrs=vattrs)
    return xr.Dataset(data_vars, coords=coords, attrs={"t_ref": float(t_ref), **(attrs or {})})


def _spatial_coords_(da, t_dim):
    """Non-time coordinates of ``da``, to attach to a coefficient dataset"""
    return {k: c for k, c in da.coords.items() if t_dim not in c.dims}


def _flat_coef_(coef, name, nf=None):
    """Values of a coefficient variable as ``(nf, ns)``, or ``(ns,)`` if ``nf`` is None"""
    values = coef[name].values
    return values.ravel() if nf is None else values.reshape(nf, -1)


def _prediction_da_(values, t_out, coef, ref, t_dim, long_name):
    spatial_dims = list(coef[ref].dims)
    spatial_shape = tuple(coef.sizes[d] for d in spatial_dims)
    coords = {t_dim: xr.DataArray(np.asarray(t_out), dims=[t_dim])}
    coords.update(
        {k: c for k, c in coef.coords.items() if c.dims and set(c.dims) <= set(spatial_dims)}
    )
    return xr.DataArray(
        values.reshape((len(coords[t_dim]), *spatial_shape)),
        dims=[t_dim] + spatial_dims,
        coords=coords,
        attrs={"long_name": long_name},
    )


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
    t_dim: str
        Name of the time dimension of the data arrays.
    nodal: bool
        Apply the nodal corrections.
    rayleigh_min: float
        Rayleigh criterion multiplier, used only when ``names`` is None.

    Note
    ----
    Do not chunk along the time dimension when using dask arrays.
    Spatial chunking is fine.
    """

    def __init__(self, t, names=None, t_dim="time", nodal=True, rayleigh_min=1.0):
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

    def _apply_(self, func, ncoef, *das):
        """Apply a block function to arrays (time, ...) and return the flat (ncoef, ns) result

        The function takes arrays ``(nt, ns)`` and returns an array ``(ncoef, ns)``.
        Dask arrays are processed by spatial blocks, time being a single chunk.
        """
        t_dim = self._t_dim
        sdims = [d for d in das[0].dims if d != t_dim]
        das = [da.transpose(t_dim, *sdims) for da in das]

        def block(*arrs):
            out = func(*(a.reshape(a.shape[0], -1) for a in arrs))
            return out.reshape((ncoef, *arrs[0].shape[1:]))

        if any(da.chunks is not None for da in das):
            import dask.array

            data = [da.data for da in xr.unify_chunks(*(da.chunk({t_dim: -1}) for da in das))]
            chunks = ((ncoef,), *data[0].chunks[1:])
            out = dask.array.map_blocks(block, *data, chunks=chunks, dtype=float)
        else:
            out = block(*(da.values for da in das))
        sshape = tuple(das[0].sizes[d] for d in sdims)
        return np.asarray(out).reshape(ncoef, -1), sdims, sshape

    def analyse(self, da, uncertainty=True, greenwich=False):
        """Harmonic analysis of a scalar data array like the sea surface height

        Parameters
        ----------
        da: xarray.DataArray
            Array with the time dimension first.
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
        A, Ai = self._A, self._Ai

        def block(h):
            h0, amp, phase = tide_core.solve_ssh(Ai, h)
            out = [h0[None], amp, phase]
            if uncertainty:
                out.extend(tide_core.solve_ssh_uncertainty(A, Ai, h, h0, amp, phase))
            return np.vstack(out)

        flat, sdims, sshape = self._apply_(block, 1 + (4 if uncertainty else 2) * nf, da)
        h0, amp, phase = flat[0], flat[1 : nf + 1], flat[nf + 1 : 2 * nf + 1]
        amp_std, phase_std = (
            (flat[2 * nf + 1 : 3 * nf + 1], flat[3 * nf + 1 :]) if uncertainty else (0, 0)
        )

        if self.nodal:
            amp = amp / self._f[:, None]
            phase = phase + self._u[:, None]
            amp_std = amp_std / self._f[:, None] if uncertainty else amp_std
        if greenwich:
            v0 = equilibrium_argument(self.names, self._t_ref)
            phase = tide_core.local_to_greenwich(phase, v0, self._u)

        variables = {
            "h0": (h0, {}, False),
            "amp": (amp, {"long_name": "Amplitude"}, True),
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
                True,
            ),
        }
        if uncertainty:
            variables["amp_std"] = (
                amp_std,
                {"long_name": "1-sigma amplitude uncertainty (white-noise)"},
                True,
            )
            variables["phase_std"] = (
                phase_std,
                {"long_name": "1-sigma phase uncertainty (white-noise)", "units": "rad"},
                True,
            )
        ds = _coef_dataset_(
            self.names,
            self.omega,
            self._t_ref,
            variables,
            sdims,
            sshape,
            attrs={"nodal": self.nodal, "phase_convention": "greenwich" if greenwich else "local"},
        )
        return ds.assign_coords(_spatial_coords_(da, self._t_dim))

    def analyse_uv(self, da_u, da_v):
        """Harmonic analysis of tidal currents

        Parameters
        ----------
        da_u, da_v: xarray.DataArray
            Eastward and northward velocities with the time dimension first.

        Return
        ------
        xarray.Dataset
            Variables ``u0``, ``v0``, ``semi_major``, ``semi_minor``,
            ``inclination`` and ``phase``, in radians for angles.
        """
        nf = len(self.names)
        Mi = self._Mi

        def block(u, v):
            u0, v0, smaj, smin, incl, phas = tide_core.solve_currents(Mi, u, v)
            return np.vstack([u0[None], v0[None], smaj, smin, incl, phas])

        flat, sdims, sshape = self._apply_(block, 2 + 4 * nf, da_u, da_v)
        variables = {
            "u0": (flat[0], {}, False),
            "v0": (flat[1], {}, False),
            "semi_major": (flat[2 : nf + 2], {"long_name": "Semi-major axis"}, True),
            "semi_minor": (flat[nf + 2 : 2 * nf + 2], {"long_name": "Semi-minor axis"}, True),
            "inclination": (flat[2 * nf + 2 : 3 * nf + 2], {"units": "rad"}, True),
            "phase": (flat[3 * nf + 2 :], {"units": "rad"}, True),
        }
        ds = _coef_dataset_(self.names, self.omega, self._t_ref, variables, sdims, sshape)
        return ds.assign_coords(_spatial_coords_(da_u, self._t_dim))

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
    da, names=None, t_dim="time", nodal=True, rayleigh_min=1.0, uncertainty=True, greenwich=False
):
    """One-shot harmonic analysis of a scalar data array

    See :meth:`TidalHarmonic.analyse` and :class:`TidalHarmonic` for the arguments.
    """
    return TidalHarmonic(da[t_dim].values, names, t_dim, nodal, rayleigh_min).analyse(
        da, uncertainty=uncertainty, greenwich=greenwich
    )


def harmonic_analysis_uv(da_u, da_v, names=None, t_dim="time", nodal=True, rayleigh_min=1.0):
    """One-shot harmonic analysis of tidal currents

    See :meth:`TidalHarmonic.analyse_uv` and :class:`TidalHarmonic` for the arguments.
    """
    return TidalHarmonic(da_u[t_dim].values, names, t_dim, nodal, rayleigh_min).analyse_uv(
        da_u, da_v
    )


def harmonic_predict(coef, t_out, t_dim="time", nodal=True):
    """Predict a scalar signal from harmonic coefficients

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :func:`harmonic_analysis`. Its ``nodal`` attribute prevails over ``nodal``.
    t_out: array-like of datetime64 or float seconds
    t_dim: str
    nodal: bool

    Return
    ------
    xarray.DataArray
        Array with the time dimension first.
    """
    names = list(coef["constituent"].values)
    nf = len(names)
    t_sec = _to_seconds_(t_out)
    t_ref = float(coef.attrs.get("t_ref", t_sec[0]))
    if coef.attrs.get("nodal", nodal):
        f, u = nodal_corrections(t_ref, names)
    else:
        f, u = np.ones(nf), np.zeros(nf)
    amp, phase = _flat_coef_(coef, "amp", nf), _flat_coef_(coef, "phase", nf)
    if coef.attrs.get("phase_convention", "local") == "greenwich":
        phase = equilibrium_argument(names, t_ref)[:, None] + u[:, None] + np.pi / 2 - phase
    else:
        phase = phase - u[:, None]
    h = tide_core.predict_ssh(
        t_sec - t_ref, get_frequencies(names), _flat_coef_(coef, "h0"), amp * f[:, None], phase
    )
    return _prediction_da_(h, t_out, coef, "h0", t_dim, "Tidal prediction")


def harmonic_predict_uv(coef, t_out, t_dim="time"):
    """Predict the currents from harmonic coefficients

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :func:`harmonic_analysis_uv`.
    t_out: array-like of datetime64 or float seconds
    t_dim: str

    Return
    ------
    xarray.DataArray, xarray.DataArray
        Eastward and northward velocities with the time dimension first.
    """
    names = list(coef["constituent"].values)
    nf = len(names)
    t_sec = _to_seconds_(t_out)
    t_ref = float(coef.attrs.get("t_ref", t_sec[0]))
    u, v = tide_core.predict_currents(
        t_sec - t_ref,
        get_frequencies(names),
        _flat_coef_(coef, "u0"),
        _flat_coef_(coef, "v0"),
        *(_flat_coef_(coef, k, nf) for k in ("semi_major", "semi_minor", "inclination", "phase")),
    )
    return (
        _prediction_da_(u, t_out, coef, "u0", t_dim, "Tidal u prediction"),
        _prediction_da_(v, t_out, coef, "u0", t_dim, "Tidal v prediction"),
    )
