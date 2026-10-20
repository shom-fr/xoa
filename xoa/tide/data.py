"""
Tidal constituents database: frequencies, equilibrium arguments and nodal corrections

References
----------
Doodson, A.T. (1921). The harmonic development of the tide-generating potential.
Schureman, P. (1958). Manual of Harmonic Analysis and Prediction of Tides.
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

import os

import numpy as np
from configobj import Section
from validate import VdtValueError

from .. import cfgm
from ..core import tide as tide_core

_THISDIR = os.path.dirname(__file__)

#: Constituent database specification file
INI_FILE = os.path.join(_THISDIR, "tide.ini")

#: Default constituent database file
CFG_FILE = os.path.join(_THISDIR, "tide.cfg")

# %% Doodson argument speeds

# Angular speeds of the 6 Doodson arguments (rad/s), from Schureman (1958) Table 1
# Original rates in degrees per mean solar hour; converted here to rad/s.
DOODSON_SPEEDS = (
    np.deg2rad(
        np.array(
            [
                14.49205211,  # τ   mean lunar time
                0.54901653,  # s   mean Moon longitude      (period ≈ 27.32 d)
                0.04106864,  # h   mean Sun longitude        (period ≈ 365.25 d)
                0.00464184,  # p   Moon's perigee longitude  (period ≈ 8.85 yr)
                0.00220641,  # N'  = −N (positive nodal arg) (period ≈ 18.61 yr)
                0.00001961,  # p₁  Sun's perihelion          (period ≈ 20940 yr)
            ]
        )
    )
    / 3600.0
)

# %% constituent database


def is_satellites(value):
    """Validate a list of tidal satellite phasors

    It is a semicolon-separated string of whitespace-delimited 5-tuples
    ``r dN dp dp1 phi0_deg``.

    Return
    ------
    list of tuples
        Empty when the value is empty.
    """
    if value is None or not str(value).strip():
        return []
    sats = []
    for token in str(value).split(";"):
        vals = token.split()
        if not vals:
            continue
        try:
            if len(vals) != 5:
                raise ValueError
            sats.append(tuple(float(v) for v in vals))
        except ValueError:
            raise VdtValueError(value)
    return sats


def get_user_cfg_file():
    """Get the default user constituents file name"""
    from ..options import get_default_user_config_file

    return os.path.join(os.path.dirname(get_default_user_config_file()), "tide.cfg")


def _load_constituent_db_():
    """Load the constituent database from the default and the user files

    The sections of the user file ``tide.cfg``, located next to the xoa user
    config file, complete or replace those of the default file.
    """
    cfgmanager = cfgm.ConfigManager(
        INI_FILE, interpolation=False, validator={"satellites": is_satellites}
    )
    db = {}
    for cfgfile in CFG_FILE, get_user_cfg_file():
        if not os.path.exists(cfgfile):
            continue
        cfg = cfgmanager.load(cfgfile, validate="raise")
        for name, sec in cfg.items():
            if not isinstance(sec, Section):
                continue
            compound_of = [
                (k, v["multiplier"])
                for k, v in sec.get("compound_of", {}).items()
                if isinstance(v, Section)
            ]
            db[name] = {
                "doodson": sec["doodson"],
                "species": sec["species"],
                "satellites": sec["satellites"],
                "compound_of": compound_of or None,
            }
    return db


#: Constituent database
CONSTITUENT_DB = _load_constituent_db_()

#: Constituent names
CONSTITUENT_NAMES = list(CONSTITUENT_DB)


# %% frequencies


def get_frequencies(names):
    """Angular frequencies in rad/s for named constituents

    Parameters
    ----------
    names: list of str

    Return
    ------
    numpy.ndarray
        Array of size ``nf``
    """
    return np.array(
        [
            np.dot(tide_core.doodson_multipliers(CONSTITUENT_DB[n]["doodson"]), DOODSON_SPEEDS)
            for n in names
        ]
    )


def auto_constituents(record_days, rayleigh_min=1.0):
    """Select the constituents resolvable from a record of given length

    Two constituents are resolved when their frequency separation exceeds
    ``rayleigh_min / record_days`` (cycles per day): this is the Rayleigh criterion.

    Parameters
    ----------
    record_days: float
    rayleigh_min: float
        Rayleigh criterion multiplier

    Return
    ------
    list of str
        Subset of non-compound :data:`CONSTITUENT_NAMES`, sorted by frequency
    """
    names = [n for n in CONSTITUENT_NAMES if CONSTITUENT_DB[n]["compound_of"] is None]
    freqs = get_frequencies(names) * 86400.0 / (2 * np.pi)  # cycles per day
    order = np.argsort(freqs)
    names, freqs = [names[i] for i in order], freqs[order]
    selected = [names[0]]
    for i in range(1, len(names)):
        if freqs[i] - freqs[i - 1] >= rayleigh_min / record_days:
            selected.append(names[i])
    return selected


def beat_period(name1, name2):
    """Beat period of two constituents, in days

    This is the record length needed to separate them (Rayleigh criterion).
    """
    omega = get_frequencies([name1, name2])
    return float(2 * np.pi / abs(omega[0] - omega[1]) / 86400.0)


def unresolved_pairs(names, record_days, rayleigh_min=1.0):
    """Pairs of constituents that a record of given length cannot separate

    Parameters
    ----------
    names: list of str
    record_days: float
    rayleigh_min: float
        Rayleigh criterion multiplier

    Return
    ------
    list of (str, str)
    """
    return [
        (n1, n2)
        for i, n1 in enumerate(names)
        for n2 in names[i + 1 :]
        if beat_period(n1, n2) > record_days / rayleigh_min
    ]


# %% equilibrium argument and nodal corrections


def equilibrium_argument(names, t_ref):
    """Astronomical equilibrium argument V0 of named constituents at ``t_ref``

    It is the Greenwich phase of each tidal frequency at ``t_ref``, before any
    nodal or local phase adjustment.

    Parameters
    ----------
    names: list of str
    t_ref: float
        Seconds since 1970-01-01 00:00:00 UTC

    Return
    ------
    numpy.ndarray
        Argument in radians, in [0, 2π)

    References
    ----------
    B. Simon (2007), Analyse Harmonique des Marées, table 5.3.
    """
    s, h, p, _, Np, p1 = tide_core.astronomical_args(t_ref)
    # τ = 15°·t_UTC_hours + h − s (mean lunar civil time angle)
    tau = (15.0 * (t_ref % 86400) / 3600 + h - s) % 360.0
    args = np.array([tau, s, h, p, Np, p1])
    return np.array(
        [
            np.deg2rad(np.dot(tide_core.doodson_multipliers(CONSTITUENT_DB[n]["doodson"]), args))
            % (2 * np.pi)
            for n in names
        ]
    )


def nodal_corrections(t_ref, names):
    """Nodal amplitude factor ``f`` and phase correction ``u`` of named constituents

    Parameters
    ----------
    t_ref: float
        Reference time in seconds since 1970-01-01, typically mid-record
    names: list of str

    Return
    ------
    f: numpy.ndarray
        Amplitude factors, close to 1
    u: numpy.ndarray
        Phase corrections in radians
    """
    _, _, p, N, _, p1 = tide_core.astronomical_args(t_ref)
    f = np.ones(len(names))
    u = np.zeros(len(names))
    for i, name in enumerate(names):
        c = CONSTITUENT_DB[name]
        if c["compound_of"]:
            f[i], u[i] = 1.0, 0.0
            for comp_name, mult in c["compound_of"]:
                fc, uc = nodal_corrections(t_ref, [comp_name])
                f[i] *= float(fc[0]) ** mult
                u[i] += mult * float(uc[0])
        else:
            z = complex(1.0, 0.0)
            for r, dN, dp, dp1, phi0 in c["satellites"]:
                z += r * np.exp(1j * np.deg2rad(dN * N + dp * p + dp1 * p1 + phi0))
            f[i], u[i] = abs(z), np.angle(z)
    return f, u
