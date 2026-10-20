"""
Tidal ellipses of the currents
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

from ..core import tide as tide_core

# %% ellipses


def tidal_ellipses(coef):
    """Add ellipse descriptors to a current harmonic coefficient dataset

    Parameters
    ----------
    coef: xarray.Dataset
        Output of :meth:`TidalHarmonic.analyse_uv`.

    Return
    ------
    xarray.Dataset
        Copy of ``coef`` with the extra variables:

        - ``ellipticity``: ``semi_minor / semi_major``, in [-1, 1]
        - ``rotation``: +1 counter-clockwise, -1 clockwise, 0 rectilinear
        - ``wplus_amp``, ``wplus_phase``: counter-clockwise circular component
        - ``wminus_amp``, ``wminus_phase``: clockwise circular component

        Angles are in radians.
    """
    smaj, smin = coef["semi_major"], coef["semi_minor"]
    ellipticity, rotation = tide_core.ellipse_properties(smaj.values, smin.values)
    wp, wm = tide_core.rotary_components(
        smaj.values, smin.values, coef["inclination"].values, coef["phase"].values
    )

    def da(values, long_name, units="1"):
        return xr.DataArray(
            values,
            dims=smaj.dims,
            coords=smaj.coords,
            attrs={"long_name": long_name, "units": units},
        )

    out = coef.copy()
    out["ellipticity"] = da(ellipticity, "Ellipticity (semi-minor / semi-major)")
    out["rotation"] = da(rotation, "Rotation sense (+1 counter-clockwise, -1 clockwise)")
    out["wplus_amp"] = da(np.abs(wp), "Counter-clockwise circular component amplitude", "m s-1")
    out["wplus_phase"] = da(np.angle(wp), "Counter-clockwise circular component phase", "rad")
    out["wminus_amp"] = da(np.abs(wm), "Clockwise circular component amplitude", "m s-1")
    out["wminus_phase"] = da(np.angle(wm), "Clockwise circular component phase", "rad")
    return out


def ellipse_outline(coef, n=100):
    """Points of the tidal ellipses, centred on the origin

    Parameters
    ----------
    coef: xarray.Dataset
        Dataset with the ellipse parameters, see :func:`tidal_ellipses`.
    n: int
        Number of points along the ellipse.

    Return
    ------
    xarray.DataArray, xarray.DataArray
        Components with an additional ``ellipse_point`` dimension, ordered
        along the current rotation starting at ``ω(t - t_ref) = 0``.
    """
    args = [coef[k] for k in ("semi_major", "semi_minor", "inclination", "phase")]
    u, v = tide_core.ellipse_vector(*(a.values for a in args), np.linspace(0.0, 2 * np.pi, n))
    dims = (*args[0].dims, "ellipse_point")
    return (
        xr.DataArray(u, dims=dims, coords=args[0].coords, attrs={"units": "m s-1"}),
        xr.DataArray(v, dims=dims, coords=args[0].coords, attrs={"units": "m s-1"}),
    )
