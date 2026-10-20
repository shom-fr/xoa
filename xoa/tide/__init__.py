"""
Tides: constituents, tidal filters, harmonic analysis and prediction, ellipses and diagnostics

The numba and numpy kernels are in :mod:`xoa.core.tide` and :mod:`xoa.core.signal`.

.. rubric:: How to use it

See the :ref:`indepth.tide` section.
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

from .data import (  # noqa: F401
    CONSTITUENT_DB,
    CONSTITUENT_NAMES,
    auto_constituents,
    beat_period,
    equilibrium_argument,
    get_frequencies,
    nodal_corrections,
    unresolved_pairs,
)
from .diags import (  # noqa: F401
    form_number,
    high_waters,
    skew_surge,
    spring_high_waters,
    tidal_atlas,
    tidal_range,
)
from .ellipse import ellipse_outline, tidal_ellipses  # noqa: F401
from .filter import HOURLY_TIDAL_FILTERS_WEIGHTS, demerliac, tidal_filter  # noqa: F401
from .harmonic import (  # noqa: F401
    TidalHarmonic,
    harmonic_analysis,
    harmonic_analysis_uv,
    harmonic_predict,
    harmonic_predict_uv,
)
