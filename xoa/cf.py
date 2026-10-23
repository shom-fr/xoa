#!/usr/bin/env python3
"""
Backward compatibility layer for the cf module

This module provides backward compatibility by importing from the meta module
and aliasing to old cf names.

.. deprecated::
    The cf module is deprecated. Please use :mod:`xoa.meta` instead.
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
import warnings

from .cf_configs import CF_CONFIGS, get_cf_config_file  # noqa

# Import from meta module and create aliases
from .meta import (  # noqa
    are_similar,
    get_matching_item_specs,
    infer_coords,
    reset_cache,
    search_similar,
    show_cache,
)
from .meta.categories import (  # noqa
    _MetaCatSpecs_ as _CFCatSpecs_,
)
from .meta.general import MetaSpecs as CFSpecs  # noqa
from .meta.sglocator import SGLocator  # noqa
from .misc import Choices

# Re-create ATTRS_PATCH_MODE
ATTRS_PATCH_MODE = Choices(
    {
        "fill": "do not erase existing attributes, just fill missing ones",
        "replace": "replace existing attributes",
    },
    parameter="mode",
    description="policy for patching existing attributes",
)

warnings.warn(
    "The 'xoa.cf' module is deprecated. Please use 'xoa.meta' instead"
    " and replace all 'cf' by 'meta' in functions and classes"
)
