#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

# Import from meta module and create aliases
from .exceptions import XoaMetaError as XoaCFError
from .meta import (  # noqa
    get_matching_item_specs,
    are_similar,
    search_similar,
    set_meta_specs as set_cf_specs,
    reset_cache,
    show_cache,
    get_meta_specs_from_name as get_cf_specs_from_name,
    get_meta_specs_encoding as get_cf_specs_encoding,
    get_meta_specs_from_encoding as get_cf_specs_from_encoding,
    get_default_meta_specs as get_default_cf_specs,
    get_meta_specs as get_cf_specs,
    register_meta_specs as register_cf_specs,
    get_registered_meta_specs as get_registered_cf_specs,
    is_registered_meta_specs as is_registered_cf_specs,
    get_meta_specs_matching_score as get_cf_specs_matching_score,
    infer_meta_specs as infer_cf_specs,
    assign_meta_specs as assign_cf_specs,
    infer_coords,
    USER_META_FILE as USER_CF_FILE,
)

from .meta.sglocator import SGLocator  # noqa
from .meta.general import MetaSpecs as CFSpecs  # noqa
from .meta.categories import (  # noqa
    _MetaCatSpecs_ as _CFCatSpecs_,
    MetaVarSpecs as CFVarSpecs,
    MetaCoordSpecs as CFCoordSpecs,
)

from .cf_configs import CF_CONFIGS, get_cf_config_file  # noqa
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
    "The 'xoa.cf' module is deprecated. Pease use 'xoa.meta' instead"
    " and replace all 'cf' by 'meta' in functions and classes"
)
