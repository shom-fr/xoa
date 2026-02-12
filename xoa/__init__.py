#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xarray-based ocean analysis library
"""
# Copyright 2020-2024 Shom
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


from .exceptions import (  # noqa: F401
    XoaError,
    XoaWarning,
    XoaDeprecationWarning,
    XoaConfigError,
    xoa_warn,
)
from .meta import get_meta_config_file  # noqa: F401
from .meta.configs import META_CONFIGS  # noqa: F401
from .data_samples import (  # noqa: F401
    get_data_sample,
    show_data_samples,
    open_data_sample,
)
from .accessors import register_accessors

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "0.0.0"

__all__ = [
    "get_meta_config_file",
    "get_data_sample",
    "show_data_samples",
    "open_data_sample",
    "xoa_warn",
    "XoaError",
    "XoaWarning",
    "get_default_user_config_file",
    "load_options",
    "get_option",
    "set_options",
    "show_versions",
    "show_paths",
    "show_info",
    "register_accessors",
]


# Directory of sample files
_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "_samples")

_XOA_CACHE = {}
