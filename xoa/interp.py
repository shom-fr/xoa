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
from .core.interp import (  # noqa
    nearest1d,
    linear1d,
    cubic1d,
    hermit1d,
    extrap1d,
    cellave1d,
    closest2d,
    cell2relloc,
    grid2relloc,
    grid2rellocs,
    grid2locs,
    isoslice,
)

warnings.warn("The 'xoa.interp' module is deprecated in favour of the 'xoa.core.interp' module")
