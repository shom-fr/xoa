#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xoa exceptions
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


class XoaError(Exception):
    """Base exception for xoa"""

    pass


class XoaConfigError(XoaError):
    """Configuration error"""

    pass


class XoaCoordsError(XoaError):
    """Coordinates error"""

    pass


class XoaGridError(XoaError):
    """Grid error"""

    pass


class XoaDynError(XoaError):
    """Ocean dynamics error"""

    pass


class XoaKrigingError(XoaError):
    """Kriging error"""

    pass


class XoaMetaError(XoaError):
    """Metadata error"""

    pass


class XoaRegridError(XoaError):
    """Regridding error"""

    pass


class XoaSigmaError(XoaError):
    """Sigma coordinates error"""

    pass


class XoaThermdynError(XoaError):
    """Thermodynamics error"""

    pass


class XoaWarning(UserWarning):
    """Base warning for xoa"""

    pass


class XoaDeprecationWarning(XoaWarning, DeprecationWarning):
    """Deprecation warning for xoa"""

    pass


def xoa_warn(message, category=None, stacklevel=2):
    """Issue a :class:`XoaWarning` warning

    Example
    -------
    .. ipython:: python
        :okwarning:

        @suppress
        from xoa import xoa_warn
        xoa_warn('Be careful!')
    """
    if category is None:
        category = XoaWarning
    elif category == "deprecation":
        category = XoaDeprecationWarning
    warnings.warn(message, category, stacklevel=stacklevel)
