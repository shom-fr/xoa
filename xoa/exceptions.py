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
    pass


class XoaConfigError(XoaError):
    pass


class XoaCoordsError(XoaError):
    pass


class XoaGridError(XoaError):
    pass


class XoaDynError(XoaError):
    pass


class XoaKrigingError(XoaError):
    pass


class XoaMetaError(XoaError):
    pass


class XoaRegridError(XoaError):
    pass


class XoaSigmaError(XoaError):
    pass


class XoaThermdynError(XoaError):
    pass


class XoaWarning(UserWarning):
    pass


class XoaDeprecationWarning(XoaWarning, DeprecationWarning):
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
