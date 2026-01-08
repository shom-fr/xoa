#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Geographic utilities
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

import math
import numba


@numba.vectorize(cache=True)
def haversine(lon0, lat0, lon1, lat1):
    """Haversine distance between two points on a **unit sphere**

    Parameters
    ----------
    lon0: float, array_like
        Longitude of the first point(s)
    lat0: float, array_like
        Latitude of the first point(s)
    lon1: float, array_like
        Longitude of the second point(s)
    lat1: float, array_like
        Latitude of the second point(s)

    Return
    ------
    float, array_like
        Distance(s)
    """
    deg2rad = math.pi / 180.0
    dist = math.sin(deg2rad * (lat0 - lat1) * 0.5) ** 2
    dist += (
        math.cos(deg2rad * lat0)
        * math.cos(deg2rad * lat1)
        * math.sin(deg2rad * (lon0 - lon1) * 0.5) ** 2
    )
    dist = 2.0 * math.asin(math.sqrt(dist))
    return dist


@numba.vectorize
def bearing(lon0, lat0, lon1, lat1):
    """Compute the bearing angle (forward azimuth)

    Parameters
    ----------
    lon0: float, array_like
        Longitude of the first point(s)
    lat0: float, array_like
        Latitude of the first point(s)
    lon1: float, array_like
        Longitude of the second point(s)
    lat1: float, array_like
        Latitude of the second point(s)

    Return
    ------
    float, array_like
        Angle(s)
    """
    deg2rad = math.pi / 180.0
    a = math.atan2(
        math.cos(deg2rad * lat0) * math.sin(deg2rad * lat1)
        - math.sin(deg2rad * lat0) * math.cos(deg2rad * lat1) * math.cos(deg2rad * (lon1 - lon0)),
        math.sin(deg2rad * (lon1 - lon0)) * math.cos(deg2rad * lat1),
    )
    return a * 180 / math.pi
