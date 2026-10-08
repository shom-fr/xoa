"""
Spherical polygon geometry primitives

Provides spherical area, orientation, Sutherland-Hodgman clipping and
overlap area. All functions operate on plain numpy arrays (lon/lat in degrees).
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

import numba as nb
import numpy as np

from .geo import DEG2RAD, EARTH_RADIUS, diff_lon, normalize_longitude
from .num import EPSILON

NOT_CI = os.environ.get("CI", "false") == "false"
# %% Area


@nb.njit(cache=NOT_CI, nogil=True)
def spherical_area(lats, lons):
    """Calculate a polygon area in spherical coordinates"""
    n = len(lats)
    if n < 3:
        return 0.0

    # Radians
    lats_rad = lats * DEG2RAD
    lons_rad = lons * DEG2RAD

    area = 0.0
    for i in range(n):
        j = (i + 1) % n

        lat1, lon1 = lats_rad[i], lons_rad[i]
        lat2, lon2 = lats_rad[j], lons_rad[j]

        # Cyclic
        dlon = lon2 - lon1
        if abs(dlon) > np.pi:
            dlon = dlon - np.sign(dlon) * 2 * np.pi

        # Integrate
        area += dlon * (np.sin(lat1) + np.sin(lat2))

    return abs(area) * EARTH_RADIUS**2 / 2.0


# %% Bounding box overlap


@nb.njit(cache=NOT_CI, nogil=True)
def quick_overlap_check(src_lats, src_lons, dst_lats, dst_lons):
    """
    Quick bounding box check to eliminate obviously non-overlapping cells.
    """
    src_lat_min, src_lat_max = np.min(src_lats), np.max(src_lats)
    src_lon_min, src_lon_max = np.min(src_lons), np.max(src_lons)
    dst_lat_min, dst_lat_max = np.min(dst_lats), np.max(dst_lats)
    dst_lon_min, dst_lon_max = np.min(dst_lons), np.max(dst_lons)

    # Check latitude overlap
    if src_lat_max < dst_lat_min - EPSILON or dst_lat_max < src_lat_min - EPSILON:
        return False

    # Check longitude overlap, the longitudes being continuous and in the same range
    if src_lon_max < dst_lon_min - EPSILON or dst_lon_max < src_lon_min - EPSILON:
        return False

    return True


# %% Sutherland-Hodgman clipping


@nb.njit(cache=NOT_CI, nogil=True)
def is_point_on_edge(p1, p2, q):
    """
    Check if point q is on the inside of the edge defined by p1->p2.
    Uses cross product to determine which side of the line the point is on.
    We assume counter-clockwize order.

    Parameters
    ----------
    p1, p2:
        Two points defining the edge (tuples or arrays of length 2)
    q:
        Query point (tuple or array of length 2)

    Return
    ------
    bool
        True if point is inside (left side of edge), False otherwise
    """
    R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
    return R >= 0


@nb.njit(cache=NOT_CI, nogil=True)
def compute_intersection(p1, p2, p3, p4):
    """
    Compute intersection point between two line segments.
    First segment: p1->p2, Second segment: p3->p4

    Parameters
    ----------
    p1, p2:
        Points defining first line segment
    p3, p4:
        Points defining second line segment

    Return
    ------
    tuple
        Intersection point as (x, y) tuple
    """
    # Handle vertical lines to avoid division by zero
    if abs(p2[0] - p1[0]) < 1e-10:  # First line is vertical
        x = p1[0]
        # Slope and intercept of second line
        if abs(p4[0] - p3[0]) < 1e-10:  # Both lines vertical - should not happen in proper usage
            return (p1[0], p1[1])  # Return arbitrary point
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]
        y = m2 * x + b2

    elif abs(p4[0] - p3[0]) < 1e-10:  # Second line is vertical
        x = p3[0]
        # Slope and intercept of first line
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]
        y = m1 * x + b1

    else:  # Neither line is vertical
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]

        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]

        # Check for parallel lines
        if abs(m1 - m2) < 1e-10:
            return (p1[0], p1[1])  # Return arbitrary point for parallel lines

        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1

    return (x, y)


@nb.njit(cache=NOT_CI, nogil=True)
def clip_polygon_against_edge(subject_polygon, clip_edge_start, clip_edge_end):
    """
    Clip a polygon against a single edge using Sutherland-Hodgman algorithm.

    Parameters
    ----------
    subject_polygon:
        Input polygon as array of (x, y) points
    clip_edge_start, clip_edge_end:
        Points defining the clipping edge

    Return
    ------
    array(n, 2)
        Clipped polygon as array of (x, y) points
    """
    if len(subject_polygon) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Use a temporary list to collect output vertices
    output_vertices = []

    # Process each edge of the subject polygon
    for j in range(len(subject_polygon)):
        # Current edge of subject polygon
        s_edge_start = subject_polygon[j - 1]  # Previous vertex (wraps around)
        s_edge_end = subject_polygon[j]  # Current vertex

        if is_point_on_edge(clip_edge_start, clip_edge_end, s_edge_end):
            # End vertex is inside
            if not is_point_on_edge(clip_edge_start, clip_edge_end, s_edge_start):
                # Start vertex is outside, end vertex is inside
                # Add intersection point
                intersection = compute_intersection(
                    s_edge_start, s_edge_end, clip_edge_start, clip_edge_end
                )
                output_vertices.append(intersection)
            # Add the end vertex
            output_vertices.append((s_edge_end[0], s_edge_end[1]))

        elif is_point_on_edge(clip_edge_start, clip_edge_end, s_edge_start):
            # Start vertex is inside, end vertex is outside
            # Add only the intersection point
            intersection = compute_intersection(
                s_edge_start, s_edge_end, clip_edge_start, clip_edge_end
            )
            output_vertices.append(intersection)
        # If both vertices are outside, add nothing

    # Convert list to numpy array
    if len(output_vertices) == 0:
        return np.empty((0, 2), dtype=np.float64)

    result = np.empty((len(output_vertices), 2), dtype=np.float64)
    for i in range(len(output_vertices)):
        result[i, 0] = output_vertices[i][0]
        result[i, 1] = output_vertices[i][1]

    return result


@nb.njit(cache=NOT_CI, nogil=True)
def clip_polygon(subject_polygon, clipping_polygon):
    """
    Clip subject_polygon against clipping_polygon using Sutherland-Hodgman algorithm.

    Parameters
    ----------
    subject_polygon: array(n, 2)
        Polygon to be clipped
    clipping_polygon: array(m, 2)
        Clipping polygon

    Return
    ------
    array(k, 2)
        Clipped polygon with k <= n
    """
    if len(subject_polygon) == 0 or len(clipping_polygon) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Start with the original subject polygon
    current_polygon = subject_polygon.copy()

    # Clip against each edge of the clipping polygon
    for i in range(len(clipping_polygon)):
        if len(current_polygon) == 0:
            break

        # Define the current clipping edge
        clip_edge_start = clipping_polygon[i - 1]  # Previous vertex (wraps around)
        clip_edge_end = clipping_polygon[i]  # Current vertex

        # Clip current polygon against this edge
        current_polygon = clip_polygon_against_edge(current_polygon, clip_edge_start, clip_edge_end)

    return current_polygon


# %% Overlap area


@nb.njit(cache=NOT_CI, nogil=True)
def unwrap_longitudes(lons, ref):
    """Make the longitudes of a polygon continuous, in the 180° range around a reference

    A polygon that crosses the dateline, like ``[179, -179, -179, 179]``, becomes
    ``[179, 181, 181, 179]`` when ``ref=179``.
    """
    out = np.empty(lons.size)
    for i in range(lons.size):
        out[i] = ref + diff_lon(ref, lons[i])
    return out


@nb.njit(cache=NOT_CI, nogil=True)
def compute_overlap_area(src_bounds_lat, src_bounds_lon, dst_bounds_lat, dst_bounds_lon):
    """
    Compute overlap area between two spherical polygon cells that are defined by their bounds

    Each cell is made continuous in longitude and the destination one is shifted by a
    multiple of 360° to be next to the source one, so that cells that cross the dateline
    or that are on the other side of it are handled.
    """
    # Handle degenerate cases
    if len(src_bounds_lat) < 3 or len(dst_bounds_lat) < 3:
        return 0.0

    # Continuous longitudes, in the same range
    src_ref = normalize_longitude(src_bounds_lon[0])
    src_lons = unwrap_longitudes(src_bounds_lon, src_ref)
    dst_lons = unwrap_longitudes(dst_bounds_lon, normalize_longitude(dst_bounds_lon[0]))
    dst_lons += 360.0 * np.round((np.mean(src_lons) - np.mean(dst_lons)) / 360.0)

    # Quick overlap check first
    if not quick_overlap_check(src_bounds_lat, src_lons, dst_bounds_lat, dst_lons):
        return 0.0

    # Cells that are too wide, like the ones that contain a pole, are not handled
    src_span = np.max(src_lons) - np.min(src_lons)
    dst_span = np.max(dst_lons) - np.min(dst_lons)
    if src_span > 180.0 or dst_span > 180.0:
        return 0.0

    # Null areas
    src_area = spherical_area(src_bounds_lat, src_bounds_lon)
    dst_area = spherical_area(dst_bounds_lat, dst_bounds_lon)

    if src_area < EPSILON or dst_area < EPSILON:
        return 0.0

    # Intersection
    src_poly = np.empty((src_bounds_lat.size, 2))
    src_poly[:, 0] = src_lons
    src_poly[:, 1] = src_bounds_lat
    dst_poly = np.empty((dst_bounds_lat.size, 2))
    dst_poly[:, 0] = dst_lons
    dst_poly[:, 1] = dst_bounds_lat
    clip_poly = clip_polygon(src_poly, dst_poly)

    # Area
    clip_area = spherical_area(clip_poly[:, 1], clip_poly[:, 0])
    if clip_area < EPSILON:
        return 0.0
    return clip_area
