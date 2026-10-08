"""
Conservative regridding kernels

Compute the 2-D horizontal conservative weights from the overlap area
of source and destination cells.
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

import numba
import numpy as np

from .num import EPSILON
from .geo import normalize_longitude
from .poly import compute_overlap_area, spherical_area, unwrap_longitudes

NOT_CI = os.environ.get("CI", "false") == "false"


@numba.njit(cache=NOT_CI)
def _bin_range(vmin, vmax, origin, size, nbins):
    """Get the range of the bins of a regular 1D binning that is touched by an interval"""
    i0 = int(np.floor((vmin - origin) / size))
    i1 = int(np.floor((vmax - origin) / size))
    return max(i0, 0), min(i1, nbins - 1)


@numba.njit(cache=NOT_CI)
def compute_conservative_weights(
    src_grid_bounds_lat: np.ndarray,
    src_grid_bounds_lon: np.ndarray,
    dst_grid_bounds_lat: np.ndarray,
    dst_grid_bounds_lon: np.ndarray,
    dst_mask: np.ndarray = None,
) -> tuple:
    """
    Compute exact conservative remapping weights using robust geometric intersection.

    Source cells are indexed in bins, so that only the cells whose bounding box is close to
    the one of a destination cell are intersected, instead of all of them.

    Parameters
    ----------
    src_grid_bounds_lat, src_grid_bounds_lon: np.ndarray
        Source grid cell corners
    dst_grid_bounds_lat, dst_grid_bounds_lon: np.ndarray
        Destination grid cell corners

    Return
    ------
    tuple
        Tuple of (row_indices, col_indices, weights, valid_dst_points)
        where valid_dst_points is a boolean mask indicating which destination
        cells have been covered by at least one source cell.
    """
    n_src_cells = src_grid_bounds_lat.shape[0]
    n_dst_cells = dst_grid_bounds_lat.shape[0]

    # Pre-calculate destination cell areas for normalization
    dst_areas = np.zeros(n_dst_cells)
    for j in range(n_dst_cells):
        if dst_mask is None or dst_mask[j]:
            dst_areas[j] = spherical_area(dst_grid_bounds_lat[j], dst_grid_bounds_lon[j])

    # Track which destination cells have been covered by source cells
    valid_dst_points = np.zeros(n_dst_cells, dtype='?')  # nb.boolean)

    # Bounding boxes of the source cells, and of their whole set
    src_bbox = np.empty((n_src_cells, 4))  # lat min, lat max, lon min, lon max
    usable = np.zeros(n_src_cells, dtype='?')
    lat_lo = np.inf
    lat_hi = -np.inf
    lon_lo = np.inf
    lon_hi = -np.inf
    for i in range(n_src_cells):
        src_bbox[i, 0] = np.min(src_grid_bounds_lat[i])
        src_bbox[i, 1] = np.max(src_grid_bounds_lat[i])
        # Longitudes made continuous around the first corner, for cells across the dateline
        lons = unwrap_longitudes(
            src_grid_bounds_lon[i], normalize_longitude(src_grid_bounds_lon[i, 0])
        )
        src_bbox[i, 2] = np.min(lons)
        src_bbox[i, 3] = np.max(lons)
        usable[i] = np.all(np.isfinite(src_bbox[i]))
        if usable[i]:
            lat_lo = min(lat_lo, src_bbox[i, 0])
            lat_hi = max(lat_hi, src_bbox[i, 1])
            lon_lo = min(lon_lo, src_bbox[i, 2])
            lon_hi = max(lon_hi, src_bbox[i, 3])

    # Bins of about one cell, in which each source cell is registered where it is
    nbins = max(1, int(np.sqrt(n_src_cells)))
    lat_size = (lat_hi - lat_lo) / nbins if lat_hi > lat_lo else 1.0
    lon_size = (lon_hi - lon_lo) / nbins if lon_hi > lon_lo else 1.0
    counts = np.zeros(nbins * nbins + 1, dtype=np.int64)
    for i in range(n_src_cells):
        if usable[i]:
            j0, j1 = _bin_range(src_bbox[i, 0], src_bbox[i, 1], lat_lo, lat_size, nbins)
            k0, k1 = _bin_range(src_bbox[i, 2], src_bbox[i, 3], lon_lo, lon_size, nbins)
            for jb in range(j0, j1 + 1):
                for kb in range(k0, k1 + 1):
                    counts[jb * nbins + kb + 1] += 1
    offsets = np.cumsum(counts)
    entries = np.empty(offsets[-1], dtype=np.int64)
    fill = offsets[:-1].copy()
    for i in range(n_src_cells):
        if usable[i]:
            j0, j1 = _bin_range(src_bbox[i, 0], src_bbox[i, 1], lat_lo, lat_size, nbins)
            k0, k1 = _bin_range(src_bbox[i, 2], src_bbox[i, 3], lon_lo, lon_size, nbins)
            for jb in range(j0, j1 + 1):
                for kb in range(k0, k1 + 1):
                    entries[fill[jb * nbins + kb]] = i
                    fill[jb * nbins + kb] += 1

    # Storage for sparse matrix, that grows when needed
    capacity = max(1024, 4 * n_dst_cells)
    row_indices = np.empty(capacity, dtype=np.int64)
    col_indices = np.empty(capacity, dtype=np.int64)
    weights = np.empty(capacity, dtype=np.float64)
    link_count = 0

    candidates = np.empty(n_src_cells, dtype=np.int64)
    seen = np.full(n_src_cells, -1, dtype=np.int64)

    # Compute overlaps
    for j in range(n_dst_cells):
        if dst_areas[j] < EPSILON:
            continue

        # The source cells whose bins touch the destination cell, in ascending order.
        # The cell is also searched one turn of the globe away, for the cells across the dateline.
        dlat_min = np.min(dst_grid_bounds_lat[j]) - EPSILON
        dlat_max = np.max(dst_grid_bounds_lat[j]) + EPSILON
        dlons = unwrap_longitudes(
            dst_grid_bounds_lon[j], normalize_longitude(dst_grid_bounds_lon[j, 0])
        )
        dlon_min = np.min(dlons) - EPSILON
        dlon_max = np.max(dlons) + EPSILON
        if not (np.isfinite(dlat_min) and np.isfinite(dlon_min)):
            continue
        if dlat_max < lat_lo or dlat_min > lat_hi:
            continue
        n_cand = 0
        for shift in (-360.0, 0.0, 360.0):
            if dlon_max + shift < lon_lo or dlon_min + shift > lon_hi:
                continue
            j0, j1 = _bin_range(dlat_min, dlat_max, lat_lo, lat_size, nbins)
            k0, k1 = _bin_range(dlon_min + shift, dlon_max + shift, lon_lo, lon_size, nbins)
            for jb in range(j0, j1 + 1):
                for kb in range(k0, k1 + 1):
                    b = jb * nbins + kb
                    for e in range(offsets[b], offsets[b + 1]):
                        i = entries[e]
                        if seen[i] != j:
                            seen[i] = j
                            candidates[n_cand] = i
                            n_cand += 1
        cand = np.sort(candidates[:n_cand])

        for i in cand:
            # Compute overlap area
            overlap_area = compute_overlap_area(
                src_grid_bounds_lat[i],
                src_grid_bounds_lon[i],
                dst_grid_bounds_lat[j],
                dst_grid_bounds_lon[j],
            )

            # Ensure non-negative area
            overlap_area = max(0.0, overlap_area)

            if overlap_area > EPSILON:
                # Normalize by destination cell area
                weight = overlap_area / dst_areas[j]

                if weight > EPSILON:
                    if link_count == capacity:
                        capacity *= 2
                        new_rows = np.empty(capacity, dtype=np.int64)
                        new_cols = np.empty(capacity, dtype=np.int64)
                        new_weights = np.empty(capacity, dtype=np.float64)
                        new_rows[:link_count] = row_indices[:link_count]
                        new_cols[:link_count] = col_indices[:link_count]
                        new_weights[:link_count] = weights[:link_count]
                        row_indices = new_rows
                        col_indices = new_cols
                        weights = new_weights
                    row_indices[link_count] = j
                    col_indices[link_count] = i
                    weights[link_count] = weight
                    link_count += 1

                    # Mark this destination cell as covered
                    valid_dst_points[j] = True

    # Renormalize rows to 1: boundary dst cells only partially overlap the src domain,
    # so raw weights (overlap/dst_area) sum to < 1. Dividing by the actual row sum
    # ensures a constant field always maps to the same constant everywhere.
    row_sums = np.zeros(n_dst_cells)
    for k in range(link_count):
        row_sums[row_indices[k]] += weights[k]
    for k in range(link_count):
        j = row_indices[k]
        if row_sums[j] > EPSILON:
            weights[k] /= row_sums[j]

    return (
        row_indices[:link_count],
        col_indices[:link_count],
        weights[:link_count],
        valid_dst_points,
    )
