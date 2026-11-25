#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
To generate the explanatory figures of the in-depth guides during the documentation compilation
"""

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

from xoa.core.grid import centers2edges, edges2bounds, create_rotated_grid
from xoa.core.poly import clip_polygon, spherical_area
from xoa.core.spline import compute_cubic_weights_1d, compute_frac_indices, hermite_basis


def _polygons(blon, blat, **kwargs):
    """Polygons of cells from their corners with shape ``(ny, nx, 4)``"""
    return [
        Polygon(np.column_stack([lo, la]), **kwargs)
        for lo, la in zip(blon.reshape(-1, 4), blat.reshape(-1, 4))
    ]


def gen_interp_vs_regrid(outfile):
    """Figure that explains the difference between interpolation and regridding

    Interpolation goes from the values at points to the values at any other points, with
    weights that are the ones of the surrounding points. Conservative regridding goes from
    the mean values over cells to the mean values over the cells of another grid, with weights
    that are the overlap areas of the cells.
    """
    # The source values are known at the nodes of a regular grid, which are the centers of cells
    slon, slat = np.meshgrid(np.arange(7.0), np.arange(5.0))
    values = np.sin(0.9 * slon) * np.cos(0.8 * slat) + 0.15 * slon
    norm, cmap = plt.Normalize(values.min(), values.max()), plt.get_cmap("viridis")
    sblon = edges2bounds(centers2edges(slon))
    sblat = edges2bounds(centers2edges(slat))

    # Interpolation goes to any points, and regridding to the cells of another grid
    rng = np.random.default_rng(3)
    plon, plat = rng.uniform(0.7, 5.3, 8), rng.uniform(0.7, 3.3, 8)
    target = (2.4, 1.6)  # the point whose weights are shown
    grid = create_rotated_grid(4, 3, 3.0, 2.0, 18.0, lon_span=4.2, lat_span=2.6)
    dblon, dblat = grid["lon_bounds"], grid["lat_bounds"]
    cell = (1, 2)  # the destination cell whose weights are shown

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.4), sharex=True, sharey=True)
    for ax in axes.ravel():
        ax.set(xlim=(-1.2, 7.4), ylim=(-1.0, 5.2), xticks=[], yticks=[], aspect="equal")
    axes[0, 0].set_title("Interpolation\npoints to points", fontsize=11)
    axes[0, 1].set_title("Regridding (conservative)\ncells to cells", fontsize=11)
    axes[0, 0].set_ylabel("Input", fontsize=12)
    axes[1, 0].set_ylabel("Output", fontsize=12)
    captions = [
        "values at the nodes",
        "mean values over the cells",
        "values at any location",
        "mean values over the cells of another grid",
    ]
    for ax, caption in zip(axes.ravel(), captions):
        ax.text(0.5, 0.02, caption, transform=ax.transAxes, ha="center", fontsize=9, style="italic")

    # Input
    axes[0, 0].scatter(
        slon, slat, c=values, cmap=cmap, norm=norm, s=80, edgecolor="k", linewidth=0.5
    )
    for patch, value in zip(_polygons(sblon, sblat, edgecolor="w"), values.ravel()):
        patch.set_facecolor(cmap(norm(value)))
        axes[0, 1].add_patch(patch)

    # Output of the interpolation: a weighted mean of the surrounding points
    ax = axes[1, 0]
    ax.scatter(slon, slat, c="0.8", s=40, zorder=2)
    ax.scatter(plon, plat, marker="*", s=60, c="0.5", zorder=3)
    i0, j0 = int(target[0]), int(target[1])
    a, b = target[0] - i0, target[1] - j0
    surrounding = [
        (i0, j0, (1 - a) * (1 - b)),
        (i0 + 1, j0, a * (1 - b)),
        (i0, j0 + 1, (1 - a) * b),
        (i0 + 1, j0 + 1, a * b),
    ]
    for ci, cj, weight in surrounding:
        ax.plot(
            [target[0], ci], [target[1], cj], color="C3", lw=1 + 8 * weight, alpha=0.8, zorder=4
        )
        ax.scatter(
            [ci],
            [cj],
            c=[values[cj, ci]],
            cmap=cmap,
            norm=norm,
            s=80,
            edgecolor="k",
            linewidth=0.5,
            zorder=5,
        )
    ax.scatter(*target, marker="*", s=260, c="C3", edgecolor="k", zorder=6)
    ax.text(5.9, 4.9, "line width = weight", ha="center", fontsize=8)

    # Output of the regridding: a weighted mean of the overlapping cells
    ax = axes[1, 1]
    for patch in _polygons(sblon, sblat, facecolor="none", edgecolor="0.75", linewidth=0.8):
        ax.add_patch(patch)
    for patch in _polygons(dblon, dblat, facecolor="none", edgecolor="C3", linewidth=1.6):
        ax.add_patch(patch)
    dpoly = np.column_stack([dblon[cell], dblat[cell]])
    darea = spherical_area(dblat[cell], dblon[cell])
    overlaps = []
    for lo, la in zip(sblon.reshape(-1, 4), sblat.reshape(-1, 4)):
        clipped = clip_polygon(np.column_stack([lo, la]), dpoly)
        if len(clipped) >= 3:
            weight = spherical_area(clipped[:, 1], clipped[:, 0]) / darea
            if weight > 1e-3:
                overlaps.append((clipped, weight))
    for n, (clipped, weight) in enumerate(overlaps):
        ax.add_patch(
            Polygon(
                clipped, facecolor=plt.get_cmap("Set2")(n), edgecolor="k", linewidth=0.6, zorder=3
            )
        )
        if weight > 0.03:  # slivers are not labelled
            ax.text(
                *clipped.mean(axis=0),
                f"{weight:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                zorder=4,
            )
    ax.text(5.9, 4.9, "area = weight", ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)


def gen_method_stencils(outfile):
    """Figure that shows which points or cells each method uses, and with which weights

    The indices and the weights are the ones of the kernels, on a curvilinear grid.
    """
    # A gently curvilinear grid
    ny, nx = 6, 7
    x, y = np.meshgrid(np.arange(nx, dtype=float), np.arange(ny, dtype=float))
    lon = x + 0.18 * np.sin(0.9 * y) + 0.03 * y
    lat = y + 0.12 * np.sin(0.8 * x) - 0.02 * x
    elon, elat = centers2edges(lon), centers2edges(lat)
    cblon, cblat = edges2bounds(elon), edges2bounds(elat)

    # The target is inside a cell, from its relative position in the cell
    j, i, a, b = 2, 3, 0.4, 0.6
    corners = [(j, i), (j, i + 1), (j + 1, i + 1), (j + 1, i)]
    w4 = np.array([(1 - a) * (1 - b), a * (1 - b), a * b, (1 - a) * b])
    tlon = sum(w * lon[c] for w, c in zip(w4, corners))
    tlat = sum(w * lat[c] for w, c in zip(w4, corners))
    jb, ib, fa, fb = compute_frac_indices(
        lon, lat, np.array([[tlon]]), np.array([[tlat]]), 2, stencil_margin=1
    )
    assert (jb[0], ib[0]) == (j, i)  # the kernel finds the same cell
    xw = compute_cubic_weights_1d(hermite_basis(fa[0]), 0.0, 0.0)
    yw = compute_cubic_weights_1d(hermite_basis(fb[0]), 0.0, 0.0)
    cubic = np.outer(yw, xw)  # (4, 4) weights of the nodes j-1..j+2 and i-1..i+2

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharex=True, sharey=True)
    titles = ["Bilinear", "Bicubic", "Conservative"]
    captions = [
        "4 points: the corners of the cell\nthat contains the target, found\nwith the indices of the grid",
        "16 points: the 4 x 4 nodes around\nthe cell, with weights that may be\nnegative, for continuous slopes",
        "the cells that the target cell overlaps,\nweighted by the fractions of its area:\nthe cell bounds are needed",
    ]
    for ax, title, caption in zip(axes, titles, captions):
        for patch in _polygons(cblon, cblat, facecolor="none", edgecolor="0.7", linewidth=0.8):
            ax.add_patch(patch)
        ax.set(xlim=(-0.8, nx - 0.2), ylim=(-0.8, ny - 0.2), xticks=[], yticks=[], aspect="equal")
        ax.set_title(title, fontsize=13, color="C0")
        ax.text(0.5, -0.02, caption, transform=ax.transAxes, ha="center", va="top", fontsize=10)
        ax.scatter(lon, lat, s=12, c="0.55", zorder=2)

    # Bilinear
    ax = axes[0]
    ax.add_patch(Polygon([(lon[c], lat[c]) for c in corners], facecolor="C3", alpha=0.15, zorder=1))
    for w, c in zip(w4, corners):
        ax.plot([tlon, lon[c]], [tlat, lat[c]], color="C3", lw=1 + 6 * w, zorder=3)
        ax.scatter(lon[c], lat[c], s=40 + 260 * w, c="C3", edgecolor="k", linewidth=0.5, zorder=4)
        ax.text(lon[c] + 0.1, lat[c] + 0.1, f"{w:.2f}", fontsize=8, zorder=5)
    ax.scatter(tlon, tlat, s=170, c="C1", edgecolor="k", zorder=6)

    # Bicubic
    ax = axes[1]
    for r, jj in enumerate(range(j - 1, j + 3)):
        for c, ii in enumerate(range(i - 1, i + 3)):
            w = cubic[r, c]
            ax.plot(
                [tlon, lon[jj, ii]],
                [tlat, lat[jj, ii]],
                color="C3" if w >= 0 else "C0",
                lw=0.4 + 5 * abs(w),
                alpha=0.7,
                zorder=3,
            )
            ax.scatter(
                lon[jj, ii],
                lat[jj, ii],
                s=30 + 400 * abs(w),
                c="C3" if w >= 0 else "C0",
                edgecolor="k",
                linewidth=0.5,
                zorder=4,
            )
    ax.add_patch(Polygon([(lon[c], lat[c]) for c in corners], facecolor="C3", alpha=0.15, zorder=1))
    ax.scatter(tlon, tlat, s=170, c="C1", edgecolor="k", zorder=6)
    ax.text(
        0.97,
        0.97,
        f"sum of weights = {cubic.sum():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )

    # Conservative: an axis-aligned destination cell around the target
    ax = axes[2]
    half = 0.85
    dpoly = np.array(
        [
            [tlon - half, tlat - half],
            [tlon + half, tlat - half],
            [tlon + half, tlat + half],
            [tlon - half, tlat + half],
        ]
    )
    darea = spherical_area(dpoly[:, 1], dpoly[:, 0])
    overlaps = []
    for lo, la in zip(cblon.reshape(-1, 4), cblat.reshape(-1, 4)):
        clipped = clip_polygon(np.column_stack([lo, la]), dpoly)
        if len(clipped) >= 3:
            weight = spherical_area(clipped[:, 1], clipped[:, 0]) / darea
            if weight > 1e-3:
                overlaps.append((clipped, weight))
    for clipped, weight in overlaps:
        ax.add_patch(
            Polygon(clipped, facecolor="C3", alpha=0.15 + 0.7 * weight, edgecolor="none", zorder=1)
        )
        ax.text(
            *clipped.mean(axis=0), f"{weight:.0%}", ha="center", va="center", fontsize=8, zorder=5
        )
    ax.add_patch(Polygon(dpoly, facecolor="none", edgecolor="C1", linewidth=2.5, zorder=4))

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.17)
    fig.savefig(outfile, dpi=110)
    plt.close(fig)


def _cuboid(ax, x, y, width, depth, height, **kwargs):
    """Draw a parallelepiped in oblique projection with its front face at ``(x, y)``"""
    dx, dy = 0.5 * depth, 0.3 * depth
    faces = [
        ([(x, y), (x + width, y), (x + width, y + height), (x, y + height)], "#eef2f6"),
        (
            [
                (x, y + height),
                (x + dx, y + height + dy),
                (x + width + dx, y + height + dy),
                (x + width, y + height),
            ],
            "#d5dde6",
        ),
        (
            [
                (x + width, y),
                (x + width + dx, y + dy),
                (x + width + dx, y + height + dy),
                (x + width, y + height),
            ],
            "#bfcad6",
        ),
    ]
    for corners, color in faces:
        ax.add_patch(Polygon(corners, facecolor=color, edgecolor="#1f3a5f", lw=0.8, **kwargs))


def gen_chunks(outfile):
    """Figure that shows how to chunk an array whose working dimension is ``z``

    The working dimension is never split: a block is a whole water column for the tile that it
    covers, with one block per time step or per tile of a time step.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    titles = ['chunks={"time": 1}', 'chunks={"time": 1, "y": 500, "x": 500}']
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10, family="monospace")
        ax.set(xlim=(-0.7, 4.7), ylim=(-0.9, 2.2), aspect="equal")
        ax.axis("off")
        ax.annotate("", (-0.25, 1.2), (-0.25, 0), arrowprops=dict(arrowstyle="->"))
        ax.text(-0.35, 0.6, "z", ha="right", va="center", fontsize=9, family="monospace")

    # One block per time step
    ax = axes[0]
    for t in range(3):
        _cuboid(ax, 1.5 * t, 0, 1.0, 0.8, 1.2)
        ax.text(1.5 * t + 0.5, -0.3, f"time={t}", ha="center", fontsize=8, family="monospace")

    # Tiles in y and x for each time step, back row first
    ax = axes[1]
    for t in range(2):
        for j in (1, 0):
            for i in range(3):
                _cuboid(ax, 2.2 * t + 0.55 * i + 0.25 * j, 0.15 * j, 0.55, 0.5, 1.2)
        ax.text(2.2 * t + 0.9, -0.3, f"time={t}", ha="center", fontsize=8, family="monospace")
    ax.annotate("", (1.9, -0.55), (0.0, -0.55), arrowprops=dict(arrowstyle="->"))
    ax.text(0.95, -0.85, "x", ha="center", fontsize=9, family="monospace")
    ax.annotate("", (2.7, -0.35), (2.2, -0.65), arrowprops=dict(arrowstyle="->"))
    ax.text(2.8, -0.45, "y", fontsize=9, family="monospace")

    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)


def gen_dask_threads(outfile):
    """Figure that shows the dask threads, the blocks they process and the numba threads"""
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.set(xlim=(-1.9, 5.0), ylim=(-0.2, 2.8), aspect="equal")
    ax.axis("off")
    rows = {"dask threads": 2.3, "blocks": 1.2, "numba threads": 0.1}
    for label, y in rows.items():
        ax.text(-0.75, y + 0.2, label, ha="right", va="center", fontsize=10)

    for i in range(4):  # 2 blocks in progress, the others wait
        active = i < 2
        ax.add_patch(
            FancyBboxPatch(
                (1.4 * i - 0.45, 1.2),
                0.9,
                0.4,
                boxstyle="round,pad=0,rounding_size=0.06",
                facecolor="#cde9f5" if active else "#eeeeee",
                edgecolor="#1f3a5f",
                lw=0.8,
            )
        )
        if active:
            ax.add_patch(
                FancyBboxPatch(
                    (1.4 * i - 0.45, 2.3),
                    0.9,
                    0.4,
                    boxstyle="round,pad=0,rounding_size=0.06",
                    facecolor="#5ac2e7",
                    edgecolor="#1f3a5f",
                    lw=0.8,
                )
            )
            ax.text(1.4 * i, 2.5, "dask", ha="center", va="center", fontsize=10)
            for y0, y1 in ((2.3, 1.6), (1.2, 0.6)):
                ax.annotate("", (1.4 * i, y1), (1.4 * i, y0), arrowprops=dict(arrowstyle="->"))
            for k in range(4):
                ax.add_patch(
                    Polygon(
                        [(1.4 * i + 0.22 * (k - 2) + 0.02 + dx, 0.1 + dy) for dx, dy in
                         ((0, 0), (0.18, 0), (0.18, 0.3), (0, 0.3))],
                        facecolor="#cde9f5",
                        edgecolor="#1f3a5f",
                        lw=0.8,
                    )
                )
    ax.text(4.2, 0.25, "blocks waiting", ha="center", va="center", fontsize=9, style="italic")

    fig.tight_layout()
    fig.savefig(outfile, dpi=110)
    plt.close(fig)


def genfigs(app):
    """Generate the figures during the documentation compilation"""
    gendir = os.path.join(app.env.srcdir, "_static")
    gen_interp_vs_regrid(os.path.join(gendir, "interp-vs-regrid.png"))
    gen_method_stencils(os.path.join(gendir, "method-stencils.png"))
    gen_chunks(os.path.join(gendir, "chunks.png"))
    gen_dask_threads(os.path.join(gendir, "dask-threads.png"))


def setup(app):
    app.connect('builder-inited', genfigs)
    return {'version': '0.1'}


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    gen_interp_vs_regrid(os.path.join(outdir, "interp-vs-regrid.png"))
    gen_method_stencils(os.path.join(outdir, "method-stencils.png"))
    gen_chunks(os.path.join(outdir, "chunks.png"))
    gen_dask_threads(os.path.join(outdir, "dask-threads.png"))
