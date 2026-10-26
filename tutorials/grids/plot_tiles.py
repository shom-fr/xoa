#!/usr/bin/env python
"""
Merge tiles of a regular grid
=============================

In this tutorial, we show how to:

* make tile files that are subsets of a larger rectangular grid,
* plot the tiles on the grid they are a subset of, to spot overlaps and gaps,
* merge them with :class:`xoa.tiles.TileMerger` and :func:`xoa.tiles.merge_tiles`,
* choose how the overlaps are merged,
* give the grid instead of inferring it, and check the memory needed.

The grid must have constant steps in degrees.
See also: :ref:`indepth.grids`.
"""

# %%
# Initialisations
# ---------------

import glob
import os
import tempfile
import warnings

import matplotlib.pyplot as plt

import xoa
from xoa.plot import plot_field
from xoa.tiles import TileMerger, merge_tiles

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})

# %%
# Make the tiles
# --------------
# We start from a GLORYS12 run of the Gulf Stream, with a single time step, and write
# three files: a reference tile, a tile at its north and a tile at its west.
# The northern tile overlaps the reference one by four rows, whereas the western tile
# just touches it, with no overlap. To see how overlaps are merged, the northern tile
# is given a bias of 1 °C. The north-west corner is not covered by any tile.

ds = xoa.open_data_sample("MODELS/CMEMS-GLORYS/glorys_gulfstream_3d.nc")
ds = ds[["thetao", "uo", "vo"]].isel(time=slice(0, 1), depth=slice(0, 3))
print(dict(ds.sizes))

tmpdir = tempfile.TemporaryDirectory()
parts = {
    "ref": dict(longitude=slice(20, 60), latitude=slice(0, 30)),
    "north": dict(longitude=slice(20, 60), latitude=slice(26, 48)),
    "west": dict(longitude=slice(0, 20), latitude=slice(0, 30)),
}
for i, (name, sel) in enumerate(parts.items()):
    tile = ds.isel(sel).copy(deep=True)
    if name == "north":
        tile["thetao"] = tile.thetao + 1
    tile.to_netcdf(os.path.join(tmpdir.name, f"tile{i + 1}-{name}.nc"))
paths = sorted(glob.glob(os.path.join(tmpdir.name, "tile*.nc")))
[os.path.basename(path) for path in paths]

# %%
# Describe the grid
# -----------------
# Tiles are added as datasets, as paths or as glob patterns. Files are opened lazily and
# only their coordinates are read at this stage.
# The grid steps, origin and size are inferred from the coordinates of the tiles.
# The representation shows them, the position of each tile as index slices and the
# estimated memory of the merge. The merged longitudes are put in ``lon_range``, which would
# otherwise be inferred from the first tile.

tm = TileMerger(lon_range=(-180, 180))
tm.add(os.path.join(tmpdir.name, "tile*.nc"))
print(tm)

# %%
# The plot shows the extent of the grid and the tiles on it. The overlaps are hatched
# in red and the gap, which will be filled with nans, in grey.

tm.plot()

# %%
# Merge the tiles
# ---------------
# The :meth:`~xoa.tiles.TileMerger.merge` method puts each tile at its place.
# Where tiles overlap, the last valid value is kept by default. The
# :func:`~xoa.tiles.merge_tiles` function does it all in a single call, from files,
# a glob pattern or datasets. Variables are transposed with longitude and latitude last.

merged = merge_tiles(os.path.join(tmpdir.name, "tile*.nc"))
merged

# %%
# The sea surface temperature shows the missing corner as nans, and the bias of the
# northern tile.

plot_field(merged.thetao.isel(time=0, depth=0), title="Merged tiles")

# %%
# Overlaps
# --------
# The ``overlap`` argument is ``"last"`` or ``"first"`` to keep the value of the last or first
# tile, ``"mean"`` to average them or ``"raise"`` to fail when they differ.
# The nans of a tile never erase the values of another one.
# Here is a meridional section across the overlap, in the northern tile
# and the reference tile, which differ by 1 °C.

fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
for overlap in "first", "mean", "last":
    out = merge_tiles(paths, overlap=overlap)
    out.thetao.isel(time=0, depth=0, longitude=40).plot(ax=ax, label=overlap, marker=".")
lat = merged.latitude
ax.axvspan(float(lat[26]), float(lat[29]), color="r", alpha=0.15, lw=0)
ax.set_xlim(float(lat[18]), float(lat[36]))
ax.set_ylim(21, 28)
ax.legend(title="overlap")
ax.set_title("Meridional section at the overlap")

# %%
# With ``"raise"``, we get an error that tells which tile differs.

try:
    merge_tiles(paths, overlap="raise")
except xoa.exceptions.XoaError as exc:
    print(exc)

# %%
# Numerous tiles
# --------------
# With many tiles, filling each one with its own color gets messy. The tiles still get
# different colors from their neighbours, but a single ``color`` is better: the tiles are
# then only framed, and their names are written when they fit. Here, a 6 by 6 tiles split
# of the same grid, with two points of overlap and three tiles missing.

many = []
missing = {(1, 4), (2, 4), (5, 0)}
for j in range(6):
    for i in range(6):
        if (i, j) not in missing:
            tile = ds.isel(longitude=slice(10 * i, 10 * i + 12), latitude=slice(8 * j, 8 * j + 10))
            tile.attrs = {"name": f"{j}{i}"}
            many.append(tile)
tm_many = TileMerger(lon_range=(-180, 180))
tm_many.add(many)
print(tm_many)
tm_many.plot()

# %%
tm_many.plot(color="C0")

# %%
# Give the grid
# -------------
# The grid may be given with the steps, the origin and the size, instead of being inferred.
# This is useful to merge the tiles on a larger domain, and to check that the tiles fit
# a known grid. A tile that is not on the grid, within ``tol`` times the step, raises an error.
# Here, we add a margin of 5 points all around.

grid = tm.grid
tm2 = TileMerger(
    dx=grid["dx"],
    dy=grid["dy"],
    lon0=grid["lon0"] - 5 * grid["dx"],
    lat0=grid["lat0"] - 5 * grid["dy"],
    nx=grid["nx"] + 10,
    ny=grid["ny"] + 10,
)
tm2.add(paths)
tm2.plot()

# %%
# Memory
# ------
# The merge is eager, so that the whole grid is loaded in memory. The estimate is
# given by :attr:`~xoa.tiles.TileMerger.nbytes`, and a warning is issued by
# :meth:`~xoa.tiles.TileMerger.merge` when it exceeds ``max_memory`` which defaults to half
# the physical memory. Here is what happens with a limit of 100 kB.

print(tm2.memory)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    TileMerger(max_memory=100e3).add(paths).merge()
print(caught[0].message)

# %%
# Close the files and remove the temporary folder.

tm.close()
tm2.close()
tmpdir.cleanup()
