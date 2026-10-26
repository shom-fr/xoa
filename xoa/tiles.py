"""
Merge tiles of a regular longitude/latitude grid
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

import glob
import os

import numpy as np
import xarray as xr

from . import coords as xcoords
from . import exceptions
from . import grid as xgrid
from .core import plot as cplot
from .core import tiles as ctiles

_SINGLE_TYPES_ = (xr.Dataset, xr.DataArray, str, os.PathLike)


def _format_bytes_(nbytes):
    """Format a number of bytes with a binary unit"""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if nbytes < 1024:
            return f"{nbytes:.3g} {unit}"
        nbytes /= 1024
    return f"{nbytes:.3g} TiB"


def _get_out_dtype_(var):
    """Data type of a merged variable: floats are kept and the others are converted"""
    return var.dtype if var.dtype.kind == "f" else np.dtype("d")


def _get_default_max_memory_():
    """Half of the physical memory, or 4 GiB when unknown"""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") // 2
    except (ValueError, OSError, AttributeError):
        return 4 * 1024**3


class _Tile_:
    """A tile with its axes, which are 1D and increasing"""

    def __init__(self, obj, index):
        self.opened = isinstance(obj, (str, os.PathLike))
        name = None
        if self.opened:
            name = os.path.splitext(os.path.basename(obj))[0]
            obj = xr.open_dataset(obj)
        elif isinstance(obj, xr.DataArray):
            obj = obj.to_dataset(name=obj.name or "data")
        self.name = name or obj.attrs.get("name") or str(index)

        obj = xgrid.to_rect(obj, errors="raise")
        lon = xcoords.get_lon(obj)
        lat = xcoords.get_lat(obj)
        if lon.ndim != 1 or lat.ndim != 1:
            raise exceptions.XoaError(f"Tile {self.name!r} must have 1D longitudes and latitudes")
        self.xdim, self.ydim = lon.dims[0], lat.dims[0]
        raw = lon.values.astype("d")
        self.lon_min, self.lon_max = np.nanmin(raw), np.nanmax(raw)
        # Longitudes are continuous across the dateline and the 0/360 seam
        values = {"lon": np.unwrap(raw, period=360), "lat": lat.values.astype("d")}
        self.axes = {}
        for axis, dim, coord in (("lon", self.xdim, lon), ("lat", self.ydim, lat)):
            diff = np.diff(values[axis])
            if (diff < 0).all() and diff.size:
                obj = obj.isel({dim: slice(None, None, -1)})
                values[axis] = values[axis][::-1]
            elif not (diff > 0).all():
                raise exceptions.XoaError(
                    f"The coordinate {coord.name!r} of the tile {self.name!r} is not monotonic"
                )
            self.axes[axis] = values[axis]
        self.ds = obj
        self.lon = xcoords.get_lon(obj)
        self.lat = xcoords.get_lat(obj)


class TileMerger:
    """Merge tiles of a regular longitude/latitude grid

    The tiles are subsets of a larger rectangular grid with constant steps.
    They may overlap, and the cells that are not covered by any tile are filled with nans.
    The grid is given by its steps, origin and size, and what is missing is inferred
    from the tiles: the steps from their coordinates, the origin from the lowest
    coordinates and the size from the extent of the tiles.

    The tiles must have the same variables and the same dimensions apart from
    longitude and latitude, whose coordinates may be 1D or 2D if the grid is rectilinear.
    Coordinates are sorted in increasing order.

    Longitudes are processed modulo 360 degrees, so tiles may use different conventions
    or cross the dateline or the Greenwich meridian. The grid is built on the window
    of 360 degrees with the smallest extent, or starting at `lon0`, and the merged dataset
    is put in `lon_range`.

    Parameters
    ----------
    dx: None, float
        Longitude step in degrees
    dy: None, float
        Latitude step in degrees
    lon0: None, float
        Longitude of the first column of the grid
    lat0: None, float
        Latitude of the first row of the grid
    nx: None, int
        Number of columns
    ny: None, int
        Number of rows
    overlap: str
        How to merge the cells covered by several tiles: ``"last"`` or ``"first"`` to keep
        the value of the last or first tile that is valid, ``"mean"`` to average them
        or ``"raise"`` to fail when they differ.
    tol: float
        Maximal distance, in fraction of the step, between the coordinates of a tile
        and the grid
    max_memory: None, int
        Estimated memory in bytes, see :attr:`nbytes`, above which :meth:`merge` issues
        a warning. It defaults to half of the physical memory.
    lon_range: None, tuple
        Range of the longitudes of the merged dataset, like ``(-180, 180)`` or ``(0, 360)``,
        whose size must be 360. It is inferred from the first tile: ``(0, 360)`` if its
        longitudes exceed 180, and ``(-180, 180)`` otherwise.
    tiles: None, list
        Initial tiles, see :meth:`add`

    Examples
    --------
    >>> tm = TileMerger()
    >>> tm.add(["tile1.nc", "tile2.nc"])  # doctest: +SKIP
    >>> tm.plot()  # doctest: +SKIP
    >>> ds = tm.merge()  # doctest: +SKIP
    """

    def __init__(
        self, dx=None, dy=None, lon0=None, lat0=None, nx=None, ny=None, overlap="last", tol=0.1,
        max_memory=None, lon_range=None, tiles=None,
    ):  # fmt: skip
        if overlap not in ctiles.OVERLAPS:
            raise exceptions.XoaError(f"Invalid overlap {overlap!r}: {list(ctiles.OVERLAPS)}")
        self.overlap = overlap
        self.tol = tol
        self.max_memory = max_memory or _get_default_max_memory_()
        if lon_range is not None:
            lon_range = tuple(float(lon) for lon in lon_range)
            if len(lon_range) != 2 or abs(lon_range[1] - lon_range[0] - 360) > 1e-6:
                raise exceptions.XoaError("lon_range must have a size of 360 degrees")
        self._lon_range = lon_range
        self._spec = {"dx": dx, "dy": dy, "lon0": lon0, "lat0": lat0, "nx": nx, "ny": ny}
        self._tiles = []
        self._grid = None
        if tiles is not None:
            self.add(tiles)

    def add(self, tiles):
        """Add tiles

        Parameters
        ----------
        tiles: xarray.Dataset, xarray.DataArray, str, list
            A tile, the path to a netcdf file or a glob pattern, or a list of them.
            Files are opened lazily and closed by :meth:`close`.

        Return
        ------
        TileMerger
        """
        if isinstance(tiles, _SINGLE_TYPES_):
            tiles = [tiles]
        for tile in tiles:
            paths = sorted(glob.glob(str(tile))) if isinstance(tile, (str, os.PathLike)) else None
            for item in paths if paths is not None else [tile]:
                self._tiles.append(_Tile_(item, len(self._tiles)))
        self._grid = None
        return self

    def close(self):
        """Close the files opened by the merger"""
        for tile in self._tiles:
            if tile.opened:
                tile.ds.close()

    @property
    def ntiles(self):
        """Number of tiles"""
        return len(self._tiles)

    @property
    def lon_range(self):
        """Range of the longitudes of the merged dataset"""
        if self._lon_range is not None:
            return self._lon_range
        if self._tiles and self._tiles[0].lon_max > 180:
            return (0.0, 360.0)
        return (-180.0, 180.0)

    def _wrap_tiles_(self):
        """Shift the longitudes of the tiles into the window of the grid"""
        spec = self._spec
        axes = [tile.axes["lon"] for tile in self._tiles]
        if spec["lon0"] is not None:
            start = spec["lon0"] - (0.5 * spec["dx"] if spec["dx"] else 1e-3)
        else:
            start = ctiles.get_lon_window_start(axes, self.lon_range[0])
        for tile in self._tiles:
            tile.lonv = ctiles.wrap_lon(tile.axes["lon"][0], start) + (
                tile.axes["lon"] - tile.axes["lon"][0]
            )

    @property
    def grid(self):
        """Specifications of the grid, as a dict with ``dx``, ``dy``, ``lon0``, ``lat0``,
        ``nx`` and ``ny``"""
        if self._grid is None:
            if not self._tiles and None in self._spec.values():
                raise exceptions.XoaError("Add tiles to infer the grid")
            spec = self._spec
            grid = {}
            if self._tiles:
                self._wrap_tiles_()
            for xy, axis in (("x", "lon"), ("y", "lat")):
                axes = [tile.lonv if axis == "lon" else tile.axes[axis] for tile in self._tiles]
                step, origin, size = spec["d" + xy], spec[axis + "0"], spec["n" + xy]
                if axes:
                    step, origin, size = ctiles.fit_regular_axis(axes, step, origin, size, axis)
                grid.update({"d" + xy: step, axis + "0": origin, "n" + xy: size})
            self._grid = {key: grid[key] for key in self._spec}
            self._index_tiles_()
        return self._grid

    def _index_tiles_(self):
        """Compute the position of the tiles in the grid"""
        grid = self._grid
        for tile in self._tiles:
            tile.index = {}
            for axis, xy in (("lon", "x"), ("lat", "y")):
                coord = tile.lonv if axis == "lon" else tile.axes[axis]
                ind, err = ctiles.get_grid_indices(coord, grid[axis + "0"], grid["d" + xy])
                if err > self.tol or (np.diff(ind) != 1).any() or ind[0] < 0:
                    raise exceptions.XoaError(
                        f"The {axis} axis of the tile {tile.name!r} is not on the grid"
                    )
                if ind[-1] >= grid["n" + xy]:
                    raise exceptions.XoaError(f"The tile {tile.name!r} exceeds the grid")
                tile.index[xy] = int(ind[0]), int(ind[-1]) + 1

    @property
    def nbytes(self):
        """Estimated memory in bytes of the merged variables, which is the minimal need of
        :meth:`merge`, that also loads one tile variable at a time"""
        if not self._tiles:
            return 0
        grid = self.grid
        first = self._tiles[0]
        nbytes = 0
        for var in first.ds.data_vars.values():
            if first.xdim in var.dims and first.ydim in var.dims:
                size = var.size // (var.sizes[first.xdim] * var.sizes[first.ydim])
                size *= grid["nx"] * grid["ny"]
                nbytes += size * _get_out_dtype_(var).itemsize
                if self.overlap == "mean":
                    nbytes += size * np.dtype("i").itemsize
        return nbytes

    @property
    def memory(self):
        """Estimated memory of :meth:`merge` as a string, like ``"1.5 GiB"``"""
        return _format_bytes_(self.nbytes)

    def __repr__(self):
        grid = self.grid
        return (
            f"<TileMerger: {self.ntiles} tiles, overlap={self.overlap!r}, "
            f"grid {grid['nx']}x{grid['ny']} (dx={grid['dx']:.4g}, dy={grid['dy']:.4g}), "
            f"lon0={grid['lon0']:.6g}, lat0={grid['lat0']:.6g}, memory~{self.memory}>"
        )

    def __str__(self):
        lines = [repr(self)]
        for tile in self._tiles:
            (ix0, ix1), (iy0, iy1) = tile.index["x"], tile.index["y"]
            lines.append(f"  {tile.name}: x[{ix0}:{ix1}] y[{iy0}:{iy1}]")
        return "\n".join(lines)

    @property
    def lon(self):
        """Longitudes of the grid, continuous and so possibly out of :attr:`lon_range`"""
        grid = self.grid
        return grid["lon0"] + grid["dx"] * np.arange(grid["nx"])

    @property
    def lat(self):
        """Latitudes of the grid"""
        grid = self.grid
        return grid["lat0"] + grid["dy"] * np.arange(grid["ny"])

    @property
    def extent(self):
        """Extent of the grid cells ``[xmin, xmax, ymin, ymax]``"""
        grid = self.grid
        x0 = grid["lon0"] - grid["dx"] / 2
        y0 = grid["lat0"] - grid["dy"] / 2
        return [x0, x0 + grid["nx"] * grid["dx"], y0, y0 + grid["ny"] * grid["dy"]]

    def merge(self, tiles=None):
        """Merge the tiles

        The variables with the longitude and latitude dimensions are merged and transposed
        with these dimensions last. Integers are converted to floats.
        The other variables are taken from the first tile.

        Parameters
        ----------
        tiles: None, list
            Tiles to add before, see :meth:`add`

        Return
        ------
        xarray.Dataset
        """
        if tiles is not None:
            self.add(tiles)
        if not self._tiles:
            raise exceptions.XoaError("No tile to merge")
        grid = self.grid
        if self.nbytes > self.max_memory:
            exceptions.xoa_warn(
                f"The merged dataset needs about {self.memory}, which is more than "
                f"the {_format_bytes_(self.max_memory)} allowed: "
                "reduce the grid, drop variables or increase max_memory"
            )
        first = self._tiles[0]
        xdim, ydim = first.xdim, first.ydim
        mode = ctiles.OVERLAPS[self.overlap]

        out = xr.Dataset(attrs=first.ds.attrs)
        for name, coord in first.ds.coords.items():
            if xdim not in coord.dims and ydim not in coord.dims:
                out.coords[name] = coord
        out = out.assign_coords(
            {
                first.lon.name: xr.DataArray(
                    self.lon, dims=xdim, attrs=first.lon.attrs, name=first.lon.name
                ),
                first.lat.name: xr.DataArray(
                    self.lat, dims=ydim, attrs=first.lat.attrs, name=first.lat.name
                ),
            }
        )
        for name, var in first.ds.data_vars.items():
            if xdim not in var.dims or ydim not in var.dims:
                if xdim not in var.dims and ydim not in var.dims:
                    out[name] = var
                continue
            dims = [dim for dim in var.dims if dim not in (xdim, ydim)]
            shape = tuple(var.sizes[dim] for dim in dims) + (grid["ny"], grid["nx"])
            dtype = _get_out_dtype_(var)
            data = np.full(shape, np.nan, dtype)
            data3d = data.reshape(-1, grid["ny"], grid["nx"])
            count = np.zeros(data3d.shape, "i") if mode == 2 else np.zeros((1, 1, 1), "i")
            for tile in self._tiles:
                if name not in tile.ds or {xdim, ydim} - set(tile.ds[name].dims):
                    continue
                arr = tile.ds[name].transpose(*dims, ydim, xdim).values
                if arr.shape[:-2] != shape[:-2]:
                    raise exceptions.XoaError(
                        f"Variable {name!r} of tile {tile.name!r} has an incompatible shape"
                    )
                nconf = ctiles.paste(
                    data3d,
                    arr.reshape(-1, *arr.shape[-2:]).astype(dtype, copy=False),
                    tile.index["y"][0],
                    tile.index["x"][0],
                    mode,
                    count,
                )
                if nconf:
                    raise exceptions.XoaError(
                        f"Tile {tile.name!r} differs from the previous ones for {nconf} "
                        f"values of {name!r} in the overlaps"
                    )
            if mode == 2:
                ctiles.finalize_mean(data3d, count)
            out[name] = xr.DataArray(data, dims=dims + [ydim, xdim], attrs=var.attrs)
        # Put the longitudes in the range, which may reorder the columns
        lon, lo = self.lon, self.lon_range[0]
        norm = np.where((lon < lo) | (lon >= lo + 360), lo + (lon - lo) % 360, lon)
        if (norm != lon).any():
            order = np.argsort(norm, kind="stable")
            out = out.isel({xdim: order}).assign_coords(
                {first.lon.name: (xdim, norm[order], first.lon.attrs)}
            )
        return out

    def get_coverage(self):
        """Number of tiles that cover the cells of the grid

        The grid is reduced to the cells delimited by the edges of the tiles, which
        keeps it small whatever the resolution.

        Return
        ------
        numpy.ndarray(nxe-1)
            Longitudes of the cell edges
        numpy.ndarray(nye-1)
            Latitudes of the cell edges
        numpy.ndarray(nye-1, nxe-1)
            Number of tiles
        """
        grid = self.grid
        edges = {}
        for xy, axis in (("x", "lon"), ("y", "lat")):
            ind = {0, grid["n" + xy]}
            for tile in self._tiles:
                ind.update(tile.index[xy])
            edges[xy] = np.array(sorted(ind))
        counts = np.zeros((edges["y"].size - 1, edges["x"].size - 1), "i")
        for tile in self._tiles:
            (ix0, ix1), (iy0, iy1) = tile.index["x"], tile.index["y"]
            ix = np.searchsorted(edges["x"], [ix0, ix1])
            iy = np.searchsorted(edges["y"], [iy0, iy1])
            counts[iy[0] : iy[1], ix[0] : ix[1]] += 1
        lon = grid["lon0"] + grid["dx"] * (edges["x"] - 0.5)
        lat = grid["lat0"] + grid["dy"] * (edges["y"] - 0.5)
        return lon, lat, counts

    def _get_tile_colors_(self):
        """Color indices that differ between touching or overlapping tiles"""
        self.grid  # positions the tiles
        ind = np.array([[*t.index["x"], *t.index["y"]] for t in self._tiles])
        colors = np.zeros(len(ind), "i")
        for i in range(1, len(ind)):
            near = (
                (ind[:i, 0] <= ind[i, 1])
                & (ind[i, 0] <= ind[:i, 1])
                & (ind[:i, 2] <= ind[i, 3])
                & (ind[i, 2] <= ind[:i, 3])
            )
            used = set(colors[:i][near])
            colors[i] = next(c for c in range(len(ind)) if c not in used)
        return colors

    def plot(self, ax=None, color=None, labels=True, cells=None, pad=0.15, **kwargs):
        """Plot the grid and the tiles on a map

        The extent of the grid is framed, the gaps are hatched in grey, the overlaps are
        hatched in red, and the tiles are filled and named. Touching tiles have different colors.

        Parameters
        ----------
        ax: None, matplotlib.axes.Axes
            Axes to draw on, which can be a cartopy one. A map is created when not provided.
        color: None, str
            A single color for all the tiles, which are then only framed, as it is better
            with numerous tiles. By default, tiles have their own color and are filled.
        labels: bool
            Name the tiles that are large enough to hold their name
        cells: None, bool
            Draw the cell edges, which defaults to small grids
        pad: float
            Margin around the grid, in fraction of its size, for the extent of the new map
        kwargs:
            Extra parameters are passed to :func:`xoa.core.plot.create_map`

        Return
        ------
        matplotlib.axes.Axes
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection, PatchCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch, Rectangle

        grid = self.grid
        x0, x1, y0, y1 = self.extent
        lon, lat, counts = self.get_coverage()
        if ax is None:
            dx, dy = pad * (x1 - x0), pad * (y1 - y0)
            title = (
                f"{self.ntiles} tiles on a {grid['nx']}×{grid['ny']} grid "
                f"({grid['dx']:.4g}°×{grid['dy']:.4g}°)"
            )
            kwargs.setdefault("title", title)
            if "projection" not in kwargs:  # centered, for grids that cross the dateline
                ccrs, _ = cplot.import_cartopy()
                kwargs["projection"] = ccrs.Mercator(central_longitude=0.5 * (x0 + x1))
            _, ax = cplot.create_map([x0 - dx, x1 + dx, y0 - dy, y1 + dy], **kwargs)
        kw = {}
        to_display = ax.transData
        if hasattr(ax, "projection"):
            ccrs, _ = cplot.import_cartopy()
            kw["transform"] = ccrs.PlateCarree()
            to_display = kw["transform"]._as_mpl_transform(ax)

        def rects(mask):
            """Rectangles of the cells of a mask, merged along x"""
            out = []
            for j, row in enumerate(mask):
                edges = np.flatnonzero(np.diff(np.r_[False, row, False].astype("i")))
                for i0, i1 in zip(edges[::2], edges[1::2]):
                    out.append(Rectangle((lon[i0], lat[j]), lon[i1] - lon[i0], lat[j + 1] - lat[j]))
            return out

        # Gaps, below everything
        handles = [Line2D([], [], color="k", lw=2, label="Grid")]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec="k", lw=2, zorder=5, **kw))
        if (counts == 0).any():
            ax.add_collection(
                PatchCollection(
                    rects(counts == 0), fc="0.93", ec="0.6", hatch="///", lw=0, zorder=1, **kw
                )
            )
            handles.append(Patch(fc="0.93", ec="0.6", hatch="///", label="Gap"))

        # Tiles
        palette = plt.get_cmap("tab10")
        indices = None if color else self._get_tile_colors_()
        dpi = ax.figure.dpi
        for i, tile in enumerate(self._tiles):
            (ix0, ix1), (iy0, iy1) = tile.index["x"], tile.index["y"]
            xt0 = grid["lon0"] + grid["dx"] * (ix0 - 0.5)
            yt0 = grid["lat0"] + grid["dy"] * (iy0 - 0.5)
            wt, ht = grid["dx"] * (ix1 - ix0), grid["dy"] * (iy1 - iy0)
            tcolor = color or palette(indices[i] % 10)
            if color:
                ax.add_patch(
                    Rectangle((xt0, yt0), wt, ht, fc="none", ec=color, lw=1, zorder=4, **kw)
                )
            else:
                ax.add_patch(
                    Rectangle((xt0, yt0), wt, ht, fc=tcolor, alpha=0.35, ec="none", zorder=2, **kw)
                )
                ax.add_patch(
                    Rectangle((xt0, yt0), wt, ht, fc="none", ec=tcolor, lw=1.5, zorder=4, **kw)
                )
                if self.ntiles <= 12:
                    handles.append(Patch(fc=tcolor, alpha=0.5, ec=tcolor, label=tile.name))
            if labels:
                # Only if the name fits in the tile, with a rough estimate of its size
                (px0, py0), (px1, py1) = to_display.transform([(xt0, yt0), (xt0 + wt, yt0 + ht)])
                size = 8 * dpi / 72
                if abs(px1 - px0) > 0.7 * size * len(tile.name) + 6 and abs(py1 - py0) > 2 * size:
                    ax.text(
                        xt0 + wt / 2,
                        yt0 + ht / 2,
                        tile.name,
                        ha="center",
                        va="center",
                        fontsize=8,
                        fontweight="bold",
                        color=tcolor,
                        bbox={"fc": "w", "ec": "none", "alpha": 0.8, "pad": 1.5},
                        zorder=6,
                        clip_on=True,
                        **kw,
                    )
        if color:
            handles.append(Patch(fc="none", ec=color, label="Tiles"))

        # Overlaps
        if (counts > 1).any():
            ax.add_collection(
                PatchCollection(
                    rects(counts > 1), fc="none", ec="r", hatch="xxx", lw=0, zorder=3, **kw
                )
            )
            handles.append(Patch(fc="none", ec="r", hatch="xxx", label="Overlap"))

        # Cells
        if cells is None:
            cells = grid["nx"] <= 100 and grid["ny"] <= 100
        if cells:
            xe = grid["lon0"] + grid["dx"] * (np.arange(grid["nx"] + 1) - 0.5)
            ye = grid["lat0"] + grid["dy"] * (np.arange(grid["ny"] + 1) - 0.5)
            lines = [[(x, y0), (x, y1)] for x in xe] + [[(x0, y), (x1, y)] for y in ye]
            ax.add_collection(LineCollection(lines, colors="0.4", lw=0.3, zorder=3, **kw))

        ax.legend(
            handles=handles,
            loc="upper left",
            fontsize=8,
            framealpha=0.9,
            title=f"Memory ~{self.memory}",
            title_fontsize=8,
        )
        return ax


def merge_tiles(tiles, plot=False, **kwargs):
    """Merge tiles of a regular longitude/latitude grid in a single call

    Parameters
    ----------
    tiles: xarray.Dataset, xarray.DataArray, str, list
        A tile, the path to a netcdf file or a glob pattern, or a list of them.
    plot: bool
        Plot the grid and the tiles with :meth:`TileMerger.plot`
    kwargs:
        Extra parameters are passed to :class:`TileMerger`

    Return
    ------
    xarray.Dataset

    See also
    --------
    TileMerger
    """
    tm = TileMerger(tiles=tiles, **kwargs)
    try:
        if plot:
            tm.plot()
        return tm.merge()
    finally:
        tm.close()
