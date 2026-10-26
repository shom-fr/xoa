"""
Statistics of xarray objects

See :func:`rms` for root mean square errors and :class:`StatAccum` for statistics
that are accumulated in time.
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
import json
from typing import ClassVar

import numpy as np
import xarray as xr

from . import coords as xcoords
from . import exceptions
from . import grid as xgrid
from .core import stats as cstats

#: Version of the file format of :meth:`StatAccum.dump`
FORMAT_VERSION = 1


def rms(da, ref=None, mask=None, dim=None, by=None):
    """Compute a root mean square, or a root mean square error

    Parameters
    ----------
    da: xarray.DataArray
        Estimate, or the error when `ref` is not provided.
    ref: None, xarray.DataArray
        Reference, which is subtracted from `da`.
    mask: None, xarray.DataArray
        Boolean array that tells where to compute the statistics.
        Missing values are always discarded.
    dim: None, str, list(str)
        Dimensions to reduce, which default to all of them.
    by: None, dict
        Regions as a dict that maps names to boolean arrays. The root mean square is then
        computed in each of them, in addition to `mask`, and the result has
        an extra ``region`` dimension.

    Return
    ------
    xarray.DataArray

    Example
    -------
    .. code-block:: python

        rms(rossby_z, ref=rossby_ref, mask=valid)
        rms(speed_error, by={"Everywhere": valid, "Cores": valid & core})
    """
    sq = (da if ref is None else da - ref) ** 2
    if mask is not None:
        sq = sq.where(mask)
    if by is None:
        return np.sqrt(sq.mean(dim)).rename("rms")
    if not len(by):
        raise exceptions.XoaError("by must not be empty")
    res = [np.sqrt(sq.where(region).mean(dim)) for region in by.values()]
    region = xr.DataArray(list(by), dims="region", name="region")
    return xr.concat(res, dim=region).rename("rms")


class StatAccum:
    """Statistics accumulator

    It accumulates single or comparative statistics in time, as additive moments
    (weighted sums, squares, products, minima and maxima) that are updated with the
    :meth:`append` method. Statistics are computed on demand with :meth:`get_stats`.
    The accumulation can be interrupted with :meth:`dump` and restarted with :meth:`from_file`.

    Two kinds of statistics are available, and their names are prefixed with
    ``"t"`` and ``"s"``:

    - **Temporal**: they are accumulated along time at each point of the other dimensions,
      without weights.
    - **Spatial**: they are computed over `sdims` at each time step, with weights,
      and they are cumulated as a time series. Other dimensions are kept.

    Parameters
    ----------
    temporal: bool
        Accumulate temporal statistics.
    spatial: bool
        Accumulate spatial statistics.
    sdims: None, str, list(str)
        Dimensions to reduce for spatial statistics, which default to all
        the dimensions but time. For vertical profiles, use for instance ``"depth"``.
    weights: None, False, xarray.DataArray
        Weights of spatial statistics.
        By default, they are automatic: the layer thicknesses from :func:`xoa.grid.get_dz`
        if the vertical dimension is reduced, times the grid cell sizes from
        :func:`xoa.grid.get_dx_dy` along the horizontal dimensions that are reduced.
        An error is raised if the thicknesses cannot be found, and the horizontal
        dimensions are not weighted, with a warning, if there are no longitudes and latitudes.
        Use ``False`` to disable the weights, or an array to provide your own,
        which must be broadcastable to the data and, if it has a time dimension,
        its coordinates must include the appended times.
        Samples with a missing or non-positive weight are ignored.
    on_overlap: {"raise", "skip"}
        What to do when appended times are not after the last accumulated time,
        which happens when a restarted run replays steps.
        They are either refused or ignored.

    Example
    -------
    .. code-block:: python

        sa = StatAccum(spatial=True, sdims=("depth", "x"), weights=dz)
        for ds in datasets:
            sa.append(ds.temp, ds.temp_ref)
        sa.dump("stats.nc")
        stats = StatAccum.from_file("stats.nc", weights=dz).append(more).get_stats()
    """

    #: Available statistics, which are prefixed with ``"t"`` or ``"s"``.
    #: Per-variable ones are suffixed with ``"0"`` and ``"1"`` for comparisons.
    all_stats = (
        "count",
        "avail",
        "mean",
        "var",
        "std",
        "min",
        "max",
        "bias",
        "rms",
        "crms",
        "cov",
        "corr",
    )

    def __init__(self, temporal=True, spatial=False, sdims=None, weights=None, on_overlap="raise"):
        if on_overlap not in ("raise", "skip"):
            raise exceptions.XoaStatsError("on_overlap must be 'raise' or 'skip'")
        if isinstance(sdims, str):
            sdims = [sdims]
        self.temporal = temporal
        self.spatial = spatial
        self.sdims = None if sdims is None else list(sdims)
        self.weights = weights
        self.on_overlap = on_overlap
        self.tdim = None
        self.dual = None
        self.nt = 0
        self.ns = None
        self.last = None
        self.names = [None, None]
        self.attrs = [{}, {}]
        self._tmom = None
        self._smom = None

    def _init_(self, da0, da1):
        self.tdim = xcoords.get_tdim(da0, errors="raise")
        self.dual = da1 is not None
        sdims = [dim for dim in da0.dims if dim != self.tdim]
        if self.sdims is None:
            self.sdims = sdims
        elif missing := [dim for dim in self.sdims if dim not in sdims]:
            raise exceptions.XoaStatsError(f"Spatial dimensions not found in the data: {missing}")
        for i, da in enumerate((da0, da1)):
            if da is not None:
                self.names[i] = da.name
                self.attrs[i] = dict(da.attrs)
        self.spatial &= bool(self.sdims)
        self.ns = int(np.prod([da0.sizes[dim] for dim in self.sdims]))

    def append(self, da0, da1=None):
        """Append data to the accumulator

        Parameters
        ----------
        da0: xarray.DataArray
            First array, which must have a time dimension.
        da1: None, xarray.DataArray
            Second array, with the same dimensions.
            If provided, comparative statistics are accumulated.
            Only the points that are valid in both arrays are used.

        Return
        ------
        StatAccum
            The accumulator itself
        """
        if self.tdim is None:
            self._init_(da0, da1)
        elif (da1 is not None) != self.dual:
            raise exceptions.XoaStatsError("Cannot mix single and comparative accumulations")
        tdim = self.tdim
        if tdim not in da0.dims:
            raise exceptions.XoaStatsError(f"The time dimension '{tdim}' is missing")
        da0 = da0.transpose(tdim, ...)
        if self.dual:
            da1 = da1.transpose(*da0.dims)
            if da1.shape != da0.shape:
                raise exceptions.XoaStatsError("The two arrays must have the same shape")

        # Do not accumulate twice the same times
        if self.last is not None and tdim in da0.coords:
            keep = (da0[tdim] > self.last).values
            if not keep.all():
                if self.on_overlap == "raise":
                    raise exceptions.XoaStatsError(
                        f"Times to append must be after the last accumulated one: {self.last}"
                    )
                da0 = da0.isel({tdim: keep})
                if self.dual:
                    da1 = da1.isel({tdim: keep})
                if not keep.any():
                    return self

        nt = da0.sizes[tdim]
        if self.temporal:
            self._append_temporal_(da0, da1)
        if self.spatial:
            self._append_spatial_(da0, da1)
        self.nt += nt
        if tdim in da0.coords:
            self.last = da0[tdim].values[-1]
        return self

    def _as_moments_(self, mom, dims, template):
        coords = dict(template.coords)
        coords["moment"] = list(cstats.MOMENT_NAMES)
        return xr.DataArray(mom, dims=("moment",) + dims, coords=coords)

    def _append_temporal_(self, da0, da1):
        shape = da0.shape[1:]
        x = da0.values.reshape(da0.shape[0], -1).T
        y = da1.values.reshape(da1.shape[0], -1).T if self.dual else None
        mom = cstats.get_moments(x, y).reshape((cstats.NMOM,) + shape)
        if self._tmom is None:
            template = da0.isel({self.tdim: 0}, drop=True)
            self._tmom = self._as_moments_(mom, template.dims, template)
        else:
            self._tmom.values[...] = cstats.merge_moments(self._tmom.values, mom)

    def _append_spatial_(self, da0, da1):
        rest = [dim for dim in da0.dims[1:] if dim not in self.sdims]
        order = (self.tdim, *rest, *self.sdims)
        da0 = da0.transpose(*order)
        shape = da0.shape[: 1 + len(rest)]
        x = da0.values.reshape(int(np.prod(shape)), -1)
        y = da1.transpose(*order).values.reshape(x.shape) if self.dual else None
        w = self.weights
        if w is None:
            # Layer thicknesses and grid cell sizes along the reduced dimensions
            zdim = xcoords.get_zdim(da0, errors="ignore")
            xdim = xcoords.get_xdim(da0, errors="ignore")
            ydim = xcoords.get_ydim(da0, errors="ignore")
            parts = []
            if zdim in self.sdims:
                dz = xgrid.get_dz(da0, errors="ignore")
                if dz is None:
                    raise exceptions.XoaStatsError(
                        "Can't get the layer thicknesses from the data. "
                        "Please provide weights, or disable them with weights=False"
                    )
                parts.append(dz)
            if xdim in self.sdims or ydim in self.sdims:
                try:
                    dx, dy = xgrid.get_dx_dy(da0)
                except exceptions.XoaError as e:
                    exceptions.xoa_warn(
                        f"Horizontal dimensions are not weighted, since no grid was found: {e}"
                    )
                else:
                    parts.extend(res for dim, res in ((xdim, dx), (ydim, dy)) if dim in self.sdims)
            w = None
            for part in parts:
                w = part if w is None else w * part
        elif w is False:
            w = None
        if w is not None:
            if self.tdim in w.dims and self.tdim in da0.coords:
                w = w.sel({self.tdim: da0[self.tdim].values})
            try:
                w = w.broadcast_like(da0).transpose(*order)
            except (KeyError, ValueError) as e:
                raise exceptions.XoaStatsError(f"The weights do not match the data: {e}")
            if w.shape != da0.shape:
                raise exceptions.XoaStatsError("The weights do not match the data")
            w = w.values.reshape(x.shape)
        mom = cstats.get_moments(x, y, w).reshape((cstats.NMOM,) + shape)
        template = da0.isel({dim: 0 for dim in self.sdims}, drop=True)
        new = self._as_moments_(mom, template.dims, template)
        if self._smom is None:
            self._smom = new
        else:
            self._smom = xr.concat(
                [self._smom, new], dim=self.tdim, coords="minimal", compat="override"
            )

    def __iadd__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        return self.append(*args)

    def _parse_select_(self, select):
        if select is None:
            return {"t": self.all_stats, "s": self.all_stats}
        if isinstance(select, str):
            select = [select]
        sel = {"t": set(), "s": set()}
        for name in select:
            name = name.rstrip("01")
            if name[1:] in self.all_stats and name[0] in sel:
                sel[name[0]].add(name[1:])
            elif name in self.all_stats:
                sel["t"].add(name)
                sel["s"].add(name)
            else:
                raise exceptions.XoaStatsError(f"Invalid statistics: {name}")
        return sel

    _descs: ClassVar[dict] = {
        "count": "number of valid samples",
        "avail": "fraction of valid samples",
        "mean": "mean",
        "var": "variance",
        "std": "standard deviation",
        "min": "minimum",
        "max": "maximum",
        "bias": "bias",
        "rms": "root mean square difference",
        "crms": "centered root mean square difference",
        "cov": "covariance",
        "corr": "correlation",
    }
    _cell_methods: ClassVar[dict] = {
        "mean": "mean",
        "var": "variance",
        "std": "standard_deviation",
        "min": "minimum",
        "max": "maximum",
    }

    def _get_attrs_(self, pre, stat, suf):
        """Attributes of a statistics from those of the input variables"""
        if suf:
            idx = [int(suf)]
        else:
            idx = [0, 1] if self.dual and stat in ("bias", "rms", "crms", "cov", "corr") else [0]
        longs = [self.attrs[i].get("long_name") or self.names[i] for i in idx]
        longs = [long for long in longs if long]
        units = self.attrs[idx[0]].get("units")
        title = ("Temporal " if pre == "t" else "Spatial ") + self._descs[stat]
        attrs = {}
        if longs:
            attrs["long_name"] = title + " of " + " and ".join(longs)
        else:
            attrs["long_name"] = title
        if units and stat in ("mean", "std", "min", "max", "bias", "rms", "crms"):
            attrs["units"] = units
        elif units and stat in ("var", "cov"):
            attrs["units"] = f"({units})2"
        if stat in ("mean", "min", "max") and "standard_name" in self.attrs[idx[0]]:
            attrs["standard_name"] = self.attrs[idx[0]]["standard_name"]
        if stat in self._cell_methods:
            dims = [self.tdim] if pre == "t" else self.sdims
            attrs["cell_methods"] = (
                " ".join(dim + ":" for dim in dims) + " " + self._cell_methods[stat]
            )
        return attrs

    def get_stats(self, select=None, ddof=0):
        """Get the current statistics from the accumulated moments

        Parameters
        ----------
        select: None, str, list(str)
            Statistics to compute like ``"smean"`` or ``["tcorr", "scount"]``,
            or unprefixed ones like ``"rms"`` to get both temporal and spatial ones.
            If None, all available statistics are returned.
            See :attr:`all_stats`.
        ddof: int
            Delta degrees of freedom of variances and covariances.

        Return
        ------
        xarray.Dataset
            Statistics prefixed with ``"t"`` or ``"s"``, and suffixed with ``"0"``
            and ``"1"`` for per-variable ones of comparative accumulations.
            The *avail* statistics are the fractions of valid samples.
            Their attributes are made from those of the input arrays:
            ``long_name``, ``units`` and, when relevant, ``standard_name`` and ``cell_methods``.
        """
        if self.tdim is None:
            raise exceptions.XoaStatsError("Nothing has been accumulated yet")
        sel = self._parse_select_(select)
        ds = xr.Dataset()
        for pre, mom, total in (("t", self._tmom, self.nt), ("s", self._smom, self.ns)):
            if mom is None:
                continue
            stats = cstats.moments2stats(mom.values, self.dual, ddof)
            stats["avail"] = stats["count"] / total
            template = mom.isel(moment=0, drop=True)
            for name, values in stats.items():
                stat = name.rstrip("01")
                if stat in sel[pre]:
                    da = template.copy(data=values)
                    da.attrs = self._get_attrs_(pre, stat, name[len(stat) :])
                    ds[pre + name] = da
        return ds

    def dump(self, path):
        """Save the current state to a netcdf file, from which it can be restarted

        See also
        --------
        from_file
        """
        if self.tdim is None:
            raise exceptions.XoaStatsError("Nothing has been accumulated yet")
        attrs = {
            "xoa_stataccum_format": FORMAT_VERSION,
            "temporal": int(self.temporal),
            "spatial": int(self.spatial),
            "sdims": ",".join(self.sdims),
            "tdim": self.tdim,
            "dual": int(self.dual),
            "nt": self.nt,
            "ns": self.ns,
            "names": json.dumps(self.names),
        }
        root = xr.Dataset(attrs=attrs)
        if self.last is not None:
            root["last"] = xr.DataArray(self.last)
        root.to_netcdf(path, mode="w")
        for group, mom in (("t", self._tmom), ("s", self._smom)):
            if mom is not None:
                ds = mom.to_dataset(dim="moment")
                for name, attrs in zip(("mx", "my"), self.attrs):
                    ds[name].attrs = attrs
                ds.to_netcdf(path, mode="a", group=group)

    @classmethod
    def from_file(cls, path, weights=None, on_overlap="raise"):
        """Restart an accumulator from a file created by :meth:`dump`

        Parameters
        ----------
        path: str
        weights: None, xarray.DataArray
            Weights of spatial statistics, which are not stored in the file.
        on_overlap: str
            See the class parameters.

        See also
        --------
        dump
        """
        with xr.open_dataset(path) as root:
            attrs = root.attrs
            if attrs.get("xoa_stataccum_format") != FORMAT_VERSION:
                raise exceptions.XoaStatsError(f"Not a valid accumulator file: {path}")
            self = cls(
                temporal=bool(attrs["temporal"]),
                spatial=bool(attrs["spatial"]),
                sdims=[dim for dim in attrs["sdims"].split(",") if dim],
                weights=weights,
                on_overlap=on_overlap,
            )
            self.tdim = attrs["tdim"]
            self.dual = bool(attrs["dual"])
            self.nt = int(attrs["nt"])
            self.ns = int(attrs["ns"])
            self.names = json.loads(attrs["names"])
            if "last" in root:
                self.last = root["last"].values[()]
        for flag, group in ((self.temporal, "t"), (self.spatial, "s")):
            if flag:
                with xr.open_dataset(path, group=group) as ds:
                    self.attrs = [dict(ds["mx"].attrs), dict(ds["my"].attrs)]
                    mom = ds.load().to_array("moment").sel(moment=list(cstats.MOMENT_NAMES))
                setattr(self, "_" + group + "mom", mom)
        return self

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(temporal={self.temporal}, spatial={self.spatial}, "
            f"sdims={self.sdims}, nt={self.nt})"
        )
