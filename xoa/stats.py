# -*- coding: utf-8 -*-
"""
Stastical tools

"""
# Copyright 2020-2022 Shom
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

import re

import numpy as np
import numba
import xarray as xr

from .__init__ import xoa_warn, XoaError
from . import misc
from . import cf
from . import coords as xcoords


class XoaStatAccumError(XoaError):
    pass


class StatAccum:
    """Statistics accumulator

    It allows to perform single array and comparative statistics along
    spatial and/or temporal dimensions.

    It works by accumulating the sum of values, their square, their products, etc.
    Then statistics are computed on demande.

    The accumulation is make with the :meth:`append` method and it can be interrupted
    and restarted respectively by the :meth:`dump` and :meth:`from_file` methods.

    Parameters
    ----------
    temporal: bool
        Accumulate for temporal statistics
    spatial: bool
        Accumulate for spatial statistics
    """

    single_stats = 'mean', 'std', 'var', 'min', 'max'
    dual_stats = 'bias', 'rms', 'crms', 'corr', 'avail', 'count'

    #: Available statistics
    all_stats = single_stats + dual_stats
    # _single_accums = 'sum', 'sqr', 'min', 'max', 'hist'
    # _dual_accums = ('prod',)

    def __init__(self, temporal=True, spatial=False):

        self.ds = xr.Dataset()
        self.ds.attrs.update(tstats=temporal, sstats=spatial, nt=0)
        self.available_stats = []
        # if temporal:

    def _add_(self, name, da):
        if name not in self.ds:
            self.ds[name] = da
        else:
            self.ds[name] += da

    def append(self, da0, da1=None):
        """Append data to the accumulator

        Parameters
        ----------
        da0: xarray.DataArray
            First array
        da0: xarray.DataArray
            Second array.
            If provided, comparative statistics are performed.
        """
        self.ds.attrs["tdim"] = xcoords.get_tdim(da0, errors="raise")

        # Initialization
        if self.ds.nt == 0:

            # Two variables?
            self.ds.attrs["dual"] = da1 is not None
            if self.ds.dual:
                valid = da0.notnull() & da1.notnull()
                da0 = da0.where(valid, np.nan)
                da1 = da1.where(valid, np.nan)

            # Do we have time and/or space in input variables
            self.ds.attrs["sdims"] = [dim for dim in da0.dims if dim != self.ds.tdim]
            self.ds.attrs["sstats"] &= bool(self.ds.sdims)
            self.ds.attrs["ns"] = int(da0.size / da0.sizes[self.ds.tdim])

        # Accumulate

        # - temporal statistics
        if self.ds.tstats:
            self._add_("tcount", da0.count(self.ds.tdim))
            valid = self.ds["tcount"] != 0
            targets = [("0", da0)]
            if self.ds.dual:
                targets.append(("1", da1))
            for suffix, da in targets:
                damin = da.min(self.ds.tdim)
                damax = da.max(self.ds.tdim)
                if self.ds.nt == 0:
                    self.ds["tmin" + suffix] = damin
                    self.ds["tmax" + suffix] = damax
                else:
                    self.ds["tmin" + suffix] = self.ds["tmin" + suffix].where(
                        (self.ds["tmin" + suffix] < damin), damin
                    )
                    self.ds["tmax" + suffix] = self.ds["tmax" + suffix].where(
                        (self.ds["tmax" + suffix] > damax), damax
                    )

                da = da.fillna(0)
                self._add_("tsum" + suffix, da.sum(self.ds.tdim))
                self._add_("tsqr" + suffix, (da**2).sum(self.ds.tdim))
            if self.ds.dual:
                self._add_("tprod", (da0 * da1).sum(self.ds.tdim))

        # - spatial statistics
        sds = xr.Dataset()
        if self.ds.sstats:
            # if self.ds.dual:
            #     sds["scount"] = da0.count(self.ds.sdims)
            sds["scount"] = da.count(self.ds.sdims)
            targets = [("0", da0)]
            if self.ds.dual:
                targets.append(("1", da1))
            for suf, da in targets:
                sds["smin" + suf] = da.min(self.ds.sdims)
                sds["smax" + suf] = da.max(self.ds.sdims)
                da = da.fillna(0)
                sds["ssum" + suf] = da.sum(self.ds.sdims)
                sds["ssqr" + suf] = (da**2).sum(self.ds.sdims)
            if self.ds.dual:
                sds["sprod"] = (da0 * da1).sum(self.ds.sdims)
        elif self.ds.tdim in da0.coords:
            sds.coords[self.ds.tdim] = da0.coords[self.ds.tdim]

        # Update the dataset
        if self.ds.nt == 0:
            self.ds = xr.merge([self.ds, sds])
        else:
            self.ds = xr.concat([self.ds, sds], self.ds.tdim, data_vars="minimal")
        self.ds.attrs["nt"] += da0.sizes[self.ds.tdim]
        return self

    def __iadd__(self, args):
        return self.append(*args)

    def dump(self, ncfile):
        """Save the current accumulator to a netcdf file

        See also
        --------
        from_file
        """
        self.ds.to_netcdf(ncfile)

    @staticmethod
    def from_file(ncfile):
        """Initialize an accumulator from an old state

        See also
        --------
        dump
        """
        ds = xr.open_dataset(ncfile)
        sa = StatAccum(temporal=ds.tstats, spatial=ds.stats)
        sa.ds = ds
        return sa

    def __str__(self):
        return str(self.ds)

    def __repr__(self):
        ss = self.__class__.__name__
        ss += "(temporal={tstats}, spatial={sstats})\nds=".format(**self.ds.attrs)
        ss += str(self.ds)
        return ss

    def _check_(self, select, this):
        if isinstance(this, str):
            this = [this]
        return select is None or any([th.strip("01") in select for th in this])

    def get_stats(self, select=None):
        """Get current statistics from cumulated numerics

        Parameters
        ----------
        select: None, str, list(str)
            A selection of statistics types like ``"smean"`` or ``["tcorr", "scount"]``.
            If None, all statistics are returned.
            All statistics types are prefixed with either ``"t"`` or `"s"`` depending on
            wether temporal or spatial statistics are requested.
            The list of available unprefixed statistics is available here: :attr:`all_stats`
        ddof: int
            Degree of freedom used in variances and covariances.

        Returns
        -------
        xarray.Dataset
        """
        if select is not None:
            if isinstance(select, str):
                select = [select]
            select = [sel.strip("01") for sel in select]
        ds = xr.Dataset()
        prefixes = []
        if self.ds.tstats:
            prefixes.append("t")
        if self.ds.sstats:
            prefixes.append("s")
        sufs = ["0"]
        names = ["count", "min0", "max0"]
        if self.ds.dual:
            sufs.append("1")
            names.extend(["min1", "max1"])

        for pre in prefixes:
            # pre_ln = "Temporal " if pre == "t" else "Spatial "
            for cum in names:
                if self._check_(select, pre + cum):
                    ds[pre + cum] = self.ds[pre + cum]
            count = ds[pre + "count"]
            bad = count == 0
            dcount = xr.where(bad, np.nan, 1 / count)
            # dcountddof = xr.where(count < ddof + 1, np.nan, 1 / (count - ddof))
            sqrp = []
            for suf in sufs:

                if self._check_(select, pre + "mean"):
                    ds[pre + "mean" + suf] = self.ds[pre + "sum" + suf] * dcount

                if self._check_(
                    select, [pre + "std", pre + "rms", pre + "crms", pre + "cov", pre + "corr"]
                ):
                    sqrp.append(-(self.ds[pre + "sum" + suf] ** 2) * dcount)
                    sqrp[-1] += self.ds[pre + "sqr" + suf]

                if self._check_(select, pre + "var"):
                    ds[pre + "var" + suf] = sqrp[-1] * dcount

                if self._check_(select, pre + "std"):
                    ds[pre + "std" + suf] = np.sqrt(sqrp[-1] * dcount)

                # ds[pre + "mean" + suf].attrs["long_name"] = pre_ln +

            if self.ds.dual:
                sum0, sum1 = self.ds[pre + "sum0"], self.ds[pre + "sum1"]
                sqr0, sqr1 = self.ds[pre + "sqr0"], self.ds[pre + "sqr1"]
                prod = self.ds[pre + "prod"]

                if self._check_(select, pre + "bias"):
                    ds[pre + "bias"] = (sum1 - sum0) * dcount

                if self._check_(select, pre + "rms"):
                    rms = sqr0 + sqr1
                    rms -= prod
                    ds[pre + 'rms'] = np.sqrt(rms * dcount)

                if self._check_(select, [pre + "crms", pre + "cov", pre + "corr"]):
                    prodp = prod - sum0 * sum1 * dcount

                if self._check_(select, pre + "crms"):
                    crms = sqrp[0] + sqrp[1]
                    crms -= prodp * 2
                    ds[pre + 'crms'] = np.sqrt(crms * dcount)

                if self._check_(select, pre + "cov"):
                    ds[pre + 'cov'] = prodp * dcount

                if self._check_(select, pre + "corr"):
                    ds[pre + 'corr'] = prodp / np.sqrt(sqrp[0] * sqrp[1])

        return ds
