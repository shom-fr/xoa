# -*- coding: utf-8 -*-
"""
Colors and colormaps utilities
"""
# Copyright 2020-2021 Shom
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

import cmocean
import matplotlib.pyplot as plt

from .__init__ import xoa_warn


def crop_cmap(cmapin, vmin, vmax, pivot=0):
    """Crop a colormap so that it is centered around pivot

    This is a wrapper for :func:`cmocean.tools.crop`.

    Parameters
    ----------
    cmap: colormap
        Compatible with :func:`matplotlib.pyplot.get_cmap`.
    vmin: float
        Min data value
    vmax: float
        Max data value
    pivot: float
        The colormap will be centered on this value.
        Should be lower than vmax et greater than vmin.
    """
    cmapin = plt.get_cmap(cmapin)
    return cmocean.tools.crop(cmapin, vmin, vmax, pivot)


class CmapAdapter(object):
    """Adapt a given colormap to data

    Parameters
    ----------
    cmap:
        The colormap to adapt
    specs: str, None
        Transformation specifications or None.

        When a string, it must have the format ``"<type><value>"``, where
        type is ``"piv"`` or ``"cyc"``, and value is convertible to a float.

        If type is ``"piv"``, that colormap is expected typically
        to be **diverging**, and is cropped
        using :func:`crop_cmap`, after min and max value are set with
        :meth:`set_vlim`.

        If type is ``"cyc"``, min is set to 0 and max to ``value``,
        and the colormap is
        expected to be **circular**, like ``"cmo.phase"``.

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.color import CmapAdapter
        @suppress
        import matplotlib.pyplot as plt, numpy as np
        cma = CmapAdapter('cmo.balance', 'piv0')
        data = np.arange(100).reshape(10, 10) - 20
        cma.set_vlim(data.min(), data.max())
        plt.contourf(data, **cma.get_dict());
        @savefig api.color.cmapadapter.png
        plt.colorbar();
    """

    def __init__(self, cmap, specs=None):
        self.cmap = plt.get_cmap(cmap)
        self.vmin = self.vmax = None
        if specs is None:
            self.specs = None
        if specs.startswith("piv"):
            self.specs = ("pivot", float(specs[3:]))
        elif specs.startswith("cyc"):
            self.specs = ("cycle", float(specs[3:]))
            self.vmin = 0
            self.vmax = self.specs[1]

    def set_vlim(self, vmin, vmax):
        """Set the min and max data value for scaling"""
        if self.specs and self.specs[0] == "cycle":
            vmin, vmax = 0, 360.0
        self.vmin = vmin
        self.vmax = vmax
        return vmin, vmax

    def get_cmap(self):
        """Get the adapted colormap"""
        if self.specs[0] == "pivot":
            if self.vmin is None or self.vmax is None:
                xoa_warn(
                    "cmap not adapted since vmin and vmin are not set"
                )
                return self.cmap
            return crop_cmap(self.cmap, self.vmin, self.vmax, self.specs[1])

    def get_dict(self):
        """The specs for plots as a dict whose keys are cmap, vmin and vmax"""
        return dict(cmap=self.get_cmap(), vmin=self.vmin, vmax=self.vmax)
