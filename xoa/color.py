# -*- coding: utf-8 -*-
"""
Colors and colormaps utilities
"""
# Copyright or © or Copr. Shom/Ifremer/Actimar
#
# stephane.raynaud@shom.fr, charria@ifremer.fr, wilkins@actimar.fr
#
# This software is a computer program whose purpose is to [describe
# functionalities and technical features of your software].
#
# This software is governed by the CeCILL license under French law and
# abiding by the rules of distribution of free software.  You can  use,
# modify and/ or redistribute the software under the terms of the CeCILL
# license as circulated by CEA, CNRS and INRIA at the following URL
# "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.

import warnings

import matplotlib.pyplot as plt
import cmocean


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
        When a string, it must have the format "<type><value>", where
        type is "piv" or "cyc", and value is convertible to a float.
        If "piv", that colormap expects typically diverging, and cropped
        using :func:`crop_cmap`, after min and max value are set with
        :meth:`set_vlim`.
        If "cyc", min is set to 0 and max to "value", and the colormap is
        expected to be circular, like "cmo.phase".

    Example
    -------
    >>> cma = CmapAdapter('cmo.balance', 'piv0')
    >>> cma.set_vlim(data.min(), data.max())
    >>> plt.pcolormesh(xx, yy, data, **cma.get_dict())
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
                warnings.warn(
                    "cmap not adapted since vmin and vmin are not set"
                )
                return self.cmap
            return crop_cmap(self.cmap, self.vmin, self.vmax, self.specs[1])

    def get_dict(self):
        """The specs for plots as a dict whose keys are cmap, vmin and vmax"""
        return dict(cmap=self.get_cmap(), vmin=self.vmin, vmax=self.vmax)
