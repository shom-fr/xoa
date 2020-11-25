#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regridding utilities
"""
# Copyright or © or Copr. Shom, 2020
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

from .__init__ import XoaError
from . import misc
from . import coords
from . import grid
from . import _interp


class XoaRegridError(XoaError):
    pass


class regrid1d_methods(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :func:`regrid1d` methods"""
    #: Linear iterpolation (default)
    linear = 1
    interp = 1  # compat
    # nearest = 0
    #: Cubic iterpolation
    cubic = 2
    #: Hermitian iterpolation
    hermit = 3
    hermitian = 3
    #: Cell-averaging or conservative regridding
    cellave = -1
    # cellerr = -2


class extrap_modes(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported extrapolation modes"""
    #: No extrapolation (default)
    no = 0
    none = 0
    #: Above (after)
    above = 1
    #: Below (before)
    below = -1
    #: Both below and above
    both = 2
    all = 2
    yes = 2


def regrid1d(da, coord, method=None, dim=None, coord_in_name=None,
             conserv=False, extrap=0):
    """Regrid along a single dimension

    The input and output coordinates may vary along other dimensions,
    which useful for instance for vertical interpolation in coastal
    ocean models.

    Parameters
    ----------
    da: xarray.DataArray
        Array to interpolate
    coord: xarray.DataArray
        Output coordinate
    method: str, int
        Regridding method as one of the following:
        {regrid1d_methods.rst_with_links}
    dim:  str, tuple(str), None
        Dimension on which to operate. If a string, it is expected to
        be the same dimension for both input and output coordinates.
        Else, provide a two-element tuple: ``(dim_coord_in, dim_coord_out)``.
    coord_in_name: str, None
        Name of the input coordinate array, which must be known of ``da``
    conserv: bool
        Use conservative regridding when using ``cellave`` method.
    extrap: str, int
        Extrapolation mode as one of the following:
        {extrap_modes.rst_with_links}

    Returns
    -------
    xarray.DataArray
        Regridded array with ``coord`` as new coordinate array.
    """

    # Array manager
    dfl = coords.DimFlusher1D(da, coord, dim=dim,
                              coord_in_name=coord_in_name)

    # Method
    method = regrid1d_methods[method]

    # Fortran function name and arguments
    func_name = str(method) + "1d"
    yi = dfl.coord_in_data
    yo = dfl.coord_out_data
    func_kwargs = {"vari": dfl.da_in_data}
    if dfl.coord_in_data.shape[0] == dfl.coord_out_data.shape[0] == 1:  # 1d
        yi = yi.reshape(-1)
        yo = yo.reshape(-1)
    else:
        # if method == regrid1d_methods.cellerr:
        #     raise XoaRegridError("cellerr regrid method is works only "
        #                          "with 1D input and output cordinates")
        if dfl.coord_out_data.shape[0] > 1:  # nd -> nd
            func_name += 'xx'
        else:  # nd -> 1d
            func_name += 'x'
            yo = yo.reshape(-1)
    if method < 0:
        func_kwargs.update(yib=grid.get_edges_1d(yi, axis=-1),
                           yob=grid.get_edges_1d(yo, axis=-1))
    else:
        func_kwargs.update(yi=yi, yo=yo, method=method)
    # if method != regrid1d_methods.cellerr:
    #     func_kwargs["extrap"] = extrap
    func = getattr(_interp, func_name)

    # Regrid
    varo = func(**func_kwargs)

    # Reform back
    return dfl.get_back(varo)


regrid1d.__doc__ = regrid1d.__doc__.format(**locals())
