#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regridding utilities
"""
from enum import IntEnum


from .__init__ import XoaError
from . import misc
from . import coords
from . import grid
from . import _interp


class XoaRegridError(XoaError):
    pass


class _intchoices_(IntEnum):

    def __str__(self):
        return self.name


class regrid1d_methods(_intchoices_, metaclass=misc.DefaultEnumMeta):
    """Supported :func:`regrid1d` methods"""
    #: Linear iterpolation
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


def regrid1d(da, coord, method=None, dim=None, coord_in_name=None,
             conserv=False, extrap=0):

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
