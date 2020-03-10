# -*- coding: utf-8 -*-
"""
Coordinates and dimensions utilities
"""

from .__init__ import XoaError
from . import cf


def get_lon(da):
    """Get the longitude coordinate"""
    return cf.get_cf_specs().search_coord(da, 'lon')


def get_lat(da):
    """Get the latitude coordinate"""
    return cf.get_cf_specs().search_coord(da, 'lat')


def get_depth(da):
    """Get the depth coordinate"""
    return cf.get_cf_specs().search_coord(da, 'depth')


def get_altitude(da):
    """Get the altitude coordinate"""
    return cf.get_cf_specs().search_coord(da, 'altitude')


def get_level(da, with_level=False):
    """Get the level coordinate"""
    return cf.get_cf_specs().search_coord(da, 'level')


def get_coords(da, coord_names):
    """Get several coordinates"""
    cfspecs = cf.get_cf_specs()
    return [cfspecs.search_coord(da, coord_name)
            for coord_name in coord_names]


def flush_work_dim_right(da, coord, dim=None):
    """Flush right the working dimension


    Returns
    -------
    xarray.DataArray
        Data or coordinate array
    xarray.DataArray
        Coordinate array
    """
    # Get the working dimensions
    if dim is None:
        cfspecs = cf.get_cf_specs()
        dim1 = cfspecs.coords.search_dim(coord)
        if dim1 is None:
            raise cf.XoaCFError("No CF dimension found for output coord. "
                                "Please specifiy the working dimension.")
        dim1, dim_type = dim1
        for c0 in da.coords.values():
            dim0 = cfspecs.coords.search_dim(c0, dim_type)
            if dim0:
                break
        else:
            raise cf.XoaCFError(
                "No CF {dim_type }dimension found for datarray. "
                "Please specifiy the working dimension.")
    else:
        if isinstance(dim, str):
            dim0 = dim1 = dim
        else:
            dim0, dim1 = dim
            assert dim0 in da.dims
            assert dim1 in coord.dims

    # Check other dimensions
    odims0 = set(da.dims).difference({dim0})
    odims1 = set(coord.dims).difference({dim1})
    if odims0.difference(odims1) and odims1.difference(odims0):
        raise XoaError("Conflicting non working dimensions")
    cdims = odims0.intersection(odims1).difference({dim0})
    for cdim in cdims:
        assert da.sizes[cdim] == coord.sizes[cdim]

    # Transpose to reflect coord order
    cdims1 = []
    for cdim1 in coord.dims:
        if cdim1 in cdims:
            cdims1.append(cdim1)
    coord = coord.transpose(Ellipsis, *(cdims1 + [dim1]))
    da = da.transpose(Ellipsis, *(cdims1 + [dim0]))

    # Broadcast array
    if da.ndim < coord.ndim:
        da = da.broadcast_like(coord, exclude=(dim0, dim1))

    return da, coord
