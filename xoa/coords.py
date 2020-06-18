# -*- coding: utf-8 -*-
"""
Coordinates and dimensions utilities
"""
from collections.abc import Mapping

import xarray as xr

from .__init__ import XoaError, xoa_warn
from . import misc
from . import cf


def get_lon(da):
    """Get the longitude coordinate"""
    return cf.get_cf_specs().search_coord(da, 'lon')


def get_lat(da):
    """Get the latitude coordinate"""
    return cf.get_cf_specs().search(da, 'lat')


def get_depth(da):
    """Get the depth coordinate"""
    return cf.get_cf_specs().search(da, 'depth')


def get_altitude(da):
    """Get the altitude coordinate"""
    return cf.get_cf_specs().search(da, 'altitude')


def get_level(da):
    """Get the level coordinate"""
    return cf.get_cf_specs().search(da, 'level')


def get_vertical(da):
    """Get either depth or altitude"""
    cfspecs = cf.get_cf_specs()
    height = cfspecs.search(da, 'depth')
    if height is None:
        height = cfspecs.search(da, 'altitude')
    return height


def get_coords(da, coord_names):
    """Get several coordinates"""
    cfspecs = cf.get_cf_specs()
    return [cfspecs.search_coord(da, coord_name)
            for coord_name in coord_names]


class transpose_modes(misc.IntEnumChoices, metaclass=misc.DefaultEnumMeta):
    """Supported :func:`regrid1d` methods"""
    #: Basic xarray transpose with :meth:`xarray.DataArray.transpose`
    classic = 0
    basic = 0
    xarray = 0
    #: Transpose skipping incompatible dimensions
    compat = -1
    #: Transpose adding missing dimensions with a size of 1
    insert = 1
    #: Transpose resizing to missing dimensions.
    #: Note that dims must be an array or a dict of sizes
    #: otherwize new dimensions will have a size of 1.
    resize = 2


def transpose(da, dims, mode='compat'):
    """Transpose an array

    Parameters
    ----------
    da: xarray.DataArray
        Array to tranpose
    dims: tuple(str), xarray.DataArray, dict
        Target dimensions or array with dimensions
    mode: str, int
        Transpose mode as one of the following:
        {transpose_modes.rst_with_links}

    Return
    ------
    xarray.DataArray
        Transposed array

    Example
    -------
    .. ipython:: python

        @suppress
        import xarray as xr, numpy as np
        @suppress
        from xoa.coords import transpose
        a = xr.DataArray(np.ones((2, 3, 4)), dims=('y', 'x', 't'))
        b = xr.DataArray(np.ones((10, 3, 2)), dims=('m', 'y', 'x'))
        # classic
        transpose(a, (Ellipsis, 'y', 'x'), mode='classic')
        # insert
        transpose(a, ('m', 'y', 'x', 'z'), mode='insert')
        transpose(a, b, mode='insert')
        # resize
        transpose(a, b, mode='resize')
        transpose(a, b.sizes, mode='resize') # with dict
        # compat mode
        transpose(a, ('y', 'x'), mode='compat').dims
        transpose(a, b.dims, mode='compat').dims
        transpose(a, b, mode='compat').dims  # same as with b.dims

    See also
    --------
    xarray.DataArray.transpose
    """
    # Inits
    if hasattr(dims, 'dims'):
        sizes = dims.sizes
        dims = dims.dims
    elif isinstance(dims, Mapping):
        sizes = dims
        dims = list(dims.keys())
    else:
        sizes = None
    mode = str(transpose_modes[mode])
    kw = dict(transpose_coords=True)

    # Classic
    if mode == "classic":
        return da.transpose(*dims, **kw)

    # Get specs
    odims = ()
    expand_dims = {}
    with_ell = False
    for dim in dims:
        if dim is Ellipsis:
            with_ell = True
            odims += dim,
        elif dim in da.dims:
            odims += dim,
        elif mode == "insert":
            expand_dims[dim] = 1
            odims += dim,
        elif mode == "resize":
            if sizes is None or dim not in sizes:
                print(sizes is None, dim not in sizes)
                xoa_warn(f"new dim '{dim}' in transposition is set to one"
                         " since no size is provided to it")
                size = 1
            else:
                size = sizes[dim]
            expand_dims[dim] = size
            odims += dim,

    # Expand
    if expand_dims:
        da = da.expand_dims(expand_dims)

    # Input dimensions that were not specified in transposition
    # are flushed to the left
    if not with_ell and set(odims) < set(da.dims):
        odims = (...,) + odims

    # Transpose
    return da.transpose(*odims, **kw)


transpose.__doc__ = transpose.__doc__.format(**locals())


class DimFlusher1D(object):

    def __init__(self, da_in, coord_out, dim=None, coord_in_name=None):
        """Right-flush the working dimension

        Parameters
        ----------
        da_in: xarray.DataArray
            Input data array
        coord_out: xarray.DataArray
            Output coordinate array
        dim: str, tuple, None
            Working dimension
        coord_in_name: str, None
            Input coordinate name. If not provided, it is infered.
        """
        # Get the working dimensions
        if not isinstance(dim, (tuple, list)):
            dim = (dim, dim)
        dim0, dim1 = dim
        if None in dim or coord_in_name is None:
            cfspecs = cf.get_cf_specs()
        if dim1 is None:
            dim1 = cfspecs.search_dim(coord_out)
            if dim1 is None:
                raise cf.XoaCFError("No CF dimension found for output coord. "
                                    "Please specifiy the working dimension.")
            dim1, dim_type = dim1
        if dim0 is None:
            for c0 in da_in.coords.values():
                dim0 = cfspecs.coords.search_dim(c0, dim_type)
                if dim0:
                    break
            else:
                raise cf.XoaCFError(
                    "No CF {dim_type }dimension found for datarray. "
                    "Please specifiy the working dimension.")
        assert dim0 in da_in.dims
        assert dim1 in coord_out.dims

        # Input coordinate
        if coord_in_name:
            assert coord_in_name in da_in.coords, 'Invalid coordinate'
        else:
            coord_in = cfspecs.search_coord_from_dim(da_in, dim0)
            if coord_in is None:
                raise cf.XoaCFError(
                    f"No coordinate found matching dimension '{dim0}'")
            coord_in_name = coord_in.name

        # Check dims
        # - non-common other dimensions
        odims0 = set(da_in.dims).difference({dim0})
        odims1 = set(coord_out.dims).difference({dim1})
        if odims0.difference(odims1) and odims1.difference(odims0):
            raise XoaError("Conflicting non working dimensions")
        # - common dims, with size checking
        cdims = odims0.intersection(odims1).difference({dim0})
        for cdim in cdims:
            assert da_in.sizes[cdim] == coord_out.sizes[cdim]
        # - common dims in the order of da_in
        cdims0 = []
        for cdim0 in da_in.dims:
            if cdim0 in cdims:
                cdims0.append(cdim0)
        # - input dims with output dim
        dims01 = list(da_in.dims)
        if dim0 != dim1:
            dims01[dims01.index(dim0)] = dim1
        dims01 = tuple(dims01)

        # Store
        self._dim0, self._dim1 = dim0, dim1
        self._da_in = da_in
        self.coord_out = transpose(coord_out, (Ellipsis,) + dims01, "compat")
        self.coord_out_name = self.coord_out.name or coord_in.name
        # self._odims0 = odims0
        # self._odims1 = odims1
        # self._cdims0 = cdims0
        self.da_in = da_in

        # Transpose to push work dim right
        da_in = da_in.transpose(
            Ellipsis, *(cdims0 + [self._dim0]), transpose_coords=True)
        coord_out = coord_out.transpose(
            Ellipsis, *(cdims0 + [self._dim1]))

        # Broadcast data array
        # - data var
        if set(da_in.dims[:-1]) < set(coord_out.dims[:-1]):
            da_in = da_in.broadcast_like(coord_out,
                                         exclude=(self._dim0, self._dim1))
        # - input coordinate
        # if (set(coord_out.dims[:-1])set(da_in.coords[coord_in_name].dims[:-1])
        #         < set(coord_out.dims[:-1])):
        if (coord_out.ndim > 1 and set(coord_out.dims[:-1]) not in
                set(da_in.coords[coord_in_name].dims[:-1])):

        #set(coord_out.dims[:-1]) > set(da_in.coords[coord_in_name].dims[:-1]):
            if da_in.coords[coord_in_name].ndim == 1:
                coord_in_name, old_coord_in_name = (
                    coord_in_name + '_dimflush1d', coord_in_name)
            else:
                old_coord_in_name = coord_in_name
            da_in.coords[coord_in_name] = (
                da_in.coords[old_coord_in_name].broadcast_like(
                    coord_out, exclude=(self._dim0, self._dim1)))
        coord_in = da_in.coords[coord_in_name]
        # - output coordinate
        if (coord_out.ndim > 1 and
                set(coord_in.dims[:-1]) > set(coord_out.dims[:-1])):
            coord_out = coord_out.broadcast_like(
                coord_in, exclude=(self._dim0, self._dim1))

        # Info reverse transfoms
        # - input coords that doesn't have dim0 inside and must copied
        self.extra_coords = dict([(name, coord) for name, coord
                                  in da_in.coords.items()
                                  if dim0 not in coord.dims])
        # - da shape after reshape + broadcast
        self.work_shape = da_in.shape[:-1] + (coord_out.sizes[dim1], )
        self.work_dims = da_in.dims[:-1] + (dim1, )
        self.final_dims = list(self._da_in.dims)
        idim0 = self.final_dims.index(dim0)
        self.final_dims[idim0] = dim1
        self.final_dims = tuple(self.final_dims)
        # self.final_shape = list(self._da_in.shape)
        # self.final_shape[idim0] = coord_out.sizes[dim1]

        # Convert to numpy 2D
        self.da_in_data = da_in.data.reshape(-1, da_in.shape[-1])
        self.coord_in_data = coord_in.data.reshape(-1, coord_in.shape[-1])
        self.coord_out_data = coord_out.data.reshape(-1, coord_out.shape[-1])

    def get_back(self, data_out):

        data_out = data_out.reshape(self.work_shape)
        da_out = xr.DataArray(data_out, dims=self.work_dims)
        da_out = da_out.transpose(Ellipsis, *self.final_dims)
        da_out[self.coord_out_name] = self.coord_out
        da_out.attrs.update(self.da_in.attrs)
        da_out.coords.update(self.extra_coords)
        da_out.name = self.da_in.name

        return da_out
