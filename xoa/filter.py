"""
Filtering utilities
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

import inspect
import numpy as np
import xarray as xr
import numba

from .__init__ import XoaError, xoa_warn
from . import misc as xmisc
from . import coords as xcoords
from . import geo as xgeo

HOURLY_DEMERLIAC_WEIGHTS = [
    1,
    3,
    8,
    15,
    21,
    32,
    45,
    55,
    72,
    91,
    105,
    128,
    153,
    171,
    200,
    231,
    253,
    288,
    325,
    351,
    392,
    435,
    465,
    512,
    558,
    586,
    624,
    658,
    678,
    704,
    726,
    738,
    752,
    762,
    766,
    768,
    766,
    762,
    752,
    738,
    726,
    704,
    678,
    658,
    624,
    586,
    558,
    512,
    465,
    435,
    392,
    351,
    325,
    288,
    253,
    231,
    200,
    171,
    153,
    128,
    105,
    91,
    72,
    55,
    45,
    32,
    21,
    15,
    8,
    3,
    1,
]


def get_window_func(window, **kwargs):
    """Get a window function from its name

    Parameters
    ----------
    window: str, callable, list, array_like
        Specification to get the window function:

        - `callable`: used as is.
        - `str`: supposed to be function name of the :mod:`numpy` or
          :mod:`scipy.signal.windows` modules.
        - `list`, `array_like`: transformed to an array and interpolated
          onto ``size`` points.

    kwargs:
        Argument passed to the low level window function at calling time.

    Return
    ------
    callable
        A function that only a ``size`` argument

    Example
    -------
    .. ipython:: python

        @suppress
        import matplotlib.pyplot as plt
        @suppress
        from xoa.filter import get_window_func
        func0 = get_window_func("gaussian", std=22, sym=True)
        func1 = get_window_func([1, 2, 5, 2, 1])
        plt.plot(func0(100), label='Gaussian');
        plt.plot(func1(100), label='List/array');
        @savefig api.filter.get_window_func.png
        plt.legend();

    See also
    --------
    scipy.signal.windows
    numpy.bartlett
    numpy.blackman
    numpy.hamming
    numpy.hanning
    numpy.kaiser
    """
    # Explicit values so create a wrapper that interpolate them
    if isinstance(window, (list, np.ndarray)):

        def window_func(size):
            x = np.linspace(0, 1, size)
            xp = np.linspace(0, 1, len(window))
            return np.interp(x, xp, window)

        return window_func

    # Base function
    if isinstance(window, str):
        if hasattr(np, window):
            func = getattr(np, window)
        else:
            import scipy.signal.windows as sw

            func = getattr(sw, window, None)
            if func is None:
                raise XoaError(f'Invalid window name: {window}')

    else:
        func = window

    # Wrapper with args
    params = inspect.signature(func).parameters
    if kwargs or "sym" in params or "std" in params or "p" in params or "sig" in params:
        kwargs = kwargs.copy()

        def window_func(size):
            if "sym" in params and "sym" not in kwargs:
                kwargs["sym"] = True
            if "std" in params:
                kwargs["std"] = size * kwargs.get("std", 1 / 6)
            if "sig" in params:
                kwargs["sig"] = size * kwargs.get("sig", 1 / 6)
            if "p" in params and "p" not in kwargs:
                kwargs["p"] = 1
            return func(size, **kwargs)

        return window_func
    return func


def generate_isotropic_kernel(shape, window_func, fill_value=0, npt=None):
    """Generate an nD istropic kernel given a shape and a window function

    Parameters
    ----------
    shape: int, tuple
        Shape of the desired kernel
    window_func: str, callable
        Function that take a size parameter as unique argument.
        If a string, it is expected to be a numpy window function.
    fill_value: float
        Value to set when outside window bounds, near the corners
    npt: int, None
        Number of interpolation point to get the window value
        at all positions. It is infered from shape if not given.

    Return
    ------
    np.ndarray
        Isotropic kernel array

    Example
    -------

    .. ipython:: python

        @suppress
        import numpy as np, matplotlib.pyplot as plt
        @suppress
        from xoa.filter import generate_isotropic_kernel
        kernel = generate_isotropic_kernel((20, 30), "bartlett", np.nan)
        plt.matshow(kernel, cmap="cmo.solar");
        @savefig api.filter.generate_isotropic_kernel.png
        plt.colorbar();

    See also
    --------
    generate_orthogonal_kernel
    generate_kernel
    scipy.signal.windows
    numpy.bartlett
    numpy.blackman
    numpy.hamming
    numpy.hanning
    numpy.kaiser
    numpy.interp

    """
    # Window function
    window_func = get_window_func(window_func)

    # Normalised indices
    indices = np.indices(shape).astype('d')
    for i, width in enumerate(shape):
        indices[i] /= width - 1
        indices[i] -= 0.5

    # Distance from bounds with 0.5 at center and < 0 outside bounds
    x = 0.5 - np.sqrt((indices**2).sum(axis=0))

    # Window values
    if npt is None:
        npt = 2 * max(shape)
    fp = window_func(npt)
    xp = np.linspace(0, 1, npt)

    # Interpolation
    kernel = np.interp(x.ravel(), xp, fp).reshape(x.shape)
    kernel[x < 0] = fill_value

    return kernel


def generate_orthogonal_kernel(kernels, window_func="ones", fill_value=0.0):
    """Generate an nD kernel from orthogonal 1d kernels

    Parameters
    ----------
    kernels: list
        List of scalars and/or 1d kernels.
        In case of a scalar, it is converted to an 1d kernel with
        the ``window_func`` parameter.
    window_func: str, callable
        Function that take a size parameter as unique argument.
        If a string, it is expected to be a numpy window function.
    fill_value: float
        Value to set when outside window bounds, when creating an 1d kernel
        from a floating size

    Return
    ------
    np.ndarray
        Orthogonal kernel array

    Example
    -------
    .. ipython:: python

        @suppress
        import numpy as np, matplotlib.pyplot as plt
        @suppress
        from matplotlib.gridspec import GridSpec
        @suppress
        from xoa.filter import generate_orthogonal_kernel
        # From sizes and functions
        ny, nx = (21, 31)
        kernel = generate_orthogonal_kernel((ny, nx), "bartlett")
        j, i = 5, 20
        plt.plot([3, 4])
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 3, figure=fig)
        ax0 = fig.add_subplot(gs[1:, :2])
        ax0.matshow(kernel)
        ax0.axhline(j, color='tab:red')
        ax1 = fig.add_subplot(gs[0, :2], sharex=ax0)
        ax1.plot(np.arange(nx), kernel[j], color='tab:red')
        ax2 = fig.add_subplot(gs[1:, -1], sharey=ax0)
        ax2.plot(kernel[:, i], np.arange(ny), color='tab:orange');
        @savefig api.filter.generate_orthogonal_kernel_0.png
        ax0.axvline(i, color='tab:orange')
        # From 1d kernels
        kernel = generate_orthogonal_kernel(([1, 1, 1], [1, 2, 3, 2, 1]))
        plt.figure()
        @savefig api.filter.generate_orthogonal_kernel_1.png
        plt.matshow(kernel);



    See also
    --------
    generate_isotropic_kernel
    generate_kernel
    scipy.signal.windows
    numpy.bartlett
    numpy.blackman
    numpy.hamming
    numpy.hanning
    numpy.kaiser
    numpy.interp

    """
    kernel = None
    for k1d in kernels:
        # From scalar to 1d
        if np.isscalar(k1d):
            window_func = get_window_func(window_func)
            if isinstance(k1d, int):
                k1d = window_func(k1d)
            else:  # Float case
                size = int(k1d)
                if size != k1d:
                    size += 2
                k1d = np.full(size, fill_value)
                k1d[1:-1] = window_func(size - 2)
        else:
            k1d = np.asarray(k1d)

        # Orthogonal merge
        if kernel is None:
            kernel = k1d
        else:
            kernel = np.tensordot(kernel[:, None], k1d[None], axes=1)

    return kernel


def generate_kernel(
    kernel, data, window_func="ones", isotropic=False, fill_value=0.0, window_kwargs=None, **kwargs
):
    """Generate a kernel that is compatible with a given data array

    Parameters
    ----------
    kernel: xarray.DataArray, np.ndarray, int, list, dictorthokernels
        Ready to use kernel or specs to generate it.

        - If an int, the kernel built with ones with a size
          of `kernel` along all dimensions.
        - If a tuple, the kernel is built with ones and a shape
          equal to `kernel`.
        - If a numpy array, it is used as is.

        The final data array is transposed and/or expanded with
        :func:`xoa.coords.transpose` to fit into the input data array.
    data: xarray.DataArray
        Data array that the kernel must be compatible with.
        If the kernel has more than one dimension, it is expanded with a size of 1
        for missing dimensions.
    isotropic: bool, tuple
        Tuple of the dimensions over which must be computed isotropically.
    fill_value: float
        Value used by :func:`generate_isotropic_kernel` to fill the isotropic kernel
        in its corners.
    window_func: str
        Function to generate the kernel from its size by calling :func:`get_window_func`
    window_kwargs: dict
        Optional arguments passed to :func:`get_window_func`
    kwargs: dict
        Extra parameters are merged with `window_kwargs`

    Return
    ------
    xarray.DataArray
        Kernel array with suitable dimensions and shape

    See also
    --------
    generate_isotropic_kernel
    generate_orthogonal_kernel
    get_window_func
    scipy.signal.windows
    numpy.bartlett
    numpy.blackman
    numpy.hamming
    numpy.hanning
    numpy.kaiser
    xoa.coords.transpose
    """
    # Isotropic
    if isotropic is True:
        isotropic = data.dims

    # Convert to tuple
    if isinstance(kernel, int):
        kernel = (kernel,) * data.ndim

    # Convert tuple to dict with dims
    if isinstance(kernel, tuple):
        if len(kernel) > data.ndim:
            raise XoaError(
                "Too many dimensions for your kernel: {} > {}".format(len(kernel), data.ndim)
            )
        kernel = dict(item for item in zip(data.dims[-len(kernel) :], kernel))

    # Convert list to dict with dims
    elif isinstance(kernel, list):
        kernel = dict((dim, kernel) for dim in data.dims)

    # From an size or 1d kernel for given dimensions
    if isinstance(kernel, dict):
        # Isotropic parameter
        if isotropic:
            if isotropic is True:
                isotropic = data.dims
            elif not set(isotropic).issubset(data.dims):
                raise XoaError("invalid dimensions for isotropic keyword")

        # Split into consistant isotropic and orthogonal kernels
        isokernels_sizes = {}
        isokernel = None
        orthokernels = {}
        for dim, kn in kernel.items():
            if not isotropic or dim not in isotropic:
                orthokernels[dim] = kn
            else:
                if isokernel:
                    if (np.isscalar(isokernel) and not np.isscalar(kn)) or (
                        not np.isscalar(isokernel) and np.isscalar(kn)
                    ):
                        raise XoaError(
                            "Inconsistant mix of 1d and scalar "
                            "kernel specs for building isotropic "
                            "kernel"
                        )
                    if (
                        not np.isscalar(kn)
                        and not np.isscalar(isokernel)
                        and not np.allclose(kn, isokernel)
                    ):
                        raise XoaError("Inconsistant 1d kernels for building " "isotropic kernel")
                else:
                    isokernel = kn
                size = kn if np.isscalar(kn) else len(kn)
                isokernels_sizes[dim] = size

        # Merge orthogonal kernels
        dims = ()
        kernel = None
        window_kwargs = {} if window_kwargs is None else window_kwargs
        if kwargs:
            window_kwargs.update(kwargs)
        if orthokernels:
            dims += tuple(orthokernels.keys())
            window_func = get_window_func(window_func, **window_kwargs)
            sizes = tuple(orthokernels.values())
            kernel = generate_orthogonal_kernel(
                sizes, window_func=window_func, fill_value=fill_value
            )

        # Build isotropic kernel
        if isokernel:
            # List/array
            if not np.isscalar(isokernel):
                window_func = get_window_func(isokernel, **window_kwargs)

            # nD isotropic kernel
            isokernels = generate_isotropic_kernel(
                tuple(isokernels_sizes.values()), window_func, fill_value=fill_value
            )

            # Update final kernel
            dims += tuple(isokernels_sizes.keys())
            if kernel is None:
                kernel = isokernels
            else:
                kernel = np.tensordot(kernel[..., None], isokernels[None, ...])

    # Numpy
    elif isinstance(kernel, np.ndarray):
        if kernel.ndim > data.ndim:
            raise XoaError(
                "too many dimensions for your numpy kernel: {} > {}".format(kernel.dim, data.ndim)
            )
        dims = data.dims[-kernel.ndim :]

    # Data array
    if not isinstance(kernel, xr.DataArray):
        kernel = xr.DataArray(kernel, dims=dims)
    elif not set(kernel.dims).issubset(set(data.dims)):
        raise XoaError(f"kernel dimensions {kernel.dims} " f"are not a subset of {data.dims}")

    # Finalize
    kernel = kernel.astype(data.dtype)
    if kernel.ndim == 1:
        return kernel
    return xcoords.transpose(kernel, data, mode="insert").astype(data.dtype)


def shapiro_kernel(dims):
    """Generate a shapiro kernel

    Parameters
    ----------
    dims: str, tuple
        Dimension names

    Return
    ------
    xarray.DataArray
        The kernel as a data array with provided dims and a shape of
        ``(3,)*len(dims)``

    Example
    -------
    .. ipython:: python

        @suppress
        from xoa.filter import shapiro_kernel
        shapiro_kernel('nx')
        shapiro_kernel(('ny', 'nx'))
        shapiro_kernel(('nt', 'ny', 'nx'))

    """
    if isinstance(dims, str):
        dims = (dims,)
    ndim = len(dims)
    kernel = np.zeros((3,) * ndim, dtype='d')
    indices = np.indices(kernel.shape)
    for idx in indices:
        idx[idx == 2] = 0
        kernel += idx
    return xr.DataArray(kernel, dims=dims)


def _convolve_(data, kernel, normalize, na_thres, axis=None):
    """Pure numpy convolution that takes care of nans"""
    # Convolution function
    kwc = {"mode": "constant"}
    if kernel.ndim != 1:
        from scipy.ndimage.filters import convolve as convolve_func

        # from scipy.signal import convolve
        assert data.ndim == kernel.ndim
    # elif data.ndim == 1:
    #     convolve_func = np.convolve
    else:
        from scipy.ndimage.filters import convolve1d as convolve_func

        kwc["axis"] = axis

    # Kernel
    if kernel.dtype is not data.dtype:
        xoa_warn("The dtype of your kernel is not the same as that of your data. Converting it...")
        kernel = kernel.astype(data.dtype)

    # Guess mask
    bad = np.isnan(data)
    data = np.where(bad, 0, data)

    # Convolutions
    cdata = convolve_func(data, kernel, cval=0.0, **kwc)
    weights = convolve_func((~bad).astype('i'), kernel, cval=0, **kwc)
    # weights = np.clip(weights, 0, kernel.sum())

    # Weigthing and masking
    bad = weights <= kernel.sum() * np.clip(1e-6, 1 - na_thres, 1 - 1e-6)
    if normalize:
        weights = np.where(bad, 1, weights)
        cdata /= weights
    return np.where(bad, np.nan, cdata)


def convolve(data, kernel, normalize=False, na_thres=0, kernel_kwargs=None, **kwargs):
    """N-dimensional convolution that takes care of nans

    Parameters
    ----------
    data: xarray.DataArray
        Array to filter
    kernel: int, tuple, numpy.ndarray, xarray.DataArray
        Convolution kernel. See :func:`generate_kernel`.
    normalize: bool
        Divide the convolution product by the local sum weights.
        The result is then a weighted average.
    na_thres: float
        A float between 0 and 1 that defines the allowed level a NaN contamination.
        Examples of the behavioir at a single location:

            - `0`: Output is masked if a single NaN is found.
            - `0.5`: Output is masked only more than 50% of the input data are masked.
            - `1`: Output is masked if all input data are masked.
    kernel_kwargs: dict, None
        Extra parameters passed to :func:`generate_kernel`.
    kwargs: dict
        Extra parameters are merged with `kernel_kwargs`

    Return
    ------
    xarray.DataArray
        The filtered array with the same shape, attributes and coordinates
        as the input array.

    See also
    --------
    scipy.signal.convolve
    generate_kernel

    Example
    -------
    .. ipython:: python

        @suppress
        import xarray as xr, numpy as np, matplotlib.pyplot as plt
        @suppress
        from xoa.filter import convolve
        data = xr.DataArray(np.random.normal(size=(50, 70)), dims=('y', 'x'))
        data[10:20, 10:20] = np.nan # introduce missing data
        kernel = dict(x=[1, 2, 5, 2, 1], y=[1, 2, 1])
        datac = convolve(data, kernel, normalize=True, na_thres=1)
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 3))
        kw = dict(vmin=data.min(), vmax=data.max())
        data.plot.pcolormesh(ax=ax0, **kw);
        @savefig api.filter.convolve.png
        datac.plot.pcolormesh(ax=ax1, **kw);

    """
    # Adapt the kernel to the data
    kernel_kwargs = kernel_kwargs if kernel_kwargs is not None else {}
    if kwargs:
        kernel_kwargs.update(kwargs)
    kernel = generate_kernel(kernel, data, **kernel_kwargs)

    # Numpy convolution
    axis = data.get_axis_num(kernel.dims[0]) if kernel.ndim == 1 else None
    datac = _convolve_(data.data, kernel.data, normalize, na_thres, axis)

    # Format
    return xr.DataArray(datac, coords=data.coords, attrs=data.attrs, dims=data.dims)


def _get_xydims_(data, xdim, ydim):
    if xdim is None:
        xdim = xcoords.get_xdims(data, 'x', allow_positional=False, errors="raise")
    else:
        assert xdim in data.dims, f"Invalid x dimension: {xdim}"
    if ydim is None:
        ydim = xcoords.get_ydims(data, 'y', allow_positional=False, errors="raise")
    else:
        assert ydim in data.dims, f"Invalid y dimension: {ydim}"
    return xdim, ydim


def _reduce_mask_(data, excluded_dims):
    rmask = data.isnull()
    for dim in set(data.dims) - set(excluded_dims):
        rmask = rmask.any(dim=dim)
    return rmask


def _convolve_and_fill_(data, kernel):
    return data.fillna(convolve(data, kernel, normalize=True, na_thres=1))


def smooth(data, kernel, **kwargs):
    """A short to ``convolve(data, kernel, normalize=True)``

    See :func:`convolve` for the complete list of options.

    Parameters
    ----------
    data: xarray.DataArray
        Array to filter
    kernel: int, tuple, numpy.ndarray, xarray.DataArray
        Convolution kernel. See :func:`generate_kernel`.

    See also
    --------
    convolve
    """
    kwargs["normalize"] = True
    return convolve(data, kernel, **kwargs)


def erode_mask(data, until=1, kernel=None):
    """Erode the horizontal mask using smoothing

    Missing values are filled with the smoothed field in a iterative way.
    Two cases:

        - Erode a fixed number of times.
        - Erode the data mask until there is no missing value where
          a given horirizontal mask is False.

    Parameters
    ----------
    data: xarray.DataArray
        Array of at least 2 dimensions, that are supposed to be horizontal.
    until: xarray.DataArray, int
        Either a minimal mask, or a max number of iteration.
    kernel: None, "shapiro", xarray.DataArray
        Defaults to a :func:`shapiro <shapiro_kernel>` kernel designed
        with all data dimensions.
        If ``kernel`` is provided, it must a compatible with
        :func:`generate_kernel`.

    Return
    ------
    xarray.DataArray
        Data array similar to input array, with its eroded
        along x and y dimensions.

    See also
    --------
    erode_coast
    sharpiro_kernel
    """
    # Kernel
    if kernel is None:
        kernel = "shapiro"
    if isinstance(kernel, str) and kernel == "shapiro":
        kernel = shapiro_kernel(data.dims)
    kernel = generate_kernel(kernel, data)
    kdims = kernel.squeeze().dims

    # Iteration or mask
    if isinstance(until, int):
        niter = until
        mask = None
    else:
        mask = until
        if not set(mask.dims).issubset(data.dims):
            raise XoaError('mask dims must be a subset of data dims')
        mask = xcoords.transpose(mask, data, mode="compat")

    # Filter
    if mask is not None:
        nmask_min = int(mask.sum())
        nmask = np.inf
        while nmask > nmask_min:
            data = _convolve_and_fill_(data, kernel)
            rmask = _reduce_mask_(data, kdims)
            nmask = (mask | rmask).sum()
    else:
        for i in range(niter):
            data = _convolve_and_fill_(data, kernel)

    return data


def erode_coast(data, until=1, kernel=None, xdim=None, ydim=None):
    """Just like :func:`erode_mask` but specialized for the horizontal dimensions

    Parameters
    ----------
    data: xarray.DataArray
        Array of at least 2 dimensions, that are supposed to be horizontal.
    until: array_like, int
        Either a minimal mask, or a max number of iteration.
    kernel:
        Defaults to a :func:`shapiro <shapiro_kernel>` kernel.
        In this case, ``xdim`` and ``ydim`` can be set to the
        horizontal dimensions, otherwise they are inferred.
    xdim: None
        Name of the X dimension, which is infered by default.
    ydim: None
        Name of the Y dimension, which is infered by default.

    Return
    ------
    xarray.DataArray
        Data array similar to input array, with its eroded
        along x and y dimensions.

    See also
    --------
    erode_mask
    sharpiro_kernel
    """
    # We must have X and Y dimensions
    if xdim is None:
        xdim = xcoords.get_xdim(data, errors="raise")
    else:
        assert xdim in data.dims, f"Invalid x dimension: {xdim}"
    if ydim is None:
        ydim = xcoords.get_ydim(data, errors="raise")
    else:
        assert ydim in data.dims, f"Invalid y dimension: {ydim}"

    # Kernel
    if isinstance(kernel, xr.DataArray):
        assert xdim in kernel.dims, f"kernel must have dimension: {xdim}"
        assert ydim in kernel.dims, f"kernel must have dimension: {ydim}"
    elif kernel is None or kernel == "shapiro":
        kernel = shapiro_kernel((ydim, xdim))

    # Mask array
    if not isinstance(until, int):
        assert xdim in until.dims, f"mask must have dimension: {xdim}"
        assert ydim in until.dims, f"mask must have dimension: {ydim}"

    # Filter
    return erode_mask(data, until=until, kernel=kernel)


def demerliac(da, na_thres=0, dt_tol=0.01):
    """Apply a dermerliac filter on a data array

    Note that the data array must have a valid time dimension.
    When the time step is less than an hour, an interpolation is made on the weights
    since they are made for hourly time series.

    Parameters
    ----------
    da: xarray.DataArray
    dt_tol: float
        Relative tolerance for the time step variability
    na_thres: float
        A float between 0 and 1 that defines the allowed level a NaN contamination.
        See :func:`convolve`.

    Return
    ------
    xarray.DataArray
    """
    # Get time dimension
    tdim = xcoords.get_tdim(da, errors="ignore")
    if tdim is None:
        xoa_warn("Cannot apply the Demerliac filter since to time dimension found")
        return da.copy()

    # Weights
    weights = np.array(HOURLY_DEMERLIAC_WEIGHTS, "d")
    if tdim not in da.indexes:
        xoa_warn("Not time coordinate found so we assume hourly data")
    else:
        dt = np.diff(da[tdim].values) / np.timedelta64(3600, "s")
        ddt = dt.ptp()
        mdt = dt.mean()
        if ddt > dt_tol * mdt:
            raise XoaError(
                "The variability of your time steps is above the allowed level "
                " to apply a Dermerliac filter"
            )
        if mdt > 1 + dt_tol:
            xoa_warn(
                "You should not apply a Demerliac filter to data that are less "
                f"than hourly sampled. Current time step: {dt:1.2f}"
            )
        elif mdt < 1 - dt_tol:
            nw = len(weights)
            from scipy.interpolate import interp1d

            weights = interp1d(weights, np.linspace(0, 1, nw), "cubic")(np.linspace(0, 1, nw / dt))

    # Apply
    return convolve(da, {tdim: weights}, normalize=True, na_thres=na_thres)


@numba.njit(cache=True)
def _get_decimate_arg_(lons, lats, radius):
    """Get which point to keep to make sure they are enough distant of one another

    A loop is made on all points: a point that is at a distance less that `radius`
    from a point that was previously marked as True is marked as False.

    Parameters
    ----------
    x: numpy.array
        1D array of longitudes
    y: numpy.array
        1D array of latitudes
    radius: float
        Radius relative to the sphere radius

    Returns
    -------
    numpy.array
        1D array of boolean with True set for the points to keep

    See also
    --------
    xoa.geo.haversine
    decimate
    """
    npts = lons.size
    keep = np.ones(npts, dtype="?")
    for i in range(1, npts):
        for j in range(i):
            if keep[j]:
                dist = xgeo._haversine_(lons[i], lats[i], lons[j], lats[j])
                if dist < radius:
                    keep[i] = False
                    break
    return keep


@numba.njit(cache=True)
def _decimate_by_average_(lons, lats, radius, keep, data):
    npts = lons.size
    nkept = keep.sum()
    cdata = np.zeros(data.shape[:-1] + (nkept,), data.dtype)
    ccount = np.zeros(nkept, data.dtype)
    k = 0
    for i in numba.prange(npts):
        if keep[i]:
            for j in range(npts):
                dist = xgeo._haversine_(lons[i], lats[i], lons[j], lats[j])
                if dist <= radius:
                    cdata[..., k] += data[..., j]
                    ccount[k] += 1.0
            k += 1
    for i in numba.prange(nkept):
        cdata[..., i] = cdata[..., i] / ccount[i]
    return cdata


class decimation_methods(xmisc.IntEnumChoices, metaclass=xmisc.DefaultEnumMeta):
    """Supported :func:`regrid1d` methods"""

    #: Average (default)
    average = 1
    #: Pick
    pick = 0
    kill = 0


def decimate(
    obj,
    radius,
    method="average",
    stack_dim="npts",
    sphere_radius=xgeo.EARTH_RADIUS,
    smooth_factor=1.0,
):
    """Decimate a data array or dataset by removing too close points

    It typical use is for undersampling a huge dataset before a geostatistical interpolation.

    A loop is made on all points: a point that is at a distance less that `radius`
    from a previously selected point is not selected.
    In the case of the "average" method, an average is made at selected points within a radius
    of `radius*smooth_factor`.

    .. warning:: X and Y dimensions are stacked during this process if not already stacked.

    Parameters
    ----------
    obj: xarray.DataArray, xarray.Dataset
        Array or dataset to decimate with lon and lat coordinates
    radius: float
        Radius in meters
    method:str, int
        Decimation method:
        {decimation_methods.rst_with_links}.
        `pick` operates with crude undersampling, while `average` performs
        an average with a radius of size `radius*smooth_factor`.
    smooth_factor: float
        Factor applied to `radius` for the average process, not for the selection
        process.
        A `smooth_factor` of zero is equivalent to `method` set to "pick".
        A `smooth_factor` which is equal to the infinite returns a spatial average over the domain
        at all selected points.
    stack_dim: str
        When lon and lat coordinates have several or uncommon dimensions,
        they are stacked onto a single dimension whose name is `stack_dim`,
        with function :func:`~xoa.coords.geo_stack`.
    sphere_radius: float
        Radius of the sphere in meters

    Example
    -------

    .. ipython:: python

        @suppress
        import xarray as xr, numpy as np, cmocean, matplotlib.pyplot as plt
        @suppress
        from xoa.filter import decimate
        # Create the sample
        npts = 1000
        x = np.random.uniform(-20, -10, npts)
        y = np.random.uniform(40, 50, npts)
        ds = xr.Dataset(
            {{"temp": ("npts", 20+5*np.exp(-(x+15)**2/3**2-(y-45)**2/3**2))}},
            coords={{"lon": ("npts", x), "lat": ("npts", y)}})

        # Decimate it with a radius of 150 km
        dsc = decimate(ds, radius=150e3)

        # Plot
        fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        axs[0].scatter(ds.lon, ds.lat, c=ds.temp, cmap='cmo.thermal')
        axs[0].set_title("Complete")
        axs[1].scatter(dsc.lon, dsc.lat, c=dsc.temp, cmap='cmo.thermal')
        @savefig api.filter.decimate.png
        axs[1].set_title("Decimated")

    See also
    --------
    xoa.geo.haversine
    xarray.DataArray.stack
    """
    # Stacked coordinates
    obj = xcoords.geo_stack(obj, stack_dim)
    lon = xcoords.get_lon(obj)
    x = lon.values
    y = xcoords.get_lat(obj).values

    # Decimation arg
    keep = xr.DataArray(_get_decimate_arg_(x, y, radius / sphere_radius), dims=lon.dims)

    # Compress
    objc = obj.where(keep, drop=True)

    # By average
    method = decimation_methods[method]
    if method.name == "average":
        targets = list(objc) if isinstance(objc, xr.Dataset) else [0]
        for target in targets:
            if lon.dims[0] in obj[target].dims:
                values = obj[target].values.reshape(-1, obj[target].shape[-1])
                objc[target][:] = _decimate_by_average_(
                    x, y, smooth_factor * radius / sphere_radius, keep.values, values
                ).reshape((objc[target].shape[:-1] + (-1,)))

    return objc


decimate.__doc__ = decimate.__doc__.format(**locals())
