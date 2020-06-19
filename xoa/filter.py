"""
Filtering utilities
"""
import numpy as np
import xarray as xr

from .__init__ import XoaError
from .coords import transpose


def generate_isotropic_kernel(shape, func, fill_value=0, npt=None):
    """Generate an nD istropic kernel given a shape and a window function

    Parameters
    ----------
    func: str, callable
        Function that take a size parameter as unique argument.
        If a string, it is expected to be a numpy window function.
    shape: int, tuple
        Shape of the desired kernel
    fill_value: float
        Value to set when outside window bounds, near the corners
    npt: int, None
        Number of interpolation point to get the window value
        at all positions. It is infered from shape if not given.

    Return
    ------
    array_like(shape)
        Output kernel

    Example
    -------
    .. ipython:: python

        @suppress
        import numpy as np, matplotlib.pyplot as plt
        @suppress
        from xoa.filter import generate_isotropic_kernel
        kernel = generate_isotropic_kernel((150, 200), "bartlett", np.nan)
        plt.axes(aspect=1);
        plt.pcolormesh(kernel, cmap="cmo.solar");
        @savefig api.filter.generate_isotropic_kernel.png
        plt.colorbar();

    See also
    --------
    numpy.bartlett
    numpy.blackman
    numpy.hamming
    numpy.hanning
    numpy.kaiser
    numpy.interp

    """
    # Windox function
    if isinstance(func, str):
        func = getattr(np, func)

    # Normalised indices
    indices = np.indices(shape).astype('d')
    for i, width in enumerate(shape):
        indices[i] /= (width-1)
        indices[i] -= 0.5

    # Distance from bounds with 0.5 at center and < 0 outside bounds
    x = 0.5 - np.sqrt((indices**2).sum(axis=0))

    # Window values
    if npt is None:
        npt = 2*max(shape)
    fp = func(npt)
    xp = np.linspace(0, 1, npt)

    # Interpolation
    kernel = np.interp(x.ravel(), xp, fp).reshape(x.shape)
    kernel[x < 0] = fill_value

    return kernel


def generate_kernel(kernel, data):
    """Generate a kernel that is compatible with a given data array

    Parameters
    ----------
    data: xarray.DataArray
        Data array that the kernel must be compatible with
    kernel: xarray.DataArray, np.ndarray, int, list, dict
        Ready to use kernel or specs to generate it.

        - If an int, the kernel built with ones with a size
          of `kernel` along all dimensions.
        - If a tuple, the kernel is built with ones and a shape
          equal to `kernel`.
        - If a numpy array, it is used as is.

        The final data array is transposed and/or expanded with
        :func:`xoa.coords.transpose` to fit into the input data array.

    Return
    ------
    xarray.DataArray
        Kernel array with suitable dimensions and shape

    See also
    --------
    generate_isotropic_kernel
    xoa.coords.transpose
    """
    # Constant kernel of the given size
    if isinstance(kernel, int):
        kernel = (kernel,)*data.ndim

    # Constant kernel of the given sizes
    if isinstance(kernel, tuple):
        if len(kernel) > data.ndim:
            raise XoaError("Too many dimensions for your kernel: {} > {}"
                           .format(len(kernel), data.ndim))
        kernel = np.ones(kernel, dtype=data.dtype)
        dims = data.dims[-kernel.ndim:]

    # From a single 1d kernel
    elif isinstance(kernel, list):
        kernel = np.array(kernel)
        k1d = kernel
        while kernel.ndim < data.ndim:
            kernel = np.tensordot(k1d[:, None], kernel[None], axes=1)
        dims = data.dims

    # From an 1d kernel for given dimensions
    elif isinstance(kernel, dict):
        kd = kernel
        for dim, k1d in kd.items():
            k1d = np.array(k1d)
            if dim not in data.dims:
                raise XoaError(f"invalid dimension for your kernel: {dim}")
            if kernel is kd:
                kernel = k1d
            else:
                kernel = np.tensordot(kernel[:, None], k1d[None], axes=1)
        dims = list(kd.keys())

    # Numpy
    elif isinstance(kernel, np.ndarray):
        if kernel.ndim > data.ndim:
            raise XoaError("Too many dimensions for your numpy kernel: {} > {}"
                           .format(kernel.dim, data.ndim))
        dims = data.dims[-kernel.ndim:]

    # Data array
    if not isinstance(kernel, xr.DataArray):
        kernel = xr.DataArray(kernel, dims=dims)
    elif not set(kernel.dims).issubset(set(data.dims)):
        raise XoaError(f"kernel dimensions {kernel.dims} "
                       f"are not a subset of {data.dims}")

    # Finalize
    return transpose(kernel, data, mode="insert").astype(data.dtype)


def _convolve_(data, kernel, normalize):
    """Pure numpy convolution that takes care of nans"""
    import scipy.signal as ss

    # Kernel
    assert data.ndim == kernel.ndim

    # Guess mask
    bad = np.isnan(data)
    with_mask = bad.any()
    if with_mask:
        data = np.where(bad, 0, data)

    # Convolutions
    cdata = ss.convolve(data, kernel, mode='same')
    if normalize:
        weights = ss.convolve((~bad).astype('i'), kernel, mode='same')
        weights = np.clip(weights, 0, kernel.sum())

    # Weigthing and masking
    if normalize:
        if with_mask:
            bad = np.isclose(weights, 0)
            weights[bad] = 1
        cdata /= weights
    if with_mask:
        cdata[bad] = np.nan
    return cdata


def convolve(data, kernel, normalize=False):
    """N-dimensional convolution that take care of nans

    Parameters
    ----------
    data: xarray.DataArray
        Array to filter
    kernel: int, tuple, numpy.ndarray, xarray.DataArray
        Convolution kernel. See :func:`generate_kernel`.
    normalize: bool
        Divide the convolution product by the local sum weights.
        The result is then a weighted average.

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
        datac = convolve(data, kernel, normalize=True)
        fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(7, 3))
        kw = dict(vmin=data.min(), vmax=data.max())
        data.plot.pcolormesh(ax=ax0, **kw);
        @savefig api.filter.convolve.png
        datac.plot.pcolormesh(ax=ax1, **kw);

    """
    # Adapt the kernel to the data
    kernel = generate_kernel(kernel, data)

    # Numpy convolution
    datac = _convolve_(data, kernel.data, normalize)

    # Format
    return xr.DataArray(datac, coords=data.coords, attrs=data.attrs)


# def erode_mask(data, until=1, kernel=3):

#     if isinstance(until, int):
#         niter = until
#         mask = None
#     else:
#         mask = until
#         # if mask.shape != data.shape

#     for i in range(niter):
#         datac = xr_filternd(data, kernel)
#         data = xr.where(np.isnan(data), datac, data)
#     return data

