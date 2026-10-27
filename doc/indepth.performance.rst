.. _indepth.performance:

Performance with dask and numba
###############################

Introduction
============

Many computations work along a single dimension, like a vertical interpolation, a
mixed layer depth or a tidal analysis, on arrays of dimensions like ``(time, z, y, x)``.
This guide explains how xoa processes them to be fast and to use little memory,
with numpy and dask arrays, and how to choose the dask scheduler.

.. ipython:: python

    @suppress
    import warnings
    @suppress
    warnings.simplefilter("ignore")
    import numpy as np
    import xarray as xr
    from xoa import grid
    from xoa.core import num

The working dimension is kept in place
======================================

:func:`xarray.apply_ufunc` moves the working dimension, also called the core dimension,
to the end, which transposes the arrays and often copies them.
The numba kernels of xoa rather keep it in place: an array of shape
``(nt, nz, ny, nx)`` is viewed as ``(nouter, nz, ninner)`` with
``nouter = nt`` and ``ninner = ny * nx``, which is a reshape without a copy.
The kernels then loop in parallel on the ``nouter * ninner`` columns.

The other arrays, like the coordinates, may have only some of the dimensions.
:func:`xoa.core.num.broadcast_along_axis` views them as ``(mouter, n, minner)``,
where ``mouter`` and ``minner`` are 1 or the full sizes, so that a depth of dimensions
``(z, y, x)`` is not copied for each time step:

.. ipython:: python

    data = np.ones((3, 10, 4, 5))  # (t, z, y, x)
    depth = np.ones((10, 4, 5))  # (z, y, x)
    levels = np.ones((20, 1, 1))  # (zo,)
    views, outer, inner = num.broadcast_along_axis(data, depth, levels, axis=1)
    [view.shape for view in views]

At the xarray level, :func:`xoa.grid.map_along_dim` gives the arrays to such a function
with all the dimensions in the same order, and processes the dask arrays block by block:

.. ipython:: python

    def double(data, axis):
        return np.repeat(data, 2, axis=axis)

    da = xr.DataArray(np.ones((3, 4, 5)), dims=("time", "z", "x"))
    grid.map_along_dim(double, da, "z", out_size=8).dims

This is used by :func:`xoa.regrid.regrid1d`, :func:`xoa.regrid.extrap1d`,
:func:`xoa.interp.isoslice` and :func:`xoa.thermdyn.get_n2`.
The tidal functions of :mod:`xoa.tide` follow the same rule with the time dimension first.

Dask arrays
===========

The working dimension is the one along which the kernel computes, like ``z`` for a
vertical interpolation. It must be a single chunk, and the other dimensions keep their
chunks, so chunk your data along the other dimensions. We recommend one block per
time step:

.. code-block:: python

    ds = xr.open_dataset("model.nc", chunks={"time": 1})
    temp = xoa.regrid.regrid1d(ds.temp, levels)

If a single time step does not fit in memory, chunk ``y`` and ``x`` too,
like ``chunks={"time": 1, "y": 500, "x": 500}``, but never ``z`` here.

.. figure:: _static/chunks.png
    :width: 90%
    :alt: Blocks of one time step, or tiles of a time step, that keep the whole z dimension

    The working dimension ``z`` is never split: a block is a time step (left) or a tile of
    a time step (right).

The memory is then about the number of blocks processed at the same time,
times the size of an input block and of an output block.

Threads or processes?
---------------------

Use threads, which is the default scheduler of dask. They share the memory, so the
blocks are not copied, and the numba kernels release the GIL, so the threads
really run in parallel.
With processes, each worker imports numpy, numba and xarray again, and the blocks are
copied between the processes: in our tests, it was slower and used four times more memory.

Speed or memory
---------------

With the ``threads`` scheduler, ``num_workers`` is the number of dask threads, which is
the number of blocks processed at the same time, hence the memory. The number of numba
threads, set with :func:`numba.set_num_threads`, is the number of threads of the
``prange`` loops of the kernel, which is the parallelism within a block:

.. figure:: _static/dask-threads.png
    :width: 70%
    :alt: Dask threads take blocks, and numba threads compute inside each of them

    With ``num_workers=2`` and ``numba.set_num_threads(4)``: 2 dask threads take 2 blocks
    at the same time, and 4 numba threads compute in each of them, so 8 cpus are used.
    The other blocks wait for a free dask thread.

.. code-block:: python

    import dask
    import numba

    # Lowest memory: one block at a time, with all the numba threads inside
    with dask.config.set(scheduler="synchronous"):
        temp.to_netcdf("temp.nc")

    # A compromise: 2 blocks at a time, with 4 numba threads each
    numba.set_num_threads(4)
    with dask.config.set(scheduler="threads", num_workers=2):
        temp.to_netcdf("temp.nc")

    # Fastest: as many blocks as cpus, and the most memory
    temp.to_netcdf("temp.nc")

When the data are read from and written to files, these operations often take most
of the time: more dask threads then mostly speed up the input and output.

The numba threading layer
-------------------------

Several dask threads may call the parallel numba kernels at the same time.
The ``workqueue`` threading layer of numba does not support it, unlike the ``omp``
and ``tbb`` ones, which are used when OpenMP or TBB are available.
Check it with :func:`numba.threading_layer` after a first computation, and see
the `numba documentation <https://numba.readthedocs.io/en/stable/user/threading-layer.html>`_
to select one.

Some figures
============

Vertical interpolation of a ``(16, 40, 300, 400)`` array of 614 MB from sigma levels
to 50 fixed depths with :func:`xoa.regrid.regrid1d`, on 8 cpus, compared to the former
implementation with :func:`xarray.apply_ufunc`:

.. list-table::
    :header-rows: 1

    * - Mode
      - Speed up
      - Memory
    * - numpy
      - 1.4 to 1.6
      - halved: only the output is allocated
    * - dask, one block at a time
      - 1.9 to 2.1
      - same
    * - dask, 8 workers
      - 1.8 to 2.0
      - same

With depths that don't have the time dimension, :func:`xoa.interp.isoslice` is
5 times faster with numpy, since the depths are no longer broadcasted to the shape
of the data, which also saves the memory of a copy of the data.
