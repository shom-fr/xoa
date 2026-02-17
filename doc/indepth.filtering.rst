.. _indepth.filtering:

Filtering and data processing
##############################

Introduction
============

The :mod:`xoa.filter` module provides comprehensive filtering and data processing
utilities for oceanographic and atmospheric data. It includes spatial filtering
with convolution kernels, specialized temporal filters for tidal signals, mask
erosion operations, and data decimation tools.

Key features:

- **Spatial filtering**: N-dimensional convolution with NaN handling
- **Temporal filtering**: Tidal filters (Demerliac, Godin) for time series
- **Mask operations**: Iterative mask erosion and coast filling
- **Data decimation**: Intelligent undersampling based on spatial proximity

.. _indepth.filtering.spatial:

Spatial filtering with convolution
===================================

The core of spatial filtering in xoa is the :func:`~xoa.filter.convolve` function,
which performs N-dimensional convolution while properly handling missing data (NaNs).

Basic convolution
-----------------

.. ipython:: python

    import xarray as xr
    import numpy as np
    import xoa.filter

    # Create sample data with some noise
    np.random.seed(42)
    data = xr.DataArray(
        np.random.normal(size=(50, 70)),
        dims=('y', 'x'),
        name='temperature'
    )

    # Add some missing data
    data.values[10:20, 10:20] = np.nan

    # Apply spatial smoothing with a simple box kernel
    smoothed = xoa.filter.convolve(data, kernel=5, normalize=True, na_thres=0.5)

The ``convolve`` function has important parameters:

- **normalize**: If True, divides by local sum of weights (creates weighted average)
- **na_thres**: Controls NaN contamination tolerance (0=strict, 1=permissive)

Smoothing shortcut
------------------

For convenience, :func:`~xoa.filter.smooth` is a shortcut that automatically
sets ``normalize=True``:

.. ipython:: python

    # Equivalent to convolve with normalize=True
    smoothed = xoa.filter.smooth(data, kernel=5, na_thres=0.5)

Window functions
----------------

The :func:`~xoa.filter.get_window_func` function provides access to various
window functions from NumPy and SciPy:

.. ipython:: python

    # Get a Gaussian window function
    gaussian_func = xoa.filter.get_window_func("gaussian", std=0.2)

    # Get a Hamming window
    hamming_func = xoa.filter.get_window_func("hamming")

    # Create custom window from array
    custom_func = xoa.filter.get_window_func([1, 2, 5, 2, 1])

Available window types include:

- **NumPy windows**: ``bartlett``, ``blackman``, ``hamming``, ``hanning``, ``kaiser``
- **SciPy windows**: ``gaussian``, ``tukey``, ``bohman``, ``parzen``, and many more
- **Custom arrays**: Provide your own weight array

Kernel generation
-----------------

Kernels can be specified in multiple ways:

**1. Simple integer (box kernel):**

.. ipython:: python

    # 5x5 box kernel
    smoothed = xoa.filter.smooth(data, kernel=5)

**2. Dictionary for anisotropic kernels:**

.. ipython:: python

    # Different sizes along different dimensions
    kernel_spec = {'x': 7, 'y': 3}
    smoothed = xoa.filter.smooth(data, kernel=kernel_spec)

**3. Dictionary with a window function:**

.. ipython:: python

    # Gaussian window along both dimensions
    kernel_spec = {'x': 9, 'y': 5}
    smoothed = xoa.filter.smooth(data, kernel=kernel_spec,
        kernel_kwargs=dict(window_func='gaussian'))

**4. Explicit arrays:**

.. ipython:: python

    # Custom 1D weights
    kernel_spec = {'x': [1, 2, 5, 2, 1], 'y': [1, 2, 1]}
    smoothed = xoa.filter.smooth(data, kernel=kernel_spec)

Isotropic kernels
^^^^^^^^^^^^^^^^^

For isotropic (radially symmetric) kernels, use :func:`~xoa.filter.generate_isotropic_kernel`:

.. ipython:: python

    # Create 2D isotropic Gaussian kernel
    iso_kernel = xoa.filter.generate_isotropic_kernel(
        shape=(15, 15),
        window_func='gaussian'
    )
    print(iso_kernel.shape)

Shapiro kernel
^^^^^^^^^^^^^^

The Shapiro kernel is a specialized filter kernel commonly used in numerical
models for smoothing:

.. ipython:: python

    # 2D Shapiro kernel
    shapiro = xoa.filter.shapiro_kernel(('y', 'x'))
    print(shapiro.values)

    # Apply Shapiro smoothing
    smoothed = xoa.filter.smooth(data, kernel=shapiro)


.. _indepth.filtering.temporal:

Temporal filtering for tidal signals
=====================================

Oceanographic time series often contain tidal signals that need to be filtered out.
The :mod:`xoa.filter` module provides specialized tidal filters.

Demerliac filter
----------------

The Demerliac filter is a widely-used tidal filter designed to remove tidal
components while preserving lower-frequency signals:

.. ipython:: python

    # Create hourly time series with tidal signal
    import pandas as pd
    time = pd.date_range('2020-01-01', periods=30*24, freq='h')

    # Simulate signal: trend + tidal component + noise
    t_hours = np.arange(len(time))
    signal = (
        0.5 * t_hours / 24 +  # trend
        0.3 * np.sin(2 * np.pi * t_hours / 12.42) +  # M2 tide
        0.1 * np.random.normal(size=len(time))  # noise
    )

    sea_level = xr.DataArray(
        signal,
        coords={'time': time},
        dims='time',
        name='sea_level'
    )

    # Apply Demerliac filter
    filtered = xoa.filter.demerliac(sea_level, na_thres=0.2)

The Demerliac filter uses pre-defined weights optimized for hourly data.
It automatically handles sub-hourly data by interpolating the weights.

Other tidal filters
-------------------

The :func:`~xoa.filter.tidal_filter` function provides access to multiple
tidal filter types:

.. ipython:: python

    # Godin filter (alternative tidal filter)
    godin_filtered = xoa.filter.tidal_filter(sea_level, 'godin')

    # Simple moving average (24-hour)
    mean_filtered = xoa.filter.tidal_filter(sea_level, 'mean')

Available filter types:

- **demerliac**: Demerliac tidal filter (default for :func:`~xoa.filter.demerliac`)
- **godin**: Godin tidal filter
- **mean**: Simple 24-hour moving average

Important notes for tidal filtering:

- Data must have a valid time coordinate
- Time step should be hourly or sub-hourly (sub-hourly is interpolated)
- Time step should be relatively constant (checked with ``dt_tol`` parameter)
- The ``na_thres`` parameter controls tolerance for missing data in the time series


.. _indepth.filtering.mask:

Mask operations
===============

Mask erosion and coast filling are important operations for preparing
ocean model output or observational data.

Eroding masks
-------------

The :func:`~xoa.filter.erode_mask` function iteratively fills masked values
using smoothing:

.. ipython:: python

    # Create data with a masked region
    data_with_mask = data.copy()
    data_with_mask.values[20:30, 30:40] = np.nan

    # Erode mask by 3 iterations
    eroded = xoa.filter.erode_mask(data_with_mask, until=3)

The erosion process:

1. Smooth the data (using Shapiro kernel by default)
2. Fill NaN values with smoothed values
3. Repeat for specified number of iterations

You can also erode until a specific mask is satisfied:

.. ipython:: python

    # Define a target mask (True = should remain masked)
    target_mask = xr.DataArray(
        np.zeros((50, 70), dtype=bool),
        dims=('y', 'x')
    )
    target_mask.values[25:27, 35:37] = True  # Keep small region masked

    # Erode until only target_mask regions remain masked
    eroded = xoa.filter.erode_mask(data_with_mask, until=target_mask)

Coast erosion
-------------

The :func:`~xoa.filter.erode_coast` function is specialized for horizontal
(geographic) dimensions and automatically identifies X and Y dimensions:

.. ipython:: python

    # Create data with land mask
    lon = xr.DataArray(np.linspace(-10, 0, 70), dims='lon')
    lat = xr.DataArray(np.linspace(40, 50, 50), dims='lat')

    sst = xr.DataArray(
        np.random.normal(15, 2, size=(50, 70)),
        coords={'lat': lat, 'lon': lon},
        dims=('lat', 'lon'),
        name='sst'
    )

    # Add coastal mask
    sst.values[:, :10] = np.nan  # "land" on the left

    # Fill coastal regions
    sst_filled = xoa.filter.erode_coast(sst, until=5)

The function automatically:

- Identifies longitude and latitude dimensions using CF conventions
- Uses Shapiro kernel on horizontal dimensions only
- Preserves vertical or temporal dimensions


.. _indepth.filtering.decimation:

Data decimation
===============

When working with very dense observational data (e.g., satellite altimetry,
drifter trajectories), it's often necessary to reduce the number of points
while preserving spatial coverage. The :func:`~xoa.filter.decimate` function
provides intelligent spatial undersampling.

Basic decimation
----------------

.. ipython:: python

    # Create dense random points
    np.random.seed(123)
    npts = 500
    lons = np.random.uniform(-20, -10, npts)
    lats = np.random.uniform(40, 50, npts)
    values = np.random.normal(15, 2, npts)

    dense_data = xr.Dataset({
        'temp': (['npts'], values)
    }, coords={
        'lon': (['npts'], lons),
        'lat': (['npts'], lats)
    })

    # Decimate to ~50km spacing
    sparse_data = xoa.filter.decimate(dense_data, radius=50e3, method='pick')

    print(f"Original points: {npts}")
    print(f"Decimated points: {sparse_data.sizes['npts']}")

Decimation methods
------------------

Two methods are available:

**1. Pick method** (method='pick'):
Simply selects points ensuring minimum spacing

.. ipython:: python

    decimated_pick = xoa.filter.decimate(dense_data, radius=50e3, method='pick')

**2. Average method** (method='average'):
At each selected point, averages all nearby points within a radius

.. ipython:: python

    # Average with smoothing factor
    decimated_avg = xoa.filter.decimate(
        dense_data,
        radius=50e3,
        method='average',
        smooth_factor=1.5  # Average over 75km radius
    )

The ``smooth_factor`` parameter controls the averaging radius:

- ``smooth_factor=0``: Equivalent to 'pick' method
- ``smooth_factor=1``: Average over same radius as spacing
- ``smooth_factor>1``: Larger averaging radius (smoother result)

Typical use cases
-----------------

**1. Preparing data for kriging:**

.. code-block:: python

    # Reduce dense satellite data before kriging
    satellite_decimated = xoa.filter.decimate(
        satellite_obs,
        radius=25e3,  # 25 km spacing
        method='average',
        smooth_factor=1.0
    )

**2. Visualization of dense trajectories:**

.. code-block:: python

    # Thin drifter trajectories for clearer plotting
    drifter_thinned = xoa.filter.decimate(
        drifter_positions,
        radius=10e3,  # 10 km spacing
        method='pick'
    )

**3. Reducing computational cost:**

.. code-block:: python

    # Subsample before expensive geostatistical processing
    data_subset = xoa.filter.decimate(
        observations,
        radius=100e3,  # 100 km spacing
        method='average',
        smooth_factor=2.0  # Smooth over 200 km
    )


Best practices and tips
=======================

Handling missing data
---------------------

The ``na_thres`` parameter is critical for controlling how NaNs propagate:

.. code-block:: python

    # Strict: output masked if any input is masked
    strict = xoa.filter.smooth(data, kernel=5, na_thres=0)

    # Moderate: output masked if >50% input is masked
    moderate = xoa.filter.smooth(data, kernel=5, na_thres=0.5)

    # Permissive: output masked only if all input is masked
    permissive = xoa.filter.smooth(data, kernel=5, na_thres=1.0)

Choosing kernel sizes
---------------------

General guidelines:

- **Smaller kernels** (3-5): Preserve features, less smoothing
- **Medium kernels** (7-15): Balance smoothing and feature preservation
- **Larger kernels** (>15): Heavy smoothing, removes fine structure

For anisotropic smoothing:

.. code-block:: python

    # Smooth more along one direction
    kernel = {'x': 15, 'y': 5}  # More smoothing in x-direction
    smoothed = xoa.filter.smooth(data, kernel=kernel)

Computational considerations
----------------------------

- Convolution is computationally intensive for large kernels
- Consider using FFT-based methods for very large kernels (not yet in xoa)
- For repeated filtering, pre-generate kernels:

.. code-block:: python

    # Generate kernel once
    kernel = xoa.filter.generate_kernel({'x': 11, 'y': 11}, data, window_func='gaussian')

    # Reuse for multiple datasets
    smoothed1 = xoa.filter.smooth(data1, kernel=kernel)
    smoothed2 = xoa.filter.smooth(data2, kernel=kernel)

Working with multi-dimensional data
------------------------------------

Filtering works seamlessly with multi-dimensional arrays:

.. code-block:: python

    # 4D data: time, depth, lat, lon
    kernel_4d = {
        'time': 3,      # Light temporal smoothing
        'depth': 1,     # No vertical smoothing
        'lat': 5,       # Spatial smoothing
        'lon': 5
    }
    smoothed_4d = xoa.filter.smooth(data_4d, kernel=kernel_4d)

See also
========

- :mod:`xoa.filter`: Complete filtering module reference
- :mod:`xoa.regrid`: For interpolation and regridding operations
- :mod:`scipy.signal`: Underlying signal processing functions
- :mod:`scipy.ndimage`: N-dimensional image processing functions
