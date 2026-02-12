.. _indepth.grids:

Grid operations, regridding and sigma coordinates
##################################################

Introduction
============

The :mod:`xoa` library provides comprehensive tools for working with grids, performing
regridding operations, and handling terrain-following parametric vertical coordinates
(sigma coordinates). This guide covers three main modules:

- :mod:`xoa.grid`: Grid utilities for 1D to nD grid operations on a single grid
- :mod:`xoa.regrid`: Regridding utilities for operations between different grids
- :mod:`xoa.sigma`: Terrain-following parametric vertical coordinates following CF conventions

.. _indepth.grids.grid:

Grid operations with :mod:`xoa.grid`
====================================

The :mod:`xoa.grid` module provides utilities to get information or perform
operations on a grid.

.. note:: For operations **between different grids**, use :mod:`xoa.regrid` instead.

Basic grid operations
---------------------

The :mod:`xoa.grid` module offers various functions to manipulate grids:

**Getting grid centers and edges:**

.. ipython:: python

    import xarray as xr
    import numpy as np
    import xoa.grid as xgrid

    # Create a simple 1D coordinate
    x_edges = xr.DataArray(np.arange(0, 5), dims='x', name='x')

    # Get centers from edges
    x_centers = xgrid.get_centers(x_edges)
    print(x_centers.values)

    # Get edges from centers
    x_back = xgrid.get_edges(x_centers)
    print(x_back.values)

**Applying operations along dimensions:**

The :func:`~xoa.grid.apply_along_dim` function allows you to apply an operator
on data array or dataset dimensions, potentially changing the size of the array.

.. ipython:: python

    # Create sample dataset
    ds = xr.Dataset({
        'temp': (['x', 'y'], np.random.rand(4, 3)),
    }, coords={
        'x': np.arange(4),
        'y': np.arange(3)
    })

    # Apply mean operation along x dimension
    ds_mean = xgrid.apply_along_dim(ds, 'x', np.mean)


Staggered grids
---------------

The xoa library supports staggered grids commonly used in ocean and atmospheric models.
Grid location information can be encoded in variable names using the staggered grid
locator (see :ref:`indepth.meta` for more details on the :class:`~xoa.meta.SGLocator`).


.. _indepth.grids.regrid:

Regridding with :mod:`xoa.regrid`
==================================

The :mod:`xoa.regrid` module provides utilities for regridding data from one grid
to another. It supports various interpolation methods for both horizontal and
vertical regridding.

1D regridding
-------------

The :func:`~xoa.regrid.regrid1d` function performs 1D interpolation/regridding
with support for multiple methods:

Supported methods
^^^^^^^^^^^^^^^^^

.. ipython:: python

    from xoa.regrid import regrid1d_methods
    print(list(regrid1d_methods))

These include:

- **linear**: Linear interpolation (default)
- **nearest**: Nearest neighbor interpolation
- **cubic**: Cubic interpolation
- **hermit/hermitian**: Hermitian interpolation
- **cellave**: Cell-averaging or conservative regridding

Example usage
^^^^^^^^^^^^^

.. ipython:: python

    from xoa import regrid

    # Create source data
    x_old = xr.DataArray(np.arange(0, 10, 2.0), dims='x')
    data_old = xr.DataArray(np.sin(x_old), dims='x', coords={'x': x_old})

    # Create new grid
    x_new = xr.DataArray(np.arange(0, 9, 0.5), dims='x')

    # Regrid with linear interpolation
    data_new = regrid.regrid1d(data_old, x_new, method='linear')

Extrapolation modes
^^^^^^^^^^^^^^^^^^^

You can control extrapolation behavior:

.. ipython:: python

    from xoa.regrid import extrap_modes
    print(list(extrap_modes))

Available modes include:

- **no/none/false**: No extrapolation (default)
- **top/above/after**: Extrapolate toward the top
- **bottom/below**: Extrapolate toward the bottom
- **both/all/yes/true**: Extrapolate both directions


Horizontal regridding
---------------------

For 2D horizontal regridding, xoa provides the :func:`~xoa.regrid.grid2loc` function
to interpolate from a grid to specific locations.

Vertical regridding
-------------------

Vertical regridding is particularly important for ocean models. The :func:`~xoa.regrid.isoslice`
function can extract data on isobaric or isopycnal surfaces.


.. _indepth.grids.sigma:

Sigma coordinates with :mod:`xoa.sigma`
========================================

The :mod:`xoa.sigma` module handles terrain-following parametric vertical coordinates
according to the CF conventions for
`Parametric Vertical Coordinates <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord>`_.

What are sigma coordinates?
---------------------------

Sigma coordinates are terrain-following vertical coordinates commonly used in ocean
and atmospheric models. Instead of using fixed depth or pressure levels, sigma
coordinates follow the bathymetry (ocean models) or topography (atmospheric models),
which allows better resolution near boundaries.

Supported coordinate types
---------------------------

The module supports the following CF-compliant sigma coordinate types:

.. ipython:: python

    from xoa.sigma import SIGMA_COORDINATE_TYPES
    for coord_type in SIGMA_COORDINATE_TYPES:
        print(f"- {coord_type}")

These include:

- **atmosphere_sigma_coordinate**: For atmospheric models
- **ocean_sigma_coordinate**: Basic ocean sigma coordinates
- **ocean_s_coordinate**: Generalized ocean s-coordinates
- **ocean_s_coordinate_g1**: Song & Haidvogel s-coordinates (generic form 1)
- **ocean_s_coordinate_g2**: Shchepetkin s-coordinates (generic form 2)

Converting sigma to physical coordinates
-----------------------------------------

The main functionality is to decode sigma coordinates into physical depths (for ocean)
or pressures (for atmosphere) using the appropriate formula terms.

.. note:: The core sigma conversion functions use Numba's guvectorize decorator
    to create optimized universal functions (ufuncs). This provides:

    - Efficient computation through JIT compilation
    - Automatic broadcasting over horizontal dimensions
    - Compatibility with dask arrays for lazy evaluation
    - Seamless integration with :func:`xarray.apply_ufunc`

Ocean sigma coordinates
^^^^^^^^^^^^^^^^^^^^^^^^

For ocean models, sigma coordinates are converted to depths using formulas that
depend on:

- **sigma** (σ): The sigma level (typically from -1 to 0)
- **eta** (η): Sea surface height
- **depth**: Bathymetry (bottom depth)
- Additional parameters depending on the coordinate type (e.g., hc, theta_s, theta_b for s-coordinates)

.. ipython:: python

    # Example with CROCO model output
    import xoa

    # Open a sample with sigma coordinates
    ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.meridional.nc")

    # Check the sigma coordinate
    print(ds.s_rho.attrs.get('standard_name'))

    # Decode sigma to depths using the sigma module
    # The decode_cf_sigma function will automatically detect
    # the coordinate type and apply the correct formula
    from xoa.sigma import decode_cf_sigma
    depths = decode_cf_sigma(ds.s_rho, ds)

Atmosphere sigma coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For atmospheric models, sigma coordinates are converted to pressure levels:

.. code-block:: python

    # p(k) = ptop + sigma(k) * (ps - ptop)
    # where:
    # - p: pressure at level k
    # - ptop: pressure at top of model
    # - sigma: sigma coordinate value
    # - ps: surface pressure

Working with vertical sections
-------------------------------

Once sigma coordinates are decoded to physical depths, you can work with
vertical sections and perform operations like:

- Extracting data at specific depth levels
- Interpolating to regular depth levels using :mod:`xoa.regrid`
- Computing vertical derivatives
- Creating vertical transects

See the gallery example :ref:`sphx_glr_examples_plot_croco_section.py` for a practical
demonstration of working with sigma coordinates and vertical sections.

Formula terms mapping
---------------------

The module provides automatic mapping from formula terms (as defined in the CF
conventions) to the standard xoa CF names:

.. ipython:: python

    from xoa.sigma import FORMULA_TERMS_TO_CF_NAMES
    from pprint import pprint
    pprint(FORMULA_TERMS_TO_CF_NAMES)

This allows the module to find the required variables in your dataset regardless
of their actual names, as long as they follow CF conventions.

Example: Working with CROCO model output
=========================================

Here's a complete example showing how to work with grid operations, regridding,
and sigma coordinates using CROCO model output:

.. ipython:: python

    import xoa
    import matplotlib.pyplot as plt

    # Load CROCO sample data
    ds = xoa.open_data_sample("croco.south-africa.meridional.nc")

    # Decode the dataset (handles sigma coordinates automatically)
    ds_decoded = ds.xoa.decode()

    # The sigma coordinate has been converted to depths
    # You can now regrid to regular depth levels if needed
    from xoa import regrid

    # Define regular depth levels
    import numpy as np
    regular_depths = xr.DataArray(
        np.linspace(0, -100, 20),
        dims='depth',
        attrs={'units': 'm', 'positive': 'up'}
    )

    # Regrid temperature to regular depths
    # (this would require the actual depth coordinate from the decoded dataset)
    # temp_regular = regrid.regrid1d(ds_decoded.temp, regular_depths, dim='z')

See also
========

- :mod:`xoa.grid`: Grid utilities module
- :mod:`xoa.regrid`: Regridding module
- :mod:`xoa.sigma`: Sigma coordinates module
- :mod:`xoa.coords`: Coordinate utilities
- :ref:`examples`: Gallery of examples including grid and sigma operations
- :ref:`indepth.meta`: For more on CF metadata and grid location encoding
