.. _indepth.grids:

Grid operations, regridding and sigma coordinates
##################################################

Introduction
============

The :mod:`xoa` library provides comprehensive tools for working with grids, performing
regridding operations, and handling terrain-following parametric vertical coordinates
(sigma coordinates). This guide covers three main modules:

- :mod:`xoa.grid`: Grid utilities for 1D to nD grid operations on a single grid
- :mod:`xoa.regrid`: Regridding utilities for operations between different grids,
  with the 1D regridding detailed here and the horizontal one in :ref:`indepth.horizontal`
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
    x_centers = xgrid.get_centers(x_edges, "x")
    print(x_centers.values)

    # Get edges from centers
    x_back = xgrid.get_edges(x_centers, "x", mode="linear_extrap")
    print(x_back.values)

.. note:: :func:`~xoa.grid.pad`, :func:`~xoa.grid.get_centers`, :func:`~xoa.grid.get_edges`,
    :func:`~xoa.grid.shift`, :func:`~xoa.grid.diff` and :func:`~xoa.grid.gradient` run numba
    kernels from :mod:`xoa.core.grid` on single or double precision floats, with the ``"edge"``,
    ``"linear_extrap"``, ``"constant"`` and ``"wrap"`` pad modes. Other types and modes use xarray.
    Dask arrays are loaded in memory, so loop over chunks to bound memory usage.
    Longitudes are kept continuous across the dateline: the centers of 178° and -174° are -178°.

**Applying operations along dimensions:**

The :func:`~xoa.grid.apply_along_dim` function allows you to apply an xarray operator
on data array or dataset dimensions, potentially changing the size of the array.

.. ipython:: python

    # Create sample dataset
    ds = xr.Dataset({
        'temp': (['x', 'y'], np.random.rand(4, 3)),
    }, coords={
        'x': np.arange(4),
        'y': np.arange(3)
    })

    # Interpolate between grid points
    ds_mean = xgrid.apply_along_dim(ds, 'x', xgrid.get_centers)


Horizontal resolution and edges
-------------------------------

The :func:`~xoa.grid.get_resolution` function computes the distance in meters
between adjacent points along x and y, and :func:`~xoa.grid.get_median_resolution` gives
a single value in degrees. The :func:`~xoa.grid.get_edge_extents` function gives the
geographic extent of the strips along the edges of a grid, which follow it when it
is rotated or curvilinear:

.. ipython:: python

    lon = xr.DataArray(
        np.arange(0.0, 5.0), dims="lon",
        attrs={"standard_name": "longitude", "units": "degrees_east"})
    lat = xr.DataArray(
        np.arange(0.0, 4.0), dims="lat",
        attrs={"standard_name": "latitude", "units": "degrees_north"})
    ds_ll = xr.Dataset(coords={"lon": lon, "lat": lat})

    dx, dy = xgrid.get_resolution(ds_ll)
    dx.values[0]
    xgrid.get_median_resolution(ds_ll)
    xgrid.get_edge_extents(ds_ll, ["north", "west"], n_cells=2)

The :func:`~xoa.grid.to_rect` function converts 2D longitudes and latitudes to 1D axes
when the grid is not curvilinear, according to :func:`xoa.core.grid.check_grid_type`.
The :func:`xoa.plot.plot_grid` function draws the edges and centers of a grid, or its bathymetry or
resolution, on a map (see :ref:`indepth.plot`).
See the :ref:`sphx_glr_tutorials_grids_plot_grid_tools.py` tutorial for a complete tutorial.


Staggered grids
---------------

The xoa library supports staggered grids commonly used in ocean and atmospheric models.
Grid location information can be encoded in variable names using the staggered grid
locator (see :ref:`indepth.meta` for more details on the :class:`~xoa.meta.SGLocator`).


Neighbors with :mod:`xoa.geo`
-----------------------------

The :class:`xoa.geo.GeoKDTree` class finds the nearest points of a grid or of unstructured points,
or those within a distance, for target points given as an object or a ``(lon, lat)`` pair.
The tree is built once and can be queried several times, with :meth:`~xoa.geo.GeoKDTree.query`
for the k nearest neighbors and :meth:`~xoa.geo.GeoKDTree.query_radius` for those within a distance.
Both return the neighbors of each target (``output="index"``), or an array on the points of the tree
with the distance to the closest target and nan elsewhere (``"dist"``), or its boolean version (``"mask"``).
The :func:`xoa.geo.get_neighbors` function is a one-shot wrapper, and the ``xoa`` accessor
method of the same name caches the tree.

.. code-block:: python

    tree = GeoKDTree(ds)
    dist, idx = tree.query(obs, k=3, units="km")
    nearest = ds.isel(idx)  # pointwise selection
    dist = tree.query_radius(obs, 30, units="km", output="dist")  # nan beyond 30 km
    mask = tree.query_radius(obs, 30, units="km", output="mask")

Longitudes and latitudes are converted to cartesian coordinates on the sphere, so
neighbors are right at the poles and across the dateline, and distances are great circle ones
(:class:`xoa.core.geo.SphericalKDTree` is the numeric version, with flat arrays).
Missing neighbors, for instance beyond a maximal distance, have an infinite distance
and an index of -1, which must not be used to select data.
See the :ref:`sphx_glr_tutorials_grids_plot_neighbors.py` tutorial.

Distances and directions on the sphere are computed by :func:`xoa.geo.haversine`,
:func:`xoa.geo.bearing`, :func:`xoa.geo.cdist`, :func:`xoa.geo.pdist` and
:func:`xoa.geo.get_distances`. The bearing is counter-clockwise from the east, like the angle
of a vector ``(u, v)``. See the :ref:`sphx_glr_tutorials_grids_plot_distances.py` tutorial.


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

The :class:`~xoa.regrid.Regridder` and :class:`~xoa.interp.Interpolator` classes regrid
and interpolate horizontally, on regular, rectangular and curvilinear grids, with the
``bilinear``, ``bicubic`` and ``conservative`` methods. They have their own guide,
:ref:`indepth.horizontal`, and a :ref:`tutorial <sphx_glr_tutorials_regrid_plot_regrid_interp.py>`.

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
which allows better resolution near boundaries.
The tutorial :ref:`sphx_glr_tutorials_vertical_plot_sigma_levels.py` shows how the stretching
parameters distribute the levels over a slope.

Supported coordinate types
---------------------------

The module supports the following CF-compliant sigma coordinate types:

.. ipython:: python

    from xoa.sigma import SIGMA_COORDINATE_FUNCTIONS
    for coord_type in SIGMA_COORDINATE_FUNCTIONS:
        print(f"- {coord_type}")

These include:

- **atmosphere_sigma_coordinate**: For atmospheric models
- **ocean_sigma_coordinate**: Basic ocean sigma coordinates
- **ocean_s_coordinate**: Generalized ocean s-coordinates
- **ocean_s_coordinate_g1**: Song & Haidvogel s-coordinates (generic form 1)
- **ocean_s_coordinate_g2**: Shchepetkin s-coordinates (generic form 2)

Converting sigma to physical coordinates
-----------------------------------------

The main functionality is to decode sigma coordinates into physical heights ``z`` (for ocean)
or pressures (for atmosphere) using the appropriate formula terms.

.. note:: The core sigma conversion functions use Numba's guvectorize decorator
    to create optimized universal functions (ufuncs). This provides:

    - Efficient computation through JIT compilation
    - Automatic broadcasting over horizontal dimensions
    - Compatibility with dask arrays for lazy evaluation
    - Seamless integration with :func:`xarray.apply_ufunc`

Ocean sigma coordinates
^^^^^^^^^^^^^^^^^^^^^^^^

For ocean models, sigma coordinates are converted to ``z`` heights, positive up, using formulas that
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

    # Decode sigma to z heights, which are negative in the ocean, using the sigma module
    # The decode_sigma function will automatically detect
    # the coordinate type and apply the correct formula.
    # The croco specs name the heights z_rho and z_w, according to their vertical location.
    from xoa.sigma import decode_sigma
    xoa.meta.assign_meta_specs(ds, "croco")
    dsz = decode_sigma(ds)
    print(dsz.z_rho)

Heights are relative to the fixed datum of the formulas by default, so that the free
surface is at ``z = ssh``. Use ``ref="surface"`` to get them relative to the free surface:
see :ref:`indepth.vertical`.

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

Once sigma coordinates are decoded to physical heights, you can work with
vertical sections and perform operations like:

- Extracting data at specific depth levels
- Interpolating to regular depth levels using :mod:`xoa.regrid`
- Computing vertical derivatives
- Creating vertical transects

See the gallery tutorial :ref:`sphx_glr_tutorials_vertical_plot_croco_section.py` for a practical
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

    @suppress
    import xarray as xr
    import xoa
    import xoa.meta
    import xoa.regrid
    import numpy as np
    
    # Load CROCO sample data
    ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.meridional.nc")
    
    # Set internal croco decoding rules and decode s-coordinate and dataset names
    xoa.meta.set_meta_specs('croco')
    ds_decoded = ds.xoa.decode_sigma().xoa.decode()
    
    # The sigma coordinate has been converted to z heights
    # You can now regrid to regular z levels if needed
    
    # Define regular z levels
    regular_z = xr.DataArray(np.linspace(-100, 0, 20), dims='z', attrs={'units': 'm', 'positive': 'up'})
    
    # Regrid temperature to regular z levels
    temp_regular = xoa.regrid.regrid1d(ds_decoded.ptemp, regular_z, extrap="top")
    print(temp_regular[0, :, 0, 0])

See also
========

- :mod:`xoa.grid`: Grid utilities module
- :mod:`xoa.regrid`: Regridding module
- :mod:`xoa.sigma`: Sigma coordinates module
- :mod:`xoa.coords`: Coordinate utilities
- :ref:`tutorials`: Gallery of tutorials including grid and sigma operations
- :ref:`indepth.meta`: For more on CF metadata and grid location encoding
