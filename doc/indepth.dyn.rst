.. _indepth.dyn:

Dynamics diagnostics and polygon masks
######################################

Introduction
============

:mod:`xoa.dyn` computes the usual kinematic diagnostics of ocean currents from the velocities
and the sea surface height, and :func:`xoa.grid.get_polygon_mask` tells which cells of a
grid are inside polygons. Both are numba accelerated and keep the missing values.

.. list-table::
    :header-rows: 1
    :widths: 32 38 30

    * - Function
      - Result
      - Units
    * - :func:`xoa.dyn.get_relvort`
      - Relative vorticity :math:`\partial_x v - \partial_y u`
      - s-1
    * - :func:`xoa.dyn.get_div`
      - Horizontal divergence :math:`\partial_x u + \partial_y v`
      - s-1
    * - :func:`xoa.dyn.get_strain`
      - Normal :math:`\partial_x u - \partial_y v` and shear :math:`\partial_x v + \partial_y u` strains
      - s-1
    * - :func:`xoa.dyn.get_okuboweiss`
      - Squared strains minus squared vorticity
      - s-2
    * - :func:`xoa.dyn.get_coriolis`
      - Coriolis parameter :math:`2 \Omega \sin\varphi`
      - s-1
    * - :func:`xoa.dyn.get_geos`
      - Geostrophic currents from the sea surface height
      - m s-1

The tutorial :ref:`sphx_glr_tutorials_plot_croco_dynamics.py` applies them to the surface
currents of a CROCO simulation, and this guide explains the conventions and the pitfalls.

Two layers
==========

As elsewhere in xoa, the numpy routines are in a ``core`` module, :mod:`xoa.core.dyn`, whose
last two axes are Y and X and whose resolutions are in meters. The xarray functions of
:mod:`xoa.dyn` find the X and Y dimensions and the resolutions from the coordinates, and
accept extra dimensions like time, and dask arrays.
The diagnostics of the velocities are also methods of the ``xoa`` accessor of a dataset,
which finds the ``u`` and ``v`` variables with :mod:`xoa.meta`, and the geostrophic
currents are a method of the accessor of the sea surface height:

.. code-block:: python

    relvort = ds.xoa.get_relvort()
    ug, vg = ds.ssh.xoa.get_geos()

Conventions
===========

.. ipython:: python

    @suppress
    import warnings
    @suppress
    warnings.simplefilter("ignore")
    import numpy as np
    import xarray as xr
    import xoa.dyn

    lon = xr.DataArray(np.arange(30) * 0.05 + 5, dims="lon", attrs={"standard_name": "longitude"})
    lat = xr.DataArray(np.arange(20) * 0.05 + 40, dims="lat", attrs={"standard_name": "latitude"})
    # Solid body, counter-clockwise rotation of 1e-5 s-1
    omega = 1e-5
    x = (lon - lon.mean()) * 111e3 * np.cos(np.radians(40.5))
    y = (lat - lat.mean()) * 111e3
    u = (-omega * y + 0 * x).transpose("lat", "lon").rename("u")
    v = (omega * x + 0 * y).transpose("lat", "lon").rename("v")
    uv = xr.Dataset({"u": u, "v": v}, coords={"lon": lon, "lat": lat})
    relvort = xoa.dyn.get_relvort(uv.u, uv.v)
    float(relvort.mean()) / omega
    okubo = xoa.dyn.get_okuboweiss(uv.u, uv.v)
    float(okubo.mean()) / omega**2

- The relative vorticity is **positive for a counter-clockwise rotation**, whatever the
  hemisphere. Divided by the Coriolis parameter, it gives the Rossby number, that is
  positive for a cyclonic rotation, which is counter-clockwise in the northern hemisphere
  and clockwise in the southern one.
- The Okubo-Weiss parameter of a solid body rotation is :math:`-4\omega^2`: rotation dominates
  strain where it is negative, and strain dominates where it is positive. It is usually
  normalized by its standard deviation, and eddy cores are classically
  defined by thresholds like -0.2 times that value.
- The geostrophic currents are
  :math:`u_g = -\frac{g}{f} \partial_y \eta` and :math:`v_g = \frac{g}{f} \partial_x \eta`.
  They are missing where the Coriolis parameter is null, and are not meaningful close to
  the equator.
- Derivatives are centered finite differences, and one-sided at the edges of the grid,
  like :func:`numpy.gradient`.
  A value is missing when a point it depends on is missing, so land points
  erode the diagnostics of their neighbors.

Resolution of the grid
======================

The derivatives are divided by the resolution in meters, which is computed by
:func:`xoa.grid.get_dx_dy` from the longitudes and latitudes with the haversine formula.
It has the shape of the grid, since the distances between adjacent points are averaged
at the points, contrary to :func:`xoa.grid.get_resolution`, whose arrays are one point shorter.
You can pass your own ``dx`` and ``dy``, as scalars or arrays, to any function, like the
metrics of the model, which can be more accurate than the ones computed from the coordinates.

.. ipython:: python

    dx, dy = xoa.grid.get_dx_dy(uv.u)
    float(dx.mean()) * 1e-3, float(dy.mean()) * 1e-3

Pitfalls
========

**Collocated grids only.**
    The velocities must be on the same points, which are the ones the diagnostics are
    computed at. On an Arakawa C grid, like the CROCO or NEMO ones, ``u`` and ``v`` must
    first be interpolated to the same points, for instance with :meth:`xarray.DataArray.interp`
    or with :func:`xoa.grid.shift`. The tutorial shows how to do it.

**Land.**
    Mask the land before computing diagnostics: velocities that are zero on land give
    large spurious gradients along the coast, whereas masked values stay missing.

**Resolution.**
    The grid must be fine enough to resolve what is looked for: vorticity and strain
    are computed over two grid steps and are noisy and underestimated at coarse resolutions.
    Smoothing the currents first, for instance with :func:`xoa.filter.smooth`, may help.

**Latitude of the geostrophic currents.**
    The Coriolis parameter is computed from the latitudes of the sea surface height. Use
    the absolute dynamic topography or the sea surface height rather than the sea level
    anomaly to get the total currents, as an anomaly gives the anomaly of the currents.

Polygon masks
=============

:func:`xoa.grid.get_polygon_mask` tells which cells of a grid are inside one or several
polygons, which are given as arrays of ``(lon, lat)`` points, a list of them, or shapely
polygons and multi-polygons (their holes are ignored).
The cells are the ones delimited by the edges that are half way between the grid points.
It works with 1D or 2D coordinates and the result has the Y and X dimensions of the grid.

The ``threshold`` parameter is the minimal fraction of a cell that must be covered by a
polygon for the cell to be inside:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - ``threshold``
      - Selected cells
    * - ``None``
      - Those whose center is inside the polygon, which is the fastest option
    * - ``0``
      - All those that touch the polygon
    * - ``0.5`` (default)
      - Those that are at least half covered
    * - ``1``
      - Only those that are entirely inside

.. ipython:: python

    import xoa.grid
    da = xr.DataArray(
        np.zeros((4, 5)), dims=("lat", "lon"),
        coords={"lon": ("lon", np.arange(5.0), {"standard_name": "longitude"}),
                "lat": ("lat", np.arange(4.0), {"standard_name": "latitude"})},
    )
    polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    for threshold in None, 0.0, 0.5, 1.0:
        print(threshold, int(xoa.grid.get_polygon_mask(da, polygon, threshold).sum()))

- The covered fraction is the ratio of areas computed in longitude and latitude coordinates:
  cells that are cut in half by a polygon edge have exactly a fraction of one half.
- The union of several polygons is returned, so cells that are partly covered by two polygons
  are only selected if one of them covers enough of them: polygons are not merged before.
- A polygon may be concave and cross the dateline, but must not span more than 180° of
  longitude.
- The numba functions of :mod:`xoa.core.poly` are available for numpy arrays:
  :func:`~xoa.core.poly.point_in_polygon` and :func:`~xoa.core.poly.points_in_polygon`
  for points, and :func:`~xoa.core.poly.polygon_cell_fraction` and
  :func:`~xoa.core.poly.polygon_mask` for cells.
