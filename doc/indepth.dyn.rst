.. _indepth.dyn:

Dynamics diagnostics and polygon masks
######################################

Introduction
============

:mod:`xoa.dyn` computes the usual kinematic diagnostics of ocean currents from the velocities
and the sea surface height, and :func:`xoa.mask.get_polygon_mask` tells which cells of a
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
    * - :func:`xoa.dyn.get_rossby`
      - Rossby number, which is the relative vorticity over the Coriolis parameter
      - 1
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
    * - :func:`xoa.dyn.get_cyclogeos`
      - Cyclogeostrophic currents from the sea surface height
      - m s-1
    * - :class:`xoa.dyn.VerticalModes`
      - Vertical modes of the water columns, their speeds and deformation radii, and the
        decomposition of profiles on them
      - 1, m s-1, m

The tutorials :ref:`sphx_glr_tutorials_plot_gulfstream_dynamics.py` and
:ref:`sphx_glr_tutorials_plot_croco_rossby.py` apply them to the surface currents of a GLORYS
reanalysis of the Gulf Stream, and to the sigma levels of a CROCO simulation.
This guide explains the conventions and the pitfalls.

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

Levels with a variable z
========================

On sigma or hybrid vertical coordinates, the levels follow the topography, so the depth of
a level varies in space. Horizontal derivatives that are taken along the levels then
mix in the vertical shear of the currents, since

.. math::

    \partial_x u|_z = \partial_x u|_s - \partial_x z|_s \, \partial_z u

where :math:`|_s` stands for the derivative along the level and :math:`|_z` for the one at
constant depth, which is the one that is wanted. The error is large where the levels are
tilted and the currents are sheared, typically over shelf breaks and continental slopes,
and it is invisible on flat levels.

All the diagnostics that need horizontal derivatives accept a ``z`` parameter:
:func:`~xoa.dyn.get_relvort`, :func:`~xoa.dyn.get_rossby`, :func:`~xoa.dyn.get_div`,
:func:`~xoa.dyn.get_strain` and :func:`~xoa.dyn.get_okuboweiss`.
It is the vertical coordinate of each level, with the same dimensions as the velocities,
possibly without the extra ones like time. Its sign convention does not matter, since both
:math:`\partial_x z` and :math:`\partial_z` change sign with it.

.. code-block:: python

    rossby = xoa.dyn.get_rossby(ds.u, ds.v, z=ds.depth)
    rossby = ds.xoa.get_rossby(z=ds.depth)  # with the accessor

- The results are still on the model levels, which have different depths at different points.
  Interpolate them to regular depths afterwards with :func:`xoa.regrid.regrid1d`.
- The vertical derivative is centered, and one-sided at the first and last levels.
- The velocities must be at the same points, like for 2D arrays. Staggered grids must be
  handled first, for instance with
  `pycroco <https://github.com/shom-fr/pycroco>`_ for CROCO outputs.
- The tutorial :ref:`sphx_glr_tutorials_plot_croco_rossby.py` validates the Rossby number
  at constant depth against the one computed from currents that the model interpolated
  to regular depths.

Cyclogeostrophic currents
=========================

The geostrophic currents ignore the advection terms of the momentum equations, which are
significant where the flow is strongly curved, like in eddies and meanders. The
cyclogeostrophic currents :math:`\mathbf{u}` satisfy

.. math::

    f \mathbf{k} \times \mathbf{u} + (\mathbf{u} \cdot \nabla) \mathbf{u}
    = f \mathbf{k} \times \mathbf{u}_g

for the same sea surface height as the geostrophic currents :math:`\mathbf{u}_g`.
:func:`xoa.dyn.get_cyclogeos` solves it with an iterative method like the one of
Penven et al. (2014), see :ref:`references`, that starts from the geostrophic currents
and corrects them with the advection terms of the previous iteration until they stop changing:

.. math::

    u_{n+1} = u_g - \frac{u_n \partial_x v_n + v_n \partial_y v_n}{f}, \quad
    v_{n+1} = v_g + \frac{u_n \partial_x u_n + v_n \partial_y u_n}{f}

.. code-block:: python

    ucg, vcg = xoa.dyn.get_cyclogeos(ssh)
    ucg, vcg = ssh.xoa.get_cyclogeos()  # with the accessor

- These are the four terms of the complete advection, which reproduce the gradient wind
  balance of an axisymmetric vortex. With ``terms="penven"``, only the two terms
  :math:`v \partial_y v` and :math:`u \partial_x u` are kept, as in the formulation that is
  usually taught, which makes a smaller correction since the dropped cross terms
  carry the curvature of the flow in an eddy.
  Cyclones are slower than what geostrophy says, and anticyclones are faster.
- The iterations amplify the shortest scales when the currents are strong compared to the
  Coriolis parameter times the resolution, which makes the complete advection diverge,
  even where a solution exists, as soon as the resolution is fine enough.
  The ``smooth`` parameter damps these scales by smoothing the advection terms at each
  iteration, which is the default with the complete advection.
- Some strong anticyclones have no solution at all, like when the Rossby number of the
  geostrophic currents gets below about -1/2. The currents are then missing where the
  iterations diverge or do not converge in ``niter`` iterations, to the tolerance ``tol``.
- The currents are also missing where the geostrophic ones or one of their neighbors are,
  and where the Coriolis parameter is smaller than ``fmin``, like close to the equator.
- The tutorial :ref:`sphx_glr_tutorials_plot_gulfstream_dynamics.py` compares them with the
  currents of a reanalysis.

Vertical modes
==============

The quasi-geostrophic vertical modes are the solutions :math:`F_n(z)` of the problem

.. math::

    -\frac{d}{dz} \left( \frac{1}{N^2} \frac{dF}{dz} \right) = \frac{1}{c^2} F

for the stratification :math:`N^2(z)` of a water column, with :math:`dF/dz = 0` at the bottom,
and either :math:`dF/dz = 0` (rigid lid) or :math:`g \, dF/dz + N^2 F = 0` (free surface)
at the surface. The eigenvalues give the speeds :math:`c_n` of the modes, and the
deformation radii :math:`R_n = c_n / |f|`. The first mode is the barotropic one, that is
constant along the vertical, and the next ones are the baroclinic modes, that have more and
more nodes and that are slower and slower. They are orthogonal, so that profiles can be split
into the amplitudes of the modes, which is a compact description of the vertical structure
of currents, and of the partition of their energy. See the tutorial
:ref:`sphx_glr_tutorials_plot_vertical_modes.py`.

The :class:`xoa.dyn.VerticalModes` class computes the modes of all the water columns at once
from the squared Brunt-Vaisala frequency, that :func:`xoa.thermdyn.get_n2` computes from the
temperature and the salinity, and decomposes and recomposes profiles:

.. code-block:: python

    n2 = xoa.thermdyn.get_n2(ds.temp, ds.sal)
    vm = xoa.dyn.VerticalModes(n2, nmodes=5)
    amps = vm.decomp(ds.u)                 # amplitudes of the modes
    part = vm.recomp(amps, modes=[1, 2])   # contribution of the first baroclinic modes
    vm.rd.isel(mode=1)                     # first deformation radius, in meters

- The object is a class since the modes are computed once, and then used to decompose and
  recompose any number of fields. They are computed by the numba routines of
  :mod:`xoa.core.modes`, in parallel over the columns, and everything works with extra
  dimensions, like time and horizontal ones, and with dask, through :func:`xarray.apply_ufunc`.
  The problem is tridiagonal: the few lowest modes are found with bisection and inverse
  iteration, so that the cost is proportional to the number of levels and not to its cube,
  which makes it practical for models with hundreds of levels and millions of columns.
- The normalization is such that the mean of the squares of the modes over the water column
  is one, so that the amplitude of the barotropic mode is the vertical mean of the profile
  and that the mean of the squares of a profile is the sum of the squares of its amplitudes.
  Modes are positive at the surface.
- The rigid lid is the default: the barotropic mode then has no speed, and the speeds
  of the baroclinic modes are hardly modified by the free surface (``bc="free"``),
  that gives the barotropic mode the speed :math:`\sqrt{gH}`.
- The levels may vary in space, like with sigma coordinates, and the levels that are missing,
  like below the bottom, are not used, so that each column has the modes of its own depth.
  The profiles to decompose must be defined at the levels of the stratification.
- The modes are the ones of the water column that is described by the data: if it stops at
  2000 m in a deeper ocean, the barotropic mode is the mean of the upper 2000 m.
- The stratification is clipped to a small positive value, ``n2min``, in the unstable and mixed
  layers. In a column that is entirely mixed, there is no baroclinic mode and its speed vanishes.
- The deformation radii are missing where the Coriolis parameter is smaller than ``fmin``, which
  is the case close to the equator, where the equatorial beta-plane radius is the appropriate one.
- To focus on a layer, like the upper ocean, use ``zrange=(zmin, zmax)``, in the convention
  of the vertical coordinate: the modes of the layer have no flux through its limits, and only
  the layer is decomposed. The modes of the thermocline are faster to oscillate, and have
  smaller deformation radii than the ones of the whole water column.

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
    or with :func:`xoa.grid.shift`. Staggered grids are not handled by :mod:`xoa.dyn`.

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

:func:`xoa.mask.get_polygon_mask` tells which cells of a grid are inside one or several
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

    import xoa.mask
    da = xr.DataArray(
        np.zeros((4, 5)), dims=("lat", "lon"),
        coords={"lon": ("lon", np.arange(5.0), {"standard_name": "longitude"}),
                "lat": ("lat", np.arange(4.0), {"standard_name": "latitude"})},
    )
    polygon = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]
    for threshold in None, 0.0, 0.5, 1.0:
        print(threshold, int(xoa.mask.get_polygon_mask(da, polygon, threshold).sum()))

- The covered fraction is the ratio of areas computed in longitude and latitude coordinates:
  cells that are cut in half by a polygon edge have exactly a fraction of one half.
- The union of several polygons is returned, so cells that are partly covered by two polygons
  are only selected if one of them covers enough of them: polygons are not merged before.
- A polygon may be concave and cross the dateline, but must not span more than 180° of
  longitude.
- The numba functions are available for numpy arrays:
  :func:`xoa.core.poly.point_in_polygon` and :func:`xoa.core.poly.points_in_polygon`
  for points, and :func:`xoa.core.mask.polygon_cell_fraction` and
  :func:`xoa.core.mask.polygon_mask` for cells.
