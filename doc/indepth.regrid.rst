.. _indepth.horizontal:

Horizontal interpolation and regridding
#######################################

Introduction
============

This guide explains how the horizontal interpolation and regridding tools work, what they
assume and how to choose between them. The :ref:`regridding tutorial
<sphx_glr_examples_plot_regrid_interp.py>` shows them at work on real data, with figures:
read it first for a walkthrough, and come back here for the rules behind it.

.. ipython:: python

    @suppress
    import warnings
    @suppress
    warnings.simplefilter("ignore")
    import numpy as np
    import xarray as xr
    import xoa
    from xoa import interp, regrid
    xoa.register_accessors()

Which tool for which job?
-------------------------

.. list-table::
    :header-rows: 1
    :widths: 28 36 36

    * - Tool
      - Destination
      - Methods
    * - :class:`xoa.regrid.Regridder`
      - Another structured 2D grid (1D or 2D coordinates)
      - ``bilinear``, ``bicubic``, ``conservative``
    * - :class:`xoa.interp.Interpolator`
      - Any points: a point, a transect, scattered positions or a grid
      - ``bilinear``, ``bicubic``
    * - ``da.xoa.regrid``, ``da.xoa.interp``
      - Same as above, in a single call
      - Same as above
    * - :func:`xoa.regrid.regrid1d`
      - Along a single dimension, typically the vertical
      - See :ref:`indepth.grids.regrid`
    * - :func:`xoa.interp.grid2loc`
      - Random positions, with depth and time
      - Linear only

The first three share the same numba kernels, which are available for plain numpy arrays in
:class:`xoa.core.interp.XYInterpolator` and :class:`xoa.core.regrid.XYRegridder`.
The classes of :mod:`xoa.regrid` and :mod:`xoa.interp` add the xarray layer: they find the
coordinates with :mod:`xoa.meta`, apply the kernels to all the variables of a dataset and
restore the dimensions, coordinates and attributes.

Interpolation or regridding?
============================

.. figure:: _static/interp-vs-regrid.png
    :width: 95%
    :alt: Interpolation goes from points to points, regridding from cells to cells

    Interpolation estimates values at points from the values at points, with the weights of
    the points around each destination point. Conservative regridding transfers the mean values
    over cells to the cells of another grid, with the overlap areas of the cells as weights.

Both words are used for moving data from a grid to another, but they do not describe the input
and the output in the same way, which explains the choices that follow.

.. list-table::
    :header-rows: 1
    :widths: 16 42 42

    * -
      - Interpolation
      - Regridding (conservative)
    * - Input
      - Values **at points**: the nodes of the grid, which are only known by their coordinates
      - Mean values **over cells**: the centers of the grid with their edges, that are
        computed from the centers and extrapolated at the ends of the grid
    * - Output
      - Values at any **points**: scattered, along a transect, or the points of a grid
      - Mean values over the **cells** of a structured destination grid
    * - Weights
      - Of the 4 (or 16) surrounding source points
      - Overlap areas of the destination cell with the source cells
    * - Keeps
      - The values: exact at the nodes and for linear fields
      - The integral: the total quantity is the same on both grids
    * - In xoa
      - :class:`xoa.interp.Interpolator`, and :class:`xoa.regrid.Regridder` with ``bilinear``
        or ``bicubic``, which interpolate at the points of the destination grid
      - :class:`xoa.regrid.Regridder` with ``conservative``

What it means in practice:

- Going to a **coarser** grid, interpolation samples the field at a few points and ignores what
  is between them, while regridding averages everything that falls in each destination cell.
  Use regridding for fluxes, concentrations and any quantity whose integral matters.
- Going to a **finer** grid, interpolation is smooth, and regridding is piecewise constant:
  it cannot invent variations inside a source cell.
- A point has no size, so interpolation does not care about the cells of the destination,
  whereas regridding needs cells and therefore a **structured 2D destination grid**:
  scattered points can only be interpolated.
- The weights of an interpolation depend on the positions of the points, and the ones of
  a regridding on the shapes of the cells, so that they also work for curvilinear cells.

Choosing a method
=================

The three methods differ by what they assume about the field, and by what they preserve.
They also differ by what they use around a target: the figure shows the points or cells that
are involved, and their weights, on a curvilinear grid.

.. figure:: _static/method-stencils.png
    :width: 100%
    :alt: Points and cells used by the bilinear, bicubic and conservative methods

    The points or cells used by each method for a target (orange), with their weights: the
    marker sizes and line widths of the first two are the weights, and the weights of the second
    one can be negative (blue). The indices of the grid give the cell that contains a point, so the
    cost does not depend on the number of source points. The destination cell of the conservative
    method is drawn as a rectangle, but cells may be any quadrilaterals.

.. list-table::
    :header-rows: 1
    :widths: 18 27 27 28

    * -
      - ``bilinear``
      - ``bicubic``
      - ``conservative``
    * - Uses
      - The 4 surrounding points
      - The 16 surrounding points
      - The overlap area of source and destination cells
    * - Smoothness
      - Continuous, with slope breaks
      - Continuous slope (Hermite)
      - Piecewise constant
    * - Order of accuracy
      - 2
      - 3
      - 1 (cell averages)
    * - Linear fields
      - Exact
      - Exact when the spacing of the grid is uniform, approximate otherwise since the
        tangents ignore the irregular spacing
      - Not exact: it is an average over each cell
    * - Conserves integrals
      - No
      - No
      - Yes, as long as the destination is covered
    * - Best for
      - Most fields, and the safest default
      - Smooth fields, where overshoots are acceptable
      - Fluxes and quantities that must be conserved, going to coarser grids

Aliases are accepted, like ``"linear"`` and ``"cubic"``, and the methods are listed in
:class:`xoa.regrid.xy_regrid_methods` and :class:`xoa.interp.xy_interp_methods`.
The ``conservative`` method is not available to :class:`~xoa.interp.Interpolator` since it
needs destination *cells*, not points.

The following example uses a field that is linear in longitude and latitude:

.. ipython:: python

    lon = xr.DataArray(
        np.linspace(0, 5, 11), dims="lon",
        attrs={"standard_name": "longitude", "units": "degrees_east"})
    lat = xr.DataArray(
        np.linspace(0, 4, 9), dims="lat",
        attrs={"standard_name": "latitude", "units": "degrees_north"})
    src = (2 * lon + lat).transpose("lat", "lon").rename("temp")
    src = src.assign_coords(lon=lon, lat=lat)
    dst = xr.Dataset(coords={
        "lon": ("lon", np.linspace(0, 5, 6), lon.attrs),
        "lat": ("lat", np.linspace(0, 4, 5), lat.attrs)})

    for method in "bilinear", "bicubic", "conservative":
        out = regrid.Regridder(src, dst, method).regrid(src)
        error = abs(out - (2 * out.lon + out.lat))
        print(method, int(out.isnull().sum()), float(error.max()))

The error of the conservative method is not a defect: the result is the mean value over
the destination cell, not the value at its center.

Comparison with a triangulation
-------------------------------

A common way to interpolate from a curvilinear grid is to forget that it is a grid and to
use a triangulation of its nodes, for instance with :func:`scipy.interpolate.griddata`
(``method="linear"``) or :class:`matplotlib.tri.LinearTriInterpolator`. It is worth knowing
how it differs from the methods above, which all take advantage of the structure of the grid.

**Principle.** The nodes are considered as a cloud of scattered points, which is
triangulated (Delaunay). For a target, the triangle that contains it gives **3 points**, and the
weights are its barycentric coordinates: they are positive inside the triangle, sum to one,
and reproduce linear fields exactly. The grid indices, the cells and the mask play no role.

**What it shares with the bilinear method.** Both are piecewise linear, exact for linear
fields and second order accurate, so that on a smooth field their errors are of the same
order.

**What differs.**

- *Shape of the interpolant.* A quadrilateral cell can be cut along either of its two diagonals,
  and the triangulation chooses from the geometry, so the value inside a cell depends on that
  choice and has visible diagonal artifacts. The bilinear method uses the whole cell
  and does not depend on any choice.
- *Edges and holes.* A triangulation covers the **convex hull** of the points: its long thin
  triangles fill the concavities of the grid, its corners and the gaps where nodes were
  removed, so that targets *outside* the grid, or over a masked area, get a value. The grid-aware
  methods return NaN there, since no cell contains the target.
- *Location and cost.* The triangulation is built from all the nodes, which costs a lot of
  time and memory for large grids, and has to be kept to avoid building it again. With a grid,
  the cell is found from the indices at a negligible cost, and the weights are
  small arrays that are shared and saved to files (see :ref:`indepth.regrid.weights`).
- *Higher orders and averages.* There is no bicubic counterpart, since there is no
  regular stencil of 16 points, and neither a conservative one, since cells are not used.
- *Missing values.* Removing masked nodes changes the triangulation, whereas ``skipna`` and
  ``na_thres`` renormalize the weights of the valid corners of the same cell.

Triangulation remains the right tool when the source really is **a cloud of scattered
points**, like observations: all the methods of this guide need a structured grid. It
is not part of :class:`~xoa.interp.Interpolator`, and :mod:`xoa.krig` offers an
optimal interpolation for scattered data.

Points near the edges of the source grid
----------------------------------------

Interpolation needs the cell that contains the point, and cells are closed:

- Destination points on the **first and last lines** of the source grid are inside, so that
  regridding to the grid itself gives back the data with the first two methods,
  and points that are outside are NaN on all sides: nothing is extrapolated. The
  small rounding errors of the coordinates are tolerated.
- ``bicubic`` needs one more cell around, since it uses 16 points, so that it returns NaN
  in the outer ring of cells, between the first and second lines and between the last two.
  It falls back to the bilinear interpolation of the 4 central points where some of the 16 are
  not valid.
- ``conservative`` only needs the destination cells to overlap the source ones. The
  weights of the cells that are partially covered are renormalized, so that a constant field
  stays constant up to the boundary.

Grids and coordinates
=====================

Longitudes and latitudes are found with :func:`xoa.coords.get_lon` and
:func:`xoa.coords.get_lat`, and may be 1D (rectilinear grids) or 2D (curvilinear grids), in
which case they are broadcast. :func:`xoa.grid.ds2grid_dict` builds the dictionary
that the core classes consume.

.. ipython:: python

    grid = xoa.grid.ds2grid_dict(src)
    grid["type"], grid["dims"], grid["lon"].shape

The type of the grid, given by :func:`xoa.core.grid.check_grid_type`, selects the search
algorithm:

- ``regular``: constant steps along both axes, so the cell is computed directly,
- ``rectangular``: longitudes only depend on the column and latitudes on the row,
  so the cell is found by searching each axis,
- ``curvilinear``: no assumption, so the closest point is searched and the
  neighbouring cells are tested.

Things to know:

- **Longitudes must be in [-180, 180] and latitudes in [-90, 90]**, for both grids and
  points, otherwise a :class:`ValueError` is raised. Convert 0-360 longitudes beforehand.
- Axes may increase or decrease along their dimension, like latitudes that go from north to south.
- Grids may cross the **dateline**, with longitudes that jump from 180 to -180, for all
  the methods and types of grids, except the ``bicubic`` method on rectangular grids. Cells that
  contain a pole, whose longitudes span more than 180°, are not supported by the
  ``conservative`` method.
- Curvilinear grids are searched around the closest node of each target, so cells must not be
  extremely sheared.
- On **staggered grids**, a dataset may have several longitude/latitude pairs, one for
  each location. Pass a data array to select its own location,
  since the pair is ambiguous in a dataset.
- Dimensions are free: the horizontal dimensions of the destination of a regridding are
  the ones of its coordinates, and the ones of an interpolation are the dimensions of
  the longitudes and latitudes of the points, or a new ``pts`` dimension when they are
  plain arrays (see :func:`xoa.coords.geo_merge`).
- The other dimensions, like time and depth, are untouched. In a dataset, variables that do
  not have the horizontal dimensions are copied as is.

Missing values and masks
========================

NaN values and masks are handled by the ``skipna`` and ``na_thres`` parameters, and the
``src_mask`` parameter, where ``True`` means valid.

- ``skipna=False`` (default): a single NaN among the points that are used gives NaN.
- ``skipna=True``: NaNs are ignored and the weights of the valid points are
  renormalized.
- ``na_thres`` is the fraction of the weights that may be missing for a value to be
  computed. At ``1.0``, a single valid point is enough, and at ``0.5`` the missing points
  must weigh less than half. Values close to 0 are strict.
- ``src_mask`` is turned into NaNs before the interpolation, **only when ``skipna`` is True**
  for ``bilinear`` and ``bicubic``, and always for ``conservative``.

Here are the rules on a single point in the middle of four corners, one of which is invalid:

.. ipython:: python

    lon3, lat3 = np.meshgrid([0.0, 1, 2], [0.0, 1, 2])
    data = np.arange(9.0).reshape(3, 3)
    data[1, 1] = np.nan
    mask = np.ones((3, 3), bool)
    mask[1, 1] = False
    from xoa.core.interp import XYInterpolator
    ip = XYInterpolator({"lon": lon3, "lat": lat3, "mask": mask}, [0.5], [0.5], "bilinear")
    for skipna, na_thres in (False, 1.0), (True, 1.0), (True, 0.5), (True, 0.1):
        print(skipna, na_thres, ip.interp(data, skipna=skipna, na_thres=na_thres))

The missing corner has a weight of 0.25, so the value is computed down to ``na_thres=0.25``.

Time
====

A regridding interpolates linearly in time with its ``dst_time`` parameter, or with the time of
the destination grid if it has one. Nothing is done if the source has no time dimension,
or if the time is a scalar coordinate.

The :meth:`xoa.interp.Interpolator.interp_with_time` method is the counterpart for points
that come with times, like an observing platform. It interpolates in space at the two
source times that frame each point and then in time. Points that are outside the time range
of the source are NaN:

.. ipython:: python

    times = xr.DataArray(
        np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]"), dims="time",
        attrs={"standard_name": "time"})
    series = (src + xr.DataArray([0.0, 1.0], dims="time")).transpose("time", "lat", "lon")
    series = series.assign_coords(time=times)
    points = np.array(["2020-01-01T12", "2020-01-03"], dtype="datetime64[ns]")
    interp.Interpolator(series, [1.0, 2.0], [1.0, 2.0]).interp_with_time(series, points)

.. _indepth.regrid.weights:

Weights
=======

The expensive part, locating the destination points in the source grid, is split from the
cheap one, applying the weights, and happens at most once.

Lazy computation
----------------

Weights are computed at the first regridding, not at the creation of the object, and are
reused by all the following calls, on any variable that has the same horizontal dimensions.
The count of the weights computed is independent of the number of variables of a dataset.

Sharing in memory
-----------------

Regridders and interpolators that have the same source and destination grids, mask,
method, parameters and dimension names share their weights, whatever the way they have been
created. This is why the following calls only compute them once, even if each call builds
a new object:

.. ipython:: python

    ds = xr.Dataset({"a": src, "b": 2 * src, "c": 3 * src})
    outs = [ds[name].xoa.regrid(dst) for name in "abc"]
    ds.xoa.regrid(dst)["c"].equals(outs[2])

The grids are recognised from their content, with :func:`xoa.misc.get_array_fingerprint`,
and not from the identity of the arrays. The 8 most recently used weights are kept, and
:func:`xoa.regrid.clear_weights_cache` and :func:`xoa.interp.clear_weights_cache` free them.
Weights hold arrays of the size of the destination (or of the number of links for the
conservative method), plus the source grid.

Files
-----

With the ``weights_file`` parameter, weights are saved to a netcdf file and reused in other
sessions. A single file serves all the grids, since each set of weights lives in its own
group, whose name is made of its kind, the method and a fingerprint of the grids:

.. ipython:: python

    import os, tempfile
    from xoa import weights
    path = os.path.join(tempfile.mkdtemp(), "weights.nc")
    for method in "bilinear", "conservative":
        regrid.Regridder(src, dst, method, weights_file=path).regrid(src)
    interp.Interpolator(src, [1.0, 2.0], [1.0, 2.0], weights_file=path).interp(src);
    for group in weights.list_groups(path):
        print(group)

To find the weights of a grid in a file, get the fingerprint of the grid and search the groups
of the file that use it, either as a source or a destination. The fingerprints are also available
from the objects, and stored in the attributes of the groups:

.. ipython:: python

    from xoa import grid
    fingerprint = grid.get_fingerprint(src)
    weights.find_groups(path, fingerprint)
    weights.find_groups(path, fingerprint, kind="regrid", method="conservative")
    regridder = regrid.Regridder(src, dst, "bilinear")
    regridder.src_fingerprint == fingerprint, regridder.weights_group
    weights.describe_groups(path)[0]

The rules are:

- The file is read at initialization if it has the group of these grids, and written
  after the first computation if not. Nothing is computed when the file already holds the weights.
- Weights cannot be loaded for other grids, even with the same sizes, since the group is
  found from the fingerprint, which is checked again once loaded. Calling
  :meth:`~xoa.regrid.Regridder.load_weights` on another grid raises a :class:`ValueError`
  that lists the groups of the file.
- Groups are appended and never modified. Do not write to the same file from several
  processes at the same time.
- Files that were written before the support of groups are still read, with a warning,
  but the grids cannot be checked, except their sizes and the method.

Cost and memory
===============

- The weights of ``bilinear`` and ``bicubic`` are cheap: 4 arrays of the size of the destination.
- The weights of ``conservative`` intersect cells, which is more expensive. Source cells are
  indexed, so that the time grows roughly with the number of cells that overlap and not with
  all the pairs: a few hundredths of a second for 120 x 120 source points and half a second
  for 300 x 300. It is still the one to keep in a weights file for large grids.
- Applying weights is parallel and fast. The data are copied at most once, to the layout
  of the kernels, and not at all for a single field without mask. The input is never modified.
- Kernels work in double precision, so that other types are converted.
- Dask arrays are loaded in memory by the kernels, variable by variable.

Use with numpy arrays
=====================

The core classes take dictionaries of arrays, with ``lon`` and ``lat`` keys and optional
``mask`` and ``type`` ones, and know nothing about xarray:

.. ipython:: python

    from xoa.core.regrid import XYRegridder
    lons, lats = np.meshgrid(np.linspace(0, 5, 11), np.linspace(0, 4, 9))
    dlons, dlats = np.meshgrid(np.linspace(1, 4, 4), np.linspace(1, 3, 3))
    regridder = XYRegridder({"lon": lons, "lat": lats}, {"lon": dlons, "lat": dlats}, "bilinear")
    regridder.regrid(2 * lons + lats).round(2)

See also
========

- The :ref:`tutorial <sphx_glr_examples_plot_regrid_interp.py>` for maps, the curvilinear case,
  time and masks on real data.
- :ref:`indepth.grids` for the 1D regridding, edges, resolutions and ``to_rect``.
- :ref:`indepth.plot` for the plotting of fields, grids and sections.
- :mod:`xoa.weights` for the format of the weights files.
