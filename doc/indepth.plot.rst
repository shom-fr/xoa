.. _indepth.plot:

Plotting fields, grids and sections
###################################

Introduction
============

The :mod:`xoa.plot` module has functions that draw maps, grids, sections and series
from xarray objects, and finds what to draw with :mod:`xoa.meta`. The lower level
:mod:`xoa.core.plot` module draws from plain numpy arrays and matplotlib axes.

This guide explains how the functions find their inputs, what their options mean, and
how to use the low level routines. The tutorials show them at work, with figures:

- :ref:`sphx_glr_tutorials_plot_grid_tools.py`: grids, resolutions and edges,
- :ref:`sphx_glr_tutorials_plot_regrid_interp.py`: maps of regridded fields,
- :ref:`sphx_glr_tutorials_plot_croco_section.py`: sections.

.. ipython:: python

    @suppress
    import warnings
    @suppress
    warnings.simplefilter("ignore")
    @suppress
    import matplotlib
    @suppress
    matplotlib.use("Agg")
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import xoa
    from xoa import plot as xplot
    from xoa.core import plot as cplot

The main ``xoa`` accessors give access to the high level functions through the ``plot``
subaccessor, with the array or dataset as the first argument:
``da.xoa.plot.field()``, ``grid()``, ``section()``, ``stick()`` and ``taylor(ref)``.

Two layers
==========

.. list-table::
    :header-rows: 1
    :widths: 22 24 54

    * - High level (xarray)
      - Low level (numpy)
      - Role
    * - :func:`~xoa.plot.plot_field`
      - (xarray plotting)
      - A field on a map, with contour overlays
    * - :func:`~xoa.plot.plot_grid`
      - :func:`~xoa.core.plot.plot_mesh`
      - Edges and centers, bathymetry or resolution of a grid
    * - :func:`~xoa.plot.plot_section`
      - :func:`~xoa.core.plot.plot_depth_section`
      - A vertical section with a variable depth
    * - :func:`~xoa.plot.plot_stick`
      - :func:`~xoa.core.plot.plot_sticks`
      - A current time series as sticks
    * - :func:`~xoa.plot.plot_taylor`
      - :func:`~xoa.core.plot.plot_taylor`
      - A Taylor diagram of arrays against a reference
    * - :func:`~xoa.plot.add_colorbar`
      - :func:`~xoa.core.plot.add_colorbar`
      - A shrunk and labelled colorbar
    * - :func:`~xoa.plot.get_label`
      -
      - ``"Long name [units]"`` from the meta-data
    * - (re-exported)
      - :func:`~xoa.core.plot.create_base_map`, :func:`~xoa.core.plot.setup_map_axes`,
        :func:`~xoa.core.plot.add_land`
      - Decorated cartopy maps

The high level functions find the data and the labels, then delegate the drawing to the
low level ones, which makes the latter usable on any array, with a plain matplotlib axes
when no map is needed.

How inputs are found
====================

Names are never needed: variables and coordinates are identified from their names and
attributes with the specifications of :mod:`xoa.meta`.

.. list-table::
    :header-rows: 1
    :widths: 26 74

    * - Function
      - What is searched
    * - :func:`~xoa.plot.plot_field`
      - Longitude and latitude, with :func:`xoa.coords.get_lon` and :func:`xoa.coords.get_lat`
    * - :func:`~xoa.plot.plot_grid`
      - Longitude and latitude, and the ``bathy`` and ``mask`` variables of a dataset
    * - :func:`~xoa.plot.plot_section`
      - The vertical dimension, the depth (with its ``positive`` attribute),
        longitude and latitude
    * - :func:`~xoa.plot.plot_stick`
      - The ``u`` and ``v`` variables of a dataset, and the time
    * - overlays of :func:`~xoa.plot.plot_field`
      - Any generic name that is given as a string, like ``"bathy"``

The generic names are the ones of the current specifications, which can be tuned for your
own files with :func:`xoa.meta.set_meta_specs`. This is how to see what a function will
use:

.. ipython:: python

    from xoa.meta import get_meta_specs
    ds = xr.Dataset(
        {
            "h": (("y", "x"), np.full((3, 4), 100.0),
                  {"standard_name": "model_sea_floor_depth_below_geoid"}),
            "land": (("y", "x"), np.ones((3, 4)), {"standard_name": "land_binary_mask"}),
        },
        coords={
            "lon": ("x", np.arange(4.0), {"standard_name": "longitude"}),
            "lat": ("y", np.arange(3.0), {"standard_name": "latitude"}),
        },
    )
    specs = get_meta_specs(ds)
    specs.search(ds, "bathy").name, specs.search(ds, "mask").name
    xoa.coords.get_lon(ds).name

When a name is ambiguous, or not recognised, pass the arrays yourself: a data array
for an overlay, or an array for the horizontal axis of a section.

Labels
======

:func:`~xoa.plot.get_label` builds ``"Long name [units]"``: it takes ``long_name``, then
``standard_name`` and then the name of the array, capitalizes the first letter and appends
the units. Missing attributes are first completed from the meta specifications, so a bare
``temp`` array is labelled properly:

.. ipython:: python

    xplot.get_label(xr.DataArray([1.0], dims="x", name="temp"))
    xplot.get_label(xr.DataArray([1.0], dims="x", attrs={"long_name": "my field", "units": "m"}))
    xplot.get_label(xr.DataArray([1.0], dims="x", name="temp"), units=False)

Maps
====

Map functions need `cartopy <https://scitools.org.uk/cartopy>`_, which is only imported
when a map is drawn.

- ``transform`` is the coordinate system of the data and defaults to ``PlateCarree``.
  The projection of the map, which defaults to ``Mercator``, is the one of the axes.
- :func:`~xoa.plot.plot_field` and :func:`~xoa.plot.plot_grid` create the map when no axes is
  provided, from the extent of the data computed with :func:`xoa.geo.get_extent` (a
  ``margin`` can be added to the field). Their ``map_kw`` parameter holds the
  options of :func:`~xoa.core.plot.setup_map_axes` (``gridlines``,
  ``gridlines_labels_on``, ``land``, ``land_scale``, ``coastlines``, ``bbox``), plus ``figsize``
  and ``projection`` for the creation of the figure.
- On axes that you provide, only the decoration keys are applied, so that panels of
  multi-panel figures can be created with their own projection.
- A field must only have its horizontal dimensions: select the time and the level first.

Contour overlays
----------------

The ``overlay_contours`` parameter of :func:`~xoa.plot.plot_field` draws contours over the
field, for instance the coast or isobaths. ``True`` contours the field itself. A list gives
one dictionary per layer, with the parameters of :meth:`matplotlib.axes.Axes.contour` and a
``field`` key that is a data array, which may be on another grid, or a generic name that is
searched in the dataset given by ``ds``:

.. ipython:: python

    import cmocean
    from xoa.core.grid import create_rotated_grid
    rgrid = create_rotated_grid(41, 31, -4.0, 47.5, 20.0, 4.0, 3.0)
    lon2d, lat2d = rgrid["lon"], rgrid["lat"]
    depth = 100 + 1500 * (lon2d - lon2d.min()) / np.ptp(lon2d)
    ds = xr.Dataset(
        {
            "temp": (("y", "x"), 12 + 3 * np.sin(lon2d) * np.cos(lat2d),
                     {"standard_name": "sea_water_temperature", "units": "degC"}),
            "h": (("y", "x"), depth, {"standard_name": "model_sea_floor_depth_below_geoid"}),
        },
        coords={
            "lon": (("y", "x"), lon2d, {"standard_name": "longitude"}),
            "lat": (("y", "x"), lat2d, {"standard_name": "latitude"}),
        },
    )

    @savefig indepth.plot.overlay.png width=5in
    xplot.plot_field(
        ds.temp,
        cmap=cmocean.cm.thermal,
        overlay_contours=[dict(field="bathy", levels=[200, 1000], colors="0.4", linestyles="--")],
        ds=ds,
    )

Colorbars
=========

All the colorbars of the module are shrunk (``shrink=0.7``, :data:`xoa.core.plot.CBAR_SHRINK`)
and labelled with :func:`~xoa.plot.get_label`, so that they do not dwarf the maps.

- The functions that make a colorbar accept ``cbar_kwargs`` (:func:`~xoa.plot.plot_field`,
  :func:`~xoa.plot.plot_section`) or ``colorbar_kwargs`` (:func:`~xoa.plot.plot_ts`), which
  override the defaults, and ``add_colorbar=False`` to skip it.
- For several panels, skip the colorbars and add a single shared one with
  :func:`~xoa.plot.add_colorbar`, which takes the axes and the array that gives the label:

.. ipython:: python

    import cartopy.crs as ccrs
    fields = [ds.temp + d for d in (-2, 0, 2)]

    fig, axes = plt.subplots(
        1, 3, figsize=(11, 3.6), subplot_kw={"projection": ccrs.Mercator()},
        constrained_layout=True,
    )
    for ax, da in zip(axes, fields):
        mappable = xplot.plot_field(da, ax=ax, add_colorbar=False, vmin=8, vmax=18,
                                  cmap=cmocean.cm.thermal)
    @savefig indepth.plot.colorbar.png width=6in
    xplot.add_colorbar(mappable, axes, da=fields[0])

The adaptive grid stride
========================

:func:`~xoa.plot.plot_grid` draws the edges and the centers of the cells. Edges are computed
from the centers by :func:`xoa.core.grid.centers2edges`, and are extrapolated at the ends.
Drawing every line of a large grid would only give a solid color, so the grid is under-sampled.

The ``stride`` is either ``"auto"`` (default), an integer or a ``(y, x)`` tuple:

- In the ``"auto"`` mode, the spacing of adjacent centers is **measured on the screen**, once
  projected on the axes, and the stride is the smallest one that leaves ``min_spacing``
  pixels (12 by default) between the lines drawn. It follows the size of the figure, the
  zoom, the projection and the rotation of the grid, and it is independent for each direction.
  The spacing is measured on a bounded number of lines, so that it is cheap for any grid.
- The first and last edges are always drawn.
- When the grid is under-sampled, the centers that are drawn are the ones of the
  coarse cells that are delimited by the lines, and not a subset of the true centers,
  that would sit on the lines.

The low level :func:`~xoa.core.plot.get_strides` does the measure. On a plain axes whose data
coordinates are pixels, the rule is easy to follow:

.. ipython:: python

    lon, lat = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # 5 pixels between points: 3 points are needed to get at least 12 pixels
    cplot.get_strides("auto", lon, lat, ax, min_spacing=12)
    cplot.get_strides("auto", lon, lat, ax, min_spacing=5)
    cplot.get_strides((2, 4), lon, lat)

The same grid is drawn with :func:`~xoa.plot.plot_grid`, with a stride that suits the figure:

.. ipython:: python

    @savefig indepth.plot.grid.png width=4in
    xplot.plot_grid(ds.temp, stride="auto", cmap=cmocean.cm.thermal)

Other options of :func:`~xoa.plot.plot_grid`:

- ``kind``: ``"mesh"`` (edges and centers, the default), ``"bathy"`` (the bathymetry found in a
  dataset, with the land masked, and the mesh if there is none) or ``"resolution"`` (the
  :func:`~xoa.core.grid.compute_center_resolution`, in km).
- ``edges`` and ``centers`` switch the lines and points on or off. They are drawn by
  default with the mesh and not with the fields, but can be added over them.
- ``strips``, with ``n_cells``, outlines the strips along the edges that
  :func:`xoa.grid.get_edge_extents` returns.
- ``edge_kw`` and ``center_kw`` style the lines and the points.
- On a staggered dataset, pass a data array, since the longitudes and latitudes are
  ambiguous. The ``"bathy"`` kind uses the grid of the bathymetry.

Sections
========

:func:`~xoa.plot.plot_section` draws an array that has a vertical dimension and a single
horizontal one. The depth may be 1D or vary along the section, as with terrain-following
coordinates; when no depth coordinate is found, the vertical coordinate is used, so decode sigma
coordinates first (see :ref:`indepth.grids.sigma`). The vertical axis is inverted when depths
are positive down.

The ``x`` parameter selects the horizontal axis: ``None`` picks longitude or latitude from
the one with the largest extent, ``"lon"`` and ``"lat"`` force one, ``"distance"`` is the
distance in km along the section, and any array that broadcasts to the horizontal
dimension is accepted.

.. ipython:: python

    depths = xr.DataArray(
        np.linspace(0, 200, 30), dims="z", attrs={"standard_name": "ocean_depth", "positive": "down", "units": "m"}
    )
    lons1d = xr.DataArray(
        np.linspace(-6, -2, 20), dims="x", attrs={"standard_name": "longitude", "units": "degrees_east"}
    )
    sec = xr.DataArray(
        20 - depths.values[:, None] / 20 * (1 + 0.5 * np.sin(lons1d.values[None])),
        dims=("z", "x"),
        coords={"depth": depths, "lon": lons1d},
        attrs={"standard_name": "sea_water_temperature", "units": "degC"},
    )

    fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    @savefig indepth.plot.section.png width=5in
    xplot.plot_section(sec, ax=ax, cmap=cmocean.cm.thermal)

Sticks
======

:func:`~xoa.plot.plot_stick` shows a time series of currents, as sticks that start from a
line, are oriented along the current and are as long as its speed. This is a compact way to
show the rotation of tidal currents. Pass ``u`` and ``v``, or a dataset that holds them,
and use ``scale`` to change the lengths (the smaller, the longer).

Taylor diagrams
===============

:func:`~xoa.plot.plot_taylor` summarizes how well arrays match a reference: the angle is the
correlation, the radius is the standard deviation and the distance to the reference
point is the centered root mean square difference (dashed contours). The ``ref`` argument
is mandatory.

- ``dim`` lists the dimensions where the statistics are computed. By default, all of them, which
  gives a single point per variable. The other dimensions create the points, labelled
  with their coordinates, and each variable of a dataset adds its own points.
- ``normalize=True`` divides the standard deviations by the one of the reference, which is
  then at 1. It is needed when the points do not share the same reference.
- ``labels`` are listed in the legend. With ``values``, the markers are colored instead and
  a colorbar is added, and the labels are written next to the points.
- ``markers`` is one marker, or one per point (cycled).
- The diagram is a quarter of circle, and a half circle if some correlations are negative.

The tuning parameters are the ones of :class:`~xoa.core.plot.TaylorDiagram`: ``rmax``,
``rms_levels``, ``corr_ticks``, ``grid_kwargs``, ``contour_kwargs`` and ``ref_kwargs``.
The returned diagram can receive more points with
:meth:`~xoa.core.plot.TaylorDiagram.add_points`, using the ``diagram`` argument.

.. ipython:: python

    rng = np.random.default_rng(0)
    obs = xr.DataArray(rng.normal(size=200), dims="time", name="obs")
    models = xr.Dataset({
        name: obs * a + rng.normal(scale=b, size=200)
        for name, a, b in [("m1", 1.0, 0.3), ("m2", 0.7, 0.6), ("m3", 1.3, 0.9)]
    })

    @savefig indepth.plot.taylor.png width=4in
    xplot.plot_taylor(models, obs, normalize=True, markers=["o", "s", "^"])

Without a reference array, already normalized statistics are drawn with the low level
:func:`~xoa.core.plot.plot_taylor`, whose ``ref_std`` is 1 by default:

.. ipython:: python

    @savefig indepth.plot.taylor_low.png width=4in
    cplot.plot_taylor([0.9, 1.2, 0.7], [0.95, 0.8, -0.3], labels=["a", "b", "c"])

The statistics themselves come from :func:`xoa.core.stats.taylor_stats`.

Low level routines
==================

The routines of :mod:`xoa.core.plot` need no xarray and no map. The mesh of a grid is
computed on its own, which is useful to check what will be drawn:

.. ipython:: python

    lons, lats = np.meshgrid(np.arange(4.0), np.arange(3.0))
    segments, clon, clat = cplot.get_mesh(lons, lats)
    len(segments), segments[0]
    segments, clon, clat = cplot.get_mesh(lons, lats, stride=(2, 2))
    len(segments), clon

and then drawn on any axes:

.. ipython:: python

    fig, ax = plt.subplots(figsize=(3, 2.5))
    cplot.plot_mesh(ax, lons, lats, stride=1)
    @savefig indepth.plot.mesh.png width=3in
    ax.set_aspect("equal")

The other routines are the same: :func:`~xoa.core.plot.plot_depth_section` takes the 2D arrays
of the horizontal coordinates, depths and values, and :func:`~xoa.core.plot.plot_sticks` takes
the positions and the components.
