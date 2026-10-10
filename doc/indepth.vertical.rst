.. _indepth.vertical:

Vertical coordinates: z and depth
#################################

Introduction
============

Ocean data come with two opposite conventions for the vertical coordinate, and mixing them
silently flips a section or shifts a mixed layer depth.
:mod:`xoa` names and handles them separately:

.. list-table::
    :header-rows: 1

    * -
      - ``z``
      - ``depth``
    * - Sign in the ocean
      - negative
      - positive
    * - Increases towards
      - the surface
      - the bottom
    * - ``positive`` attribute
      - ``"up"``
      - ``"down"``
    * - ``standard_name``
      - ``altitude``
      - ``ocean_layer_depth``
    * - Produced by
      - :func:`xoa.sigma.decode_sigma`, :func:`xoa.grid.dz2depth` with ``positive="up"``
      - :func:`xoa.grid.dz2depth` with ``positive="down"``

``z`` is the same thing as ``depth`` with the opposite sign: ``z = -depth``.
The ``altitude`` coordinate is something else: it belongs to the atmosphere, in meters or
hPa.

This guide explains how the two are found, converted and produced, and what to watch for.
The tutorial :ref:`sphx_glr_tutorials_plot_croco_section.py` shows them at work on a
CROCO section.

.. ipython:: python

    @suppress
    import warnings
    @suppress
    warnings.simplefilter("ignore")
    import numpy as np
    import xarray as xr
    import xoa
    from xoa import coords

Sign and order are two different things
=======================================

The sign and the order of the levels along the vertical dimension are independent.
A ``depth`` may go from the surface to the bottom (the usual case for observations and
z-level models), but also from the bottom to the surface, like a sigma-level model whose
first level is the deepest.
Likewise, a ``z`` of such a model goes from the bottom to the surface, but a ``z`` can also
be stored from the surface to the bottom.

The ``positive`` attribute only tells the sign.
The functions that change the sign, :func:`~xoa.coords.to_z` and :func:`~xoa.coords.to_depth`,
**do not change the order**, unless you ask for it with ``reverse``:

.. ipython:: python

    depth = xr.DataArray([0., 10, 50], dims="k", name="depth", attrs={"units": "m"})
    z = coords.to_z(depth)
    print(z.values, z.attrs)
    print(coords.to_z(depth, reverse=True).values)

``reverse=True`` reverses along the first dimension, and a dimension name reverses along that one.
The conversion only negates the array, so it is lazy with dask arrays, and it costs
one pass over the values. The output keeps only the ``units`` attribute and gets the new
``positive`` one: other attributes, like the ``standard_name``, are not valid anymore.
The stale copy of the input that a coordinate carries is dropped.

Finding z and depth
===================

The :func:`~xoa.coords.get_z` and :func:`~xoa.coords.get_depth` functions look for
the coordinate with :mod:`xoa.meta`, and fall back on the other one, with the sign changed when its ``positive`` attribute says so:

.. ipython:: python

    da = xr.DataArray(
        np.ones(3), dims="k", coords={"z": coords.to_z(depth, reverse=True)}
    )
    coords.get_z(da).name
    coords.get_depth(da).values
    coords.get_depth(da).attrs["positive"]

When neither is found in a dataset, they are computed, depending on the ``type`` of the
``[vertical]`` section of the meta configuration: from sigma-like coordinates with
:func:`xoa.sigma.decode_sigma`, or from layer thicknesses with
:func:`xoa.grid.decode_dz2depth`.
Like the other finders, they take an ``errors`` argument, with ``"raise"`` as default
and ``"ignore"`` that returns ``None`` silently.
The ``xoa`` accessors have the same ``get_z`` and ``get_depth`` methods, and the
``da.xoa.z`` shortcut.

Use :func:`~xoa.coords.get_vertical` instead when you want what is there, **without
conversion**: it looks for a depth, then a ``z``, then an altitude.

.. ipython:: python

    coords.get_vertical(da).name

Computing z and depth
=====================

Sigma coordinates give a ``z``, since the formulas of the CF conventions give
heights that are negative in the ocean.
The meta specs are inferred from the dataset, and here they are explicitly set
to the default ones with :func:`xoa.meta.assign_meta_specs`:

.. ipython:: python

    from xoa import sigma
    ds = xr.Dataset(
        {"temp": (("sig", "x"), np.ones((5, 2)))},
        coords={
            "sig": ("sig", np.linspace(-1, 0, 5), {
                "standard_name": "ocean_sigma_coordinate",
                "formula_terms": "sigma: sig eta: ssh depth: bathy"}),
            "ssh": ("x", np.zeros(2)),
            "bathy": ("x", [50., 100.]),
        },
    )
    ds = xoa.meta.assign_meta_specs(ds, "default")
    dsz = sigma.decode_sigma(ds)
    dsz.z.isel(x=1).values
    coords.get_depth(ds).isel(x=1).values

Note that ``depth`` is also the name of the bathymetry term of the formulas, which
is positive down but is not a coordinate.

Layer thicknesses are integrated by :func:`xoa.grid.dz2depth`, which returns
a ``z`` when ``positive="up"`` and a ``depth`` otherwise. The ``positive`` argument is
also what decides which end of the levels is the surface, as well as the
``[vertical] positive`` option of the configuration or the ``positive`` attribute
of the vertical dimension coordinate when it is inferred.

.. ipython:: python

    from xoa import grid
    dz = xr.DataArray(np.full((3, 2), 10.), dims=("lev", "x"), name="dz")
    grid.dz2depth(dz, "down").isel(x=0).values
    z = grid.dz2depth(dz, "up")
    z.name, z.isel(x=0).values

What uses them
==============

- :func:`xoa.plot.plot_section` plots the vertical coordinate as it is found, and inverts
  the vertical axis when it is positive down.
- :func:`xoa.thermdyn.mixed_layer_depth` also uses the coordinate as it is found, since
  the ``positive`` attribute also tells it where the surface is. It always returns
  a positive value, which you may negate to draw it on a ``z`` axis.
- :func:`xoa.plot.plot_ts` converts to ``z`` with :func:`~xoa.coords.get_z` before
  computing the pressure with ``gsw.p_from_z``.
- :func:`xoa.regrid.regrid1d` works with any coordinate you give it: interpolate
  from ``z`` to ``z`` or from ``depth`` to ``depth``, and convert before, never across.

Pitfalls
========

- **Trust the attribute, not the name.** A coordinate is returned as found, and a variable
  named ``depth`` whose ``positive`` attribute is ``"up"`` is already a ``z``:
  :func:`~xoa.coords.get_z` returns it as is, without changing its sign, and
  :func:`~xoa.coords.get_depth` returns it as is too. Without this attribute, the
  ``depth`` name is taken as positive down. Set the attribute of your coordinates when
  their sign is not the usual one.
- **Mixing signs in a regridding or an isoline.** Values and bounds must have the same sign:
  ``-15`` m is a ``z``, ``15`` m is a ``depth``.
- **A dimension called** ``z``. A vertical dimension named like the ``z`` coordinate,
  or a dimension coordinate holding level indices, is read as a ``z`` coordinate.
  A 2D ``z`` computed from the thicknesses cannot be assigned to a dataset with a ``z``
  dimension: :func:`xoa.grid.decode_dz2depth` then raises an error, or leaves the dataset
  unchanged with ``errors="ignore"``, and you have to use :func:`xoa.grid.dz2depth`
  and give it another name.
- **Staggered locations.** With a configuration like ``croco``, ``z`` gets the location
  suffix of its grid, like ``z_w``, as the other coordinates.
- **Datasets in which the levels are not ordered like their sign suggests.** Functions that
  need the surface level, like the mixed layer depth, take it from the ``positive`` attribute
  and the order of the levels: check that they are consistent after converting
  with ``reverse``.
