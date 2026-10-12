.. _indepth.stataccum:

Accumulating statistics
#######################

The :class:`xoa.stats.StatAccum` class computes statistics step by step, for data
that are too large to be loaded at once, or that are produced along a run.
See :ref:`sphx_glr_tutorials_plot_stataccum.py` for a walkthrough.

Principle
=========

Each call to :meth:`~xoa.stats.StatAccum.append` updates a small set of
*centered moments*: sum of weights, weighted mean, and sums of squared and cross deviations,
plus minima and maxima. Statistics such as the standard deviation, the root mean square
error or the correlation are derived from them on demand with
:meth:`~xoa.stats.StatAccum.get_stats`.

Moments are merged with a pairwise update, so that the result is the same whatever
the way the data are split in time, and it does not suffer from the cancellation errors
of sums of squares, which matters for fields with a large mean like temperature in kelvins.
The numba kernels are in :mod:`xoa.core.stats`.

Temporal and spatial statistics
===============================

- **Temporal** statistics, prefixed with ``t``, are accumulated along time
  at each point of the other dimensions. They are not weighted.
- **Spatial** statistics, prefixed with ``s``, are computed at each time step over `sdims`,
  and they form a time series. `sdims` can be a vertical dimension only,
  the horizontal ones, or all of them.

Two variables can be given to get comparative statistics: ``bias`` (second minus first),
``rms``, centered ``crms``, ``cov`` and ``corr``. Only the points valid in both are used.

Names and attributes
====================

Statistics are named after their kind, like ``tmean`` or ``scorr``, suffixed with ``0`` and
``1`` for the per-variable ones of comparative accumulations. Their attributes are made from
those of the input arrays: ``long_name`` mentions the input, ``units`` follow
(squared for variances), ``standard_name`` is kept for means and extrema,
and ``cell_methods`` tells the reduced dimensions.
Names and attributes of the inputs are saved by :meth:`~xoa.stats.StatAccum.dump`.

Weights
=======

Spatial statistics are weighted, since cells do not have the same size.
A vertical mean must be weighted by the layer thicknesses, which vary in space and time
with sigma coordinates, and a horizontal mean by the cell areas, which vary with latitude.
Weights are automatic by default and they only concern the reduced dimensions:

- the layer thicknesses :func:`xoa.grid.get_dz` if the vertical dimension is reduced,
  which are the thickness variable of the data if any, else they are computed from its
  ``z`` or ``depth`` coordinate (see :func:`xoa.grid.z2dz`);
- the grid cell sizes :func:`xoa.grid.get_dx_dy` along the reduced horizontal dimensions,
  from the longitudes and latitudes.

The vertical coordinate must therefore be attached to the data, and an error is raised
when it cannot be found. Horizontal dimensions are not weighted, with a warning, if
longitudes and latitudes cannot be found, which is fine for a regular Cartesian grid.
Thicknesses computed from a ``z`` are exact for evenly spaced levels and approximate
otherwise: pass your own thicknesses with ``weights=dz`` when the layers are stretched,
and ``weights=False`` to disable all the weights.
Samples with a missing or non-positive weight are ignored, as the ones below the sea floor.

Restarting
==========

:meth:`~xoa.stats.StatAccum.dump` writes the moments to a netcdf file with one group
for each kind of statistics, and :meth:`~xoa.stats.StatAccum.from_file` restores them.
Explicit weights are not stored and must be provided again, and automatic ones are recomputed. Appending the same times twice
would bias the statistics, so the times that are not after the last accumulated one
are refused, or skipped with ``on_overlap="skip"``, which is convenient to replay a step.
Therefore the data must have a time coordinate to benefit from this protection.
