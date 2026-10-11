.. _roadmap:

Development roadmap
===================

The base
--------

The fundamental objective of xoa is to facilitate the analysis,
transformation and comparison of ocean data.
Facilitate here means that it must be possible to perform
complex operations in a few lines, without having to know all the
API and all the parameters to obtain a suitable and accurate result.


Full regridding capabilities
----------------------------

Extend :mod:`xoa.regrid` into a comprehensive regridding framework
with full support for structured and unstructured grids,
including curvilinear and staggered ocean model grids.
All core interpolation routines will be accelerated with
:mod:`numba` and exposed through a high-level :mod:`xarray` interface,
so that regridding a dataset is a one-liner.

Oceanographic grid operators
----------------------------

Classical differential operators on ocean grids are available in :mod:`xoa.dyn`:
vorticity, divergence, strain, Okubo-Weiss parameter and geostrophic currents,
accelerated with :mod:`numba` for collocated grids.
They will be extended to handle staggered grids natively,
and to other diagnostics commonly used in physical oceanography.

Graphics
--------

Add specialized oceanographic diagram types:

- **Taylor diagrams** for multi-model or multi-variable skill assessment.
- **Target diagrams** for compact representation of model biases
  and root-mean-square differences.

Geographic tools
----------------

Provide more efficient geographic utilities,
starting with spherical nearest-neighbour searches
based on ball-tree or k-d tree structures suited
for latitude/longitude coordinates.
These tools will speed up operations like
observation-to-model matching and spatial selection.

