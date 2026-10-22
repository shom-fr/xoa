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

xoa provides the numerical building blocks and the generic naming layer.
The numeric kernels are accelerated with :mod:`numba` and exposed
through :mod:`xarray` accessors.

Next diagnostics
----------------

- **Transports and sections**: volume, heat and salt transports across a section,
  normal velocity and stream function, based on :func:`xoa.grid.vintegrate`
  and the velocity getters of :mod:`xoa.dyn`.
- **Integrated contents**: heat and salt contents and dynamic height.
- **Spectra and scales**: variance spectra in frequency and wavenumber,
  coherence and wavelets, with proper handling of masks and irregular grids.
- **Water masses**: T-S diagrams with density classes and isopycnal depths.
- **Waves**: directional spectra, integrated parameters and Stokes drift,
  as a follow-up of :func:`xoa.plot.plot_rose`.

Tools
-----

- Parallel numba kernels for large grids.
- More regridding and kriging options with the same one-line interface.
- More tutorials, as the entry point of each new feature.

Ecosystem
---------

Some domains are developed on top of xoa and are out of its scope:

- **shoot**: detection and tracking of eddies and fronts.
- **armoor**: evaluation of ocean models.

Staggered grids are not handled natively.
