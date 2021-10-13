What's new
##########

0.5.1 (2021-10-13)
==================

New features
------------
- Switch the CI workflow to github  [:pull:`36`].

Bug fixes
---------
- Fix :meth:`xoa.cf.CFSpecs.to_loc` that which failing with dataset  [:pull:`23`].


0.5.0 (2021-10-12)
==================

New features
------------
- Add the `hlocs` argument to :func:`xoa.sigma.get_sigma_terms`
  and :func:`xoa.sigma.decode_cf_sigma` to decode at several horizontal
  staggered grid locations.
- Add the `edges` argument to :func:`xoa.regrid.regrid1d` to manually specify
  the edges that are used by the "cellave" regridding method.
- Add back the `loc` argument to the formatting methods of :mod:`xoa.cf`.
- Add dimension checking and support for dask arrays in :mod:`xoa.sigma`.
- Expose a few options of :meth:`xoa.cfgm.ConfigManager` to the
  :func:`xoa.cfgm.cfgargparse` function.
- Add the :confval:`cfgm_cfg_file` sphinx configuration option
  to save the default configuration of a :meth:`xoa.cfgm.ConfigManager`.

Bug fixes
---------
- Fix :func:`xoa.regrid.regrid1d` with "cellave" method.
- Fix :meth:`xoa.cf.CFSpecs.get_location_mapping` for coordinates that have
  no axis attribute specifications.
- Fix :func:`xoa.grid.dz2depth` that was not working properly with 4D+ arrays.


Breaking changes
----------------
- The `loc` argument of :func:`xoa.sigma.get_sigma_terms` is renamed `vloc`.


0.4.0 (2021-09-02)
==================

New features
------------
- :meth:`xoa.cf.CFSpecs.decode` better supports staggered grids.
- :meth:`xoa.cf.CFSpecs.search_dim` supports generic names in addition
  to dimension types as second argument.
- Add the :meth:`xoa.cf.CFSpecs.match_dim` method to check if a given
  dimension name is known.
- Add the :meth:`~xoa.cf.CFSpecs.reloc` and :meth:`~xoa.cf.CFSpecs.to_loc` methods
  to :class:`xoa.cf.CFSpecs` for quickly changing the staggered grid indicators
  in names.
- Add the :meth:`xoa.cf.SGLocator.add_loc` method to quickly change the location
  markers in a data array.

Breaking changes
----------------
- :func:`xoa.coords.get_dims` is renamed to :func:`xoa.coords.get_cf_dims`.
- The `name` argument of :class:`xoa.cf.CFSpecs` methods is renamed to `cf_name`,
  and the `dim_type(s)` argument is renamed to `cf_arg(s)`.
- :meth:`xoa.cf.SGLocator.get_location` is renamed to
  :meth:`~xoa.cf.SGLocator.get_loc_from_da` and the :meth:`~xoa.cf.SGLocator.get_loc` is added.

Bug fixes
---------
- Fix the output formatting of :func:`xoa.grid.dz2depth`.

Documentation
-------------
- The :ref:`uses.cf` section and :ref:`sphx_glr_examples_plot_hycom_gdp.py` example
  are adapted to reflect changes.


v0.3.1 (2021-05-21)
===================

New features
------------

- Add an `autolim` keyword to :func:`xoa.plot.plot_flow` to speedup
  the processing with cartopy maps.

Breaking changes
----------------

- Rename the `cf` and `sigma` keyword of :func:`xoa.register_accessors`
  respectively to `xcf` and `decode_sigma` to match the default
  name of accessors.
- Rename the `sigma` accessor to `decode_sigma`.

Bug fixes
---------

- Fix the access to the xoa executable on windows.
- Fix the minimal version for xarray [:pull:`23`].

Documentation
-------------

- Add a "How to start" section.
- Accessors are now documented separately with `sphinx-autosummary-accessors`
  [:pull:`20`].
- The Hycom-GDP example now uses :func:`xoa.plot.plot_flow`.


v0.3.0 (2021-05-12)
===================

New features
------------

- Add the :func:`xoa.plot.plot_flow` function [:pull:`9`].
- Improve :func:`xoa.coords.get_depth` so that it can compute
  depth from sigma coordinates or layer thinknesses [:pull:`8`].
- Add the :func:`xoa.dyn.flow2d` function [:pull:`7`].
- Add the :func:`xoa.regrid.extrap1d` function.
- Add the :func:`xoa.filter.erode_coast` function which is specialized version
  of the :func:`xoa.filter.erode_mask` for horizontal data.
- Add the :func:`xoa.coords.get_xdim`, :func:`~xoa.coords.get_ydim`,
  :func:`~xoa.coords.get_zdim`, :func:`~xoa.coords.get_tdim` and
  :func:`~xoa.coords.get_fdim` for quickly finding standard dimensions.

Bug fixes
---------

- Fix u and v CF config [:pull:`6`]


0.2.0
=====

New features
------------

Breaking changes
----------------

Deprecations
------------

Bug fixes
---------

Documentation
-------------

