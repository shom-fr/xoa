What's new
##########


v0.3.1 (2021-05-21)
===================

New features
------------

- Add an ``autolim`` keyword to :func:`xoa.plot.plot_flow` to speedup
  the processing with cartopy maps.

Breaking changes
----------------

- Rename the ``cf`` and ``sigma`` keyword of :func:`xoa.register_accessors`
  respectively to ``xcf`` and ``decode_sigma`` to match the default
  name of accessors.
- Rename the ``sigma`` accessor to ``decode_sigma``.

Bug fixes
---------

- Fix the access to the xoa executable on windows.
- Fix the minimal version for xarray [:pull:`23`].

Documentation
-------------

- Add a "How to start" section.
- Accessors are not documented separately with ``sphinx-autosummary-accessors``
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

