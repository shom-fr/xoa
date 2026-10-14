What's new
##########

Unreleased
==========

New features
------------
- Add :class:`xoa.krig.LocalKriger`, which krigs with the nearest points only using the new :class:`~xoa.core.geo.SphericalKDTree` and the numba kernels of :mod:`xoa.core.krig`. See :ref:`sphx_glr_tutorials_grids_plot_kriging.py` and :ref:`sphx_glr_tutorials_grids_plot_kriging_sst.py`.
- Add :func:`xoa.coords.geo_unstack` to restore the grid of an object stacked with :func:`~xoa.coords.geo_stack`, and its ``singleton`` argument. The stacked dimensions are now in a fixed order, latitude first.
- Add :class:`xoa.krig.ClusteredKriger`, which now holds the clusterization (``npmax`` and ``nproc``) that moved out of :class:`xoa.krig.Kriger`.
- Add :class:`xoa.geo.GeoKDTree` and :func:`xoa.geo.get_neighbors` to find the nearest points or those within a distance, also as ``xoa`` accessor methods with a cached tree, based on the new :class:`xoa.core.geo.SphericalKDTree`. See :ref:`sphx_glr_tutorials_grids_plot_neighbors.py`.
- Add the :ref:`indepth.dyn` guide and the :mod:`xoa.dyn` diagnostics :func:`~xoa.dyn.get_relvort`, :func:`~xoa.dyn.get_div`, :func:`~xoa.dyn.get_strain`, :func:`~xoa.dyn.get_okuboweiss`, :func:`~xoa.dyn.get_coriolis` and :func:`~xoa.dyn.get_geos`, also available as ``xoa`` accessor methods, with numba kernels in the new :mod:`xoa.core.dyn` module. Grid resolution comes from :func:`xoa.grid.get_dx_dy`.
- Add :func:`xoa.dyn.get_cyclogeos`, an iterative cyclogeostrophic solver after Penven et al. (2014), with full (smoothed) advection terms or the two classic ones with ``terms="penven"``.
- The :mod:`xoa.dyn` diagnostics accept a variable vertical coordinate ``z`` to compute horizontal derivatives at constant depth. See the :ref:`sphx_glr_tutorials_dynamics_plot_croco_rossby.py` tutorial.
- Add :class:`xoa.dyn.VerticalModes` to compute vertical modes, speeds and deformation radii, and to decompose and recompose profiles, based on the new :mod:`xoa.core.modes` module. See :ref:`sphx_glr_tutorials_dynamics_plot_vertical_modes.py`.
- Add :func:`xoa.thermdyn.get_n2` for the squared Brunt-Vaisala frequency, and :func:`xoa.coords.as_z` to get a vertical coordinate as z, positive up.
- Add the :mod:`xoa.numbers` module with :func:`~xoa.numbers.get_rossby`, :func:`~xoa.numbers.get_burger`, :func:`~xoa.numbers.get_froude` and :func:`~xoa.numbers.get_ekman`. ``xoa.dyn.get_rossby`` moved to it, without backward compatibility. See :ref:`sphx_glr_tutorials_dynamics_plot_dimensionless_numbers.py`.
- Add :class:`xoa.stats.StatAccum` to accumulate temporal and spatial statistics, with automatic vertical and horizontal weights, dump and restart, based on numba moments in :mod:`xoa.core.stats`. See :ref:`indepth.stataccum`.
- Add :func:`xoa.grid.dz2z`, :func:`xoa.grid.z2dz`, :func:`xoa.grid.get_dz`, :func:`xoa.grid.get_dz_terms` and :func:`xoa.grid.decode_dz2z`, and the ``positive`` and ``errors`` arguments of :func:`xoa.coords.get_positive_attr`. :func:`xoa.coords.get_z` now relies on :func:`~xoa.grid.decode_dz2z`. Fix :func:`xoa.grid.decode_dz2depth` with a bathymetry or a sea surface height and ``positive="down"``.
- Group the tutorials in categories: graphics, grids, dynamics and evaluation, and add the :ref:`sphx_glr_tutorials_graphics_plot_maps_vectors.py`, :ref:`sphx_glr_tutorials_grids_plot_filtering.py` and :ref:`sphx_glr_tutorials_grids_plot_kriging.py` tutorials, and the new :ref:`sphx_glr_tutorials_meta_plot_naming.py` and :ref:`sphx_glr_tutorials_dynamics_plot_mixed_layer.py` ones. Figures are narrower to fit the doc pages, and the :ref:`indepth.meta` guide is shorter and clearer.
- Add the :ref:`references` page.
- Add :func:`xoa.mask.get_polygon_mask`, also an ``xoa`` accessor method, to flag grid cells inside polygons with a coverage threshold, based on the new :mod:`xoa.core.mask` and :mod:`xoa.core.poly` numba functions.
- Add :func:`xoa.plot.plot_fields` to plot several fields on maps of a single figure, with a shared colorbar if needed.
- Add the ``normalize`` parameter to :func:`xoa.dyn.get_div`, :func:`~xoa.dyn.get_strain` and :func:`~xoa.dyn.get_okuboweiss`.
- Add :func:`xoa.mask.get_common_mask`, :func:`xoa.stats.rms` and :func:`xoa.poly.get_contour_polygons`, with the new :mod:`xoa.stats` and :mod:`xoa.poly` modules.
- Add the :meth:`~xoa.dyn.VerticalModes.energy_share` and :meth:`~xoa.dyn.VerticalModes.truncation_error` methods to :class:`xoa.dyn.VerticalModes`.
- Add :func:`xoa.plot.plot_vectors` and :func:`xoa.core.plot.plot_vectors` to draw currents as under-sampled arrows.
- Add the :ref:`sphx_glr_tutorials_dynamics_plot_gulfstream_dynamics.py` tutorial, based on a new GLORYS data sample.
- The ``mercator`` naming specifications are also used for GLORYS.
- Add :func:`xoa.coords.to_surface_ref` to refer a ``z`` or a ``depth`` to the free surface instead of the fixed datum, and the ``ref`` argument of :func:`xoa.sigma.decode_sigma` and of the ``decode_sigma`` accessor to do it while decoding.
- Add :func:`xoa.coords.copy_coord_attrs` to restore coordinate attributes after :func:`xarray.apply_ufunc`.
- Add the :ref:`indepth.vertical` guide and the ``z`` coordinate (positive up) as distinct from ``depth`` (positive down): ``z`` meta spec, :func:`xoa.coords.get_z`, :func:`~xoa.coords.is_z`, :func:`~xoa.coords.to_z`, :func:`~xoa.coords.to_depth`, :func:`~xoa.coords.reverse_dim` and the ``xoa.get_z`` accessor method. :func:`~xoa.coords.get_depth` and :func:`~xoa.coords.get_z` fall back on each other and flip the sign when needed.
- Add :func:`xoa.core.plot.get_projection` to get a map projection from an instance, a cartopy lower case name or the ``"merc"``, ``"pc"`` and ``"ortho"`` aliases. The default of :func:`xoa.plot.plot_grid` is ``"ortho"``, centered on the grid.
- The main ``xoa`` accessor is now registered when :mod:`xoa` is imported.
- Add the :class:`xoa.core.interp.XYInterpolator` and :class:`xoa.core.regrid.XYRegridder` numba classes for horizontal bilinear, bicubic and conservative interpolation and regridding, with their kernels in the new :mod:`xoa.core.spline` and :mod:`xoa.core.conserv` modules.
- Add the :class:`xoa.interp.Interpolator` and :class:`xoa.regrid.Regridder` xarray classes for horizontal (and temporal) interpolation and regridding, with the :func:`xoa.coords.geo_merge` and :func:`xoa.grid.ds2grid_dict` helpers.
- The methods of :class:`xoa.interp.Interpolator` and :class:`xoa.regrid.Regridder` are listed in the :class:`xoa.interp.xy_interp_methods` and :class:`xoa.regrid.xy_regrid_methods` enums, that accept aliases like ``"linear"``.
- :func:`xoa.regrid.regrid1d` and :func:`xoa.regrid.extrap1d` now accept datasets: only variables that have the working dimension are processed.
- Add the :func:`xoa.grid.get_resolution`, :func:`xoa.grid.get_median_resolution` and :func:`xoa.grid.get_edge_extents` functions, and their numpy counterparts :func:`xoa.core.grid.compute_resolution` and :func:`xoa.core.grid.median_resolution_deg`.
- Add the plotting helpers :func:`xoa.plot.plot_field`, :func:`xoa.plot.plot_grid`, :func:`xoa.plot.plot_section`, :func:`xoa.plot.plot_stick`, :func:`xoa.plot.get_label`, :func:`xoa.plot.create_base_map`, :func:`xoa.plot.setup_map_axes` and :func:`xoa.plot.add_land`. Their inputs are found with :mod:`xoa.meta`.
- The weights of :class:`xoa.interp.Interpolator` and :class:`xoa.regrid.Regridder` are computed only once for the same grids, points and parameters: they are shared between the objects, and thus between the calls of the ``xoa`` accessors on the variables of a dataset. The :func:`xoa.interp.clear_weights_cache` and :func:`xoa.regrid.clear_weights_cache` functions free them.
- The weights files of :class:`xoa.interp.Interpolator` and :class:`xoa.regrid.Regridder` are now netcdf files with one group per grid and method, named after a fingerprint of the grids, so that a single file serves all your grids, the weights are found automatically and cannot be loaded for other grids. A group is added to the file when it is not found. Files in the previous format are still read, with a warning, but the grids cannot be checked. See the new :mod:`xoa.weights` module.
- The fingerprint of a grid is available with :func:`xoa.grid.get_fingerprint`, and from the ``src_fingerprint``, ``dst_fingerprint``, ``fingerprint`` and ``weights_group`` attributes of :class:`xoa.regrid.Regridder` and :class:`xoa.interp.Interpolator`. The :func:`xoa.weights.find_groups` and :func:`xoa.weights.describe_groups` functions find and describe the weights of a grid in a weights file.
- Add the in-depth guides :ref:`indepth.horizontal` and :ref:`indepth.plot`.
- Add the :func:`xoa.misc.get_array_fingerprint` and :func:`xoa.misc.combine_fingerprints` functions and the :class:`xoa.misc.SmallCache` class.
- Add the :func:`xoa.plot.add_colorbar` and :func:`xoa.core.plot.add_colorbar` functions that add shrunk colorbars, labelled from the data array, which are used by all the colorbars of :mod:`xoa.plot`.
- Add the :mod:`xoa.core.plot` module with the low level plotting routines that work on numpy arrays and axes, like :func:`xoa.core.plot.plot_mesh`, :func:`xoa.core.plot.plot_depth_section` and :func:`xoa.core.plot.plot_sticks`, which are used by the high level functions of :mod:`xoa.plot`.
- :mod:`xoa.plot` imports cartopy lazily, only when a map is drawn.
- Add the :func:`xoa.regrid.regridxy` and :func:`xoa.interp.interpxy` functions, that build and apply a regridder or an interpolator in a single call, or reuse one that is passed to them. The ``interp`` and ``regrid`` methods of the ``xoa`` accessors are now based on them.
- Add the ``interp`` and ``regrid`` methods to the ``xoa`` accessors.
- Add the :mod:`xoa.core.grid`, :mod:`xoa.core.poly` and :mod:`xoa.core.time` modules.
- Add Taylor diagrams with :func:`xoa.plot.plot_taylor` for xarray objects, and the :class:`xoa.core.plot.TaylorDiagram` class and :func:`xoa.core.plot.plot_taylor` function for numpy arrays, with labels, markers, colored values and negative correlations. The numba statistics are computed by :func:`xoa.core.stats.taylor_stats` of the new :mod:`xoa.core.stats` module.

Breaking changes
----------------
- :func:`xoa.sigma.decode_sigma` and the ``ocean_*`` functions of :mod:`xoa.sigma` now return a ``z`` array, which is positive up, instead of a ``depth`` array that was negative. :func:`xoa.grid.dz2depth` and :func:`xoa.grid.decode_dz2depth` return a ``z`` array when ``positive="up"``, and :func:`xoa.grid.decode_dz2depth` no longer overwrites a dimension coordinate that has the same name. Use :func:`xoa.coords.get_depth` to get a depth.
- :func:`xoa.coords.get_vertical` now uses the meta specs of its argument and looks for a depth, a ``z`` and then an altitude. The ``z`` name is no longer an alternative name of the ``level`` coordinate, and the ``z`` entry of the ``hycom`` configuration is now a ``depth`` entry.
- :func:`xoa.grid.get_edges` now extrapolates the outer edges linearly by default (``mode="linear_extrap"``) instead of replicating the end values. Pass ``mode="edge"`` to recover the previous behaviour. This also changes the edges inferred by the ``cellave`` method of :func:`xoa.regrid.regrid1d`.
- The minimal version of xarray is now 2024.6, which provides the ``on_missing_core_dim`` argument of :func:`xarray.apply_ufunc` used to interpolate and regrid datasets.
- :func:`xoa.grid.to_rect` now relies on :func:`xoa.core.grid.check_grid_type` and converts a longitude and the latitude that shares its dimensions together, only if the grid is not curvilinear.

Deprecations
------------
- :func:`xoa.core.interp.grid2rellocs` is deprecated, use :func:`xoa.core.spline.compute_frac_indices` or :class:`xoa.core.interp.XYInterpolator` instead.
- :func:`xoa.core.interp.cell2relloc` is deprecated, use :func:`xoa.core.geo.relative_cell_coords` instead (vertices 2 and 4 are swapped).
- :func:`xoa.core.interp.closest2d` is deprecated, use :func:`xoa.core.geo.closest_point_fast` instead.

Bug fixes
---------
- Fix :mod:`xoa.krig`: simple kriging now uses the covariance, block kriging averages per point, :func:`~xoa.krig.empirical_variogram` no longer fails when undersampling, drops pairs at the same distance or overwrites the long name, a fitted :class:`~xoa.krig.VariogramModel` is not fitted again and the first variable of a dataset is used to fit it. The krigers now return the grid or the dimensions of the target with its coordinates and their attributes, and the attributes of the input data, and accept a single position. The kriging error is now named variance, and it is conservative for the :class:`~xoa.krig.ClusteredKriger`. The latter and :func:`xoa.geo.clusterize` accept a ``seed`` for reproducible clusters. The kriging matrices are factorized (Cholesky or LU) instead of inverted, with a pseudo inverse for singular ones, and the fitted nugget is at least ``1e-6`` of the sill.
- The ``set_meta_specs`` method of the ``xoa`` accessors accepts the same arguments as :class:`xoa.meta.set_meta_specs`, like a name or a dict.
- Fix :class:`xoa.krig.Kriger` that computed the distances between input points in meters whatever its `dist_units`.
- :func:`xoa.coords.get_depth` no longer fails when it has to decode a dataset, and no longer warns when ``errors="ignore"``.
- Fix the ``decode_sigma`` subaccessor of the ``xoa`` accessor of datasets, that returned ``None``.
- Fix :func:`xoa.sigma.decode_sigma` that gave the same name to the heights at all vertical locations, like ``z`` at rho and w levels of CROCO, when the name format of the meta specs has no location, as the default one, so that one of them was lost. The location is now always appended to the name, like ``z_w``, and it is also parsed from the name of the sigma variable, so that the heights at rho levels of CROCO are named ``z_rho`` with the ``croco`` specs, as in :ref:`sphx_glr_tutorials_grids_plot_croco_section.py`.
- Fix the meta appendix: the meta configurations are no longer called specifications, the specifications provided by the internal :file:`meta.ini` file have their own page, and the details pages of the configuration items are not listed in the left table of contents anymore.
- Fix the bilinear and bicubic interpolation with ``skipna=True`` and ``na_thres=0``, that set isolated points to nan because of rounding errors in the sum of the weights of valid neighbours.
- Fix the conservative weights of the cells that cross the dateline or that are on the other side of it, which were missing. Longitudes of the cells are made continuous before computing their overlaps, with :func:`xoa.core.poly.unwrap_longitudes` and :func:`xoa.core.grid.unwrap_grid_longitudes`.
- Fix the points that are on the first and last lines of a regular source grid, that were NaN on the last one and extrapolated up to one cell before the first one, with the bilinear and bicubic methods. Cells are now closed and what is outside is NaN on all sides.
- Fix the bicubic method that returned NaN for the points that are exactly on the second and last but one lines of a rectangular grid, whereas they are valid on a regular grid.
- Fix the interpolation on rectangular grids, which have an irregular spacing, whose latitudes decrease along their dimension, that gave NaN everywhere, and a longitude search that wrongly found points that are on the opposite side of the globe of a cell that crosses the dateline.
- Fix the interpolation on curvilinear grids that cross the dateline, that gave NaN or wrong values near it.
- Fix the search of the closest node of large curvilinear grids, that missed the cell of some points of the inside of the grid (0.1% on a 400 x 400 grid): the closest node of a subsampled grid is now refined by walking downhill instead of searching a window of fixed size, and invalid nodes trigger a full search.
- Fix the conservative weights that were silently truncated to ten links per source cell, so that regridding to a much finer grid gave wrong values, and make their computation much faster with an index of the source cells, instead of testing all the pairs of cells.
- :func:`xoa.core.grid.check_grid_type` no longer warns for a grid with a single row or column.
- :meth:`xoa.interp.Interpolator.interp_with_time` no longer renames an unnamed data array.
- :func:`xoa.plot.plot_ts` only adds a colorbar automatically when the scatter colors are a data array, and not a numpy array, and its colorbar is now shrunk and labelled like the others. Its axis labels are built with :func:`xoa.plot.get_label`.
- Fix a data race in :func:`xoa.core.interp.closest2d` (parallel scan writing to shared accumulators) that made :func:`xoa.interp.grid2loc` silently return wrong results, or crash, on genuinely curvilinear (non-separable) grids.
- Remove ``fastmath=True`` from :func:`xoa.core.interp.closest2d`, :func:`xoa.core.interp.cell2relloc` and :func:`xoa.core.interp.grid2relloc`: it could break the NaN-skip comparisons these functions rely on to ignore invalid/land-masked grid points, silently returning a wrong result.


2026.7.0
========

Bug fixes
---------
- Remove the ``xoa reset_cf_cache`` CLI command: the disk cache no longer exists in :mod:`xoa.meta` and the command was raising :exc:`AttributeError` at runtime [:pull:`117`].


2026.2.0
========

New features
------------
- Add the ``within`` parameter to :mod:`xoa.cf` search methods.
- Add the ``rename_args`` argument to formatting routing of :mod:`xoa.meta` to retrieve renaming information [:pull:`102`].
- Add the :func:`xoa.dyn.get_sea_level` function to get either SSH, ADT, SLA or MDT in a dataset [:pull:`102`].
- Allow passing a list of "meta_name" to :meth:`xoa.cf.MetaSpecs.get` for enlarging a search [:pull:`102`].
- The :mod:`xoa.data_samples` module now use Pooch to manage sample files [:pull:`111`].
- Add the ``multi`` parameter to :class:`xoa.misc.Choices` to allow multiple choices [:pull:`111`].

Deprecations
------------
- The :mod:`xoa.cf` module is deprecated. Use :mod:`xoa.meta` instead. All references to "cf" must be converted to "meta" in modules, functions, classes, methods and accessors. A backward compatibility layer has been added [:pull:`111`].
- The :mod:`xoa.cf_configs` module is deprecated in favour of module :mod:`xoa.meta.configs` [:pull:`111`].
- Pure core regridding 1D function of the :mod:`xoa.core.interp` module are moved to :mod:`xoa.core.regrid` module.

Bug fixes
---------
- Fix :func:`xoa.coords.get_time` that was returning None.
- Fix :func:`xoa.interp.linear1d` extrapolation [:pull:`107`]
- Fix :func:`xoa.grid.to_rect` warnings.
- Fix :meth:`xoa.meta.MetaSpecs.get_dim_types`.
- Fix ``test_accessors.py`` to use ``meta`` accessor instead of deprecated ``xmeta`` and ``meta_specs`` property instead of ``metaspecs``.
- Fix examples to use :func:`xoa.get_meta_config_file` instead of deprecated ``get_cf_config_file``.

Documentation
-------------
- Rename "Using the modules" section to "In-depth guides".
- Added in the in-depth guide to metadata mentions of :mod:`xoa.coords`, :mod:`xoa.dyn` and :mod:`xoa.thermdyn` helper function to access coordinates and physical variables like lon or temp.
- Added in-depth guides about sigma-like coordinates, grids and regridding, and filtering.


0.8.0 (2025-05-12)
==================

New features
------------
- Add modules :mod:`xoa.data_samples` and :mod:`xoa.cf_configs`.
- :class:`xoa.cf.CFSpecs`, :func:`xoa.cf.register_cf_specs`, :func:`xoa.cf.get_cf_specs` and :class:`xoa.cf.set_cf_specs` now accepts short names of internal CF config like "croco" or "argo".
- Add `absolute` option for salinity and the `ref_dens` option to enable using different formulae for density to :func:`xoa.plot.plot_ts`.
- Add the `exclude_names` option to :mod:`xoa.cf` configurations to ignore data array names.
- Call :func:`xoa.cf.infer_cf_specs` from :func:`xoa.cf.get_cf_specs`  when a data array or dataset is passed [:pull:`96`].
- Add the `drop_na` and `maxgap` parameters to the :func:`xoa.regrid.regrid1d` function to fill gaps [:pull:`92`].
- Add the :func:`xoa.filter.tidal_filter` function to apply a tidal filter to a time series [:pull:`87`].

Breaking changes
----------------
- :meth:`xoa.cf.CFSpecs` tests the generic pivot name last, not first.
- The :meth:`xoa.cf.CFSpecs.decode` method no longer sets the "cf_specs" encoding [:pull:`96`].
- The default :metaopt:`name_format <[sglocator] name_format>` option of the :metasec:`sglocator <[sglocator]>` section of the meta specifications is set to False to prevent parsing the grid location in names [:pull:`96`].
- :func:`xoa.filter.tidal_filter` raises an :class:`~xoa.XoaError` instead of a simple warning when the time step is greater than an hour.
- :func:`xoa.regrid.regrid1d`: extrapolation from input data instead of output data [:pull:`91`].

Deprecations
------------
- Now using ``platformdirs`` instead of ``appdirs``.
- :func:`xoa.filter.demerliac` is deprecated, use :func:`xoa.filter.tidal_filter` instead [:pull:`88`].

Bug fixes
---------
- Fix :func:`xoa.cf.infer_cf_specs` that was missing attributes and encoding [:pull:`96`].
- Fix :func:`xoa.plot.plot_ts` that now uses absolute salinity along with potential temperature and potential density anomaly formulas.
- Fix `lat` keyword in :func:`~xoa.geo.deg2m` and :func:`~xoa.geo.m2deg` which now accepts arrays.
- Fix weighting of :func:`xoa.filter.convolve`.
- Fix :func:`xoa.grid.to_rect` raising of errors and handling of missing data in coordinates.
- Fix string formatting in :func:`xoa.filter.tidal_filter` warning.
- Fix default value of `deltadens` in :func:`xoa.thermdyn.mixed_layer_depth`.

Documentation
-------------
- Now using `Sphinx book theme <https://sphinx-book-theme.readthedocs.io/en/stable/#>`_

0.7.1 (2023-07-21)
==================

New features
------------
- Add support for not creating multi-indexes in :func:`xoa.coords.geo_stack`.

Bug fixes
---------
- Fix :func:`xoa.geo.get_distances` that was not able to handle multi-indexes from :func:`xoa.coords.geo_stack`.
- Fix doc generation [:pull:`83`].
- Fix :func:`xoa.sigma.get_cs` to keep it compatible with pure numpy arrays


Documentation
-------------
- Add a link to the github repository.


0.7.0 (2023-07-17)
==================

New features
------------
- Add issue and pull request templates.
- Add the :func:`xoa.filter.smooth` function [:pull:`76`].
- Improved the default `sig` and `std` parameter values for filter windows that accept them [:pull:`76`].
- Add the :func:`xoa.plot.plot_minimap` and :func:`xoa.plot.plot_double_minimap` functions to display the geographic situation of a set of coordinates [:pull:`73`].
- Add support for the `min_extent` keyword to :func:`xoa.geo.get_extent` [:pull:`73`]
- Add dask support to :mod:`xoa.sigma` sigma to depth converters [:pull:`72`].
- Add the :mod:`xoa.num` module that contains low level numeric utilities.
- Add the :func:`xoa.thermdyn.mixed_layer_depth` function to compute the mixed layer depth with three different methods [:pull:`67`, :pull:`75`].
- Add the :func:`xoa.thermdyn.is_temp`, :func:`xoa.thermdyn.is_sal` and :func:`xoa.thermdyn.is_dens` functions to infer if an array of temperature, salinity or density type, and added the related :func:`xoa.thermdyn.get_temp`, :func:`xoa.thermdyn.get_sal` and :func:`xoa.thermdyn.get_dens` function to search in datasets[:pull:`67`, :pull:`79`].
- Add `kernel_kwargs` keyword to :func:`xoa.filter.convolve` to better control the kernel generation by :func:`xoa.filter.generate_kernel` [:pull:`64`].
- Add inference of parameters for some window functions, like the gaussian shape, in :func:`xoa.filter.get_window_func` [:pull:`64`].
- Add :func:`xoa.regrid.isoslice` based on :func:`xoa.interp.isoslice` core function [:pull:`63`].

Breaking changes
----------------
- :func:`~xoa.filter.get_window_func` now accepts only one positional argument and all other arguments must be named.

Bug fixes
---------
- Fix :func:`xoa.cfgm.is_boolstr` which now supports the new :mod:`configobj`.
- Fix broadcasting :mod:`xoa.interp` 1d interpolation routines [:issue:`69`].
- Fix :func:`xoa.sigma.get_sigma_terms` so that it works in case of multiple levels coordinates [:pull:`60`].
- Fix :func:`xoa.grid.to_rect` that now infers coordinates and can emit a warning or raise an error.

Documentation
-------------
- Add an example of `xoa.plot.plot_double_minimap` to :ref:`sphx_glr_tutorials_evaluation_plot_mercator_argo.py` and :ref:`sphx_glr_tutorials_evaluation_plot_hycom_gdp.py` examples [:pull:`73`].
- Add an example of `xoa.thermdyn.mixed_layer_depth` to :ref:`sphx_glr_tutorials_grids_plot_croco_section.py` example [:pull:`67`].


0.6.1 (2022-02-24)
==================

New features
------------
- Add a warning to :func:`xoa.open_data_sample` that is emitted when the requested file is not an internal data sample [:pull:`47`].
- Add the :func:`xoa.plot.add_shadow`, :func:`xoa.plot.add_glow` and :func:`xoa.plot.add_lightshading` function to add path effects to plots [:pull:`44`].
- Add the :func:`xoa.plot.plot_ts` function to make T-S diagrams [:pull:`43`].
- Add the :func:`xoa.filter.demerliac` function to apply a Demerliac filter to a time series [:pull:`41`].
- Add support for fine tuning masking in :func:`xoa.filter.convolve` through the `na_thres` parameter [:pull:`41`].
- Add the :func:`xoa.geo.cdist` and :func:`xoa.geo.pdist` functions to compute haversine distances respectively between two dataset and with a dataset  [:pull:`40`].
- Add the :func:`xoa.coords.geo_stack` function to stack longitudes and latitudes into another dimension, in a dataset or data array  [:pull:`40`].
- Add the :func:`xoa.filter.decimate` function to crudely undersample a geographic dataset or data array with a radius of proximity [:pull:`40`].
- Add the :func:`xoa.geo.get_distances` to compute the Haversine distances between locations inside a single dataset or between two datasets [:pull:`40`].
- Add the :func:`xoa.krig.empirical_variogram` function to estimate variogram parameters.
- Add the :class:`xoa.krig.VariogramModel` to manage a variogram model [:pull:`40`].
- Add the :class:`xoa.krig.Kriger` and :func:`xoa.krig.krig` to perform kriging [:pull:`40`].
- Add the `exclude` option to data var and coordinate specifications of :class:`xoa.cf.CFSpecs` instances [:pull:`38`].

Breaking changes
----------------
- A single Nan now contaminates the data over the kernel emprise in :func:`xoa.filter.convolve` since `na_thres` is set to zero by default  [:pull:`40`].
- xoa now requires the :mod:`gsw` package.

Bug fixes
---------
- Fix :func:`xoa.regrid.regrid1d` so that it works now with time coordinates [:pull:`48`].
- Fix :func:`xoa.regrid.grid2loc` so that it works with scalar output coordinates.
- Fix :func:`xoa.regrid.regrid1d` to prevent conflict in the presence of MultiIndexes.
- Fix search for coordinates that are hidden due to :meth:`xarray.DataArray.stack`.

Documentation
-------------
- Add the :ref:`Compare Mercator to ARGO <sphx_glr_tutorials_evaluation_plot_mercator_argo.py>` example.


0.6.0 (2022-02-24)
==================

Empty with non existing tag.


0.5.1 (2021-10-13)
==================

New features
------------
- Switch the CI workflow to github  [:pull:`36`].

Bug fixes
---------
- Fix :meth:`xoa.cf.CFSpecs.to_loc` that was failing with dataset [:pull:`23`].


0.5.0 (2021-10-12)
==================

New features
------------
- Add the `hlocs` argument to :func:`xoa.sigma.get_sigma_terms` and :func:`xoa.sigma.decode_cf_sigma` to decode at several horizontal staggered grid locations  [:pull:`34`].
- Add the `edges` argument to :func:`xoa.regrid.regrid1d` to manually specify the edges that are used by the "cellave" regridding method  [:pull:`34`].
- Add back the `loc` argument to the formatting methods of :mod:`xoa.cf` [:pull:`34`].
- Add dimension checking and support for dask arrays in :mod:`xoa.sigma` [:pull:`34`].
- Expose a few options of :meth:`xoa.cfgm.ConfigManager` to the :func:`xoa.cfgm.cfgargparse` function.
- Add the :confval:`cfgm_cfg_file` sphinx configuration option to save the default configuration of a :meth:`xoa.cfgm.ConfigManager`.

Bug fixes
---------
- Fix :class:`xoa.cf.SGLocator.parse_attr` parsing since regular expressions were not matching full strings.
- Fix :func:`xoa.regrid.regrid1d` with "cellave" method  [:pull:`34`].
- Fix :meth:`xoa.cf.CFSpecs.get_location_mapping` for coordinates that have no axis attribute specifications [:pull:`34`].
- Fix :func:`xoa.grid.dz2depth` that was not working properly with 4D+ arrays [:pull:`34`].


Breaking changes
----------------
- The default `name_format` of :class:`xoa.cf.SGLocator` no longer contains the location pattern `{loc}` to prevent unwanted location catching.
- The `loc` argument of :func:`xoa.sigma.get_sigma_terms` is renamed `vloc` [:pull:`34`].


0.4.0 (2021-09-02)
==================

New features
------------
- :meth:`xoa.cf.CFSpecs.decode` better supports staggered grids.
- :meth:`xoa.cf.CFSpecs.search_dim` supports generic names in addition to dimension types as second argument.
- Add the :meth:`xoa.cf.CFSpecs.match_dim` method to check if a given dimension name is known.
- Add the :meth:`~xoa.cf.CFSpecs.reloc` and :meth:`~xoa.cf.CFSpecs.to_loc` methods to :class:`xoa.cf.CFSpecs` for quickly changing the staggered grid indicators in names.
- Add the :meth:`xoa.cf.SGLocator.add_loc` method to quickly change the location markers in a data array.

Breaking changes
----------------
- :func:`xoa.coords.get_dims` is renamed to :func:`xoa.coords.get_cf_dims`.
- The `name` argument of :class:`xoa.cf.CFSpecs` methods is renamed to `cf_name`, and the `dim_type(s)` argument is renamed to `cf_arg(s)`.
- :meth:`xoa.cf.SGLocator.get_location` is renamed to :meth:`~xoa.cf.SGLocator.get_loc_from_da` and the :meth:`~xoa.cf.SGLocator.get_loc` is added.

Bug fixes
---------
- Fix the output formatting of :func:`xoa.grid.dz2depth`.

Documentation
-------------
- The :ref:`indepth.meta` section and :ref:`sphx_glr_tutorials_evaluation_plot_hycom_gdp.py` example are adapted to reflect changes.


v0.3.1 (2021-05-21)
===================

New features
------------
- Add an `autolim` keyword to :func:`xoa.plot.plot_flow` to speedup the processing with cartopy maps.

Breaking changes
----------------
- Rename the `cf` and `sigma` keyword of :func:`xoa.register_accessors` respectively to `xcf` and `decode_sigma` to match the default name of accessors.
- Rename the `sigma` accessor to `decode_sigma`.

Bug fixes
---------
- Fix the access to the xoa executable on windows.
- Fix the minimal version for xarray [:pull:`23`].

Documentation
-------------
- Add a "How to start" section.
- Accessors are now documented separately with `sphinx-autosummary-accessors` [:pull:`20`].
- The Hycom-GDP example now uses :func:`xoa.plot.plot_flow`.


v0.3.0 (2021-05-12)
===================

New features
------------
- Add the :func:`xoa.plot.plot_flow` function [:pull:`9`].
- Improve :func:`xoa.coords.get_depth` so that it can compute depth from sigma coordinates or layer thicknesses [:pull:`8`].
- Add the :func:`xoa.dyn.flow2d` function [:pull:`7`].
- Add the :func:`xoa.regrid.extrap1d` function.
- Add the :func:`xoa.filter.erode_coast` function which is specialized version of the :func:`xoa.filter.erode_mask` for horizontal data.
- Add the :func:`xoa.coords.get_xdim`, :func:`~xoa.coords.get_ydim`, :func:`~xoa.coords.get_zdim`, :func:`~xoa.coords.get_tdim` and :func:`~xoa.coords.get_fdim` for quickly finding standard dimensions.

Bug fixes
---------
- Fix u and v CF config [:pull:`6`]


YYYY.M.MICRO
============

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

