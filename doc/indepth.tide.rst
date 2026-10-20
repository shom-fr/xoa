.. _indepth.tide:

Tides
#####

Introduction
============

:mod:`xoa.tide` analyses and predicts the tide of sea level and currents, filters the tide
out of time series and computes tidal diagnostics. The numba and numpy kernels are in
:mod:`xoa.core.tide` and :mod:`xoa.core.signal`.

.. list-table::
    :header-rows: 1
    :widths: 40 60

    * - Function
      - Purpose
    * - :func:`xoa.tide.harmonic_analysis`, :func:`~xoa.tide.harmonic_analysis_uv`
      - Amplitudes and phases of sea level, and ellipses of currents
    * - :func:`xoa.tide.harmonic_predict`, :func:`~xoa.tide.harmonic_predict_uv`
      - Tide reconstructed from the coefficients
    * - :func:`xoa.tide.tidal_ellipses`, :func:`~xoa.tide.ellipse_outline`
      - Descriptors and outlines of current ellipses
    * - :func:`xoa.tide.high_waters`, :func:`~xoa.tide.skew_surge`, :func:`~xoa.tide.tidal_range`
      - High waters, skew surge and range of a series
    * - :func:`xoa.tide.tidal_filter`
      - Demerliac, Godin and mean filters

The tutorials :ref:`sphx_glr_tutorials_tides_plot_tide_station.py` and
:ref:`sphx_glr_tutorials_tides_plot_tide_map.py` show them at work, on a series and
on gridded fields.

.. _indepth.tide.constituents:

Constituents
============

The constituents are defined in a database: Doodson number, nodal satellites and,
for the shallow-water ones like M4, the constituents they are made of.
Their angular frequencies come from :func:`~xoa.tide.get_frequencies`.

.. ipython:: python

    import xoa.tide as xtide
    xtide.CONSTITUENT_NAMES
    2 * 3.14159 / xtide.get_frequencies(["M2", "K1"]) / 3600  # periods in hours

The database is read from ``tide.cfg`` and checked with the specifications of ``tide.ini``,
using :mod:`xoa.cfgm`. A ``tide.cfg`` file located next to the xoa user config file
(see :func:`xoa.get_default_user_config_file`) completes or overrides the default sections.

Two constituents can only be separated by an analysis if the record is longer than
their beat period (Rayleigh criterion). :func:`~xoa.tide.auto_constituents` selects
the constituents that a record resolves, :func:`~xoa.tide.beat_period` and
:func:`~xoa.tide.unresolved_pairs` check your own choice.

.. _indepth.tide.harmonic:

Harmonic analysis
=================

The sea level is fitted with a mean and the sine and cosine of each frequency, by least
squares. The design matrix only depends on the time axis, so its pseudo-inverse is computed
once and applied to all the points with a single matrix product.
:class:`~xoa.tide.TidalHarmonic` keeps it for several analyses of the same time axis.

The data must have time as their first dimension, followed by any others like ``y`` and ``x``.
The result is a :class:`xarray.Dataset` indexed by ``constituent``:

- ``h0``, ``amp`` and ``phase``, with the white-noise uncertainties ``amp_std``
  and ``phase_std``,
- ``t_ref``, the middle of the record that the phases refer to, in the attributes.

Missing values propagate: a point with a gap gives NaN coefficients.
Do not chunk along time when using dask arrays; chunks along the other dimensions are fine.

.. rubric:: Pitfalls

- **Phase convention**: phases are *local* by default, i.e. relative to ``t_ref``.
  Use ``greenwich=True`` to get the Greenwich phase lag that tidal atlases use.
  The convention is stored in the ``phase_convention`` attribute and used when predicting.
- **Nodal corrections** account for the 18.6 year modulation of the amplitudes and phases.
  They are on by default and matter for records of a year or more, and for predictions.
- **Uncertainties** assume white residuals. They are too small when the residual is a
  surge: :meth:`~xoa.tide.TidalHarmonic.ensemble` accepts an ``inflate`` factor, 3 to 5 being typical.
- **Currents** are analysed as rotating ellipses: the semi-minor axis is negative for a
  clockwise rotation, the inclination is relative to east and angles are in radians.

.. _indepth.tide.diags:

Diagnostics
===========

The diagnostics work on regularly sampled series with time as first dimension, at all points
in parallel. The high waters are the local maxima separated by at least ``min_separation`` samples.
The skew surge is the difference between the highest observed level around a predicted
high water and the predicted level, whatever the timing offset.

:func:`~xoa.tide.form_number` classifies the tide from the K1, O1, M2 and S2 amplitudes:
below 0.25 it is semi-diurnal, above 3 diurnal and mixed in between.

.. _indepth.tide.plot:

Plots
=====

The :mod:`xoa.tide.plot` module, imported apart because it needs matplotlib and cartopy,
draws the results of the analyses:

- :func:`~xoa.tide.plot.plot_cotidal`: amplitude in colors and phase in lines of a sea level constituent,
- :func:`~xoa.tide.plot.plot_tidal_ellipses` and :func:`~xoa.tide.plot.plot_tidal_constituents`:
  maps of current ellipses, coloured by rotation sense, phase, ellipticity or semi-major axis,
  at geographic scale and optionally over a background field,
- :func:`~xoa.tide.plot.plot_tidal_ellipse`: one annotated ellipse,
- :func:`~xoa.tide.plot.plot_tidal_spectrum`: amplitudes against frequencies.

On a map, the scale of the ellipses is automatic and set from the distance between the points:
use ``step`` to thin dense grids and ``scale`` to enlarge weak constituents.

.. _indepth.tide.filters:

Tidal filters
=============

Oceanographic time series often contain a tidal signal that must be removed.
:func:`~xoa.tide.tidal_filter` convolves a series with weights that are designed for
hourly data, with the following names:

- **demerliac**: Demerliac filter, 25 hours,
- **godin**: Godin filter, 70 hours,
- **mean**: simple 24 hour moving average.

.. ipython:: python

    import numpy as np, pandas as pd, xarray as xr
    time = pd.date_range("2020-01-01", periods=30 * 24, freq="h")
    hours = np.arange(len(time))
    sea_level = xr.DataArray(
        0.5 * hours / 24 + 0.3 * np.sin(2 * np.pi * hours / 12.42),
        coords={"time": time}, dims="time",
    )
    filtered = xtide.tidal_filter(sea_level, "demerliac", na_thres=0.2)

Notes:

- the data must have a time dimension, with a regular time step (``dt_tol``)
  that is an hour or less: weights are interpolated for finer steps,
- ``na_thres`` is the tolerance to missing values, see :func:`xoa.filter.convolve`,
- :func:`xoa.filter.tidal_filter` and :func:`xoa.filter.demerliac` are deprecated aliases.

See the tutorial :ref:`sphx_glr_tutorials_filtering_plot_filter_time.py`.
