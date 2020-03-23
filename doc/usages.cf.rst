Using the :mod:`xoa.cf` module
##############################

Introduction
============

This module is an application and extension of a subset of the
`CF conventions <http://cfconventions.org/>`_.
It has two intents:

* Searching :class:`xarray.DataArray` variables or coordinates with another :class:`xarray.DataArray` or a :class:`xarray.Dataset`, by scanning name and attributes like ``units`` and ``standard_name``.
* Formatting :class:`xarray.DataArray` variables or coordinates with unique name and ``standard_name`` and ``long_name`` attributes, with support of staggered grid location syntax.

Accessing the current specifications
====================================
Scanning and formatting actions are based on specifications, and this module natively includes a default configuration for various oceanographic, sea surface and atmospheric variables and coordinates. A distinction is made between data variables ``data_vars`` and coordinates ``coords``, like in :mod:`xarray`.

Getting the current specifications for data variables and coordinates:

.. ipython:: python

    from xoa import cf
    cfspecs = cf.get_cf_specs()
    cfspecs["data_vars"] is cfspecs.data_vars
    cfspecs["data_vars"] is cfspecs.coords
    cfspecs.data_vars.names[:3]
    cfspecs.coords.names[:3]

Data variables
--------------

.. ipython:: python

    from pprint import pprint
    pprint(cfspecs.data_vars['sst'])


Description of specification keys:

.. list-table:: CF specs for ``data_vars``

    * - Key
      - Type
      - Description
    * - name
      - list(str)
      - Names
    * - ``standard_name``
      - list(str)
      - "standard_name" attributes
    * - ``long_name``
      - list(str)
      - "long_name" attributes
    * - ``units``
      - list(str)
      - "units" attributes
    * - ``domain``
      - choice
      - Domain of application, within {'generic', 'atmos', 'ocean', 'surface'}
    * - ``search_order``
      - str
      - Search order within properties as combination of letters: `[n]name`, `[s]tandard_name`, `[u]nits`
    * - ``cmap``
      - str
      - Colormap specification
    * - ``inherit``
      - str
      - Inherit specification from another data variable
    * - ``select``
      - eval
      - Item selection evaluated and applied to the array
    * - ``squeeze``
      - list(str)
      - List of dimensions that must be squeezed out

Coordinates
-----------

.. ipython:: python

    from pprint import pprint
    pprint(cfspecs.coords['lon'])

Description of specification keys:

.. list-table:: CF specs for ``coords``

    * - Key
      - Type
      - Description
    * - name
      - list(str)
      - Names
    * - ``standard_name``
      - list(str)
      - "standard_name" attributes
    * - ``long_name``
      - list(str)
      - "long_name" attributes
    * - ``units``
      - list(str)
      - "units" attributes
    * - ``axis``
      - str
      - "axis" attribute like X, Y, Z, T or F
    * - ``search_order``
      - str
      - Search order within properties as combination of letters: `[n]name`, `[s]tandard_name`, `[u]nits`
    * - ``inherit``
      - str
      - Inherit specification from another data variable

Searching within a :class:`~xarray.Dataset` or  :class:`~xarray.DataArray`
==========================================================================

Let's define a minimal dataset:

.. ipython:: python

    @suppress
    import xarray as xr, numpy as np
    nx = 3
    lon = xr.DataArray(np.arange(3, dtype='d'), dims='mylon',
        attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(np.arange(20, 23, dtype='d'), dims='mylon',
        coords={'mylon': lon},
        attrs={'standard_name': 'sea_water_temperature'})
    sal = xr.DataArray(np.arange(33, 36, dtype='d'), dims='mylon',
        coords={'mylon': lon},
        attrs={'standard_name': 'sea_water_salinity'})
    ds = xr.Dataset({'mytemp': temp, 'mysal': sal})

All these arrays are CF compliant according to their
``standard_name`` attribute, despite their name is not really explicit.

Check if they match known or explicit CF items:

.. ipython:: python

    cfspecs.coords.match(lon, "lon") # explicit
    cfspecs.coords.match(lon, "lat") # explicit
    cfspecs.coords.match(lon)
    cfspecs.data_vars.match(temp)
    cfspecs.data_vars.match(sal)

Search for known CF items:

.. ipython:: python

    mytemp = cfspecs.search(ds, "temp")
    mylon = cfspecs.search(mytemp, "lon")

Datasets are searched for data variables ("data_vars") and
data variables are searched for coordinates ("coords").
You can also search for coordinates in datasets, for instance like this:

.. ipython:: python

    cfspecs.coords.search(ds, "lon")

Formatting
==========

It is possible to format, or even auto-format data variables and coordinates.

During an auto-formatting, each array is matched against CF specs,
and the array is formatting when a matching is successfull.
If the array contains coordinates, the same process is applied on them,
as soon as the ``format_coords`` keyword is ``True``.

**Explicit formatting:**

.. ipython:: python

    cfspecs.format_coord(lon, "lon")
    cfspecs.format_data_var(temp, "temp")

**Auto-formatting:**

.. ipython:: python

    ds2 = cfspecs.auto_format(ds)
    ds2.temp
    ds2.lon


Using the accessors
===================

Accessors for :class:`xarray.Dataset` and :class:`xarray.DataArray`
can be registerd with the :func:`xoa.cf.register_cf_accessors`:

.. ipython:: python

    cf.register_cf_accessors()

These accessors make it easy to use some of the :class:`xoa.cf.CFSpecs`
capabilities.
Here are some example of usage:

.. ipython:: python

    temp
    temp.cf.get("lon")
    ds.cf.get("temp")



