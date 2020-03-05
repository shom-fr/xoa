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
    print(cfspecs.data_vars.names[:3])
    print(cfspecs.coords.names[:3])

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
