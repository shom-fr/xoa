.. _indepth.meta:

Metadata, CF and naming conventions
###################################

.. warning::
    Have a quick look at the appendix :ref:`appendix.meta.default`
    before going any further!

Introduction
============

The :mod:`xoa.meta` subpackage is an application and extension of a subset of the
`CF conventions <http://cfconventions.org/>`_ in application
to naming conventions.
It has two main intents:

* Searching :class:`xarray.DataArray` variables or coordinates within another
  :class:`xarray.DataArray` or a :class:`xarray.Dataset`,
  by scanning name and attributes like ``units`` and ``standard_name``.
* Formatting :class:`xarray.DataArray` variables or coordinates with
  unique ``name``, and ``standard_name``, ``long_name`` and ``units``
  attributes, with support of staggered grid location syntax.

The module offers capabilities for the user to extend and specialize
default behaviors for user's special datasets.

:mod:`xoa` provides **ready-to-use configurations** for decoding and encoding a few **standard dataset types**.
Look at :ref:`appendix.meta.specialized`.


.. note:: This module shares common features with the excellent and long
    awaited `cf_xarray <https://cf-xarray.readthedocs.io/en/latest/>`_
    package and started a long time before with the Vacumm package.
    The most notable differences include:

    - The current module is also designed for data variables and dimensions, not only
      coordinates.
    - It searches for items not only using standard_names but also
      specialized names. It means that **it works even with datasets that are not well formatted**.
    - It is not only available as accessors, but also as independent
      specification objects that can be configured by the user
      for each type of dataset.


Quick start with accessors
==========================

The easiest way to use :mod:`xoa.meta` is through xarray accessors.
Let's set up a sample dataset and register the accessors:

.. ipython:: python

    from xoa import meta
    meta_specs = meta.get_meta_specs()

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

.. ipython:: python

    import xoa
    xoa.register_accessors()

All these arrays are CF compliant according to their
``standard_name`` attribute, despite their name is not really explicit.
The accessors make it easy to find, format and chain operations:

.. ipython:: python
    :okwarning:

    temp
    temp.meta.get("lon") # access by .get
    ds.meta.get("temp") # access by .get
    ds.meta.lon # access by attribute
    ds.meta.coords.lon  # specific search = ds.meta.coords.get("lon")
    ds.meta.temp # access by attribute
    ds.meta["temp"].name # access by item
    ds.meta.data_vars.temp.name  # specific search = ds.meta.data_vars.get("temp")
    ds.meta.data_vars.bathy is None # returns None when not found
    ds.meta.temp.meta.lon.name  # chaining
    ds.meta.temp.meta.name # generic meta name, not real name
    ds.meta.temp.meta.attrs # attributes, merged with CF attrs
    ds.meta.temp.meta.standard_name # single attribute
    ds.mytemp.meta.auto_format()
    ds.meta.auto_format()

Accessing an accessor attribute or item makes an
implicit call to :meth:`~xoa.accessors.MetaDataArrayAccessor.get`.
The accessor gives access to
two sub-accessors, :attr:`~xoa.accessors.MetaDatasetAccessor.data_vars`
and :attr:`~xoa.accessors.MetaDatasetAccessor.coords`,
for specializing the searches.

.. seealso::
    :class:`xoa.accessors.MetaDataArrayAccessor`
    :class:`xoa.accessors.MetaDatasetAccessor`


Searching and matching
======================

Under the hood, accessor lookups are powered by the :class:`~xoa.meta.general.MetaSpecs`
methods. You can use them directly.

Check if arrays match known or explicit meta items:

.. ipython:: python

    meta_specs.coords.match(lon, "lon") # explicit
    meta_specs.coords.match(lon, "lat") # explicit
    meta_specs.coords.match(lon) # any known
    meta_specs.data_vars.match(temp) # any known
    meta_specs.data_vars.match(sal) # any known

Search for known meta items:

.. ipython:: python

    mytemp = meta_specs.search(ds, "temp")
    mylon = meta_specs.search(mytemp, "lon")

Datasets are searched for data variables ("data_vars") and
data variables are searched for coordinates ("coords").
You can also search for coordinates in datasets, for instance like this:

.. ipython:: python

    meta_specs.coords.search(ds, "lon")

.. seealso::
    - Meta items:
      :metacoord:`lon` :metacoord:`lat` :metadatavar:`temp` :metadatavar:`sal`
    - Methods: :meth:`xoa.meta.MetaCoordSpecs.match`
      :meth:`xoa.meta.MetaVarSpecs.match` :meth:`xoa.meta.general.MetaSpecs.search`
      :meth:`xoa.meta.MetaCoordSpecs.search` :meth:`xoa.meta.MetaVarSpecs.search`


Formatting, encoding and decoding
==================================

The idea
--------
Formatting means changing or setting names and attributes.
It is possible to format, or even auto-format data variables and coordinates.

During an auto-formatting, each array is matched against meta specs,
and the array is formatted when a matching is successful.
If the array contains coordinates, the same process is applied on them,
as soon as the ``format_coords`` keyword is ``True``.
If the ``name`` key is set in the matching item specs, its value can be used
to name the variable or coordinate array, else the generic name is used
like ``"lon"``.

**Explicit formatting:**

.. ipython:: python

    meta_specs.format_coord(lon, "lon")
    meta_specs.format_data_var(temp, "temp")

**Auto-formatting:**

.. ipython:: python

    ds2 = meta_specs.auto_format(ds)
    ds2.temp
    ds2.lon

It can be applied to a data array or a full dataset.

.. seealso::
    :meth:`xoa.meta.general.MetaSpecs.format_coord`
    :meth:`xoa.meta.general.MetaSpecs.format_data_var`
    :meth:`xoa.meta.general.MetaSpecs.auto_format`

Encoding and decoding
---------------------

By default, formatting renames known arrays to their generic name,
like "temp" in the example above. We speak here of **encoding**.
If the ``specialize`` keyword is set to ``True``, arrays are
renamed with their specialized name if set in the specs with the
:metaopt:`name <[data_vars] [__many__] name>` option.
We speak here of **decoding**.
Two shortcut methods exists for these tasks:

- Decoding: :meth:`~xoa.meta.general.MetaSpecs.decode`
- Encoding: :meth:`~xoa.meta.general.MetaSpecs.encode`

Chaining the two methods should lead to the initial dataset or data array.
See the last section of this page for an example:
:ref:`indepth.meta.croco`.


.. seealso::
    :meth:`xoa.meta.general.MetaSpecs.decode`
    :meth:`xoa.meta.general.MetaSpecs.encode`

Staggered grid locations
------------------------

A :class:`~xoa.meta.general.MetaSpecs` instance comes with a :class:`~xoa.meta.SGLocator`
that is accessible through the :attr:`~xoa.meta.general.MetaSpecs.sglocator` attribute.
A :class:`~xoa.meta.SGLocator` helps parsing and formatting staggered grid
location from :attr:`name`, :attr:`standard_name` and :attr:`long_name`
data array attributes.
It is configured at the :metasec:`sglocator <[sglocator]>` section
in which you can specify the
format of the name (:metaopt:`name_format <[sglocator] name_format>` and
the allowed location names :metaopt:`allowed_locations <[sglocator] allowed_locations>`.

.. ipython:: python

    sglocator = meta_specs.sglocator
    sglocator.formats
    sglocator.update(name_format="{root}_{loc}")
    sglocator.format_attr("name", "lon", "rho")
    sglocator.format_attr("standard_name", "lon", "rho")
    sglocator.format_attr("long_name", "lon", "rho")
    sglocator.parse_attr("name", "lon_rho")


Understanding the specifications
================================

Scanning and formatting actions are based on specifications,
and this module natively includes a
:ref:`default configuration <appendix.meta.default>`
for various oceanographic, sea surface and atmospheric variables and coordinates.
A distinction is made between
data variables (:ref:`data_vars <appendix.meta.data_vars>`)
and coordinates (:ref:`coords <appendix.meta.coords>`), like in :mod:`xarray`.

Getting the current specifications for data variables and coordinates
with the :func:`~xoa.meta.get_meta_specs` function:

.. ipython:: python

    meta_specs["data_vars"] is meta_specs.data_vars
    meta_specs["coords"] is meta_specs.coords
    meta_specs.data_vars.names[:3]
    meta_specs.coords.names[:3]

See the appendix :ref:`appendix.meta.default` for the
list of available default specifications.

An instance of the :class:`MetaSpecs` has other configuration
:ref:`sections <appendix.meta.sections>`
than
:metasec:`data_vars <[data_vars]>`,
:metasec:`coords <[coords]>` and
:metasec:`dims <[dims]>`:
:metasec:`registration <[registration]>`,
:metasec:`sglocator <[sglocator]>`,
:metasec:`vertical <[vertical]>` and
:metasec:`accessors <[accessors]>`.


Data variables
--------------

Here is the example of the :metadatavar:`sst` data variable:

.. ipython:: python

    from pprint import pprint
    pprint(meta_specs.data_vars['sst'])


Description of specification keys:

.. list-table:: meta specs for :ref:`appendix.meta.data_vars`

    * - Key
      - Type
      - Description
    * - ``name``
      - str
      - Specialized name for decoding and encoding which is empty by default
    * - ``alt_names``
      - list(str)
      - Alternate names for decoding
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
      - Inherit specifications from another data variable
    * - ``select``
      - eval
      - Item selection evaluated and applied to the array
    * - ``squeeze``
      - list(str)
      - List of dimensions that must be squeezed out

.. note:: The ``standard_name``, ``long_name`` and ``units`` specifications are
    internally stored as a dict in the ``attrs`` key.

Get the specialized name and the attributes only:

.. ipython:: python

    meta_specs.data_vars.get_name("sst")
    meta_specs.data_vars.get_attrs("sst")


Coordinates
-----------

Here is the example of the :metacoord:`lon` coordinate:

.. ipython:: python

    from pprint import pprint
    pprint(meta_specs.coords['lon'])

Description of specification keys:

.. list-table:: meta specs for :ref:`appendix.meta.coords`

    * - Key
      - Type
      - Description
    * - ``name``
      - str
      - Specialized name for decoding and encoding which is empty by default
    * - ``alt_names``
      - list(str)
      - Alternate names for decoding
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
      - Inherit specifications from another data variable or coordinate

.. note:: Like for data variables, the ``standard_name``, ``long_name``, ``units``
    and ``axis`` specifications are internally stored as a dict in the ``attrs`` key.

Get the specialized name and the attributes only:

.. ipython:: python

    meta_specs.coords.get_name("lon")
    meta_specs.coords.get_attrs("lon")


Dimensions
----------

.. list-table:: meta specs for :ref:`appendix.meta.dims`

    * - Key
      - Type
      - Description
    * - ``x``
      - list(str)
      - Possible names of X-type dimensions
    * - ``y``
      - list(str)
      - Possible names of Y-type dimensions
    * - ``z``
      - list(str)
      - Possible names of Z-type vertical dimensions
    * - ``t``
      - list(str)
      - Possible names of time dimensions
    * - ``f``
      - list(str)
      - Possible names of forecast dimensions


This section of the specs defines the names that the dimensions
of type ``x``, ``y``, ``z``, ``t`` and ``f`` (forecast) can take.
It is filled automatically by default with possible names (``name`` and ``alt_names``)
of the coordinates that have their ``axis`` defined.
The user can also add more specific names.

These lists of names are used when searching for **dimensions that are not coordinates**:
since they don't have attributes, their type can only be guessed from their name.

.. warning:: It is recommended to not fill the **dims** section,
    but fill the coordinate section instead.

Here is the default content:

.. ipython:: python

    meta_specs.dims  # or meta_specs["dims"]


Other sections
--------------

.. ipython:: python

    meta_specs["register"]
    meta_specs["sglocator"]
    meta_specs["accessors"]
    meta_specs["vertical"]


Customizing the specifications
==============================

Default user file
-----------------

The :mod:`xoa.meta` module has internal defaults as shown
in appendix :ref:`appendix.meta.default`.

You can extend these defaults with a user file,
whose location is printable with the following command,
at the line containing "user meta specs file":

.. command-output:: xoa info paths

Updating current specs
----------------------

The current specs can be updated with different methods.

From a well **structured dictionary**:

.. ipython:: python

    meta_specs.load_cfg({"data_vars": {"banana": {"standard_name": "banana"}}})
    meta_specs.data_vars["banana"]

From a **configuration file**: instead of the dictionary as an argument
to :meth:`~xoa.meta.general.MetaSpecs.load_cfg` method, you can give either a
file name or a **multi-line string** with the same content as
the file.
Following the previous example:

.. code-block:: ini

    [data_vars]
        [[banana]]
            standard_name: banana

If you only want to update a :attr:`~xoa.meta.general.MetaSpecs.category`,
you can use such method (here :meth:`~xoa.meta.MetaVarSpecs.set_specs`):

.. ipython:: python

    meta_specs.data_vars.set_specs("banana", name="bonono")
    meta_specs.data_vars["banana"]["name"]

Alternatively, a :class:`xoa.meta.general.MetaSpecs` instance can be loaded
with the :meth:`~xoa.meta.general.MetaSpecs.load_cfg` method, as explained below.

Creating new specs
------------------

To create new specs, you must instantiate the :class:`xoa.meta.general.MetaSpecs` class,
with an input type as those presented above:

- A config file name.
- A multi-line string in the format of a config file.
- A dictionary.
- A :class:`configobj.ConfigObj` instance.
- Another :class:`~xoa.meta.general.MetaSpecs` instance.
- A list of them, with the first having priority over the lasts.

The initialization also accepts two options:

- ``default``: whether to load or not the default internal config.
- ``user``: whether to load or not the user config file.

A config created **from default and user configs**:

.. ipython:: python

    banana_specs = {"data_vars": {"banana": {"attrs": {"standard_name": "banana"}}}}
    mymeta_specs = meta.general.MetaSpecs(banana_specs)
    mymeta_specs["data_vars"]["sst"]["attrs"]["standard_name"]
    mymeta_specs["data_vars"]["banana"]["attrs"]["standard_name"]

A config created **from scratch**:

.. ipython:: python

    mymeta_specs = meta.general.MetaSpecs(banana_specs, default=False, user=False)
    mymeta_specs.pprint(depth=2)

A config created **from two other configs**:

.. ipython:: python

    meta_specs_banana = meta.general.MetaSpecs(banana_specs, default=False, user=False)
    apple_specs = {"data_vars": {"apple": {"attrs": {"long_name": "Big apple"}}}}
    meta_specs_apple = meta.general.MetaSpecs(apple_specs, default=False, user=False)
    meta_specs_fruits = meta.general.MetaSpecs((meta_specs_apple, meta_specs_banana),
        default=False, user=False)
    meta_specs_fruits.data_vars.names

Temporarily replacing specs
---------------------------

As shown before, the current meta specs are accessible with the
:func:`xoa.meta.get_meta_specs` function.
You can replace them with the :class:`xoa.meta.set_meta_specs` class,
to be used as a function.

.. ipython:: python

    meta_specs_old = meta.get_meta_specs()
    meta.set_meta_specs(meta_specs_banana)
    meta.get_meta_specs() is meta_specs_banana
    meta.set_meta_specs(meta_specs_old)
    meta.get_meta_specs() is meta_specs_old


In case of a temporary change, you can use :class:`~xoa.meta.set_meta_specs`
in a context statement:

.. ipython:: python

    with meta.set_meta_specs(meta_specs_banana) as myspecs:
        print('inside', meta.get_meta_specs() is meta_specs_banana)
        print('inside', myspecs is meta.get_meta_specs())
    print('outside', meta.get_meta_specs() is meta_specs_old)

For convenience, you can set specs directly with a dictionary:

 .. ipython:: python

    with meta.set_meta_specs({"data_vars": {"apple": {}}}) as myspecs:
        print("apple" in meta.get_meta_specs())
    print("apple" in meta.get_meta_specs())

Application with an accessor usage:

.. ipython:: python


    data = xr.DataArray([5], attrs={'standard_name': 'sea_surface_banana'})
    ds = xr.Dataset({'toto': data})
    mymeta_specs = meta.general.MetaSpecs({"data_vars": {"ssb":
        {"standard_name": "sea_surface_banana"}}})
    with meta.set_meta_specs(mymeta_specs):
        print(ds.meta.get("ssb"))


Registering and inferring specs
-------------------------------

It is possible to register specialized :class:`~xoa.meta.general.MetaSpecs` instances
with :func:`~xoa.meta.register_meta_specs` for future access.

Here we register new specs with an internal registration name ``"mycroco"``:

.. ipython:: python

    content = {
        "register": {
            "name": "mycroco"
        },
        "data_vars": {
            "temp": {
                "name": "supertemp"
            }
        },
        "coords": {
            "lon": {
                "name": "mylon"
            }
        }
    }
    mymeta_specs = meta.general.MetaSpecs(content)
    meta.register_meta_specs(mymeta_specs)

We can now access it with the :func:`~xoa.meta.get_meta_specs` function:

.. ipython:: python

    these_meta_specs = meta.get_meta_specs('mycroco')
    these_meta_specs is mymeta_specs

If you set the :attr:`meta_specs` attribute or encoding of a dataset
to the name of a registered :class:`~xoa.meta.general.MetaSpecs` instance, you can
get it automatically with the :func:`~xoa.meta.get_meta_specs` function.

Let's register another :class:`~xoa.meta.general.MetaSpecs` instance:

.. ipython:: python

    content = {
        "register": {
            "name": "myhycom"
        },
        "data_vars": {
            "sal": {
                "name": "supersal"
            }
        },
    }
    mymeta_specs2 = meta.general.MetaSpecs(content)
    meta.register_meta_specs(mymeta_specs2)

Let's create a dataset:

.. ipython:: python

    ds = xr.Dataset({'supertemp': ("mylon", [0, 2])}, coords={"mylon": [10, 20]})

Now find the best registered specs instance which has either the name
``myhycom`` or ``mycroco``:


.. ipython:: python

    meta_specs_auto = meta.infer_meta_specs(ds)
    print(meta_specs_auto.name)
    ds_decoded = meta_specs_auto.decode(ds)
    ds_decoded
    meta_specs_auto.encode(ds)

It is ``mycroco`` as expected.

Assigning specs to datasets
----------------------------

All xoa routines that need to access specific coordinates
or variables try to infer the appropriate specs, which default
to the current specs.
When the :attr:`meta_specs` **attribute** or **encoding** is set,
:func:`~xoa.meta.get_meta_specs` uses it to search within
registered specs.

.. ipython:: python

    ds.encoding.update(meta_specs="mycroco")
    meta_specs = meta.get_meta_specs(ds)
    meta_specs.encode(ds)

The :attr:`meta_specs` encoding is set at the dataset level,
not at the data array level:

.. ipython:: python

    meta.get_meta_specs(ds.supertemp) is meta_specs

To propagate to all the data arrays, use :func:`~xoa.meta.assign_meta_specs`:

.. ipython:: python

    meta.assign_meta_specs(ds, "mycroco")
    ds.mylon.encoding
    meta.get_meta_specs(ds.supertemp) is meta_specs


Specialized helper modules
==========================

The :mod:`xoa.meta` module provides the foundation for finding and identifying
variables and coordinates. Building on this, xoa provides specialized
modules with convenient functions for common oceanographic and atmospheric variables.
All these functions use the meta specs under the hood.

Finding coordinates with :mod:`xoa.coords`
-------------------------------------------

The :mod:`xoa.coords` module provides high-level functions to find specific
coordinates in a dataset or data array.

Available coordinate finder functions:

- :func:`~xoa.coords.get_lon` / :func:`~xoa.coords.is_lon`: Longitude
- :func:`~xoa.coords.get_lat` / :func:`~xoa.coords.is_lat`: Latitude
- :func:`~xoa.coords.get_depth` / :func:`~xoa.coords.is_depth`: Depth
- :func:`~xoa.coords.get_altitude` / :func:`~xoa.coords.is_altitude`: Altitude
- :func:`~xoa.coords.get_level` / :func:`~xoa.coords.is_level`: Generic vertical level
- :func:`~xoa.coords.get_vertical`: Any vertical coordinate (depth, altitude, or level)
- :func:`~xoa.coords.get_time` / :func:`~xoa.coords.is_time`: Time

Dimension finder functions:

- :func:`~xoa.coords.get_xdim` / :func:`~xoa.coords.get_ydim` /
  :func:`~xoa.coords.get_zdim` / :func:`~xoa.coords.get_tdim` /
  :func:`~xoa.coords.get_fdim`

Example usage:

.. ipython:: python

    import xoa.coords

    # Rebuild a sample dataset with coordinates
    lon = xr.DataArray(np.arange(3, dtype='d'), dims='mylon',
        attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(np.arange(20, 23, dtype='d'), dims='mylon',
        coords={'mylon': lon},
        attrs={'standard_name': 'sea_water_temperature'})

    # Get longitude coordinate from a data array
    found_lon = xoa.coords.get_lon(temp)
    print(found_lon.name)

    # Check if a coordinate is latitude
    print(xoa.coords.is_lon(found_lon))
    print(xoa.coords.is_lat(found_lon))

Finding thermodynamic variables with :mod:`xoa.thermdyn`
---------------------------------------------------------

The :mod:`xoa.thermdyn` module provides functions to find thermodynamic variables
like temperature, salinity, and density in datasets.

- :func:`~xoa.thermdyn.get_temp` / :func:`~xoa.thermdyn.is_temp`: Temperature
- :func:`~xoa.thermdyn.get_sal` / :func:`~xoa.thermdyn.is_sal`: Salinity
- :func:`~xoa.thermdyn.get_dens` / :func:`~xoa.thermdyn.is_dens`: Density

Each function accepts a ``variant`` keyword to target a specific type:

- **Temperature variants**: ``"temp"`` (in situ), ``"ptemp"`` (potential),
  ``"ctemp"`` (conservative), ``"atemp"`` (absolute)
- **Salinity variants**: ``"sal"`` (in situ), ``"psal"`` (practical),
  ``"asal"`` (absolute), ``"pfsal"`` (preformed)

.. ipython:: python

    import xoa.thermdyn

    sal = xr.DataArray(np.arange(33, 36, dtype='d'), dims='mylon',
        coords={'mylon': lon},
        attrs={'standard_name': 'sea_water_salinity'})
    ds_thd = xr.Dataset({'mytemp': temp, 'mysal': sal})

    # Find any temperature variable
    found_temp = xoa.thermdyn.get_temp(ds_thd)
    print(found_temp.name)

    # Check if a variable is temperature-like
    print(xoa.thermdyn.is_temp(found_temp))

    # Find salinity
    found_sal = xoa.thermdyn.get_sal(ds_thd)
    print(found_sal.name)

Finding dynamical variables with :mod:`xoa.dyn`
------------------------------------------------

The :mod:`xoa.dyn` module provides functions for ocean dynamics variables,
particularly sea level.

- :func:`~xoa.dyn.get_sea_level`: Find a sea level variable

Supported variants: ``"ssh"`` (sea surface height), ``"adt"`` (absolute dynamic
topography), ``"sla"`` (sea level anomaly), ``"mdt"`` (mean dynamic topography),
``"mss"`` (mean sea surface).

.. seealso::
    :mod:`xoa.coords`, :mod:`xoa.thermdyn`, :mod:`xoa.dyn`


.. _indepth.meta.croco:

Example: decoding/encoding Croco model outputs
==============================================

Here are the specs for Croco:

.. literalinclude:: ../xoa/meta/configs/croco.cfg
    :language: ini

Register them:

.. ipython:: python

    @suppress
    import xoa, xoa.meta, xoa.meta.configs
    meta_config_file = xoa.meta.get_meta_config_file("croco")
    print(meta_config_file)
    xoa.meta.register_meta_specs(meta_config_file) # xoa.meta.register_meta_specs("croco")
    xoa.meta.is_registered_meta_specs("croco")


Register the :class:`xoa <~xoa.accessors.XoaDatasetAccessor>` accessor:

.. ipython:: python

    xoa.register_accessors()

Now let's open a Croco sample as a xarray dataset:

.. ipython:: python

    ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.meridional.nc")
    ds

Let's **decode** it:

.. ipython:: python

    dsd = ds.xoa.decode()
    dsd

Let's **re-encode** it!

.. ipython:: python

    dse = dsd.xoa.encode()
    dse

Et voilà !

.. _indepth.meta.hycom:

Example: decoding and merging Hycom splitted outputs
====================================================

At Shom, the Hycom model outputs are splitted into separate files, one for each variable.
Conflicts may occur when using variables that are not at the same staggered grid location,
since all variables are stored with dimensions ``(y, x)`` and ``lon`` and ``lat``
coordinates, all with the same name.
To avoid this problem, the horizontal dimensions and coordinates of
staggered variables are renamed to indicate their location.

Here are the specs to take care of the staggered grid indicators in the names:

.. literalinclude:: ../xoa/meta/configs/hycom.cfg
    :language: ini

Note the ``add_coords_loc`` sections.
Location is not added to the ``u`` and ``v`` variables
but to their dimensions and coordinates.

Register them:

.. ipython:: python

    @suppress
    import xoa, xoa.meta
    xoa.meta.register_meta_specs("hycom")
    xoa.meta.is_registered_meta_specs("hycom")

Overview of the U dataset:

.. ipython:: python

    dsu = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.u.nc")
    dsu

Decoding:

.. ipython:: python

    dsu = dsu.xoa.decode()
    dsu

Now we can read, decode and merge all files without any conflict:

.. ipython:: python

    dsv = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.u.nc").xoa.decode()
    dsh = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.h.nc").xoa.decode()
    print(dsu)
    ds = xr.merge([dsu, dsv, dsh])
    ds
