.. _uses.meta:

Naming conventions with :mod:`xoa.meta`
#######################################

.. warning::
    Have a quick loop to the appendix :ref:`appendix.meta.default`
    before going any further!

Introduction
============

This module is an application and extension of a subset of the
`CF conventions <http://cfconventions.org/>`_ in application
to naming conventions.
It has two main intents:

* Searching :class:`xarray.DataArray` variables or coordinates with another
  :class:`xarray.DataArray` or a :class:`xarray.Dataset`,
  by scanning name and attributes like ``units`` and ``standard_name``.
* Formatting :class:`xarray.DataArray` variables or coordinates with
  unique ``name``, and ``standard_name``, ``long_name`` and ``units``
  attributes, with support of staggered grid location syntax.

The module offers capabilities for the user to extend and specialize
default behaviors for user's special datasets.

:mod:`xoa` provides **ready-to-use configurations** for decoding and encoding a few **standard dataset types**. 
Look at :ref:`appendix.meta.specialized`.


.. note:: This module shares common feature with the excellent and long
    awaited `cf_xarray <https://cf-xarray.readthedocs.io/en/latest/>`_
    package and started a long time before with the Vacumm package.
    The most notable differences differences include:

    - The current module is also designed for data variables an dimensions, not only
      coordinates.
    - It searches for items not only using standard_names but also
      specialized names. It means that **it works even with datasets that are not well formatted**.
    - It is not only available as accessors, but also as independant
      specification objects that can be configured by the user
      for each type of dataset.



:mod:`xoa` make available **ready-to-use configurations** for decoding and encoding a few **standard dataset types**. 
Looks at :ref:`appendix.meta.specialized`.

Accessing the current specifications
====================================

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

    from xoa import meta
    meta_specs = meta.get_meta_specs()
    meta_specs["data_vars"] is meta_specs.data_vars
    meta_specs["data_vars"] is meta_specs.coords
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

Here is the example of the :metadatavar:`lon` coordinate:

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
The user can also add more spcific names.

Theses list of names are used when searching for **dimensions that are not coordinates**:
since they don't have attribute, their type can only be guessed from their name.

.. warning:: It is recommended to not fill the  **dims** section,
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

    meta_specs.coords.match(lon, "lon") # explicit
    meta_specs.coords.match(lon, "lat") # explicit
    meta_specs.coords.match(lon) # any known
    meta_specs.data_vars.match(temp) # any known
    meta_specs.data_vars.match(sal) # any known

Search for known CF items:

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
      :meth:`xoa.meta.MetaVarSpecs.match` :meth:`xoa.meta.MetaSpecs.search`
      :meth:`xoa.meta.MetaCoordSpecs.search` :meth:`xoa.meta.MetaVarSpecs.search`


The staggered-grid locator :class:`~xoa.meta.SGLocator`
=====================================================

A :class:`~xoa.meta.MetaSpecs` instance comes with :class:`~xoa.meta.SGLocator`
that is accessible through the :attr:`~xoa.meta.MetaSpecs.sglocator` attribute.
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
    sglocator.format_attr("name", "lon", "rho")
    sglocator.format_attr("standard_name", "lon", "rho")
    sglocator.format_attr("long_name", "lon", "rho")
    sglocator.parse_attr("name", "lon_rho")

Formatting i.e encoding and decoding
====================================

The idea
--------
Formatting means changing or setting names and attributes.
It is possible to format, or even auto-format data variables and coordinates.

During an auto-formatting, each array is matched against meta specs,
and the array is formatted when a matching is successfull.
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
    :meth:`xoa.meta.MetaSpecs.format_coord`
    :meth:`xoa.meta.MetaSpecs.format_data_var`
    :meth:`xoa.meta.MetaSpecs.auto_format`

Encoding and decoding
---------------------

By default, formatting renames known arrays to their generic name,
like "temp" in the example above. We speak here of **encoding**.
If the ``specialize`` keyword is set to ``True``, arrays are
renamed with their specialized name if set in the specs with the
:metaopt:`name <[data_vars] [__many__] name>` option.
We speak here of **decoding**.
Two shortcut methods exists for these tasks:

- Decoding: :meth:`~xoa.meta.MetaSpecs.decode`
- Encoding: :meth:`~xoa.meta.MetaSpecs.encode`

Chaining the two methods should lead to the initial dataset or data array.
See the last section of this page for an exemple:
:ref:`uses.meta.croco`.


.. seealso::
    :meth:`xoa.meta.MetaSpecs.decode`
    :meth:`xoa.meta.MetaSpecs.encode`

Using the accessors
===================

Accessors for :class:`xarray.Dataset` and :class:`xarray.DataArray`
can be registered with the :func:`xoa.meta.register_meta_specs_accessors`:

.. ipython:: python

    import xoa
    xoa.register_accessors(xmeta=True)

The accessor is named here :class:`xmeta <xoa.accessors.CFDatasetAccessor>`.


.. note:: All xoa accessors can be be registered with
    :func:`xoa.register_accessors`. Note also that all functionalities
    of the :class:`xmeta <xoa.accessors.CFDatasetAccessor>`
    accessor are also available with the more global
    :class:`xoa <xoa.accessors.XoaDatasetAccessor>` accessor.

These accessors make it easy to use some of the :class:`xoa.meta.MetaSpecs`
capabilities.
Here are examples of use:

.. ipython:: python
    :okwarning:

    temp
    temp.xmeta.get("lon") # access by .get
    ds.xmeta.get("temp") # access by .get
    ds.xmeta.lon # access by attribute
    ds.xmeta.coords.lon  # specific search = ds.meta.coords.get("lon")
    ds.xmeta.temp # access by attribute
    ds.xmeta["temp"].name # access by item
    ds.xmeta.data_vars.temp.name  # specific search = ds.meta.coords.get("temp")
    ds.xmeta.data_vars.bathy is None # returns None when not found
    ds.xmeta.temp.xmeta.lon.name  # chaining
    ds.xmeta.temp.xmeta.name # CF name, not real name
    ds.xmeta.temp.xmeta.attrs # attributes, merged with CF attrs
    ds.xmeta.temp.xmeta.standard_name # single attribute
    ds.mytemp.xmeta.auto_format() # or ds.temp.xmeta()
    ds.xmeta.auto_format() # or ds.xmeta()

As you can see, accessing an accessor attribute or item make an
implicit call to :class:`~xoa.meta.DataArrayCFAccessor.get`.
The root accessor :attr:`cf` give access to
two sub-accessors, :attr:`~xoa.meta.DatasetCFAccessor.data_vars`
and :attr:`~xoa.meta.DatasetCFAccessor.coords`,
for being able to specialize the searches.

.. seealso::
    :class:`xoa.meta.DataArrayCFAccessor`
    :class:`xoa.meta.DatasetCFAccessor`

Changing the meta specs
=====================

Default user file
-----------------

The :mod:`xoa.meta` module has internal defaults as shown
in appendix :ref:`appendix.meta.default`.

You can extend these defaults with a user file,
whose location is printable with the following command,
at the line containing "user meta specs file":

.. command-output:: xoa info paths

Update the current specs
------------------------

The current specs can be updated with different methods.

From a well **structured dictionary**:

.. ipython:: python

    meta_specs.load_cfg({"data_vars": {"banana": {"standard_name": "banana"}}})
    meta_specs.data_vars["banana"]

From a **configuration file**: instead of the dictionary as an argument
to :meth:`~xoa.meta.MetaSpecs.load_cfg` method, you can give either a
file name or a **multi-line string** with the same content as
the file.
Following the previous example:

.. code-block:: ini

    [data_vars]
        [[banana]]
            standard_name: banana

If you only want to update a :attr:`~xoa.meta.MetaSpecs.category`,
you can use such method (here :meth:`~xoa.meta.CFVarSpecs.set_specs`):

.. ipython:: python

    meta_specs.data_vars.set_specs("banana", name="bonono")
    meta_specs.data_vars["banana"]["name"]

Alternatively, a :class:`xoa.meta.MetaSpecs` instance can be loaded
with the :meth:`~xoa.meta.CfSpecs.load_cfg` method, as explained below.

Create new specs from scratch
-----------------------------

To create new specs, you must instantiate the :class:`xoa.meta.MetaSpecs` class,
with an input type as those presented above:

- A config file name.
- A Multi-line string in the format of a config file.
- A dictionary.
- A :class:`configobj.ConfigObj` instance.
- Another :class:`~xoa.meta.MetaSpecs` instance.
- A list of them, with the having priority over the lasts.

The initialization also accepts two options:

- ``default``: wether to load or not the default internal config.
- ``user``: wether to load or not the user config file.

An config created **from default and user configs**:

.. ipython:: python

    banana_specs = {"data_vars": {"banana": {"attrs": {"standard_name": "banana"}}}}
    mymeta_specs = meta.MetaSpecs(banana_specs)
    mymeta_specs["data_vars"]["sst"]["attrs"]["standard_name"]
    mymeta_specs["data_vars"]["banana"]["attrs"]["standard_name"]

An config created **from scratch**:

.. ipython:: python

    mymeta_specs = meta.MetaSpecs(banana_specs, default=False, user=False)
    mymeta_specs.pprint(depth=2)

An config created **from two other configs**:

.. ipython:: python

    meta_specs_banana = meta.MetaSpecs(banana_specs, default=False, user=False)
    apple_specs = {"data_vars": {"apple": {"attrs": {"long_name": "Big apple"}}}}
    meta_specs_apple = meta.MetaSpecs(apple_specs, default=False, user=False)
    meta_specs_fruits = meta.MetaSpecs((meta_specs_apple, meta_specs_banana),
        default=False, user=False)
    meta_specs_fruits.data_vars.names

Replacing the currents meta specs
-------------------------------

As shown before, the currents meta specs are accessible with the
:func:`xoa.meta.get_meta_specs` function.
You can replace them with the :class:`xoa.meta.set_meta_specs` class,
to be used as a fonction.

.. ipython:: python

    meta_specs_old = meta.get_meta_specs()
    meta.set_meta_specs(meta_specs_banana)
    meta.get_meta_specs() is meta_specs_banana
    meta.set_meta_specs(meta_specs_old)
    meta.get_meta_specs() is meta_specs_old


In case of a temporary change, you can used :class:`~xoa.meta.set_meta_specs`
in a context statement:

.. ipython:: python

    with meta.set_meta_specs(meta_specs_banana) as myspecs:
        print('inside', meta.get_meta_specs() is meta_specs_banana)
        print('inside', myspecs is meta.get_meta_specs())
    print('outside', meta.get_meta_specs() is meta_specs_old)

For convience, you can set specs directly with a dictionary:

 .. ipython:: python

    with meta.set_meta_specs({"data_vars": {"apple": {}}}) as myspecs:
        print("apple" in meta.get_meta_specs())
    print("apple" in meta.get_meta_specs())

Application with an accessor usage:

.. ipython:: python


    data = xr.DataArray([5], attrs={'standard_name': 'sea_surface_banana'})
    ds = xr.Dataset({'toto': data})
    mymeta_specs = meta.MetaSpecs({"data_vars": {"ssb":
        {"standard_name": "sea_surface_banana"}}})
    with meta.set_meta_specs(mymeta_specs):
        print(ds.xmeta.get("ssb"))


Working with registered specs
=============================

Registering and accessing new specs
-----------------------------------

It is possible to register specialized :class:`~xoa.meta.MetaSpecs` instances
with :func:`~xoa.meta.register_meta_specs` for future access.

Here we register new specs with a internal registration name ``"mycroco"``:

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
    mymeta_specs = meta.MetaSpecs(content)
    meta.register_meta_specs(mymeta_specs)

We can now access with it the :func:`~xoa.meta.get_meta_specs` function:

.. ipython:: python

    these_meta_specs = meta.get_meta_specs('mycroco')
    these_meta_specs is mymeta_specs

Inferring the best specs for my dataset
---------------------------------------

If you set the :attr:`meta_specs` attribute or encoding of a dataset
to the name of a registered :class:`~xoa.meta.MetaSpecs` instance, you can
get it automatically with the :func:`~xoa.meta.get_meta_specs` function.

Let's register another :class:`~xoa.meta.MetaSpecs` instance:

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
    mymeta_specs2 = meta.MetaSpecs(content)
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


Assigning registered specs to a dataset or data array
-----------------------------------------------------

All xoa routines that needs to access specific coordinates
or variables try to infer the approriate specs, which default
to the current specs.
When the :attr:`meta_specs` **attribute** or **encoding** is set,
:meth:`~xoa.meta.get_meta_specs` uses it to search within
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


.. _uses.meta.croco:

Example: decoding/encoding Croco model outputs
==============================================

Here are the specs for Croco:

.. literalinclude:: ../xoa/meta/configs/croco.cfg
    :language: ini

Register them:

.. ipython:: python

    @suppress
    import xoa, xoa.meta, xoa.meta.configs
    cf_config_file = xoa.meta.configs.get_meta_config_file("croco")
    print(cf_config_file)
    xoa.meta.register_meta_specs(cf_config_file) # xoa.meta.register_meta_specs("croco")
    xoa.meta.is_registered_meta_specs("croco")


Register the :class:`xoa <~xoa.accessors.XoaDatasetAccessor>` accessor:

.. ipython:: python

    xoa.register_accessors()

Now let's open a Croco sample as a xarray dataset:

.. ipython:: python

    ds = xoa.open_data_sample("croco.south-africa.meridional.nc")
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

.. _uses.meta.hycom:

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
    import xoa, xoa.cf
    xoa.meta.register_cf_specs("hycom")
    xoa.meta.is_registered_cf_specs("hycom")

Overview of the U dataset:

.. ipython:: python

    dsu = xoa.open_data_sample("hycom.gdp.u.nc")
    dsu

Decoding:

.. ipython:: python

    dsu = dsu.xoa.decode()
    dsu

Now we can read, decode and merge all files without any conflict:

.. ipython:: python

    dsv = xoa.open_data_sample("hycom.gdp.u.nc").xoa.decode()
    dsh = xoa.open_data_sample("hycom.gdp.h.nc").xoa.decode()
    print(dsu)
    ds = xr.merge([dsu, dsv, dsh])
    ds
