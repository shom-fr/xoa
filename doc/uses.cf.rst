.. _uses.cf:

Naming conventions with :mod:`xoa.cf`
#####################################

.. warning::
    Have a quick loop to the appendix :ref:`appendix.cf.default`
    before going any further!

Introduction
============

This module is an application and extension of a subset of the
`CF conventions <http://cfconventions.org/>`_ in application
to naming conventions.
It has two intents:

* Searching :class:`xarray.DataArray` variables or coordinates with another
  :class:`xarray.DataArray` or a :class:`xarray.Dataset`,
  by scanning name and attributes like ``units`` and ``standard_name``.
* Formatting :class:`xarray.DataArray` variables or coordinates with
  unique ``name``, and ``standard_name``, ``long_name`` and ``units``
  attributes, with support of staggered grid location syntax.

The module offers capabilities for the user to extend and specialize
default behaviors for user's special datasets.


.. note:: This module shares common feature with the excellent and long
    awaited `xf_xarray <https://cf-xarray.readthedocs.io/en/latest/>`_
    package and started a long time before with the Vacumm package.
    The most notable differences differences include:

    - The current module is also designed for data variables, not only
      coordinates.
    - It search for items not only using standard_names but also
      specialized names.
    - It is not only available as accessors, but also as independant
      objects that can be configured for each type of dataset or in
      contexts by the user.

Accessing the current specifications
====================================

Scanning and formatting actions are based on specifications,
and this module natively includes a
:ref:`default configuration <appendix.cf.default>`
for various oceanographic, sea surface and atmospheric variables and coordinates.
A distinction is made between
data variables (:ref:`data_vars <appendix.cf.data_vars>`)
and coordinates (:ref:`coords <appendix.cf.coords>`), like in :mod:`xarray`.

Getting the current specifications for data variables and coordinates
with the :func:`~xoa.cf.get_cf_specs` function:

.. ipython:: python

    from xoa import cf
    cfspecs = cf.get_cf_specs()
    cfspecs["data_vars"] is cfspecs.data_vars
    cfspecs["data_vars"] is cfspecs.coords
    cfspecs.data_vars.names[:3]
    cfspecs.coords.names[:3]

See the appendix :ref:`appendix.cf.default` for the
list of available default specifications.

An instance of the :class:`CFSpecs` has other configuration
:ref:`sections <appendix.cf.sections>`
than
:cfgmsec:`data_vars <[data_vars]>`,
:cfgmsec:`coords <[coords]>` and
:cfgmsec:`dims <[dims]>`:
:cfgmsec:`registration <[registration]>`,
:cfgmsec:`sglocator <[sglocator]>`,
:cfgmsec:`vertical <[vertical]>` and
:cfgmsec:`accessors <[accessors]>`.


Data variables
--------------

Here is the example of the :cfdatavar:`sst` data variable:

.. ipython:: python

    from pprint import pprint
    pprint(cfspecs.data_vars['sst'])


Description of specification keys:

.. list-table:: CF specs for :ref:`appendix.cf.data_vars`

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

    cfspecs.data_vars.get_name("sst")
    cfspecs.data_vars.get_attrs("sst")


Coordinates
-----------

Here is the example of the :cfdatavar:`lon` coordinate:

.. ipython:: python

    from pprint import pprint
    pprint(cfspecs.coords['lon'])

Description of specification keys:

.. list-table:: CF specs for :ref:`appendix.cf.coords`

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

    cfspecs.coords.get_name("lon")
    cfspecs.coords.get_attrs("lon")


Dimensions
----------

.. list-table:: CF specs for :ref:`appendix.cf.dims`

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

    cfspecs.dims  # or cfspecs["dims"]


Other sections
--------------

.. ipython:: python

    cfspecs["register"]
    cfspecs["sglocator"]
    cfspecs["accessors"]
    cfspecs["vertical"]


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
    cfspecs.coords.match(lon) # any known
    cfspecs.data_vars.match(temp) # any known
    cfspecs.data_vars.match(sal) # any known

Search for known CF items:

.. ipython:: python

    mytemp = cfspecs.search(ds, "temp")
    mylon = cfspecs.search(mytemp, "lon")

Datasets are searched for data variables ("data_vars") and
data variables are searched for coordinates ("coords").
You can also search for coordinates in datasets, for instance like this:

.. ipython:: python

    cfspecs.coords.search(ds, "lon")

.. seealso::
    - CF items:
      :cfcoord:`lon` :cfcoord:`lat` :cfdatavar:`temp` :cfdatavar:`sal`
    - Methods: :meth:`xoa.cf.CFCoordSpecs.match`
      :meth:`xoa.cf.CFVarSpecs.match` :meth:`xoa.cf.CFSpecs.search`
      :meth:`xoa.cf.CFCoordSpecs.search` :meth:`xoa.cf.CFVarSpecs.search`


The staggered-grid locator :class:`~xoa.cf.SGLocator`
=====================================================

A :class:`~xoa.cf.CFSpecs` instance comes with :class:`~xoa.cf.SGLocator`
that is accessible through the :attr:`~xoa.cf.CFSpecs.sglocator` attribute.
A :class:`~xoa.cf.SGLocator` helps parsing and formatting staggered grid
location from :attr:`name`, :attr:`standard_name` and :attr:`long_name`
data array attributes.
It is configured at the :cfgmsec:`sglocator <[sglocator]>` section
in which you can specify the
format of the name (:cfgmopt:`name_format <[sglocator] name_format>` and
the allowed location names :cfgmopt:`allowed_locations <[sglocator] allowed_locations>`.

.. ipython:: python

    sglocator = cfspecs.sglocator
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

During an auto-formatting, each array is matched against CF specs,
and the array is formatted when a matching is successfull.
If the array contains coordinates, the same process is applied on them,
as soon as the ``format_coords`` keyword is ``True``.
If the ``name`` key is set in the matching item specs, its value can be used
to name the variable or coordinate array, else the generic name is used
like ``"lon"``.

**Explicit formatting:**

.. ipython:: python

    cfspecs.format_coord(lon, "lon")
    cfspecs.format_data_var(temp, "temp")

**Auto-formatting:**

.. ipython:: python

    ds2 = cfspecs.auto_format(ds)
    ds2.temp
    ds2.lon

It can be applied to a data array or a full dataset.

.. seealso::
    :meth:`xoa.cf.CFSpecs.format_coord`
    :meth:`xoa.cf.CFSpecs.format_data_var`
    :meth:`xoa.cf.CFSpecs.auto_format`

Encoding and decoding
---------------------

By default, formatting renames known arrays to their generic name,
like "temp" in the example above. We speak here of **encoding**.
If the ``specialize`` keyword is set to ``True``, arrays are
renamed with their specialized name if set in the specs with the
:cfgmopt:`name <[data_vars] [__many__] name>` option.
We speak here of **decoding**.
Two shortcut methods exists for these tasks:

- Decoding: :meth:`~xoa.cf.CFSpecs.decode`
- Encoding: :meth:`~xoa.cf.CFSpecs.encode`

Chaining the two methods should lead to the initial dataset or data array.
See the last section of this page for an exemple:
:ref:`uses.cf.croco`.


.. seealso::
    :meth:`xoa.cf.CFSpecs.decode`
    :meth:`xoa.cf.CFSpecs.encode`

Using the accessors
===================

Accessors for :class:`xarray.Dataset` and :class:`xarray.DataArray`
can be registered with the :func:`xoa.cf.register_cf_accessors`:

.. ipython:: python

    import xoa
    xoa.register_accessors(xcf=True)

The accessor is named here :class:`xcf <xoa.accessors.CFDatasetAccessor>`
to not conflict with the :class:`cf` accessor of
`cf-xarray <https://cf-xarray.readthedocs.io/en/latest/>`_.


.. note:: All xoa accessors can be be registered with
    :func:`xoa.register_accessors`. Note also that all functionalities
    of the :class:`xcf <xoa.accessors.CFDatasetAccessor>`
    accessor are also available with the more global
    :class:`xoa <xoa.accessors.XoaDatasetAccessor>` accessor.

These accessors make it easy to use some of the :class:`xoa.cf.CFSpecs`
capabilities.
Here are examples of use:

.. ipython:: python
    :okwarning:

    temp
    temp.xcf.get("lon") # access by .get
    ds.xcf.get("temp") # access by .get
    ds.xcf.lon # access by attribute
    ds.xcf.coords.lon  # specific search = ds.cf.coords.get("lon")
    ds.xcf.temp # access by attribute
    ds.xcf["temp"].name # access by item
    ds.xcf.data_vars.temp.name  # specific search = ds.cf.coords.get("temp")
    ds.xcf.data_vars.bathy is None # returns None when not found
    ds.xcf.temp.xcf.lon.name  # chaining
    ds.xcf.temp.xcf.name # CF name, not real name
    ds.xcf.temp.xcf.attrs # attributes, merged with CF attrs
    ds.xcf.temp.xcf.standard_name # single attribute
    ds.mytemp.xcf.auto_format() # or ds.temp.xcf()
    ds.xcf.auto_format() # or ds.xcf()

As you can see, accessing an accessor attribute or item make an
implicit call to :class:`~xoa.cf.DataArrayCFAccessor.get`.
The root accessor :attr:`cf` give access to
two sub-accessors, :attr:`~xoa.cf.DatasetCFAccessor.data_vars`
and :attr:`~xoa.cf.DatasetCFAccessor.coords`,
for being able to specialize the searches.

.. seealso::
    :class:`xoa.cf.DataArrayCFAccessor`
    :class:`xoa.cf.DatasetCFAccessor`

Changing the CF specs
=====================

Default user file
-----------------

The :mod:`xoa.cf` module has internal defaults as shown
in appendix :ref:`appendix.cf.default`.

You can extend these defaults with a user file,
whose location is printable with the following command,
at the line containing "user CF specs file":

.. command-output:: xoa info paths

Update the current specs
------------------------

The current specs can be updated with different methods.

From a well **structured dictionary**:

.. ipython:: python

    cfspecs.load_cfg({"data_vars": {"banana": {"standard_name": "banana"}}})
    cfspecs.data_vars["banana"]

From a **configuration file**: instead of the dictionary as an argument
to :meth:`~xoa.cf.CFSpecs.load_cfg` method, you can give either a
file name or a **multi-line string** with the same content as
the file.
Following the previous example:

.. code-block:: ini

    [data_vars]
        [[banana]]
            standard_name: banana

If you only want to update a :attr:`~xoa.cf.CFSpecs.category`,
you can use such method (here :meth:`~xoa.cf.CFVarSpecs.set_specs`):

.. ipython:: python

    cfspecs.data_vars.set_specs("banana", name="bonono")
    cfspecs.data_vars["banana"]["name"]

Alternatively, a :class:`xoa.cf.CFSpecs` instance can be loaded
with the :meth:`~xoa.cf.CfSpecs.load_cfg` method, as explained below.

Create new specs from scratch
-----------------------------

To create new specs, you must instantiate the :class:`xoa.cf.CFSpecs` class,
with an input type as those presented above:

- A config file name.
- A Multi-line string in the format of a config file.
- A dictionary.
- A :class:`configobj.ConfigObj` instance.
- Another :class:`~xoa.cf.CFSpecs` instance.
- A list of them, with the having priority over the lasts.

The initialization also accepts two options:

- ``default``: wether to load or not the default internal config.
- ``user``: wether to load or not the user config file.

An config created **from default and user configs**:

.. ipython:: python

    banana_specs = {"data_vars": {"banana": {"attrs": {"standard_name": "banana"}}}}
    mycfspecs = cf.CFSpecs(banana_specs)
    mycfspecs["data_vars"]["sst"]["attrs"]["standard_name"]
    mycfspecs["data_vars"]["banana"]["attrs"]["standard_name"]

An config created **from scratch**:

.. ipython:: python

    mycfspecs = cf.CFSpecs(banana_specs, default=False, user=False)
    mycfspecs.pprint(depth=2)

An config created **from two other configs**:

.. ipython:: python

    cfspecs_banana = cf.CFSpecs(banana_specs, default=False, user=False)
    apple_specs = {"data_vars": {"apple": {"attrs": {"long_name": "Big apple"}}}}
    cfspecs_apple = cf.CFSpecs(apple_specs, default=False, user=False)
    cfspecs_fruits = cf.CFSpecs((cfspecs_apple, cfspecs_banana),
        default=False, user=False)
    cfspecs_fruits.data_vars.names

Replacing the currents CF specs
-------------------------------

As shown before, the currents CF specs are accessible with the
:func:`xoa.cf.get_cf_specs` function.
You can replace them with the :class:`xoa.cf.set_cf_specs` class,
to be used as a fonction.

.. ipython:: python

    cfspecs_old = cf.get_cf_specs()
    cf.set_cf_specs(cfspecs_banana)
    cf.get_cf_specs() is cfspecs_banana
    cf.set_cf_specs(cfspecs_old)
    cf.get_cf_specs() is cfspecs_old


In case of a temporary change, you can used :class:`~xoa.cf.set_cf_specs`
in a context statement:

.. ipython:: python

    with cf.set_cf_specs(cfspecs_banana) as myspecs:
        print('inside', cf.get_cf_specs() is cfspecs_banana)
        print('inside', myspecs is cf.get_cf_specs())
    print('outside', cf.get_cf_specs() is cfspecs_old)

For convience, you can set specs directly with a dictionary:

 .. ipython:: python

    with cf.set_cf_specs({"data_vars": {"apple": {}}}) as myspecs:
        print("apple" in cf.get_cf_specs())
    print("apple" in cf.get_cf_specs())

Application with an accessor usage:

.. ipython:: python


    data = xr.DataArray([5], attrs={'standard_name': 'sea_surface_banana'})
    ds = xr.Dataset({'toto': data})
    mycfspecs = cf.CFSpecs({"data_vars": {"ssb":
        {"standard_name": "sea_surface_banana"}}})
    with cf.set_cf_specs(mycfspecs):
        print(ds.xcf.get("ssb"))


Working with registered specs
=============================

Registering and accessing new specs
-----------------------------------

It is possible to register specialized :class:`~xoa.cf.CFSpecs` instances
with :func:`~xoa.cf.register_cf_specs` for future access.

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
    mycfspecs = cf.CFSpecs(content)
    cf.register_cf_specs(mycfspecs)

We can now access with it the :func:`~xoa.cf.get_cf_specs` function:

.. ipython:: python

    these_cfspecs = cf.get_cf_specs('mycroco')
    these_cfspecs is mycfspecs

Inferring the best specs for my dataset
---------------------------------------

If you set the :attr:`cfspecs` attribute or encoding of a dataset
to the name of a registered :class:`~xoa.cf.CFSpecs` instance, you can
get it automatically with the :func:`~xoa.cf.get_cf_specs` or
:func:`~xoa.cf.infer_cf_specs` functions.

Let's register another :class:`~xoa.cf.CFSpecs` instance:

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
    mycfspecs2 = cf.CFSpecs(content)
    cf.register_cf_specs(mycfspecs2)

Let's create a dataset:

.. ipython:: python

    ds = xr.Dataset({'supertemp': ("mylon", [0, 2])}, coords={"mylon": [10, 20]})

Now find the best registered specs instance which has either the name
``myhycom`` or ``mycroco``:


.. ipython:: python

    cf_specs_auto = cf.infer_cf_specs(ds)
    print(cf_specs_auto.name)
    ds_decoded = cf_specs_auto.decode(ds)
    ds_decoded
    cf_specs_auto.encode(ds)

It is ``mycroco`` as expected.


Assigning registered specs to a dataset or data array
-----------------------------------------------------

All xoa routines that needs to access specific coordinates
or variables try to infer the approriate specs, which default
to the current specs.
When the :attr:`cfspecs` **attribute** or **encoding** is set,
:meth:`~xoa.cf.get_cf_specs` uses it to search within
registered specs.

.. ipython:: python

    ds.encoding.update(cfspecs="mycroco")
    cfspecs = cf.get_cf_specs(ds)
    cfspecs.encode(ds)

The :attr:`cfspecs` encoding is set at the dataset level,
not at the data array level:

.. ipython:: python

    cf.get_cf_specs(ds.supertemp) is cfspecs

To propagate to all the data arrays, use :func:`~xoa.cf.assign_cf_specs`:

.. ipython:: python

    cf.assign_cf_specs(ds, "mycroco")
    ds.mylon.encoding
    cf.get_cf_specs(ds.supertemp) is cfspecs


.. _uses.cf.croco:

Example: decoding/encoding Croco model outputs
==============================================

Here are the specs for Croco:

.. literalinclude:: ../xoa/_samples/croco.cfg
    :language: ini

Register them:

.. ipython:: python

    @suppress
    import xoa, xoa.cf
    xoa.cf.register_cf_specs(xoa.get_data_sample("croco.cfg"))
    xoa.cf.is_registered_cf_specs("croco")


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

.. _uses.cf.hycom:

Example: decoding and merging Hycom splitted outputs
====================================================

At Shom, the Hycom model outputs are splitted into separate files, one for each variable.
Conflicts may occur when using variables that are not at the same staggered grid location,
since all variables are stored with dimensions ``(y, x)`` and ``lon`` and ``lat``
coordinates, all with the same name.
To avoid this problem, the horizontal dimensions and coordinates of
staggered variables are renamed to indicate their location.

Here are the specs to take care of the staggered grid indicators in the names:

.. literalinclude:: ../xoa/_samples/hycom.cfg
    :language: ini

Note the ``add_coords_loc`` sections.
Location is not added to the ``u`` and ``v`` variables
but to their dimensions and coordinates.

Register them:

.. ipython:: python

    @suppress
    import xoa, xoa.cf
    xoa.cf.register_cf_specs(xoa.get_data_sample("hycom.cfg"))
    xoa.cf.is_registered_cf_specs("hycom")

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
    ds = xr.merge([dsu, dsv, dsh])
    ds