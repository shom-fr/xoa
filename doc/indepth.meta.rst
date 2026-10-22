.. _indepth.meta:

Metadata, CF and naming conventions
###################################

The :mod:`xoa.meta` subpackage finds and formats variables and coordinates whatever the
names that a dataset gives them. It applies a subset of the
`CF conventions <http://cfconventions.org/>`_ to naming, and also works with datasets
that are not well formatted, since it searches the names and the attributes
(``standard_name``, ``long_name``, ``units``).

See :ref:`sphx_glr_tutorials_meta_plot_naming.py` for a walkthrough with real datasets,
:ref:`sphx_glr_tutorials_meta_plot_naming_specs.py` to write your own specs,
and the appendix :ref:`appendix.meta.default` for the list of known items.

.. note:: It shares features with `cf_xarray <https://cf-xarray.readthedocs.io/en/latest/>`_,
    but it also handles data variables and dimensions, not only coordinates, and it is
    available as independent specs objects that can be configured for each type of dataset.

.. ipython:: python

    @suppress
    import warnings
    @suppress
    warnings.simplefilter("ignore")
    @suppress
    import xarray as xr, numpy as np
    import xoa
    from xoa import meta
    meta_specs = meta.get_meta_specs()

Concepts
========

**Generic names** are the unique names that xoa uses to refer to a quantity,
like ``"temp"``, ``"u"``, ``"lon"`` or ``"depth"``.
**Specialized names** are those of a given dataset or model, like ``"thetao"`` or ``"lon_rho"``.
The **specs** (:class:`~xoa.meta.general.MetaSpecs`) tell how to go from one to the other:
for each generic name, the alternate names and the attributes that identify the quantity,
and the specialized name to use for encoding.
They have two main sections, like in xarray:
:ref:`data_vars <appendix.meta.data_vars>` and :ref:`coords <appendix.meta.coords>`.

Three operations are available:

- **search**: find an array from its generic name, with ``get``;
- **decode**: rename the arrays to their generic names and fill their attributes;
- **encode**: the reverse, rename to the specialized names.

Quick start with accessors
==========================

The easiest way is the ``xoa`` accessor. Let's make a dataset whose names are not explicit:

.. ipython:: python

    lon = xr.DataArray(np.arange(3, dtype='d'), dims='mylon',
        attrs={'standard_name': 'longitude'})
    temp = xr.DataArray(np.arange(20, 23, dtype='d'), dims='mylon',
        coords={'mylon': lon}, attrs={'standard_name': 'sea_water_temperature'})
    sal = xr.DataArray(np.arange(33, 36, dtype='d'), dims='mylon',
        coords={'mylon': lon}, attrs={'standard_name': 'sea_water_salinity'})
    ds = xr.Dataset({'mytemp': temp, 'mysal': sal})

Arrays are found from their attributes, whatever their names:

.. ipython:: python

    ds.xoa.get("temp").name
    ds.xoa.temp.name  # same, as an attribute
    ds.xoa.lon.name  # coordinates are also searched
    ds.xoa.data_vars.get("sal").name  # restrict to data variables
    ds.xoa.get("bathy") is None  # None when not found
    ds.xoa.temp.xoa.lon.name  # chaining

Decoding renames and completes the attributes, encoding goes back to specialized names
when the specs have some:

.. ipython:: python

    ds.xoa.decode()

.. seealso::
    :class:`xoa.accessors.MetaDataArrayAccessor`
    :class:`xoa.accessors.MetaDatasetAccessor`

Specs objects
=============

The accessors rely on :class:`~xoa.meta.general.MetaSpecs` methods that can be used directly.

.. ipython:: python

    meta_specs.search(ds, "temp").name  # data variable in a dataset
    meta_specs.coords.match(lon, "lon")  # does it match this item?
    meta_specs.coords.match(lon)  # does it match any known item?
    meta_specs.format_data_var(temp, "temp")  # explicit formatting
    meta_specs.decode(ds)

``decode`` and ``encode`` are the same operation, where ``encode`` renames with the
specialized names. They are consistent: encoding a decoded dataset leads back
to the initial dataset. See the example :ref:`indepth.meta.croco`.

.. seealso::
    :meth:`xoa.meta.general.MetaSpecs.search`
    :meth:`xoa.meta.general.MetaSpecs.auto_format`
    :meth:`xoa.meta.general.MetaSpecs.decode`
    :meth:`xoa.meta.general.MetaSpecs.encode`

What an item looks like
=======================

.. ipython:: python

    from pprint import pprint
    pprint(meta_specs.data_vars['sst'])
    pprint(meta_specs.coords['lon'])

The keys of the items of both sections are:

.. list-table::
    :header-rows: 1
    :widths: 20 80

    * - Key
      - Description
    * - ``name``
      - Specialized name for encoding. Empty by default.
    * - ``alt_names``
      - Alternate names for decoding.
    * - ``standard_name``, ``long_name``, ``units``
      - Attributes to match and to set. They are stored in the ``attrs`` key.
    * - ``search_order``
      - Properties to search and in which order, from the letters ``n`` (name),
        ``s`` (standard name) and ``u`` (units).
    * - ``inherit``
      - Inherit the specs of another item.

Data variables also have ``cmap`` (colormap), ``domain`` (generic, atmos, ocean or surface),
``select`` (selection applied to the array) and ``squeeze`` (dimensions to remove).
Coordinates also have ``axis`` (X, Y, Z, T or F).

.. ipython:: python

    meta_specs.data_vars.get_name("sst")
    meta_specs.data_vars.get_attrs("sst")

.. note:: The ``dims`` section lists the names of the dimensions of type x, y, z, t and f,
    that are used to guess the type of **dimensions that are not coordinates**,
    since they have no attributes. It is filled from the coordinates, so it
    is better to edit the coordinates than this section.

The other sections (:metasec:`register`,
:metasec:`sglocator`, :metasec:`vertical` and
:metasec:`accessors`) are described in :ref:`appendix.meta.sections`.
The ``sglocator`` one handles the staggered grid locations of the names, like
``"lon_rho"`` or ``"u_u"``. See the example :ref:`indepth.meta.hycom`.

Choosing the specs
==================

Most xoa routines need the specs of the dataset they work on. They are found in this order:

#. The ones the dataset was assigned, through its ``meta_specs`` attribute or encoding,
   that is the name of registered specs.
#. The registered specs that best match the dataset, from its names, and from its
   ``source`` attribute for the Mercator ones.
#. The current specs, which are the default ones unless you change them.

Setting the current specs
-------------------------

:class:`~xoa.meta.set_meta_specs` takes specs, the name of registered or internal specs, a
dict or a file. It is applied at once, or only within a ``with`` block:

.. ipython:: python

    specs = {"data_vars": {"ssb": {"standard_name": "sea_surface_banana"}}}
    ds = xr.Dataset({"toto": ("x", [5], {"standard_name": "sea_surface_banana"})})
    with meta.set_meta_specs(specs):
        print(ds.xoa.get("ssb").name)

The accessor reads the specs when it is first used on a dataset: open the dataset
within the block, or set the specs of the dataset only, with the same arguments:
``ds.xoa.set_meta_specs("croco")``.

Registering and inferring specs
-------------------------------

Specs with a registration name can be recalled by their name and are candidates for
inference. xoa comes with a few for specialized models (see :ref:`appendix.meta.specialized`):

.. ipython:: python

    meta.register_meta_specs("croco")  # from the name of an internal configuration
    meta.get_meta_specs("croco").name

You can register your own:

.. ipython:: python

    content = {
        "register": {"name": "mycroco"},
        "data_vars": {"temp": {"name": "supertemp"}},
        "coords": {"lon": {"name": "mylon"}},
    }
    mycroco = meta.general.MetaSpecs(content)
    meta.register_meta_specs(mycroco)
    meta.get_meta_specs("mycroco") is mycroco

Inference selects the registered specs that know the names of a dataset:

.. ipython:: python

    ds = xr.Dataset({'supertemp': ("mylon", [0, 2])}, coords={"mylon": [10, 20]})
    meta.infer_meta_specs(ds, named=True).name
    meta.infer_meta_specs(ds).decode(ds)

To assign specs to a dataset once and for all, set the ``meta_specs`` encoding
of the dataset, or of all its arrays with :func:`~xoa.meta.assign_meta_specs`:

.. ipython:: python

    meta.assign_meta_specs(ds, "mycroco")
    ds.mylon.encoding
    meta.get_meta_specs(ds.supertemp) is mycroco

Writing specs
-------------

:class:`~xoa.meta.general.MetaSpecs` is created from a dict, a configuration file,
a multi-line string with the content of a file, or another instance, or a list of them
where the first has priority. By default, they extend the internal defaults and the
user file, whose location is shown by ``xoa info paths``. Use ``default=False`` and
``user=False`` to start from scratch:

.. ipython:: python

    banana = {"data_vars": {"banana": {"attrs": {"standard_name": "banana"}}}}
    apple = {"data_vars": {"apple": {"attrs": {"long_name": "Big apple"}}}}
    fruits = meta.general.MetaSpecs(
        (meta.general.MetaSpecs(apple, default=False, user=False),
         meta.general.MetaSpecs(banana, default=False, user=False)),
        default=False, user=False)
    fruits.data_vars.names

The same banana in a configuration file:

.. code-block:: ini

    [data_vars]
        [[banana]]
            standard_name: banana

Existing specs can be updated with :meth:`~xoa.meta.general.MetaSpecs.load_cfg`, and
a single item with :meth:`~xoa.meta.MetaVarSpecs.set_specs`.

Helper modules
==============

These modules rely on the specs to offer high-level finders:

- :mod:`xoa.coords`: :func:`~xoa.coords.get_lon`, :func:`~xoa.coords.get_lat`,
  :func:`~xoa.coords.get_time`, :func:`~xoa.coords.get_depth`,
  :func:`~xoa.coords.get_z`, :func:`~xoa.coords.get_altitude`,
  :func:`~xoa.coords.get_level` and :func:`~xoa.coords.get_vertical` (any vertical coordinate,
  without conversion), their ``is_*`` counterparts, and the dimensions finders
  :func:`~xoa.coords.get_xdim`, :func:`~xoa.coords.get_ydim`, :func:`~xoa.coords.get_zdim`,
  :func:`~xoa.coords.get_tdim` and :func:`~xoa.coords.get_fdim`.
- :mod:`xoa.thermdyn`: :func:`~xoa.thermdyn.get_temp`, :func:`~xoa.thermdyn.get_sal` and
  :func:`~xoa.thermdyn.get_dens`, with a ``variant`` argument to target a type:
  ``"temp"`` (in situ), ``"ptemp"`` (potential), ``"ctemp"`` (conservative) for temperature, and ``"sal"``, ``"psal"``, ``"asal"``,
  ``"pfsal"`` for salinity.
- :mod:`xoa.dyn`: :func:`~xoa.dyn.get_sea_level` with the variants ``"ssh"``, ``"adt"``,
  ``"sla"``, ``"mdt"`` and ``"mss"``, and :func:`~xoa.dyn.get_u` and :func:`~xoa.dyn.get_v`
  with the variants ``"geo"`` (eastward and northward, ``ugeo``, ``uo``), ``"grid"`` (along the
  grid axes, ``ux``, ``vy``), ``"geos"`` (geostrophic, ``ugeos``, ``ugos``), ``"bt"``, ``"bc"``,
  ``"surf"`` and ``"generic"``. The geostrophic and baroclinic ones are not searched by default.

.. _indepth.meta.zdepth:

The ``z`` coordinate is positive up and the ``depth`` is positive down:
see :ref:`indepth.vertical`.

.. ipython:: python

    import xoa.coords, xoa.thermdyn
    xoa.coords.get_lon(temp).name
    xoa.coords.is_lat(lon)
    ds_thd = xr.Dataset({'mytemp': temp, 'mysal': sal})
    xoa.thermdyn.get_sal(ds_thd).name

.. _indepth.meta.croco:

Example: decoding and encoding Croco outputs
============================================

The Croco specs are in :file:`xoa/meta/configs/croco.cfg`. They are internal, so they
are registered from their name (done above), and a Croco dataset
is decoded with them whatever the names of the file:

.. ipython:: python

    ds = xoa.open_data_sample("MODELS/CROCO/SOUTH-AFRICA/croco.south-africa.meridional.nc")
    dsd = ds.xoa.decode()
    list(ds.data_vars)[-8:]
    list(dsd.data_vars)[-8:]

Encoding leads back to the Croco names:

.. ipython:: python

    list(dsd.xoa.encode().data_vars)[-8:]

.. _indepth.meta.hycom:

Example: decoding and merging Hycom split outputs
=================================================

At Shom, the Hycom outputs are split into one file per variable, and all variables are
stored with dimensions ``(y, x)`` and ``lon`` and ``lat`` coordinates, whatever their
staggered grid location. Merging them would lead to conflicts. The Hycom specs rename the
horizontal dimensions and coordinates of staggered variables to indicate their location,
with the ``add_coords_loc`` sections: the location is not added to the ``u`` and ``v``
variables, but to their dimensions and coordinates.

.. literalinclude:: ../xoa/meta/configs/hycom.cfg
    :language: ini

Decoded datasets can then be merged without any conflict:

.. ipython:: python

    meta.register_meta_specs("hycom")
    dsu = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.u.nc").xoa.decode()
    dsv = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.v.nc").xoa.decode()
    dsh = xoa.open_data_sample("MODELS/HYCOM/hycom.gdp.h.nc").xoa.decode()
    xr.merge([dsu, dsv, dsh])
