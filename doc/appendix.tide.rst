.. _appendix.tide:

Tidal constituents
==================

This appendix lists the tidal constituents that are available by default in :mod:`xoa.tide`,
shows the file that defines them, and gives the specifications of the files that complete or
change them.
Their usage is introduced in the :ref:`indepth.tide.constituents` section.

Default constituents
--------------------

.. include:: gentide/constituents.txt

The species is 0 for the long-period constituents, 1 for the diurnal ones, 2 for the
semidiurnal ones, and so on. The satellites are the nodal modulation terms of the constituent,
and the compound constituents are made of those they are listed with.

.. _appendix.tide.default:

The default file
----------------

The constituents are read from the internal :file:`xoa/tide/tide.cfg` file,
whose path is :data:`xoa.tide.data.CFG_FILE`.

.. literalinclude:: ../xoa/tide/tide.cfg
    :language: ini

.. _appendix.tide.user:

Adding or changing constituents
-------------------------------

The default constituents are completed by a :file:`tide.cfg` file located next to the xoa user
config file, whose path is returned by :func:`xoa.tide.data.get_user_cfg_file`, and by the files
that you give to :class:`xoa.tide.set_constituents`, like a file of your working directory.
Their sections complete the default constituents, or replace those of the same name.
See :ref:`indepth.tide.custom` for how to use them.

It has one section per constituent, whose name is that of the constituent.
For instance, the shallow-water constituent 2MS6 is made of twice M2 and once S2::

    [2MS6]
    doodson = 673.555
    species = 6
    [[compound_of]]
    [[[M2]]]
    multiplier = 2
    [[[S2]]]
    multiplier = 1

The keys of a section are:

.. list-table::
    :header-rows: 1

    * - Key
      - Type
      - Description
    * - ``doodson``
      - string
      - Doodson number, like ``255.555`` for M2, that sets the frequency
    * - ``species``
      - integer, at least 0
      - Tidal species: 0 for long period, 1 for diurnal, 2 for semidiurnal, etc.
    * - ``phase``
      - float, ``0`` by default
      - Constant phase in degrees added to the Doodson argument, for the diurnal constituents
    * - ``satellites``
      - string, empty by default
      - Nodal satellites, as ``r dN dp dp1 phi0`` groups of 5 floats separated by semicolons,
        like ``0.188 1 0 0 0 ; 0.0089 0 0 0 0``. See :func:`xoa.tide.data.is_satellites`.
    * - ``[[compound_of]]``
      - subsection
      - Only for compound constituents: one ``[[[name]]]`` subsection per source constituent,
        with its integer ``multiplier``, ``1`` by default

.. _appendix.tide.specs:

Specifications
--------------

The syntax of the files is validated with the specifications of the internal
:file:`xoa/tide/tide.ini` file, whose path is :data:`xoa.tide.data.INI_FILE`,
using :mod:`xoa.cfgm`.
See `configobj <https://configobj.readthedocs.io/en/latest/index.html>`_.
Any section name is accepted (``__many__``).

.. literalinclude:: ../xoa/tide/tide.ini
    :language: ini
