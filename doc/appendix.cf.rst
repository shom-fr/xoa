.. _appendix.cf:

Default and specialized CF specs
================================

This appendix refers to the searching and formatting specifications
for data variables and coordinates, and related tools,
available in the :mod:`xoa.cf` module.
Their usage is introduced in the :ref:`uses.cf` section.

.. _appendix.cf.specialized:

Specialized configurations
--------------------------

A few configurations are made available internally for decoding specialized datasets.
You can use them at your own risk.

.. highlight:: python

For instance, load the croco specs with directly::

    import xoa.cf
    xoa.cf.set_cf_specs("croco")

Register it with::

    xoa.cf.register_cf_specs("croco")

You can access the associated `.cfg` file with :func:`xoa.cf_congs.get_cf_conf_file`.


.. include:: gencfspecs/specialized.txt

.. _appendix.cf.default:

The default configuration
-------------------------

.. note:: You can define you own specifications for each of your datasets.
    Have a look to the :ref:`uses.cf` section and to the :ref:`examples`.

As a :file:`.cfg` file
^^^^^^^^^^^^^^^^^^^^^^

Look at :ref:`appendix.cf.specialized.default`.


.. include:: gencfspecs/index.txt

The configuration syntax specifications
---------------------------------------

The syntax of all configurations are validated with these specifications.

See `configobj <https://configobj.readthedocs.io/en/latest/index.html>`_.

Details
^^^^^^^

.. include:: cf.txt

File
^^^^

.. literalinclude:: ../xoa/cf.ini
    :language: ini


