.. _appendix.meta:

Default and specialized meta specs
==================================

This appendix refers to the searching and formatting specifications
for data variables and coordinates, and related tools,
available in the :mod:`xoa.meta` module.
Their usage is introduced in the :ref:`indepth.meta` section.

.. _appendix.meta.specialized:

Specialized configurations
--------------------------

A few configurations are made available internally for decoding specialized datasets.
You can use them at your own risk.

.. highlight:: python

For instance, load the croco specs with directly::

    import xoa.meta
    xoa.meta.set_meta_specs("croco")

Register it with::

    xoa.meta.register_meta_specs("croco")

You can access the associated `.cfg` file with :func:`xoa.meta_configs.get_meta_config_file`.


.. include:: genmetaspecs/specialized.txt

.. _appendix.meta.default:

The default configuration
-------------------------

.. note:: You can define your own specifications for each of your datasets.
    Have a look to the :ref:`indepth.meta` section and to the :ref:`examples`.

As a :file:`.cfg` file
^^^^^^^^^^^^^^^^^^^^^^

Look at :ref:`appendix.meta.specialized.default`.


.. include:: genmetaspecs/index.txt

The configuration syntax specifications
---------------------------------------

The syntax of all configurations are validated with these specifications.

See `configobj <https://configobj.readthedocs.io/en/latest/index.html>`_.

Details
^^^^^^^

.. include:: meta.txt

File
^^^^

.. literalinclude:: ../xoa/meta/meta.ini
    :language: ini


