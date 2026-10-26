.. _appendix.meta:

Meta configurations and specifications
======================================

This appendix refers to the searching and formatting configurations
for data variables and coordinates, and related tools,
available in the :mod:`xoa.meta` module,
and to the specifications that are used to validate them.
Their usage is introduced in the :ref:`indepth.meta` section.

Configurations
--------------

.. _appendix.meta.specialized:

Specialized configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^

A few configurations are made available internally for decoding specialized datasets.
You can use them at your own risk.

.. highlight:: python

For instance, load the croco configuration directly with::

    import xoa.meta
    xoa.meta.set_meta_specs("croco")

Register it with::

    xoa.meta.register_meta_specs("croco")

You can access the associated `.cfg` file with :func:`xoa.meta.get_meta_config_file`.


.. include:: genmetaspecs/specialized.txt

.. _appendix.meta.default:

The default configuration
^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: You can define your own configurations for each of your datasets.
    Have a look to the :ref:`indepth.meta` section and to the :ref:`tutorials`.

As a :file:`.cfg` file
""""""""""""""""""""""

Look at :ref:`appendix.meta.specialized.default`.


.. include:: genmetaspecs/index.txt

Specifications
--------------

The syntax of all configurations is validated with the specifications
provided by the internal :file:`meta.ini` file.

.. toctree::
    :maxdepth: 1

    appendix.meta.specs
