.. _appendix.cf:

CF specs and defaults
=====================

This appendix refers to the searching and formatting specifications
for data variables and coordinates, and related tools,
available in the :mod:`xoa.cf` module.
Their usage is introduced in the :ref:`uses.cf` section.

.. _appendix.cf.default:

The default configuration
-------------------------

.. note:: You can define you own specifications for each of your datasets.
    Have a look to the :ref:`uses.cf` section and to the :ref:`examples`.

.. include:: gencfspecs/index.txt

As a :file:`.cfg` file
^^^^^^^^^^^^^^^^^^^^^^

.. literalinclude:: ../xoa/cf.cfg
    :language: ini

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


