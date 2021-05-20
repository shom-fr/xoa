Configuration
#############

List of xoa options
===================

Some options are accessible to alter the default behavior of xoa.
These options are of a particular **type** and have a **default value**,
as shown in the following table.

.. include:: genoptions/table.txt

.. note:: The xoa configuration system is based on the
    `configobj <https://configobj.readthedocs.io/en/latest/configobj.html>`_
    package.

Setting options
===============

Permanent settings
------------------

You can permanently change default settings by editing
the xoa user configuration file, which is typicall here on linux:
:file:`~/.local/share/xoa/xoa.cfg`.
Use the following command to see where this is located:

.. code-block:: bash

    $ xoa info paths

This file is organised in sections, which correspond to the
string before the dot "." in the "flat" view of the option names.

Inline settings
---------------

You can change options from python with
the :class:`xoa.set_option` function and the :class:`xoa.set_options` class::

    xoa.set_option("plot.cmapdiv", "cmo.balance")  # single option
    xoa.set_options("plot", cmapdiv="cmo.balance", cmappos="cmo.amp")  # several options

:class:`xoa.set_options` can be used as a context manager to temporarily
change options within a block::

    with xoa.set_options("plot", cmapdiv="cmo.balance"):

        # you code with temporary settings

    # back to previous options

Getting options
===============

From commandline
----------------

Default options are printable with the following command:

.. code-block:: bash

    $ xoa info options

Options are organised in sections.

From python
-----------

You can use the :func:`xoa.get_option` function to access a single option::

    cmap = xoa.get_option('plot.cmapdiv') # flat mode
    cmap = xoa.get_option('plot', 'cmapdiv') # section mode


