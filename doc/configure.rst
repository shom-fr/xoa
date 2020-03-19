Configuration
#############

List of xoa options
===================

Some options are accessible to alter the default behavior of xoa.
These options are of particular type and have a default value,
as shown in the following table.

.. include:: genoptions/table.txt


Setting options
===============

Permanent settings
------------------

You permanently change default settings by editing
your xoa configuration file, which is typicall here on linux:
:file:`~/.local/share/xoa/xoa.cfg`.
Use the following command to make sure this file is at
this place:

.. code-block:: bash

    $ xoa info paths

This file is organised in sections, which correspond to the
string before the dot "." in the options name.

Inline settings
---------------

You can change options within your code with
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

Options are printed organised in sections.

From your code
--------------

You the :func:`xoa.get_option` function to access a single option::

    cmap = xoa.get_option('plot.cmapdiv') # flat mode
    cmap = xoa.get_option('plot', 'cmapdiv') # section mode


