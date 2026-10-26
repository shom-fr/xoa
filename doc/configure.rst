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
the xoa user configuration file, which is typically here on linux:
:file:`~/.local/share/xoa/xoa.cfg`.
Use the following command to see where this is located:

.. code-block:: bash

    $ xoa info paths

This file is organised in sections, which correspond to the
string before the dot "." in the "flat" view of the option names.

Settings from a file
--------------------

Give the path of a config file with the same format to :func:`xoa.load_options`
to keep its options for the session, or to :class:`xoa.set_options`
to use them temporarily as a context manager::

    xoa.load_options("myproject/xoa.cfg")  # for the session

    with xoa.set_options("myproject/xoa.cfg"):  # within the block only
        ...

Options are merged, so the file only needs to contain the ones to change.
Unknown options and invalid values raise an :class:`xoa.exceptions.XoaConfigError`.

Precedence, from the lowest to the highest: the default values, the user configuration file,
the options loaded or set in python, and the arguments of the functions.
The plot functions use the ``plot`` options when the corresponding arguments are ``None``.

Inline settings
---------------

You can change options from python with
the :func:`xoa.set_option` function and the :class:`xoa.set_options` class::

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

The tutorial :ref:`sphx_glr_tutorials_config_plot_options.py` shows the options at work, with the
``cmap="auto"`` argument of the maps that chooses the colormap from the sign of the data.
