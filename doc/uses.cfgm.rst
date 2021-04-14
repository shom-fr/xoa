.. _uses.cfgm:

Config management with  :mod:`xoa.cfgm`
#######################################


Introduction
============

The :mod:`xoa.cfgm` module facilitates and extends the functionality
offered by the excellent
`configobj <https://configobj.readthedocs.io/en/latest/index.html>`_
configuration file reading and validation package.
It makes the global usage slightly easier and adds support for
plurals forms and for exporting the specifications
to a commandline options and rst declarations.


Loading and validating
======================


Let's define the configuration specifications:

.. ipython:: python

    cfgspecs_lines = ["[logger] # logging system", "level=option(debug,info,error,default=info) # logging level", "name=string(default=root) # logger name", "[domain]", "xlim=floats(n=2, default=list(-20,0)) # min and max longitudes", "ylim=floats(n=2,default=list(40,50))"]
    print("\n".join(cfgspecs_lines))

.. note:: Note that the built-in `float` validation function has been
    used here in its plurals form.

Let's set the user configuration:

.. ipython:: python

    cfg_lines = ["[logger]", "name=xoa", "[domain]", "xminmax=-10,0"]
    print("\n".join(cfg_lines))

Now, initialize the config manager:

.. ipython:: python

    from xoa import cfgm
    CFGM = cfgm.ConfigManager(cfgspecs_lines)

.. note:: To configure a library, the `cfgspecs_lines` argument is typically the name
    of a static file placed in the same directory as the module that will
    initializes an instance of the `~xoa.cfg.ConfigManager` class.

And finally, load and validate the user configuration:

.. ipython:: python

    cfg = CFGM.load(cfg_lines)

    from pprint import pprint
    pprint(cfg)

Get the default values:

.. ipython:: python

    pprint(CFGM.defaults.dict())

Extending the validation capabilities
=====================================

One can add user validation fonctions to the default ones.

Use the :func:`~xoa.cfgm.print_validation_fonctions` function to print
the list of validation functions.
In the following example, only those matching ``*datetime*`` are printed:

.. ipython:: python

    cfgm.print_validation_functions("*time*") # use a pattern to restrict search

To define a new validation function, use the
:func:`~xoa.cfgm.register_validation_fonctions`  function.
Here we define and register a validation function that converts
an entry to an angle in degrees within [0, 360):

.. ipython:: python

    # Define
    from validate import VdtTypeError
    def is_angle(value, radians=False):
        """Validate an angle with optional convertion to radians"""
        try:
            value = float(value) % 360.
        except Exception:
            raise VdtTypeError("Invalid angle")
        value = float(value)
        if radians:
            value = np.radians(value)
        return value

    # Register
    cfgm.register_validation_functions(angle=is_angle)

    # Check that it is registered
    cfgm.print_validation_functions("angle")

    # Check that it works
    validator = cfgm.get_validator()
    print(validator.check("angle(radians=True)", 180+360))


Using the :mod:`argparse` capabilities
======================================

The :class:`~xoa.cfgm.ConfigManager` has the capability to generate
commandline options from the configuration specifications thanks to
the :class:`~xoa.cfgm.ConfigManager.arg_parse` method.
The goal is to add more control to the configuration
for the user that use it from an executable script that parses the
commandline:

* Default values are internally defined in the config specifications.
* The user optionally alter these value with its configuration file.
* The user optionally alter its own configuration using the
  commandline options, which are set to None by default.

In other word, the commandline arguments take precedence over
the user configuration, which takes precedence over the default
internal configuration.

Taking advantage of the example above, we create a parser
and add arguments that reflect the configuration specifications:


.. ipython:: python
    :okwarning:

    # Write the user config file
    with open("config.cfg", "w") as f:
        f.write("\n".join(cfg_lines))

    # Define our commandline options
    argv = ["--logger-level", "error", "myfile.nc"]

    # Init the parser
    from argparse import ArgumentParser
    parser = ArgumentParser(description="My script")
    parser.add_argument("ncfile", help="netcdf file")

    # Add options and parse
    cfg, args = CFGM.arg_parse(
        parser=parser, getargs=True, args=argv, cfgfile="config.cfg")

    # Args
    print(args)

    # Cfg
    pprint(cfg)

    # See the help like with the -h option
    parser.print_help()

    # Long help
    parser.parse_args(["--long-help"])

    # Very short help
    parser.parse_args(["--short-help"])


The :class:`~xoa.cfgm.ConfigManager.arg_parse` method makes also available
the ``--short-help`` and ``--long--help`` in addition to the
``--help`` option.

.. note:: The previous example can be compacted using the
    :func:`~xoa.cfgm.cfgargparse` function::

        cfg, args = cfgargparse(cfgspecsfile, parser=parser, getargs=True)


Converting to ``.rst`` format
==============================

The config manager instance is exportable to `rst` declaration
for documentation purpose.

.. ipython:: python

    print(CFGM.get_rst())

The result is the following.

.. include:: uses.cfgm.cfg2rst.txt

These above sphinx declarations need two objet types to be declared::

    app.add_object_type('confopt', 'confopt',
                        objname='configuration option',
                        indextemplate='pair: %s; configuration option')
    app.add_object_type('confsec', 'confsec',
                        objname='configuration section',
                        indextemplate='pair: %s; configuration section')

The name of these types are parameters of the
:func:`~xoa.cfgm.cfg2rst` function
(and :func:`xoa.cfgm.ConfigManager.get_rst` method).

Fortunately, the :mod:`xoa.cfgm` comes also as a **Sphinx extension**:

- Add ``"xoa.cfgm"`` to the list of Sphinx extensions that are declared in the :file:`conf.py` file.
- Declare the following Sphinx config variables in the :file:`conf.py`:

  .. confval:: cfgm_get_cfgm_func

     Function that returns a :class:`xoa.cfgm.ConfigManager` instance.

  .. confval:: cfgm_rst_file

     Name of the outfile file in which rst declarations are written.

