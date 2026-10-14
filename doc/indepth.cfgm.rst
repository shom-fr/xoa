.. _indepth.cfgm:

Config management with :mod:`xoa.cfgm`
#######################################

The :mod:`xoa.cfgm` module simplifies and extends the
`configobj <https://configobj.readthedocs.io/en/latest/index.html>`_
configuration file reader and validator package. It is what xoa uses for its own
:doc:`options <configure>` and for its naming specifications (see :ref:`indepth.meta`).
It makes global usage a little easier and adds support for plural forms,
for command line options and for rst declarations of the options.

The tutorial :ref:`sphx_glr_tutorials_config_plot_cfgm.py` goes through all of this with code.

Concepts
========

- **Specifications** describe the options: a section is in brackets, and an option is a
  ``name=function(parameters, default=value) # help``. They are typically in a static file
  next to the module, which initializes a :class:`~xoa.cfgm.ConfigManager`.
- **Validation functions** convert and check the values: the ones of configobj, like
  ``string``, ``float`` or ``option``, and the ones of xoa, like ``figsize``,
  ``cmap`` or the dates. :func:`xoa.cfgm.print_validation_functions` lists them
  and :func:`xoa.cfgm.register_validation_functions` adds yours.
- **Plural forms** validate lists: ``floats(n=2)`` is a list of two floats.
- **Precedence**: the command line options, if any, override the user configuration file,
  which overrides the defaults of the specifications.

The workflow is to create the manager from the specifications, then call
:meth:`~xoa.cfgm.ConfigManager.load` to validate a user configuration, or
:meth:`~xoa.cfgm.ConfigManager.arg_parse` to also get the command line options:

.. code-block:: python

    from xoa import cfgm

    CFGM = cfgm.ConfigManager("myspecs.cfg")
    cfg = CFGM.load("user.cfg")                  # validated, with defaults
    cfg, args = CFGM.arg_parse(parser=parser, getargs=True, cfgfile="user.cfg")

:func:`xoa.cfgm.cfgargparse` does both in a single call.

Command line options
====================

:meth:`~xoa.cfgm.ConfigManager.arg_parse` adds one option per option of the specifications,
named after its section, like ``--logger-level``. They are set to ``None`` by default so that
the ones that are not passed do not override the user configuration. It also adds
``--cfgfile`` for the user configuration file, and ``--short-help`` and ``--long-help``
in addition to ``--help``.

Documenting the options
=======================

:meth:`~xoa.cfgm.ConfigManager.to_rst` exports the specifications as Sphinx declarations
of options and sections, which need two object types to be declared::

    app.add_object_type('confopt', 'confopt',
                        objname='configuration option',
                        indextemplate='pair: %s; configuration option')
    app.add_object_type('confsec', 'confsec',
                        objname='configuration section',
                        indextemplate='pair: %s; configuration section')

The names of these types are parameters of the :func:`~xoa.cfgm.cfg2rst` function.

Fortunately, :mod:`xoa.cfgm` also comes as a **Sphinx extension**:

- Add ``"xoa.cfgm"`` to the list of Sphinx extensions that are declared in the :file:`conf.py` file.
- Declare the following Sphinx config variables in the :file:`conf.py`:

  .. confval:: cfgm_get_cfgm_func

     Function that returns a :class:`xoa.cfgm.ConfigManager` instance.

  .. confval:: cfgm_rst_file

     Name of the outfile file in which rst declarations are written.

Pitfalls
========

- **Keys that are not in the specifications are kept as unvalidated strings.** A typo in a
  user file, like ``xminmax`` instead of ``xlim``, does not raise an error, and the default
  value of ``xlim`` is used.
- **Validation errors name the option** and are :class:`~xoa.cfgm.XoaValidateError`.
