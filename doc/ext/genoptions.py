"""Generate files to declare and list xoa options"""

import os
import logging

from configobj import ConfigObj
from validate import Validator

import xoa


decl_format = """.. xoaoption:: {optname}

    Type: ``{opttype}``

    Default: ``{optdef}``

    {optdesc}

"""

table_line_format = """    * - :xoaoption:`{optname}`
      - ``{opttype}``
      - ``{optdef}``
      - {optdesc}
"""

table_header = """
.. list-table:: Flat view of available xoa options
    :header-rows: 1
    :widths: 20 20 20 40

    * - Name
      - Type
      - Default value
      - Description
"""


def genrst(app):

    logging.info("Generating rst files for xoa options")

    srcdir = app.env.srcdir

    file_table = os.path.join(srcdir, app.config.genoptions_table)
    file_declarations = os.path.join(
        srcdir, app.config.genoptions_declarations)

    for ff in file_table, file_declarations:
        fd = os.path.dirname(ff)
        if not os.path.exists(fd):
            os.makedirs(fd)

    cfgspecs = ConfigObj(
        xoa.CONFIG_SPECS.split("\n"),
        list_values=False,
        interpolation=False,
        raise_errors=True,
        file_error=True,
    )
    validator = Validator()

    ftable = open(file_table, 'w')
    ftable.write(table_header)

    fdecl = open(file_declarations, 'w')

    for sec in cfgspecs.sections:
        for optname, optspecs in cfgspecs[sec].items():

            opttype, fun_args, fun_kwargs, optdef = validator._parse_check(
                optspecs)
            optdesc = cfgspecs[sec].inline_comments[optname].strip("# ")
            optdesc = optdesc.capitalize()
            optname = f"{sec}.{optname}"

            ftable.write(table_line_format.format(**locals()))
            fdecl.write(decl_format.format(**locals()))

    ftable.close()
    fdecl.close()


def setup(app):

    app.add_object_type(
        'xoaoption', 'xoaoption',
        objname='xoa flat configuration option',
        indextemplate='pair: %s; xoa flat configuration option')

    app.add_config_value('genoptions_table',
                         'genoptions/table.txt', 'html')
    app.add_config_value('genoptions_declarations',
                         'genoptions/declarations.txt', 'html')

    app.connect('builder-inited', genrst)

    return {'version': '0.1'}
