"""Generate files to declare fortran functions"""

import os
import logging
import importlib

from docutils.statemachine import string2lines
from sphinx.util.docutils import SphinxDirective

import xoa.cf


path_pat_mod_dir = os.path.join("{gendir}", "{mod_name}")
path_pat_mod_file = os.path.join(path_pat_mod_dir, "index.rst")
path_pat_func_file = os.path.join(path_pat_mod_dir, "{func_name}.rst")


def checkdir(path):
    pdir = os.path.dirname(path)
    if not os.path.exists(pdir):
        os.makedirs(pdir)


class GenFortran(SphinxDirective):

    required_arguments = 0
    has_content = True

    def run(self):

        srcdir = self.env.srcdir
        gendir = os.path.join(srcdir, "genfortran")
        rst_entry_toctree = ".. toctree::\n\t:hidden:\n\n"
        rst_entry_table = ".. list-table::\n\n"

        # Get names
        for mod_name_desc in self.content:

            smod = mod_name_desc.split(" ")
            mod_name = smod[0]
            description = " ".join(smod[1:])
            logging.info(f"Generating rst files for fortran wrapper "+mod_name)
            mod_content = importlib.import_module(mod_name)
            func_names = [func for func in dir(mod_content)
                          if not func.startswith('_')]

            # Write files
            mod_dir = path_pat_mod_dir.format(**locals())
            if not os.path.exists(mod_dir):
                os.makedirs(mod_dir)
            mod_file = path_pat_mod_file.format(**locals())
            with open(mod_file, "w") as f:
                f.write(mod_name + "\n" + len(mod_name)*"=" + "\n\n")
                f.write(f".. module:: {mod_name}\n\n")
                rst_table = ".. list-table::\n\n"
                rst_toctree = ".. toctree::\n\t:hidden:\n\n"
                for func_name in func_names:

                    rst_table += f"\t* - :func:`{mod_name}.{func_name}`\n"
                    func = getattr(mod_content, func_name)
                    signature = func.__doc__.split("\n")[0]
                    rst_table += f"\t  - {signature}\n"
                    rst_toctree += f"\t{func_name}\n"

                    with open(path_pat_func_file.format(
                            **locals()), "w") as ff:
                        ff.write(func_name+"\n"+len(func_name)*"="+"\n\n")
                        ff.write(f".. currentmodule:: {mod_name}\n\n")
                        out, call = signature.split('=')
                        ff.write(f".. autofunction:: {call}\n\n")
                f.write(rst_toctree+"\n\n")
                f.write(rst_table)

            # Complete entry points
            rst_entry_toctree += f"\tgenfortran/{mod_name}/index\n"
            rst_entry_table += f"\t* - :mod:`{mod_name}`\n"
            rst_entry_table += f"\t  - {description}\n"

        # Write entry point
        rst_entry = rst_entry_toctree + "\n\n" + rst_entry_table + "\n"
        print(rst_entry)
        source = self.state_machine.input_lines.source(
            self.lineno - self.state_machine.input_offset - 1)
        include_lines = string2lines(rst_entry, convert_whitespace=1)
        self.state_machine.insert_input(include_lines, source)

        return []

import numpy.f2py

def setup(app):

    app.add_directive("genfortran", GenFortran)

    return {'version': '0.1'}
