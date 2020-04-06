"""Generate files to declare fortran functions"""

import os
import logging
import importlib

from docutils import nodes
from docutils.statemachine import string2lines, StringList
from sphinx import addnodes
from sphinx.util.docutils import SphinxDirective, switch_source_input


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

    def get_table(self, items):
        """Inspired from autosummary code"""
        table_spec = addnodes.tabular_col_spec()
        table_spec['spec'] = r'\X{1}{2}\X{1}{2}'

        real_table = nodes.table('', classes=['longtable'])
        group = nodes.tgroup('', cols=2)
        real_table.append(group)
        group.append(nodes.colspec('', colwidth=10))
        group.append(nodes.colspec('', colwidth=90))
        body = nodes.tbody('')
        group.append(body)

        for mod_name, mod_desc in items:
            source, line = self.state_machine.get_source_and_line()
            row = nodes.row('')
            for text in (f":mod:`{mod_name}`", mod_desc):
                node = nodes.paragraph('')
                vl = StringList()
                vl.append(text, '%s:%d:<tonq>' % (source, line))
                with switch_source_input(self.state, vl):
                    self.state.nested_parse(vl, 0, node)
                    try:
                        if isinstance(node[0], nodes.paragraph):
                            node = node[0]
                    except IndexError:
                        pass
                    row.append(nodes.entry('', node))
            body.append(row)

        return real_table

    def run(self):

        srcdir = self.env.srcdir
        gendir = os.path.join(srcdir, "genfortran")
        rst_entry_table = ".. list-table::\n\n"

        toctree_entries = []
        table_entries = []

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

            # Append entry points
            toctree_entries.append(f"genfortran/{mod_name}/index")
            table_entries.append((mod_name, description))

        # Create entry point
        nodes = []
        if toctree_entries:

            tocnode = addnodes.toctree(glob=None, hidden=True)
            tocnode['includefiles'] = toctree_entries
            tocnode['entries'] = [(None, en) for en in toctree_entries]
            nodes.append(tocnode)

            nodes.append(self.get_table(table_entries))



        return nodes


def setup(app):

    app.add_directive("genfortran", GenFortran)

    return {'version': '0.1'}
