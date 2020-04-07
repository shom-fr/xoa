"""Generate files to declare fortran functions"""

import os
import re
import logging
import importlib

from docutils.statemachine import string2lines
from sphinx.util.docutils import SphinxDirective


path_pat_mod_dir = os.path.join("{gendir}", "{mod_name}")
path_pat_mod_file = os.path.join(path_pat_mod_dir, "index.rst")
path_pat_func_file = os.path.join(path_pat_mod_dir, "{func_name}.rst")


def checkdir(path):
    pdir = os.path.dirname(path)
    if not os.path.exists(pdir):
        os.makedirs(pdir)


class GenFortran(SphinxDirective):

    has_content = True

    def run(self):

        if not self.content:
            return []

        # Loop on modules and descriptions
        rst_toctree = ".. toctree::\n\t:hidden:\n\n"
        rst_table = ".. list-table::\n\n"
        for mod_name_desc in self.content:
            smod = mod_name_desc.split(" ")
            mod_name = smod[0]
            mod_desc = " ".join(smod[1:])
            rst_toctree += f"\tgenfortran/{mod_name}/index\n"
            rst_table += f"\t* - :mod:`{mod_name}`\n"
            rst_table += f"\t  - {mod_desc}\n"

        # Insert toctree and tables
        rst_all = rst_toctree + "\n\n" + rst_table + "\n"
        source = self.state_machine.input_lines.source(
            self.lineno - self.state_machine.input_offset - 1)
        include_lines = string2lines(rst_all, convert_whitespace=1)
        self.state_machine.insert_input(include_lines, source)

        return []


re_directive_match = re.compile(
    r"^(?P<indent>\s*)\.\.\s+genfortran::\s*\n$").match
re_indent_match = re.compile(r"^(?P<indent>\s*)\S.+\n$").match


def generate_stub_files(srcdir, mod_name, mod_desc):

    gendir = os.path.join(srcdir, "genfortran")

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
        f.write(mod_desc + "\n\n")
        f.write(f".. module:: {mod_name}\n\n")
        rst_table = ".. list-table::\n\n"
        rst_toctree = ".. toctree::\n\t:hidden:\n\n"
        for func_name in func_names:

            rst_table += f"\t* - :func:`{mod_name}.{func_name}`\n"
            func = getattr(mod_content, func_name)
            func_sig = func.__doc__.split("\n")[0]
            rst_table += f"\t  - {func_sig}\n"
            rst_toctree += f"\t{func_name}\n"

            with open(path_pat_func_file.format(
                    **locals()), "w") as ff:
                ff.write(func_name+"\n"+len(func_name)*"="+"\n\n")
                ff.write(f".. currentmodule:: {mod_name}\n\n")
                out, call = func_sig.split('=')
                ff.write(f".. autofunction:: {call}\n\n")
        f.write(rst_toctree+"\n\n")
        f.write(rst_table)


def parse_and_generate(app):
    """Parse rst files to find directives and generate stub files"""

    # Get file list
    env = app.builder.env
    srcdir = env.srcdir
    if app.config.genfortran_src_files:
        srcfiles = [os.path.join(srcdir, srcfile) for srcfile
                    in app.config.genfortran_src_files]
    else:
        env = app.builder.env
        srcfiles = [env.doc2path(x, base=None) for x in env.found_docs
                    if os.path.isfile(env.doc2path(x))]

    # Parse files
    for srcfile in srcfiles:
        if not os.path.exists(srcfile):
            logging.warning("[genfortran] file not found: "+srcfile)
            continue
        with open(srcfile) as f:
            indent = None
            for line in f:
                m = re_directive_match(line)
                if m:
                    indent = m.group('indent')
                    continue
                if indent is None:
                    continue
                m = re.match("^"+indent + r"\s+(?P<mod_name>[\w.]+)" +
                             r"(?P<mod_desc>\s.*)\n$", line)
                if m:
                    generate_stub_files(
                        srcdir, m.group("mod_name"),
                        m.group("mod_desc").strip())
                    continue
                m = re_indent_match(line)
                if m and len(m.group('indent')) <= len(indent):
                    indent = None


def setup(app):

    app.add_directive("genfortran", GenFortran)
    app.connect('builder-inited', parse_and_generate)
    app.add_config_value('genfortran_src_files', [], [], [list])

    return {'version': '0.1'}
