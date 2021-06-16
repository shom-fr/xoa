# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# %% Path setup

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.join(os.path.abspath(
    os.path.dirname(__file__)), 'ext'))


# %% Project information

project = 'xoa'
copyright = '2020-2021, Shom'
author = 'Shom'

# The full version, including alpha/beta/rc tags
import xoa
release = xoa.__version__
xoa.register_accessors(xoa=True, xcf=True, decode_sigma=True)


# %% General configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    'genoptions',
    'gencfspecs',
    'sphinxarg.ext',
    'sphinxcontrib.programoutput',
    'sphinx_autosummary_accessors',
    'sphinx_gallery.gen_gallery',
    'xoa.cfgm'
]

# Add any paths that contain templates here, relative to this directory.
import sphinx_autosummary_accessors
templates_path = ['_templates', sphinx_autosummary_accessors.templates_path]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# A list of ignored prefixes for module index sorting.
modindex_common_prefix = ['xoa.']


# %% Options for HTML output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# %% Autosumarry
autosummary_generate = True

# %% Intersphinx
intersphinx_mapping = {
    'python': ('https://docs.python.org/fr/3/', None),
    'cmocean': ('https://matplotlib.org/cmocean/', None),
    'configobj': ('https://configobj.readthedocs.io/en/latest/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'numba:': ('https://numba.readthedocs.io/en/stable/', None),
    'numpy': ("https://numpy.org/doc/stable/", None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'proplot': ('https://proplot.readthedocs.io/en/latest/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'xarray': ('http://xarray.pydata.org/en/stable/', None),
    # 'xesmf': ("https://xesmf.readthedocs.io/en/latest/", None)
    }

# %% Napoleon
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_rtype = False

# %% Genoptions
genoptions_table = 'genoptions/table.txt'
genoptions_declarations = 'genoptions/declarations.txt'

# %% Cfgm
import xoa.cf
cfgm_get_cfgm_func = xoa.cf._get_cfgm_
cfgm_rst_file = "cf.txt"

# %% Extlinks
extlinks = {
    "issue": ("https://github.com/pydata/xarray/issues/%s", "GH"),
    "pull": ("https://github.com/pydata/xarray/pull/%s", "PR"),
}

# %% Nbsphinx
#nbsphinx_timeout = 120  # in seconds

# %% Sphinx gallery
sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "examples",
    "binder": {
        'org': 'VACUMM',
        'repo': 'xoa',
        'branch': 'master',
        'binderhub_url': 'https://mybinder.org',
        'dependencies': [
            './binder/environment.yml',
            './binder/apt.txt',
            './binder/setup.py'
            ],
        'notebooks_dir': 'notebooks',
        'use_jupyter_lab': True,
        },
    }

# %% User directives

def setup(app):

    app.add_css_file('custom.css')

    app.add_object_type('confopt', 'confopt',
                        objname='configuration option',
                        indextemplate='pair: %s; configuration option')
    app.add_object_type('confsec', 'confsec',
                        objname='configuration section',
                        indextemplate='pair: %s; configuration section')
    app.add_object_type('confval', 'confval',
                        objname='sphinx configuration value',
                        indextemplate='pair: %s; sphinx configuration value')
