Installation
============

.. highlight:: bash

Dependencies
------------

xoa requires ``python>3`` and depends on the following packages:

.. list-table::
   :widths: 10 90

   * - `appdirs <http://github.com/ActiveState/appdirs>`_
     - A small Python module for determining appropriate platform-specific
       dirs, e.g. a "user data dir".
   * - `cmocean <https://matplotlib.org/cmocean>`_
     - Beautiful colormaps for oceanography.
   * - `configobj <https://configobj.readthedocs.io/en/latest/configobj.html>`_
     - ConfigObj is a simple but powerful config file reader and writer:
       an ini file round tripper.
   * - `gsw <https://teos-10.github.io/GSW-Python/>`_
     - gsw is the python implementation of the Thermodynamic Equation of
       Seawater 2010 (TEOS-10).
   * - `matplotlib <https://matplotlib.org/>`_
     - Matplotlib is a comprehensive library for creating static, animated,
       and interactive visualizations in Python.
   * - `numba <https://numba.pydata.org/>`_
     - A high performance python compiler.
   * - `pandas <https://pandas.pydata.org/>`_
     - pandas is a fast, powerful, flexible and easy to use open source
       data analysis and manipulation tool, built on top of the
       Python programming language.
   * - `scipy <https://www.scipy.org/scipylib/index.html>`_
     - Scipy provides many user-friendly and efficient numerical routines,
       such as routines for numerical integration, interpolation,
       optimization, linear algebra, and statistics.
   * - `xarray <http://xarray.pydata.org/en/stable/>`_
     - xarray is an open source project and Python package that makes working
       with labelled multi-dimensional arrays simple, efficient, and fun!



From packages
-------------

xoa is on `conda-forge <https://anaconda.org/conda-forge>`_
and `pypi <https://pypi.org>`_::

    $ conda install -c conda-forge xoa

or::

    $ pip install xoa


From sources
------------

Clone the repository::

    $ git clone https://github.com/shom-fr/xoa.git

Run the installation command::

    $ cd xoa
    $ python setup.py install
