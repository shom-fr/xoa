Installation
============

.. highlight:: bash

Dependencies
------------

xoa requires ``python>3`` and depends on the following packages:

.. list-table:: Requirements
   :widths: 10 90

   * - `appdirs <http://github.com/ActiveState/appdirs>`_
     - A small Python module for determining appropriate platform-specific
       dirs, e.g. a "user data dir".
   * - `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_
     - Cartopy is a Python package designed for geospatial data processing in
       order to produce maps and other geospatial data analyses.
   * - `configobj <https://configobj.readthedocs.io/en/latest/configobj.html>`_
     - ConfigObj is a simple but powerful config file reader and writer:
       an ini file round tripper.
   * - `matplotlib <https://matplotlib.org/>`_
     - Matplotlib is a comprehensive library for creating static, animated,
       and interactive visualizations in Python.
   * - libgfortran
     - Needed to use the fortran extension.
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
   * - `xesmf <https://xesmf.readthedocs.io/en/latest/>`_
     - xESMF is a powerful, easy-to-use and fast Python package for regridding.

You can install them with `conda <https://docs.conda.io/en/latest/>`_::

    $ conda install -c conda-forge appdirs cartopy configobj pandas scipy xarray xesmf

You also need a **fortran compiler** to install xoa from sources.
Conda packages exists, like for instance on linux: ``gfortran_linux-64``.


From sources
------------

Clone the repository::

    $ git clone https://github.com/VACUMM/xoa.git

Run the installation command::

    $ cd xoa
    $ python setup.py install
