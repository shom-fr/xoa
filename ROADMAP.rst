.. _roadmap:

Development roadmap
===================

The base
--------

The fundamental objective of xoa is to facilitate the analysis,
transformation and comparison of ocean data.
Facilitate here means that it must be possible to perform
complex operations in a few lines, without having to know all the
API and all the parameters to obtain a suitable and accurate result.
To this end, the library relies on the following libraries, among others:

`xarray <http://xarray.pydata.org>`_
    All numerical vectors are by default supposed to be of
    :class:`xarray.DataArray` type, i.e. with numerical values, attributes and
    potentially multi-dimensional coordinates.
    The library adds some :ref:`CF support <usages.cf>`
    to identify known data arrays and coordinates in datasets.
`matplotlib <https://matplotlib.org>`_ and `cartopy <https://scitools.org.uk/cartopy/docs/latest/>`_
    All plots are made with them, with an extra layer that
    prevents the user from making certain choices or
    explicitly providing certain parameters.
`xesmf <https://xesmf.readthedocs.io/en/latest/>`_ and `scipy.interpolate <https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_
    :mod:`xesmf` is used to perform efficient and valid regridding between
    structured grids.
    :mod:`scipy.interpolate` helps performing interpolations
    between unstructured data.
    :mod:`xoa.regrid` is a convenient interface to these libraries.
    xoa adds missing capabilities such as the ability to make 1D interpolations
    with coordinates that varies along other dimensions, which is often
    the case for the vertical coordinate of ocean models.

For those who know `vacumm <https://github.com/VACUMM/vacumm>`_,
xoa a light weight, faster and more flexible version,
now in python 3, and based on the xarray data model instead of
CDAT.
vacumm provide alot of facilities that are now available in other packages,
thus no longer needed.


Extension of the Climate and Forecast (CF) conventions
------------------------------------------------------

It is essential to easily find longitude and latitude coordinates
to make a map, or temperature and salinity to make a TS diagram.
The :mod:`xoa.cf` module helps finding numerous variables in data arrays
and datasets thanks to their attributes, the most important one being
the standard_name.

- Not all known variables with a standard name have been declared,
  but the list can be completed.
- Variables are organised by their short name which must be unique,
  and is generally chosen from an ocean point of view.

In addition to variable identification, xoa is intended to
facilitate depth and altitude calculations based on sigma coordinates,
with the :mod:`xoa.sigma` module.

Interpolations
--------------

The :mod:`fortran interpolation routines <xoa._interp>` are implemented.
Calling python routines such as :func:`xoa.regrid.regrid1d` still have to be
implemented.
The interpolation engine must always be these xoa fortran routines by default
because they fulfill all basic requirements.
One can however imagine the possibility of using scipy's
abilities when it is appropriate to do so.

Graphics
--------

The idea is to guess how to plot a variable
from its attributes and coordinates.
For example, know that we draw a hovmoeller diagram of the temperature,
with depths as the y-axis, and a cmo.thermal palette.

In addition, most of convenient graphic routines of `vacumm <https://github.com/VACUMM/vacumm>`_
are now implemented by `proplot <https://proplot.readthedocs.io/en/latest/>`_
in a nice way.
The xoa graphic routines will probably make an intensive usage of
this package for core operations.

