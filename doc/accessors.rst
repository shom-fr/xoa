.. _accessors:

Accessors API
=============

The main ``xoa`` accessor is registered when :mod:`xoa` is imported.
The ``meta`` and ``decode_sigma`` accessors are available as subaccessors of it,
like ``da.xoa.meta`` and ``ds.xoa.decode_sigma`` (the latter for datasets only),
and can also be registered at the top level with :func:`xoa.register_accessors`,
like ``ds.meta`` and ``ds.decode_sigma()``.

.. currentmodule:: xarray

.. _accessors.dataarray:

DataArray
---------

Attributes
~~~~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_attribute.rst

    DataArray.xoa.xdim
    DataArray.xoa.ydim
    DataArray.xoa.zdim
    DataArray.xoa.tdim
    DataArray.xoa.name
    DataArray.xoa.attrs
    DataArray.xoa.coords
    DataArray.xoa.data_vars
    DataArray.xoa.meta


Methods
~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_method.rst

    DataArray.xoa.set_meta_specs
    DataArray.xoa.get_meta_specs
    DataArray.xoa.decode
    DataArray.xoa.encode
    DataArray.xoa.auto_format
    DataArray.xoa.fill_attrs
    DataArray.xoa.infer_coords
    DataArray.xoa.get
    DataArray.xoa.get_coord
    DataArray.xoa.get_z
    DataArray.xoa.get_depth
    DataArray.xoa.interp
    DataArray.xoa.regrid


Dataset
-------

.. _accessors.dataset:

Attributes
~~~~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_attribute.rst

    Dataset.xoa.coords
    Dataset.xoa.data_vars
    Dataset.xoa.meta
    Dataset.xoa.decode_sigma


Methods
~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_method.rst

    Dataset.xoa.set_meta_specs
    Dataset.xoa.get_meta_specs
    Dataset.xoa.decode
    Dataset.xoa.encode
    Dataset.xoa.auto_format
    Dataset.xoa.fill_attrs
    Dataset.xoa.infer_coords
    Dataset.xoa.get
    Dataset.xoa.get_coord
    Dataset.xoa.get_z
    Dataset.xoa.get_depth
    Dataset.xoa.interp
    Dataset.xoa.regrid
    Dataset.decode_sigma.decode
    Dataset.decode_sigma.get_sigma_terms



Callables
~~~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_callable.rst

    Dataset.decode_sigma


Plotting
~~~~~~~~

The ``plot`` attribute of the ``xoa`` accessors, like ``da.xoa.plot.field()``,
has the following methods.

.. autoclass:: xoa.accessors.XoaPlotAccessor
    :members:
