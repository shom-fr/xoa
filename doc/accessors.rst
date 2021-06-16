.. _accessors:

Accessors API
=============


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
    DataArray.xoa.cf


Methods
~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_method.rst

    DataArray.xoa.set_cf_specs
    DataArray.xoa.get_cf_specs
    DataArray.xoa.decode
    DataArray.xoa.encode
    DataArray.xoa.auto_format
    DataArray.xoa.fill_attrs
    DataArray.xoa.infer_coords
    DataArray.xoa.get
    DataArray.xoa.get_coord
    DataArray.xoa.get_depth


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
    Dataset.xoa.cf
    Dataset.xoa.decode_sigma


Methods
~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_method.rst

    Dataset.xoa.set_cf_specs
    Dataset.xoa.get_cf_specs
    Dataset.xoa.decode
    Dataset.xoa.encode
    Dataset.xoa.auto_format
    Dataset.xoa.fill_attrs
    Dataset.xoa.infer_coords
    Dataset.xoa.get
    Dataset.xoa.get_coord
    Dataset.xoa.get_depth
    Dataset.decode_sigma.decode
    Dataset.decode_sigma.get_sigma_terms



Callables
~~~~~~~~~

.. autosummary::
    :toctree: accessors
    :template: autosummary/accessor_callable.rst

    Dataset.decode_sigma
