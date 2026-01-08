"""
Backward compatibility layer for the xoa.meta.configs module

.. deprecated::
    The xoa.meta_configs module is deprecated. Please use :mod:`xoa.meta.configs` instead.
"""

import warnings

warnings.warn("The 'xoa.cf_configs' module is deprecated. Pease use 'xoa.meta.configs' instead")

from xoa.meta import get_meta_config_file as get_cf_config_file  # noqa
from xoa.meta.configs import META_CONFIGS as CF_CONFIGS  # noqa
