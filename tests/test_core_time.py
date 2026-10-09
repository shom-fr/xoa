# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.time` module
"""

import numpy as np

from xoa.core import time as ctime


def test_compute_time_frac_indices():
    it, frac = ctime.compute_time_frac_indices(
        np.array([0.0, 1.0, 3.0]), np.array([0.5, 2.0, 5.0, np.nan])
    )
    np.testing.assert_array_equal(it, [0, 1, -1, -1])
    np.testing.assert_allclose(frac[:2], [0.5, 0.5])
