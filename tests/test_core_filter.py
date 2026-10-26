"""
Test the :mod:`xoa.core.filter` module
"""

import numpy as np
import pytest

from xoa.core import filter as cfilter


@pytest.mark.parametrize(
    "radius, expected",
    [
        (0.0, [True, True, True, True, True]),
        (0.025, [True, False, True, True, False]),
        (np.inf, [True, False, False, False, False]),
    ],
)
def test_get_decimate_arg(radius, expected):

    x = np.array([0.0, 1.0, 2.0, 5.0, 6.0])
    y = x.copy()
    keep = cfilter.get_decimate_arg(x, y, radius)
    np.testing.assert_allclose(keep, expected)
