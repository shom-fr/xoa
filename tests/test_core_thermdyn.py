"""
Test the :mod:`xoa.core.thermdyn` module
"""

import numpy as np
import pytest

from xoa.core import thermdyn

gsw = pytest.importorskip("gsw")


def test_sal_roundtrip():
    p, lon, lat = 100.0, 5.0, 45.0
    sp = np.array([34.0, 35.5])
    sa = thermdyn.to_sa(sp, "psal", p, lon, lat)
    np.testing.assert_allclose(sa, gsw.SA_from_SP(sp, p, lon, lat))
    for kind in "psal", "pfsal", "asal":
        back = thermdyn.to_sa(thermdyn.from_sa(sa, kind, p, lon, lat), kind, p, lon, lat)
        np.testing.assert_allclose(back, sa, rtol=1e-8)


def test_temp_roundtrip():
    p, sa = 100.0, np.array([34.0, 35.5])
    ct = np.array([10.0, 15.0])
    for kind in "temp", "ptemp", "ctemp":
        back = thermdyn.to_ct(thermdyn.from_ct(ct, kind, sa, p), kind, sa, p)
        np.testing.assert_allclose(back, ct, rtol=1e-8)


def test_convert_same_kind_is_identity():
    t = np.array([10.0, 12.0])
    np.testing.assert_array_equal(
        thermdyn.convert_temp(t, t, -10.0, 45.0, 0.0, "ptemp", "ptemp", "psal"), t
    )
    np.testing.assert_array_equal(thermdyn.convert_sal(t, -10.0, 45.0, 0.0, "psal", "psal"), t)
