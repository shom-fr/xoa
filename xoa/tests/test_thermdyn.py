# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.thermdyn` module
"""

import numpy as np
import xarray as xr

from xoa import thermdyn


def test_thermdyn_mixed_layer_depth():

    # 1d arrays
    depth = xr.DataArray(
        np.linspace(-50, 0.0, 6), dims="z", attrs={"long_name": "Depth", "positive": "up"}
    )
    temp = xr.DataArray(np.linspace(10, 20.0, 6), dims="z", coords={"depth": depth}, name="temp")
    kz = xr.DataArray(np.linspace(0.0005 * 10, 0, 6), dims="z", coords={"depth": depth})

    # temp 1d
    mld = thermdyn.mixed_layer_depth(temp, method="deltatemp")
    np.testing.assert_allclose(mld, 1)
    assert mld.long_name == "Mixed layer depth"
    assert mld.dims == ()

    # implicit
    mld = thermdyn.mixed_layer_depth(temp)
    np.testing.assert_allclose(mld, 1)

    # kz
    mld = thermdyn.mixed_layer_depth(kz, method="kzmax")
    np.testing.assert_allclose(mld, 5)

    # 2d arrays
    temp2d = temp.expand_dims("lon").transpose(..., "lon")
    depth2d = depth.expand_dims("lon").transpose(..., "lon")
    temp2d.coords["depth"] = depth2d
    mld2d = thermdyn.mixed_layer_depth(temp2d, method="deltatemp")
    assert mld2d.dims == ("lon",)
    assert mld2d.shape == (1,)
    np.testing.assert_allclose(mld2d, 1)
