# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.weights` module
"""

import os

import numpy as np
import pytest
import xarray as xr

from xoa import weights


def get_variables(n=5):
    return {
        "j_base": ("n_dst", np.arange(n)),
        "frac_a": ("n_dst", np.linspace(0, 1, n)),
        "valid_dst_mask": ("n_dst", np.arange(n) % 2 == 0),
    }


def test_get_group_name():
    assert weights.get_group_name("regrid", "bilinear", "abc") == "regrid_bilinear_abc"


class TestSaveAndLoad:
    def test_missing_file(self, tmp_path):
        path = str(tmp_path / "missing.nc")
        assert weights.list_groups(path) == []
        assert not weights.has_group(path, "x")
        assert not weights.is_legacy(path)

    def test_create_file_and_group(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        assert weights.save_group(path, "g1", get_variables(), {"method": "bilinear", "n_dst": 5})
        assert weights.list_groups(path) == ["g1"]
        assert weights.has_group(path, "g1") and not weights.has_group(path, "g2")
        with xr.open_dataset(path) as root:
            assert root.attrs["xoa_weights_format"] == weights.FORMAT_VERSION
        ds = weights.load_group(path, "g1")
        assert ds.attrs["method"] == "bilinear"
        np.testing.assert_array_equal(ds.j_base, np.arange(5))
        np.testing.assert_allclose(ds.frac_a, np.linspace(0, 1, 5))
        assert ds.valid_dst_mask.dtype == bool
        assert not weights.is_legacy(path)

    def test_groups_are_appended(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        weights.save_group(path, "g1", get_variables(5))
        assert weights.save_group(path, "g2", get_variables(8))
        assert weights.list_groups(path) == ["g1", "g2"]
        assert weights.load_group(path, "g1").sizes["n_dst"] == 5
        assert weights.load_group(path, "g2").sizes["n_dst"] == 8

    def test_existing_group_is_left_untouched(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        weights.save_group(path, "g1", get_variables(5))
        size = os.path.getsize(path)
        assert not weights.save_group(path, "g1", get_variables(7))
        assert os.path.getsize(path) == size
        assert weights.load_group(path, "g1").sizes["n_dst"] == 5

    def test_arrays_are_compressed(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        weights.save_group(path, "g1", {"x": ("n", np.zeros(10000))})
        with xr.open_dataset(path, group="g1") as ds:
            assert ds.x.encoding["zlib"]
        assert os.path.getsize(path) < 10000 * 8 / 4

    def test_loaded_group_is_in_memory(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        weights.save_group(path, "g1", get_variables())
        ds = weights.load_group(path, "g1")
        os.remove(path)
        assert ds.j_base.values.sum() == 10


class TestLegacy:
    def test_legacy_file(self, tmp_path):
        path = str(tmp_path / "legacy.nc")
        xr.Dataset(get_variables(), attrs={"method": "bilinear", "n_dst": 5}).to_netcdf(path)
        assert weights.is_legacy(path)
        assert weights.list_groups(path) == []
        assert weights.load_group(path).attrs["n_dst"] == 5
        # Adding a group to a legacy file keeps its content
        weights.save_group(path, "g1", get_variables())
        assert weights.list_groups(path) == ["g1"]
        assert weights.load_group(path).sizes["n_dst"] == 5

    def test_unrelated_file_is_not_legacy(self, tmp_path):
        path = str(tmp_path / "other.nc")
        xr.Dataset({"x": ("n", np.arange(3))}).to_netcdf(path)
        assert not weights.is_legacy(path)


def test_fingerprint_attribute_roundtrip(tmp_path):
    path = str(tmp_path / "weights.nc")
    weights.save_group(path, "g1", get_variables(), {"fingerprint": "abc", "n_dst": 5})
    attrs = weights.load_group(path, "g1").attrs
    assert attrs["fingerprint"] == "abc"
    with pytest.raises(Exception):
        weights.load_group(path, "missing")


class TestDescribeAndFind:
    @staticmethod
    def make_file(path):
        for group, attrs in (
            (
                "g1",
                {
                    "kind": "regrid",
                    "method": "bilinear",
                    "fingerprint": "AB",
                    "src_fingerprint": "A",
                    "dst_fingerprint": "B",
                    "n_dst": 5,
                    "n_src": 9,
                },
            ),
            (
                "g2",
                {
                    "kind": "regrid",
                    "method": "conservative",
                    "fingerprint": "AC",
                    "src_fingerprint": "A",
                    "dst_fingerprint": "C",
                    "n_dst": 4,
                    "n_src": 9,
                },
            ),
            (
                "g3",
                {
                    "kind": "interp",
                    "method": "bilinear",
                    "fingerprint": "DB",
                    "src_fingerprint": "D",
                    "dst_fingerprint": "B",
                    "n_dst": 3,
                    "n_src": 6,
                },
            ),
        ):
            weights.save_group(path, group, get_variables(), attrs)

    def test_describe_groups(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        assert weights.describe_groups(path) == []
        self.make_file(path)
        infos = weights.describe_groups(path)
        assert [info["group"] for info in infos] == ["g1", "g2", "g3"]
        assert infos[0]["method"] == "bilinear" and infos[0]["n_src"] == 9
        assert isinstance(infos[0]["n_dst"], int)  # python types, not numpy ones
        assert infos[2]["kind"] == "interp"

    def test_find_by_fingerprint_as_source_or_destination(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        self.make_file(path)
        assert weights.find_groups(path, "A") == ["g1", "g2"]  # source
        assert weights.find_groups(path, "B") == ["g1", "g3"]  # destination
        assert weights.find_groups(path, "AC") == ["g2"]  # couple
        assert weights.find_groups(path, "unknown") == []

    def test_find_by_kind_and_method(self, tmp_path):
        path = str(tmp_path / "weights.nc")
        self.make_file(path)
        assert weights.find_groups(path) == ["g1", "g2", "g3"]
        assert weights.find_groups(path, kind="interp") == ["g3"]
        assert weights.find_groups(path, method="bilinear") == ["g1", "g3"]
        assert weights.find_groups(path, "B", kind="regrid", method="bilinear") == ["g1"]
        assert weights.find_groups(str(tmp_path / "missing.nc"), "A") == []
