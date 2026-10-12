# -*- coding: utf-8 -*-
"""
Test the :mod:`xoa.core.num` module
"""

import numpy as np

from xoa.core import num


class TestArrayIndexing:
    """Test array indexing utilities"""

    def test_get_iminmax(self):
        """Test finding first and last valid values in 1D array"""
        data = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, np.nan])
        imin, imax = num.get_iminmax(data)
        assert imin == 2
        assert imax == 4

    def test_get_iminmax_all_nan(self):
        """Test with all NaN values"""
        data = np.array([np.nan, np.nan, np.nan])
        imin, imax = num.get_iminmax(data)
        assert imin == -1
        assert imax == -1

    def test_get_iminmax_no_nan(self):
        """Test with no NaN values"""
        data = np.array([1.0, 2.0, 3.0, 4.0])
        imin, imax = num.get_iminmax(data)
        assert imin == 0
        assert imax == 3


class TestIndexConversion:
    """Test multi-dimensional index conversion utilities"""

    def test_unravel_index(self):
        """Test converting flat index to multi-dimensional"""
        shape = np.array([3, 4, 5], dtype=np.int64)
        i = 27
        ii = num.unravel_index(i, shape)
        expected = np.array([1, 1, 2], dtype=np.int64)
        np.testing.assert_array_equal(ii, expected)

    def test_ravel_index(self):
        """Test converting multi-dimensional index to flat"""
        shape = np.array([3, 4, 5], dtype=np.int64)
        ii = np.array([1, 1, 2], dtype=np.int64)
        i = num.ravel_index(ii, shape)
        assert i == 27

    def test_ravel_unravel_roundtrip(self):
        """Test that ravel and unravel are inverse operations"""
        shape = np.array([3, 4, 5], dtype=np.int64)
        for i in [0, 10, 27, 59]:
            ii = num.unravel_index(i, shape)
            i_back = num.ravel_index(ii, shape)
            assert i == i_back


class TestCsr:
    """Test the minimal CSR sparse matrix tools"""

    def test_coo_to_csr(self):
        mat = num.coo_to_csr([1, 0, 0], [1, 1, 0], [3.0, 2.0, 1.0], 2, 2)
        assert mat.nnz == 3
        np.testing.assert_array_equal(mat.indptr, [0, 2, 3])
        np.testing.assert_array_equal(mat.indices, [1, 0, 1])

    def test_csr_spmm_skipna_unified(self):
        mat = num.coo_to_csr([0, 0, 1], [0, 1, 1], [0.5, 0.5, 1.0], 2, 2)
        data = np.array([[1.0], [np.nan]])
        out = np.empty((2, 1))
        num.csr_spmm_skipna_unified(
            mat.indptr, mat.indices, mat.data, mat.data, num.EMPTY_OFFSETS, data, out
        )
        np.testing.assert_allclose(out[0], 1.0)  # renormalized over valid neighbour
        assert np.isnan(out[1, 0])
        out[:] = 0
        num.csr_spmm_skipna_unified(
            mat.indptr, mat.indices, mat.data, mat.data, num.EMPTY_OFFSETS, np.ones((2, 1)), out
        )
        np.testing.assert_allclose(out, 1.0)


def _random_tridiagonal(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(1, 3, n), rng.uniform(0.2, 1, n - 1)


def test_tridisolve_matches_dense():
    n = 30
    rng = np.random.default_rng(1)
    dl, d, du = rng.normal(size=n - 1), rng.normal(size=n), rng.normal(size=n - 1)
    a = np.diag(d) + np.diag(dl, -1) + np.diag(du, 1)
    b = rng.normal(size=n)
    x = b.copy()
    num.tridisolve(dl.copy(), d.copy(), du.copy(), x)
    np.testing.assert_allclose(a @ x, b, atol=1e-10)


def test_sturm_count():
    d, e = _random_tridiagonal(20)
    w = np.linalg.eigvalsh(np.diag(d) + np.diag(e, 1) + np.diag(e, -1))
    for k in range(1, 20):
        x = 0.5 * (w[k - 1] + w[k])
        assert num.sturm_count(d, e * e, x) == k


def test_trideigh_matches_dense():
    d, e = _random_tridiagonal(40)
    a = np.diag(d) + np.diag(e, 1) + np.diag(e, -1)
    w, v = np.linalg.eigh(a)
    values, vectors = num.trideigh(d, e, 5)
    np.testing.assert_allclose(values, w[:5], rtol=1e-10)
    np.testing.assert_allclose(np.abs(np.sum(vectors * v[:, :5], axis=0)), 1.0, atol=1e-8)
