import numpy as np
import pytest
from arsenal.maths import logsumexp as arsenal_lse
from genlm.control._fast import logsumexp


@pytest.mark.parametrize(
    "arr",
    [
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.arange(10, dtype=np.float64),
        np.array([1000.0, 0.0, 0.0]),
        np.array([-1e10, -1e10, -1e10]),
        np.linspace(-50.0, 50.0, 200_000),
        np.array([-np.inf, 0.0, -np.inf]),
        np.random.default_rng(0).standard_normal(8),
        np.random.default_rng(1).standard_normal(152_064),
    ],
)
def test_matches_arsenal(arr):
    have = logsumexp(arr)
    want = arsenal_lse(arr)
    if np.isinf(want):
        assert have == want
    else:
        assert np.isclose(have, want, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n", [1, 8, 1024])
def test_all_neg_inf(n):
    assert logsumexp(np.full(n, -np.inf)) == -np.inf


def test_empty_returns_neg_inf():
    assert logsumexp(np.array([], dtype=np.float64)) == -np.inf


def test_accepts_python_list():
    assert np.isclose(logsumexp([0.0, 1.0, 2.0]), arsenal_lse([0.0, 1.0, 2.0]))


def test_accepts_non_contiguous():
    arr = np.linspace(-10, 10, 1000)
    assert np.isclose(logsumexp(arr[::3]), arsenal_lse(arr[::3]))
