"""Contract every resampler owes `smc_standard`, which picks one by name at runtime."""

import numpy as np
import pytest

from genlm.control.sampler.resampling import (
    RESAMPLING_METHODS,
    get_resampling_fn,
)

METHODS = sorted(RESAMPLING_METHODS)


@pytest.fixture(autouse=True)
def _seed():
    np.random.seed(0)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "weights",
    [
        [0.25, 0.25, 0.25, 0.25],
        [0.7, 0.2, 0.05, 0.05],
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.5],
        [1.0],
    ],
)
def test_returns_n_indices_in_range(method, weights):
    idx = get_resampling_fn(method)(np.asarray(weights))
    assert len(idx) == len(weights)
    assert idx.min() >= 0 and idx.max() < len(weights)


@pytest.mark.parametrize("method", METHODS)
def test_degenerate_weights_select_only_the_survivor(method):
    weights = np.zeros(8)
    weights[5] = 1.0
    idx = get_resampling_fn(method)(weights)
    assert set(idx.tolist()) == {5}


@pytest.mark.parametrize("method", METHODS)
def test_ancestor_counts_are_unbiased(method):
    weights = np.array([0.5, 0.3, 0.15, 0.05])
    fn = get_resampling_fn(method)
    n, trials = len(weights), 4000
    counts = np.zeros(n)
    for _ in range(trials):
        counts += np.bincount(fn(weights), minlength=n)
    assert np.allclose(counts / (trials * n), weights, atol=0.02)


@pytest.mark.parametrize("method", ["systematic", "stratified"])
def test_one_probe_per_stratum_bounds_each_count(method):
    """Each ancestor count lies within floor/ceil of its expected share."""
    weights = np.array([0.5, 0.3, 0.15, 0.05])
    fn = get_resampling_fn(method)
    n = len(weights)
    for _ in range(200):
        counts = np.bincount(fn(weights), minlength=n)
        expected = n * weights
        assert np.all(counts >= np.floor(expected))
        assert np.all(counts <= np.ceil(expected))


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown resampling method"):
        get_resampling_fn("bootstrap")

