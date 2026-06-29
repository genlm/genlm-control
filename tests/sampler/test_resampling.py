"""Parity tests for the vendored resampling functions.

Asserts the vendored `genlm.control.sampler.resampling` functions return
identical indices to `llamppl.inference.resampling` under a fixed numpy seed,
for each resampling method. This is the guarantee that dropping the llamppl
dependency does not change resampling RNG behavior.
"""

import numpy as np
import pytest

from genlm.control.sampler import resampling as vendored

llamppl_resampling = pytest.importorskip("llamppl.inference.resampling")


def _weight_grids():
    rng = np.random.default_rng(0)
    grids = []
    for n in (1, 2, 5, 10, 64):
        for _ in range(3):
            w = rng.random(n) + 1e-6
            w = w / w.sum()
            grids.append(w)
    # Include a peaked and a near-uniform distribution explicitly.
    grids.append(np.array([0.97, 0.01, 0.01, 0.01]))
    grids.append(np.full(8, 1 / 8))
    return grids


@pytest.mark.parametrize(
    "method", ["multinomial", "stratified", "systematic", "residual"]
)
def test_vendored_resample_matches_llamppl(method):
    vendored_fn = vendored.get_resampling_fn(method)
    llamppl_fn = llamppl_resampling.get_resampling_fn(method)

    for weights in _weight_grids():
        np.random.seed(12345)
        got = vendored_fn(weights)
        np.random.seed(12345)
        expected = llamppl_fn(weights)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_get_resampling_fn_unknown():
    with pytest.raises(ValueError, match="Unknown resampling method"):
        vendored.get_resampling_fn("does-not-exist")
