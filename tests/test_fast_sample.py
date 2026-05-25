import numpy as np
import pytest

from genlm.control.util import fast_sample_logprobs


def test_size_one_returns_int_array_of_length_one():
    out = fast_sample_logprobs(np.array([0.0, 0.0, 0.0, 0.0]), size=1)
    assert out.shape == (1,)
    assert 0 <= int(out[0]) < 4


def test_size_many_returns_array_of_correct_length():
    out = fast_sample_logprobs(np.array([0.0, 0.0, 0.0]), size=37)
    assert out.shape == (37,)
    assert ((out >= 0) & (out < 3)).all()


def test_empirical_distribution_matches_softmax():
    logprobs = np.array([0.0, 1.0, 2.0, -1.0, 0.5])
    expected = np.exp(logprobs - logprobs.max())
    expected /= expected.sum()

    n = 200_000
    samples = fast_sample_logprobs(logprobs, size=n)
    counts = np.bincount(samples, minlength=len(logprobs))
    empirical = counts / counts.sum()

    assert np.allclose(empirical, expected, atol=5e-3), (
        f"empirical={empirical}  expected={expected}  diff={empirical - expected}"
    )


def test_handles_very_negative_logprobs():
    logprobs = np.array([-1e10, 0.0, -1e10, -1e10])
    n = 1000
    samples = fast_sample_logprobs(logprobs, size=n)
    assert (samples == 1).all()


def test_handles_neg_inf_in_logprobs():
    logprobs = np.array([-np.inf, 0.0, -np.inf, 0.0, -np.inf])
    n = 5000
    samples = fast_sample_logprobs(logprobs, size=n)
    finite_mask = np.isin(samples, [1, 3])
    assert finite_mask.all(), f"sampled a -inf index: {samples[~finite_mask][:5]}"


@pytest.mark.parametrize("v", [4, 32, 1024, 152_064])
def test_argmax_dominates_when_one_logprob_is_huge(v):
    logprobs = np.full(v, -100.0)
    logprobs[v // 2] = 100.0
    samples = fast_sample_logprobs(logprobs, size=500)
    assert (samples == v // 2).all()


def test_np_random_seed_gives_deterministic_samples():
    logprobs = np.linspace(-3.0, 3.0, 256)
    np.random.seed(42)
    a = fast_sample_logprobs(logprobs, size=1000)
    np.random.seed(42)
    b = fast_sample_logprobs(logprobs, size=1000)
    assert np.array_equal(a, b)
    np.random.seed(43)
    c = fast_sample_logprobs(logprobs, size=1000)
    assert not np.array_equal(a, c)


def _legacy_fast_sample_logprobs(logprobs, size=1):
    noise = -np.log(-np.log(np.random.random((size, len(logprobs)))))
    return (logprobs + noise).argmax(axis=1)


@pytest.mark.parametrize(
    "logprobs,n",
    [
        (np.array([0.0, 1.0, 2.0, -1.0, 0.5]), 200_000),
        (np.linspace(-3.0, 3.0, 50), 200_000),
        (np.random.default_rng(0).standard_normal(1024), 50_000),
    ],
)
def test_matches_legacy_implementation_distributionally(logprobs, n):
    new = np.bincount(fast_sample_logprobs(logprobs, size=n),
                      minlength=len(logprobs)) / n
    legacy = np.bincount(_legacy_fast_sample_logprobs(logprobs, size=n),
                         minlength=len(logprobs)) / n
    assert np.allclose(new, legacy, atol=5e-3), (
        f"V={len(logprobs)}  max diff={np.abs(new - legacy).max():.4f}"
    )
