import pytest
import numpy as np
from genlm.control.sampler.sequence import Sequences, EndOfSequence, EOS


def test_initialization():
    sequences = Sequences(
        contexts=[[b"a"], [b"b"]],
        log_weights=[np.log(0.4), np.log(0.6)],
    )
    assert sequences.size == 2
    assert np.isclose(np.exp(sequences.log_total), 1.0)  # weights sum to 1
    assert len(sequences) == 2


def test_initialization_validation():
    # Test mismatched lengths
    with pytest.raises(AssertionError):
        Sequences(contexts=[[b"a"]], log_weights=[0.0, 0.0])


@pytest.mark.parametrize(
    "contexts, log_weights, expected",
    [
        (
            [[b"hello"], [b"world", EndOfSequence()]],  # no EOS filtering
            [np.log(0.4), np.log(0.6)],
            {tuple([b"hello"]): 0.4, tuple([b"world", EndOfSequence()]): 0.6},
        ),
        (
            [
                [b"hello", EndOfSequence()],
                [b"world", EndOfSequence()],
                [b"test", EndOfSequence()],
            ],
            [np.log(2), np.log(5), np.log(3)],
            {
                tuple([b"hello", EndOfSequence()]): 0.2,
                tuple([b"world", EndOfSequence()]): 0.5,
                tuple([b"test", EndOfSequence()]): 0.3,
            },
        ),
    ],
    ids=["no_eos_filtering", "three_way_normalization"],
)
def test_posterior(contexts, log_weights, expected):
    sequences = Sequences(contexts=contexts, log_weights=log_weights)
    posterior = sequences.posterior
    assert len(posterior) == len(expected)
    for key, prob in expected.items():
        assert np.isclose(posterior[key], prob)
    # Posterior probabilities must sum to 1.
    assert np.isclose(sum(posterior.values()), 1.0)


def test_normalized_weights():
    sequences = Sequences(
        contexts=[[b"a"], [b"b"]],
        log_weights=[np.log(3), np.log(7)],
    )
    weights = sequences.normalized_weights
    assert np.allclose(weights, [0.3, 0.7])
    assert np.isclose(np.sum(weights), 1.0)


def test_iteration_and_indexing():
    contexts = [[b"a"], [b"b"]]
    log_weights = [np.log(0.3), np.log(0.7)]
    sequences = Sequences(contexts=contexts, log_weights=log_weights)

    # Test __iter__
    for i, (ctx, weight) in enumerate(sequences):
        assert ctx == contexts[i]
        assert weight == log_weights[i]

    # Test __getitem__
    assert sequences[0] == (contexts[0], log_weights[0])
    assert sequences[1] == (contexts[1], log_weights[1])


def test_effective_sample_size():
    # Test equal weights (maximum ESS)
    sequences = Sequences(
        contexts=[[b"a"], [b"b"], [b"c"]],
        log_weights=[0.0, 0.0, 0.0],  # equal weights
    )
    assert np.isclose(sequences.ess, 3.0)  # ESS should equal number of particles

    # Test completely unbalanced weights (minimum ESS)
    sequences = Sequences(
        contexts=[[b"a"], [b"b"], [b"c"]],
        log_weights=[
            np.log(1.0),
            float("-inf"),
            float("-inf"),
        ],  # one particle has all weight
    )
    assert np.isclose(sequences.ess, 1.0)


def test_log_ml_calculation():
    # Test log marginal likelihood calculation
    sequences = Sequences(
        contexts=[[b"a"], [b"b"]],
        log_weights=[np.log(0.3), np.log(0.7)],
    )
    assert np.isfinite(sequences.log_ml)
    assert sequences.log_ml <= sequences.log_total


def test_empty_sequences():
    sequences = Sequences(contexts=[], log_weights=[])
    assert sequences.size == 0
    assert len(sequences.posterior) == 0
    assert len(sequences.decoded_posterior) == 0


@pytest.mark.parametrize(
    "contexts, log_weights, expected",
    [
        (
            [[b"hello", EndOfSequence()]],
            [0.0],
            {"hello": 1.0},
        ),
        (
            [[b"hello", EndOfSequence()], [b"world", EndOfSequence()]],
            [np.log(0.7), np.log(0.3)],
            {"hello": 0.7, "world": 0.3},
        ),
        (
            [
                [b"hello", EndOfSequence()],
                [b"hello", EndOfSequence()],
                [b"world", EndOfSequence()],
            ],
            [np.log(4), np.log(4), np.log(2)],
            {"hello": 0.8, "world": 0.2},
        ),
        (
            [[b"hello"], [b"world"]],  # no sequence ends with EOS
            [np.log(0.6), np.log(0.4)],
            {},
        ),
        (
            [
                [b"hello", EndOfSequence()],
                [b"world"],  # no EOS -- filtered out
                [b"test", EndOfSequence()],
            ],
            [np.log(5), np.log(2), np.log(3)],
            {"hello": 5 / 8, "test": 3 / 8},  # renormalized after filtering
        ),
        (
            [
                [b"hello", EndOfSequence()],
                [bytes([0xFF, 0xFF]), EndOfSequence()],  # invalid UTF-8
                [b"world", EndOfSequence()],
            ],
            [np.log(4), np.log(2), np.log(4)],
            {"hello": 0.5, "world": 0.5},
        ),
        (
            [[EndOfSequence()]],  # just EOS
            [0.0],
            {"": 1.0},
        ),
        (
            [
                ["🌟".encode("utf-8"), EndOfSequence()],
                ["こんにちは".encode("utf-8"), EndOfSequence()],
            ],
            [np.log(3), np.log(7)],
            {"🌟": 0.3, "こんにちは": 0.7},
        ),
    ],
    ids=[
        "basic_sequence",
        "multiple_sequences",
        "duplicate_sequences_summed",
        "no_eos_is_empty",
        "mixed_eos_and_non_eos_renormalizes",
        "invalid_utf8_dropped",
        "empty_sequence_with_eos",
        "multi_byte_utf8",
    ],
)
def test_decoded_posterior(contexts, log_weights, expected):
    sequences = Sequences(contexts=contexts, log_weights=log_weights)
    posterior = sequences.decoded_posterior
    assert len(posterior) == len(expected)
    for key, prob in expected.items():
        assert np.isclose(posterior[key], prob)


def test_all_negative_infinity_weights():
    # Test handling of case where all weights are -inf
    sequences = Sequences(
        contexts=[[b"hello", EndOfSequence()], [b"world", EndOfSequence()]],
        log_weights=[-np.inf, -np.inf],
    )

    # Check all the derived quantities
    assert sequences.log_total == float("-inf")
    assert sequences.log_ml == float("-inf")
    assert np.all(np.isneginf(sequences.log_normalized_weights))
    assert sequences.log_ess == float("-inf")
    assert sequences.ess == 0.0

    # Check that posterior methods handle this case
    assert len(sequences.posterior) == 2
    assert len(sequences.decoded_posterior) == 2


@pytest.mark.parametrize(
    "contexts, log_weights",
    [
        ([[b"test", EndOfSequence()]], [0.0]),
        ([[b"a", b"b", b"c", EOS], [b"a", b"b", b"d"]], [np.log(1), np.log(9)]),
    ],
    ids=["single_sequence", "mixed_eos_and_incomplete"],
)
def test_shows(contexts, log_weights):
    """str/repr/_repr_html_/show() must not raise, incl. on an incomplete (no-EOS) sequence."""
    sequences = Sequences(contexts=contexts, log_weights=log_weights)
    sequences.show()
    repr(sequences)
    sequences._repr_html_()
    str(sequences)
