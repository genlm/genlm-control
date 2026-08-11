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


def test_posterior():
    # Test posterior without EOS filtering
    sequences = Sequences(
        contexts=[
            [b"hello"],  # No EOS
            [b"world", EndOfSequence()],
        ],
        log_weights=[np.log(0.4), np.log(0.6)],
    )
    posterior = sequences.posterior
    assert len(posterior) == 2
    assert np.isclose(posterior[tuple([b"hello"])], 0.4)
    assert np.isclose(posterior[tuple([b"world", EndOfSequence()])], 0.6)


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


def test_posterior_normalization():
    # Test that posterior probabilities sum to 1
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [b"world", EndOfSequence()],
            [b"test", EndOfSequence()],
        ],
        log_weights=[np.log(2), np.log(5), np.log(3)],
    )
    posterior = sequences.posterior
    assert np.isclose(sum(posterior.values()), 1.0)


def test_string_representation():
    sequences = Sequences(contexts=[[b"test", EndOfSequence()]], log_weights=[0.0])
    # Test that string representation doesn't raise errors
    str(sequences)
    repr(sequences)


def test_decoded_posterior_basic_sequence():
    # Simple case with one valid UTF-8 sequence
    sequences = Sequences(contexts=[[b"hello", EndOfSequence()]], log_weights=[0.0])
    posterior = sequences.decoded_posterior
    assert len(posterior) == 1
    assert posterior["hello"] == 1.0


def test_decoded_posterior_multiple_sequences():
    # Multiple different valid sequences
    sequences = Sequences(
        contexts=[[b"hello", EndOfSequence()], [b"world", EndOfSequence()]],
        log_weights=[np.log(0.7), np.log(0.3)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    assert np.isclose(posterior["hello"], 0.7)
    assert np.isclose(posterior["world"], 0.3)


def test_duplicate_sequences():
    # Test that duplicate sequences have their probabilities summed
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [b"hello", EndOfSequence()],
            [b"world", EndOfSequence()],
        ],
        log_weights=[np.log(4), np.log(4), np.log(2)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    assert np.isclose(posterior["hello"], 0.8)
    assert np.isclose(posterior["world"], 0.2)


def test_no_eos_sequences():
    # Test when no sequences end with EOS
    sequences = Sequences(
        contexts=[[b"hello"], [b"world"]],
        log_weights=[np.log(0.6), np.log(0.4)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 0


def test_mixed_eos_and_non_eos():
    # Test mixture of EOS and non-EOS sequences
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [b"world"],  # No EOS
            [b"test", EndOfSequence()],
        ],
        log_weights=[np.log(5), np.log(2), np.log(3)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    # Note: weights should be renormalized after filtering
    total_weight = 5 + 3
    assert np.isclose(posterior["hello"], 5 / total_weight)
    assert np.isclose(posterior["test"], 3 / total_weight)


def test_invalid_utf8_sequences():
    # Test handling of invalid UTF-8 sequences
    invalid_bytes = bytes([0xFF, 0xFF])  # Invalid UTF-8
    sequences = Sequences(
        contexts=[
            [b"hello", EndOfSequence()],
            [invalid_bytes, EndOfSequence()],
            [b"world", EndOfSequence()],
        ],
        log_weights=[np.log(4), np.log(2), np.log(4)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    total_weight = 4 + 4
    assert np.isclose(posterior["hello"], 4 / total_weight)
    assert np.isclose(posterior["world"], 4 / total_weight)


def test_empty_sequence_with_eos():
    # Test sequence that's just EOS
    sequences = Sequences(contexts=[[EndOfSequence()]], log_weights=[0.0])
    posterior = sequences.decoded_posterior
    assert len(posterior) == 1
    assert posterior[""] == 1.0


def test_multi_byte_utf8():
    # Test with multi-byte UTF-8 characters
    sequences = Sequences(
        contexts=[
            ["🌟".encode("utf-8"), EndOfSequence()],
            ["こんにちは".encode("utf-8"), EndOfSequence()],
        ],
        log_weights=[np.log(3), np.log(7)],
    )
    posterior = sequences.decoded_posterior
    assert len(posterior) == 2
    assert np.isclose(posterior["🌟"], 0.3)
    assert np.isclose(posterior["こんにちは"], 0.7)


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


def test_shows():
    sequences = Sequences(
        contexts=[[b"a", b"b", b"c", EOS], [b"a", b"b", b"d"]],
        log_weights=[np.log(1), np.log(9)],
    )
    sequences.show()
    repr(sequences)
    sequences._repr_html_()
    str(sequences)
