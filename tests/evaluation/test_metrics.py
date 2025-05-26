import pytest
import numpy as np
from unittest.mock import AsyncMock

from genlm.control.evaluation.metrics import (
    kl_divergence_direct,
    kl_divergence_potentials,
    kl_divergence_sequences,
    effective_sample_size,
    perplexity_from_kl,
)
from genlm.control.sampler.sequence import Sequences
from genlm.control.constant import EndOfSequence


class MockLanguageModel:
    """Mock language model for testing."""

    def __init__(self, vocab_probs):
        self.vocab_probs = vocab_probs
        self.vocab = list(vocab_probs.keys())
        self.probs = np.array(list(vocab_probs.values()))
        self.probs = self.probs / self.probs.sum()

    def log_prob(self, samples):
        """Return log probabilities for samples."""
        if isinstance(samples, str):
            samples = [samples]

        log_probs = []
        for sample in samples:
            if sample in self.vocab_probs:
                prob = self.vocab_probs[sample] / sum(self.vocab_probs.values())
                log_probs.append(np.log(prob))
            else:
                log_probs.append(np.log(1e-10))

        return np.array(log_probs)


class TestKLDivergenceDirect:
    """Test direct KL divergence computation."""

    def test_identical_models(self):
        """KL divergence should be near 0 for identical models."""
        model_p = MockLanguageModel({"hello": 0.6, "world": 0.4})
        model_q = MockLanguageModel({"hello": 0.6, "world": 0.4})

        samples = ["hello", "world", "hello", "hello", "world"]
        kl = kl_divergence_direct(model_p, model_q, samples)

        assert np.isclose(kl, 0.0, atol=1e-6)

    def test_different_models(self):
        """KL divergence should be positive for different models."""
        model_p = MockLanguageModel({"hello": 0.8, "world": 0.2})
        model_q = MockLanguageModel({"hello": 0.2, "world": 0.8})

        samples = ["hello", "hello", "hello", "world"]
        kl = kl_divergence_direct(model_p, model_q, samples)

        assert kl > 0
        assert np.isfinite(kl)

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        model_p = MockLanguageModel({"a": 0.7, "b": 0.3})
        model_q = MockLanguageModel({"a": 0.3, "b": 0.7})

        samples = ["a"] * 10 + ["b"] * 5

        # Test with different batch sizes
        kl_batch1 = kl_divergence_direct(model_p, model_q, samples, batch_size=1)
        kl_batch5 = kl_divergence_direct(model_p, model_q, samples, batch_size=5)
        kl_batch32 = kl_divergence_direct(model_p, model_q, samples, batch_size=32)

        # Results should be the same regardless of batch size
        assert np.isclose(kl_batch1, kl_batch5, atol=1e-6)
        assert np.isclose(kl_batch5, kl_batch32, atol=1e-6)

    def test_numerical_stability(self):
        """Test numerical stability with extreme probabilities."""
        model_p = MockLanguageModel({"common": 0.999, "rare": 0.001})
        model_q = MockLanguageModel({"common": 0.001, "rare": 0.999})

        samples = ["common"] * 100 + ["rare"]
        kl = kl_divergence_direct(model_p, model_q, samples)

        assert np.isfinite(kl)
        assert kl > 0


@pytest.mark.asyncio
class TestKLDivergencePotentials:
    """Test KL divergence with Potential objects."""

    async def test_mock_potentials(self):
        """Test with mock potential objects."""
        # Create mock potentials
        potential_p = AsyncMock()
        potential_q = AsyncMock()

        # Set up return values
        potential_p.complete.side_effect = [np.log(0.6), np.log(0.4)]
        potential_q.complete.side_effect = [np.log(0.4), np.log(0.6)]

        samples = ["hello", "world"]
        kl = await kl_divergence_potentials(potential_p, potential_q, samples)

        assert np.isfinite(kl)
        # Verify potentials were called correctly
        assert potential_p.complete.call_count == 2
        assert potential_q.complete.call_count == 2


class TestKLDivergenceSequences:
    """Test KL divergence between Sequences objects."""

    def create_test_sequences(self, contexts, weights):
        """Helper to create test sequences."""
        return Sequences(contexts, weights)

    def test_identical_sequences(self):
        """KL divergence should be 0 for identical sequence distributions."""
        contexts = [
            [b"hello", b" world", EndOfSequence()],
            [b"goodbye", b" world", EndOfSequence()],
        ]
        weights = [np.log(0.6), np.log(0.4)]

        seq1 = self.create_test_sequences(contexts, weights)
        seq2 = self.create_test_sequences(contexts, weights)

        kl = kl_divergence_sequences(seq1, seq2)
        assert np.isclose(kl, 0.0, atol=1e-6)

    def test_different_sequences(self):
        """KL divergence should be positive for different distributions."""
        contexts = [
            [b"hello", b" world", EndOfSequence()],
            [b"goodbye", b" world", EndOfSequence()],
        ]

        # Different weight distributions
        weights1 = [np.log(0.8), np.log(0.2)]  # Peaked
        weights2 = [np.log(0.4), np.log(0.6)]  # Different peak

        seq1 = self.create_test_sequences(contexts, weights1)
        seq2 = self.create_test_sequences(contexts, weights2)

        kl = kl_divergence_sequences(seq1, seq2)
        assert kl > 0
        assert np.isfinite(kl)

    def test_empty_sequences_error(self):
        """Should raise error for empty sequences."""
        empty_contexts = []
        empty_weights = []

        seq1 = self.create_test_sequences(empty_contexts, empty_weights)
        seq2 = self.create_test_sequences(empty_contexts, empty_weights)

        with pytest.raises(ValueError, match="No decodable sequences found"):
            kl_divergence_sequences(seq1, seq2)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_effective_sample_size(self):
        """Test effective sample size extraction."""
        contexts = [
            [b"hello", EndOfSequence()],
            [b"world", EndOfSequence()],
        ]
        weights = [np.log(0.7), np.log(0.3)]

        sequences = Sequences(contexts, weights)
        ess = effective_sample_size(sequences)

        assert isinstance(ess, float)
        assert ess > 0
        assert ess <= len(sequences)

    def test_perplexity_from_kl(self):
        """Test perplexity calculation from KL divergence."""
        kl_div = 1.0
        perplexity = perplexity_from_kl(kl_div)

        expected = np.exp(1.0)
        assert np.isclose(perplexity, expected)

    def test_perplexity_zero_kl(self):
        """Test perplexity with zero KL divergence."""
        kl_div = 0.0
        perplexity = perplexity_from_kl(kl_div)

        assert np.isclose(perplexity, 1.0)
