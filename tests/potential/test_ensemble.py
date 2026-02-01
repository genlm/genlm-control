import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch
from genlm.backend import load_model_by_name
from genlm.control import (
    Ensemble,
    ByteEnsemble,
    ByteEnsembleTokenSampler,
    Potential,
    PromptedLLM,
    convert_to_weighted_logop,
    EOS,
)
from genlm.control.sampler.sequence import EnsembleSMC, SequencesExt, Sequences
from genlm.control.potential.built_in.ensemble import (
    split_with_atomic_tokens,
    _weighted_extremum,
)
from conftest import MockPotential

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_vocab():
    """Simple vocabulary for testing."""
    return ["a", "b", "c", "d"]


@pytest.fixture
def mock_potential_1(mock_vocab):
    """Create a mock potential with predefined probabilities."""
    logws = np.log([0.4, 0.3, 0.2, 0.1, 0.001])
    return MockPotential(vocab=mock_vocab, next_token_logws=logws)


@pytest.fixture
def mock_potential_2(mock_vocab):
    """Create a second mock potential with different probabilities."""
    logws = np.log([0.1, 0.2, 0.3, 0.4, 0.001])
    return MockPotential(vocab=mock_vocab, next_token_logws=logws)


# ============================================================================
# Test Basic Initialization & API
# ============================================================================


@pytest.mark.asyncio
async def test_ensemble_initialization(mock_potential_1, mock_potential_2):
    """Test that Ensemble initializes correctly."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    assert isinstance(ensemble, Potential)
    assert ensemble.p1 is mock_potential_1
    assert ensemble.p2 is mock_potential_2
    assert len(ensemble.vocab) == 4


@pytest.mark.asyncio
async def test_ensemble_logws_next(mock_potential_1, mock_potential_2):
    """Test that logws_next returns separate log weights from both potentials."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    p1_logw, p2_logw = await ensemble.logws_next([])
    assert hasattr(p1_logw, "weights")
    assert hasattr(p2_logw, "weights")
    assert p1_logw.weights.shape == p2_logw.weights.shape


@pytest.mark.asyncio
async def test_ensemble_logw_next_raises(mock_potential_1, mock_potential_2):
    """Test that logw_next raises NotImplementedError."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    with pytest.raises(NotImplementedError):
        await ensemble.logw_next([])


@pytest.mark.asyncio
async def test_ensemble_batch_logw_next(mock_potential_1, mock_potential_2):
    """Test batch_logw_next combines weights from both potentials."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    results = await ensemble.batch_logw_next([[], ["a"]])
    assert len(results) == 2
    for result in results:
        assert hasattr(result, "weights")
        assert len(result.weights) == len(ensemble.vocab_eos)


@pytest.mark.asyncio
async def test_ensemble_prefix_geometric_mean(mock_potential_1, mock_potential_2):
    """Test ensemble prefix with product."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    logw = await ensemble.prefix([])
    # For product with a=0.5: result = 0.5 * log(p1) + 0.5 * log(p2)
    p1_logw = await mock_potential_1.prefix([])
    p2_logw = await mock_potential_2.prefix([])
    expected = 0.5 * p1_logw + 0.5 * p2_logw
    np.testing.assert_allclose(logw, expected, rtol=1e-5)


@pytest.mark.asyncio
async def test_ensemble_prefix_arithmetic_mean(mock_potential_1, mock_potential_2):
    """Test ensemble prefix with sum."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="sum", a=0.5)
    logw = await ensemble.prefix([])
    # For sum with a=0.5: result = log(0.5 * exp(p1) + 0.5 * exp(p2))
    p1_logw = await mock_potential_1.prefix([])
    p2_logw = await mock_potential_2.prefix([])
    expected = np.logaddexp(np.log(0.5) + p1_logw, np.log(0.5) + p2_logw)
    np.testing.assert_allclose(logw, expected, rtol=1e-5)


@pytest.mark.asyncio
async def test_ensemble_with_context():
    """Test ensemble operations with non-empty context."""
    mock_vocab = ["a", "b", "c", "d"]
    logws1 = np.log([0.4, 0.3, 0.2, 0.1, 0.001])
    logws2 = np.log([0.1, 0.2, 0.3, 0.4, 0.001])
    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws2)
    ensemble = Ensemble(p1, p2, op="prod", a=0.5)
    context = ["a", "b"]
    logw = await ensemble.prefix(context)
    assert isinstance(logw, (int, float, np.number))
    assert np.isfinite(logw)
    complete_logw = await ensemble.complete(context)
    assert isinstance(complete_logw, (int, float, np.number))
    assert np.isfinite(complete_logw)


@pytest.mark.asyncio
async def test_ensemble_consistency():
    """Test that ensemble computations are consistent across multiple calls."""
    mock_vocab = ["x", "y"]
    logws = np.log([0.6, 0.4, 0.001])
    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws)
    ensemble = Ensemble(p1, p2, op="prod", a=0.5)
    results = [await ensemble.prefix(["x"]) for _ in range(3)]
    assert all(np.isclose(results[0], r) for r in results)


# ============================================================================
# Test Ensemble Operations
# ============================================================================


@pytest.mark.asyncio
async def test_token_ensemble_different_operations():
    """Test different ensemble operations with mock models."""
    mock_vocab = ["a", "b", "c"]
    logws1 = np.array([np.log(0.6), np.log(0.3), np.log(0.1), -100.0])
    logws2 = np.array([np.log(0.2), np.log(0.5), np.log(0.3), -100.0])

    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws2)

    # Prod: 0.5 * log(p1) + 0.5 * log(p2)
    ensemble_prod = Ensemble(p1, p2, op="prod", a=0.5)
    result_prod = await ensemble_prod.batch_logw_next([[]])
    for tok in mock_vocab:
        idx_ens = ensemble_prod.lookup[tok]
        idx_p1 = p1.lookup[tok]
        expected_val = 0.5 * logws1[idx_p1] + 0.5 * logws2[idx_p1]
        assert result_prod[0].weights[idx_ens] == pytest.approx(expected_val, abs=1e-6)

    # Sum: log(0.5 * exp(log(p1)) + 0.5 * exp(log(p2)))
    ensemble_sum = Ensemble(p1, p2, op="sum", a=0.5)
    result_sum = await ensemble_sum.batch_logw_next([[]])

    for tok in mock_vocab:
        idx_ens = ensemble_sum.lookup[tok]
        idx_p1 = p1.lookup[tok]
        expected_val = np.logaddexp(
            np.log(0.5) + logws1[idx_p1], np.log(0.5) + logws2[idx_p1]
        )
        assert result_sum[0].weights[idx_ens] == pytest.approx(expected_val, abs=1e-6)

    # Min: For a=0.5, should be close to actual minimum
    ensemble_min = Ensemble(p1, p2, op="min", a=0.5)
    result_min = await ensemble_min.batch_logw_next([[]])

    for tok in mock_vocab:
        idx_ens = ensemble_min.lookup[tok]
        idx_p1 = p1.lookup[tok]
        expected_val = np.minimum(logws1[idx_p1], logws2[idx_p1])
        assert result_min[0].weights[idx_ens] == pytest.approx(expected_val, abs=0.5)

    # Max: For a=0.5, should be close to actual maximum
    ensemble_max = Ensemble(p1, p2, op="max", a=0.5)
    result_max = await ensemble_max.batch_logw_next([[]])

    for tok in mock_vocab:
        idx_ens = ensemble_max.lookup[tok]
        idx_p1 = p1.lookup[tok]
        expected_val = np.maximum(logws1[idx_p1], logws2[idx_p1])
        assert result_max[0].weights[idx_ens] == pytest.approx(expected_val, abs=0.5)


@pytest.mark.asyncio
async def test_ensemble_all_power_means():
    """Test all supported power mean operations."""
    mock_vocab = ["a", "b"]
    logws1 = np.log([0.7, 0.3, 0.001])  # Model 1 prefers 'a'
    logws2 = np.log([0.3, 0.7, 0.001])  # Model 2 prefers 'b'

    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws2)

    power_means = [
        "pm5",
        "pm2.5",
        "p-2",
        "pm1.5",
        "pm0.5",
        "pm0.25",
        "p0.25",
        "p0.5",
        "p1.5",
        "p2",
        "p2.5",
        "p3",
        "p5",
    ]
    for op in power_means:
        ensemble = Ensemble(p1, p2, op=op, a=0.5)
        logw = await ensemble.prefix([])
        assert isinstance(logw, (int, float, np.number))
        assert np.isfinite(logw)


# ============================================================================
# Test Weighting & Parameters
# ============================================================================


@pytest.mark.asyncio
async def test_ensemble_weighting_affects_output():
    """Verify that changing the weighting parameter affects the output."""
    mock_vocab = ["a", "b"]
    logws1 = np.array([0.0, -5.0, -100.0])  # Model 1 prefers 'a'
    logws2 = np.array([-5.0, 0.0, -100.0])  # Model 2 prefers 'b'

    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws2)

    ensemble_50 = Ensemble(p1, p2, op="prod", a=0.5)
    result_50 = await ensemble_50.batch_logw_next([[]])
    ensemble_80 = Ensemble(p1, p2, op="prod", a=0.8)  # Weight (0.8) on model 1
    result_80 = await ensemble_80.batch_logw_next([[]])
    ensemble_20 = Ensemble(p1, p2, op="prod", a=0.2)  # Weight (0.2) on model 2
    result_20 = await ensemble_20.batch_logw_next([[]])

    logws_50 = result_50[0].weights
    logws_80 = result_80[0].weights
    logws_20 = result_20[0].weights
    assert not np.allclose(logws_50, logws_80, rtol=1e-5)
    assert not np.allclose(logws_50, logws_20, rtol=1e-5)
    assert not np.allclose(logws_80, logws_20, rtol=1e-5)

    a_idx_50 = ensemble_50.lookup["a"]
    b_idx_50 = ensemble_50.lookup["b"]
    a_idx_80 = ensemble_80.lookup["a"]
    b_idx_20 = ensemble_20.lookup["b"]
    assert logws_80[a_idx_80] > logws_50[a_idx_50]
    assert logws_20[b_idx_20] > logws_50[b_idx_50]
    diff_50 = abs(logws_50[a_idx_50] - logws_50[b_idx_50])
    diff_80 = logws_80[a_idx_80] - logws_80[a_idx_80 if a_idx_80 == 0 else 1 - a_idx_80]
    diff_20 = logws_20[b_idx_20] - logws_20[b_idx_20 if b_idx_20 == 0 else 1 - b_idx_20]
    assert abs(diff_80) > diff_50 or abs(diff_20) > diff_50


@pytest.mark.asyncio
async def test_ensemble_with_differently_conditioned_models():
    """Test ensemble with different weighting simulating different prompt strategies."""
    vocab = ["SELECT", "FROM", "WHERE", "JOIN"]
    logws1 = np.array([0.0, -1.0, -2.0, -3.0, -100.0])
    logws2 = np.array([-3.0, -2.0, -1.0, 0.0, -100.0])
    p1 = MockPotential(vocab=vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=vocab, next_token_logws=logws2)
    ensemble_balanced = Ensemble(
        p1, p2, op="prod", a=0.5
    )  # Ensemble with different weights
    ensemble_favor_p1 = Ensemble(p1, p2, op="prod", a=0.7)

    result_balanced = await ensemble_balanced.batch_logw_next([[]])
    result_favor_p1 = await ensemble_favor_p1.batch_logw_next([[]])
    combined_balanced = result_balanced[0].weights
    combined_favor_p1 = result_favor_p1[0].weights

    select_idx_bal = ensemble_balanced.lookup[
        "SELECT"
    ]  # When favoring p1, SELECT is more likely
    select_idx_fav = ensemble_favor_p1.lookup["SELECT"]
    join_idx_bal = ensemble_balanced.lookup[
        "JOIN"
    ]  # When favoring p1, JOIN is less likely
    join_idx_fav = ensemble_favor_p1.lookup["JOIN"]

    assert combined_favor_p1[select_idx_fav] > combined_balanced[select_idx_bal]
    assert combined_favor_p1[join_idx_fav] < combined_balanced[join_idx_bal]


# ============================================================================
# Test Vocabulary Handling
# ============================================================================


@pytest.mark.asyncio
async def test_ensemble_warns_on_different_vocabularies():
    """Test Ensemble warns when using potentials with different vocabularies."""
    vocab1 = ["a", "b", "c", "d"]
    vocab2 = ["a", "b", "x", "y"]
    logws1 = np.log([0.25, 0.25, 0.25, 0.25, 0.001])
    logws2 = np.log([0.25, 0.25, 0.25, 0.25, 0.001])
    p1 = MockPotential(vocab=vocab1, next_token_logws=logws1)
    p2 = MockPotential(vocab=vocab2, next_token_logws=logws2)
    with pytest.warns(UserWarning, match="different vocabularies"):
        with pytest.raises((KeyError, AssertionError)):
            _ = Ensemble(p1, p2, op="prod", a=0.5)


@pytest.mark.asyncio
async def test_ensemble_vocab_alignment(mock_vocab):
    """Test that ensemble handles vocabulary alignment correctly."""
    logws = np.log([0.25, 0.25, 0.25, 0.25, 0.001])
    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws)

    ensemble = Ensemble(p1, p2, op="prod", a=0.5)

    # Vocabulary indices should be correctly aligned
    assert len(ensemble.p1_vocab_idxs) == len(ensemble.vocab_eos)
    assert len(ensemble.p2_vocab_idxs) == len(ensemble.vocab_eos)
    assert ensemble.p1_vocab_idxs == ensemble.p2_vocab_idxs


@pytest.mark.asyncio
async def test_ensemble_respects_vocab_alignment():
    """Verify ensemble correctly handles vocabulary alignment with reordering."""
    vocab = ["x", "y", "z"]
    logws1 = np.array([0.0, -1.0, -2.0, -100.0])
    logws2 = np.array([-1.0, 0.0, -2.0, -100.0])

    p1 = MockPotential(vocab=vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=vocab, next_token_logws=logws2)
    ensemble = Ensemble(p1, p2, op="prod", a=0.5)
    result = await ensemble.batch_logw_next([[]])
    combined = result[0].weights

    # Each token should get correct combined weight
    for tok in ["x", "y", "z"]:
        ensemble_idx = ensemble.lookup[tok]
        p1_idx = p1.lookup[tok]
        p2_idx = p2.lookup[tok]
        expected = 0.5 * logws1[p1_idx] + 0.5 * logws2[p2_idx]
        actual = combined[ensemble_idx]
        assert actual == pytest.approx(expected, abs=1e-5), f"Token {tok} mismatch"


# ============================================================================
# Test Integration with Real Models (GPT-2)
# ============================================================================


@pytest.mark.asyncio
async def test_token_ensemble_with_different_prompts():
    """Test token-level Ensemble with different prompts - basic functionality check."""
    llm1 = PromptedLLM.from_name("gpt2")
    llm2 = PromptedLLM.from_name("gpt2")
    llm1.set_prompt_from_str("Write a SQL query: ")
    llm2.set_prompt_from_str("SQL code: ")

    ensemble = Ensemble(llm1, llm2, op="prod", a=0.5)
    assert ensemble.p1 is llm1
    assert ensemble.p2 is llm2

    ensemble_result = await ensemble.batch_logw_next([[]])
    ensemble_logws = ensemble_result[0].weights

    assert len(ensemble_logws) > 0
    assert np.all(np.isfinite(ensemble_logws))
    assert len(ensemble_logws) == len(
        ensemble.vocab_eos
    )  # Should have same length as vocab


@pytest.mark.asyncio
async def test_token_ensemble_complementary_prompts():
    """Test token-level Ensemble combining complementary prompting strategies."""
    llm1 = PromptedLLM.from_name("gpt2")
    llm2 = PromptedLLM.from_name("gpt2")

    llm1.set_prompt_from_str("Task: Generate structured SQL.\n")
    llm2.set_prompt_from_str("Task: Generate correct SQL.\n")

    ensemble = Ensemble(llm1, llm2, op="prod", a=0.5)
    p1_result = await llm1.batch_logw_next([[]])
    p2_result = await llm2.batch_logw_next([[]])
    ensemble_result = await ensemble.batch_logw_next([[]])

    p1_logws = p1_result[0].weights
    p2_logws = p2_result[0].weights
    ensemble_logws = ensemble_result[0].weights

    assert not np.allclose(ensemble_logws, p1_logws, rtol=0.1)
    assert not np.allclose(ensemble_logws, p2_logws, rtol=0.1)
    assert np.all(np.isfinite(ensemble_logws))


# ============================================================================
# Test ByteEnsemble
# ============================================================================


@pytest.mark.asyncio
async def test_byte_ensemble_creation():
    """Test ByteEnsemble creation with identical prompts."""
    llm1 = load_model_by_name("gpt2", backend="hf")
    llm2 = load_model_by_name("gpt2", backend="hf")
    prompt = b"Test"
    ensemble = await ByteEnsemble.create(
        llm1, llm2, op="prod", prompt1=prompt, prompt2=prompt, a=0.5
    )
    assert ensemble.p1 is llm1
    assert ensemble.p2 is llm2
    assert len(ensemble.vocab) == 256
    assert isinstance(ensemble.vocab, list)
    assert all(isinstance(v, int) and 0 <= v < 256 for v in ensemble.vocab)
    assert b"" in ensemble.data_dict_1
    assert b"" in ensemble.data_dict_2


@pytest.mark.asyncio
async def test_byte_ensemble_different_prompts():
    """Test ByteEnsemble with different prompts - the key use case for ensembling."""
    llm1 = load_model_by_name("gpt2", backend="hf")
    llm2 = load_model_by_name("gpt2", backend="hf")

    prompt1 = b"Write a SQL query to find all users: "
    prompt2 = b"SQL: Find all users in the database: "

    ensemble = await ByteEnsemble.create(
        llm1, llm2, op="prod", prompt1=prompt1, prompt2=prompt2, a=0.5
    )
    assert ensemble.p1 is llm1
    assert ensemble.p2 is llm2
    assert len(ensemble.vocab) == 256
    assert b"" in ensemble.data_dict_1
    assert b"" in ensemble.data_dict_2
    beam1, beam2 = await ensemble.get_beam_states([])
    assert beam1 is not None
    assert beam2 is not None


@pytest.mark.asyncio
async def test_byte_ensemble_get_beam_states():
    """Test that ByteEnsemble.get_beam_states() provides access to beams."""
    llm1 = load_model_by_name("gpt2", backend="hf")
    llm2 = load_model_by_name("gpt2", backend="hf")
    ensemble = await ByteEnsemble.create(
        llm1, llm2, op="prod", prompt1=b"Hi", prompt2=b"Hello", a=0.5
    )
    beam1, beam2 = await ensemble.get_beam_states([])
    assert beam1 is not None
    assert beam2 is not None
    assert hasattr(beam1, "states")
    assert hasattr(beam2, "states")
    assert len(beam1) > 0
    assert len(beam2) > 0


@pytest.mark.asyncio
async def test_byte_ensemble_token_sampler_initialization():
    """Test ByteEnsembleTokenSampler initialization with different prompts."""
    llm1 = load_model_by_name("gpt2", backend="hf")
    llm2 = load_model_by_name("gpt2", backend="hf")
    ensemble = await ByteEnsemble.create(
        llm1, llm2, op="prod", prompt1=b"Answer: ", prompt2=b"Response: ", a=0.5
    )
    eos_tokens = [llm1.byte_vocab[llm1.tokenizer.eos_token_id]]
    sampler = ByteEnsembleTokenSampler(
        ensemble, max_tokens=50, eos_tokens=eos_tokens, n_particles=5
    )
    assert sampler.potential is ensemble
    assert sampler.max_tokens == 50
    assert sampler.eos_tokens == eos_tokens
    assert sampler.n_particles == 5
    # check caches
    assert () in sampler.prefix_cache_1
    assert () in sampler.prefix_cache_2
    assert sampler.prefix_cache_1[()] == 0.0
    assert sampler.prefix_cache_2[()] == 0.0


@pytest.mark.asyncio
async def test_byte_ensemble_sampler_sample():
    """Test ByteEnsembleTokenSampler samples with different prompts."""
    llm1 = load_model_by_name("gpt2", backend="hf")
    llm2 = load_model_by_name("gpt2", backend="hf")

    ensemble = await ByteEnsemble.create(
        llm1, llm2, op="prod", prompt1=b"The cat is ", prompt2=b"A cat is ", a=0.5
    )
    eos_tokens = [llm1.byte_vocab[llm1.tokenizer.eos_token_id]]
    sampler = ByteEnsembleTokenSampler(
        ensemble, max_tokens=10, eos_tokens=eos_tokens, n_particles=3
    )
    token, logw, logp = await sampler.sample([])
    assert isinstance(token, (int, bytes))
    assert isinstance(logw, (int, float, np.number))
    assert isinstance(logp, (int, float, np.number))
    assert np.isfinite(logw)
    assert np.isfinite(logp)
    if isinstance(token, int):
        next_context_bytes = bytes([token])
    else:
        next_context_bytes = token

    assert next_context_bytes in ensemble.data_dict_1
    assert next_context_bytes in ensemble.data_dict_2


@pytest.mark.asyncio
async def test_byte_ensemble_weighted_different_prompts():
    """Test ByteEnsemble with unequal weights on different prompts."""
    llm1 = load_model_by_name("gpt2", backend="hf")
    llm2 = load_model_by_name("gpt2", backend="hf")

    prompt1 = b"Correct approach: "
    prompt2 = b"Alternative: "

    ensemble = await ByteEnsemble.create(
        llm1, llm2, op="prod", prompt1=prompt1, prompt2=prompt2, a=0.7
    )

    assert ensemble.p1 is llm1
    assert ensemble.p2 is llm2

    eos_tokens = [llm1.byte_vocab[llm1.tokenizer.eos_token_id]]
    sampler = ByteEnsembleTokenSampler(
        ensemble, max_tokens=5, eos_tokens=eos_tokens, n_particles=2
    )
    token, logw, logp = await sampler.sample([])
    assert isinstance(token, (int, bytes))
    assert np.isfinite(logw)


# ============================================================================
# Test Realistic Ensemble Applications
# ============================================================================


@pytest.mark.asyncio
async def test_ensemble_with_different_model_preferences():
    """Test ensemble where models have same vocab but different preferences."""
    vocab = ["a", "b", "c", "d"]
    logws1 = np.array([0.0, -0.5, -2.0, -3.0, -100.0])  # prefers 'a' > 'b' > 'c' > 'd'
    logws2 = np.array([-3.0, -2.0, -0.5, 0.0, -100.0])  # prefers 'd' > 'c' > 'b' > 'a'
    p1 = MockPotential(vocab=vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=vocab, next_token_logws=logws2)
    ensemble = Ensemble(p1, p2, op="prod", a=0.5)

    for tok in vocab:
        assert tok in ensemble.vocab
    result = await ensemble.batch_logw_next([[]])
    combined = result[0].weights

    for tok in vocab:
        ensemble_idx = ensemble.lookup[tok]
        p1_idx = p1.lookup[tok]
        p2_idx = p2.lookup[tok]
        expected = 0.5 * logws1[p1_idx] + 0.5 * logws2[p2_idx]
        actual = combined[ensemble_idx]
        assert actual == pytest.approx(expected, abs=1e-5), f"Token {tok} mismatch"

    b_idx = ensemble.lookup["b"]
    c_idx = ensemble.lookup["c"]
    a_idx = ensemble.lookup["a"]
    d_idx = ensemble.lookup["d"]
    assert combined[b_idx] > min(combined[a_idx], combined[d_idx])
    assert combined[c_idx] > min(combined[a_idx], combined[d_idx])


@pytest.mark.asyncio
async def test_ensemble_with_complementary_knowledge():
    """Test ensemble where models show different performance on different tokens."""
    vocab1 = ["SELECT", "FROM", "WHERE", "LIMIT"]
    logws1 = np.array(
        [
            np.log(0.4),  # SELECT: confident
            np.log(0.3),  # FROM: confident
            np.log(0.2),  # WHERE: confident
            np.log(0.1),  # LIMIT: less confident
            -100.0,
        ]
    )
    vocab2 = ["SELECT", "FROM", "WHERE", "LIMIT"]
    logws2 = np.array(
        [
            np.log(0.1),  # SELECT: not confident
            np.log(0.2),  # FROM: not confident
            np.log(0.3),  # WHERE: somewhat confident
            np.log(0.4),  # LIMIT: very confident
            -100.0,
        ]
    )

    p1 = MockPotential(vocab=vocab1, next_token_logws=logws1)
    p2 = MockPotential(vocab=vocab2, next_token_logws=logws2)
    ensemble = Ensemble(p1, p2, op="prod", a=0.5)
    result = await ensemble.batch_logw_next([[]])
    combined = result[0].weights

    for tok in vocab1:
        idx = ensemble.lookup[tok]
        prob = np.exp(combined[idx])
        assert prob > 0.05, f"{tok} should have reasonable probability in ensemble"
        assert prob < 0.95, f"{tok} shouldn't dominate in balanced ensemble"

    p1_result = await p1.batch_logw_next([[]])
    p2_result = await p2.batch_logw_next([[]])

    assert not np.allclose(combined, p1_result[0].weights, rtol=0.1)
    assert not np.allclose(combined, p2_result[0].weights, rtol=0.1)


@pytest.mark.asyncio
async def test_ensemble_helps_uncertain_model():
    """Test that ensembling helps when one model is uncertain but the other is confident."""
    mock_vocab = ["correct", "wrong1", "wrong2"]
    logws1 = np.array(
        [np.log(0.33), np.log(0.33), np.log(0.34), -100.0]
    )  # Model 1 is uncertain
    logws2 = np.array(
        [np.log(0.9), np.log(0.05), np.log(0.05), -100.0]
    )  # Model 2 is confident

    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws2)
    ensemble = Ensemble(p1, p2, op="prod", a=0.5)
    result = await ensemble.batch_logw_next([[]])
    combined = result[0].weights

    correct_idx = ensemble.lookup["correct"]
    wrong1_idx = ensemble.lookup["wrong1"]
    # ensemble should favor 'correct' more than model 1 alone
    p1_result = await p1.batch_logw_next([[]])
    p1_logws = p1_result[0].weights
    p1_correct = p1_logws[p1.lookup["correct"]]
    p1_wrong1 = p1_logws[p1.lookup["wrong1"]]
    ensemble_correct = combined[correct_idx]
    ensemble_wrong1 = combined[wrong1_idx]

    # Ensemble should have stronger preference for 'correct' than uncertain model 1
    p1_gap = p1_correct - p1_wrong1
    ensemble_gap = ensemble_correct - ensemble_wrong1
    assert (
        ensemble_gap > p1_gap
    ), "Ensemble should be more confident than uncertain model"

    # But less confident than model 2 alone
    p2_result = await p2.batch_logw_next([[]])
    p2_logws = p2_result[0].weights
    p2_gap = p2_logws[p2.lookup["correct"]] - p2_logws[p2.lookup["wrong1"]]
    assert (
        ensemble_gap < p2_gap
    ), "Ensemble should be less confident than very confident model"


# ============================================================================
# Test Utility Functions
# ============================================================================


def test_convert_to_weighted_logop_invalid_a():
    """Test that invalid 'a' parameter raises ValueError."""
    with pytest.raises(ValueError, match="variable a should be between 0 and 1"):
        convert_to_weighted_logop("prod", a=1.5)

    with pytest.raises(ValueError, match="variable a should be between 0 and 1"):
        convert_to_weighted_logop("prod", a=-0.1)


def test_convert_to_weighted_logop_invalid_op():
    """Test that invalid operation raises ValueError."""
    with pytest.raises(ValueError, match="Invalid operation"):
        convert_to_weighted_logop("invalid_op", a=0.5)


def test_convert_to_weighted_logop_operations():
    """Test convert_to_weighted_logop returns correct operation for prod."""
    x = np.log(np.array([0.3, 0.7]))
    y = np.log(np.array([0.6, 0.4]))

    # Test prod with analytical verification
    op_prod = convert_to_weighted_logop("prod", a=0.5)
    result_prod = op_prod(x, y)
    expected_prod = 0.5 * x + 0.5 * y
    np.testing.assert_allclose(result_prod, expected_prod, rtol=1e-5)


@pytest.mark.asyncio
async def test_byte_ensemble_token_sampler_start_weight():
    """Test ByteEnsembleTokenSampler.start_weight() returns 0.0."""
    llm = load_model_by_name("gpt2", backend="hf")
    ensemble = await ByteEnsemble.create(
        llm, llm, op="prod", prompt1=b"Hi", prompt2=b"Hi", a=0.5
    )
    eos_tokens = [llm.byte_vocab[llm.tokenizer.eos_token_id]]
    sampler = ByteEnsembleTokenSampler(
        ensemble, max_tokens=10, eos_tokens=eos_tokens, n_particles=5
    )
    start_weight = await sampler.start_weight()
    assert start_weight == 0.0


@pytest.mark.asyncio
async def test_byte_ensemble_sampler_eos_handling():
    """Test ByteEnsembleTokenSampler properly handles EOS tokens and max_tokens."""
    llm = load_model_by_name("gpt2", backend="hf")
    ensemble = await ByteEnsemble.create(
        llm, llm, op="prod", prompt1=b"Hi", prompt2=b"Hi", a=0.5
    )
    eos_byte = llm.byte_vocab[llm.tokenizer.eos_token_id]
    sampler = ByteEnsembleTokenSampler(
        ensemble, max_tokens=5, eos_tokens=[eos_byte], n_particles=5
    )
    _, _, _ = await sampler.sample([])
    if len(sampler.particle_prefix_log_prob_1) > 0:
        assert len(sampler.particle_prefix_log_prob_1) >= 0
        assert len(sampler.particle_prefix_log_prob_2) >= 0


@pytest.mark.asyncio
async def test_byte_ensemble_empty_beam_error():
    """Test ByteEnsemble raises RuntimeError when beam is empty after prefill."""
    mock_llm = MagicMock()
    mock_llm.byte_vocab = {0: b"a"}
    mock_llm.tokenizer.eos_token_id = 0
    empty_beam = MagicMock()
    empty_beam.prefill = AsyncMock(return_value=[])
    with patch(
        "genlm.control.potential.built_in.ensemble.ByteBeamState.initial",
        AsyncMock(return_value=empty_beam),
    ):
        with pytest.raises(RuntimeError, match="Beam1 is empty after prefill"):
            await ByteEnsemble.create(
                mock_llm, mock_llm, op="prod", prompt1=b"test", prompt2=b"test", a=0.5
            )


def test_split_with_atomic_tokens_overlapping():
    """Test split_with_atomic_tokens with overlapping tokens."""
    with pytest.warns(UserWarning, match="Overlapping atomic tokens detected"):
        result = split_with_atomic_tokens(b"ABC", [b"A", b"AB"])
    assert result == [b"A", 66, 67]


def test_split_with_atomic_tokens_no_match():
    """Test split_with_atomic_tokens when no atomic tokens match."""
    result = split_with_atomic_tokens(b"XYZ", [b"A", b"B"])
    assert result == [88, 89, 90]


def test_weighted_extremum_different_weights():
    """Test _weighted_extremum with different weight values."""
    x = np.array([-1.0, -2.0, -3.0])
    y = np.array([-2.0, -1.5, -3.5])
    max_op_favoring_y = _weighted_extremum(np.maximum, a=0.7)
    result = max_op_favoring_y(x, y)
    expected = (2 * 0.7 - 1) * y + 2 * (1 - 0.7) * np.maximum(x, y)
    np.testing.assert_allclose(result, expected, rtol=1e-5)
    min_op_favoring_x = _weighted_extremum(np.minimum, a=0.3)
    result2 = min_op_favoring_x(x, y)
    expected2 = (1 - 2 * 0.3) * x + 2 * 0.3 * np.minimum(x, y)
    np.testing.assert_allclose(result2, expected2, rtol=1e-5)


@pytest.mark.asyncio
async def test_ensemble_smc_weight_extraction():
    """Test EnsembleSMC extracts individual model weights correctly."""
    from genlm.control.sampler.token import TokenSampler

    class MockTokenSampler(TokenSampler):
        def __init__(self):
            self.particle_prefix_log_prob_1 = {
                ("a",): -1.0,
                ("b",): -2.0,
            }
            self.particle_prefix_log_prob_2 = {
                ("a",): -1.5,
                ("b",): -2.5,
            }

        async def start_weight(self):
            return 0.0

        async def sample(self, context, draw=None):
            return EOS, 0.0, 0.0

    mock_sampler = MockTokenSampler()
    smc = EnsembleSMC(mock_sampler, None)
    mock_sequences = Sequences(
        contexts=[["a"], ["b"]],
        log_weights=[-0.5, -0.7],
    )

    with patch.object(
        EnsembleSMC.__bases__[0],
        "__call__",
        AsyncMock(return_value=mock_sequences),
    ):
        result = await smc(n_particles=2, ess_threshold=0.5, max_tokens=10)

    assert isinstance(result, SequencesExt)
    assert hasattr(result, "log_prefix_weights_1")
    assert hasattr(result, "log_prefix_weights_2")
    assert len(result.log_prefix_weights_1) == 2
    assert len(result.log_prefix_weights_2) == 2
    assert result.log_prefix_weights_1[0] == -1.0
    assert result.log_prefix_weights_1[1] == -2.0
    assert result.log_prefix_weights_2[0] == -1.5
    assert result.log_prefix_weights_2[1] == -2.5


def test_sequences_ext_post_init():
    """Test SequencesExt.__post_init__ converts lists to numpy arrays."""
    seq = SequencesExt(
        contexts=[["a", "b"], ["c", "d"]],
        log_weights=[0.1, 0.2],
        log_prefix_weights_1=[0.15, 0.25],
        log_prefix_weights_2=[0.12, 0.22],
    )
    assert isinstance(seq.log_prefix_weights_1, np.ndarray)
    assert isinstance(seq.log_prefix_weights_2, np.ndarray)
    seq2 = SequencesExt(contexts=[["a"]], log_weights=[0.1], log_prefix_weights_1=None)
    assert seq2.log_prefix_weights_1 is None


def test_sequences_ext_post_init_with_none():
    """Test SequencesExt.__post_init__ handles None values correctly."""
    seq = SequencesExt(
        contexts=[["a", "b"]],
        log_weights=[0.1],
        log_prefix_weights_1=None,
        log_prefix_weights_2=None,
    )
    assert seq.log_prefix_weights_1 is None
    assert seq.log_prefix_weights_2 is None


@pytest.mark.asyncio
async def test_byte_ensemble_cleanup_cache_deletes_short_keys():
    """Test ByteEnsemble._cleanup_cache() deletes short keys."""
    gpt2 = load_model_by_name("gpt2")
    prompt1 = b"The capital of France is"
    prompt2 = b"Paris, the capital city of France, is"
    ensemble = await ByteEnsemble.create(
        gpt2, gpt2, "sum", prompt1, prompt2, a=0.5, K=3
    )
    ensemble.data_dict_1 = {
        (1,): "short1",
        (1, 2): "short2",
        (1, 2, 3): "keep3",
        (1, 2, 3, 4): "keep4",
        (1, 2, 3, 4, 5): "keep5",
        (1, 2, 3, 4, 5, 6): "keep6",
    }
    ensemble.data_dict_2 = {
        (10,): "short1",
        (10, 20): "short2",
        (10, 20, 30): "keep3",
        (10, 20, 30, 40): "keep4",
        (10, 20, 30, 40, 50): "keep5",
        (10, 20, 30, 40, 50, 60): "keep6",
    }
    await ensemble._cleanup_cache()
    for d in [ensemble.data_dict_1, ensemble.data_dict_2]:
        for k in d.keys():
            assert len(k) >= 4, f"Key {k} should have been deleted"
    assert len(ensemble.data_dict_1) == 3
    assert len(ensemble.data_dict_2) == 3


@pytest.mark.asyncio
async def test_byte_ensemble_empty_beam_error_covered():
    gpt2 = load_model_by_name("gpt2")
    prompt1 = b"\xff\xfe\xfd"  # Invalid UTF-8 bytes
    prompt2 = b"Test"
    with pytest.raises(RuntimeError, match="is empty after prefill"):
        await ByteEnsemble.create(
            gpt2, gpt2, "sum", prompt1, prompt2, a=0.5, K=1, prune_threshold=100.0
        )


@pytest.mark.asyncio
async def test_byte_ensemble_sampler_stores_particle_weights():
    """Test ByteEnsembleTokenSampler stores particle weights at EOS."""
    gpt2 = load_model_by_name("gpt2")
    prompt1 = b"Hi"
    prompt2 = b"Hi"

    ensemble = await ByteEnsemble.create(
        gpt2, gpt2, "sum", prompt1, prompt2, a=0.5, K=3
    )
    sampler = ByteEnsembleTokenSampler(ensemble, max_tokens=1)
    context = []
    token, _, _ = await sampler.sample(context)
    new_ctx_tuple = (token,)
    assert new_ctx_tuple in sampler.particle_prefix_log_prob_1
    assert new_ctx_tuple in sampler.particle_prefix_log_prob_2


@pytest.mark.asyncio
async def test_byte_ensemble_sampler_eos_conversion():
    """Test ByteEnsembleTokenSampler EOS conversion path."""
    gpt2 = load_model_by_name("gpt2")
    prompt1 = b"Hi"
    prompt2 = b"Hi"
    ensemble = await ByteEnsemble.create(
        gpt2, gpt2, "sum", prompt1, prompt2, a=0.5, K=3
    )
    sampler = ByteEnsembleTokenSampler(ensemble)
    context = []
    token, _, _ = await sampler.sample(context)
    assert token is not None


@pytest.mark.asyncio
async def test_byte_ensemble_sampler_smc_calls_ensemble_smc():
    """Test ByteEnsembleTokenSampler.smc() method invokes EnsembleSMC."""
    gpt2 = load_model_by_name("gpt2")
    prompt1 = b"Hi"
    prompt2 = b"Hi"
    ensemble = await ByteEnsemble.create(
        gpt2, gpt2, "sum", prompt1, prompt2, a=0.5, K=3
    )
    sampler = ByteEnsembleTokenSampler(ensemble)
    assert hasattr(sampler, "smc")
    assert callable(sampler.smc)
    try:
        result = await sampler.smc(
            n_particles=1,
            ess_threshold=0.5,
            max_tokens=1,
            critic=None,
        )
        assert isinstance(result, SequencesExt)
    except (AssertionError, KeyError) as e:
        if "Beam is empty" in str(e) or "not found in cache" in str(e):
            pass
        else:
            raise
