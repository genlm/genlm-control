import pytest
import numpy as np
from genlm.backend import load_model_by_name
from genlm.control import (
    Ensemble,
    ByteLLM,
    Potential,
    PromptedLLM,
    convert_to_weighted_logop,
    direct_token_sampler,
    EOS,
)
from genlm.control.potential.built_in.ensemble import _weighted_extremum
from genlm.bytes import BeamParams
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


def _normalized_mock(vocab, probs):
    """A mock LM whose next-token weights are a normalized distribution."""
    return MockPotential(vocab=vocab, next_token_logws=np.log(probs))


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["sum", "prod", "harmonic", "min", "max", 2.5])
async def test_ensemble_logw_next_matches_prefix(op):
    """Test logw_next(ctx)[x] == prefix(ctx + [x]) - prefix(ctx), the Potential contract."""
    vocab = ["a", "b", "c"]
    p1 = _normalized_mock(vocab, [0.5, 0.2, 0.2, 0.1])
    p2 = _normalized_mock(vocab, [0.1, 0.3, 0.4, 0.2])
    ensemble = Ensemble(p1, p2, op=op, a=0.3)
    for context in [[], ["a"], ["c", "b"]]:
        logws = await ensemble.logw_next(context)
        base = await ensemble.prefix(context)
        for token in vocab:
            expected = await ensemble.prefix(context + [token]) - base
            assert logws[token] == pytest.approx(expected, abs=1e-10)
        expected_eos = await ensemble.complete(context) - base
        assert logws[ensemble.eos] == pytest.approx(expected_eos, abs=1e-10)


@pytest.mark.asyncio
async def test_ensemble_memo_matches_recomputation():
    """Test prefix weights read from the memo equal those recomputed without it."""
    vocab = ["a", "b"]
    p1 = _normalized_mock(vocab, [0.6, 0.3, 0.1])
    p2 = _normalized_mock(vocab, [0.2, 0.5, 0.3])
    warm = Ensemble(p1, p2, op="sum", a=0.4)
    await warm.batch_logw_next([[], ["a"]])
    cold = Ensemble(p1, p2, op="sum", a=0.4)
    for context in [["a"], ["b"], ["a", "b"]]:
        assert await warm.prefix(context) == pytest.approx(
            await cold.prefix(context), abs=1e-12
        )


@pytest.mark.asyncio
async def test_ensemble_component_logws():
    """Test per-model weights: prefix weights, and complete weights after EOS."""
    vocab = ["a", "b"]
    p1 = _normalized_mock(vocab, [0.6, 0.3, 0.1])
    p2 = _normalized_mock(vocab, [0.2, 0.5, 0.3])
    for warm in [False, True]:
        ensemble = Ensemble(p1, p2, op="sum", a=0.4)
        if warm:
            await ensemble.batch_logw_next([[], ["a"]])
        w1, w2 = await ensemble.component_logws(["a", "b"])
        assert w1 == pytest.approx(await p1.prefix(["a", "b"]), abs=1e-12)
        assert w2 == pytest.approx(await p2.prefix(["a", "b"]), abs=1e-12)
        w1, w2 = await ensemble.component_logws(["a", EOS])
        assert w1 == pytest.approx(await p1.complete(["a"]), abs=1e-12)
        assert w2 == pytest.approx(await p2.complete(["a"]), abs=1e-12)


@pytest.mark.asyncio
async def test_ensemble_component_logws_after_smc():
    """Test per-model weights are available for every SMC sample, incl. forced EOS."""
    vocab = ["a", "b"]
    p1 = _normalized_mock(vocab, [0.45, 0.45, 0.1])
    p2 = _normalized_mock(vocab, [0.2, 0.5, 0.3])
    ensemble = Ensemble(p1, p2, op="sum", a=0.4)
    sequences = await direct_token_sampler(ensemble).smc(
        n_particles=5, ess_threshold=0.5, max_tokens=3
    )
    for context in sequences.contexts:
        w1, w2 = await ensemble.component_logws(context)
        assert w1 == pytest.approx(await p1.complete(context[:-1]), abs=1e-12)
        assert w2 == pytest.approx(await p2.complete(context[:-1]), abs=1e-12)


@pytest.mark.asyncio
async def test_ensemble_prod_is_local_product():
    """Test the weighted product's next-token weights don't depend on the prefix."""
    vocab = ["a", "b"]
    p1 = _normalized_mock(vocab, [0.6, 0.3, 0.1])
    p2 = _normalized_mock(vocab, [0.2, 0.5, 0.3])
    ensemble = Ensemble(p1, p2, op="prod", a=0.3)
    expected = 0.3 * p1.next_token_logws + 0.7 * p2.next_token_logws
    for context in [[], ["a", "b", "a"]]:
        np.testing.assert_allclose(
            (await ensemble.logw_next(context)).weights, expected, atol=1e-12
        )


@pytest.mark.asyncio
async def test_ensemble_sum_preserves_mass():
    """Test a sum ensemble of normalized models has every SMC step weight log 1 = 0."""
    vocab = ["a", "b"]
    p1 = _normalized_mock(vocab, [0.6, 0.3, 0.1])
    p2 = _normalized_mock(vocab, [0.2, 0.5, 0.3])
    sampler = direct_token_sampler(Ensemble(p1, p2, op="sum", a=0.3))
    context = []
    for _ in range(10):
        token, logw, _ = await sampler.sample(context)
        assert logw == pytest.approx(0.0, abs=1e-12)
        if token is EOS:
            break
        context.append(token)


@pytest.mark.asyncio
async def test_ensemble_batch_logw_next(mock_potential_1, mock_potential_2):
    """Test batch_logw_next combines weights from both potentials."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    results = await ensemble.batch_logw_next([[], ["a"]])
    assert results.weights.shape == (2, len(ensemble.vocab_eos))
    expected = 0.5 * mock_potential_1.next_token_logws + 0.5 * (
        mock_potential_2.next_token_logws
    )
    for row in results.weights:
        np.testing.assert_allclose(row, expected)


@pytest.mark.asyncio
async def test_ensemble_logw_eos(mock_potential_1, mock_potential_2):
    """Test logw_eos combines both potentials' EOS weights."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    logw = await ensemble.logw_eos(["a"])
    np.testing.assert_allclose(logw, np.log(0.001))


@pytest.mark.asyncio
async def test_ensemble_smc_forces_eos_at_max_tokens(
    mock_potential_1, mock_potential_2
):
    """Test SMC over a token-level ensemble terminates at max_tokens."""
    ensemble = Ensemble(mock_potential_1, mock_potential_2, op="prod", a=0.5)
    sampler = direct_token_sampler(ensemble)
    sequences = await sampler.smc(n_particles=4, ess_threshold=0.5, max_tokens=3)
    assert len(sequences) == 4
    for ctx, logw in zip(sequences.contexts, sequences.log_weights):
        assert ctx[-1] is EOS
        assert len(ctx) <= 3
        assert np.isfinite(logw)


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
        assert result_prod.weights[0][idx_ens] == pytest.approx(expected_val, abs=1e-6)

    # Sum: log(0.5 * exp(log(p1)) + 0.5 * exp(log(p2)))
    ensemble_sum = Ensemble(p1, p2, op="sum", a=0.5)
    result_sum = await ensemble_sum.batch_logw_next([[]])

    for tok in mock_vocab:
        idx_ens = ensemble_sum.lookup[tok]
        idx_p1 = p1.lookup[tok]
        expected_val = np.logaddexp(
            np.log(0.5) + logws1[idx_p1], np.log(0.5) + logws2[idx_p1]
        )
        assert result_sum.weights[0][idx_ens] == pytest.approx(expected_val, abs=1e-6)

    # Min: For a=0.5, should be close to actual minimum
    ensemble_min = Ensemble(p1, p2, op="min", a=0.5)
    result_min = await ensemble_min.batch_logw_next([[]])

    for tok in mock_vocab:
        idx_ens = ensemble_min.lookup[tok]
        idx_p1 = p1.lookup[tok]
        expected_val = np.minimum(logws1[idx_p1], logws2[idx_p1])
        assert result_min.weights[0][idx_ens] == pytest.approx(expected_val, abs=0.5)

    # Max: For a=0.5, should be close to actual maximum
    ensemble_max = Ensemble(p1, p2, op="max", a=0.5)
    result_max = await ensemble_max.batch_logw_next([[]])

    for tok in mock_vocab:
        idx_ens = ensemble_max.lookup[tok]
        idx_p1 = p1.lookup[tok]
        expected_val = np.maximum(logws1[idx_p1], logws2[idx_p1])
        assert result_max.weights[0][idx_ens] == pytest.approx(expected_val, abs=0.5)


@pytest.mark.asyncio
async def test_ensemble_all_power_means():
    """Test power means across negative and positive exponents."""
    mock_vocab = ["a", "b"]
    logws1 = np.log([0.7, 0.3, 0.001])  # Model 1 prefers 'a'
    logws2 = np.log([0.3, 0.7, 0.001])  # Model 2 prefers 'b'

    p1 = MockPotential(vocab=mock_vocab, next_token_logws=logws1)
    p2 = MockPotential(vocab=mock_vocab, next_token_logws=logws2)

    power_means = [-5, -2.5, -2, -1.5, -0.5, -0.25, 0.25, 0.5, 1.5, 2, 2.5, 3, 5]
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

    logws_50 = result_50.weights[0]
    logws_80 = result_80.weights[0]
    logws_20 = result_20.weights[0]
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
    combined_balanced = result_balanced.weights[0]
    combined_favor_p1 = result_favor_p1.weights[0]

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
async def test_ensemble_raises_on_different_vocabularies():
    """Test Ensemble raises when using potentials with different vocabularies."""
    vocab1 = ["a", "b", "c", "d"]
    vocab2 = ["a", "b", "x", "y"]
    logws1 = np.log([0.25, 0.25, 0.25, 0.25, 0.001])
    logws2 = np.log([0.25, 0.25, 0.25, 0.25, 0.001])
    p1 = MockPotential(vocab=vocab1, next_token_logws=logws1)
    p2 = MockPotential(vocab=vocab2, next_token_logws=logws2)
    with pytest.raises(ValueError, match="same vocabulary"):
        Ensemble(p1, p2, op="prod", a=0.5)


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
    combined = result.weights[0]

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
    llm1 = PromptedLLM.from_name("openai-community/gpt2")
    llm2 = PromptedLLM.from_name("openai-community/gpt2")
    llm1.set_prompt_from_str("Write a SQL query: ")
    llm2.set_prompt_from_str("SQL code: ")

    ensemble = Ensemble(llm1, llm2, op="prod", a=0.5)
    assert ensemble.p1 is llm1
    assert ensemble.p2 is llm2

    ensemble_result = await ensemble.batch_logw_next([[]])
    ensemble_logws = ensemble_result.weights[0]

    assert len(ensemble_logws) > 0
    assert np.all(np.isfinite(ensemble_logws))
    assert len(ensemble_logws) == len(
        ensemble.vocab_eos
    )  # Should have same length as vocab


@pytest.mark.asyncio
async def test_token_ensemble_complementary_prompts():
    """Test token-level Ensemble combining complementary prompting strategies."""
    llm1 = PromptedLLM.from_name("openai-community/gpt2")
    llm2 = PromptedLLM.from_name("openai-community/gpt2")

    llm1.set_prompt_from_str("Task: Generate structured SQL.\n")
    llm2.set_prompt_from_str("Task: Generate correct SQL.\n")

    ensemble = Ensemble(llm1, llm2, op="prod", a=0.5)
    p1_result = await llm1.batch_logw_next([[]])
    p2_result = await llm2.batch_logw_next([[]])
    ensemble_result = await ensemble.batch_logw_next([[]])

    p1_logws = p1_result.weights[0]
    p2_logws = p2_result.weights[0]
    ensemble_logws = ensemble_result.weights[0]

    p1_logws = np.asarray(p1_logws)[ensemble.p1_vocab_idxs]
    p2_logws = np.asarray(p2_logws)[ensemble.p2_vocab_idxs]
    assert not np.allclose(p1_logws, p2_logws)
    np.testing.assert_allclose(
        ensemble_logws, 0.5 * p1_logws + 0.5 * p2_logws, rtol=1e-5
    )
    assert np.all(np.isfinite(ensemble_logws))


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
    combined = result.weights[0]

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
    combined = result.weights[0]

    for tok in vocab1:
        idx = ensemble.lookup[tok]
        prob = np.exp(combined[idx])
        assert prob > 0.05, f"{tok} should have reasonable probability in ensemble"
        assert prob < 0.95, f"{tok} shouldn't dominate in balanced ensemble"

    p1_result = await p1.batch_logw_next([[]])
    p2_result = await p2.batch_logw_next([[]])

    assert not np.allclose(combined, p1_result.weights[0], rtol=0.1)
    assert not np.allclose(combined, p2_result.weights[0], rtol=0.1)


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
    combined = result.weights[0]

    correct_idx = ensemble.lookup["correct"]
    wrong1_idx = ensemble.lookup["wrong1"]
    # ensemble should favor 'correct' more than model 1 alone
    p1_result = await p1.batch_logw_next([[]])
    p1_logws = p1_result.weights[0]
    p1_correct = p1_logws[p1.lookup["correct"]]
    p1_wrong1 = p1_logws[p1.lookup["wrong1"]]
    ensemble_correct = combined[correct_idx]
    ensemble_wrong1 = combined[wrong1_idx]

    # Ensemble should have stronger preference for 'correct' than uncertain model 1
    p1_gap = p1_correct - p1_wrong1
    ensemble_gap = ensemble_correct - ensemble_wrong1
    assert ensemble_gap > p1_gap, (
        "Ensemble should be more confident than uncertain model"
    )

    # But less confident than model 2 alone
    p2_result = await p2.batch_logw_next([[]])
    p2_logws = p2_result.weights[0]
    p2_gap = p2_logws[p2.lookup["correct"]] - p2_logws[p2.lookup["wrong1"]]
    assert ensemble_gap < p2_gap, (
        "Ensemble should be less confident than very confident model"
    )


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
    with pytest.raises(ValueError, match="Invalid operation"):
        convert_to_weighted_logop(True, a=0.5)


@pytest.mark.parametrize("name, p", [("sum", 1), ("prod", 0), ("harmonic", -1)])
@pytest.mark.parametrize("a", [0.3, 0.5])
def test_named_ops_are_power_means(name, p, a):
    """Test sum, prod and harmonic are the power means with p = 1, 0 and -1."""
    x = np.log([0.2, 0.5, 0.3])
    y = np.log([0.6, 0.1, 0.3])
    np.testing.assert_allclose(
        convert_to_weighted_logop(name, a)(x, y),
        convert_to_weighted_logop(p, a)(x, y),
        rtol=1e-12,
    )


def test_power_mean_matches_definition():
    """Test the log-space power mean against (a x^p + (1-a) y^p)^(1/p)."""
    x, y = np.array([0.2, 0.5]), np.array([0.6, 0.1])
    for p, a in [(2.5, 0.3), (-1.5, 0.7), (0.5, 0.5)]:
        expected = (a * x**p + (1 - a) * y**p) ** (1 / p)
        result = convert_to_weighted_logop(p, a)(np.log(x), np.log(y))
        np.testing.assert_allclose(np.exp(result), expected, rtol=1e-12)


def test_convert_to_weighted_logop_operations():
    """Test convert_to_weighted_logop returns correct operation for prod."""
    x = np.log(np.array([0.3, 0.7]))
    y = np.log(np.array([0.6, 0.4]))

    # Test prod with analytical verification
    op_prod = convert_to_weighted_logop("prod", a=0.5)
    result_prod = op_prod(x, y)
    expected_prod = 0.5 * x + 0.5 * y
    np.testing.assert_allclose(result_prod, expected_prod, rtol=1e-5)


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


@pytest.mark.parametrize("op", ["sum", "prod", "harmonic", "min", "max", -0.5, 2])
@pytest.mark.parametrize("a", [0.3, 0.5, 0.7])
def test_ops_zero_weights_give_neginf_not_nan(op, a):
    """Test ops combine -inf (zero weight) entries without producing nan."""
    fn = convert_to_weighted_logop(op, a)
    x = np.array([-1.0, -np.inf, -np.inf, -1.0])
    y = np.array([-2.0, -np.inf, -1.0, -np.inf])
    result = fn(x, y)
    assert not np.any(np.isnan(result))
    assert result[1] == -np.inf
    assert fn(-np.inf, -np.inf) == -np.inf


def test_weighted_max_equal_weights_ignores_zero_term():
    """Test weighted max at a=0.5 is the plain max even with a -inf input."""
    fn = convert_to_weighted_logop("max", a=0.5)
    np.testing.assert_allclose(fn(np.array([-np.inf]), np.array([-1.0])), [-1.0])


async def _byte_ensemble(op, prompt1, prompt2, a=0.3):
    """Ensemble two GPT-2 byte-level potentials with different prompts."""
    llm = load_model_by_name("openai-community/gpt2", backend="hf")
    params = BeamParams(K=3, eos_byte_strings=[b"<|endoftext|>"])
    b1, b2 = ByteLLM(llm, params), ByteLLM(llm, params)
    b1.set_prompt_from_str(prompt1)
    b2.set_prompt_from_str(prompt2)
    return Ensemble(b1, b2, op=op, a=a)


async def _walk(ensemble, steps=8):
    """Sample a single path with the library's direct token sampler; return its weights."""
    sampler = direct_token_sampler(ensemble)
    context, weights = [], []
    for _ in range(steps):
        token, logw, _ = await sampler.sample(context)
        weights.append(logw)
        if token is EOS:
            break
        context.append(token)
    return np.array(weights)


@pytest.mark.asyncio
async def test_byte_ensemble_sum_preserves_mass():
    """A byte-level sum ensemble has every step weight log 1 = 0."""
    ensemble = await _byte_ensemble(
        "sum", "My favorite physicist is", "My favorite author is"
    )
    np.testing.assert_allclose(await _walk(ensemble), 0.0, atol=1e-8)


@pytest.mark.asyncio
async def test_byte_ensemble_prod_weights_bounded():
    """Product weights are <= 0 (Hoelder), and 0 when both models agree."""
    different = await _byte_ensemble(
        "prod", "My favorite physicist is", "My favorite author is"
    )
    assert np.all(await _walk(different) <= 1e-9)
    same = await _byte_ensemble(
        "prod", "My favorite author is", "My favorite author is"
    )
    np.testing.assert_allclose(await _walk(same), 0.0, atol=1e-8)


@pytest.mark.asyncio
async def test_byte_ensemble_smc_component_logws():
    """End-to-end byte SMC; every particle gets finite per-model weights."""
    ensemble = await _byte_ensemble("max", "The cat", "A dog")
    sequences = await direct_token_sampler(ensemble).smc(
        n_particles=3, ess_threshold=0.5, max_tokens=6
    )
    for context in sequences.contexts:
        assert context[-1] is EOS and len(context) <= 6
        w1, w2 = await ensemble.component_logws(context)
        assert np.isfinite(w1) and np.isfinite(w2)
    assert np.all(np.isfinite(sequences.log_weights))
