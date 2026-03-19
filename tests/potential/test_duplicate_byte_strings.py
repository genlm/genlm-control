"""Regression tests for models with duplicate byte strings in their vocabularies.

These models previously failed due to the bijection assumption between tokens
and byte strings. The old code used ``encode = {x: i for i, x in enumerate(decode)}``
where decode was ``list[bytes]``. When multiple token_ids mapped to the same byte
string, later entries silently overwrote earlier ones, making those tokens
inaccessible.

The refactored code uses ``Token`` objects keyed by ``token_id``, so each token
is a distinct dict key regardless of byte string collisions.

Models tested:
- google/gemma-3-1b-pt (110 duplicate byte strings)
- codellama/CodeLlama-7b-hf (96 duplicate byte strings)
- EleutherAI/llemma_7b (96 duplicate byte strings)
"""

import pytest
import torch
import numpy as np
from collections import Counter

from genlm.backend.llm import MockAsyncLM
from genlm.backend.tokenization import Token
from genlm.control import PromptedLLM, BoolFSA, AWRS

MODELS_WITH_DUPLICATES = [
    "google/gemma-3-1b-pt",
    "codellama/CodeLlama-7b-hf",
    "EleutherAI/llemma_7b",
]


@pytest.fixture(scope="module", params=MODELS_WITH_DUPLICATES)
def model_name(request):
    return request.param


@pytest.fixture(scope="module")
def mock_llm(model_name):
    try:
        return MockAsyncLM.from_name(model_name)
    except OSError:
        pytest.skip(f"Model {model_name} not available")


@pytest.fixture(scope="module")
def llm(mock_llm):
    return PromptedLLM(mock_llm)


# ---------------------------------------------------------------------------
# Precondition
# ---------------------------------------------------------------------------


def test_model_has_duplicate_byte_strings(mock_llm):
    """Verify these models actually have duplicate byte strings in their vocab."""
    byte_strings = [t.byte_string for t in mock_llm.byte_vocab]
    counts = Counter(byte_strings)
    duplicates = {bs: cnt for bs, cnt in counts.items() if cnt > 1}
    assert len(duplicates) > 0, "Expected duplicate byte strings in vocabulary"


# ---------------------------------------------------------------------------
# Core bijection-removal tests: these verify the semantic fix.
# The old code lost tokens because `{bytes: id}` overwrote duplicates.
# The new code uses Token objects (keyed by token_id) so nothing is lost.
# ---------------------------------------------------------------------------


def test_no_tokens_lost_in_vocab(llm, mock_llm):
    """Every non-EOS token_id must appear in the potential's vocab.

    The old bijection code built ``encode = {bytes: id}`` which silently dropped
    earlier entries when duplicate byte strings appeared. This test checks that
    the vocab size equals decode size minus the number of EOS tokens.
    """
    expected_size = len(mock_llm.byte_vocab) - len(llm.token_maps.eos_idxs)
    assert len(llm.vocab) == expected_size, (
        f"Vocab has {len(llm.vocab)} tokens but expected {expected_size}. "
        f"Tokens were lost (the old bijection bug)."
    )


def test_all_duplicate_tokens_in_lookup(llm):
    """Both tokens sharing a byte string must be independently addressable in lookup.

    The old ``encode`` dict mapped ``bytes -> int``, so for duplicate byte strings
    only the last token_id survived. The new ``lookup`` maps ``Token -> int``,
    so each token_id is a distinct key.
    """
    byte_strings = [t.byte_string for t in llm.vocab]
    counts = Counter(byte_strings)
    for bs, cnt in counts.items():
        if cnt <= 1:
            continue
        dup_tokens = [t for t in llm.vocab if t.byte_string == bs]
        for t in dup_tokens:
            assert (
                t in llm.lookup
            ), f"Token({t.token_id}, {t.byte_string!r}) missing from lookup"
        # Verify they map to DIFFERENT indices
        indices = [llm.lookup[t] for t in dup_tokens]
        assert len(set(indices)) == len(
            indices
        ), f"Duplicate tokens for {bs!r} mapped to the same index: {indices}"


@pytest.mark.asyncio
async def test_duplicate_tokens_get_independent_weights(llm):
    """Tokens sharing a byte string must receive independent log-probabilities.

    If the old code lost a token, its weight would be inaccessible (returned as
    -inf). Here we verify both duplicate tokens have finite, distinct weights.
    """
    llm.set_prompt_from_str("Hello")

    byte_strings = [t.byte_string for t in llm.vocab]
    counts = Counter(byte_strings)
    dup_bs = next(bs for bs, cnt in counts.items() if cnt > 1)
    dup_tokens = [t for t in llm.vocab if t.byte_string == dup_bs]
    assert len(dup_tokens) >= 2

    lw = await llm.logw_next([])

    weights = [lw[t] for t in dup_tokens]
    for t, w in zip(dup_tokens, weights):
        assert np.isfinite(w), (
            f"Token({t.token_id}, {t.byte_string!r}) has weight {w} — "
            f"likely lost in encode dict"
        )

    assert weights[0] != weights[1], (
        f"Duplicate tokens {dup_tokens[0]} and {dup_tokens[1]} "
        f"should have different weights but both got {weights[0]}"
    )


@pytest.mark.asyncio
async def test_non_eos_indices_match_vocab(llm):
    """_non_eos_indices must have exactly len(vocab) entries, one per vocab token."""
    llm.set_prompt_from_str("test")
    await llm.logw_next([])

    assert len(llm._non_eos_indices) == len(llm.vocab)
    assert len(llm._non_eos_indices) == len(llm.token_maps.decode) - len(
        llm.token_maps.eos_idxs
    )


# ---------------------------------------------------------------------------
# Tokenize / encode / decode roundtrip
# ---------------------------------------------------------------------------


def test_tokenize_roundtrip(llm):
    """tokenize → encode_tokens should produce the same ids as the raw tokenizer."""
    text = "Hello, world!"
    tokens = llm.tokenize(text)
    ids = llm.encode_tokens(tokens)
    expected_ids = llm.model.tokenizer.encode(text)
    assert ids == expected_ids


# ---------------------------------------------------------------------------
# Direct use of duplicate tokens in contexts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prefix_complete_with_duplicate_token_in_context(llm):
    """Build a context that includes a duplicate token and run prefix/complete on it.

    For codellama, b' ' exists at token_ids [35, 29871]. We explicitly build
    contexts using each and verify both produce finite results.
    """
    llm.set_prompt_from_str("Hello")

    byte_strings = [t.byte_string for t in llm.vocab]
    counts = Counter(byte_strings)
    dup_bs = next(bs for bs, cnt in counts.items() if cnt > 1)
    dup_tokens = [t for t in llm.vocab if t.byte_string == dup_bs]

    for token in dup_tokens:
        context = [token]
        p = await llm.prefix(context)
        c = await llm.complete(context)
        assert np.isfinite(
            p
        ), f"prefix([Token({token.token_id}, {token.byte_string!r})]) = {p}"
        assert np.isfinite(
            c
        ), f"complete([Token({token.token_id}, {token.byte_string!r})]) = {c}"

    # The two duplicate tokens should give different scores
    # (because they have different token_ids → different positions in the LLM output)
    p0 = await llm.prefix([dup_tokens[0]])
    p1 = await llm.prefix([dup_tokens[1]])
    assert p0 != p1, (
        f"prefix should differ for Token({dup_tokens[0].token_id}) vs "
        f"Token({dup_tokens[1].token_id}), both with byte_string={dup_bs!r}"
    )


@pytest.mark.asyncio
async def test_coerced_logw_next_has_duplicate_tokens(llm):
    """logw_next on a coerced FSA should contain entries for duplicate tokens.

    Both tokens sharing a byte string must appear as independent keys in the
    coerced vocabulary and receive weights from logw_next. This verifies
    duplicate tokens are not dropped during coercion.
    """
    llm.set_prompt_from_str("The answer is")
    fsa = BoolFSA.from_regex(r" (yes|no)")
    coerced = fsa.coerce(llm, f=b"".join)

    # Verify the coerced vocab itself contains duplicate byte_strings
    coerced_byte_strings = [t.byte_string for t in coerced.vocab]
    counts = Counter(coerced_byte_strings)
    dup_pairs = {bs: cnt for bs, cnt in counts.items() if cnt > 1}
    assert len(dup_pairs) > 0, "Expected duplicate byte_strings in coerced vocab"

    # Both duplicate tokens must be addressable in the coerced lookup
    for dup_bs in dup_pairs:
        tokens = [t for t in coerced.vocab if t.byte_string == dup_bs]
        for t in tokens:
            assert (
                t in coerced.lookup
            ), f"Token({t.token_id}, {t.byte_string!r}) missing from coerced lookup"
        indices = [coerced.lookup[t] for t in tokens]
        assert len(set(indices)) == len(
            indices
        ), f"Duplicate tokens for {dup_bs!r} share an index: {indices}"


# ---------------------------------------------------------------------------
# Coerce path
# ---------------------------------------------------------------------------


def test_coerce_fsa(llm):
    """Coercing a BoolFSA onto an LLM with duplicate byte strings should succeed."""
    fsa = BoolFSA.from_regex(r" (yes|no)")
    coerced = fsa.coerce(llm, f=b"".join)
    assert len(coerced.vocab) > 0


@pytest.mark.asyncio
async def test_coerced_logw_next(llm):
    """logw_next on a coerced FSA should return valid weights."""
    llm.set_prompt_from_str("The answer is")
    fsa = BoolFSA.from_regex(r" (yes|no)")
    coerced = fsa.coerce(llm, f=b"".join)
    lw = await coerced.logw_next([])
    assert len(lw) > 0
    assert not np.all(np.isinf(lw.weights))


# ---------------------------------------------------------------------------
# Full SMC pipeline end-to-end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smc_with_duplicate_vocab(llm):
    """Full SMC sampling should work end-to-end with these models."""
    llm.set_prompt_from_str("The answer is")
    fsa = BoolFSA.from_regex(r" (yes|no)")
    coerced = fsa.coerce(llm, f=b"".join)
    sampler = AWRS(llm, coerced)

    result = await sampler.smc(n_particles=3, ess_threshold=0.5, max_tokens=10)
    assert len(result.contexts) == 3

    posterior = result.decoded_posterior
    assert len(posterior) > 0
    for seq in posterior.keys():
        assert seq in (" yes", " no"), f"Unexpected sequence: {seq!r}"


# ---------------------------------------------------------------------------
# Consistency checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_logw_next_consistency(llm):
    """logw_next weights should be consistent with prefix/complete."""
    llm.set_prompt_from_str("Once upon")
    context = llm.tokenize(" a")
    await llm.assert_logw_next_consistency(context, top=10, rtol=1e-3, atol=1e-3)


@pytest.mark.asyncio
async def test_autoreg_fact(llm):
    """Autoregressive factorization: complete(x) == sum of prefix log-probs + eos."""
    llm.set_prompt_from_str("Once upon")
    context = llm.tokenize(" a time")
    await llm.assert_autoreg_fact(context, rtol=1e-3, atol=1e-3)


@pytest.mark.asyncio
async def test_batch_consistency(llm):
    """Batch logw_next should match individual logw_next calls."""
    llm.set_prompt_from_str("Once upon")
    contexts = [llm.tokenize(" a"), llm.tokenize(" a time")]
    await llm.assert_batch_consistency(contexts, rtol=1e-3, atol=1e-3)


# ---------------------------------------------------------------------------
# encode_tokens disambiguation: the core of the bijection fix
# ---------------------------------------------------------------------------


def test_encode_tokens_distinguishes_duplicate_tokens(llm):
    """Passing Token objects with the same byte_string but different token_ids
    must produce different IDs. This is the core behavior the refactor enables."""
    byte_strings = [t.byte_string for t in llm.vocab]
    counts = Counter(byte_strings)
    dup_bs = next(bs for bs, cnt in counts.items() if cnt > 1)
    dup_tokens = [t for t in llm.vocab if t.byte_string == dup_bs]

    id_a = llm.encode_tokens([dup_tokens[0]])[0]
    id_b = llm.encode_tokens([dup_tokens[1]])[0]

    assert id_a == dup_tokens[0].token_id
    assert id_b == dup_tokens[1].token_id
    assert id_a != id_b


def test_encode_tokens_bytes_fallback_returns_first_match(llm):
    """The deprecated bytes path returns the first matching token_id.
    This is ambiguous for duplicates — users should use Token objects instead."""
    import warnings

    byte_strings = [t.byte_string for t in llm.vocab]
    counts = Counter(byte_strings)
    dup_bs = next(bs for bs, cnt in counts.items() if cnt > 1)
    dup_tokens = [t for t in llm.vocab if t.byte_string == dup_bs]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = llm.encode_tokens([dup_bs])

    # Should return the first token_id for this byte_string
    assert result[0] == dup_tokens[0].token_id


# ---------------------------------------------------------------------------
# EOS with duplicate byte strings
# ---------------------------------------------------------------------------


def test_eos_duplicate_keeps_non_eos_in_vocab(mock_llm):
    """If the EOS byte_string also appears as a non-EOS token (different token_id),
    only the designated EOS token should be excluded from potential_vocab."""
    from genlm.control.potential.built_in.llm import TokenMappings

    decode = mock_llm.byte_vocab
    eos_byte = decode[mock_llm.tokenizer.eos_token_id].byte_string

    # Check if the EOS byte_string has duplicates in this model
    eos_dupes = [t for t in decode if t.byte_string == eos_byte]
    if len(eos_dupes) < 2:
        pytest.skip("This model's EOS byte_string has no duplicate")

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        tm = TokenMappings.create(decode=decode, eos_tokens=[eos_byte])

    # The non-EOS duplicate must still be in potential_vocab
    non_eos_dupes = [t for t in eos_dupes if t.token_id not in set(tm.eos_idxs)]
    for t in non_eos_dupes:
        assert t in [
            v for v in tm.potential_vocab if v.token_id == t.token_id
        ], f"Token({t.token_id}, {t.byte_string!r}) wrongly excluded from potential_vocab"


def test_spawn_new_eos_with_duplicate_byte_string(llm):
    """spawn_new_eos with a byte_string that has duplicates in the vocab
    should work and only exclude the designated EOS token."""
    byte_strings = [t.byte_string for t in llm.vocab]
    counts = Counter(byte_strings)
    dup_bs = next(bs for bs, cnt in counts.items() if cnt > 1)

    new_llm = llm.spawn_new_eos(eos_tokens=[dup_bs])

    # Only ONE token should be excluded (the first match)
    assert len(new_llm.token_maps.eos_idxs) == 1

    # The other duplicate should still be in vocab
    dup_tokens = [t for t in llm.vocab if t.byte_string == dup_bs]
    excluded_id = new_llm.token_maps.eos_idxs[0]
    remaining = [t for t in dup_tokens if t.token_id != excluded_id]
    for t in remaining:
        assert any(
            v.token_id == t.token_id for v in new_llm.vocab
        ), f"Token({t.token_id}) wrongly excluded from vocab"


# ---------------------------------------------------------------------------
# Fewer logits than vocab entries: real models may output fewer logits than
# len(tokenizer) when the tokenizer has added tokens beyond the model's
# embedding matrix (e.g. Gemma: len(tokenizer)=262145, vocab_size=262144).
# ---------------------------------------------------------------------------


class TruncatedMockAsyncLM(MockAsyncLM):
    """Mock that returns fewer logits than len(byte_vocab), like a real HF model
    whose config.vocab_size < len(tokenizer)."""

    def __init__(self, tokenizer, truncate_by=1):
        super().__init__(tokenizer)
        self._truncate_by = truncate_by

    def _get_logprobs(self, token_ids):
        seed = sum([(i + 1) * t for i, t in enumerate(token_ids)])
        self._rng.seed(seed)
        n_logits = len(self.byte_vocab) - self._truncate_by
        logits = torch.from_numpy(
            self._rng.rand(n_logits).astype(np.float32)
        )
        return torch.log_softmax(logits, dim=-1)


@pytest.fixture(scope="module")
def truncated_mock_llm(model_name):
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except OSError:
        pytest.skip(f"Model {model_name} not available")
    return TruncatedMockAsyncLM(tokenizer, truncate_by=1)


@pytest.fixture(scope="module")
def truncated_llm(truncated_mock_llm):
    return PromptedLLM(truncated_mock_llm)


def test_byte_vocab_includes_all_tokens(mock_llm):
    """byte_vocab should include ALL tokens from the tokenizer, including added
    tokens beyond the model's embedding matrix. These tokens are part of the
    vocabulary even if the model can't produce logits for them."""
    assert len(mock_llm.byte_vocab) == len(mock_llm.tokenizer)


@pytest.mark.asyncio
async def test_logw_next_with_fewer_logits(truncated_llm):
    """logw_next must not crash when the model returns fewer logits than
    len(token_maps.decode). Tokens without logits should get -inf.

    This reproduces the bug where real HF models (e.g. Gemma) have
    config.vocab_size < len(tokenizer).
    """
    truncated_llm.set_prompt_from_str("Hello")
    lw = await truncated_llm.logw_next([])
    assert len(lw) > 0
    assert np.any(np.isfinite(lw.weights))


@pytest.mark.asyncio
async def test_smc_with_fewer_logits(truncated_llm):
    """Full SMC should work even when model returns fewer logits."""
    truncated_llm.set_prompt_from_str("The answer is")
    fsa = BoolFSA.from_regex(r" (yes|no)")
    coerced = fsa.coerce(truncated_llm, f=b"".join)
    sampler = AWRS(truncated_llm, coerced)

    result = await sampler.smc(n_particles=3, ess_threshold=0.5, max_tokens=10)
    assert len(result.contexts) == 3


def test_token_id_index_invariant(mock_llm):
    """Token IDs must equal their position index in byte_vocab.

    This invariant is assumed by the logit padding in _process_logw_next:
    when the model returns fewer logits than len(byte_vocab), we pad with -inf
    at the end, which is only correct if the extra tokens are at the highest
    indices.
    """
    for i, token in enumerate(mock_llm.byte_vocab):
        assert token.token_id == i, (
            f"byte_vocab[{i}].token_id={token.token_id}, expected {i}"
        )
