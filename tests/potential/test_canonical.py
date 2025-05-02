import pytest
import numpy as np
from genlm.control import PromptedLLM, CanonicalTokenization
from genlm.control.constant import EOS
from genlm.control.potential.built_in.canonical import FastCanonicalityFilterBPE


@pytest.fixture
def llm():
    return PromptedLLM.from_name("gpt2", temperature=0.7)


@pytest.fixture
def llm_with_multiple_eos():
    return PromptedLLM.from_name(
        "gpt2", temperature=0.7, eos_tokens=[b".", b" city", b"\n", b" "]
    )


@pytest.fixture
def canonical_potential(llm):
    """Create a CanonicalBPEPotential for testing"""
    return CanonicalTokenization(llm)


@pytest.mark.asyncio
async def test_init(llm, llm_with_multiple_eos):
    """Test that the potential initializes properly"""
    llm.set_prompt_from_str("Montreal is")
    llm_with_multiple_eos.set_prompt_from_str("Montreal is")
    potential = CanonicalTokenization(llm)
    potential_with_multiple_eos = CanonicalTokenization(llm_with_multiple_eos)

    # Check that the potential has the correct vocabulary
    assert len(potential.vocab) == len(potential.canonicality_filter._decode)
    assert len(potential_with_multiple_eos.vocab) == len(
        potential_with_multiple_eos.canonicality_filter._decode
    )
    # Check that EOS is added correctly
    assert len(potential.vocab_eos) == len(potential.vocab) + 1
    assert (
        len(potential_with_multiple_eos.vocab_eos)
        == len(potential_with_multiple_eos.vocab) + 1
    )


@pytest.mark.asyncio
async def test_empty_context_mask(llm):  # Use the llm fixture
    """
    Test FastCanonicalityFilterBPE.__call__ with an empty context tuple ().
    It should return a mask allowing all tokens initially.
    """

    filter_instance = FastCanonicalityFilterBPE.from_llm(llm)
    empty_context = ()

    mask = filter_instance(empty_context)

    assert isinstance(mask, np.ndarray), "Mask should be a numpy array"
    assert mask.dtype == bool, "Mask dtype should be boolean"
    assert len(mask) == filter_instance.V, (
        f"Mask length ({len(mask)}) should equal vocab size ({filter_instance.V})"
    )
    assert np.all(mask), "Mask should be all True for an empty context"


@pytest.mark.asyncio
async def test_complete_empty(canonical_potential):
    """Test complete method with empty context"""
    log_weight = await canonical_potential.complete([])
    assert log_weight == 0.0


@pytest.mark.asyncio
async def test_complete_canonical(canonical_potential):
    """Test complete method with canonical context"""
    tokens = [b"Token", b"ization"]
    log_weight = await canonical_potential.complete(tokens)
    assert log_weight == 0.0


@pytest.mark.asyncio
async def test_complete_non_canonical(canonical_potential):
    """Test complete method with non-canonical context"""
    tokens = [b"To", b"ken", b"ization"]
    log_weight = await canonical_potential.complete(tokens)
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_logw_next(canonical_potential):
    """Test logw_next method with non canonical context. should only extend to EOS"""
    tokens = [b"To", b"ken"]
    logw = await canonical_potential.logw_next(tokens)
    assert logw[b"ization"] == float("-inf")
    assert logw[EOS] == 0.0


@pytest.mark.asyncio
async def test_logw_next_canonical(canonical_potential):
    """Test logw_next allows canonical next tokens and disallows non-canonical ones."""
    context = [b"Token"]
    canonical_next_bytes = b"ization"
    non_canonical_next_bytes = b"tion"

    filter_encode_map = canonical_potential.canonicality_filter._encode
    if canonical_next_bytes not in filter_encode_map:
        pytest.skip(f"Test token {canonical_next_bytes!r} not in filter vocab.")
    if non_canonical_next_bytes not in filter_encode_map:
        pytest.skip(f"Test token {non_canonical_next_bytes!r} not in filter vocab.")
    if context[0] not in filter_encode_map:
        pytest.skip(f"Test token {context[0]!r} not in filter vocab.")

    logw = await canonical_potential.logw_next(context)

    # Assert canonical next token is allowed (weight is not -inf)
    assert logw[canonical_next_bytes] != float("-inf"), (
        f"Canonical next token {canonical_next_bytes!r} should be allowed"
    )

    # Assert non-canonical next token is disallowed (weight is -inf)
    assert logw[non_canonical_next_bytes] == float("-inf"), (
        f"Non-canonical next token {non_canonical_next_bytes!r} should be disallowed"
    )


@pytest.mark.asyncio
async def test_set_overrides_allows_specific_pairs(canonical_potential):
    """Test that set_overrides allows configured non-canonical pairs for gpt2."""
    _decode = canonical_potential.canonicality_filter._decode
    _encode = canonical_potential.canonicality_filter._encode

    required_ids = [198, 2637, 82]
    if any(idx >= len(_decode) or _decode[idx] is None for idx in required_ids):
        pytest.skip("Required token IDs for override test not present in vocabulary.")

    token_198_bytes = _decode[198]
    token_2637_bytes = _decode[2637]
    token_82_bytes = _decode[82]  # Corresponds to 's' for gpt2

    if (
        token_198_bytes not in _encode
        or token_2637_bytes not in _encode
        or token_82_bytes not in _encode
    ):
        pytest.skip(
            "Byte sequences for required tokens not found in potential's encode map."
        )

    # Test override (198, 198) -> \n\n
    logw_198 = await canonical_potential.logw_next([token_198_bytes])
    assert logw_198[token_198_bytes] != float("-inf"), (
        "Override (198, 198) failed in logw_next"
    )
    assert (
        await canonical_potential.complete([token_198_bytes, token_198_bytes]) == 0.0
    ), "Override (198, 198) failed in complete"

    if token_2637_bytes not in _encode:
        pytest.skip(
            f"Token ID 2637 ({token_2637_bytes!r}) not in potential encode map."
        )

    logw_2637 = await canonical_potential.logw_next([token_2637_bytes])
    assert logw_2637[token_82_bytes] != float("-inf"), (
        "Override (2637, 82) failed in logw_next"
    )
    assert (
        await canonical_potential.complete([token_2637_bytes, token_82_bytes]) == 0.0
    ), "Override (2637, 82) failed in complete"


@pytest.mark.asyncio
async def test_check_canonicality(canonical_potential):
    """Test check_canonicality method with canonical context"""
    assert canonical_potential._check_canonicality([])
    # Single token is always canonical
    assert canonical_potential._check_canonicality([b" the"])
    # Valid token sequence should be canonical
    assert canonical_potential._check_canonicality([b"Token", b"ization"])
    # This should be non-canonical
    assert not canonical_potential._check_canonicality([b"hel", b"lo", b" world"])


@pytest.mark.asyncio
async def test_example(canonical_potential):
    """Test example method with canonical context"""
    sentences = [
        "Natural language processing",
        "The quick brown fox jumps over the lazy dog",
        "Artificial intelligence and machine learning",
    ]
    for sentence in sentences:
        tokens = canonical_potential.tokenizer.encode(
            sentence, add_special_tokens=False
        )
        token_bytes = [
            canonical_potential.tokenizer.decode([token]).encode("utf-8")
            for token in tokens
        ]

        # This should be canonical
        log_weight = await canonical_potential.complete(token_bytes)
        assert log_weight == 0.0

        # Also test prefix for each subsequence
        for i in range(1, len(token_bytes) + 1):
            prefix = token_bytes[:i]
            log_weight = await canonical_potential.prefix(prefix)
            assert log_weight == 0.0

        # Test that each valid prefix allows appropriate next tokens
        for i in range(len(token_bytes)):
            prefix = token_bytes[:i]
            next_token = (
                token_bytes[i] if i < len(token_bytes) else canonical_potential.eos
            )
            print(prefix)
            print(next_token)
            lazy_weights = await canonical_potential.logw_next(prefix)

            # The next token in the sequence should be allowed
            token_idx = lazy_weights.encode.get(next_token)
            if token_idx is not None:
                assert not np.isneginf(lazy_weights.weights[token_idx])


if __name__ == "__main__":
    pytest.main()
