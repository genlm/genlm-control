import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from transformers import GPT2Tokenizer, BertTokenizer
from genlm.backend.tokenization import decode_vocab
from genlm.control import PromptedLLM, CanonicalTokenization
from genlm.control.potential.built_in.canonical import (
    FastCanonicalityFilterBPE,
    _extract_bpe_merges,
)
from hypothesis import given, strategies as st, settings


class MockAsyncTransformer:  # Mock the backend LLM object
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Restore calculation of byte_vocab; PromptedLLM init needs it.
        # decode_vocab will raise ValueError for unsupported tokenizers (like BERT).
        try:
            self.byte_vocab, _ = decode_vocab(tokenizer)
        except ValueError:
            self.byte_vocab = None  # Handle cases like BERT where byte vocab fails
        # maybe add other attributes if PromptedLLM.__init__ needs them
        # e.g., self.model_name_or_path = tokenizer.name_or_path


class MockLLM(PromptedLLM):
    def __init__(self, tokenizer, model_name="mock_model", eos_tokens=None):
        # Create the mock backend object
        mock_backend_llm = MockAsyncTransformer(tokenizer)

        # Call the parent PromptedLLM initializer
        # Use provided eos_tokens if available, otherwise extract from tokenizer
        if eos_tokens is None:
            eos_token_bytes = (
                tokenizer.eos_token.encode("utf-8") if tokenizer.eos_token else None
            )
            eos_token_list = [eos_token_bytes] if eos_token_bytes else []
        else:
            # Assume provided eos_tokens are already bytes or handle conversion if needed
            eos_token_list = eos_tokens

        # Need to handle cases where byte_vocab is None for unsupported tokenizers
        if mock_backend_llm.byte_vocab is None:
            # Provide some dummy value or handle appropriately based on PromptedLLM needs
            # For now, let's skip super init if vocab fails, as Canonical won't work anyway
            print(
                f"Warning: Skipping PromptedLLM super().__init__ for {tokenizer.name_or_path} due to missing byte_vocab."
            )
            self.model = (
                mock_backend_llm  # Still need self.model for tests accessing tokenizer
            )
            self.token_maps = None  # Indicate maps aren't properly initialized
        else:
            super().__init__(llm=mock_backend_llm, eos_tokens=eos_token_list)
            # The super init should handle setting up self.model and self.token_maps


@pytest.fixture(scope="module")
def llm():
    return PromptedLLM.from_name("gpt2", temperature=0.7, backend="hf")


@pytest.fixture(scope="module")
def llm_with_multiple_eos(llm):
    return llm.spawn_new_eos(eos_tokens=[b".", b" city", b"\n", b" "])


@pytest.fixture(scope="module")
def canonical_potential(llm):
    """Create a CanonicalTokenization for testing"""
    return CanonicalTokenization.from_llm(llm)


def test_init(llm, llm_with_multiple_eos):
    """Test that the potential initializes properly via from_llm"""
    # Instantiate using the new factory method
    potential = CanonicalTokenization.from_llm(llm)
    potential_with_multiple_eos = CanonicalTokenization.from_llm(llm_with_multiple_eos)

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


def test_no_eos_init(llm):
    canonicality_filter = FastCanonicalityFilterBPE.from_tokenizer(llm.model.tokenizer)
    assert canonicality_filter.eos_token_ids == {llm.model.tokenizer.eos_token_id}


def test_empty_context_mask(llm):  # Use the llm fixture
    """
    Test FastCanonicalityFilterBPE.__call__ with an empty context tuple ().
    It should return a mask allowing all tokens initially.
    """
    # Use the new factory method for the filter
    filter_instance = FastCanonicalityFilterBPE.from_tokenizer(
        llm.model.tokenizer, llm.token_maps.eos_idxs
    )
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
async def test_complete_non_canonical(canonical_potential):
    """Test complete method with non-canonical context"""
    tokens = [b"To", b"ken", b"ization"]
    log_weight = await canonical_potential.complete(tokens)
    assert log_weight == float("-inf")


@pytest.mark.asyncio
async def test_logw_next_invalid_prefix(canonical_potential):
    """Test logw_next method with non canonical context. should only extend to EOS"""
    tokens = [b"To", b"ken"]
    with pytest.raises(ValueError):
        await canonical_potential.logw_next(tokens)


@pytest.mark.asyncio
async def test_logw_next_canonical(canonical_potential):
    """Test logw_next allows canonical next tokens and disallows non-canonical ones."""
    context = [b"Token"]
    canonical_next_bytes = b"ization"
    non_canonical_next_bytes = b"tion"
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
async def test_set_overrides(canonical_potential):
    """Test that set_overrides allows configured non-canonical pairs for gpt2."""
    _decode = canonical_potential.canonicality_filter._decode

    required_ids = [198, 2637, 82]
    if any(idx >= len(_decode) or _decode[idx] is None for idx in required_ids):
        pytest.skip("Required token IDs for override test not present in vocabulary.")

    token_198_bytes = _decode[198]
    token_2637_bytes = _decode[2637]
    token_82_bytes = _decode[82]  # Corresponds to 's' for gpt2

    # Test override (198, 198) -> \n\n
    logw_198 = await canonical_potential.logw_next([token_198_bytes])
    assert logw_198[token_198_bytes] != float("-inf"), (
        "Override (198, 198) failed in logw_next"
    )
    assert (
        await canonical_potential.complete([token_198_bytes, token_198_bytes]) == 0.0
    ), "Override (198, 198) failed in complete"

    logw_2637 = await canonical_potential.logw_next([token_2637_bytes])
    assert logw_2637[token_82_bytes] != float("-inf"), (
        "Override (2637, 82) failed in logw_next"
    )
    assert (
        await canonical_potential.complete([token_2637_bytes, token_82_bytes]) == 0.0
    ), "Override (2637, 82) failed in complete"


def test_check_canonicality(canonical_potential):
    """Test check_canonicality method with canonical context"""
    assert canonical_potential._check_canonicality([])
    # Single token is always canonical
    assert canonical_potential._check_canonicality([b" the"])
    # Valid token sequence should be canonical
    assert canonical_potential._check_canonicality([b"Token", b"ization"])
    # This should be non-canonical
    assert not canonical_potential._check_canonicality([b"hel", b"lo", b" world"])


@pytest.mark.asyncio
@settings(deadline=None)
@given(st.text(min_size=1, max_size=10))
async def test_example(canonical_potential, llm, text):
    """Test example method with canonical context"""
    tokens = llm.tokenize(text)
    log_weight = await canonical_potential.complete(tokens)
    assert log_weight == 0.0
    # Also test prefix for each subsequence
    for i in range(1, len(tokens) + 1):
        prefix = tokens[:i]
        log_weight = await canonical_potential.prefix(prefix)
        assert log_weight == 0.0

    # Test that each valid prefix allows appropriate next tokens
    for i in range(len(tokens)):
        prefix = tokens[:i]
        next_token = tokens[i]
        lazy_weights = await canonical_potential.logw_next(prefix)
        # The next token in the sequence should be allowed
        assert lazy_weights[next_token] == 0.0


def test_from_llm_extract_merges_slow_tokenizer():
    """Test that merges are extracted correctly from a slow tokenizer (using bpe_ranks)."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)
    mock_llm = MockLLM(tokenizer)  # MockLLM needs to handle token_maps now
    if mock_llm.token_maps is None:  # Handle case where super init was skipped
        pytest.skip(
            "MockLLM failed to initialize token maps, likely unsupported tokenizer."
        )
    # Instantiate filter using the new method
    filter_instance = FastCanonicalityFilterBPE.from_tokenizer(
        mock_llm.model.tokenizer, mock_llm.token_maps.eos_idxs
    )
    assert filter_instance._merges, (
        "Merges should be extracted from the slow GPT2 tokenizer."
    )
    # Check a known merge (example: 'a' + 't' -> 'at')
    g_id = tokenizer.encode("a")[0]
    t_id = tokenizer.encode("t")[0]
    gt_id = tokenizer.encode("at")[0]
    assert (g_id, t_id, gt_id) in filter_instance._merges, (
        "Known merge (a, t) not found in extracted merges."
    )


def test_from_llm_extract_merges_fallback():
    """Test that creating the Filter/Potential fails for unsupported tokenizers."""
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )  # WordPiece tokenizer

    # MockLLM should handle the decode_vocab error gracefully now
    mock_llm = MockLLM(tokenizer)

    # Assert that MockLLM skipped its super init due to the unsupported tokenizer
    assert mock_llm.token_maps is None, (
        "MockLLM should have token_maps=None for unsupported tokenizer"
    )

    # Directly calling from_tokenizer should still raise the ValueError from decode_vocab
    with pytest.raises(ValueError, match="Could not decode byte representation"):
        FastCanonicalityFilterBPE.from_tokenizer(
            mock_llm.model.tokenizer, []
        )  # Pass empty eos_ids


def test_from_llm_duplicate_byte_error(llm):
    """Test that from_tokenizer raises ValueError if decode_vocab returns duplicates."""

    # Define the vocabulary with duplicates we want decode_vocab to return
    duplicate_vocab = [
        b"a",  # ID 0
        b"b",  # ID 1
        b"c",  # ID 2
        b"a",  # ID 3 - DUPLICATE of ID 0
    ]

    # Patch decode_vocab within the canonical module for this test
    with patch(
        "genlm.control.potential.built_in.canonical.decode_vocab",
        return_value=(duplicate_vocab, None),
    ):
        # Assert that from_tokenizer raises the expected ValueError when called
        with pytest.raises(ValueError, match="Duplicate byte sequences found"):
            FastCanonicalityFilterBPE.from_tokenizer(
                llm.model.tokenizer, llm.token_maps.eos_idxs
            )


def test_canonical_tokenization_init_type_error():
    """Test that CanonicalTokenization.from_llm raises TypeError for wrong llm type."""

    not_an_llm = object()
    with pytest.raises(
        TypeError, match="Expected llm to be an instance of PromptedLLM"
    ):
        # Call the factory method which performs the check
        CanonicalTokenization.from_llm(not_an_llm)


def test_call_unknown_last_token(llm):
    """Test FastCanonicalityFilterBPE.__call__ handles unknown last_token (KeyError)."""
    # Instantiate filter using the new method
    filter_instance = FastCanonicalityFilterBPE.from_tokenizer(
        llm.model.tokenizer, llm.token_maps.eos_idxs
    )

    unknown_token = b"@@@totally_unknown_token@@@"
    # Check it's really not in the encode map (optional sanity check)
    assert unknown_token not in filter_instance._encode
    context = (
        None,
        unknown_token,
    )

    with pytest.raises(KeyError):
        filter_instance(context)


def test_extract_merges_slow_id_mapping_failure():
    """Test warning when slow tokenizer has bpe_ranks but vocab mapping fails."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = False
    mock_tokenizer.name_or_path = "mock_slow_map_fail_tokenizer"

    # Make JSON and direct access fail (e.g., raise exceptions)
    mock_tokenizer._tokenizer = MagicMock()
    mock_tokenizer._tokenizer.to_str.side_effect = Exception("JSON parsing failed")
    type(mock_tokenizer._tokenizer).model = PropertyMock(
        side_effect=Exception("Direct access failed")
    )  # Ensure direct access also fails

    # Provide bpe_ranks directly on the mock
    mock_tokenizer.bpe_ranks = {("a", "b"): 0}

    # Make vocab lookup fail
    mock_tokenizer.get_vocab.return_value = {}

    # Patch hasattr to return True for bpe_ranks check
    with patch(
        "builtins.hasattr",
        lambda obj, name: True
        if obj is mock_tokenizer and name == "bpe_ranks"
        else hasattr(obj, name),
    ):
        # Catch ALL UserWarnings
        with pytest.raises(ValueError):
            _extract_bpe_merges(mock_tokenizer)


# @pytest.mark.asyncio
# def test_extract_merges_slow_exception():
#     """Test warning when accessing slow tokenizer bpe_ranks raises an exception."""
#     mock_tokenizer = MagicMock()
#     mock_tokenizer.is_fast = False
#     mock_tokenizer.name_or_path = "mock_slow_exception_tokenizer"

#     # Make JSON and direct access fail
#     mock_tokenizer._tokenizer = MagicMock()
#     mock_tokenizer._tokenizer.to_str.side_effect = Exception("JSON parsing failed")
#     type(mock_tokenizer._tokenizer).model = PropertyMock(side_effect=Exception("Direct access failed"))

#     # Make accessing bpe_ranks raise an error using PropertyMock
#     exception_message = "Cannot access bpe_ranks"
#     type(mock_tokenizer).bpe_ranks = PropertyMock(
#         side_effect=Exception(exception_message)
#     )
#     with pytest.raises(ValueError):
#         _extract_bpe_merges(mock_tokenizer)


if __name__ == "__main__":
    pytest.main()
