import pytest
import numpy as np
from unittest.mock import patch, MagicMock, PropertyMock
from transformers import GPT2Tokenizer, BertTokenizer
from genlm.backend.tokenization import decode_vocab
from genlm.control import PromptedLLM, CanonicalTokenization
from genlm.control.constant import EOS
from genlm.control.potential.built_in.canonical import (
    FastCanonicalityFilterBPE,
    _extract_bpe_merges,
)
import json


class MockAsyncTransformer:  # Mock the backend LLM object
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Restore calculation of byte_vocab; PromptedLLM init needs it.
        # decode_vocab will raise ValueError for unsupported tokenizers (like BERT).
        self.byte_vocab, _ = decode_vocab(tokenizer)
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

        super().__init__(llm=mock_backend_llm, eos_tokens=eos_token_list)
        # The super init should handle setting up self.model and self.token_maps


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
async def test_set_overrides(canonical_potential):
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
            lazy_weights = await canonical_potential.logw_next(prefix)

            # The next token in the sequence should be allowed
            token_idx = lazy_weights.encode.get(next_token)
            if token_idx is not None:
                assert not np.isneginf(lazy_weights.weights[token_idx])


@pytest.mark.asyncio
async def test_from_llm_extract_merges_slow_tokenizer():
    """Test that merges are extracted correctly from a slow tokenizer (using bpe_ranks)."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)
    mock_llm = MockLLM(tokenizer)
    filter_instance = FastCanonicalityFilterBPE.from_llm(mock_llm)
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


@pytest.mark.asyncio
async def test_from_llm_extract_merges_fallback():
    """Test that creating the LLM/Filter fails for unsupported tokenizers."""
    tokenizer = BertTokenizer.from_pretrained(
        "bert-base-uncased"
    )  # WordPiece tokenizer

    # Expect MockLLM init to fail because MockAsyncTransformer calls decode_vocab, which fails for BERT.
    with pytest.raises(ValueError, match="Could not decode byte representation"):
        MockLLM(tokenizer)


@pytest.mark.asyncio
async def test_from_llm_duplicate_byte_error(llm):
    """Test that from_llm raises ValueError if decode_vocab returns duplicates."""

    # Define the vocabulary with duplicates we want decode_vocab to return
    duplicate_vocab = [
        b"a",  # ID 0
        b"b",  # ID 1
        b"c",  # ID 2
        b"a",  # ID 3 - DUPLICATE of ID 0
    ]

    # Patch decode_vocab within the canonical module for this test
    # Make the mocked function return the duplicate vocab and None for the second value.
    with patch(
        "genlm.control.potential.built_in.canonical.decode_vocab",
        return_value=(duplicate_vocab, None),
    ) as mock_decode:
        # Assert that from_llm raises the expected ValueError when called
        # It will now call the mocked decode_vocab internally.
        with pytest.raises(ValueError, match="Duplicate byte sequences found"):
            FastCanonicalityFilterBPE.from_llm(llm)

        # Assert that the mock was called
        mock_decode.assert_called_once_with(llm.model.tokenizer)


@pytest.mark.asyncio
async def test_from_llm_empty_eos_warning():  # No fixture needed
    """Test that from_llm issues a warning if llm.token_maps.eos_idxs is empty."""

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False)
    mock_llm = MagicMock(spec=PromptedLLM)
    mock_llm.model = MagicMock()
    mock_llm.model.tokenizer = tokenizer
    mock_llm.token_maps = MagicMock()
    mock_llm.token_maps.eos_idxs = []

    # Expect the warning when calling from_llm with the fully mocked LLM
    with pytest.warns(UserWarning, match="llm.token_maps.eos_idxs is empty or None"):
        filter_instance = FastCanonicalityFilterBPE.from_llm(mock_llm)

    # Assert that the resulting filter has no EOS tokens registered
    assert not filter_instance.eos_token_ids, (
        "Filter should have an empty set of eos_token_ids"
    )


@pytest.mark.asyncio
async def test_canonical_tokenization_init_type_error():
    """Test that CanonicalTokenization.__init__ raises TypeError for wrong llm type."""

    not_an_llm = object()
    with pytest.raises(
        TypeError, match="Expected llm to be an instance of PromptedLLM"
    ):
        CanonicalTokenization(not_an_llm)


@pytest.mark.asyncio
def test_extract_merges_json_id_mapping_failure():
    """Test _extract_bpe_merges warning when JSON merges exist but vocab mapping fails."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = True
    mock_tokenizer._tokenizer = MagicMock()

    # Mock to_str to return valid JSON with merges
    mock_tokenizer._tokenizer.to_str.return_value = (
        '{"model": {"merges": [("a", "b")]}}'
    )

    # Mock get_vocab to return empty dict, causing ID lookup failure
    mock_tokenizer.get_vocab.return_value = {}

    with pytest.warns(UserWarning, match="ID mapping failed for ALL pairs"):
        merges = _extract_bpe_merges(mock_tokenizer)

    assert merges == [], "Merges should be empty if ID mapping failed for all pairs"


@pytest.mark.asyncio
def test_extract_merges_json_parsing_exception():
    """Test _extract_bpe_merges warning when JSON parsing raises an exception."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = True
    mock_tokenizer._tokenizer = MagicMock()

    # Mock to_str to raise an exception
    mock_tokenizer._tokenizer.to_str.side_effect = json.JSONDecodeError(
        "Mock decode error", "", 0
    )
    # can also do: mock_tokenizer._tokenizer.to_str.side_effect = Exception("Mock general error")

    # Mock get_vocab needed for later stages if the first try fails (shouldn't be reached here)
    mock_tokenizer.get_vocab.return_value = {"a": 1, "b": 2}

    with pytest.warns(
        UserWarning, match="Failed to extract merges using fast tokenizer JSON parsing"
    ):
        merges = _extract_bpe_merges(mock_tokenizer)

    assert merges == [], "Merges should be empty if JSON parsing failed"


@pytest.mark.asyncio
def test_extract_merges_direct_access_id_mapping_failure():
    """Test warning when direct access merges exist but vocab mapping fails."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = True
    mock_tokenizer._tokenizer = MagicMock()
    mock_tokenizer._tokenizer.model = MagicMock()

    # Make JSON parsing fail
    mock_tokenizer._tokenizer.to_str.side_effect = Exception("JSON Failed")

    # Provide direct merges
    mock_tokenizer._tokenizer.model.merges = [("a", "b")]

    # Make vocab lookup fail
    mock_tokenizer.get_vocab.return_value = {}

    with pytest.warns(
        UserWarning, match="Accessed direct merges, but ID mapping failed"
    ):
        merges = _extract_bpe_merges(mock_tokenizer)

    assert merges == [], "Merges should be empty if direct access ID mapping failed"


@pytest.mark.asyncio
def test_extract_merges_direct_access_exception():
    """Test warning when accessing direct merges raises an exception."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = True
    mock_tokenizer._tokenizer = MagicMock()
    mock_tokenizer._tokenizer.model = MagicMock()  # Ensure model attribute exists

    # Make JSON parsing succeed but find no merges (e.g., return empty JSON)
    mock_tokenizer._tokenizer.to_str.return_value = "{}"
    # Also provide a vocab so ID mapping wouldn't fail if merges WERE found
    mock_tokenizer.get_vocab.return_value = {"a": 1, "b": 2}

    # Make accessing the .merges attribute itself raise an Exception
    type(mock_tokenizer._tokenizer.model).merges = PropertyMock(
        side_effect=Exception("Cannot access merges")
    )

    with pytest.warns(
        UserWarning, match="Failed to extract merges using fast tokenizer direct access"
    ) as record:
        merges = _extract_bpe_merges(mock_tokenizer)

    assert merges == [], "Merges should be empty if direct access failed"

    assert "Cannot access merges" in str(record[0].message)


@pytest.mark.asyncio
def test_extract_merges_slow_id_mapping_failure():
    """Test warning when slow tokenizer has bpe_ranks but vocab mapping fails."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = False
    # Provide bpe_ranks directly on the mock
    mock_tokenizer.bpe_ranks = {("a", "b"): 0}

    # Make vocab lookup fail
    mock_tokenizer.get_vocab.return_value = {}

    with pytest.warns(UserWarning, match="ID mapping failed for ALL pairs"):
        # Patch hasattr check for bpe_ranks to return True for this mock
        with patch(
            "builtins.hasattr",
            lambda obj, name: name == "bpe_ranks"
            if obj is mock_tokenizer
            else hasattr(obj, name),
        ):
            merges = _extract_bpe_merges(mock_tokenizer)

    assert merges == [], "Merges should be empty if slow ID mapping failed"


@pytest.mark.asyncio
def test_extract_merges_slow_exception():
    """Test warning when accessing slow tokenizer bpe_ranks raises an exception."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = False

    # Make accessing bpe_ranks raise an error using PropertyMock
    type(mock_tokenizer).bpe_ranks = PropertyMock(
        side_effect=Exception("Cannot access bpe_ranks")
    )

    # Patch hasattr check for bpe_ranks to return True for this mock
    with patch(
        "builtins.hasattr",
        lambda obj, name: name == "bpe_ranks"
        if obj is mock_tokenizer
        else hasattr(obj, name),
    ):
        with pytest.warns(
            UserWarning, match="Failed to extract merges using tokenizer.bpe_ranks"
        ) as record:
            merges = _extract_bpe_merges(mock_tokenizer)

    assert merges == [], "Merges should be empty if slow access failed"
    assert "Cannot access bpe_ranks" in str(record[0].message)


@pytest.mark.asyncio
def test_extract_merges_final_fallback():
    """Test the final fallback warning when no merge extraction method works."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = False

    # Mock hasattr to return False for bpe_ranks check
    def mock_hasattr_false(obj, name):
        if obj is mock_tokenizer and name == "bpe_ranks":
            return False
        return hasattr(obj, name)

    mock_tokenizer.hasattr.side_effect = mock_hasattr_false

    mock_tokenizer.name_or_path = "mock_fallback_tokenizer"
    # Patch builtins.hasattr for the check within the function
    with patch("builtins.hasattr", mock_hasattr_false):
        with pytest.warns(
            UserWarning,
            match="Could not determine BPE merges for tokenizer mock_fallback_tokenizer",
        ):
            merges = _extract_bpe_merges(mock_tokenizer)

    assert merges == [], "Merges should be empty on final fallback"


@pytest.mark.asyncio
def test_call_unknown_last_token(llm):
    """Test FastCanonicalityFilterBPE.__call__ handles unknown last_token (KeyError)."""

    filter_instance = FastCanonicalityFilterBPE.from_llm(llm)

    unknown_token = b"@@@totally_unknown_token@@@"
    # Check it's really not in the encode map (optional sanity check)
    assert unknown_token not in filter_instance._encode
    context = (
        None,
        unknown_token,
    )

    with pytest.warns(UserWarning, match=f"Last token {unknown_token!r} not found"):
        mask = filter_instance(context)

    # 4. Verify the mask allows only EOS tokens
    expected_mask = np.zeros(filter_instance.V, dtype=bool)
    # Ensure EOS IDs are within vocab bounds before indexing
    valid_eos_ids = [
        eid for eid in filter_instance.eos_token_ids if eid < filter_instance.V
    ]
    if valid_eos_ids:
        expected_mask[valid_eos_ids] = True

    np.testing.assert_array_equal(
        mask, expected_mask, "Mask should only allow EOS for unknown context token"
    )


# @pytest.mark.asyncio
# def test_extract_merges_json_id_mapping_failure():
#     """Test _extract_bpe_merges warning when JSON merges exist but vocab mapping fails."""
#     mock_tokenizer = MagicMock()
#     mock_tokenizer.is_fast = True
#     mock_tokenizer._tokenizer = MagicMock()
#     mock_tokenizer.name_or_path = "mock_json_fail_tokenizer"

#     # Mock to_str to return valid JSON with merges
#     mock_tokenizer._tokenizer.to_str.return_value = (
#         '{"model": {"merges": [["a", "b"]]}}'  # <--- VALID: uses square brackets
#     )

#     # Mock get_vocab to return empty dict, causing ID lookup failure
#     mock_tokenizer.get_vocab.return_value = {}

#     with pytest.warns(UserWarning) as record:
#         merges = _extract_bpe_merges(mock_tokenizer)

#     assert merges == [], "Merges should be empty if ID mapping failed for all pairs"
#     recorded_messages = [str(w.message) for w in record]

#     expected_warning = "Parsed JSON merges, but ID mapping failed for ALL pairs."
#     assert any(expected_warning in msg for msg in recorded_messages), (
#         f"Expected warning '{expected_warning}' not found in recorded warnings: {recorded_messages}"
#     )


def test_extract_merges_direct_access_success():
    """Test _extract_bpe_merges successfully extracts merges via direct access."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.is_fast = True
    mock_tokenizer._tokenizer = MagicMock()
    mock_tokenizer._tokenizer.model = MagicMock()
    mock_tokenizer.name_or_path = "mock_direct_success_tokenizer"

    # Make JSON parsing fail (e.g., raise exception or return no merges)
    mock_tokenizer._tokenizer.to_str.side_effect = Exception("JSON parsing failed")

    # add direct merges
    direct_merges_list = [("hel", "lo")]
    mock_tokenizer._tokenizer.model.merges = direct_merges_list

    # Provide vocab to allow successful mapping
    mock_tokenizer.get_vocab.return_value = {"hel": 10, "lo": 20, "hello": 100}
    merges = _extract_bpe_merges(mock_tokenizer)

    # Assert that the correct merges were returned, confirming the target line was hit
    assert merges == [(10, 20, 100)], (
        "Merges extracted via direct access are incorrect."
    )


if __name__ == "__main__":
    pytest.main()
