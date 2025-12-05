
import pytest
import numpy as np
import asyncio
from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
from genlm.control import AWRS, BoolFSA, Potential, ByteLLM


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


@pytest.fixture(scope="module")
def beam_params(model_name):
    """Provides BeamParams configured with the model's default EOS token."""
    from genlm.backend import load_model_by_name

    llm = load_model_by_name(model_name)
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    return BeamParams(K=5, prune_threshold=0.0, eos_tokens={model_eos_token})


@pytest.fixture
def byte_llm(model_name, beam_params):
    """Provides a fresh ByteLLM instance for each test and handles cleanup."""
    instance = ByteLLM.from_name(model_name, beam_params)
    yield instance
    # Cleanup code will run after the test has finished
    asyncio.run(instance.cleanup())


@pytest.mark.asyncio
async def test_initialization(byte_llm: ByteLLM):
    """Tests that the ByteLLM is initialized correctly."""
    assert isinstance(byte_llm, Potential)
    assert len(byte_llm.vocab) == 256
    assert byte_llm.prompt_bytes == b""
    assert byte_llm._initial_beam is None
    assert not byte_llm._beam_cache


@pytest.mark.asyncio
async def test_set_prompt(byte_llm: ByteLLM):
    """Tests the set_prompt_from_str method."""
    prompt = "Hello"
    byte_llm.set_prompt_from_str(prompt)
    assert byte_llm.prompt_bytes == prompt.encode("utf-8")

    # Call it once to populate cache
    await byte_llm.prefix([b" "])
    assert byte_llm._beam_cache

    # Setting the same prompt should not clear the cache
    cache_before = dict(byte_llm._beam_cache)
    byte_llm.set_prompt_from_str(prompt)
    assert byte_llm._beam_cache == cache_before

    # Setting a new prompt should clear the cache
    byte_llm.set_prompt_from_str("New prompt")
    assert not byte_llm._beam_cache
    assert byte_llm._initial_beam is None


@pytest.mark.asyncio
async def test_prefix_and_complete_methods(byte_llm: ByteLLM):
    """Tests the prefix and complete methods for calculating log probabilities."""
    context = [b'H', b'e', b'l', b'l', b'o']

    # --- Test prefix --- #
    logp_hello = await byte_llm.prefix(context)
    assert isinstance(logp_hello, float)
    assert logp_hello < 0, "Invalid prefix log probability"

    # --- Test complete --- #
    complete_logp = await byte_llm.complete(context)
    assert isinstance(complete_logp, float)
    assert complete_logp < logp_hello, "Complete logp should be less than prefix logp"

    # --- Test consistency --- #
    # complete(C) = prefix(C) + logp(EOS|C)
    # So, logp(EOS|C) = complete(C) - prefix(C)
    logp_eos_given_c = complete_logp - logp_hello
    assert -100 < logp_eos_given_c < 0, f"Implied EOS logp is out of reasonable bounds: {logp_eos_given_c}"


@pytest.mark.asyncio
async def test_logw_next_values(byte_llm: ByteLLM):
    """Tests that logw_next returns sensible, finite values."""
    context = [b'H', b'e', b'l', b'l', b'o']
    lazy_weights = await byte_llm.logw_next(context)
    weights = lazy_weights.materialize()

    # The main point is to ensure we don't get -inf for valid next tokens
    space_logp = weights[b' ']
    comma_logp = weights[b',']
    eos_logp = weights[byte_llm.eos]

    print(f"logp(' ' | 'Hello') = {space_logp}")
    print(f"logp(',' | 'Hello') = {comma_logp}")
    print(f"logp(EOS | 'Hello') = {eos_logp}")

    assert np.isfinite(space_logp), "Logp for space should be finite"
    assert np.isfinite(comma_logp), "Logp for comma should be finite"
    assert np.isfinite(eos_logp), "Logp for EOS should be finite"


@pytest.mark.asyncio
async def test_bytelm_smc(byte_llm: ByteLLM):

    prompt = "Here is my honest opinion:"
    byte_llm.set_prompt_from_str(prompt)

    fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

    sampler = AWRS(byte_llm, fsa.coerce(byte_llm, f=b"".join))

    sequences = await sampler.smc(
        n_particles=10,
        max_tokens=30,
        ess_threshold=0.5,
        verbosity=1,
    )
    print(f"Output: {sequences.decoded_posterior}")
    assert len(sequences) > 0, "SMC should generate at least one sequence"
    assert len(sequences.decoded_posterior) == 1, "SMC did not terminate"


# -------------------------
# Adaptive token healing tests
# -------------------------

TEXT_THAT_NEEDS_HEALING = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."


async def _try_prefix(byte_llm: ByteLLM, text: str):
    """Helper to compute prefix probability for text using ByteLLM.

    Returns:
        (success, bytes_consumed, error):
            - success: True if completed successfully
            - bytes_consumed: Number of bytes successfully processed
            - error: Exception if failed, None otherwise
    """
    bs = text.encode("utf-8")
    context = [b.to_bytes(1, 'big') for b in bs]
    try:
        # This will call _get_or_create_beam_for_context which advances byte-by-byte
        await byte_llm.prefix(context)
        return True, len(bs), None
    except ValueError as e:
        # Extract bytes consumed from error message if available
        # Error format: "Beam became empty at byte X. Context so far: b'...'"
        import re
        match = re.search(r"Context so far: b'([^']*)'", str(e))
        if match:
            consumed = len(match.group(1).encode().decode('unicode_escape').encode('latin1'))
            return False, consumed, e
        return False, 0, e


@pytest.mark.asyncio
async def test_healing_disabled_fails(model_name):
    """Without healing, K=1 beam fails on text requiring alternative tokenization."""
    llm = load_model_by_name(model_name)
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(K=1, eos_tokens={model_eos_token})

    byte_llm = ByteLLM(llm, beam_params, heal=False)
    try:
        success, bytes_consumed, error = await _try_prefix(byte_llm, TEXT_THAT_NEEDS_HEALING)
        assert not success, "Expected failure with heal disabled, but succeeded"
        assert isinstance(error, ValueError), f"Expected ValueError, got {type(error)}"
        assert "Beam became empty" in str(error), "Error should mention empty beam"
        # Should fail before completing the full text
        assert bytes_consumed < len(TEXT_THAT_NEEDS_HEALING), \
            f"Should fail before consuming all {len(TEXT_THAT_NEEDS_HEALING)} bytes, but consumed {bytes_consumed}"
    finally:
        await byte_llm.cleanup()


@pytest.mark.asyncio
async def test_healing_enabled_succeeds(model_name):
    """With healing enabled, K=1 beam gets further or completes text."""
    llm = load_model_by_name(model_name)
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(K=1, eos_tokens={model_eos_token})

    # First, get baseline of how far we can get without healing
    byte_llm_no_heal = ByteLLM(llm, beam_params, heal=False)
    try:
        _, bytes_no_heal, _ = await _try_prefix(byte_llm_no_heal, TEXT_THAT_NEEDS_HEALING)
    finally:
        await byte_llm_no_heal.cleanup()

    # Now test with healing enabled
    byte_llm_heal = ByteLLM(llm, beam_params, heal=True)
    try:
        success_heal, bytes_heal, _ = await _try_prefix(byte_llm_heal, TEXT_THAT_NEEDS_HEALING)

        # Healing should allow us to get further than without healing
        assert bytes_heal > bytes_no_heal, \
            f"Healing should consume more bytes ({bytes_heal}) than no healing ({bytes_no_heal})"

        # Ideally it completes, but even getting further is evidence of healing working
        if success_heal:
            print(f"✓ Healing enabled complete success: {bytes_heal}/{len(TEXT_THAT_NEEDS_HEALING)} bytes")
        else:
            print(f"✓ Healing helped: {bytes_heal} bytes vs {bytes_no_heal} bytes without healing")
    finally:
        await byte_llm_heal.cleanup()


@pytest.mark.asyncio
async def test_healing_max_backoff_limited(model_name):
    """With limited backoff, healing is constrained."""
    llm = load_model_by_name(model_name)
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(K=1, eos_tokens={model_eos_token})

    # Test with unlimited healing
    byte_llm_unlimited = ByteLLM(llm, beam_params, heal=True, heal_max_backoff=None)
    try:
        _, bytes_unlimited, _ = await _try_prefix(byte_llm_unlimited, TEXT_THAT_NEEDS_HEALING)
    finally:
        await byte_llm_unlimited.cleanup()

    # Test with limited healing
    byte_llm_limited = ByteLLM(llm, beam_params, heal=True, heal_max_backoff=2)
    try:
        _, bytes_limited, _ = await _try_prefix(byte_llm_limited, TEXT_THAT_NEEDS_HEALING)

        # Limited backoff should consume same or fewer bytes than unlimited
        assert bytes_limited <= bytes_unlimited, \
            f"Limited backoff ({bytes_limited} bytes) should not exceed unlimited ({bytes_unlimited} bytes)"

        print(f"Limited backoff=2: {bytes_limited} bytes, Unlimited: {bytes_unlimited} bytes")
    finally:
        await byte_llm_limited.cleanup()
    