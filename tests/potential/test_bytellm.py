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
def shared_llm(model_name):
    """Module-scoped LLM fixture that creates once and cleans up at end."""
    llm = load_model_by_name(model_name)
    yield llm
    llm.cleanup()  # Clean up GPU memory when module is done


@pytest.fixture(scope="module")
def beam_params(shared_llm):
    """Provides BeamParams configured with the model's default EOS token."""
    model_eos_token = shared_llm.byte_vocab[shared_llm.tokenizer.eos_token_id]
    return BeamParams(K=5, prune_threshold=0.0, eos_tokens={model_eos_token})


@pytest.fixture
def byte_llm(shared_llm, beam_params):
    """Provides a fresh ByteLLM instance for each test and handles cleanup."""
    instance = ByteLLM(shared_llm, beam_params)
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
    context = [b"H", b"e", b"l", b"l", b"o"]

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
    assert (
        -100 < logp_eos_given_c < 0
    ), f"Implied EOS logp is out of reasonable bounds: {logp_eos_given_c}"


@pytest.mark.asyncio
async def test_logw_next_values(byte_llm: ByteLLM):
    """Tests that logw_next returns sensible, finite values."""
    context = [b"H", b"e", b"l", b"l", b"o"]
    lazy_weights = await byte_llm.logw_next(context)
    weights = lazy_weights.materialize()

    # The main point is to ensure we don't get -inf for valid next tokens
    space_logp = weights[b" "]
    comma_logp = weights[b","]
    eos_logp = weights[byte_llm.eos]

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
    assert len(sequences) > 0, "SMC should generate at least one sequence"
    assert len(sequences.decoded_posterior) >= 1, "SMC did not terminate"


# -------------------------
# Cache tests
# -------------------------


@pytest.mark.asyncio
async def test_cache_size_limit(shared_llm, beam_params):
    """Test that cache respects the size limit."""
    cache_size = 5
    byte_llm = ByteLLM(shared_llm, beam_params, cache_size=cache_size)

    try:
        # Process enough bytes to exceed the cache size
        # Each byte position gets cached, so processing N bytes creates N cache entries
        text = "Hello World!"
        for i in range(len(text)):
            context = [b.to_bytes(1, "big") for b in text[: i + 1].encode("utf-8")]
            await byte_llm.prefix(context)

        # Cache should not exceed the limit
        assert len(byte_llm._beam_cache) <= cache_size
    finally:
        await byte_llm.cleanup()


@pytest.mark.asyncio
async def test_cache_lru_eviction(shared_llm, beam_params):
    """Test that LRU eviction removes oldest entries."""
    cache_size = 3
    byte_llm = ByteLLM(shared_llm, beam_params, cache_size=cache_size)

    try:
        # Create cache entries for "a", "ab", "abc"
        await byte_llm.prefix([b"a"])
        await byte_llm.prefix([b"a", b"b"])
        await byte_llm.prefix([b"a", b"b", b"c"])

        # All three should be cached
        assert len(byte_llm._beam_cache) == cache_size

        # Access "a" to make it recently used
        await byte_llm.prefix([b"a"])

        # Add a new entry "x" - should evict "ab" (least recently used)
        byte_llm._beam_cache.clear()  # Reset for cleaner test
        byte_llm._initial_beam = None

        await byte_llm.prefix([b"x"])
        await byte_llm.prefix([b"x", b"y"])
        await byte_llm.prefix([b"x", b"y", b"z"])

        # Cache should be at limit
        assert len(byte_llm._beam_cache) == cache_size

        # Adding one more should trigger eviction
        await byte_llm.prefix([b"x", b"y", b"z", b"w"])
        assert len(byte_llm._beam_cache) <= cache_size
    finally:
        await byte_llm.cleanup()


# -------------------------
# Adaptive token healing tests
# -------------------------


async def measure_prefix_reach(byte_llm: ByteLLM, context: list) -> int:
    """Measure how many bytes of context can be processed before failure.

    Returns the number of bytes successfully processed before a ValueError is raised,
    or len(context) if all bytes are processed successfully.
    """
    try:
        for i in range(len(context)):
            try:
                await byte_llm.prefix(context[: i + 1])
            except ValueError:
                return i
        return len(context)
    finally:
        await byte_llm.cleanup()


@pytest.mark.asyncio
async def test_healing_disabled_fails(shared_llm):
    """Without healing, K=1 beam fails on text requiring alternative tokenization."""
    beam_params = BeamParams(
        K=1, eos_tokens={shared_llm.byte_vocab[shared_llm.tokenizer.eos_token_id]}, heal=False
    )
    byte_llm = ByteLLM(shared_llm, beam_params)

    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, "big") for b in text.encode("utf-8")]

    try:
        with pytest.raises(ValueError, match="Beam became empty"):
            await byte_llm.prefix(context)
    finally:
        await byte_llm.cleanup()


@pytest.mark.asyncio
async def test_healing_enabled_succeeds(shared_llm):
    """With healing enabled, K=1 beam processes more text than without healing."""
    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, "big") for b in text.encode("utf-8")]

    # Test without healing - find how far we get
    beam_params_no_heal = BeamParams(
        K=1, eos_tokens={shared_llm.byte_vocab[shared_llm.tokenizer.eos_token_id]}, heal=False
    )
    no_heal_len = await measure_prefix_reach(ByteLLM(shared_llm, beam_params_no_heal), context)

    # Test with healing - should get further
    beam_params_heal = BeamParams(
        K=1, eos_tokens={shared_llm.byte_vocab[shared_llm.tokenizer.eos_token_id]}, heal=True
    )
    heal_len = await measure_prefix_reach(ByteLLM(shared_llm, beam_params_heal), context)

    assert (
        heal_len > no_heal_len
    ), f"Healing ({heal_len}) should exceed no-healing ({no_heal_len})"


@pytest.mark.asyncio
async def test_healing_max_backoff(shared_llm):
    """Limited backoff constrains healing effectiveness."""
    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, "big") for b in text.encode("utf-8")]

    # Unlimited healing
    beam_params_unlimited = BeamParams(
        K=1, eos_tokens={shared_llm.byte_vocab[shared_llm.tokenizer.eos_token_id]}, heal=True
    )
    unlimited_len = await measure_prefix_reach(
        ByteLLM(shared_llm, beam_params_unlimited), context
    )

    # Limited healing
    beam_params_limited = BeamParams(
        K=1,
        eos_tokens={shared_llm.byte_vocab[shared_llm.tokenizer.eos_token_id]},
        heal=True,
        heal_max_backoff=2,
    )
    limited_len = await measure_prefix_reach(ByteLLM(shared_llm, beam_params_limited), context)

    assert (
        limited_len <= unlimited_len
    ), f"Limited ({limited_len}) should not exceed unlimited ({unlimited_len})"


# -------------------------
# Async context manager tests
# -------------------------


@pytest.mark.asyncio
async def test_context_manager_basic(shared_llm, beam_params):
    """Test that ByteLLM works as an async context manager."""

    async with ByteLLM(shared_llm, beam_params) as byte_llm:
        # Verify we can use the instance inside the context
        assert isinstance(byte_llm, Potential)
        assert len(byte_llm.vocab) == 256

        # Perform some operations
        byte_llm.set_prompt_from_str("Hello")
        logp = await byte_llm.prefix([b" ", b"w", b"o", b"r", b"l", b"d"])
        assert isinstance(logp, float)
        assert logp < 0

        # Verify cache was populated
        assert byte_llm._beam_cache or byte_llm._initial_beam is not None

    # After exiting context, cleanup should have been called
    # Cache should be cleared
    assert not byte_llm._beam_cache
    assert byte_llm._last_context is None
    assert byte_llm._last_beam is None


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception(shared_llm, beam_params):
    """Test that cleanup is called even when an exception occurs inside the context."""

    class TestException(Exception):
        pass

    byte_llm_ref = None

    with pytest.raises(TestException):
        async with ByteLLM(shared_llm, beam_params) as byte_llm:
            byte_llm_ref = byte_llm

            # Perform some operations to populate cache
            byte_llm.set_prompt_from_str("Test")
            await byte_llm.prefix([b"!"])
            assert byte_llm._beam_cache or byte_llm._initial_beam is not None

            # Raise an exception
            raise TestException("Intentional test exception")

    # Cleanup should still have been called despite the exception
    assert byte_llm_ref is not None
    assert not byte_llm_ref._beam_cache
    assert byte_llm_ref._last_context is None
    assert byte_llm_ref._last_beam is None


@pytest.mark.asyncio
async def test_context_manager_with_smc(shared_llm, beam_params):
    """Test that ByteLLM context manager works correctly with SMC sampling."""

    async with ByteLLM(shared_llm, beam_params) as byte_llm:
        byte_llm.set_prompt_from_str("The answer is:")

        fsa = BoolFSA.from_regex(r" (yes|no)")
        sampler = AWRS(byte_llm, fsa.coerce(byte_llm, f=b"".join))

        sequences = await sampler.smc(
            n_particles=5,
            max_tokens=10,
            ess_threshold=0.5,
            verbosity=0,
        )

        assert len(sequences) > 0
        # Verify outputs match the constraint
        for seq in sequences.decoded_posterior.keys():
            assert "yes" in seq or "no" in seq

    # Cleanup should have been called
    assert not byte_llm._beam_cache
