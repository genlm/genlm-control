import pytest
import numpy as np
import asyncio
import warnings

from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
from genlm.control import AWRS, BoolFSA, Potential, ByteLLM


@pytest.fixture(scope="module")
def model_name():
    return "gpt2"


# On a GPU box the default backend is vLLM at gpu_memory_utilization=0.9. This
# module instantiates BOTH a module-scoped `llm` engine and a per-test
# `byte_llm`/`ByteLLM` engine, so two 0.9-util engines must coexist on one GPU
# -> "Free memory ... less than desired GPU memory utilization". gpt2 is tiny, so
# cap each engine's footprint to a small fraction so they fit together. (CPU/HF
# backend ignores engine_opts, so this is a no-op off-GPU.)
_LOW_GPU = {"engine_opts": {"gpu_memory_utilization": 0.3}}


@pytest.fixture(scope="module")
def llm(model_name):
    """Provides the underlying LLM for the test module."""
    instance = load_model_by_name(model_name, llm_opts=_LOW_GPU)
    yield instance
    # Release the GPU engine when the module finishes so it doesn't linger and
    # starve subsequent test modules of GPU memory.
    cleanup = getattr(instance, "cleanup", None)
    if cleanup is not None:
        try:
            asyncio.run(cleanup())
        except Exception as e:
            # Surface (don't swallow) a failing teardown: a silently-failing
            # cleanup is exactly how a later module hits an unexplained OOM.
            warnings.warn(f"engine cleanup failed during teardown: {e}")


@pytest.fixture(scope="module")
def beam_params(llm):
    """Provides BeamParams configured with the model's default EOS token."""
    eos_byte_string = llm.byte_vocab[llm.tokenizer.eos_token_id].byte_string
    return BeamParams(K=5, prune_threshold=0.0, eos_byte_strings=[eos_byte_string])


@pytest.fixture
def byte_llm(llm, beam_params):
    """Provides a fresh ByteLLM instance for each test and handles cleanup.

    Reuses the single module-scoped ``llm`` engine instead of creating its own,
    so we don't accumulate multiple vLLM engines on the GPU (cleanup doesn't
    promptly reclaim GPU memory within a process)."""
    instance = ByteLLM(llm, beam_params)
    yield instance


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
async def test_cache_size_limit(llm, beam_params):
    """Test that cache respects the size limit."""
    cache_size = 5
    byte_llm = ByteLLM(llm, beam_params, cache_size=cache_size)

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
async def test_cache_lru_eviction(llm, beam_params):
    """Test that eviction removes the least-recently-*accessed* entry, not
    simply the oldest-inserted one."""
    cache_size = 3
    byte_llm = ByteLLM(llm, beam_params, cache_size=cache_size)

    try:
        # Three independent single-byte contexts: none is a prefix of another,
        # so caching one never touches another's recency as a side effect.
        await byte_llm.prefix([b"p"])
        await byte_llm.prefix([b"q"])
        await byte_llm.prefix([b"r"])
        assert set(byte_llm._beam_cache) == {b"p", b"q", b"r"}

        # Touch "p" again -- an exact cache hit -- making it most-recently-used.
        await byte_llm.prefix([b"p"])

        # A new entry pushes the cache over the limit: eviction must take the
        # least-recently-used entry ("q", never re-accessed), not "p".
        await byte_llm.prefix([b"s"])

        assert len(byte_llm._beam_cache) == cache_size
        assert b"q" not in byte_llm._beam_cache, "LRU entry should have been evicted"
        assert b"p" in byte_llm._beam_cache, (
            "recently-accessed entry should survive eviction"
        )
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
async def test_healing_disabled_fails(llm):
    """Without healing, K=1 beam fails on text requiring alternative tokenization."""
    eos = llm.byte_vocab[llm.tokenizer.eos_token_id].byte_string
    # Explicitly disable healing to test failure mode
    beam_params = BeamParams(K=1, eos_byte_strings=[eos], heal=False)
    byte_llm = ByteLLM(llm, beam_params)

    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, "big") for b in text.encode("utf-8")]

    try:
        with pytest.raises(ValueError, match="Beam became empty"):
            await byte_llm.prefix(context)
    finally:
        await byte_llm.cleanup()


@pytest.mark.asyncio
async def test_healing_enabled_succeeds(llm):
    """With healing enabled, K=1 beam processes more text than without healing."""
    eos = llm.byte_vocab[llm.tokenizer.eos_token_id].byte_string

    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, "big") for b in text.encode("utf-8")]

    # Test without healing - find how far we get
    beam_params_no_heal = BeamParams(K=1, eos_byte_strings=[eos], heal=False)
    no_heal_len = await measure_prefix_reach(ByteLLM(llm, beam_params_no_heal), context)

    # Test with healing - should get further
    beam_params_heal = BeamParams(K=1, eos_byte_strings=[eos], heal=True)
    heal_len = await measure_prefix_reach(ByteLLM(llm, beam_params_heal), context)

    assert (
        heal_len > no_heal_len
    ), f"Healing ({heal_len}) should exceed no-healing ({no_heal_len})"


@pytest.mark.asyncio
async def test_healing_max_backoff(llm):
    """heal_max_backoff bounds how far back healing may search for a valid
    retokenization point (TokenHealer.try_heal only tries k in
    [partial_len - max_backoff, partial_len]). At K=1, max_backoff=0 restricts
    the healer to the exact token boundary the normal (pre-heal) extend step
    already tried and failed at, so it can do no better than healing disabled
    -- and must reach strictly less far than unlimited backoff."""
    eos = llm.byte_vocab[llm.tokenizer.eos_token_id].byte_string
    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, "big") for b in text.encode("utf-8")]

    no_heal_len = await measure_prefix_reach(
        ByteLLM(llm, BeamParams(K=1, eos_byte_strings=[eos], heal=False)), context
    )
    no_backoff_len = await measure_prefix_reach(
        ByteLLM(
            llm, BeamParams(K=1, eos_byte_strings=[eos], heal=True, heal_max_backoff=0)
        ),
        context,
    )
    unlimited_len = await measure_prefix_reach(
        ByteLLM(
            llm,
            BeamParams(K=1, eos_byte_strings=[eos], heal=True, heal_max_backoff=None),
        ),
        context,
    )

    assert no_backoff_len == no_heal_len, (
        f"max_backoff=0 ({no_backoff_len}) retries only the boundary the "
        f"pre-heal extend step already failed at, so it should behave "
        f"exactly like heal=False ({no_heal_len})"
    )
    assert no_backoff_len < unlimited_len, (
        f"heal_max_backoff=0 ({no_backoff_len}) should reach strictly less far "
        f"than unlimited backoff ({unlimited_len}) -- the knob must constrain healing"
    )


# -------------------------
# Async context manager tests
# -------------------------


@pytest.mark.asyncio
async def test_context_manager_basic(llm, beam_params):
    """Test that ByteLLM works as an async context manager, including driving
    a full SMC run from inside the context."""

    async with ByteLLM(llm, beam_params) as byte_llm:
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

        # SMC sampling must also work while the engine is held via the context manager
        byte_llm.set_prompt_from_str("The answer is:")
        fsa = BoolFSA.from_regex(r" (yes|no)")
        sampler = AWRS(byte_llm, fsa.coerce(byte_llm, f=b"".join))
        sequences = await sampler.smc(
            n_particles=5, max_tokens=10, ess_threshold=0.5, verbosity=0
        )
        assert len(sequences) > 0
        for seq in sequences.decoded_posterior.keys():
            assert "yes" in seq or "no" in seq

    # After exiting context, cleanup should have been called
    # Cache should be cleared
    assert not byte_llm._beam_cache
    assert byte_llm._last_context is None
    assert byte_llm._last_beam is None


@pytest.mark.asyncio
async def test_context_manager_cleanup_on_exception(llm, beam_params):
    """Test that cleanup is called even when an exception occurs inside the context."""

    class TestException(Exception):
        pass

    byte_llm_ref = None

    with pytest.raises(TestException):
        async with ByteLLM(llm, beam_params) as byte_llm:
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
