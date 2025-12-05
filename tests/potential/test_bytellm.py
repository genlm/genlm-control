
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
    assert len(sequences.decoded_posterior) >= 1, "SMC did not terminate"


# -------------------------
# Adaptive token healing tests
# -------------------------

@pytest.mark.asyncio
async def test_healing_disabled_fails(model_name):
    """Without healing, K=1 beam fails on text requiring alternative tokenization."""
    llm = load_model_by_name(model_name)
    beam_params = BeamParams(K=1, eos_tokens={llm.byte_vocab[llm.tokenizer.eos_token_id]})
    byte_llm = ByteLLM(llm, beam_params, heal=False)

    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, 'big') for b in text.encode('utf-8')]

    try:
        with pytest.raises(ValueError, match="Beam became empty"):
            await byte_llm.prefix(context)
    finally:
        await byte_llm.cleanup()


@pytest.mark.asyncio
async def test_healing_enabled_succeeds(model_name):
    """With healing enabled, K=1 beam processes more text than without healing."""
    llm = load_model_by_name(model_name)
    beam_params = BeamParams(K=1, eos_tokens={llm.byte_vocab[llm.tokenizer.eos_token_id]})

    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, 'big') for b in text.encode('utf-8')]

    # Test without healing - find how far we get
    byte_llm_no_heal = ByteLLM(llm, beam_params, heal=False)
    try:
        for i in range(len(context)):
            try:
                await byte_llm_no_heal.prefix(context[:i+1])
                no_heal_len = i + 1
            except ValueError:
                no_heal_len = i
                break
    finally:
        await byte_llm_no_heal.cleanup()

    # Test with healing - should get further
    byte_llm_heal = ByteLLM(llm, beam_params, heal=True)
    try:
        for i in range(len(context)):
            try:
                await byte_llm_heal.prefix(context[:i+1])
                heal_len = i + 1
            except ValueError:
                heal_len = i
                break

        assert heal_len > no_heal_len, f"Healing ({heal_len}) should exceed no-healing ({no_heal_len})"
    finally:
        await byte_llm_heal.cleanup()


@pytest.mark.asyncio
async def test_healing_max_backoff(model_name):
    """Limited backoff constrains healing effectiveness."""
    llm = load_model_by_name(model_name)
    beam_params = BeamParams(K=1, eos_tokens={llm.byte_vocab[llm.tokenizer.eos_token_id]})

    text = ". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti ."
    context = [b.to_bytes(1, 'big') for b in text.encode('utf-8')]

    # Unlimited healing
    byte_llm_unlimited = ByteLLM(llm, beam_params, heal=True)
    try:
        for i in range(len(context)):
            try:
                await byte_llm_unlimited.prefix(context[:i+1])
                unlimited_len = i + 1
            except ValueError:
                unlimited_len = i
                break
    finally:
        await byte_llm_unlimited.cleanup()

    # Limited healing
    byte_llm_limited = ByteLLM(llm, beam_params, heal=True, heal_max_backoff=2)
    try:
        for i in range(len(context)):
            try:
                await byte_llm_limited.prefix(context[:i+1])
                limited_len = i + 1
            except ValueError:
                limited_len = i
                break

        assert limited_len <= unlimited_len, \
            f"Limited ({limited_len}) should not exceed unlimited ({unlimited_len})"
    finally:
        await byte_llm_limited.cleanup()
    