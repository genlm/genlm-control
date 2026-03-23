import pytest
import torch
import numpy as np
from arsenal.maths import logsumexp
from hypothesis import given, strategies as st, settings, reject

from genlm.backend.tokenization import Token
from genlm.control.potential.built_in import PromptedLLM
from genlm.control.potential.built_in.llm import TokenMappings

# pytest.mark.asyncio seems to cause issues with hypothesis
# and the vllm backend, so we use asyncio.run here.


async def reference_scorer(llm, context, eos=False, temp=1):
    """Compute the log probability of the context given the prompt."""
    context_ids = llm.encode_tokens(context)

    async def tempered(context_ids):
        logps = await llm.model.next_token_logprobs(context_ids)
        if temp != 1:
            logps = torch.log_softmax(logps / temp, dim=-1)
        return logps

    logps = await tempered(llm.prompt_ids)
    total_logp = logps[context_ids[0]].item()

    for i in range(1, len(context_ids)):
        logps = await tempered(llm.prompt_ids + context_ids[:i])
        total_logp += logps[context_ids[i]].item()

    if eos:
        logps = await tempered(llm.prompt_ids + context_ids)
        eos_logp = float("-inf")
        for i in llm.token_maps.eos_idxs:
            eos_logp = logsumexp([eos_logp, logps[i].item()])
        total_logp += eos_logp

    return total_logp


@pytest.fixture(
    scope="module",
    params=[
        ("hf", {"hf_opts": {"torch_dtype": "float"}}),
        # ("mock", {}),
    ],
)
def llm_config(request):
    return request.param


@pytest.fixture(scope="module")
def llm(llm_config):
    backend, opts = llm_config
    return PromptedLLM.from_name("gpt2", backend=backend, **opts)


@pytest.mark.asyncio
@given(st.text(min_size=1))
async def test_prompt_setting(llm, pre_prompt):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)

    # Test ids setter
    llm.prompt_ids = pre_prompt_ids
    assert llm.prompt_ids == pre_prompt_ids
    assert b"".join(t.byte_string for t in llm.prompt).decode() == pre_prompt

    # Test str setter
    llm.set_prompt_from_str(pre_prompt)
    assert b"".join(t.byte_string for t in llm.prompt).decode() == pre_prompt
    assert llm.prompt_ids == pre_prompt_ids


@pytest.mark.asyncio
@settings(deadline=None, max_examples=50)
@given(st.text(min_size=1), st.text(min_size=1), st.floats(min_value=1e-6, max_value=3))
async def test_scoring(llm, pre_prompt, context_str, temp):
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)
    context = llm.tokenize(context_str)

    llm.temperature = temp
    llm.prompt_ids = pre_prompt_ids

    have = await llm.prefix(context)
    want = await reference_scorer(llm, context, temp=temp)
    assert np.isclose(have, want), [have, want]

    have = await llm.complete(context)
    want = await reference_scorer(llm, context, eos=True, temp=temp)
    assert np.isclose(have, want), [have, want]


@pytest.mark.asyncio
@settings(deadline=None, max_examples=50)
@given(
    st.text(min_size=1, max_size=10),
    st.text(min_size=1, max_size=10),
    st.floats(
        min_value=0.75, max_value=3
    ),  # TODO: scrutinize precision with low temperature
)
async def test_properties(llm, pre_prompt, context, temp):
    if "!" in context or "?" in context:
        reject()  # We are using these as eos tokens, so we skip this example.
    pre_prompt_ids = llm.model.tokenizer.encode(pre_prompt)
    llm.prompt_ids = pre_prompt_ids
    context = llm.tokenize(context)
    llm.temperature = temp

    await llm.assert_logw_next_consistency(context, top=10, rtol=0.01, atol=1e-3)
    await llm.assert_autoreg_fact(context, rtol=0.01, atol=1e-3)

    new_llm = llm.spawn_new_eos(eos_byte_strings=[b"!", b"?"])
    await new_llm.assert_logw_next_consistency(context, top=10, rtol=0.01, atol=1e-3)
    await new_llm.assert_autoreg_fact(context, rtol=0.01, atol=1e-3)


@pytest.mark.asyncio
@settings(deadline=None, max_examples=50)
@given(st.lists(st.text(min_size=1), min_size=1, max_size=4))
async def test_batch_consistency(llm, contexts):
    contexts = [llm.tokenize(context) for context in contexts]
    await llm.assert_batch_consistency(contexts, rtol=1e-3, atol=1e-3)


@st.composite
def eos_test_params(draw):
    # Probably can decrase the size of these ranges for faster tests.
    eos_token_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=50256),
            min_size=1,
            max_size=3,
            unique=True,
        )
    )
    valid_ids = st.integers(min_value=0, max_value=50256).filter(
        lambda x: x not in eos_token_ids
    )
    context_ids = draw(st.lists(valid_ids, min_size=1, max_size=5))
    prompt_ids = draw(
        st.lists(st.integers(min_value=0, max_value=50256), min_size=1, max_size=5)
    )
    return eos_token_ids, context_ids, prompt_ids


@pytest.mark.asyncio
@settings(deadline=None)
@given(eos_test_params())
async def test_new_eos_tokens(llm, params):
    with pytest.raises(
        ValueError, match="Cannot reset eos_byte_strings after initialization"
    ):
        llm.eos_byte_strings = []

    eos_token_ids, context_ids, prompt_ids = params
    llm.prompt_ids = prompt_ids
    eos_bs = [llm.token_maps.decode[x].byte_string for x in eos_token_ids]
    new_llm = llm.spawn_new_eos(eos_byte_strings=eos_bs)
    assert new_llm.eos_byte_strings == eos_bs

    new_llm.temperature = 1.0

    assert new_llm.prompt_ids == prompt_ids  # check prompt_ids is not changed
    assert new_llm.token_maps.eos_idxs == eos_token_ids
    vocab_bytes = {token.byte_string for token in new_llm.vocab}
    decode_bytes = {token.byte_string for token in new_llm.token_maps.decode}
    assert decode_bytes - set(eos_bs) == vocab_bytes

    context = new_llm.decode_tokens(context_ids)
    have = await new_llm.complete(context)
    want = await reference_scorer(new_llm, context, eos=True)
    assert np.isclose(have, want), [have, want]


def test_invalid_eos_tokens(llm):
    # Test EOS token not in vocabulary
    invalid_eos = [b"THIS_TOKEN_DOES_NOT_EXIST"]
    with pytest.raises(ValueError, match="EOS token not in language model vocabulary"):
        llm.spawn_new_eos(eos_byte_strings=invalid_eos)

    # Test duplicate EOS tokens
    duplicate_eos = [
        llm.token_maps.decode[0].byte_string,
        llm.token_maps.decode[0].byte_string,
    ]
    with pytest.raises(ValueError, match="Duplicate eos byte strings"):
        llm.spawn_new_eos(eos_byte_strings=duplicate_eos)

    # Test attempting to modify eos_byte_strings directly
    with pytest.raises(
        ValueError, match="Cannot reset eos_byte_strings after initialization"
    ):
        llm.eos_byte_strings = [llm.token_maps.decode[0].byte_string]


def test_invalid_token_encoding(llm):
    # Test encoding invalid tokens
    invalid_tokens = [b"INVALID_TOKEN"]
    with pytest.raises(ValueError, match="Token .* not in vocabulary"):
        llm.encode_tokens(invalid_tokens)


def test_prompt_from_str_invalid_type(llm):
    with pytest.raises(ValueError, match="Prompt must a string"):
        llm.set_prompt_from_str(42)


def test_spawn(llm):
    new_llm = llm.spawn()
    assert new_llm.prompt_ids == llm.prompt_ids
    assert new_llm.token_maps == llm.token_maps
    assert new_llm.vocab == llm.vocab

    new_llm = llm.spawn(temperature=1.0)
    assert new_llm.temperature == 1.0
    assert new_llm.prompt_ids == llm.prompt_ids
    assert new_llm.token_maps == llm.token_maps
    assert new_llm.vocab == llm.vocab

    new_llm = llm.spawn(prompt_ids=[0])
    assert new_llm.temperature == llm.temperature
    assert new_llm.prompt_ids == [0]
    assert new_llm.token_maps == llm.token_maps
    assert new_llm.vocab == llm.vocab

    new_llm = llm.spawn(eos_byte_strings=[b"!"], temperature=1.0)
    assert new_llm.token_maps.eos_idxs == [0]
    assert new_llm.temperature == 1.0
    assert new_llm.prompt_ids == llm.prompt_ids
    assert new_llm.token_maps != llm.token_maps


def test_providing_eos_tokens_and_token_maps(llm):
    with pytest.raises(
        ValueError, match="eos_byte_strings must not be provided when token_maps is provided."
    ):
        PromptedLLM(
            llm.model,
            prompt_ids=llm.prompt_ids,
            eos_byte_strings=[b"!"],
            token_maps=llm.token_maps,
        )


def test_to_autobatched(llm):
    with pytest.raises(ValueError, match="PromptedLLMs are autobatched by default"):
        llm.to_autobatched()


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="vllm requires CUDA")
async def test_vllm_backend():
    # VLLM backend isn't playing well with hypothesis so we test it here.
    # Note though that any differences between backends are encapsulated in the AsyncLM class, which
    # is tested in genlm_backend, so we shouldn't expect any significant differences in testing outcomes.
    llm = PromptedLLM.from_name(
        "gpt2",
        backend="vllm",
        engine_opts={"dtype": "float", "gpu_memory_utilization": 0.5},
    )

    llm.set_prompt_from_str("hello")
    context = llm.tokenize(" world!")

    await llm.assert_logw_next_consistency(context, top=10, rtol=1e-3, atol=1e-3)
    await llm.assert_autoreg_fact(context, rtol=1e-3, atol=1e-3)
    await llm.assert_batch_consistency(
        [context, llm.tokenize(" world")], rtol=1e-3, atol=1e-3
    )

    new_llm = llm.spawn_new_eos(eos_byte_strings=[b"!"])
    assert new_llm.token_maps.eos_idxs == [0]
    assert new_llm.token_maps.decode[0].byte_string == b"!"

    context = llm.tokenize(" world")
    await new_llm.assert_logw_next_consistency(context, top=10, rtol=1e-3, atol=1e-3)
    await new_llm.assert_autoreg_fact(context, rtol=1e-3, atol=1e-3)
    await new_llm.assert_batch_consistency(
        [context, llm.tokenize(" worlds")], rtol=1e-3, atol=1e-3
    )


def test_llm_repr(llm):
    repr(llm)


def test_prompt_warning(llm):
    with pytest.warns(UserWarning):
        llm.set_prompt_from_str("hello ")


def test_encode_tokens_with_bytes(llm):
    """Test that encode_tokens works with bytes (deprecated path) and issues warning."""
    token = llm.vocab[0]
    byte_string = token.byte_string

    with pytest.warns(
        DeprecationWarning, match="Passing bytes to encode_tokens is deprecated"
    ):
        result = llm.encode_tokens([byte_string])

    assert result == [token.token_id]


def test_encode_tokens_invalid_bytes(llm):
    """Test that encode_tokens raises error for invalid bytes."""
    with pytest.raises(ValueError, match="Token .* not in vocabulary"):
        llm.encode_tokens([b"THIS_DOES_NOT_EXIST_IN_VOCAB_12345"])


def test_token_encode_dict_getitem_bytes(llm):
    """Test _TokenEncodeDict.__getitem__ with bytes key (deprecated path)."""
    token = llm.vocab[0]
    with pytest.warns(DeprecationWarning, match="Indexing token_maps.encode by bytes is deprecated"):
        idx = llm.token_maps.encode[token.byte_string]
    assert idx == llm.token_maps.encode[token]


def test_token_encode_dict_getitem_missing(llm):
    """Test _TokenEncodeDict.__getitem__ raises KeyError for missing key."""
    with pytest.raises(KeyError):
        llm.token_maps.encode[b"THIS_DOES_NOT_EXIST_IN_VOCAB_12345"]


def test_token_encode_dict_contains_token(llm):
    """Test _TokenEncodeDict.__contains__ with Token key."""
    token = llm.vocab[0]
    assert token in llm.token_maps.encode


def test_token_encode_dict_contains_bytes(llm):
    """Test _TokenEncodeDict.__contains__ with bytes key (deprecated fallback)."""
    token = llm.vocab[0]
    assert token.byte_string in llm.token_maps.encode


def test_token_encode_dict_contains_missing(llm):
    """Test _TokenEncodeDict.__contains__ returns False for missing key."""
    assert b"THIS_DOES_NOT_EXIST_IN_VOCAB_12345" not in llm.token_maps.encode


def test_find_token_id_for_bytes(llm):
    """Test _find_token_id_for_bytes returns first match and caches."""
    token = llm.vocab[0]
    tid = llm._find_token_id_for_bytes(token.byte_string)
    assert tid == token.token_id
    # Second call uses cache
    tid2 = llm._find_token_id_for_bytes(token.byte_string)
    assert tid2 == tid


def test_find_token_id_for_bytes_missing(llm):
    """Test _find_token_id_for_bytes returns None for missing bytes."""
    assert llm._find_token_id_for_bytes(b"THIS_DOES_NOT_EXIST_12345") is None


def test_duplicate_eos_byte_string_warning():
    """Test that TokenMappings warns when multiple tokens have the same EOS byte_string."""
    decode = [
        Token(token_id=0, byte_string=b"hello"),
        Token(token_id=1, byte_string=b"world"),
        Token(token_id=2, byte_string=b"hello"),
        Token(token_id=3, byte_string=b"foo"),
    ]

    with pytest.warns(UserWarning, match="Multiple tokens with EOS byte_string"):
        TokenMappings.create(decode=decode, eos_byte_strings=[b"hello"])
