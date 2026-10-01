"""Tests for HarmonyPotential channel extraction from harmony chat examples."""

import json

import numpy as np
import pytest
import torch
from arsenal.maths import logsumexp
from pathlib import Path

from genlm.control import SMC, direct_token_sampler
from genlm.control.constant import EOS
from genlm.control.potential.built_in import WCFG, BoolCFG
from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.potential.coerce import Coerced
from genlm.control.potential.harmony import HarmonyPotential, HarmonyChat
from genlm.control.sampler.token import AWRS
from genlm.grammar import CFG, Boolean, Float


model_name = "unsloth/gpt-oss-20b-BF16"


def coerce_bytes_to_chars(bytes_tokens):
    """Convert a sequence of byte tokens to a list of characters."""
    byte_string = b"".join(bytes_tokens)
    decoded = byte_string.decode("utf-8", errors="replace")
    return list(decoded)


@pytest.fixture
def harmony_examples_path():
    """Path to the harmony chat examples JSON file."""
    test_dir = Path(__file__).parent
    return test_dir / "harmony_chat_examples.json"


@pytest.fixture
def harmony_examples(harmony_examples_path):
    """Load harmony chat examples from JSON file."""
    with open(harmony_examples_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def tokenizer():
    """Get GPT-OSS tokenizer for tokenizing full responses."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_name)
    except ImportError:
        pytest.skip("transformers not available")
    except Exception as e:
        pytest.skip(f"Could not load tokenizer: {e}")


@pytest.fixture(scope="module")
def promptedllm():
    """Load the PromptedLLM once per module to avoid repeated GPU memory allocation."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) < (8, 0):
        pytest.skip("CUDA not available or compute capability < 8.0")
    elif (
        torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved()
    ) / 1e9 < 40:
        pytest.skip("Not enough GPU memory free")

    return PromptedLLM.from_name(model_name, backend="hf", temperature=0.5)


@pytest.fixture
def wcfg():
    grammar_str = """
    0.9 : S -> a S a
    0.1 : S -> b
    """
    cfg = CFG.from_string(grammar_str, Float)
    return WCFG(cfg)


@pytest.fixture
def BooleanCfg():
    grammar_str = """
    1 : S -> a S b
    1 : S -> a b
    """
    cfg = CFG.from_string(grammar_str, Boolean)
    return BoolCFG(cfg)


@pytest.mark.asyncio
async def test_potential_evaluation_single_channel(wcfg, tokenizer):
    """Test that the harmony potential is correctly evaluated when constraining a single channel."""
    cfg_inputs = {"analysis": "aaaabaaaa", "commentary": "bbbbbbbbb", "final": "aaa"}

    potential_vocab = HarmonyChat(tokenizer).potential_vocab
    coerced_cfg = Coerced(wcfg, potential_vocab, f=coerce_bytes_to_chars, prune=False)
    base_cfg = wcfg.cfg

    def tot_w(string):
        return np.log(base_cfg(string))

    def pfx_w(string):
        return np.log(base_cfg.prefix_grammar(string))

    chat = (
        "<|channel|>analysis<|message|>"
        + cfg_inputs["analysis"]
        + "<|end|><|start|>assistant<|channel|>commentary<|message|>"
        + cfg_inputs["commentary"]
        + "<|end|><|start|>assistant<|channel|>final<|message|>"
        + cfg_inputs["final"]
    )

    for channel in ["analysis", "final", "commentary"]:
        harmony_potential = HarmonyPotential(
            base_potential=coerced_cfg,
            llm_tokenizer=tokenizer,
            constrained_channels=[channel],
        )
        chat_ids = harmony_potential.harmony_chat.tokenizer.encode(chat)
        chat_bytes = harmony_potential.harmony_chat.decode_tokens(chat_ids)

        assert np.isclose(
            await harmony_potential.complete(chat_bytes), tot_w(cfg_inputs[channel])
        )
        assert np.isclose(
            await harmony_potential.prefix(chat_bytes), pfx_w(cfg_inputs[channel])
        )


@pytest.mark.asyncio
async def test_potential_evaluation_multiple_channels(wcfg, tokenizer):
    """Test that the harmony potential is correctly evaluated when constraining all channels at once."""
    cfg_inputs = {"analysis": "aaaabaaaa", "commentary": "aba", "final": "aaa"}

    potential_vocab = HarmonyChat(tokenizer).potential_vocab
    coerced_cfg = Coerced(wcfg, potential_vocab, f=coerce_bytes_to_chars, prune=False)
    base_cfg = wcfg.cfg

    def tot_w(string):
        return np.log(base_cfg(string))

    def pfx_w(string):
        return np.log(base_cfg.prefix_grammar(string))

    chat = (
        "<|channel|>analysis<|message|>"
        + cfg_inputs["analysis"]
        + "<|end|><|start|>assistant<|channel|>commentary<|message|>"
        + cfg_inputs["commentary"]
        + "<|end|><|start|>assistant<|channel|>final<|message|>"
        + cfg_inputs["final"]
    )

    harmony_potential = HarmonyPotential(
        base_potential=coerced_cfg,
        llm_tokenizer=tokenizer,
        constrained_channels=["analysis", "final", "commentary"],
    )
    chat_ids = harmony_potential.harmony_chat.tokenizer.encode(chat)
    chat_bytes = harmony_potential.harmony_chat.decode_tokens(chat_ids)

    # When all channels are constrained, complete weight is the sum of individual weights.
    expected_complete = sum(tot_w(cfg_inputs[ch]) for ch in cfg_inputs)
    assert np.isclose(await harmony_potential.complete(chat_bytes), expected_complete)

    # For prefix: analysis and commentary are closed (complete), only final is open (prefix).
    expected_prefix = (
        tot_w(cfg_inputs["analysis"])
        + tot_w(cfg_inputs["commentary"])
        + pfx_w(cfg_inputs["final"])
    )
    assert np.isclose(await harmony_potential.prefix(chat_bytes), expected_prefix)


def test_harmony_failure_cases(tokenizer):
    """Test that HarmonyChat raises an error for invalid chat formats."""
    harmony_chat = HarmonyChat(tokenizer)
    with pytest.raises(AssertionError):  # invalid syntax
        harmony_chat.extract_harmony_channels_from_string(
            "<|channel|>analysis<|end|>aaabbb<|end|>"
        )
    with pytest.raises(AssertionError):  # invalid channel
        harmony_chat.extract_harmony_channels_from_string(
            "<|channel|>astronomy<|message|>aaabbb<|end|>"
        )


def test_harmony_potential_validation(tokenizer, wcfg):
    """Test that HarmonyPotential validates constrained_channels correctly."""
    potential_vocab = HarmonyChat(tokenizer).potential_vocab
    coerced_cfg = Coerced(wcfg, potential_vocab, f=coerce_bytes_to_chars, prune=False)

    with pytest.raises(ValueError, match="non-empty"):
        HarmonyPotential(coerced_cfg, tokenizer, constrained_channels=[])

    with pytest.raises(ValueError, match="Invalid channel"):
        HarmonyPotential(coerced_cfg, tokenizer, constrained_channels=["invalid"])


def test_harmony_encode_tokens_bytes_fallback(tokenizer):
    """Test HarmonyChat.encode_tokens with bytes input (deprecated path)."""
    harmony_chat = HarmonyChat(tokenizer)
    # Get a token and its byte_string
    token = harmony_chat.token_maps.decode[0]
    byte_string = token.byte_string

    # Bytes path should work and return the same token_id
    result_bytes = harmony_chat.encode_tokens([byte_string])
    result_token = harmony_chat.encode_tokens([token])
    assert result_bytes == result_token


def test_harmony_encode_tokens_mixed(tokenizer):
    """Test HarmonyChat.encode_tokens with mixed Token and bytes input."""
    harmony_chat = HarmonyChat(tokenizer)
    t0 = harmony_chat.token_maps.decode[0]
    t1 = harmony_chat.token_maps.decode[1]
    # Mix Token objects and raw bytes
    result = harmony_chat.encode_tokens([t0, t1.byte_string])
    assert result == [t0.token_id, t1.token_id]


def test_harmony_channel_extraction(harmony_examples, tokenizer):
    """Test that HarmonyChat correctly extracts channels from harmony chat examples."""
    samples = harmony_examples["samples"]

    harmony_chat = HarmonyChat(tokenizer)
    for sample in samples:
        sample_id = sample["sample_id"]
        full_response = sample["full_response"]
        expected_channels = sample["channels"]

        extracted_channels = harmony_chat.extract_harmony_channels_from_string(
            full_response, add_special_tokens=False
        )

        for channel_name in ["analysis", "final", "commentary"]:
            expected = expected_channels.get(channel_name)
            extracted = extracted_channels.get(channel_name)

            if expected is None:
                assert extracted is None, (
                    f"Sample {sample_id}: Expected {channel_name} to be None, "
                    f"but got: {extracted}"
                )
            else:
                assert extracted is not None, (
                    f"Sample {sample_id}: Expected {channel_name} to exist, but got None"
                )

                assert extracted["is_prefix"] == expected["is_prefix"], (
                    f"Sample {sample_id}, {channel_name}: "
                    f"Expected is_prefix={expected['is_prefix']}, "
                    f"got {extracted['is_prefix']}"
                )

                extracted_content = tokenizer.decode(
                    harmony_chat.encode_tokens(extracted["content"])
                )
                expected_content = expected["content"]

                assert extracted_content == expected_content, (
                    f"Sample {sample_id}, {channel_name}: Content mismatch.\n"
                    f"Expected: {expected_content[:100]}...\n"
                    f"Got: {extracted_content[:100]}..."
                )


@pytest.mark.asyncio
async def test_harmony_awrs_constrained_sampling(promptedllm, tokenizer, BooleanCfg):
    """Test HarmonyPotential with AWRS and SMC for constrained generation.

    Verifies that the full pipeline using prefix and complete methods produces
    output accepted by the grammar.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    messages = [
        {
            "role": "user",
            "content": "Please sample a string from a^nb^n where n >= 5. Please provide just the final answer.",
        }
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, reasoning_effort="low"
    )
    prompt_str = tokenizer.decode(prompt_ids)
    promptedllm.set_prompt_from_str(prompt_str)

    coerced_cfg = BooleanCfg.coerce(promptedllm, f=coerce_bytes_to_chars, prune=False)
    harmony_cfg = HarmonyPotential(coerced_cfg, promptedllm.model.tokenizer, ["final"])

    sampler = AWRS(
        potential=promptedllm,
        condition=harmony_cfg,
        max_accepts=2,
        max_rejects=1000,
        prune_logws=False,
    )
    sequences = await SMC(sampler)(
        n_particles=3, ess_threshold=0.6, max_tokens=500, verbosity=0
    )

    sequence, weight = sequences[0]
    assert weight != float("-inf"), "No valid particle sampled (weight is -inf)."

    channels = harmony_cfg.harmony_chat.extract_harmony_channels_from_tokens(sequence)
    final_channel = channels.get("final")
    assert final_channel is not None and len(final_channel["content"]) > 0, (
        "Final channel is None or empty."
    )

    if final_channel["content"][-1] == EOS:
        # EOS present means the final channel is complete (AWRS replaces <|return|> with EOS).
        final_str = b"".join(final_channel["content"][:-1]).decode(
            "utf-8", errors="replace"
        )
        log_weight = await BooleanCfg.complete(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be accepted by the grammar: {final_str!r}"
        )
    else:
        # No EOS means the output was truncated by max_tokens.
        final_str = b"".join(final_channel["content"]).decode("utf-8", errors="replace")
        log_weight = await BooleanCfg.prefix(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be a valid prefix of the grammar: {final_str!r}"
        )


@pytest.mark.asyncio
async def test_logw_next_token_eos(tokenizer, wcfg):
    """Test that logw_next gives mass to the correct end-of-channel token.

    For analysis/commentary channels, mass should be on <|end|>.
    For the final channel, mass should be on EOS.
    """
    potential_vocab = HarmonyChat(tokenizer).potential_vocab
    coerced_cfg = Coerced(wcfg, potential_vocab, f=coerce_bytes_to_chars, prune=False)
    harmony_potential = HarmonyPotential(
        base_potential=coerced_cfg,
        llm_tokenizer=tokenizer,
        constrained_channels=["analysis", "final", "commentary"],
    )

    for channel in ["analysis", "final", "commentary"]:
        # "aaabaaa" is a complete string under the grammar, so the only valid
        # next token should be the channel-closing token.
        chat = "<|channel|>" + channel + "<|message|>aaabaaa"
        chat_ids = tokenizer.encode(chat, add_special_tokens=False)
        chat_bytes = harmony_potential.harmony_chat.decode_tokens(chat_ids)

        logw_next_token = await harmony_potential.logw_next(chat_bytes)
        assert logsumexp(logw_next_token.weights) == 0, (
            "The total log weight of logw_next tokens should be 0."
        )
        if channel == "final":
            assert np.isclose(logw_next_token[EOS], 0), (
                "All the mass should be on the EOS token."
            )
        else:
            assert np.isclose(logw_next_token[b"<|end|>"], 0), (
                "All the mass should be on the <|end|> token."
            )


@pytest.mark.asyncio
async def test_logw_next_token_all(tokenizer, wcfg):
    """Test that logw_next matches the naive next-prefix weight computation.

    For each non-zero-weight token, verify that the logw_next weight equals
    log(prefix_grammar(context + token)) - log(prefix_grammar(context)).
    """
    string = "aaa"
    Z = np.log(wcfg.cfg_eos.prefix_grammar(string))

    potential_vocab = HarmonyChat(tokenizer).potential_vocab
    coerced_cfg = Coerced(wcfg, potential_vocab, f=coerce_bytes_to_chars, prune=False)
    harmony_potential = HarmonyPotential(
        base_potential=coerced_cfg,
        llm_tokenizer=tokenizer,
        constrained_channels=["analysis", "final", "commentary"],
    )

    for channel in ["analysis", "final", "commentary"]:
        chat = "<|channel|>" + channel + "<|message|>" + string
        chat_ids = tokenizer.encode(chat, add_special_tokens=False)
        chat_bytes = harmony_potential.harmony_chat.decode_tokens(chat_ids)

        logw_next_token = await harmony_potential.logw_next(chat_bytes)
        non_inf_indices = np.where(logw_next_token.weights != float("-inf"))[0]

        for index in non_inf_indices:
            token_string = logw_next_token.decode[index].decode(
                "utf-8", errors="replace"
            )
            completion = string + token_string
            want = np.log(wcfg.cfg_eos.prefix_grammar(completion)) - Z
            have = logw_next_token.weights[index]
            assert np.isclose(want, have), (
                f"Token {token_string}. Want: {want}. Have: {have}"
            )


@pytest.mark.asyncio
async def test_harmony_sampling_from_product(promptedllm, tokenizer, wcfg):
    """Test sampling from the product potential of HarmonyPotential and PromptedLLM.

    This exercises the logw_next method in an end-to-end sampling pipeline.
    """
    torch.manual_seed(42)
    np.random.seed(42)

    messages = [
        {
            "role": "user",
            "content": "Please sample a string from a^nba^n where n >= 5. Please provide just the final answer.",
        }
    ]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, reasoning_effort="low"
    )
    prompt_str = tokenizer.decode(prompt_ids)
    promptedllm.set_prompt_from_str(prompt_str)

    coerced_cfg = wcfg.coerce(promptedllm, f=coerce_bytes_to_chars, prune=False)
    harmony_cfg = HarmonyPotential(coerced_cfg, promptedllm.model.tokenizer, ["final"])

    sampler = direct_token_sampler(promptedllm * harmony_cfg)
    sequences = await SMC(sampler)(
        n_particles=3, ess_threshold=0.6, max_tokens=800, verbosity=0
    )

    sequence, weight = sequences[0]
    assert weight != float("-inf"), "No valid particle sampled (weight is -inf)."

    channels = harmony_cfg.harmony_chat.extract_harmony_channels_from_tokens(sequence)
    final_channel = channels.get("final")
    assert final_channel is not None and len(final_channel["content"]) > 0, (
        "Final channel is None or empty."
    )

    if final_channel["content"][-1] == EOS:
        # EOS present means the final channel is complete (AWRS replaces <|return|> with EOS).
        final_str = b"".join(final_channel["content"][:-1]).decode(
            "utf-8", errors="replace"
        )
        log_weight = await wcfg.complete(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be accepted by the grammar: {final_str!r}"
        )
    else:
        # No EOS means the output was truncated by max_tokens.
        final_str = b"".join(final_channel["content"]).decode("utf-8", errors="replace")
        log_weight = await wcfg.prefix(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be a valid prefix of the grammar: {final_str!r}"
        )
