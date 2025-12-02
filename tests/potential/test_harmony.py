"""
Tests for HarmonyPotential channel extraction from harmony chat examples.
"""

import json
import pytest
from pathlib import Path
from genlm.grammar import CFG, Boolean, Float
from genlm.control.potential.built_in import WCFG, BoolCFG

from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.potential.harmony import HarmonyPotential, HarmonyChat
from genlm.control.sampler.token import AWRS
from genlm.control import SMC
from genlm.control.constant import EOS
import numpy as np
import torch
from arsenal.maths import logsumexp
from genlm.control import direct_token_sampler


model_name = "unsloth/gpt-oss-20b-BF16"


def coerce_bytes_to_chars(bytes_tokens):
    """Convert a sequence of bytes tokens to a list of characters."""
    byte_string = b"".join(bytes_tokens)
    # Use errors='ignore' to skip invalid UTF-8 bytes
    decoded = byte_string.decode("utf-8", errors="replace")
    return list(decoded)


@pytest.fixture(scope="function", autouse=True)
def cleanup_gpu():
    """Clear GPU cache before and after each test."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield  # Run test
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@pytest.fixture
def harmony_examples_path():
    """Path to the harmony chat examples JSON file."""
    test_dir = Path(__file__).parent
    return test_dir / "harmomy_chat_examples.json"


@pytest.fixture
def harmony_examples(harmony_examples_path):
    """Load harmony chat examples from JSON file."""
    with open(harmony_examples_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def tokenizer():
    """Get GPT-OSS tokenizer for tokenizing full responses."""
    try:
        from transformers import AutoTokenizer

        # Use the same model as in the examples (GPT-OSS)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except ImportError:
        pytest.skip("transformers not available")
    except Exception as e:
        pytest.skip(f"Could not load tokenizer: {e}")


@pytest.fixture
def promptedllm():
    return PromptedLLM.from_name(
        model_name, backend="hf"
    )  # We set the prompt with set_prompt_from_string


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
async def test_potential_evaluation(wcfg, promptedllm, tokenizer):
    """
    Tests that the harmony potential are correctly evaluated
    on a given chat and taking a wcfg as the underlying grammar
    In particular, we test the following cases: either just one of the three channels is subject to
    the constraint or all of them are subject to it.
    For each case, we test both the prefix potential and the complete one.
    """

    cfg_inputs = {"analysis": "aaaabaaaa", "commentary": "bbbbbbbbb", "final": "aaa"}

    base_cfg = wcfg.cfg
    coerced_cfg = wcfg.coerce(promptedllm, f=coerce_bytes_to_chars, prune=False)
    harmony_potential = HarmonyPotential(
        base_potential=coerced_cfg, llm_tokenizer=tokenizer
    )

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
    chat_ids = harmony_potential.harmony_chat.tokenizer.encode(
        chat
    )  # string to IDs list
    chat_bytes = harmony_potential.harmony_chat.decode_tokens(
        chat_ids
    )  # IDs list to bytes list

    # Test that each channel is correctly constrained.
    for channel in ["analysis", "final", "commentary"]:
        harmony_potential.set_constrained_channels([channel])
        assert np.isclose(
            await harmony_potential.complete(chat_bytes), tot_w(cfg_inputs[channel])
        )
        assert np.isclose(
            await harmony_potential.prefix(chat_bytes), pfx_w(cfg_inputs[channel])
        )


def test_harmony_channel_extraction(harmony_examples, tokenizer):
    """
    Test that HarmonyPotential correctly extracts channels from harmony chat examples.
    """
    samples = harmony_examples["samples"]

    harmony_chat = HarmonyChat(tokenizer)
    for sample in samples:
        sample_id = sample["sample_id"]
        full_response = sample["full_response"]
        expected_channels = sample["channels"]

        # Tokenize the full response
        token_ids = tokenizer.encode(full_response, add_special_tokens=False)
        token_bytes = harmony_chat.decode_tokens(token_ids)

        # Extract channels using HarmonyPotential
        extracted_channels = harmony_chat.extract_harmony_channels_from_tokens(
            token_bytes
        )

        # Check each channel
        for channel_name in ["analysis", "final", "commentary"]:
            expected = expected_channels.get(channel_name)
            extracted = extracted_channels.get(channel_name)

            if expected is None:
                # Expected channel is None
                assert extracted is None, (
                    f"Sample {sample_id}: Expected {channel_name} to be None, "
                    f"but got: {extracted}"
                )
            else:
                # Expected channel exists
                assert extracted is not None, (
                    f"Sample {sample_id}: Expected {channel_name} to exist, but got None"
                )

                # Check is_prefix flag
                assert extracted["is_prefix"] == expected["is_prefix"], (
                    f"Sample {sample_id}, {channel_name}: "
                    f"Expected is_prefix={expected['is_prefix']}, "
                    f"got {extracted['is_prefix']}"
                )

                # Check content matches (decode token IDs and compare)
                extracted_content = tokenizer.decode(
                    harmony_chat.encode_tokens(extracted["content"])
                )
                expected_content = expected["content"]

                assert extracted_content == expected_content, (
                    f"Sample {sample_id}, {channel_name}: Content mismatch.\n"
                    f"Expected: {expected_content[:100]}...\n"
                    f"Got: {extracted_content[:100]}..."
                )


def test_harmony_channel_extraction_token_bytes(harmony_examples, tokenizer):
    """
    Same as above, but at a byte level
    """
    samples = harmony_examples["samples"]
    harmony_chat = HarmonyChat(tokenizer=tokenizer)

    for sample in samples:
        sample_id = sample["sample_id"]
        full_response = sample["full_response"]
        expected_channels = sample["channels"]

        # Tokenize the full response
        token_ids = tokenizer.encode(full_response, add_special_tokens=False)
        token_bytes = harmony_chat.decode_tokens(token_ids)

        # Extract channels
        extracted_channels = harmony_chat.extract_harmony_channels_from_tokens(
            token_bytes
        )

        # For each expected channel, verify the extracted token IDs match
        for channel_name in ["analysis", "final", "commentary"]:
            expected = expected_channels.get(channel_name)
            extracted = extracted_channels.get(channel_name)

            if expected is None:
                assert extracted is None, (
                    f"Sample {sample_id}: {channel_name} should be None"
                )
            else:
                assert extracted is not None, (
                    f"Sample {sample_id}: {channel_name} should not be None"
                )

                # Compare token IDs directly
                expected_token_bytes = harmony_chat.decode_tokens(expected["token_ids"])
                extracted_token_bytes = extracted["content"]

                assert extracted_token_bytes == expected_token_bytes, (
                    f"Sample {sample_id}, {channel_name}: Token ID mismatch.\n"
                    f"Expected length: {len(expected_token_bytes)}, "
                    f"Got length: {len(extracted_token_bytes)}\n"
                    f"First 20 expected: {expected_token_bytes[:20]}\n"
                    f"First 20 got: {extracted_token_bytes[:20]}"
                )


@pytest.mark.asyncio
async def test_harmony_awrs_constrained_sampling(promptedllm, tokenizer, BooleanCfg):
    """Test HarmonyPotential with AWRS and SMC for constrained generation."""
    # Skip if model or tokenizer not available
    if promptedllm is None or tokenizer is None:
        pytest.skip("Model or tokenizer not available")

    # Setup prompt using chat template
    messages = [
        {
            "role": "user",
            "content": "Please sample a string from a^nb^n where n >= 5. Please provide just the final answer.",
        }
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_str = tokenizer.decode(prompt_ids)
    print(prompt_str)
    promptedllm.set_prompt_from_str(prompt_str)

    # Create harmony potential with constrained final channel
    coerced_cfg = BooleanCfg.coerce(promptedllm, f=coerce_bytes_to_chars, prune=False)
    harmony_cfg = HarmonyPotential(coerced_cfg, promptedllm.model.tokenizer, ["final"])

    # Run SMC sampling with CUDA error handling
    try:
        sampler = AWRS(
            potential=promptedllm,
            condition=harmony_cfg,
            max_accepts=2,
            max_rejects=1000,
            prune_logws=False,
        )
        sequences = await SMC(sampler)(
            n_particles=1, ess_threshold=0.6, max_tokens=500, verbosity=0
        )
    except Exception as e:
        # Catch torch.OutOfMemoryError and other CUDA-related exceptions
        error_msg = str(e).lower()
        error_type = type(e).__name__.lower()
        if (
            "outofmemory" in error_type
            or "cuda" in error_msg
            or "out of memory" in error_msg
        ):
            pytest.skip(f"CUDA/GPU memory error: {e}")
        raise

    # Get the single sequence (n_particles=1)
    sequence, weight = sequences[0]
    if weight == float("-inf"):  # We skip  if the returned particle has -inf weight
        pytest.skip("Weight is -inf. No valid particle sampled.")
    channels = harmony_cfg.harmony_chat.extract_harmony_channels_from_tokens(sequence)
    final_channel = channels.get("final")

    # Skip if final channel is None
    if final_channel is None or len(final_channel["content"]) == 0:
        pytest.skip("Final channel is None or empty")

    if (
        final_channel["content"][-1] == EOS
    ):  # Remove EOS if present. this also implies that we can return the final channel as complete (note that awrs automatically replaces teh built in <|return|> character with EOS).
        final_str = b"".join(final_channel["content"][:-1]).decode(
            "utf-8", errors="replace"
        )
        log_weight = await BooleanCfg.complete(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be accepted by the grammar: {final_str!r}"
        )
    else:  # If EOS is not the last token, it means that we can treat the final channel as prefix. CHECK: should this be treated as an error? This is only correct thet the output was truncated due to the max_tokens limit.
        final_str = b"".join(final_channel["content"]).decode("utf-8", errors="replace")
        log_weight = await BooleanCfg.prefix(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be a valid prefix of the grammar: {final_str!r}"
        )

@pytest.mark.asyncio
async def test_logw_next_token_END(promptedllm, tokenizer, wcfg):
    """This method tests that the logw_next method is correctly implemented. in the spcial cases where the mass is concentrated on the end of strinfg token"""

    coerced_cfg = wcfg.coerce(promptedllm, f=coerce_bytes_to_chars, prune=False)
    harmony_potential = HarmonyPotential(
            base_potential=coerced_cfg, llm_tokenizer=tokenizer, constrained_channels=["analysis", "final", "commentary"]
        )

    for channel in ["analysis", "final", "commentary"]:
        chat_complete_final = ( "<|channel|>"+channel+"<|message|>aaabaaa" ) # Here the only valid next-token should be EOS (because of the internal token substitution.)

        chat_complete_final_ids = tokenizer.encode(chat_complete_final, add_special_tokens=False) # string to token ids
        chat_complete_final_bytes = harmony_potential.harmony_chat.decode_tokens(chat_complete_final_ids) # token ids to bytes # todo --> this naming is ugly and should be fixed 
        
        logw_next_token = await harmony_potential.logw_next(chat_complete_final_bytes)
        assert logsumexp(logw_next_token.weights) == 0, "The total logprob of the logw_next tokens should be 0"
        if channel == "final":
            assert np.isclose(logw_next_token[EOS], 0), "All the mass should be on the EOS token"
        else:
            assert np.isclose(logw_next_token[b"<|end|>"],0), "All the mass should be on the <|end|> token"

@pytest.mark.asyncio
async def test_logw_next_token_all(promptedllm, tokenizer, wcfg):
    """This method tests that the logw_next method is correctly implemented. In particular,
    we check that the next token weights matches what we would compute with the naive next prefix weight. """
    string = "aaa"
    Z = np.log(wcfg.cfg_eos.prefix_grammar(string)) # compute the common normalizing constant 

    coerced_cfg = wcfg.coerce(promptedllm, f=coerce_bytes_to_chars, prune=False)
    harmony_potential = HarmonyPotential(
            base_potential=coerced_cfg, llm_tokenizer=tokenizer, constrained_channels=["analysis", "final", "commentary"]
        )

    for channel in ["analysis", "final", "commentary"]:
        chat_complete_final = ( "<|channel|>"+channel+"<|message|>"+string) # Here the only valid next-token should be EOS (because of teh internal token substittion.)
        chat_ids = tokenizer.encode(chat_complete_final, add_special_tokens=False)
        chat_bytes = harmony_potential.harmony_chat.decode_tokens(chat_ids)

        logw_next_token = await harmony_potential.logw_next(chat_bytes)
        non_inf_indices = np.where(logw_next_token.weights != float("-inf"))[0]

        for index in non_inf_indices:
            token_string = logw_next_token.decode[index].decode("utf-8", errors="replace")
            completion = string + token_string
            want = np.log(wcfg.cfg_eos.prefix_grammar(completion)) - Z # compute the normalized weights  
            have = logw_next_token.weights[index]
            assert np.isclose(want, have), f"Token {token_string}. Want : {want}. Have : {have}"


@pytest.mark.asyncio
async def test_harmony_sampling_from_product(promptedllm, tokenizer, wcfg):
    """ Tests sampling from the product potential of HarmonyPotential and the PromptedLLM. Importantly,
    this tests the logic of th elogw_next method """
    # Skip if model or tokenizer not available
    if promptedllm is None or tokenizer is None:
        pytest.skip("Model or tokenizer not available")

    # Setup prompt using chat template
    messages = [
        {
            "role": "user",
            "content": "Please sample a string from a^nba^n where n >= 5. Please provide just the final answer.",
        }
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_str = tokenizer.decode(prompt_ids)
    print(prompt_str)
    promptedllm.set_prompt_from_str(prompt_str)

    # Create harmony potential with constrained final channel
    coerced_cfg = wcfg.coerce(promptedllm, f=coerce_bytes_to_chars, prune=False)
    harmony_cfg = HarmonyPotential(coerced_cfg, promptedllm.model.tokenizer, ["final"])

    # Run SMC sampling with CUDA error handling
    try:
        sampler = direct_token_sampler( promptedllm * harmony_cfg)
        sequences = await SMC(sampler)(
            n_particles=1, ess_threshold=0.6, max_tokens=500, verbosity=0
        )
    except Exception as e:
        # Catch torch.OutOfMemoryError and other CUDA-related exceptions
        error_msg = str(e).lower()
        error_type = type(e).__name__.lower()
        if (
            "outofmemory" in error_type
            or "cuda" in error_msg
            or "out of memory" in error_msg
        ):
            pytest.skip(f"CUDA/GPU memory error: {e}")
        raise

    # Get the single sequence (n_particles=1)
    sequence, weight = sequences[0]
    if weight == float("-inf"):  # We skip  if the returned particle has -inf weight
        pytest.skip("Weight is -inf. No valid particle sampled.")
    channels = harmony_cfg.harmony_chat.extract_harmony_channels_from_tokens(sequence)
    final_channel = channels.get("final")

    # Skip if final channel is None
    if final_channel is None or len(final_channel["content"]) == 0:
        pytest.skip("Final channel is None or empty")

    if (
        final_channel["content"][-1] == EOS
    ):  # Remove EOS if present. this also implies that we can return the final channel as complete (note that awrs automatically replaces teh built in <|return|> character with EOS).
        final_str = b"".join(final_channel["content"][:-1]).decode(
            "utf-8", errors="replace"
        )
        log_weight = await wcfg.complete(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be accepted by the grammar: {final_str!r}"
        )
    else:  # If EOS is not the last token, it means that we can treat the final channel as prefix. CHECK: should this be treated as an error? This is only correct thet the output was truncated due to the max_tokens limit.
        final_str = b"".join(final_channel["content"]).decode("utf-8", errors="replace")
        log_weight = await wcfg.prefix(final_str)
        assert log_weight != float("-inf"), (
            f"The generated final channel should be a valid prefix of the grammar: {final_str!r}"
        )