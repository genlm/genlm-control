"""
Tests for HarmonyPotential channel extraction from harmony chat examples.
"""
import json
import pytest
from pathlib import Path
import time
from genlm.control.potential.coerce import Coerced
from genlm.backend import decode_vocab

from genlm.control.potential.harmony import HarmonyPotential, HarmonyChat


@pytest.fixture
def harmony_examples_path():
    """Path to the harmony chat examples JSON file."""
    test_dir = Path(__file__).parent
    return test_dir / "harmomy_chat_examples.json"


@pytest.fixture
def harmony_examples(harmony_examples_path):
    """Load harmony chat examples from JSON file."""
    with open(harmony_examples_path, 'r', encoding='utf-8') as f:
        return json.load(f)
 

@pytest.fixture
def tokenizer():
    """Get GPT-OSS tokenizer for tokenizing full responses."""
    try:
        from transformers import AutoTokenizer
        # Use the same model as in the examples (GPT-OSS)
        tokenizer = AutoTokenizer.from_pretrained("unsloth/gpt-oss-20b-BF16")
        return tokenizer
    except ImportError:
        pytest.skip("transformers not available")
    except Exception as e:
        pytest.skip(f"Could not load tokenizer: {e}")


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
        extracted_channels = harmony_chat.extract_harmony_channels_from_tokens(token_bytes)
        
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
                extracted_content = tokenizer.decode(harmony_chat.encode_tokens(extracted["content"]))
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
    harmony_chat = HarmonyChat(tokenizer = tokenizer)
    
    for sample in samples:
        sample_id = sample["sample_id"]
        full_response = sample["full_response"]
        expected_channels = sample["channels"]
        
        # Tokenize the full response
        token_ids = tokenizer.encode(full_response, add_special_tokens=False)
        token_bytes = harmony_chat.decode_tokens(token_ids)
        
        # Extract channels
        extracted_channels = harmony_chat.extract_harmony_channels_from_tokens(token_bytes)
        
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




