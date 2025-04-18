from genlm.control import Canonical
from transformers import AutoTokenizer
from genlm.control.constant import EOS
import pytest
import numpy as np


@pytest.fixture
def tokenizer():
    """Create a GPT-2 tokenizer for testing"""
    return AutoTokenizer.from_pretrained('gpt2', use_fast=False)


@pytest.fixture
def canonical_potential(tokenizer):
    """Create a CanonicalBPEPotential for testing"""
    return Canonical(tokenizer, "gpt2")


@pytest.mark.asyncio
async def test_init(tokenizer):
    """Test that the potential initializes properly"""
    potential = Canonical(tokenizer, "gpt2")
    
    # Check that the potential has the correct vocabulary
    assert len(potential.vocab) == len(potential.canonicality_filter._decode)
    
    # Check that EOS is added correctly
    assert len(potential.vocab_eos) == len(potential.vocab) + 1

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
    tokens = [b'To', b'ken', b'ization']
    log_weight = await canonical_potential.complete(tokens)
    assert log_weight == float("-inf")

@pytest.mark.asyncio
async def test_logw_next(canonical_potential):
    """Test logw_next method with non canonical context. should only extend to EOS"""
    tokens = [b'To', b'ken']
    logw = await canonical_potential.logw_next(tokens)
    assert logw[b'ization'] == float('-inf')
    assert logw[EOS] == 0.0

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
    "Artificial intelligence and machine learning"
    ]
    for sentence in sentences:
        tokens = canonical_potential.tokenizer.encode(sentence, add_special_tokens=False)
        token_bytes = [canonical_potential.tokenizer.decode([token]).encode('utf-8') for token in tokens]
        
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
            next_token = token_bytes[i] if i < len(token_bytes) else canonical_potential.eos
            print(prefix)
            print(next_token)
            lazy_weights = await canonical_potential.logw_next(prefix)
            
            # The next token in the sequence should be allowed
            token_idx = lazy_weights.encode.get(next_token)
            if token_idx is not None:
                assert not np.isneginf(lazy_weights.weights[token_idx])


if __name__ == "__main__":
    pytest.main()
