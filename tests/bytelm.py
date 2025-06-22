import pytest
import torch
import numpy as np
import asyncio
from typing import Dict, Optional

from genlm.control import Potential, direct_token_sampler, BoolFSA, AWRS
from genlm.backend import load_model_by_name
from genlm.bytes import ByteBeamState, BeamParams
from genlm.bytes.trie import AsyncTokenByteTrie
from genlm.backend.tokenization.bytes import get_byte_vocab
from genlm.bytes.byte_lm import LazyTrieState


class StatefulByteLevelLLM(Potential):
    """
    A stateful byte-level language model potential using beam summing algorithm.
    
    This class maintains beam states across calls and extends them incrementally,
    which is much more efficient than recreating beams from scratch.
    
    EOS bytes (like \n) are individually samplable ONLY when at the root (after completing 
    tokens via EOT) and trigger termination when sampled. EOS bytes are prevented from 
    being sampled mid-token. The termination slot contains only token-level EOS probability.
    """

    def __init__(self, llm, beam_params: BeamParams, eos_tokens=None, temperature=1.0):
        self.model = llm
        self.beam_params = beam_params
        self.temperature = temperature
        
        default_eos = llm.byte_vocab[self.model.tokenizer.eos_token_id]
        self._eos_tokens = [default_eos]
        
        # Add custom EOS tokens
        if eos_tokens:
            for eos_token in eos_tokens:
                if eos_token not in self._eos_tokens:
                    self._eos_tokens.append(eos_token)
            
        assert len(set(self._eos_tokens)) == len(self._eos_tokens), "duplicate eos tokens"
        
        # Store EOS token IDs for the underlying token model (only for default tokenizer EOS)
        self._eos_token_ids = []
        for token_id, byte_token in enumerate(llm.byte_vocab):
            if byte_token == default_eos:
                self._eos_token_ids.append(token_id)
                break
        
        self.prompt_bytes = b""
        self.trie = AsyncTokenByteTrie.from_vocab(get_byte_vocab(self.model.tokenizer))
        
        # cache beam states for different context prefixes
        self._beam_cache: Dict[bytes, ByteBeamState] = {}
        self._initial_beam: Optional[ByteBeamState] = None
        
        # The vocabulary should include ALL bytes (including EOS bytes, design decision will be changed once genlm.bytes PR is finished)
        vocab_bytes = [i.to_bytes(1, 'big') for i in range(256)]  # All possible bytes
        
        super().__init__(vocabulary=vocab_bytes)

    @property
    def eos_tokens(self):
        return self._eos_tokens

    def is_eos_token(self, token):
        """Check if a token is an EOS token (for termination logic)."""
        return token in self._eos_tokens

    def set_prompt_from_str(self, prompt_str: str):
        """Sets the internal prompt from a string and clears beam cache."""
        self.prompt_bytes = prompt_str.encode("utf-8")
        # clear cache when prompt changes
        self._beam_cache.clear()
        self._initial_beam = None

    @classmethod
    def from_name(
        cls,
        name,
        beam_params: BeamParams,
        backend=None,
        eos_tokens=None,
        temperature=1.0,
        **kwargs,
    ):
        backend = backend or ("vllm" if torch.cuda.is_available() else "hf")
        token_level_model = load_model_by_name(name, backend=backend, **kwargs)
        return cls(
            token_level_model,
            beam_params=beam_params,
            eos_tokens=eos_tokens,
            temperature=temperature,
        )

    async def _get_or_create_beam_for_context(self, context_bytes: bytes) -> ByteBeamState:
        """
        Get or create a beam state for the given context.
        Uses caching and incremental extension to avoid recomputation.
        """
        # Check if we have this exact context cached, if cached then O(1)
        if context_bytes in self._beam_cache:
            return self._beam_cache[context_bytes]
        
        # Find the longest cached prefix, O(n) n = chache size
        best_prefix = b""
        best_beam = None
        
        for cached_prefix, cached_beam in self._beam_cache.items():
            if context_bytes.startswith(cached_prefix) and len(cached_prefix) > len(best_prefix):
                best_prefix = cached_prefix
                best_beam = cached_beam
        
        # If no cached prefix found, start from initial beam
        if best_beam is None:
            if self._initial_beam is None:
                initial_lazy_state = LazyTrieState.initial(self.model, self.trie)
                self._initial_beam = ByteBeamState([await initial_lazy_state.materialize()], self.beam_params)
                
                # If we have a prompt, prefill it once
                if self.prompt_bytes:
                    self._initial_beam = await self._initial_beam.prefill(self.prompt_bytes)
            
            best_beam = self._initial_beam
            best_prefix = self.prompt_bytes if context_bytes.startswith(self.prompt_bytes) else b""
        
        # Extend from the best cached state to the target context
        remaining_bytes = context_bytes[len(best_prefix):]
        current_beam = best_beam
        current_prefix = best_prefix
        
        # Extend byte by byte and cache intermediate states, O(k) k = remaining bytes
        for byte_val in remaining_bytes:
            current_beam = await (current_beam.prune() << byte_val)
            current_prefix += bytes([byte_val])
            
            # Cache this intermediate state
            self._beam_cache[current_prefix] = current_beam
        
        return current_beam

    def _get_eos_logp_from_token_model(self, token_logprobs):
        """Extract and aggregate EOS probabilities from token model logprobs."""
        if len(self._eos_token_ids) == 1:
            return token_logprobs[self._eos_token_ids[0]].item()
        else:
            eos_logprobs = token_logprobs[self._eos_token_ids]
            return torch.logsumexp(eos_logprobs, dim=0).item()

    async def logw_next(self, context):
        """
        Compute log weights for next tokens using cached beam states.
        """
        full_context_bytes = self.prompt_bytes + b"".join(context)
        
        # Get beam state for this context
        beam = await self._get_or_create_beam_for_context(full_context_bytes)
        
        if beam.logZ == float('-inf'):
            log_weights = np.full((len(self.vocab) + 1,), float('-inf'), dtype=np.float32)
            return self.make_lazy_weights(log_weights)
        
        # Get the next byte distribution from the beam
        logp_next = await beam.logp_next()
        
        # Initialize log weights array: vocab + EOS
        log_weights = np.full((len(self.vocab) + 1,), float('-inf'), dtype=np.float32)  # Shape: [257] (bytes 0-255 + EOT)

        # Map byte probabilities to vocab indices
        # Only allow EOS bytes when we're at the root (after completing tokens via EOT)
        at_root = any(state.node == state.root for state in beam.states)
        
        for i, byte_val in enumerate(self.vocab):
            byte_int = int.from_bytes(byte_val, 'big')
            # Check if this byte is an EOS token
            if byte_val in self._eos_tokens:
                if at_root:
                    # Only allow EOS bytes when at root (after completing tokens)
                    log_weights[i] = logp_next.ps[byte_int]
                else:
                    # Prevent EOS bytes mid-token
                    log_weights[i] = float('-inf')
            else:
                # Regular bytes are always allowed if available in beam
                log_weights[i] = logp_next.ps[byte_int]
        
        # Only use token-level EOS probability for termination slot
        if full_context_bytes:
            context_str = full_context_bytes.decode('utf-8', errors='replace')
            token_ids = self.model.tokenizer.encode(context_str)
        else:
            token_ids = []
        
        token_logprobs = await self.model.next_token_logprobs(token_ids)
        token_eos_logp = self._get_eos_logp_from_token_model(token_logprobs)
        
        log_weights[-1] = token_eos_logp

        if self.temperature != 1.0:
            log_weights /= self.temperature

        finite_mask = np.isfinite(log_weights)
        if np.any(finite_mask):
            log_sum_exp = np.logaddexp.reduce(log_weights[finite_mask])
            normalized_log_probs = log_weights - log_sum_exp
        else:
            normalized_log_probs = log_weights

        return self.make_lazy_weights(normalized_log_probs)

    async def prefix(self, context):
        """Compute the log probability of the context as a prefix using cached beam states."""
        if not context and not self.prompt_bytes:
            return 0.0
        
        full_context_bytes = self.prompt_bytes + b"".join(context)
        beam = await self._get_or_create_beam_for_context(full_context_bytes)
        
        if beam.logZ == float('-inf') or len(beam.states) == 0:
            raise ValueError(f"Beam failed for context (empty beam): {full_context_bytes}")
        
        return beam.logZ

    async def log_probability(self, context):
        """Compute the log probability of `context` given the prompt."""
        return await self.prefix(context)
    
    async def complete(self, context):
        """Compute the log probability of context as a complete sequence."""
        # Check if context incorrectly includes EOS at the end and remove it so we don't double count EOS
        if context and context[-1] == self.eos:
            print(f"WARNING: complete() called with EOS in context, removing it")
            context = context[:-1]
        
        if not context and not self.prompt_bytes:
            # Empty context - get EOS probability from token-level model
            token_logprobs = await self.model.next_token_logprobs([])
            eos_logp = self._get_eos_logp_from_token_model(token_logprobs)
            return eos_logp
        
        # Check that all context items are valid bytes (should never be triggered since we only allow EOS after an EOT, but just for safety)
        for i, item in enumerate(context):
            if item == self.eos:
                print(f"ERROR: Found EOS token at position {i} in context: {context}")
                return float("-inf")
            if not isinstance(item, bytes) or len(item) != 1:
                print(f"ERROR: Invalid context item at position {i}: {item} (type: {type(item)})")
                return float("-inf")
        
        # Get the prefix probability
        prefix_logp = await self.prefix(context)
        
        if prefix_logp == float("-inf"):
            return float("-inf")
        
        # Get the combined EOS probability (only token-level EOS for termination)
        full_context_bytes = self.prompt_bytes + b"".join(context)
        
        # Only use token-level EOS probability for termination
        if full_context_bytes:
            token_ids = self.model.tokenizer.encode(full_context_bytes.decode('utf-8', errors='replace'))
        else:
            token_ids = []
        
        token_logprobs = await self.model.next_token_logprobs(token_ids)
        token_eos_logp = self._get_eos_logp_from_token_model(token_logprobs)
        
        return prefix_logp + token_eos_logp

    def spawn(self):
        """Spawn a new StatefulByteLevelLLM with the same configuration but separate cache."""
        new_instance = StatefulByteLevelLLM(
            self.model,
            beam_params=self.beam_params,
            eos_tokens=self._eos_tokens.copy(),
            temperature=self.temperature,
        )
        new_instance.prompt_bytes = self.prompt_bytes
        return new_instance

    def spawn_new_eos(self, eos_tokens):
        """Create a new StatefulByteLevelLLM with additional EOS tokens (default EOS always included)."""
        new_instance = StatefulByteLevelLLM(
            self.model,
            beam_params=self.beam_params,
            eos_tokens=eos_tokens,  # added to default EOS, doesnt replace it
            temperature=self.temperature,
        )
        new_instance.prompt_bytes = self.prompt_bytes
        return new_instance

    async def cleanup(self):
        """Clean up resources including cached beam states."""
        # Clean up all cached beams
        for beam in self._beam_cache.values():
            await beam.cleanup()
        
        if self._initial_beam:
            await self._initial_beam.cleanup()
        
        if hasattr(self, 'trie'):
            await self.trie.cleanup()
        
        # Clear caches
        self._beam_cache.clear()
        self._initial_beam = None


@pytest.mark.asyncio
async def test_byte_level_llm_smc():
    """
    Tests basic SMC sampling with byte-level generation.
    """
    llm = None
    try:
        beam_params = BeamParams(K=10)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
        llm.set_prompt_from_str("The weather today")
        sampler = direct_token_sampler(llm)

        sequences = await sampler.smc(
            n_particles=3,
            max_tokens=20,
            ess_threshold=0.5,
            verbosity=0
        )
        
        assert len(sequences.contexts) > 0, "Should generate at least one sequence"
        assert len(sequences.contexts[0]) > 0, "Sequences should have content"
        
        # Check that sequences are byte sequences
        for context in sequences.contexts:
            for token in context:
                if token != llm.eos:  # Skip EOS token
                    assert isinstance(token, bytes), f"Expected bytes, got {type(token)}"
                    assert len(token) == 1, f"Expected single byte, got {len(token)} bytes"

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_byte_level_llm_with_prompt():
    """
    Tests prompt handling and that generation continues from prompt.
    """
    llm = None
    try:
        beam_params = BeamParams(K=10)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")

        prompt = "Hello world"
        llm.set_prompt_from_str(prompt)
        
        # Test that prompt is stored correctly
        assert llm.prompt_bytes == prompt.encode("utf-8")
        
        # Test generation continues from prompt
        sampler = direct_token_sampler(llm)
        sequences = await sampler.smc(
            n_particles=2,
            max_tokens=10,
            ess_threshold=0.5,
            verbosity=0,
        )
        
        assert len(sequences.contexts) > 0, "Should generate sequences"

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_byte_level_llm_with_fsa():
    """
    Tests constrained generation with FSA - the main use case.
    """
    llm = None
    try:
        beam_params = BeamParams(K=5, prune_threshold=0.05)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")

        prompt = "The answer is"
        llm.set_prompt_from_str(prompt)
        
        # Simple regex constraint
        fsa = BoolFSA.from_regex(r" (yes|no)")
        coerced_fsa = fsa.coerce(llm, f=b"".join)
        token_sampler = AWRS(llm, coerced_fsa)

        sequences = await token_sampler.smc(
            n_particles=3,
            max_tokens=20,
            ess_threshold=0.5,
            verbosity=0,
        )
        
        assert len(sequences) > 0, "Should generate constrained sequences"
        print(sequences.decoded_posterior)
        
        for context, weight in sequences:
            # manual decoding to check constraint, filter out EOS tokens before joining, since EOS is not a byte
            context_bytes = [token for token in context if token != llm.eos]
            generated_bytes = b"".join(context_bytes)
            full_text = prompt.encode() + generated_bytes
            decoded = full_text.decode('utf-8', errors='replace')
            print(f"Generated: '{decoded}'")
            # Should contain either "yes" or "no" after "The answer is"
            assert " yes" in decoded or " no" in decoded, f"Constraint not satisfied: {decoded}"

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_byte_level_llm_with_fsa2():
    """
    Tests that the StatefulByteLevelLLM can be used with FSA constraints.
    """
    llm = None
    try:
        beam_params = BeamParams(K=5, prune_threshold=0.05)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")

        prompt = "Here is my honest opinion:"
        llm.set_prompt_from_str(prompt)
        fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")
        coerced_fsa = fsa.coerce(llm, f=b"".join)
        token_sampler = AWRS(llm, coerced_fsa)


        sequences = await token_sampler.smc(
            n_particles=5,
            max_tokens=30,
            ess_threshold=0.5,
            verbosity=1,
        )
        
        print(f"\n=== DEBUGGING SMC RESULTS ===")
        print(f"sequences type: {type(sequences)}")
        print(f"sequences length: {len(sequences)}")
        print(f"sequences.posterior: {sequences.posterior}")
        
        for i, (context, weight) in enumerate(sequences):
            print(f"Particle {i}:")
            print(f"  Context: {context}")
            print(f"  Weight: {weight}")
            print(f"  Weight type: {type(weight)}")
            print(f"  Weight is finite: {np.isfinite(weight) if isinstance(weight, (int, float)) else 'N/A'}")
        print(sequences.decoded_posterior)

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_potential_contract():
    """
    Tests that the Potential contract is satisfied:
    logw_next(token | context) ≈ score(context + token) - prefix(context)
    """
    llm = None
    try:
        beam_params = BeamParams(K=10)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
        
        llm.set_prompt_from_str("Hello")
        context = [b' ', b'w', b'o', b'r', b'l', b'd']
        
        # Get probabilities
        logw_next = await llm.logw_next(context)
        prefix_logp = await llm.prefix(context)
        
        # Test a few tokens
        test_tokens = [b'!', b'.', b' ']
        max_diff = 0.0
        
        for token in test_tokens:
            extended_context = context + [token]
            score_extended = await llm.prefix(extended_context)
            
            # Find token in vocabulary
            token_idx = llm.lookup.get(token, -1)
            if token_idx >= 0:
                logw_token = logw_next.weights[token_idx]
            else:
                continue  # Skip if token not in vocab
            
            # Check contract: logw_next ≈ score_extended - prefix
            expected = score_extended - prefix_logp
            diff = abs(logw_token - expected)
            max_diff = max(max_diff, diff)
        
        # Allow some tolerance for beam summing approximation
        tolerance = 0.1
        assert max_diff < tolerance, f"Potential contract violated. Max diff: {max_diff:.6f} > {tolerance}"

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_eos_functionality():
    """
    Tests basic EOS token functionality and recognition.
    """
    llm = None
    try:
        beam_params = BeamParams(K=10)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
        
        # default EOS token recognition
        default_eos = llm.eos_tokens[0]  # Should have at least the default EOS
        assert llm.is_eos_token(default_eos), "Should recognize default EOS token"
        assert not llm.is_eos_token(b'a'), "Should not recognize regular byte as EOS"
        
        # custom EOS tokens
        newline_eos = b'\n'
        custom_llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf", 
                                                   eos_tokens=[newline_eos])
        
        # Should have both default and custom EOS
        assert len(custom_llm.eos_tokens) >= 2, "Should have default + custom EOS tokens"
        assert custom_llm.is_eos_token(newline_eos), "Should recognize custom EOS token"
        assert custom_llm.is_eos_token(default_eos), "Should still recognize default EOS token"
        
        # EOS termination slot exists
        custom_llm.set_prompt_from_str("Hello")
        context = [b' ', b'w', b'o', b'r', b'l', b'd']
        logw_next = await custom_llm.logw_next(context)
        
        # Should have termination slot (last element)
        assert len(logw_next.weights) == len(custom_llm.vocab) + 1, "Should have termination slot"
        termination_weight = logw_next.weights[-1]
        assert np.isfinite(float(termination_weight)), "Termination weight should be finite"
        
        await custom_llm.cleanup()

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_beam_caching():
    """
    Tests that beam states are properly cached and reused.
    """
    llm = None
    try:
        beam_params = BeamParams(K=10)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
        llm.set_prompt_from_str("Test")
        
        # first call should create and cache beam
        context1 = [b'i', b'n', b'g']
        beam1 = await llm._get_or_create_beam_for_context(llm.prompt_bytes + b"".join(context1))
        
        # Check cache has intermediate states
        assert len(llm._beam_cache) > 0, "Cache should contain beam states"
        
        # Second call to same context should reuse cache
        beam2 = await llm._get_or_create_beam_for_context(llm.prompt_bytes + b"".join(context1))
        assert beam1 is beam2, "Should reuse cached beam state"
        
        # Extension should reuse prefix
        context2 = [b'i', b'n', b'g', b' ', b's']
        beam3 = await llm._get_or_create_beam_for_context(llm.prompt_bytes + b"".join(context2))
        
        # Should have cached the prefix
        prefix_key = llm.prompt_bytes + b"ing"
        assert prefix_key in llm._beam_cache, "Should cache intermediate prefix"

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_basic_methods():
    """
    Tests individual methods in isolation.
    """
    llm = None
    try:
        beam_params = BeamParams(K=10)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
        llm.set_prompt_from_str("Hi")
        
        context = [b' ', b't', b'h', b'e', b'r', b'e']
        
        # Test prefix()
        prefix_logp = await llm.prefix(context)
        assert isinstance(prefix_logp, float), "prefix() should return float"
        assert prefix_logp < 0, "Log probability should be negative"
        
        # Test logw_next()
        logw_next = await llm.logw_next(context)
        assert hasattr(logw_next, 'weights'), "logw_next should return LazyWeights"
        assert len(logw_next.weights) == len(llm.vocab) + 1, "Should have weights for vocab + EOS"
        
        # Test complete()
        complete_logp = await llm.complete(context)
        assert isinstance(complete_logp, float), "complete() should return float"
        assert complete_logp < prefix_logp, "Complete should be less likely than prefix"
        
        # Test spawn()
        spawned = llm.spawn()
        assert spawned is not llm, "spawn() should create new instance"
        assert spawned.prompt_bytes == llm.prompt_bytes, "spawn() should copy prompt"
        assert spawned._beam_cache != llm._beam_cache, "spawn() should have separate cache"
        
        await spawned.cleanup()

    finally:
        await llm.cleanup()


@pytest.mark.asyncio
async def test_error_handling():
    """
    Tests error handling and edge cases.
    """
    llm = None
    try:
        beam_params = BeamParams(K=10)
        llm = StatefulByteLevelLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
        
        # Test complete() with invalid context
        invalid_context = [b'a', "not_bytes", b'c']  # Mix of bytes and string
        try:
            result = await llm.complete(invalid_context)
            assert result == float('-inf'), "Should return -inf for invalid context"
        except Exception:
            pass  # Also fine to raise exception
        
        # test with empty context and no prompt
        llm.set_prompt_from_str("")
        try:
            empty_result = await llm.complete([])
            assert isinstance(empty_result, float), "Should handle empty context"
        except ValueError as e:
            # also fine to reject empty token sequences
            assert "empty" in str(e).lower(), f"Expected empty-related error, got: {e}"
        
        # Test that cleanup doesn't crash
        await llm.cleanup()
        await llm.cleanup()  # call cleanup twice is fine

    finally:
        await llm.cleanup()