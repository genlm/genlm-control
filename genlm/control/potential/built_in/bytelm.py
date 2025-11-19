"""
Preliminary integration from https://github.com/genlm/genlm-control/blob/yahya010/bytelm/genlm/control/potential/built_in/bytelm.py
"""
import torch
import numpy as np
from genlm.control.potential.base import Potential
from genlm.bytes import ByteBeamState, BeamParams
from genlm.backend import load_model_by_name
from collections import OrderedDict

class ByteLLM(Potential):
    def __init__(self, llm, beam_params: BeamParams, cache_size: int = 1024):
        self.llm = llm
        self.beam_params = beam_params
        self.cache_size = cache_size
        vocab = [i.to_bytes(1, 'big') for i in range(256)]
        super().__init__(vocabulary=vocab)
        # LRU cache of ByteBeamState keyed by full context bytes (prompt + context)
        self._beam_cache: OrderedDict[bytes, ByteBeamState] = OrderedDict()
        self._initial_beam = None
        self.prompt_bytes = b""

    @classmethod
    def from_name(cls, name, beam_params: BeamParams, backend=None, **kwargs):
        backend = backend or ("vllm" if torch.cuda.is_available() else "hf")
        llm = load_model_by_name(name, backend=backend, **kwargs)
        return cls(llm, beam_params)

    def set_prompt_from_str(self, prompt_str: str):
        new_prompt_bytes = prompt_str.encode("utf-8")
        if new_prompt_bytes != self.prompt_bytes:
            self.prompt_bytes = new_prompt_bytes
            self._beam_cache.clear()
            self._initial_beam = None

    async def _get_or_create_beam_for_context(self, context):
        context_bytes = b''.join(context)
        full_context_bytes = self.prompt_bytes + context_bytes
        if full_context_bytes in self._beam_cache:
            # Mark as recently used
            self._beam_cache.move_to_end(full_context_bytes)
            return self._beam_cache[full_context_bytes]

        best_prefix_bytes = b""
        best_beam = None
        for cached_prefix_bytes, cached_beam in self._beam_cache.items():
            if full_context_bytes.startswith(cached_prefix_bytes) and len(cached_prefix_bytes) > len(best_prefix_bytes):
                best_prefix_bytes = cached_prefix_bytes
                best_beam = cached_beam

        if best_beam is None:
            if self._initial_beam is None:
                self._initial_beam = await ByteBeamState.initial(self.llm, self.beam_params)
                if self.prompt_bytes:
                    self._initial_beam = await self._initial_beam.prefill(self.prompt_bytes)
                    self._cache_put(self.prompt_bytes, self._initial_beam)
            best_beam = self._initial_beam
            best_prefix_bytes = self.prompt_bytes if full_context_bytes.startswith(self.prompt_bytes) else b""

        remaining_bytes = full_context_bytes[len(best_prefix_bytes):]
        current_beam = best_beam
        current_prefix_bytes = best_prefix_bytes
        for byte_val in remaining_bytes:
            current_beam = await (current_beam.prune() << byte_val)
            current_prefix_bytes += bytes([byte_val])
            self._cache_put(current_prefix_bytes, current_beam)
        return current_beam

    def _cache_put(self, key: bytes, beam: ByteBeamState):
        self._beam_cache[key] = beam
        self._beam_cache.move_to_end(key)
        while len(self._beam_cache) > self.cache_size:
            # Evict least-recently used
            self._beam_cache.popitem(last=False)

    async def prefix(self, context):
        # Treat empty context as neutral (log 1 = 0), matching PromptedLLM semantics.
        # The prompt, if set, is incorporated into next-token distributions via the cached beam,
        # but does not contribute to the prefix weight of the empty context.
        if not context:
            return 0.0
        beam = await self._get_or_create_beam_for_context(context)
        base = self._initial_beam.logZ if self._initial_beam is not None else 0.0
        return beam.logZ - base

    async def complete(self, context):
        beam = await self._get_or_create_beam_for_context(context)
        logp_next = await beam.logp_next()
        # Assume logp_next.ps contains log-probs for 256 byte values plus EOS at the end.
        eos_logp = logp_next.ps[-1]
        base = self._initial_beam.logZ if self._initial_beam is not None else 0.0
        return (beam.logZ - base) + eos_logp

    async def logw_next(self, context):
        """Efficient next-token weights using the cached beam state.

        Uses the beam's next-token distribution directly instead of the
        default (slower) fallback that recomputes scores for each token.
        """
        beam = await self._get_or_create_beam_for_context(context)
        logp_next = await beam.logp_next()

        # Build weights over vocab_eos (256 bytes + EOS at the end)
        ps = np.asarray(logp_next.ps)
        logws = self.alloc_logws()
        v = len(self.vocab)
        logws[:v] = ps[:v]
        logws[-1] = ps[-1]
        return self.make_lazy_weights(logws)

    async def cleanup(self):
        """Cleans up resources used by the beam states."""
        if self._initial_beam:
            await self._initial_beam.cleanup()
        for beam in self._beam_cache.values():
            await beam.cleanup()