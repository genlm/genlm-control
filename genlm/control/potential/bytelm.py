import torch
from genlm.control.potential.base import Potential
from genlm.bytes import ByteBeamState, BeamParams
from genlm.backend import load_model_by_name

class ByteLLM(Potential):
    def __init__(self, llm, beam_params: BeamParams):
        self.llm = llm
        self.beam_params = beam_params
        vocab = [i.to_bytes(1, 'big') for i in range(256)]
        super().__init__(vocabulary=vocab)
        self._beam_cache = {}
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
                    self._beam_cache[self.prompt_bytes] = self._initial_beam
            best_beam = self._initial_beam
            best_prefix_bytes = self.prompt_bytes if full_context_bytes.startswith(self.prompt_bytes) else b""

        remaining_bytes = full_context_bytes[len(best_prefix_bytes):]
        current_beam = best_beam
        current_prefix_bytes = best_prefix_bytes
        for byte_val in remaining_bytes:
            current_beam = await (current_beam.prune() << byte_val)
            current_prefix_bytes += bytes([byte_val])
            self._beam_cache[current_prefix_bytes] = current_beam
        return current_beam

    async def prefix(self, context):
        if not context and not self.prompt_bytes:
            return 0.0
        beam = await self._get_or_create_beam_for_context(context)
        return beam.logZ

    async def complete(self, context):
        beam = await self._get_or_create_beam_for_context(context)
        logp_next = await beam.logp_next()
        eos_prob = logp_next.ps[257]  # Get EOS probability from the beam
        return beam.logZ + eos_prob

    async def logw_next(self, context):
        # This now correctly uses the default implementation in the Potential base class
        return await super().logw_next(context)

    async def cleanup(self):
        """Cleans up resources used by the beam states."""
        if self._initial_beam:
            await self._initial_beam.cleanup()
        for beam in self._beam_cache.values():
            await beam.cleanup()