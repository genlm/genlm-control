
import asyncio
import numpy as np
import syllables

from genlm.control import AWRS, PromptedLLM, ByteLLM
from genlm.control.constant import EndOfSequence
from genlm.control.potential.base import Potential
from genlm.bytes import BeamParams

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWrite a haiku about nature.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SYLLABLE_PATTERN = [5, 7, 5]
N_PARTICLES = 10
MAX_TOKENS = 200


def _syll_count_last_line(text: str) -> int:
    lines = text.split("\n")
    last = lines[-1] if lines else ""
    return syllables.estimate(last) if last.strip() else 0


class HaikuSyllableBoundaryPotential(Potential):
    """
    Constraint evaluated only when syllable count changes.
    If syllables do not increase, weights are neutral (no resampling trigger).
    """

    def __init__(self, vocab, token_type, decode_fn, pattern=None):
        super().__init__(vocab, token_type)
        self.pattern = pattern or SYLLABLE_PATTERN
        self.decode_fn = decode_fn

    def _state(self, text):
        lines = text.split("\n")
        completed = lines[:-1]
        current = lines[-1] if lines else ""
        line_num = len(completed)
        num_lines = len(self.pattern)
        current_syl = syllables.estimate(current) if current.strip() else 0

        if line_num > num_lines:
            return {"valid": False, "line_num": line_num, "current_syl": current_syl, "current": current}
        if line_num == num_lines and current.strip():
            return {"valid": False, "line_num": line_num, "current_syl": current_syl, "current": current}

        for i, line in enumerate(completed):
            syl = syllables.estimate(line) if line.strip() else 0
            if syl != self.pattern[i]:
                return {"valid": False, "line_num": line_num, "current_syl": current_syl, "current": current}

        if line_num < num_lines:
            target = self.pattern[line_num]
            if current_syl > target:
                return {"valid": False, "line_num": line_num, "current_syl": current_syl, "current": current}

        return {"valid": True, "line_num": line_num, "current_syl": current_syl, "current": current}

    async def prefix(self, context):
        text = self.decode_fn(context)
        state = self._state(text)
        return 0.0 if state["valid"] else float("-inf")

    async def complete(self, context):
        text = self.decode_fn(context)
        state = self._state(text)
        if not state["valid"]:
            return float("-inf")
        num_lines = len(self.pattern)
        if state["line_num"] == num_lines and not state["current"].strip():
            return 0.0
        if state["line_num"] == num_lines - 1 and state["current_syl"] == self.pattern[-1]:
            return 0.0
        return float("-inf")

    async def logw_next(self, context):
        text = self.decode_fn(context)
        state = self._state(text)
        V = len(self.vocab_eos)

        if not state["valid"]:
            return self.make_lazy_weights(np.full(V, float("-inf")))

        line_num = state["line_num"]
        num_lines = len(self.pattern)

        # If all lines are done, only EOS
        if line_num >= num_lines:
            weights = np.full(V, float("-inf"))
            if self.eos_token in self.vocab_eos:
                weights[self.lookup[self.eos_token]] = 0.0
            return self.make_lazy_weights(weights)

        target = self.pattern[line_num]
        cur = state["current_syl"]

        # Syllable-boundary trigger: compare to previous syllable count
        prev_text = self.decode_fn(context[:-1]) if context else ""
        prev_syl = _syll_count_last_line(prev_text)
        syllable_changed = cur != prev_syl

        if not syllable_changed:
            return self.make_lazy_weights(np.zeros(V))

        # If over budget, block everything
        if cur > target:
            return self.make_lazy_weights(np.full(V, float("-inf")))

        # At or under budget: neutral (let model decide when to newline/EOS)
        return self.make_lazy_weights(np.zeros(V))


def decode_bytes(ctx):
    if not ctx:
        return ""
    chunks = []
    for tok in ctx:
        if isinstance(tok, EndOfSequence):
            break
        if isinstance(tok, (bytes, bytearray)):
            chunks.append(tok)
        elif isinstance(tok, int):
            chunks.append(bytes([tok]))
    return b"".join(chunks).decode("utf-8", errors="ignore")


def decode_tokens(tokenizer, ctx):
    if not ctx:
        return ""
    # PromptedLLM vocab may be bytes, handle both ints and bytes.
    if isinstance(ctx[0], (bytes, bytearray)):
        chunks = []
        for tok in ctx:
            if isinstance(tok, EndOfSequence):
                break
            if isinstance(tok, (bytes, bytearray)):
                chunks.append(tok)
        return b"".join(chunks).decode("utf-8", errors="ignore")
    ids = []
    for tok in ctx:
        if isinstance(tok, EndOfSequence):
            break
        if isinstance(tok, int):
            ids.append(tok)
    return tokenizer.decode(ids, skip_special_tokens=False)


async def run_token():
    print("=" * 60)
    print("TOKEN-LEVEL HAIKU SMC (syllable-boundary)")
    print("=" * 60)
    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf", eos_tokens=[b"<|end_of_text|>", b"<|eot_id|>"])
    import ipdb; ipdb.set_trace()
    prompted.set_prompt_from_str(PROMPT)
    tokenizer = prompted.model.tokenizer
    constraint = HaikuSyllableBoundaryPotential(
        prompted.vocab, prompted.token_type, lambda c: decode_tokens(tokenizer, c), SYLLABLE_PATTERN
    )
    sampler = AWRS(prompted, constraint)
    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
        json_path="haiku_token_syllable.json",
    )
    await sampler.cleanup()
    print(sequences.decoded_posterior)
    prompted.model.clear_cache()
    return sequences, constraint


async def run_byte():
    print("=" * 60)
    print("BYTE-LEVEL HAIKU SMC (syllable-boundary)")
    print("=" * 60)

    beam_params = BeamParams(K=5, prune_threshold=0.0, eos_tokens={b"<|end_of_text|>", b"<|eot_id|>"})

    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)
    constraint = HaikuSyllableBoundaryPotential(byte_llm.vocab, byte_llm.token_type, decode_bytes, SYLLABLE_PATTERN)
    sampler = AWRS(byte_llm, constraint)
    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
        json_path="haiku_byte_syllable.json",
    )
    await sampler.cleanup()
    await byte_llm.cleanup()
    print(sequences.decoded_posterior)
    return sequences, constraint

async def main():
    seq_token, constraint_token = await run_token()
    seq_byte, constraint_byte = await run_byte()
    # summarize("Token (syllable-boundary)", seq_token, constraint_token)
    # summarize("Byte (syllable-boundary)", seq_byte, constraint_byte)


if __name__ == "__main__":
    asyncio.run(main())
