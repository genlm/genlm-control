"""
Byte-level Haiku SMC experiment (constraint checked every byte).

This mirrors the token-step experiment but uses ByteLLM and enforces
the syllable constraint at every byte step (no word-boundary delay).
"""

import asyncio
import numpy as np
import syllables

from genlm.control import AWRS, ByteLLM
from genlm.control.constant import EndOfSequence
from genlm.control.potential.base import Potential
from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWrite a haiku about nature.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SYLLABLE_PATTERN = [5, 7, 5]  # Standard haiku
N_PARTICLES = 5
MAX_TOKENS = 200  # bytes; allow plenty since bytes are smaller units


class HaikuBytePotential(Potential):
    """Syllable constraint evaluated after every byte token."""

    def __init__(self, vocab, token_type, pattern=None):
        super().__init__(vocab, token_type)
        self.pattern = pattern or SYLLABLE_PATTERN

    def _decode(self, context):
        """Decode bytes to string, ignoring EndOfSequence markers."""
        if not context:
            return ""
        chunks = []
        for tok in context:
            if isinstance(tok, EndOfSequence):
                break
            if isinstance(tok, (bytes, bytearray)):
                chunks.append(tok)
            elif isinstance(tok, int):
                # ByteLLM can also surface ints in some pathways; treat as single byte
                chunks.append(bytes([tok]))
        return b"".join(chunks).decode("utf-8", errors="ignore")

    def _state(self, text):
        """Return parsed state and validity."""
        lines = text.split("\n")
        completed = lines[:-1]
        current = lines[-1] if lines else ""
        line_num = len(completed)
        num_lines = len(self.pattern)
        current_syl = syllables.estimate(current) if current.strip() else 0

        # Too many lines started
        if line_num > num_lines:
            return {"valid": False, "reason": "Too many lines", "line_num": line_num, "current_syl": current_syl, "current": current}
        # Completed all lines but extra text started
        if line_num == num_lines and current.strip():
            return {"valid": False, "reason": "Extra text after final line", "line_num": line_num, "current_syl": current_syl, "current": current}

        # Check completed lines syllables
        for i, line in enumerate(completed):
            syl = syllables.estimate(line) if line.strip() else 0
            if syl != self.pattern[i]:
                return {"valid": False, "reason": f"Line {i+1} has {syl} (need {self.pattern[i]})", "line_num": line_num, "current_syl": current_syl, "current": current}

        # Check current line budget
        if line_num < num_lines:
            target = self.pattern[line_num]
            if current_syl > target:
                return {"valid": False, "reason": f"Line {line_num+1} over budget {current_syl}>{target}", "line_num": line_num, "current_syl": current_syl, "current": current}

        return {"valid": True, "line_num": line_num, "current_syl": current_syl, "current": current}

    async def prefix(self, context):
        state = self._state(self._decode(context))
        return 0.0 if state["valid"] else float("-inf")

    async def complete(self, context):
        state = self._state(self._decode(context))
        if not state["valid"]:
            return float("-inf")
        num_lines = len(self.pattern)
        # Accept when all lines complete and no extra text
        if state["line_num"] == num_lines and not state["current"].strip():
            return 0.0
        # Accept when last line exactly meets target
        if state["line_num"] == num_lines - 1 and state["current_syl"] == self.pattern[-1]:
            return 0.0
        return float("-inf")

    async def logw_next(self, context):
        """
        Apply constraint every byte:
        - If invalid so far: block all.
        - If line meets/exceeds target: only allow newline (non-final lines) or EOS/newline (final line).
        - Otherwise: neutral (0) over vocab.
        """
        state = self._state(self._decode(context))
        V = len(self.vocab_eos)

        if not state["valid"]:
            return self.make_lazy_weights(np.full(V, float("-inf")))

        line_num = state["line_num"]
        num_lines = len(self.pattern)
        last_idx = num_lines - 1

        # If all lines are done, only EOS
        # if line_num >= num_lines:
        #     weights = np.full(V, float("-inf"))
        #     if self.eos_token in self.vocab_eos:
        #         weights[self.lookup[self.eos_token]] = 0.0
        #     return self.make_lazy_weights(weights)

        target = self.pattern[line_num]
        cur = state["current_syl"]

        # If over budget: block all
        if cur > target:
            return self.make_lazy_weights(np.full(V, float("-inf")))

        # At or under budget: neutral (let model decide when to newline/EOS)
        return self.make_lazy_weights(np.zeros(V))


async def run():
    print("=" * 60)
    print("BYTE-LEVEL HAIKU SMC (per-byte constraint)")
    print("=" * 60)

    # Derive byte-level EOS from tokenizer to avoid partial endings
    # llm_for_eos = load_model_by_name(MODEL_NAME, backend="hf")
    # model_eos = llm_for_eos.byte_vocab[llm_for_eos.tokenizer.eos_token_id]
    beam_params = BeamParams(
        K=5,  # slightly wider beam for cleaner endings
        prune_threshold=0.0,
        eos_tokens={b"<|end_of_text|>", b"<|eot_id|>"},
    )
    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)

    constraint = HaikuBytePotential(byte_llm.vocab, byte_llm.token_type, SYLLABLE_PATTERN)
    sampler = AWRS(byte_llm, constraint)

    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
        json_path="haiku_byte_step.json",
    )
    await sampler.cleanup()

    norm_weights = sequences.normalized_weights
    log_weights = sequences.log_weights

    valid = 0
    uniq = set()
    valid_logweights = []
    print(f"\nProcessing {len(sequences)} sequences...")

    for i, ((seq, _), w, logw) in enumerate(zip(sequences, norm_weights, log_weights)):
        if w < 1e-6:
            continue
        text = constraint._decode(seq)
        lines = text.strip().split("\n")
        if len(lines) == len(SYLLABLE_PATTERN):
            syls = [syllables.estimate(l) for l in lines]
        else:
            syls = []
        is_valid = len(lines) == len(SYLLABLE_PATTERN) and syls == SYLLABLE_PATTERN

        if is_valid:
            valid += 1
            uniq.add(text.strip())
            valid_logweights.append(logw)
            print(f"\n✓ Valid #{valid} (w={w:.4f}, logw={logw:.2f}):")
            for j, line in enumerate(lines):
                print(f"  Line {j+1} ({syls[j]} syl): {line}")
        elif i < 5:
            print(f"\n✗ Invalid #{i+1} (w={w:.4f}, logw={logw:.2f}): {repr(text[:120])}")

    total = len(sequences)
    valid_pct = (valid / total * 100) if total else 0.0
    avg_logw = np.mean(valid_logweights) if valid_logweights else float("-inf")

    print("\n" + "-" * 60)
    print("Summary")
    print(f"  Total sequences: {total}")
    print(f"  Valid: {valid}/{total} ({valid_pct:.1f}%)")
    print(f"  Unique valid: {len(uniq)}")
    print(f"  Avg log-weight (valid): {avg_logw:.2f}")
    print("-" * 60)


if __name__ == "__main__":
    asyncio.run(run())
