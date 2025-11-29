"""
Token-level Haiku SMC experiment (constraint checked every token).

This version evaluates the syllable constraint after each token (no word-boundary delay)
to compare with byte-level or word-boundary approaches.
"""

import asyncio
import numpy as np
import syllables

from genlm.control import AWRS, PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.control.potential.base import Potential

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWrite a haiku about nature.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
SYLLABLE_PATTERN = [5, 7, 5]  # Standard haiku
N_PARTICLES = 5
MAX_TOKENS = 100


class HaikuTokenPotential(Potential):
    """Syllable constraint evaluated after every token (no boundary deferral)."""

    def __init__(self, vocab, token_type, tokenizer, pattern=None):
        super().__init__(vocab, token_type)
        self.pattern = pattern or SYLLABLE_PATTERN
        self.tokenizer = tokenizer

    def _decode(self, context):
        """Decode context (bytes or ints) to string, ignoring EndOfSequence."""
        if not context:
            return ""

        # Handle byte tokens
        if isinstance(context[0], (bytes, bytearray)):
            chunks = []
            for tok in context:
                if isinstance(tok, EndOfSequence):
                    break
                if isinstance(tok, (bytes, bytearray)):
                    chunks.append(tok)
            return b"".join(chunks).decode("utf-8", errors="ignore")

        # Handle token ids
        ids = []
        for tok in context:
            if isinstance(tok, EndOfSequence):
                break
            if isinstance(tok, int):
                ids.append(tok)
        return self.tokenizer.decode(ids, skip_special_tokens=False)

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
            return {"valid": False, "reason": "Too many lines", "line_num": line_num, "current_syl": current_syl, "lines": completed, "current": current}
        # Completed all lines but extra text started
        if line_num == num_lines and current.strip():
            return {"valid": False, "reason": "Extra text after final line", "line_num": line_num, "current_syl": current_syl, "lines": completed, "current": current}

        # Check completed lines syllables
        for i, line in enumerate(completed):
            syl = syllables.estimate(line) if line.strip() else 0
            if syl != self.pattern[i]:
                return {"valid": False, "reason": f"Line {i+1} has {syl} (need {self.pattern[i]})", "line_num": line_num, "current_syl": current_syl, "lines": completed, "current": current}

        # Check current line budget
        if line_num < num_lines:
            target = self.pattern[line_num]
            if current_syl > target:
                return {"valid": False, "reason": f"Line {line_num+1} over budget {current_syl}>{target}", "line_num": line_num, "current_syl": current_syl, "lines": completed, "current": current}

        return {"valid": True, "line_num": line_num, "current_syl": current_syl, "lines": completed, "current": current}

    async def prefix(self, context):
        """Weight for partial sequence."""
        state = self._state(self._decode(context))
        return 0.0 if state["valid"] else float("-inf")

    async def complete(self, context):
        """Weight for completed haiku (EOS)."""
        state = self._state(self._decode(context))
        if not state["valid"]:
            return float("-inf")
        num_lines = len(self.pattern)
        # Accept when all lines complete and no extra text
        if state["line_num"] == num_lines and not state["current"].strip():
            return 0.0
        # Accept when last line exactly meets target (line_num points to final line)
        if state["line_num"] == num_lines - 1 and state["current_syl"] == self.pattern[-1]:
            return 0.0
        return float("-inf")

    async def logw_next(self, context):
        """
        Apply constraint every token:
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

        newline_token = "\n"

        # If at/over budget: force line break / EOS
        if cur >= target:
            weights = np.full(V, float("-inf"))
            if line_num < last_idx:
                if newline_token in self.vocab_eos:
                    weights[self.lookup[newline_token]] = 0.0
            else:
                if self.eos_token in self.vocab_eos:
                    weights[self.lookup[self.eos_token]] = 0.0
                if newline_token in self.vocab_eos:
                    weights[self.lookup[newline_token]] = 0.0
            return self.make_lazy_weights(weights)

        # Under budget: neutral
        return self.make_lazy_weights(np.zeros(V))


async def run():
    print("=" * 60)
    print("TOKEN-LEVEL HAIKU SMC (per-token constraint)")
    print("=" * 60)

    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)

    tokenizer = prompted.model.tokenizer
    constraint = HaikuTokenPotential(prompted.vocab, prompted.token_type, tokenizer, SYLLABLE_PATTERN)
    sampler = AWRS(prompted, constraint)

    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
        json_path="=haiku_token_step.json",
    )
    await sampler.cleanup()
    prompted.model.clear_cache()

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
            print(f"\n✗ Invalid #{i+1} (w={w:.4f}, logw={logw:.2f}): {repr(text[:100])}")

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
