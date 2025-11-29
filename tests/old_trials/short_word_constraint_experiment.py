import asyncio
import numpy as np

from genlm.control import AWRS, PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.control.potential.base import Potential

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "The Fed says"
MAX_WORD_LEN = 3  # strictly less than 4
N_PARTICLES = 10
MAX_TOKENS = 100


class ShortWordPotential(Potential):
    """Hard constraint: every word length must be <= MAX_WORD_LEN."""

    def __init__(self, vocab, token_type, tokenizer, max_len=MAX_WORD_LEN):
        super().__init__(vocab, token_type)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def _decode_ctx(self, ctx):
        if not ctx:
            return ""
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
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def _current_word_len(self, text):
        if not text:
            return 0
        parts = text.rstrip().split()
        if not parts:
            return 0
        last = parts[-1].rstrip(".,;:!?")
        return len(last)

    async def prefix(self, context):
        return 0.0 if self._current_word_len(self._decode_ctx(context)) <= self.max_len else float("-inf")

    async def complete(self, context):
        return 0.0 if self._current_word_len(self._decode_ctx(context)) <= self.max_len else float("-inf")

    async def logw_next(self, context):
        # Neutral proposal; rely on prefix/complete to kill bad prefixes
        return self.make_lazy_weights(np.zeros(len(self.vocab_eos)))


async def main():
    print("=" * 60)
    print("SHORT WORD CONSTRAINT (len < 4)")
    print("=" * 60)

    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)
    tokenizer = prompted.model.tokenizer

    constraint = ShortWordPotential(prompted.vocab, prompted.token_type, tokenizer, MAX_WORD_LEN)
    sampler = AWRS(prompted, constraint)

    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
    )
    print(sequences.decoded_posterior)
    print(sequences.posterior)
    await sampler.cleanup()
    prompted.model.clear_cache()

    norm_weights = sequences.normalized_weights
    log_weights = sequences.log_weights

    print(f"\nProcessing {len(sequences)} sequences...")
    for i, ((seq, _), w, logw) in enumerate(zip(sequences, norm_weights, log_weights)):
        if w < 1e-6 or i >= 5:
            continue
        text = constraint._decode_ctx(seq)
        print(f"\nSample #{i+1} (w={w:.4f}, logw={logw:.2f}): {repr(text)}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
