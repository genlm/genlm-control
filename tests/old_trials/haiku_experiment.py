"""
Haiku Generation Experiment
Compare Token-level SMC vs Byte-level SMC on haiku generation with syllable constraints.

Haiku format: 5-7-5 syllable pattern across 3 lines
"""

import asyncio
import numpy as np
from genlm.control import AWRS, BoolFSA, ByteLLM, PromptedLLM
from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
from genlm.control.constant import EndOfSequence
from transformers import GPT2Tokenizer
import syllables  # Accurate syllable counting

# Syllable-organized vocabulary
# For simplicity, using words with known syllable counts
WORDS_1_SYL = ["spring", "fall", "wind", "rain", "snow", "sun", "moon", "star", "tree", "leaf",
               "bird", "frog", "pond", "stream", "cloud", "sky", "hill", "mountain", "field", "lake"]

WORDS_2_SYL = ["cherry", "blossom", "flower", "petal", "water", "river", "forest", "ocean",
               "winter", "summer", "autumn", "morning", "evening", "sunset", "lightning",
               "thunder", "rainbow", "mountain", "valley", "meadow"]

WORDS_3_SYL = ["butterfly", "dragonfly", "beautiful", "terrible", "messenger", "wandering",
               "whispering", "glistening", "shimmering", "flowering", "blooming", "awakens"]

# Pre-built haiku lines - VERIFIED with syllables library
# 5 syllables each
LINES_5_SYL = [
    "birds sing in the trees",      # 5 ✓
    "snow falls on the ground",     # 5 ✓
    "bright moon in the sky",       # 5 ✓
    "the old pond sits still",      # 5 ✓
    "warm sun lights the sky",      # 5 ✓
    "cold rain on the roof",        # 5 ✓
    "the rain falls down hard",     # 5 ✓
    "a bird sits on branch",        # 5 ✓
]

# 7 syllables each
LINES_7_SYL = [
    "the mountain stands tall in mist",    # 7 ✓
    "water flows down to the sea",         # 7 ✓
    "cherry petals fall to ground",        # 7 ✓
    "the forest sleeps beneath snow",      # 7 ✓
    "dark clouds gather in the sky",       # 7 ✓
    "snow covers all of the land",         # 7 ✓
]

MODEL_NAME = "gpt2"
PROMPT = "Write a haiku about nature:\n"
N_PARTICLES = 5  # Reduced from 50 to save memory
MAX_TOKENS = 150  # Increased for byte-level generation

def count_syllables_simple(text: str) -> int:
    """Use the syllables library for accurate syllable counting."""
    return syllables.estimate(text)

def verify_haiku(text: str) -> dict:
    """Verify if text is a valid 5-7-5 haiku."""
    lines = text.strip().split('\n')

    if len(lines) != 3:
        return {
            'valid': False,
            'reason': f'Expected 3 lines, got {len(lines)}',
            'syllables': [],
            'lines': lines
        }

    syllables = [count_syllables_simple(line) for line in lines]
    target = [5, 7, 5]

    valid = syllables == target

    return {
        'valid': valid,
        'syllables': syllables,
        'target': target,
        'lines': lines,
        'reason': 'OK' if valid else f'Syllables {syllables} != {target}'
    }

async def run_experiment(sampler_type: str, model, constraint):
    """Run haiku generation experiment with given model and constraint."""
    print(f"\n{'='*60}")
    print(f"Running {sampler_type}...")
    print(f"{'='*60}")

    sampler = AWRS(model, constraint)
    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=1,
    )
    await sampler.cleanup()

    norm_weights = sequences.normalized_weights
    log_weights = sequences.log_weights
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    results = []
    valid_count = 0

    print(f"\nProcessing {len(sequences)} sequences...")

    for i, ((seq, _), w, logw) in enumerate(zip(sequences, norm_weights, log_weights)):
        if w < 1e-6:
            continue

        # Decode sequence to text
        text = ""
        clean_seq = []

        if isinstance(seq, list):
            for item in seq:
                if not isinstance(item, EndOfSequence):
                    clean_seq.append(item)
        else:
            if not isinstance(seq, EndOfSequence):
                clean_seq = seq
            else:
                clean_seq = []

        if isinstance(clean_seq, bytes):
            text = clean_seq.decode("utf-8", errors="ignore")
        elif isinstance(clean_seq, list) and clean_seq:
            if isinstance(clean_seq[0], bytes):
                text = b"".join(clean_seq).decode("utf-8", errors="ignore")
            elif isinstance(clean_seq[0], int):
                text = tokenizer.decode(clean_seq)

        # Verify haiku
        verification = verify_haiku(text)
        verification['weight'] = w
        verification['log_weight'] = logw
        verification['text'] = text
        verification['rank'] = i

        results.append(verification)

        if verification['valid']:
            valid_count += 1
            print(f"\n✓ Valid Haiku #{valid_count} (weight: {w:.4f}, logw: {logw:.2f}):")
            for j, line in enumerate(verification['lines']):
                print(f"  Line {j+1} ({verification['syllables'][j]} syl): {line}")
        elif i < 3:  # Show first 3 invalid ones for debugging
            print(f"\n✗ Invalid #{i+1} (weight: {w:.4f}, logw: {logw:.2f}):")
            print(f"  Text: {repr(text[:100])}")
            print(f"  Reason: {verification['reason']}")

    # Summary statistics
    total = len(results)
    valid_pct = (valid_count / total * 100) if total > 0 else 0

    valid_logweights = [r['log_weight'] for r in results if r['valid']]
    avg_valid_logweight = np.mean(valid_logweights) if valid_logweights else float('-inf')

    print(f"\n{'-'*60}")
    print(f"{sampler_type} Summary:")
    print(f"  Total sequences: {total}")
    print(f"  Valid haikus: {valid_count}/{total} ({valid_pct:.1f}%)")
    print(f"  Avg log-weight (valid): {avg_valid_logweight:.2f}")
    print(f"{'-'*60}")

    return {
        'sampler_type': sampler_type,
        'valid_count': valid_count,
        'total': total,
        'valid_pct': valid_pct,
        'avg_valid_logweight': avg_valid_logweight,
        'results': results
    }

def build_haiku_constraint():
    """
    Build a constraint for haiku generation.
    Uses pre-built lines with correct syllable counts.
    Format: line5\nline7\nline5
    """
    # Escape special regex characters in lines
    def escape_regex(s):
        special_chars = r'\.^$*+?{}[]|()'
        for char in special_chars:
            s = s.replace(char, '\\' + char)
        return s

    lines_5_escaped = [escape_regex(line) for line in LINES_5_SYL]
    lines_7_escaped = [escape_regex(line) for line in LINES_7_SYL]

    # Build pattern: (line5)\n(line7)\n(line5)
    pattern_5 = "(" + "|".join(lines_5_escaped) + ")"
    pattern_7 = "(" + "|".join(lines_7_escaped) + ")"

    # Full haiku pattern
    haiku_pattern = pattern_5 + r"\n" + pattern_7 + r"\n" + pattern_5

    print(f"\nConstraint vocabulary:")
    print(f"  5-syllable lines: {len(LINES_5_SYL)}")
    print(f"  7-syllable lines: {len(LINES_7_SYL)}")
    print(f"  Total possible haikus: {len(LINES_5_SYL)**2 * len(LINES_7_SYL)}")

    return BoolFSA.from_regex(haiku_pattern)

async def main():
    print("="*60)
    print("HAIKU GENERATION EXPERIMENT")
    print("Comparing Token-level SMC vs Byte-level SMC")
    print("="*60)

    # Build constraint
    base_constraint = build_haiku_constraint()

    # 1. Run PromptedLLM (Token Level)
    print("\n" + "="*60)
    print("SETUP: PromptedLLM (Token-Level)")
    print("="*60)

    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)
    constraint_prompted = base_constraint.coerce(prompted, f=b"".join)

    results_token = await run_experiment("PromptedLLM (Token-Level)", prompted, constraint_prompted)
    prompted.model.clear_cache()

    # 2. Run ByteLLM (Character Level)
    print("\n" + "="*60)
    print("SETUP: ByteLLM (Byte-Level)")
    print("="*60)

    llm = load_model_by_name(MODEL_NAME, backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]

    beam_params = BeamParams(K=5, prune_threshold=0.0, eos_tokens={model_eos_token})
    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)

    constraint_byte = base_constraint.coerce(byte_llm, f=b"".join)

    results_byte = await run_experiment("ByteLLM (Byte-Level)", byte_llm, constraint_byte)
    await byte_llm.cleanup()

    # Final Comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<30} {'Token SMC':>15} {'Byte SMC':>15}")
    print("-"*60)
    print(f"{'Valid Haikus':<30} {results_token['valid_count']:>15} {results_byte['valid_count']:>15}")
    print(f"{'Valid %':<30} {results_token['valid_pct']:>14.1f}% {results_byte['valid_pct']:>14.1f}%")
    print(f"{'Avg Log-Weight (valid)':<30} {results_token['avg_valid_logweight']:>15.2f} {results_byte['avg_valid_logweight']:>15.2f}")
    print("-"*60)

    # Determine winner
    if results_byte['valid_pct'] > results_token['valid_pct']:
        print("\n🏆 WINNER: Byte-level SMC (higher valid %)")
    elif results_token['valid_pct'] > results_byte['valid_pct']:
        print("\n🏆 WINNER: Token-level SMC (higher valid %)")
    else:
        print("\n🤝 TIE: Both achieved same valid %")

    if results_byte['avg_valid_logweight'] > results_token['avg_valid_logweight']:
        print("🏆 BETTER EXPLORATION: Byte-level SMC (higher log-prob on valid haikus)")
    elif results_token['avg_valid_logweight'] > results_byte['avg_valid_logweight']:
        print("🏆 BETTER EXPLORATION: Token-level SMC (higher log-prob on valid haikus)")

if __name__ == "__main__":
    asyncio.run(main())
