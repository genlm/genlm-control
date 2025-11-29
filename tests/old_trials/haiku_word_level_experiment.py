"""
Word-Level Haiku Generation Experiment

SMC operates at WORD level, not token/byte level.
Evaluation and resampling happen after each word, not each character.

This creates a word vocabulary with syllable counts and builds an FSA
where each transition is a complete word.
"""

import asyncio
import numpy as np
from genlm.control import AWRS, BoolFSA, ByteLLM, PromptedLLM
from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
from genlm.control.constant import EndOfSequence
from transformers import GPT2Tokenizer
import syllables
import torch
import re

MODEL_NAME = "gpt2"
PROMPT = "Write a haiku about nature:\n"
N_PARTICLES = 10
MAX_TOKENS = 100  # Now counts words, not tokens

SYLLABLE_PATTERN = [5, 7, 5]

# Word vocabulary organized by syllable count
# Each word is a single "token" in our FSA
WORDS_BY_SYLLABLES = {
    1: [
        "the", "a", "sun", "moon", "sky", "rain", "snow", "wind",
        "tree", "leaf", "bird", "frog", "pond", "stream", "cloud",
        "hill", "field", "lake", "stars", "night", "day", "fall",
        "spring", "warm", "cold", "bright", "dark", "soft", "still",
        "green", "blue", "white", "red", "through", "on", "in",
        "by", "with", "and", "or", "but", "so", "now", "here",
    ],
    2: [
        "cherry", "blossom", "flower", "petal", "water", "river",
        "forest", "ocean", "winter", "summer", "autumn", "morning",
        "evening", "sunset", "lightning", "thunder", "rainbow",
        "mountain", "valley", "meadow", "gentle", "falling", "rising",
        "floating", "singing", "dancing", "sleeping", "waking",
        "under", "over", "across", "between", "into", "onto",
    ],
    3: [
        "beautiful", "wonderful", "terrible", "silently", "quietly",
        "suddenly", "gracefully", "peacefully", "butterfly", "dragonfly",
        "whispering", "glistening", "shimmering", "flowering", "awakening",
    ],
}

def build_word_haiku_regex():
    """
    Build a regex that matches haikus at the WORD level.

    Each line must have exactly the target syllables.
    We enumerate all valid word combinations for each line.
    """

    def get_line_patterns(target_syllables, max_words=6):
        """Generate regex patterns for lines with exactly target_syllables."""
        patterns = []

        # Generate all combinations of words that sum to target_syllables
        def generate_combinations(remaining, current_words, current_pattern):
            if remaining == 0:
                # Valid combination found
                pattern = " ".join(current_pattern)
                patterns.append(pattern)
                return
            if remaining < 0 or len(current_words) >= max_words:
                return

            # Try adding each word
            for syl_count in [1, 2, 3]:
                if syl_count <= remaining:
                    for word in WORDS_BY_SYLLABLES.get(syl_count, []):
                        generate_combinations(
                            remaining - syl_count,
                            current_words + [word],
                            current_pattern + [re.escape(word)]
                        )

        generate_combinations(target_syllables, [], [])
        return patterns

    # Generate patterns for each line
    print("Generating word combinations...")
    line1_patterns = get_line_patterns(5, max_words=5)
    line2_patterns = get_line_patterns(7, max_words=6)
    line3_patterns = get_line_patterns(5, max_words=5)

    print(f"  Line 1 (5 syl): {len(line1_patterns)} patterns")
    print(f"  Line 2 (7 syl): {len(line2_patterns)} patterns")
    print(f"  Line 3 (5 syl): {len(line3_patterns)} patterns")

    # Limit to avoid regex explosion
    MAX_PATTERNS = 100
    if len(line1_patterns) > MAX_PATTERNS:
        line1_patterns = line1_patterns[:MAX_PATTERNS]
    if len(line2_patterns) > MAX_PATTERNS:
        line2_patterns = line2_patterns[:MAX_PATTERNS]
    if len(line3_patterns) > MAX_PATTERNS:
        line3_patterns = line3_patterns[:MAX_PATTERNS]

    print(f"  (Limited to {MAX_PATTERNS} patterns each)")

    # Build full regex
    line1_regex = "(" + "|".join(line1_patterns) + ")"
    line2_regex = "(" + "|".join(line2_patterns) + ")"
    line3_regex = "(" + "|".join(line3_patterns) + ")"

    full_regex = line1_regex + r"\n" + line2_regex + r"\n" + line3_regex

    return full_regex


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

    syllable_counts = [syllables.estimate(line) for line in lines]
    target = SYLLABLE_PATTERN

    valid = syllable_counts == target

    return {
        'valid': valid,
        'syllables': syllable_counts,
        'target': target,
        'lines': lines,
        'reason': 'OK' if valid else f'Syllables {syllable_counts} != {target}'
    }


async def run_experiment(sampler_type: str, model, constraint):
    """Run haiku generation experiment."""
    print(f"\n{'='*60}")
    print(f"Running {sampler_type}...")
    print(f"{'='*60}")

    sampler = AWRS(model, constraint)
    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
    )
    await sampler.cleanup()

    norm_weights = sequences.normalized_weights
    log_weights = sequences.log_weights
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    results = []
    valid_count = 0
    unique_haikus = set()

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

        results.append(verification)

        if verification['valid']:
            valid_count += 1
            unique_haikus.add(text.strip())
            if valid_count <= 5:
                print(f"\n✓ Valid Haiku #{valid_count} (weight: {w:.4f}, logw: {logw:.2f}):")
                for j, line in enumerate(verification['lines']):
                    print(f"  Line {j+1} ({verification['syllables'][j]} syl): {line}")
        elif i < 3:
            print(f"\n✗ Invalid #{i+1} (weight: {w:.4f}, logw: {logw:.2f}):")
            print(f"  Text: {repr(text[:80])}")
            print(f"  Reason: {verification['reason']}")

    total = len(results)
    valid_pct = (valid_count / total * 100) if total > 0 else 0
    valid_logweights = [r['log_weight'] for r in results if r['valid']]
    avg_valid_logweight = np.mean(valid_logweights) if valid_logweights else float('-inf')

    print(f"\n{'-'*60}")
    print(f"{sampler_type} Summary:")
    print(f"  Total sequences: {total}")
    print(f"  Valid haikus: {valid_count}/{total} ({valid_pct:.1f}%)")
    print(f"  Unique valid haikus: {len(unique_haikus)}")
    print(f"  Avg log-weight (valid): {avg_valid_logweight:.2f}")
    print(f"{'-'*60}")

    return {
        'sampler_type': sampler_type,
        'valid_count': valid_count,
        'total': total,
        'valid_pct': valid_pct,
        'unique_count': len(unique_haikus),
        'avg_valid_logweight': avg_valid_logweight,
        'results': results,
        'unique_haikus': unique_haikus
    }


async def main():
    torch.cuda.empty_cache()

    print("="*60)
    print("WORD-LEVEL HAIKU GENERATION EXPERIMENT")
    print("SMC evaluates at WORD boundaries, not token/byte")
    print("="*60)

    # Build word-level constraint
    print("\nBuilding word-level haiku constraint...")
    regex_pattern = build_word_haiku_regex()

    print(f"\nRegex length: {len(regex_pattern)} characters")

    try:
        base_constraint = BoolFSA.from_regex(regex_pattern)
        print("✓ FSA built successfully")
    except Exception as e:
        print(f"✗ FSA build failed: {e}")
        print("\nTrying with simpler constraint...")

        # Fallback to simpler pre-built lines
        simple_lines_5 = ["the sun shines bright", "cold wind through trees", "soft rain on leaves"]
        simple_lines_7 = ["the mountain stands so tall here", "water flows down to the sea"]

        pattern = (
            "(" + "|".join(re.escape(l) for l in simple_lines_5) + ")" +
            r"\n" +
            "(" + "|".join(re.escape(l) for l in simple_lines_7) + ")" +
            r"\n" +
            "(" + "|".join(re.escape(l) for l in simple_lines_5) + ")"
        )
        base_constraint = BoolFSA.from_regex(pattern)
        print("✓ Simple FSA built")

    # Run Token-level SMC
    print("\n" + "="*60)
    print("SETUP: PromptedLLM (Token-Level)")
    print("="*60)

    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)
    constraint_prompted = base_constraint.coerce(prompted, f=b"".join)

    results_token = await run_experiment("PromptedLLM (Token-Level)", prompted, constraint_prompted)
    prompted.model.clear_cache()
    torch.cuda.empty_cache()

    # Run Byte-level SMC
    print("\n" + "="*60)
    print("SETUP: ByteLLM (Byte-Level)")
    print("="*60)

    llm = load_model_by_name(MODEL_NAME, backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]

    beam_params = BeamParams(K=8, prune_threshold=0.0, eos_tokens={model_eos_token})
    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)

    constraint_byte = base_constraint.coerce(byte_llm, f=b"".join)

    results_byte = await run_experiment("ByteLLM (Byte-Level)", byte_llm, constraint_byte)
    await byte_llm.cleanup()

    # Comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'Metric':<30} {'Token SMC':>15} {'Byte SMC':>15}")
    print("-"*60)
    print(f"{'Valid Haikus':<30} {results_token['valid_count']:>15} {results_byte['valid_count']:>15}")
    print(f"{'Valid %':<30} {results_token['valid_pct']:>14.1f}% {results_byte['valid_pct']:>14.1f}%")
    print(f"{'Unique Valid Haikus':<30} {results_token['unique_count']:>15} {results_byte['unique_count']:>15}")
    print(f"{'Avg Log-Weight (valid)':<30} {results_token['avg_valid_logweight']:>15.2f} {results_byte['avg_valid_logweight']:>15.2f}")
    print("-"*60)


if __name__ == "__main__":
    asyncio.run(main())
