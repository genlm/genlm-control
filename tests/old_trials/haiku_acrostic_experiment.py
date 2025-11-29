"""
Haiku + Acrostic Experiment
Compare Token-level SMC vs Byte-level SMC on haiku generation with acrostic constraint.

Constraint: Generate a 5-7-5 haiku where the first letters of each line spell a word.
This showcases the "coalition subsidy" effect in character-level SMC.
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

MODEL_NAME = "gpt2"
N_PARTICLES = 10  # Reduced for memory
MAX_TOKENS = 150

# Acrostic word - first letters should spell this
ACROSTIC_WORD = "SUN"  # Simple 3-letter word
PROMPT = f"Write a haiku about {ACROSTIC_WORD.lower()}:\n"

# Organize haiku lines by first letter and syllable count
# 5-syllable lines (verified)
LINES_5_SYL_BY_LETTER = {
    'S': [
        "snow falls on the ground",      # 5 ✓
        "sun lights up the sky",         # 5 ✓
        "spring wind blows through trees", # 5 ✓
    ],
    'U': [
        "under the bright moon",        # 5 ✓
        "under tall green trees",       # 5 ✓
    ],
    'N': [
        "no clouds in the sky",         # 5 ✓
        "night birds sing their song",  # 5 ✓
    ],
    'B': [
        "birds sing in the trees",      # 5 ✓
        "bright moon in the sky",       # 5 ✓
    ],
    'T': [
        "the old pond sits still",      # 5 ✓
        "the rain falls down hard",     # 5 ✓
    ],
    'W': [
        "warm sun lights the sky",      # 5 ✓
    ],
    'C': [
        "cold rain on the roof",        # 5 ✓
    ],
    'A': [
        "a bird sits on branch",        # 5 ✓
    ],
}

# 7-syllable lines (verified)
LINES_7_SYL_BY_LETTER = {
    'S': [
        "snow covers all of the land",  # 7 ✓
    ],
    'U': [
        "under trees the fox runs fast", # 7 ✓
    ],
    'N': [
        "new light breaks across the sky", # 7 ✓
    ],
    'T': [
        "the mountain stands tall in mist", # 7 ✓
        "the forest sleeps beneath snow",   # 7 ✓
    ],
    'W': [
        "water flows down to the sea",      # 7 ✓
    ],
    'C': [
        "cherry petals fall to ground",     # 7 ✓
    ],
    'D': [
        "dark clouds gather in the sky",    # 7 ✓
    ],
}

def verify_syllable_counts():
    """Verify all lines have correct syllable counts."""
    print("\nVerifying syllable counts...")
    all_valid = True

    for letter, lines in LINES_5_SYL_BY_LETTER.items():
        for line in lines:
            count = syllables.estimate(line)
            if count != 5:
                print(f"  ✗ {letter}: '{line}' = {count} syllables (expected 5)")
                all_valid = False

    for letter, lines in LINES_7_SYL_BY_LETTER.items():
        for line in lines:
            count = syllables.estimate(line)
            if count != 7:
                print(f"  ✗ {letter}: '{line}' = {count} syllables (expected 7)")
                all_valid = False

    if all_valid:
        print("  ✓ All lines have correct syllable counts")

    return all_valid


def build_acrostic_haiku_constraint(word):
    """
    Build constraint for haiku with acrostic (first letters spell word).

    Args:
        word: 3-letter word (e.g., "SUN")

    Returns:
        BoolFSA constraint
    """
    if len(word) != 3:
        raise ValueError("Acrostic word must be 3 letters for haiku")

    word = word.upper()

    # Get lines starting with each letter
    letter1, letter2, letter3 = word[0], word[1], word[2]

    # 5-syllable lines starting with letter1
    lines_5_1 = LINES_5_SYL_BY_LETTER.get(letter1, [])
    if not lines_5_1:
        raise ValueError(f"No 5-syllable lines starting with '{letter1}'")

    # 7-syllable lines starting with letter2
    lines_7_2 = LINES_7_SYL_BY_LETTER.get(letter2, [])
    if not lines_7_2:
        raise ValueError(f"No 7-syllable lines starting with '{letter2}'")

    # 5-syllable lines starting with letter3
    lines_5_3 = LINES_5_SYL_BY_LETTER.get(letter3, [])
    if not lines_5_3:
        raise ValueError(f"No 5-syllable lines starting with '{letter3}'")

    # Escape regex special characters
    def escape_regex(s):
        special_chars = r'\.^$*+?{}[]|()'
        for char in special_chars:
            s = s.replace(char, '\\' + char)
        return s

    # Build patterns
    pattern_1 = "(" + "|".join(escape_regex(line) for line in lines_5_1) + ")"
    pattern_2 = "(" + "|".join(escape_regex(line) for line in lines_7_2) + ")"
    pattern_3 = "(" + "|".join(escape_regex(line) for line in lines_5_3) + ")"

    # Full haiku pattern: line1\nline2\nline3
    haiku_pattern = pattern_1 + r"\n" + pattern_2 + r"\n" + pattern_3

    print(f"\nAcrostic constraint for '{word}':")
    print(f"  Line 1 (5 syl, starts with '{letter1}'): {len(lines_5_1)} options")
    print(f"  Line 2 (7 syl, starts with '{letter2}'): {len(lines_7_2)} options")
    print(f"  Line 3 (5 syl, starts with '{letter3}'): {len(lines_5_3)} options")
    print(f"  Total possible haikus: {len(lines_5_1)} × {len(lines_7_2)} × {len(lines_5_3)} = {len(lines_5_1) * len(lines_7_2) * len(lines_5_3)}")

    return BoolFSA.from_regex(haiku_pattern)


def count_syllables_simple(text: str) -> int:
    """Use the syllables library for accurate syllable counting."""
    return syllables.estimate(text)


def verify_haiku(text: str, acrostic_word: str) -> dict:
    """Verify if text is a valid 5-7-5 haiku with acrostic."""
    lines = text.strip().split('\n')

    if len(lines) != 3:
        return {
            'valid': False,
            'reason': f'Expected 3 lines, got {len(lines)}',
            'syllables': [],
            'lines': lines,
            'acrostic_valid': False
        }

    syllable_counts = [count_syllables_simple(line) for line in lines]
    target = [5, 7, 5]

    syllables_valid = syllable_counts == target

    # Check acrostic
    first_letters = ''.join(line[0].upper() if line else '' for line in lines)
    acrostic_valid = first_letters == acrostic_word.upper()

    valid = syllables_valid and acrostic_valid

    return {
        'valid': valid,
        'syllables': syllable_counts,
        'target': target,
        'lines': lines,
        'acrostic': first_letters,
        'acrostic_valid': acrostic_valid,
        'syllables_valid': syllables_valid,
        'reason': 'OK' if valid else f"Syllables {syllable_counts} != {target}" if not syllables_valid else f"Acrostic '{first_letters}' != '{acrostic_word.upper()}'"
    }


async def run_experiment(sampler_type: str, model, constraint, acrostic_word):
    """Run haiku generation experiment with given model and constraint."""
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
        verification = verify_haiku(text, acrostic_word)
        verification['weight'] = w
        verification['log_weight'] = logw
        verification['text'] = text
        verification['rank'] = i

        results.append(verification)

        if verification['valid']:
            valid_count += 1
            unique_haikus.add(text.strip())
            if valid_count <= 5:  # Show first 5 valid ones
                print(f"\n✓ Valid Haiku #{valid_count} (weight: {w:.4f}, logw: {logw:.2f}):")
                print(f"  Acrostic: {verification['acrostic']}")
                for j, line in enumerate(verification['lines']):
                    print(f"  Line {j+1} ({verification['syllables'][j]} syl): {line}")

    # Summary statistics
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
    # Clear GPU memory first
    torch.cuda.empty_cache()

    print("="*60)
    print("HAIKU + ACROSTIC EXPERIMENT")
    print("Comparing Token-level SMC vs Byte-level SMC")
    print(f"Acrostic word: {ACROSTIC_WORD}")
    print("="*60)

    # Verify syllable counts
    if not verify_syllable_counts():
        print("\n❌ Error: Some lines have incorrect syllable counts!")
        return

    # Build constraint
    base_constraint = build_acrostic_haiku_constraint(ACROSTIC_WORD)

    # 1. Run PromptedLLM (Token Level)
    print("\n" + "="*60)
    print("SETUP: PromptedLLM (Token-Level)")
    print("="*60)

    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)
    constraint_prompted = base_constraint.coerce(prompted, f=b"".join)

    results_token = await run_experiment("PromptedLLM (Token-Level)", prompted, constraint_prompted, ACROSTIC_WORD)
    prompted.model.clear_cache()
    torch.cuda.empty_cache()  # Clear GPU memory

    # 2. Run ByteLLM (Character Level)
    print("\n" + "="*60)
    print("SETUP: ByteLLM (Byte-Level)")
    print("="*60)

    llm = load_model_by_name(MODEL_NAME, backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]

    beam_params = BeamParams(K=8, prune_threshold=0.0, eos_tokens={model_eos_token})
    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)

    constraint_byte = base_constraint.coerce(byte_llm, f=b"".join)

    results_byte = await run_experiment("ByteLLM (Byte-Level)", byte_llm, constraint_byte, ACROSTIC_WORD)
    await byte_llm.cleanup()

    # Final Comparison
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

    # Determine winner
    if results_byte['valid_pct'] > results_token['valid_pct']:
        print("\n🏆 WINNER: Byte-level SMC (higher valid %)")
    elif results_token['valid_pct'] > results_byte['valid_pct']:
        print("\n🏆 WINNER: Token-level SMC (higher valid %)")
    else:
        print("\n🤝 TIE: Both achieved same valid %")

    if results_byte['unique_count'] > results_token['unique_count']:
        print("🏆 MORE DIVERSE: Byte-level SMC (more unique haikus)")
    elif results_token['unique_count'] > results_byte['unique_count']:
        print("🏆 MORE DIVERSE: Token-level SMC (more unique haikus)")

    if results_byte['avg_valid_logweight'] > results_token['avg_valid_logweight']:
        print("🏆 BETTER EXPLORATION: Byte-level SMC (higher log-prob on valid haikus)")
    elif results_token['avg_valid_logweight'] > results_byte['avg_valid_logweight']:
        print("🏆 BETTER EXPLORATION: Token-level SMC (higher log-prob on valid haikus)")


if __name__ == "__main__":
    asyncio.run(main())
