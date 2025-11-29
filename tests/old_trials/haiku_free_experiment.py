"""
Free Haiku Generation Experiment
Compare Token-level SMC vs Byte-level SMC on FREE haiku generation with syllable constraints.

Unlike the constrained version, this allows the LLM to generate ANY text,
but enforces 5-7-5 syllable pattern dynamically.
"""

import asyncio
import numpy as np
from genlm.control import AWRS, ByteLLM, PromptedLLM
from genlm.control.potential.base import Potential
from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
from genlm.control.constant import EndOfSequence
import syllables

MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
PROMPT = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWrite a short poem about nature.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
N_PARTICLES = 10
MAX_TOKENS = 100

SYLLABLE_PATTERNS = {
    "haiku": [5, 7, 5],           # Traditional haiku
    "reverse_haiku": [7, 5, 7],   # Inverted haiku
    "short": [3, 5, 3],           # Short form
    "long": [7, 9, 7],            # Extended form
    "ascending": [3, 5, 7],       # Growing lines
    "descending": [7, 5, 3],      # Shrinking lines
    "symmetric": [4, 6, 4],       # Even symmetric
    "heavy_middle": [3, 9, 3],    # Emphasis on middle
    "couplet": [5, 5],            # Two-line form
    "quatrain": [4, 4, 4, 4],     # Four equal lines
}

# Select which pattern to use (change this to test different constraints)
PATTERN_NAME = "haiku"
SYLLABLE_PATTERN = SYLLABLE_PATTERNS[PATTERN_NAME]


class SyllableHaikuPotential(Potential):
    """
    A potential that enforces haiku syllable constraints (5-7-5) on free text generation.

    The constraint checks:
    1. Line 1 must have exactly 5 syllables
    2. Line 2 must have exactly 7 syllables
    3. Line 3 must have exactly 5 syllables
    4. Must have exactly 3 lines (2 newlines)
    """

    def __init__(self, vocab, token_type, syllable_pattern=None):
        super().__init__(vocab, token_type)
        self.syllable_pattern = syllable_pattern if syllable_pattern else SYLLABLE_PATTERN
        self.max_syllables_per_line = max(self.syllable_pattern)

    def _decode_context(self, context):
        """Decode context to text string."""
        if not context:
            return ""
        # Handle both bytes and ints in context
        chunks = []
        for token in context:
            if isinstance(token, EndOfSequence):
                break
            if isinstance(token, bytes):
                chunks.append(token)
            elif isinstance(token, int):
                chunks.append(bytes([token]))
        return b"".join(chunks).decode("utf-8", errors="ignore")

    def _parse_haiku_state(self, text):
        """
        Parse current haiku state from text.

        Returns:
            dict with:
            - lines: list of completed lines
            - current_line: text of line in progress
            - current_syllables: syllables in current line
            - line_num: which line we're on (0, 1, 2)
            - valid: whether current state is valid
        """
        lines = text.split('\n')

        # Completed lines (all but last)
        completed_lines = lines[:-1]
        current_line = lines[-1] if lines else ""

        line_num = len(completed_lines)

        # Count syllables in current line
        current_syllables = syllables.estimate(current_line) if current_line.strip() else 0

        num_lines = len(self.syllable_pattern)

        # Check if we're beyond expected lines
        if line_num > num_lines:
            return {
                'lines': completed_lines,
                'current_line': current_line,
                'current_syllables': current_syllables,
                'line_num': line_num,
                'valid': False,
                'reason': f'Too many lines (max {num_lines})'
            }
        # If we have all completed lines but text continues, also invalid
        if line_num == num_lines and current_line.strip():
            return {
                'lines': completed_lines,
                'current_line': current_line,
                'current_syllables': current_syllables,
                'line_num': line_num,
                'valid': False,
                'reason': f'Too many lines (max {num_lines})'
            }

        # Check completed lines have correct syllable counts
        for i, line in enumerate(completed_lines):
            line_syllables = syllables.estimate(line) if line.strip() else 0
            expected = self.syllable_pattern[i]
            if line_syllables != expected:
                return {
                    'lines': completed_lines,
                    'current_line': current_line,
                    'current_syllables': current_syllables,
                    'line_num': line_num,
                    'valid': False,
                    'reason': f'Line {i+1} has {line_syllables} syllables, expected {expected}'
                }

        # Check if current line exceeds syllable budget
        if line_num < len(self.syllable_pattern):
            max_allowed = self.syllable_pattern[line_num]
            if current_syllables > max_allowed:
                return {
                    'lines': completed_lines,
                    'current_line': current_line,
                    'current_syllables': current_syllables,
                    'line_num': line_num,
                    'valid': False,
                    'reason': f'Line {line_num+1} has {current_syllables} syllables (max {max_allowed})'
                }

        return {
            'lines': completed_lines,
            'current_line': current_line,
            'current_syllables': current_syllables,
            'line_num': line_num,
            'valid': True,
            'reason': 'OK'
        }

    async def prefix(self, context):
        """Log weight of context as a prefix."""
        text = self._decode_context(context)
        state = self._parse_haiku_state(text)

        if not state['valid']:
            return float('-inf')

        return 0.0  # Neutral weight if valid so far

    async def complete(self, context):
        """Log weight of context as a complete haiku."""
        text = self._decode_context(context)
        state = self._parse_haiku_state(text)

        if not state['valid']:
            return float('-inf')

        num_lines = len(self.syllable_pattern)
        last_line_idx = num_lines - 1

        # Accept EOS when the last line hits its target syllables
        if state['line_num'] == last_line_idx:
            if state['current_syllables'] == self.syllable_pattern[last_line_idx]:
                return 0.0
            return float('-inf')

        # Also accept if a trailing newline was already emitted (all lines complete, empty next)
        if state['line_num'] == num_lines and not state['current_line'].strip():
            return 0.0

        return float('-inf')

    def _is_word_boundary(self, text):
        """Check if we're at a word boundary (space, newline, or punctuation)."""
        if not text:
            return True
        last_char = text[-1]
        return last_char in ' \n\t.,!?;:'

    async def logw_next(self, context):
        """
        Log weights for next token.

        KEY CHANGE: Only evaluate syllables at WORD BOUNDARIES.
        Mid-word: return neutral weights (0) - no constraint effect, no resampling trigger.
        At word boundary: evaluate syllables and apply constraints.

        This makes SMC effectively operate at word-level granularity.
        """
        text = self._decode_context(context)

        # ONLY evaluate at word boundaries!
        if not self._is_word_boundary(text):
            # Mid-word: return neutral weights - constraint has no effect here
            # This means no resampling will be triggered mid-word
            return self.make_lazy_weights(np.zeros(len(self.vocab_eos)))

        # At word boundary: NOW evaluate syllables
        state = self._parse_haiku_state(text)

        # If current state is invalid, block everything
        if not state['valid']:
            return self.make_lazy_weights(np.full(len(self.vocab_eos), float('-inf')))

        num_lines = len(self.syllable_pattern)
        last_line_idx = num_lines - 1

        # If we're beyond all lines, only allow EOS
        if state['line_num'] >= num_lines:
            weights = np.full(len(self.vocab_eos), float('-inf'))
            if self.eos_token in self.vocab_eos:
                eos_idx = self.lookup[self.eos_token]
                weights[eos_idx] = 0.0
            return self.make_lazy_weights(weights)

        target_syllables = self.syllable_pattern[state['line_num']]
        current_syllables = state['current_syllables']

        # Define newline token based on vocab type
        newline_token = b'\n' if self.token_type.name == 'bytes' else '\n'

        # If we have met or exceeded the syllable budget, force newline/EOS only
        if current_syllables >= target_syllables:
            weights = np.full(len(self.vocab_eos), float('-inf'))

            # Not last line: only allow newline
            if state['line_num'] < last_line_idx:
                if newline_token in self.vocab_eos:
                    nl_idx = self.lookup[newline_token]
                    weights[nl_idx] = 0.0
            # Last line: allow EOS (preferred) and optionally a trailing newline
            else:
                if self.eos_token in self.vocab_eos:
                    eos_idx = self.lookup[self.eos_token]
                    weights[eos_idx] = 0.0
                if newline_token in self.vocab_eos:
                    nl_idx = self.lookup[newline_token]
                    weights[nl_idx] = 0.0

            return self.make_lazy_weights(weights)

        # Otherwise, allow free generation
        return self.make_lazy_weights(np.zeros(len(self.vocab_eos)))


def count_syllables_simple(text: str) -> int:
    """Use the syllables library for accurate syllable counting."""
    return syllables.estimate(text)


def verify_poem(text: str, pattern: list) -> dict:
    """Verify if text matches the syllable pattern."""
    lines = text.strip().split('\n')
    expected_lines = len(pattern)

    if len(lines) != expected_lines:
        return {
            'valid': False,
            'reason': f'Expected {expected_lines} lines, got {len(lines)}',
            'syllables': [],
            'lines': lines
        }

    syllable_counts = [count_syllables_simple(line) for line in lines]

    valid = syllable_counts == pattern

    return {
        'valid': valid,
        'syllables': syllable_counts,
        'target': pattern,
        'lines': lines,
        'reason': 'OK' if valid else f'Syllables {syllable_counts} != {pattern}'
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
        verbosity=0,
    )
    await sampler.cleanup()

    norm_weights = sequences.normalized_weights
    log_weights = sequences.log_weights

    results = []
    valid_count = 0
    unique_haikus = set()

    print(f"\nProcessing {len(sequences)} sequences...")

    for i, ((seq, _), w, logw) in enumerate(zip(sequences, norm_weights, log_weights)):
        if w < 1e-6:
            continue

        # Decode sequence to text
        chunks = []
        for token in seq:
            if isinstance(token, EndOfSequence):
                break
            if isinstance(token, bytes):
                chunks.append(token)
            elif isinstance(token, int):
                chunks.append(bytes([token]))
        text = b"".join(chunks).decode("utf-8", errors="ignore")

        # Verify poem
        verification = verify_poem(text, SYLLABLE_PATTERN)
        verification['weight'] = w
        verification['log_weight'] = logw
        verification['text'] = text
        verification['rank'] = i

        results.append(verification)

        if verification['valid']:
            valid_count += 1
            unique_haikus.add(text.strip())
            print(f"\n✓ Valid Haiku #{valid_count} (weight: {w:.4f}, logw: {logw:.2f}):")
            for j, line in enumerate(verification['lines']):
                print(f"  Line {j+1} ({verification['syllables'][j]} syl): {line}")
        elif i < 5:  # Show first 5 invalid ones for debugging
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


async def main(pattern_name=None, pattern=None):
    global PATTERN_NAME, SYLLABLE_PATTERN
    if pattern_name:
        PATTERN_NAME = pattern_name
        SYLLABLE_PATTERN = pattern

    print("="*60)
    print("FREE POEM GENERATION EXPERIMENT")
    print("Comparing Token-level SMC vs Byte-level SMC")
    print(f"Pattern: {PATTERN_NAME} = {SYLLABLE_PATTERN}")
    print("="*60)

    # 1. Run PromptedLLM (Token Level)
    print("\n" + "="*60)
    print("SETUP: PromptedLLM (Token-Level)")
    print("="*60)

    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)

    # Create syllable constraint for token-level
    constraint_prompted = SyllableHaikuPotential(prompted.vocab, prompted.token_type, SYLLABLE_PATTERN)

    results_token = await run_experiment("PromptedLLM (Token-Level)", prompted, constraint_prompted)
    prompted.model.clear_cache()

    # 2. Run ByteLLM (Character Level)
    print("\n" + "="*60)
    print("SETUP: ByteLLM (Byte-Level)")
    print("="*60)

    # llm = load_model_by_name(MODEL_NAME, backend="hf")
    # model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    beam_params = BeamParams(K=5, prune_threshold=0.0, eos_tokens={b'<|end_of_text|>', b'<|eot_id|>'})
    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)

    # Create syllable constraint for byte-level
    constraint_byte = SyllableHaikuPotential(byte_llm.vocab, byte_llm.token_type, SYLLABLE_PATTERN)

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

    # Show some example unique haikus from each
    print("\n" + "="*60)
    print("EXAMPLE HAIKUS")
    print("="*60)

    print("\nToken-level SMC examples:")
    for i, haiku in enumerate(list(results_token['unique_haikus'])[:3]):
        print(f"\nHaiku {i+1}:")
        for line in haiku.split('\n'):
            print(f"  {line}")

    print("\nByte-level SMC examples:")
    for i, haiku in enumerate(list(results_byte['unique_haikus'])[:3]):
        print(f"\nHaiku {i+1}:")
        for line in haiku.split('\n'):
            print(f"  {line}")

    return {
        'pattern_name': PATTERN_NAME,
        'pattern': SYLLABLE_PATTERN,
        'token': results_token,
        'byte': results_byte
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pattern = sys.argv[1]
        if pattern == "all":
            # Run all patterns and collect results
            all_results = []
            for name in SYLLABLE_PATTERNS:
                print(f"\n\n{'#'*70}")
                print(f"# PATTERN: {name} = {SYLLABLE_PATTERNS[name]}")
                print(f"{'#'*70}\n")
                result = asyncio.run(main(pattern_name=name, pattern=SYLLABLE_PATTERNS[name]))
                all_results.append(result)

            # Aggregate summary
            print("\n\n" + "="*70)
            print("AGGREGATE SUMMARY ACROSS ALL PATTERNS")
            print("="*70)

            token_solved = sum(1 for r in all_results if r['token']['valid_count'] > 0)
            byte_solved = sum(1 for r in all_results if r['byte']['valid_count'] > 0)
            total_patterns = len(all_results)

            # Average log-weights (only for solved patterns)
            token_logweights = [r['token']['avg_valid_logweight'] for r in all_results
                               if r['token']['valid_count'] > 0]
            byte_logweights = [r['byte']['avg_valid_logweight'] for r in all_results
                              if r['byte']['valid_count'] > 0]

            token_avg_logw = np.mean(token_logweights) if token_logweights else float('-inf')
            byte_avg_logw = np.mean(byte_logweights) if byte_logweights else float('-inf')

            print(f"\n{'Metric':<35} {'Token SMC':>15} {'Byte SMC':>15}")
            print("-"*70)
            print(f"{'Patterns with ≥1 valid seq':<35} {token_solved:>15}/{total_patterns} {byte_solved:>15}/{total_patterns}")
            print(f"{'Avg log-weight (solved only)':<35} {token_avg_logw:>15.2f} {byte_avg_logw:>15.2f}")
            print("-"*70)

            if byte_solved > token_solved:
                print("\n🏆 WINNER: Byte-level SMC (solved more patterns)")
            elif token_solved > byte_solved:
                print("\n🏆 WINNER: Token-level SMC (solved more patterns)")
            else:
                print("\n🤝 TIE: Both solved same number of patterns")

            if byte_avg_logw > token_avg_logw:
                print("🏆 BETTER EXPLORATION: Byte-level SMC (higher avg log-prob)")
            elif token_avg_logw > byte_avg_logw:
                print("🏆 BETTER EXPLORATION: Token-level SMC (higher avg log-prob)")

            # Per-pattern breakdown
            print("\n" + "="*70)
            print("PER-PATTERN BREAKDOWN")
            print("="*70)
            print(f"\n{'Pattern':<20} {'Token Valid':<15} {'Byte Valid':<15} {'Winner':<15}")
            print("-"*70)
            for r in all_results:
                name = r['pattern_name']
                token_valid = r['token']['valid_count'] > 0
                byte_valid = r['byte']['valid_count'] > 0

                if token_valid and byte_valid:
                    # Both solved - compare by log-weight
                    if r['byte']['avg_valid_logweight'] > r['token']['avg_valid_logweight']:
                        winner = "Byte (better)"
                    elif r['token']['avg_valid_logweight'] > r['byte']['avg_valid_logweight']:
                        winner = "Token (better)"
                    else:
                        winner = "Tie"
                elif byte_valid:
                    winner = "Byte (only)"
                elif token_valid:
                    winner = "Token (only)"
                else:
                    winner = "Neither"

                token_str = "✓" if token_valid else "✗"
                byte_str = "✓" if byte_valid else "✗"
                print(f"{name:<20} {token_str:<15} {byte_str:<15} {winner:<15}")

        elif pattern in SYLLABLE_PATTERNS:
            asyncio.run(main(pattern_name=pattern, pattern=SYLLABLE_PATTERNS[pattern]))
        else:
            print(f"Unknown pattern: {pattern}")
            print(f"Available: {list(SYLLABLE_PATTERNS.keys())} or 'all'")
    else:
        asyncio.run(main())
