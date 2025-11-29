import asyncio
import syllables
from genlm.control import PromptedLLM, AWRS
from genlm.control.potential.base import Potential

SYLLABLE_PATTERN = [5, 7, 5]


class HaikuPotential(Potential):
    """
    A boolean potential that enforces the haiku 5-7-5 syllable pattern.
    
    This potential operates on BYTES (integers 0-255) since it's designed to be
    coerced to work with an LLM's token vocabulary via:
        coerced = haiku_constraint.coerce(llm, f=b"".join)
    
    The haiku is structured as:
    - Line 1: 5 syllables (ends with newline)
    - Line 2: 7 syllables (ends with newline)  
    - Line 3: 5 syllables (ends with EOS)
    
    Returns:
        0.0 for valid prefixes/completions
        -inf for invalid prefixes/completions
    """
    
    def __init__(self, pattern=None):
        self.pattern = pattern or SYLLABLE_PATTERN
        # Vocabulary is all single bytes as integers (0-255)
        # This is what FSAs use after to_bytes() conversion
        vocab = list(range(256))
        super().__init__(vocabulary=vocab)
    
    def _decode_context(self, context):
        """Convert context (list of byte integers) to string."""
        try:
            return bytes(context).decode("utf-8", errors="replace")
        except Exception:
            return ""
    
    def _count_syllables(self, text):
        """Count syllables in text using the syllables library."""
        text = text.strip()
        if not text:
            return 0
        return syllables.estimate(text)
    
    def _analyze_haiku(self, context):
        """
        Analyze the current haiku state.
        
        Returns dict with:
        - lines: list of completed lines
        - current_line: text of current (incomplete) line
        - line_syllables: syllable counts for completed lines
        - current_syllables: syllables in current line
        - is_valid_prefix: whether this could lead to a valid haiku
        - is_valid_complete: whether this is a complete valid haiku
        """
        text = self._decode_context(context)
        
        # Split by newlines to get lines
        parts = text.split("\n")
        
        # Completed lines are all parts except the last (which may be incomplete)
        completed_lines = parts[:-1] if len(parts) > 1 else []
        current_line = parts[-1] if parts else ""
        
        # Count syllables
        line_syllables = [self._count_syllables(line) for line in completed_lines]
        current_syllables = self._count_syllables(current_line)
        
        line_num = len(completed_lines)  # 0-indexed current line number
        
        result = {
            "lines": completed_lines,
            "current_line": current_line,
            "line_syllables": line_syllables,
            "current_syllables": current_syllables,
            "line_num": line_num,
            "is_valid_prefix": True,
            "is_valid_complete": False,
        }
        
        # Check completed lines have correct syllable counts
        for i, (actual, expected) in enumerate(zip(line_syllables, self.pattern)):
            if actual != expected:
                result["is_valid_prefix"] = False
                return result
        
        # Check we don't have too many lines
        if line_num > len(self.pattern):
            result["is_valid_prefix"] = False
            return result
        
        # If we're on a line (not past the last line), check we haven't exceeded its syllable budget
        if line_num < len(self.pattern):
            max_allowed = self.pattern[line_num]
            if current_syllables > max_allowed:
                result["is_valid_prefix"] = False
                return result
        
        # Check if complete: all lines done with correct syllables
        if line_num == len(self.pattern) and not current_line.strip():
            # All lines completed, check each has correct syllables
            if line_syllables == self.pattern:
                result["is_valid_complete"] = True
        
        return result
    
    async def prefix(self, context):
        """Return 0.0 if valid prefix, -inf otherwise."""
        state = self._analyze_haiku(context)
        return 0.0 if state["is_valid_prefix"] else float("-inf")
    
    async def complete(self, context):
        """Return 0.0 if valid complete haiku, -inf otherwise."""
        state = self._analyze_haiku(context)
        
        # For complete, we need all lines to have exact syllable counts
        if not state["is_valid_prefix"]:
            return float("-inf")
        
        text = self._decode_context(context)
        lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
        
        if len(lines) != len(self.pattern):
            return float("-inf")
        
        line_syllables = [self._count_syllables(line) for line in lines]
        if line_syllables == self.pattern:
            return 0.0
        
        return float("-inf")


def verify_haiku(text, pattern=SYLLABLE_PATTERN):
    """Verify if a generated text is a valid haiku."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    
    if len(lines) != len(pattern):
        return {
            "valid": False,
            "lines": lines,
            "syllables": [],
            "expected": pattern,
            "reason": f"Expected {len(pattern)} lines, got {len(lines)}"
        }
    
    syllable_counts = [syllables.estimate(line) for line in lines]
    valid = syllable_counts == pattern
    
    return {
        "valid": valid,
        "lines": lines,
        "syllables": syllable_counts,
        "expected": pattern,
        "reason": "Valid haiku!" if valid else f"Syllables {syllable_counts} != {pattern}"
    }


async def generate_haiku(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    n_particles=10,
    max_tokens=50,
    ess_threshold=0.5,
    temperature=0.8,
    verbosity=0,
    use_chat_template=True
):
    """
    Generate a haiku using controlled generation.
    
    Args:
        model_name: HuggingFace model name (default: Llama 3.1 8B Instruct)
        n_particles: Number of SMC particles (higher = better quality, slower)
        max_tokens: Maximum tokens to generate
        ess_threshold: ESS threshold for resampling (0-1)
        temperature: LLM temperature (higher = more random)
        verbosity: Print level (0=silent, 1=progress)
        use_chat_template: Whether to use chat template for instruct models
    
    Returns:
        Sequences object with generated haikus
    """
    print(f"Loading model: {model_name}")
    
    # Create language model potential with proper EOS tokens for Llama
    llm = PromptedLLM.from_name(
        model_name, 
        temperature=temperature,
        eos_tokens=[b"<|eot_id|>", b"<|eom_id|>"],  # Llama 3 EOS tokens
    )
    
    if use_chat_template and "Instruct" in model_name:
        # Use chat template for instruction-tuned models
        llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
            conversation=[
                {"role": "system", "content": "You are a haiku poet. Write haikus with exactly 5-7-5 syllables. Output only the haiku, nothing else. Each line should be on a new line."},
                {"role": "user", "content": "Write a haiku about nature."},
            ],
            tokenize=True,
            add_generation_prompt=True
        )
    else:
        # Fallback for base models - use few-shot examples
        prompt = """Here are some haiku poems (5-7-5 syllables):

An old silent pond
A frog jumps into the pond
Splash! Silence again

Autumn moonlight bright
A worm digs silently
Into the chestnut

"""
        llm.set_prompt_from_str(prompt)
    
    print(f"Creating haiku constraint (pattern: {SYLLABLE_PATTERN})")
    
    # Create haiku constraint potential (operates on single bytes)
    haiku_constraint = HaikuPotential(pattern=SYLLABLE_PATTERN)
    
    # Coerce constraint to work with LLM's token vocabulary
    # f=b"".join converts list of byte-tokens to single bytes object that the 
    # haiku constraint can process character by character
    coerced_constraint = haiku_constraint.coerce(llm, f=b"".join)
    
    print("Creating AWRS sampler")
    
    # Create token sampler using AWRS
    # AWRS efficiently combines LLM (proposal) with boolean constraint (filter)
    token_sampler = AWRS(llm, coerced_constraint)
    
    print(f"Running SMC with {n_particles} particles, max {max_tokens} tokens")
    
    # Run SMC inference
    sequences = await token_sampler.smc(
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        verbosity=verbosity
    )
    
    return sequences


async def main():
    """Main entry point for haiku generation."""
    
    print("=" * 60)
    print("HAIKU GENERATOR using GenLM-Control")
    print("=" * 60)
    print()
    
    # Generate haikus using Llama 3.1 8B Instruct
    sequences = await generate_haiku(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        n_particles=20,  # Number of particles for SMC
        max_tokens=60,   # Enough for a haiku
        ess_threshold=0.5,
        temperature=0.7,
        verbosity=1,
        use_chat_template=True
    )
    
    print("\n" + "=" * 60)
    print("GENERATED HAIKUS")
    print("=" * 60)
    
    # Display results
    print("\nPosterior distribution over decoded sequences:")
    print(sequences.decoded_posterior)
    
    # Verify each generated haiku
    print("\n" + "-" * 60)
    print("VERIFICATION")
    print("-" * 60)
    
    for i, (text, prob) in enumerate(sequences.decoded_posterior.items()):
        print(f"\n[{i+1}] Probability: {prob:.4f}")
        print(f"Text:\n{text}")
        
        verification = verify_haiku(text)
        print(f"Valid: {verification['valid']}")
        if verification['lines']:
            for j, line in enumerate(verification['lines']):
                syl = verification['syllables'][j] if j < len(verification['syllables']) else '?'
                print(f"  Line {j+1} ({syl} syllables): {line}")
        print(f"Reason: {verification['reason']}")


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())

