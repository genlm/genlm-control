"""
Haiku Generator with Line-Boundary Resampling (Approach A)

Token-level SMC loop with resampling only at line boundaries:
- Extend: Token-by-token (with AWRS filtering)
- Reweight: Token-by-token
- Resample: Only when a line completes (\n or EOS)

This keeps lines coherent by not resampling mid-line.
"""

import asyncio
import numpy as np
import syllables
from genlm.control import PromptedLLM, AWRS, EOS
from genlm.control.potential.base import Potential


# Standard haiku syllable pattern
SYLLABLE_PATTERN = [5, 7, 5]


class HaikuPotential(Potential):
    """
    Boolean potential enforcing 5-7-5 syllable haiku pattern.
    Operates on bytes (0-255), coerced to LLM token space.
    """
    
    def __init__(self, pattern=None):
        self.pattern = pattern or SYLLABLE_PATTERN
        vocab = list(range(256))
        super().__init__(vocabulary=vocab)
    
    def _decode_context(self, context):
        try:
            return bytes(context).decode("utf-8", errors="replace")
        except Exception:
            return ""
    
    def _count_syllables(self, text):
        text = text.strip()
        if not text:
            return 0
        return syllables.estimate(text)
    
    def _analyze_haiku(self, context):
        text = self._decode_context(context)
        parts = text.split("\n")
        
        completed_lines = parts[:-1] if len(parts) > 1 else []
        current_line = parts[-1] if parts else ""
        
        line_syllables = [self._count_syllables(line) for line in completed_lines]
        current_syllables = self._count_syllables(current_line)
        line_num = len(completed_lines)
        
        result = {
            "lines": completed_lines,
            "current_line": current_line,
            "line_syllables": line_syllables,
            "current_syllables": current_syllables,
            "line_num": line_num,
            "is_valid_prefix": True,
        }
        
        # Check completed lines have correct syllable counts
        for i, (actual, expected) in enumerate(zip(line_syllables, self.pattern)):
            if actual != expected:
                result["is_valid_prefix"] = False
                return result
        
        # Check not too many lines
        if line_num > len(self.pattern):
            result["is_valid_prefix"] = False
            return result
        
        # Check current line doesn't exceed budget
        if line_num < len(self.pattern):
            if current_syllables > self.pattern[line_num]:
                result["is_valid_prefix"] = False
                return result
        
        return result
    
    async def prefix(self, context):
        state = self._analyze_haiku(context)
        return 0.0 if state["is_valid_prefix"] else float("-inf")
    
    async def complete(self, context):
        state = self._analyze_haiku(context)
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


class Particle:
    """Represents a single SMC particle."""
    
    def __init__(self, context=None, log_weight=0.0):
        self.context = context if context is not None else []
        self.log_weight = log_weight
        self.finished = False
    
    def copy(self):
        p = Particle(self.context.copy(), self.log_weight)
        p.finished = self.finished
        return p
    
    def decode(self):
        """Decode context to string."""
        tokens = [t for t in self.context if t is not EOS]
        try:
            return b"".join(tokens).decode("utf-8", errors="replace")
        except:
            return str(tokens)


def compute_ess(log_weights):
    """Compute Effective Sample Size from log weights."""
    # Normalize in log space
    max_w = np.max(log_weights)
    weights = np.exp(log_weights - max_w)
    weights = weights / np.sum(weights)
    ess = 1.0 / np.sum(weights ** 2)
    return ess


def resample_particles(particles, rng=None):
    """Systematic resampling of particles."""
    if rng is None:
        rng = np.random.default_rng()
    
    n = len(particles)
    log_weights = np.array([p.log_weight for p in particles])
    
    # Normalize weights
    max_w = np.max(log_weights)
    weights = np.exp(log_weights - max_w)
    weights = weights / np.sum(weights)
    
    # Systematic resampling
    positions = (rng.random() + np.arange(n)) / n
    cumsum = np.cumsum(weights)
    indices = np.searchsorted(cumsum, positions)
    
    # Create new particles with reset weights
    new_particles = []
    for i in indices:
        p = particles[i].copy()
        p.log_weight = 0.0  # Reset weight after resampling
        new_particles.append(p)
    
    return new_particles


def is_at_line_boundary(context):
    """Check if we just completed a line (last token ends with newline)."""
    if not context:
        return False
    
    last_token = context[-1]
    if last_token is EOS:
        return True
    
    if isinstance(last_token, bytes):
        return last_token.endswith(b"\n")
    
    return False


def count_completed_lines(context):
    """Count how many lines have been completed."""
    try:
        tokens = [t for t in context if t is not EOS]
        text = b"".join(tokens).decode("utf-8", errors="replace")
        # Count newlines = completed lines
        return text.count("\n")
    except:
        return 0


async def smc_line_boundary_resample(
    token_sampler,
    n_particles=20,
    max_tokens=60,
    ess_threshold=0.5,
    verbosity=1,
    seed=None
):
    """
    Custom SMC with resampling only at line boundaries.
    
    Args:
        token_sampler: AWRS sampler for token-level sampling
        n_particles: Number of particles
        max_tokens: Maximum tokens to generate
        ess_threshold: ESS threshold for resampling (as fraction of n_particles)
        verbosity: 0=silent, 1=progress, 2=detailed
        seed: Random seed
    
    Returns:
        List of (context, log_weight) tuples
    """
    rng = np.random.default_rng(seed)
    
    # Get the start weight (weight of empty sequence)
    start_weight = await token_sampler.start_weight()
    
    # Initialize particles with start weight
    particles = [Particle(context=[], log_weight=start_weight) for _ in range(n_particles)]
    
    if verbosity >= 1:
        print(f"Initialized {n_particles} particles with start weight: {start_weight:.2f}")
    
    # Track statistics
    resample_points = []
    
    for step in range(max_tokens):
        # Count active particles
        active = [p for p in particles if not p.finished]
        if not active:
            if verbosity >= 1:
                print(f"Step {step}: All particles finished")
            break
        
        # === EXTEND: Sample next token for each active particle ===
        step_tokens = []  # Track tokens sampled this step
        for particle in active:
            try:
                # Use AWRS to sample a valid token
                # sample() returns (token, log_weight, log_prob)
                token, log_weight_increment, _ = await token_sampler.sample(
                    particle.context
                )
                
                # Check if sampling failed (returned -inf weight)
                if log_weight_increment == float("-inf"):
                    particle.log_weight = float("-inf")
                    particle.finished = True
                    step_tokens.append(("DEAD", float("-inf")))
                    continue
                
                particle.context.append(token)
                particle.log_weight += log_weight_increment
                
                # Track token for logging
                if token is EOS:
                    step_tokens.append(("EOS", log_weight_increment))
                    particle.finished = True
                else:
                    try:
                        tok_str = token.decode("utf-8", errors="replace") if isinstance(token, bytes) else str(token)
                    except:
                        tok_str = str(token)
                    step_tokens.append((tok_str, log_weight_increment))
                    
            except Exception as e:
                # No valid tokens available - particle dies
                if verbosity >= 2:
                    print(f"  Particle died: {e}")
                particle.log_weight = float("-inf")
                particle.finished = True
                step_tokens.append(("ERROR", float("-inf")))
        
        # Log token-level details
        if verbosity >= 2 and step_tokens:
            unique_tokens = {}
            for tok, lw in step_tokens:
                if tok not in unique_tokens:
                    unique_tokens[tok] = []
                unique_tokens[tok].append(lw)
            
            tok_summary = ", ".join([f"{tok!r}({np.mean(lws):.2f})" for tok, lws in unique_tokens.items()])
            print(f"  Step {step} tokens: {tok_summary}")
        
        # === Check if ANY particle just hit a line boundary ===
        # A line boundary is: newline character or EOS
        at_boundary = False
        for p in particles:
            if p.context and is_at_line_boundary(p.context):
                at_boundary = True
                break
        
        # Count completed lines for reporting
        n_lines = 0
        if particles:
            sample_particle = particles[0]
            n_lines = count_completed_lines(sample_particle.context)
        
        # === RESAMPLE: Only at line boundaries ===
        active_particles = [p for p in particles if not p.finished and p.log_weight > float("-inf")]
        
        if at_boundary and active_particles:
            log_weights = np.array([p.log_weight for p in particles])
            # Handle -inf weights
            valid_mask = log_weights > float("-inf")
            if np.sum(valid_mask) > 0:
                ess = compute_ess(log_weights[valid_mask])
                ess_fraction = ess / np.sum(valid_mask)
                
                if verbosity >= 1:
                    sample_text = particles[0].decode()[:50]
                    print(f"Step {step} | Line boundary | Lines: {n_lines} | ESS: {ess:.1f} ({ess_fraction:.2f}) | Sample: {sample_text!r}")
                
                if ess_fraction < ess_threshold:
                    if verbosity >= 1:
                        print(f"  → RESAMPLING (ESS {ess:.1f} < {ess_threshold * n_particles:.1f})")
                    
                    # Only resample non-finished particles, keep finished ones
                    finished_particles = [p for p in particles if p.finished]
                    active_particles = [p for p in particles if not p.finished]
                    
                    if active_particles:
                        resampled = resample_particles(active_particles, rng)
                        particles = finished_particles + resampled
                    
                    resample_points.append((step, n_lines))
        
        elif verbosity >= 2:
            # Non-boundary step - show weight distribution
            active_weights = [p.log_weight for p in particles if not p.finished and p.log_weight > float("-inf")]
            if active_weights:
                sample_text = particles[0].decode()[:60] if particles else ""
                print(f"Step {step} | Mid-line | Weights: [{min(active_weights):.1f}, {np.mean(active_weights):.1f}, {max(active_weights):.1f}] | {sample_text!r}")
    
    if verbosity >= 1:
        print(f"\nResampling occurred at steps: {resample_points}")
    
    # Return results
    results = [(p.context, p.log_weight) for p in particles if p.log_weight > float("-inf")]
    return results


def verify_haiku(text, pattern=SYLLABLE_PATTERN):
    """Verify if generated text is a valid haiku."""
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


async def main():
    print("=" * 70)
    print("HAIKU GENERATOR - Line Boundary Resampling (Approach A)")
    print("=" * 70)
    print()
    print("Strategy: Token-level SMC, resample ONLY at line boundaries")
    print()
    
    # Load model
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading model: {model_name}")
    
    llm = PromptedLLM.from_name(
        model_name,
        temperature=0.7,
        eos_tokens=[b"<|eot_id|>", b"<|eom_id|>"],
    )
    
    # Apply chat template
    llm.prompt_ids = llm.model.tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": "You are a haiku poet. Write haikus with exactly 5-7-5 syllables. Output only the haiku, nothing else. Each line should be on a new line."},
            {"role": "user", "content": "Write a haiku about nature."},
        ],
        tokenize=True,
        add_generation_prompt=True
    )
    
    print("Creating haiku constraint...")
    haiku_constraint = HaikuPotential(pattern=SYLLABLE_PATTERN)
    coerced_constraint = haiku_constraint.coerce(llm, f=b"".join)
    
    print("Creating AWRS token sampler...")
    token_sampler = AWRS(llm, coerced_constraint)
    
    print(f"Running SMC with LINE-BOUNDARY resampling...")
    print("-" * 70)
    
    results = await smc_line_boundary_resample(
        token_sampler,
        n_particles=20,
        max_tokens=60,
        ess_threshold=0.5,
        verbosity=2,  # Increased verbosity to see token-level logprobs
        seed=42
    )
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    # Decode and deduplicate results
    decoded_results = {}
    for context, log_weight in results:
        tokens = [t for t in context if t is not EOS]
        try:
            text = b"".join(tokens).decode("utf-8", errors="replace")
        except:
            continue
        
        if text not in decoded_results:
            decoded_results[text] = log_weight
        else:
            # Keep the higher weight
            decoded_results[text] = max(decoded_results[text], log_weight)
    
    # Sort by weight
    sorted_results = sorted(decoded_results.items(), key=lambda x: -x[1])
    
    print(f"\nGenerated {len(sorted_results)} unique sequences:")
    
    for i, (text, log_weight) in enumerate(sorted_results[:10]):
        print(f"\n[{i+1}] Log weight: {log_weight:.2f}")
        print(f"Text:\n{text}")
        
        verification = verify_haiku(text)
        print(f"Valid: {verification['valid']}")
        if verification['lines']:
            for j, line in enumerate(verification['lines']):
                syl = verification['syllables'][j] if j < len(verification['syllables']) else '?'
                print(f"  Line {j+1} ({syl} syllables): {line}")
        print(f"Reason: {verification['reason']}")
    
    # Summary statistics
    valid_count = sum(1 for text, _ in sorted_results if verify_haiku(text)['valid'])
    print(f"\n" + "=" * 70)
    print(f"SUMMARY: {valid_count}/{len(sorted_results)} valid haikus ({100*valid_count/len(sorted_results):.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

