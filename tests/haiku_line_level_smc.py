"""
Line-Level SMC for Haiku Generation

This implements a true line-level SMC where:
- Each SMC "step" generates an ENTIRE line (not a single token)
- Constraints are checked per-line (does this line have correct syllables?)
- Reweighting happens per-line
- Resampling happens per-line

Contrast with token-level SMC:
- Token-level: generate token → check constraint → reweight → resample (repeat)
- Line-level: generate full line → check constraint → reweight → resample (repeat for 3 lines)
"""

import asyncio
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
import syllables

from genlm.control import PromptedLLM


# Target syllables per line
TARGET_SYLLABLES = [5, 7, 5]


def count_syllables(text: str) -> int:
    """Count syllables in text using the syllables library."""
    words = text.split()
    total = 0
    for word in words:
        clean = ''.join(c for c in word.lower() if c.isalpha())
        if clean:
            total += syllables.estimate(clean)
    return total


def extract_lines(text: str) -> list[str]:
    """Extract completed lines from text."""
    if not text:
        return []
    parts = text.split('\n')
    # Return all complete lines (exclude last part if no trailing newline)
    if text.endswith('\n'):
        return [p for p in parts if p]  # All parts are complete
    else:
        return [p for p in parts[:-1] if p]  # Last part is incomplete


@dataclass
class LineParticle:
    """A particle in line-level SMC."""
    lines: list[str] = field(default_factory=list)  # Completed lines
    log_weight: float = 0.0
    alive: bool = True
    
    @property
    def text(self) -> str:
        """Get full text so far."""
        return '\n'.join(self.lines) + ('\n' if self.lines else '')
    
    @property
    def num_lines(self) -> int:
        return len(self.lines)
    
    def copy(self) -> 'LineParticle':
        return LineParticle(
            lines=self.lines.copy(),
            log_weight=self.log_weight,
            alive=self.alive
        )


def compute_ess(log_weights: np.ndarray) -> float:
    """Compute effective sample size from log weights."""
    # Normalize in log space
    max_w = np.max(log_weights)
    if np.isinf(max_w) and max_w < 0:
        return 0.0
    shifted = log_weights - max_w
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    return 1.0 / np.sum(weights ** 2)


def resample_particles(particles: list[LineParticle], rng: np.random.Generator) -> list[LineParticle]:
    """Resample particles based on weights."""
    n = len(particles)
    log_weights = np.array([p.log_weight for p in particles])
    
    # Handle all -inf case
    if np.all(np.isinf(log_weights) & (log_weights < 0)):
        print("  [WARNING] All particles have -inf weight!")
        return particles
    
    # Normalize weights
    max_w = np.max(log_weights)
    shifted = log_weights - max_w
    weights = np.exp(shifted)
    weights = weights / np.sum(weights)
    
    # Systematic resampling
    indices = []
    cumsum = np.cumsum(weights)
    u = rng.uniform(0, 1/n)
    i = 0
    for j in range(n):
        while cumsum[i] < u:
            i += 1
        indices.append(i)
        u += 1/n
    
    # Create new particles with reset weights
    new_particles = []
    for idx in indices:
        p = particles[idx].copy()
        p.log_weight = 0.0  # Reset weight after resampling
        new_particles.append(p)
    
    return new_particles


async def generate_single_line(
    llm: PromptedLLM,
    context: str,
    max_tokens: int = 50,
    temperature: float = 0.7,
    rng: np.random.Generator = None
) -> tuple[str, float]:
    """
    Generate a single line (until newline or EOS) from the LLM.
    
    Returns:
        (line_text, log_prob): The generated line and its log probability
    """
    if rng is None:
        rng = np.random.default_rng()
    
    generated_tokens = []
    total_log_prob = 0.0
    current_context = context.encode('utf-8') if isinstance(context, str) else context
    
    for _ in range(max_tokens):
        # Get next token distribution
        log_probs = await llm.logw_next(current_context)
        
        # Apply temperature
        if temperature != 1.0:
            log_probs = log_probs / temperature
        
        # Normalize to get probabilities
        max_lp = np.max(log_probs)
        if np.isinf(max_lp) and max_lp < 0:
            break  # No valid tokens
        
        probs = np.exp(log_probs - max_lp)
        probs = probs / np.sum(probs)
        
        # Sample token
        token_idx = rng.choice(len(probs), p=probs)
        token = llm.vocab[token_idx]
        
        # Record log prob (at temperature=1 for proper weighting)
        original_log_probs = await llm.logw_next(current_context)
        total_log_prob += original_log_probs[token_idx]
        
        generated_tokens.append(token)
        current_context = current_context + token
        
        # Check for newline or EOS
        token_str = token.decode('utf-8', errors='replace')
        if '\n' in token_str or token == llm.eos_token:
            break
    
    # Decode the generated line
    line_bytes = b''.join(generated_tokens)
    line_text = line_bytes.decode('utf-8', errors='replace').strip()
    
    return line_text, total_log_prob


def line_constraint_weight(line: str, target_syllables: int) -> float:
    """
    Compute constraint weight for a line.
    
    Returns:
        0.0 if syllables match exactly (exp(0) = 1)
        -inf if syllables don't match (exp(-inf) = 0)
    """
    actual = count_syllables(line)
    if actual == target_syllables:
        return 0.0  # Valid
    else:
        return float('-inf')  # Invalid


async def line_level_smc(
    llm: PromptedLLM,
    prompt: str,
    n_particles: int = 20,
    ess_threshold: float = 0.5,
    temperature: float = 0.7,
    max_tokens_per_line: int = 50,
    seed: int = 42,
    verbosity: int = 1
) -> list[str]:
    """
    Line-level SMC for haiku generation.
    
    The SMC loop operates at line granularity:
    1. For each line (3 total for haiku):
       a. Each particle generates a complete line
       b. Apply constraint (check syllable count)
       c. Reweight particles
       d. Resample if ESS drops below threshold
    
    Args:
        llm: The language model
        prompt: The prompt to condition on
        n_particles: Number of particles
        ess_threshold: Resample when ESS < ess_threshold * n_particles
        temperature: Sampling temperature
        max_tokens_per_line: Max tokens to generate per line
        seed: Random seed
        verbosity: 0=silent, 1=summary, 2=detailed
    
    Returns:
        List of completed haikus
    """
    rng = np.random.default_rng(seed)
    
    # Initialize particles
    particles = [LineParticle() for _ in range(n_particles)]
    
    if verbosity >= 1:
        print(f"\n{'='*60}")
        print(f"LINE-LEVEL SMC")
        print(f"{'='*60}")
        print(f"Particles: {n_particles}, ESS threshold: {ess_threshold}")
        print(f"Target syllables: {TARGET_SYLLABLES}")
        print(f"{'='*60}\n")
    
    # Generate 3 lines (haiku structure)
    for line_idx in range(3):
        target_syl = TARGET_SYLLABLES[line_idx]
        
        if verbosity >= 1:
            print(f"\n--- Generating Line {line_idx + 1} (target: {target_syl} syllables) ---")
        
        # Count alive particles
        alive_particles = [p for p in particles if p.alive]
        if not alive_particles:
            print("[ERROR] All particles dead!")
            break
        
        # STEP 1: Generate a complete line for each particle
        for i, particle in enumerate(particles):
            if not particle.alive:
                continue
            
            # Build context: prompt + previous lines
            context = prompt + particle.text
            
            # Generate line
            line, log_prob = await generate_single_line(
                llm, context, max_tokens_per_line, temperature, rng
            )
            
            # STEP 2: Apply constraint
            constraint_weight = line_constraint_weight(line, target_syl)
            
            # STEP 3: Reweight
            # Weight = LLM probability * constraint
            # In log space: log_weight += log_prob + constraint_weight
            particle.log_weight += log_prob + constraint_weight
            
            if constraint_weight == float('-inf'):
                particle.alive = False
                if verbosity >= 2:
                    actual_syl = count_syllables(line)
                    print(f"  Particle {i}: '{line}' ({actual_syl} syl) - REJECTED")
            else:
                particle.lines.append(line)
                if verbosity >= 2:
                    print(f"  Particle {i}: '{line}' ({target_syl} syl) - ACCEPTED")
        
        # Report statistics
        alive_count = sum(1 for p in particles if p.alive)
        if verbosity >= 1:
            print(f"  Alive particles: {alive_count}/{n_particles}")
        
        if alive_count == 0:
            print("[ERROR] All particles rejected!")
            break
        
        # STEP 4: Compute ESS and resample if needed
        alive_particles = [p for p in particles if p.alive]
        log_weights = np.array([p.log_weight for p in alive_particles])
        
        # Handle case where some particles are dead
        if len(alive_particles) < n_particles:
            # Fill dead particle slots with resampled alive ones
            if verbosity >= 1:
                print(f"  Filling {n_particles - len(alive_particles)} dead particle slots...")
            
            # Resample to fill back to n_particles
            particles = resample_particles(alive_particles, rng)
            while len(particles) < n_particles:
                idx = rng.integers(0, len(alive_particles))
                particles.append(alive_particles[idx].copy())
            particles = particles[:n_particles]
            
            if verbosity >= 1:
                print(f"  Resampled to {n_particles} particles")
        else:
            # Normal ESS-based resampling
            ess = compute_ess(log_weights)
            if verbosity >= 1:
                print(f"  ESS: {ess:.1f} / {n_particles}")
            
            if ess < ess_threshold * n_particles:
                if verbosity >= 1:
                    print(f"  ESS below threshold, resampling...")
                particles = resample_particles(particles, rng)
    
    # Collect completed haikus
    completed = []
    for p in particles:
        if p.alive and p.num_lines == 3:
            haiku = '\n'.join(p.lines)
            if haiku not in completed:  # Deduplicate
                completed.append(haiku)
    
    return completed


async def main():
    """Run line-level SMC haiku generation."""
    
    print("Loading Llama 3.1 8B Instruct...")
    llm = await PromptedLLM.from_name(
        "meta-llama/Llama-3.1-8B-Instruct",
        eos_tokens=["\n\n", "<|eot_id|>", "<|end_of_text|>"]
    )
    
    # Create prompt with chat template
    system_msg = "You are a haiku poet. Write haikus with exactly 5-7-5 syllable structure."
    user_msg = "Write a haiku about nature:"
    
    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    
    print(f"\nPrompt:\n{prompt}")
    
    # Run line-level SMC
    haikus = await line_level_smc(
        llm=llm,
        prompt=prompt,
        n_particles=30,
        ess_threshold=0.5,
        temperature=0.7,
        max_tokens_per_line=50,
        seed=42,
        verbosity=2
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(haikus)} unique haikus")
    print(f"{'='*60}\n")
    
    for i, haiku in enumerate(haikus):
        print(f"Haiku {i+1}:")
        print("-" * 40)
        for line in haiku.split('\n'):
            syl = count_syllables(line)
            print(f"  {line} ({syl} syllables)")
        print()


if __name__ == "__main__":
    asyncio.run(main())

