import asyncio
import numpy as np
from genlm.control import AWRS, BoolFSA, ByteLLM, PromptedLLM
from genlm.bytes import BeamParams
from genlm.backend import load_model_by_name
from genlm.control.constant import EndOfSequence
from transformers import GPT2Tokenizer
import torch

# Setup
MODEL_NAME = "gpt2"
PROMPT = "Pick a fruit: [" 
# EXPERIMENT DESIGN:
# 1. PromptedLLM (Token Level):
#    - P(apple) >> P(pineapple_start_token).
#    - Collapses to 'apple'.
# 2. ByteLLM (Character Level):
#    - P(p) > P(a) due to "Coalition Subsidy" (many fruits start with p).
#    - Keeps 'p' branch alive.
#    - Finds 'pineapple'.
# Note: 'apple' is a substring of 'pineapple', so careful with matching logic.
CHOICES = ["apple", "pineapple"] 
N_PARTICLES = 10
MAX_TOKENS = 10

async def run_experiment(sampler_type: str, model, constraint):
    print(f"\nRunning {sampler_type}...")
    sampler = AWRS(model, constraint)
    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=1,
    )
    await sampler.cleanup()
    
    # Analyze results
    counts = {c: 0 for c in CHOICES}
    norm_weights = sequences.normalized_weights
    
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    for i, ((seq, _), w) in enumerate(zip(sequences, norm_weights)):
        if w == 0: continue
        
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
        
        text = text.strip()
        
        # Match EXACTLY or sort by length descending to avoid substring issues
        # CHOICES sorted by length descending: ["pineapple", "apple"]
        sorted_choices = sorted(CHOICES, key=len, reverse=True)
        
        matched = False
        for choice in sorted_choices:
            if choice in text:
                counts[choice] += w
                matched = True
                break
        if not matched:
            # Debug unmatched
            # print(f"Unmatched text: {text}")
            pass
    
    total_mass = sum(counts.values())
    
    print(f"Results for {sampler_type}:")
    for choice in CHOICES:
        pct = (counts[choice] / total_mass * 100) if total_mass > 0 else 0
        print(f"  {choice}: {pct:.1f}%")
    
    return counts

def build_constraint(vocab_words):
    pattern = "(" + "|".join(vocab_words) + ")"
    regex = pattern
    return BoolFSA.from_regex(regex)

async def main():
    print(f"Comparing {CHOICES}...")
    
    # 2. Run PromptedLLM (Token Level)
    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)
    constraint_prompted = build_constraint(CHOICES).coerce(prompted, f=b"".join)
    
    await run_experiment("PromptedLLM (Token-Level)", prompted, constraint_prompted)
    prompted.model.clear_cache()

    # 3. Run ByteLLM (Character Level)
    llm = load_model_by_name(MODEL_NAME, backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    
    beam_params = BeamParams(K=16, prune_threshold=0.0, eos_tokens={model_eos_token})
    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)
    
    constraint_byte = build_constraint(CHOICES).coerce(byte_llm, f=b"".join)
    
    await run_experiment("ByteLLM (Char-Level)", byte_llm, constraint_byte)
    await byte_llm.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
