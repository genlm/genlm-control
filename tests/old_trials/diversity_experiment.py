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
# Fix prompt: remove trailing space? No, "List of fruits: " is fine.
# Maybe change constraint to regex that ALLOWS spaces/commas in between?
PROMPT = "List of fruits: " 
FRUITS = [
    "apple", "apricot", "avocado", "banana", "blackberry", "blueberry", 
    "cantaloupe", "cherry", "coconut", "cranberry", "date", "dragonfruit", 
    "durian", "elderberry", "fig", "grape", "grapefruit", "guava", "honeydew", 
    "jackfruit", "kiwi", "kumquat", "lemon", "lime", "lychee", "mango", 
    "melon", "mulberry", "nectarine", "orange", "papaya", "passionfruit", 
    "peach", "pear", "persimmon", "pineapple", "plum", "pomegranate", 
    "quince", "raspberry", "starfruit", "strawberry", "tangerine", "watermelon"
]
FRUITS = sorted(FRUITS, key=len, reverse=True)
N_PARTICLES = 50
MAX_TOKENS = 20

async def run_experiment(sampler_type: str, model, constraint):
    print(f"\nRunning {sampler_type}...")
    sampler = AWRS(model, constraint)
    sequences = await sampler.smc(
        n_particles=N_PARTICLES,
        ess_threshold=0.5,
        max_tokens=MAX_TOKENS,
        verbosity=0,
    )
    await sampler.cleanup()
    
    norm_weights = sequences.normalized_weights
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    
    found_fruits = set()
    
    for i, ((seq, _), w) in enumerate(zip(sequences, norm_weights)):
        if w < 1e-6: continue
        
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
        
        text = text.strip().lower()
        for fruit in FRUITS:
            if fruit in text:
                found_fruits.add(fruit)
    
    print(f"Unique Fruits Found by {sampler_type}: {len(found_fruits)}")
    print(f"Fruits: {sorted(list(found_fruits))}")
    return found_fruits

def build_constraint(vocab_words):
    # Allow spaces before word?
    # pattern = "(" + "|".join(vocab_words) + ")"
    # But prompt "List of fruits: " -> might naturally want space?
    # Let's allow optional space
    pattern = " *" + "(" + "|".join(vocab_words) + ")"
    regex = pattern
    return BoolFSA.from_regex(regex)

async def main():
    print(f"Targeting {len(FRUITS)} fruits...")
    
    # 2. Run PromptedLLM (Token Level)
    prompted = PromptedLLM.from_name(MODEL_NAME, backend="hf")
    prompted.set_prompt_from_str(PROMPT)
    constraint_prompted = build_constraint(FRUITS).coerce(prompted, f=b"".join)
    
    fruits_token = await run_experiment("PromptedLLM (Token-Level)", prompted, constraint_prompted)
    prompted.model.clear_cache()

    # 3. Run ByteLLM (Character Level)
    llm = load_model_by_name(MODEL_NAME, backend="hf")
    model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
    
    beam_params = BeamParams(K=16, prune_threshold=0.0, eos_tokens={model_eos_token})
    byte_llm = ByteLLM.from_name(MODEL_NAME, beam_params=beam_params, backend="hf")
    byte_llm.set_prompt_from_str(PROMPT)
    
    constraint_byte = build_constraint(FRUITS).coerce(byte_llm, f=b"".join)
    
    fruits_byte = await run_experiment("ByteLLM (Char-Level)", byte_llm, constraint_byte)
    await byte_llm.cleanup()
    
    # Comparison
    only_token = fruits_token - fruits_byte
    only_byte = fruits_byte - fruits_token
    print(f"\nComparison:")
    print(f"Found ONLY by Token SMC: {only_token}")
    print(f"Found ONLY by Byte SMC: {only_byte}")

if __name__ == "__main__":
    asyncio.run(main())
