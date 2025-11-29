import asyncio
import numpy as np

from genlm.control import AWRS
from genlm.control.potential.base import Potential
from genlm.control.potential.built_in.bytelm import ByteLLM, BeamParams

class WordLengthPotential(Potential):
    """ A potential that restricts the length of generated words. """
    def __init__(self, vocabulary, max_length, eos_tokens=None):
        super().__init__(vocabulary)
        self.max_length = max_length
        # For byte-level generation, eos_tokens are likely specific bytes or sequences.
        # ByteLLM typically doesn't emit "EOS tokens" in the token-LLM sense in the stream,
        # but let's handle potential special bytes if needed.
        self.eos_tokens = set(eos_tokens) if eos_tokens else set()

    async def complete(self, context):
        if self._is_valid(context):
            return 0.0
        return float('-inf')

    async def prefix(self, context):
        if self._is_valid(context):
            return 0.0
        return float('-inf')

    def _is_valid(self, context):
        # Filter out EOS tokens so they don't count towards word length
        clean_context = [b for b in context if b not in self.eos_tokens]
        
        if not clean_context:
            return True
            
        # Decode everything (safest, simplest approach as requested)
        # Note: clean_context is a list of bytes objects, so we join them first.
        try:
            text = b"".join(clean_context).decode("utf-8", errors="ignore")
        except Exception:
            return True # If it can't be decoded yet, we assume valid partial state
            
        # Split by whitespace
        words = text.split()
        
        if not words:
            return True
            
        # Check ONLY the last word. 
        # Previous words were already checked in previous steps of generation.
        last_word = words[-1]
        if len(last_word) > self.max_length:
            return False
            
        return True


async def main():
    # 1. Initialize ByteLLM
    # We use BeamParams to control the conversion from Token LM to Byte LM
    beam_params = BeamParams(K=5, prune_threshold=0.0)
    llm = ByteLLM.from_name("meta-llama/Llama-3.1-8B-Instruct", beam_params, backend="hf")
    llm.set_prompt_from_str("The Fed says")
    
    # 2. Create the potential
    # For ByteLLM, the vocabulary is single bytes [b'\x00', ..., b'\xff']
    word_length_potential = WordLengthPotential(vocabulary=llm.vocab, max_length=5)
    
    # 3. Use AWRS sampler
    # AWRS works with any Potential. Since ByteLLM is a Potential, it should work out of the box.
    token_sampler = AWRS(llm, word_length_potential)
    
    print("Starting generation with ByteLLM + WordLengthPotential...")
    
    # 4. Generate
    # Note: max_tokens here effectively means max_bytes since our "tokens" are bytes
    sequences = await token_sampler.smc(n_particles=10, ess_threshold=0.5, max_tokens=100, verbosity=1)
    
    print("\nResults:")
    print(sequences.decoded_posterior)
    print(sequences.posterior)


if __name__ == "__main__":
    asyncio.run(main())

