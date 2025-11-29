import asyncio
import numpy as np

from genlm.control import AWRS, PromptedLLM
from genlm.control.constant import EndOfSequence
from genlm.control.potential.base import Potential

class WordLengthPotential(Potential):
    """ A potential that restricts the length of generated words. """
    def __init__(self, vocabulary, max_length, eos_tokens=None):
        super().__init__(vocabulary)
        self.max_length = max_length
        self.eos_tokens = set(eos_tokens) if eos_tokens else set()

    async def complete(self, context):
        # This potential doesn't enforce a specific sequence length, 
        # so any complete sequence that satisfies the word length constraint is valid.
        if self._is_valid(context):
            return 0.0
        return float('-inf')

    async def prefix(self, context):
        # Check if the sequence so far satisfies the constraint
        if self._is_valid(context):
            return 0.0
        return float('-inf')

    def _is_valid(self, context):
        # Filter out EOS tokens so they don't count towards word length
        clean_context = [b for b in context if b not in self.eos_tokens]
        
        # Decode the sequence of byte-tokens into a string
        # errors='ignore' handles potential partial multi-byte characters at the very end of the buffer
        text = b"".join(clean_context).decode("utf-8", errors="ignore")
        
        # Split by whitespace to get "words"
        words = text.split()
        
        for word in words:
            if len(word) > self.max_length:
                return False
        return True


async def main():
    llm = PromptedLLM.from_name("meta-llama/Llama-3.1-8B-Instruct", backend="hf")
    llm.set_prompt_from_str("The Fed says")
    
    # Create the potential with a max word length of 5 characters
    word_length_potential = WordLengthPotential(vocabulary=llm.vocab, max_length=5, eos_tokens=llm.eos_tokens)
    
    token_sampler = AWRS(llm, word_length_potential)
    
    # Generate text
    # We increase max_tokens to demonstrate it continues generating as long as words are short
    sequences = await token_sampler.smc(n_particles=5, ess_threshold=0.5, max_tokens=10, verbosity=1)
    print(sequences.decoded_posterior)
    print(sequences.posterior)
    


if __name__ == "__main__":
    asyncio.run(main())
