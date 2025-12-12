from genlm.control import PromptedLLM, BoolFSA, AWRS
# model = PromptedLLM.from_name("google/gemma-2-2b")
llm = PromptedLLM.from_name("meta-llama/CodeLlama-7b-hf")
# llm = PromptedLLM.from_name("EleutherAI/llemma_7b")
llm.set_prompt_from_str("Here is my honest opinion:")

# Create a finite-state automaton potential using a regular expression.
fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")

# Coerce the FSA so that it operates on the token type of the language model.
coerced_fsa = fsa.coerce(llm, f=b"".join)

# Create a token sampler that combines the language model and FSA.
token_sampler = AWRS(llm, coerced_fsa)

# Generate text using SMC.
# Generation is asynchronous; use `await` if calling in an async context (like in an async
# function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
import asyncio
if __name__ == "__main__":
    sequences = asyncio.run(token_sampler.smc(
    n_particles=5, # Number of candidate sequences to maintain
    ess_threshold=0.5, # Threshold for resampling
    max_tokens=30, # Maximum sequence length
    verbosity=1 # Print particles at each step
))

    print(sequences.decoded_posterior)