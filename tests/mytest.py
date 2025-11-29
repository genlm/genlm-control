from genlm.control import PromptedLLM, BoolFSA, AWRS, ByteLLM
import asyncio
from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams

llm = load_model_by_name("gpt2", backend="hf")
model_eos_token = llm.byte_vocab[llm.tokenizer.eos_token_id]
beam_params = BeamParams(K=5, prune_threshold=0.0, eos_tokens={model_eos_token})
llm = ByteLLM.from_name("gpt2", beam_params=beam_params, backend="hf")
    llm.set_prompt_from_str("My story is about:")

fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")
# fsa = BoolFSA.from_regex(r". Boulter starred in the 2011 film Mercenaries directed by Paris Leonti .")



coerced_fsa = fsa.coerce(llm, f=b"".join)

token_sampler = AWRS(llm, coerced_fsa)

async def main():
    return await token_sampler.smc(
        n_particles=5, # Number of candidate sequences to maintain
        ess_threshold=0.5, # Threshold for resampling
        max_tokens=150, # Maximum sequence length
        verbosity=1 # Print particles at each step
    )
# Generate text using SMC.
# Generation is asynchronous; use `await` if calling in an async context (like in an async
# function or in a Jupyter notebook) and `asyncio.run(token_sampler.smc(...))` otherwise.
if __name__ == "__main__":
    sequences = asyncio.run(main())
    print(sequences.decoded_posterior)
