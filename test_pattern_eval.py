import pandas as pd
from itertools import islice
from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import ByteLLM, AWRS
from genlm.eval import run_evaluation
from genlm.eval import ModelOutput, ModelResponse
from genlm.eval.domains.pattern_matching import (
    default_prompt_formatter,
    PatternPotential,
    PatternMatchingEvaluator,
    PatternMatchingDataset,
)


llm = load_model_by_name("gpt2", backend="hf")
beam_params = BeamParams(K=1, prune_threshold=0.0, eos_tokens={b"\n", b"\n\n"}, heal=True)
bytelm = ByteLLM(llm, beam_params)



path = "/Users/yemara/Desktop/genlm/genlm-eval/assets/pattern_matching/patterns.csv"
print(pd.read_csv(path, nrows=0).columns.tolist())  # see the header names


dataset = PatternMatchingDataset.from_csv("/Users/yemara/Desktop/genlm/genlm-eval/assets/pattern_matching/patterns.csv", pattern_column="regex")
instance = next(iter(dataset))

# first_k = list(islice(dataset, 6))
# instance = first_k[-1]

print(instance.pattern)
prompt_ids = default_prompt_formatter(bytelm.llm.tokenizer, instance, use_chat_format=False)
bytelm.set_prompt_from_str(bytelm.llm.tokenizer.decode(prompt_ids))
potential = PatternPotential(instance.pattern).coerce(bytelm, f=b"".join)

sampler = AWRS(bytelm, potential)
async def run():
    sequences = await sampler.smc(
        n_particles=1,
        ess_threshold=0.5,
        max_tokens=300,
        verbosity=1,
    )
    return sequences

if __name__ == "__main__":
    import asyncio
    asyncio.run(run())