import asyncio
import os
from typing import List

import pandas as pd

from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import ByteLLM, AWRS
from genlm.eval.core import run_evaluation
from genlm.eval.core.model import ModelOutput, ModelResponse
from genlm.eval.domains.pattern_matching import (
    default_prompt_formatter,
    PatternPotential,
    PatternMatchingEvaluator,
    PatternMatchingDataset,
)


# Path to the pattern dataset (same as in test_eval.py)
PATTERNS_CSV = \
    "/Users/yemara/Desktop/genlm/genlm-eval/assets/pattern_matching/patterns.csv"


def build_bytelm():
    llm = load_model_by_name("gpt2", backend="hf")
    # Match test_eval.py EOS handling and healing config
    beam_params = BeamParams(
        K=1,
        prune_threshold=0.0,
        eos_tokens={b"\n", b"\n\n"},
        heal=True,
    )
    return ByteLLM(llm, beam_params)


async def pattern_model_adaptor(instance, output_dir: str, replicate: int) -> ModelOutput:
    # Globals initialized in main
    global BYTE_LLM

    # Prepare prompt for this instance
    prompt_ids = default_prompt_formatter(
        BYTE_LLM.llm.tokenizer, instance, use_chat_format=False
    )
    BYTE_LLM.set_prompt_from_str(BYTE_LLM.llm.tokenizer.decode(prompt_ids))

    # Build constraint and sampler
    potential = PatternPotential(instance.pattern).coerce(BYTE_LLM, f=b"".join)
    sampler = AWRS(BYTE_LLM, potential)

    # Generate sequences
    sequences = await sampler.smc(
        n_particles=5,
        ess_threshold=0.5,
        max_tokens=200,
        verbosity=1,  # print SMC particle states each step
    )

    # Show outputs for this instance
    print("Decoded posterior:", sequences.decoded_posterior)

    # Convert to ModelOutput (string responses with weights)
    responses: List[ModelResponse] = [
        ModelResponse(response=seq, weight=float(prob))
        for seq, prob in sequences.decoded_posterior.items()
    ]
    # Proactively cleanup beam/trie background tasks between instances
    await BYTE_LLM.cleanup()

    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    global BYTE_LLM
    BYTE_LLM = build_bytelm()

    # Quick sanity: show CSV headers (useful if column changes)
    print(pd.read_csv(PATTERNS_CSV, nrows=0).columns.tolist())

    dataset = PatternMatchingDataset.from_csv(PATTERNS_CSV, pattern_column="regex")
    evaluator = PatternMatchingEvaluator()

    # Limit to first 5 instances using max_instances
    results = await run_evaluation(
        dataset=dataset,
        model=pattern_model_adaptor,
        evaluator=evaluator,
        output_dir=None,  # set a path to cache outputs/results if desired
        n_replicates=1,
        overwrite_results=False,
        overwrite_outputs=False,
        max_instances=20,
        verbosity=1,
    )

    print("\nSummary:")
    print(f"n_instances: {results['n_instances']}")
    print(f"average_weighted_accuracy: {results['average_weighted_accuracy']}")

    # Clean up ByteLLM beam state resources
    await BYTE_LLM.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
