import asyncio
import os
from typing import List
import argparse

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
    "../genlm-eval/assets/pattern_matching/patterns.csv"


def build_bytelm():
    # llm = load_model_by_name("unsloth/Qwen3-4B-128K-GGUF", backend="hf")
    # llm = load_model_by_name("allenai/OLMo-2-0425-1B", backend="hf") 
    # llm = load_model_by_name("meta-llama/Llama-3.2-1B-Instruct", backend="hf")
    llm = load_model_by_name("gpt2-large", backend="hf")
    # llm = load_model_by_name("deepseek-ai/deepseek-coder-1.3b-base", backend="hf")
    # Match test_eval.py EOS handling and healing config
    beam_params = BeamParams(
        K=5,
        prune_threshold=0.0,
        eos_tokens={b"\n", b"\n\n"},
        # eos_tokens={b"\n"},
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
        max_tokens=100,
        verbosity=0,  # print SMC particle states each step
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)  # exclusive
    parser.add_argument(
        "--skip_idxs",
        type=str,
        default="",
        help="Comma or space separated original indices to skip (e.g., '133,275').",
    )
    args = parser.parse_args()

    global BYTE_LLM
    BYTE_LLM = build_bytelm()

    # Quick sanity: show CSV headers (useful if column changes)
    print(pd.read_csv(PATTERNS_CSV, nrows=0).columns.tolist())

    dataset = PatternMatchingDataset.from_csv(PATTERNS_CSV, pattern_column="regex")

    # Parse skip indices into a set
    skip_set = set()
    if args.skip_idxs:
        for tok in args.skip_idxs.replace(",", " ").split():
            try:
                skip_set.add(int(tok))
            except ValueError:
                pass

    # Slice and filter by original indices (start/end are applied to original order)
    selected = []
    for i, inst in enumerate(dataset):
        if i in skip_set:
            continue
        if i < args.start_idx:
            continue
        if args.end_idx is not None and i >= args.end_idx:
            break
        selected.append(inst)
    dataset = selected
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
        # max_instances=5,
        verbosity=1,
    )

    print("\nSummary:")
    print(f"n_instances: {results['n_instances']}")
    print(f"average_weighted_accuracy: {results['average_weighted_accuracy']}")

    # Clean up ByteLLM beam state resources
    await BYTE_LLM.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
