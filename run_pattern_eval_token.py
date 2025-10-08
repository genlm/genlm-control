import asyncio
import argparse
from typing import List

import pandas as pd

from genlm.control import AWRS
from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.eval.core import run_evaluation
from genlm.eval.core.model import ModelOutput, ModelResponse
from genlm.eval.domains.pattern_matching import (
    default_prompt_formatter,
    PatternPotential,
    PatternMatchingEvaluator,
    PatternMatchingDataset,
)


# Path to the pattern dataset
PATTERNS_CSV = \
    "../genlm-eval/assets/pattern_matching/patterns.csv"


def build_tokenlm(model_name: str = "meta-llama/Llama-3.2-1B-Instruct") -> PromptedLLM:
    print(f"Building token model for {model_name}")
    return PromptedLLM.from_name(
        model_name,
        backend="hf",
        eos_tokens=[b"\n", b"\n\n"],
    )


async def pattern_model_adaptor(instance, output_dir: str, replicate: int) -> ModelOutput:
    global TOKEN_LLM

    # Prepare prompt for this instance
    prompt_ids = default_prompt_formatter(
        TOKEN_LLM.model.tokenizer, instance, use_chat_format=False
    )
    TOKEN_LLM.prompt_ids = prompt_ids

    # Build constraint and sampler (coerce regex potential to token vocab)
    potential = PatternPotential(instance.pattern).coerce(TOKEN_LLM, f=b"".join)
    sampler = AWRS(TOKEN_LLM, potential)

    # Generate sequences
    sequences = await sampler.smc(
        n_particles=5,
        ess_threshold=0.5,
        max_tokens=100,
        verbosity=0,
    )

    # Convert to ModelOutput (string responses with weights)
    responses: List[ModelResponse] = [
        ModelResponse(response=seq, weight=float(prob))
        for seq, prob in sequences.decoded_posterior.items()
    ]

    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)  # exclusive
    parser.add_argument(
        "--skip_idxs",
        type=str,
        default="",
        help="Comma or space separated original indices to skip (e.g., '133,275').",
    )
    args = parser.parse_args()

    global TOKEN_LLM
    TOKEN_LLM = build_tokenlm(args.model)

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

    results = await run_evaluation(
        dataset=dataset,
        model=pattern_model_adaptor,
        evaluator=evaluator,
        output_dir=None,
        n_replicates=1,
        overwrite_results=False,
        overwrite_outputs=False,
        verbosity=1,
    )

    print("\nSummary:")
    print(f"n_instances: {results['n_instances']}")
    print(f"average_weighted_accuracy: {results['average_weighted_accuracy']}")


if __name__ == "__main__":
    asyncio.run(main())
