import asyncio
import argparse
from typing import List

from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control import AWRS, direct_token_sampler
from genlm.eval.core import run_evaluation
from genlm.eval.core.model import ModelOutput, ModelResponse

from genlm.eval.domains.json_schema import (
    JSONSchemaBenchDataset,
    JSONSchemaBenchEvaluator,
    default_prompt_formatter,
)


def build_prompted_llm(model_name: str = "meta-llama/Llama-3.2-1B-Instruct") -> PromptedLLM:
    # Token-level model; use chat formatting downstream
    # EOS tokens: allow newline to be a possible stop, keep default tokenizer EOS as well
    return PromptedLLM.from_name(
        model_name,
        backend="hf",
        eos_tokens=[b"\n", b"\n\n"],
        temperature=1.0,
    )


async def json_model_adaptor(instance, output_dir: str, replicate: int) -> ModelOutput:
    # Build a fresh PromptedLLM per instance to set the prompt
    llm = build_prompted_llm()

    # Get chat-formatted prompt ids for the schema instance
    prompt_ids = default_prompt_formatter(
        tokenizer=llm.model.tokenizer,
        instance=instance,
        use_chat_format=True,
    )
    llm.prompt_ids = prompt_ids

    # No boolean constraint here; just token-level sampling
    sampler = direct_token_sampler(llm)

    # Use SMC wrapper for a small beam of candidates (no critic)
    sequences = await sampler.smc(
        n_particles=3,
        ess_threshold=0.5,
        max_tokens=128,
        verbosity=0,
    )

    # Convert decoded posterior (bytes) to strings
    responses: List[ModelResponse] = []
    for seq, prob in sequences.decoded_posterior.items():
        if isinstance(seq, (bytes, bytearray)):
            text = bytes(seq).decode("utf-8", errors="ignore")
        else:
            # Some backends may already return str
            text = str(seq)
        responses.append(ModelResponse(response=text, weight=float(prob)))

    return ModelOutput(responses=responses, runtime_seconds=None)


async def main():
    parser = argparse.ArgumentParser(description="JSONSchemaBench eval with token-level PromptedLLM")
    parser.add_argument(
        "--tasks",
        type=str,
        default="Glaiveai2K,JsonSchemaStore",
        help="Comma-separated JSONSchemaBench task names (see epfl-dlab/JSONSchemaBench)",
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--max_instances", type=int, default=10)
    args = parser.parse_args()

    # Build dataset from requested tasks (validation split by default)
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if len(task_list) == 1:
        dataset = JSONSchemaBenchDataset.from_tasks(task_list, split="val")
    else:
        # Load each task separately, then merge while namespacing unique_ids
        all_schemas = []
        all_tasks = []
        all_unique_ids = []
        for task in task_list:
            ds = JSONSchemaBenchDataset.from_tasks([task], split="val")
            for schema, uid in zip(ds.schemas, ds.unique_ids):
                all_schemas.append(schema)
                all_tasks.append(task)
                all_unique_ids.append(f"{task}:{uid}")  # ensure uniqueness across tasks
        dataset = JSONSchemaBenchDataset(all_schemas, all_tasks, all_unique_ids)

    evaluator = JSONSchemaBenchEvaluator()

    # Monkey-patch model name into builder if user provided a different one
    global build_prompted_llm
    _orig_builder = build_prompted_llm

    def _patched_build():
        return PromptedLLM.from_name(args.model, backend="hf", eos_tokens=[b"\n", b"\n\n"], temperature=1.0)

    build_prompted_llm = _patched_build  # type: ignore

    results = await run_evaluation(
        dataset=dataset,
        model=json_model_adaptor,
        evaluator=evaluator,
        output_dir=None,
        n_replicates=1,
        overwrite_results=False,
        overwrite_outputs=False,
        max_instances=args.max_instances,
        verbosity=1,
    )

    print("\nSummary:")
    print(f"n_instances: {results['n_instances']}")
    print(f"average_weighted_accuracy: {results['average_weighted_accuracy']}")


if __name__ == "__main__":
    asyncio.run(main())


