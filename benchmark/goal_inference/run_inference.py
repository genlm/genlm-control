import os
import sys
import json
import time
import pickle
import asyncio
import warnings
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import polars as pl
import subprocess
import re
import functools
import hashlib
import tempfile
from functools import lru_cache
from lark import Lark
from genlm_control.potential import Potential
import string

from genlm_control import InferenceEngine, PromptedLLM, BoolCFG

problem_text = "You are a PDDL expert, who writes valid PDDL code that \
describes user-provided planning problems directly without further \
explanations or texts.\n\n"

os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "600"


@functools.cache
def parsable_energy_function(pddl, context):
    task_hash = hashlib.sha256(pddl.encode()).hexdigest()
    task_filepath = f"benchmark/goal_inference/data/pddl_tasks/{task_hash}.pddl"
    if not os.path.exists(task_filepath):
        with open(task_filepath, "w") as f:
            f.write(pddl)

    plan_filepath = f"benchmark/goal_inference/data/pddl_plans/{task_hash}.pddl"
    if not os.path.exists(plan_filepath):
        proc = subprocess.run(
            [
                f'./fast-downward.sif --plan-file {plan_filepath} domain.pddl {task_filepath} --search "astar(ipdb())"'
            ],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

        if proc.returncode != 0:
            raise ValueError(
                f"Planner exited with nonzero status: {proc.stderr.decode()}"
            )

    generated_pddl = re.sub(
        "\(:goal \(and .*?\)\)\n", f"(:goal (and {context}))\n", pddl
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        task_filepath = os.path.join(tmpdir, "task.pddl")
        with open(task_filepath, "w") as f:
            f.write(generated_pddl)

        proc = subprocess.run(
            [f"Validate domain.pddl {task_filepath} {plan_filepath}"],
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            # planner exited with nonzero status
            return float("-inf")
        return 0.0


def energy_function(masked_pddl, context):
    context = context.replace(")))\n", ")")
    context = context.replace(")))", ")")
    context = context.replace("))", ")")
    parsable_context = context.rpartition(")")
    if not parsable_context[0]:
        return 0.0
    parsable_context = parsable_context[0] + ")"

    return parsable_energy_function(masked_pddl, parsable_context)


class GoalInference(Potential):
    def __init__(self, pddl, grammar):
        self.pddl = pddl
        self.parser = Lark(grammar)
        vocab = list(map(ord, string.printable))
        super().__init__(vocab)

    @lru_cache(maxsize=None)
    def _parse(self, query):
        return self.parser.parse(query)

    async def prefix(self, context):
        try:
            string = bytes(context).decode("utf-8")
        except Exception as e:
            return float("-inf")

        return energy_function(self.pddl, string)

    async def complete(self, context):
        try:
            string = bytes(context).decode("utf-8")
        except Exception as e:
            return float("-inf")

        return energy_function(self.pddl, string + ")")


project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warnings.filterwarnings("once", category=RuntimeWarning)

os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "360"


def make_sampler(
    sampler_name, llm, pddl, bool_cfg, grammar, sampler_args, time_sampler=False
):
    if sampler_name == "eager":
        from genlm_control.sampler import EagerSetSampler
        from benchmark.util import LoggedSetTokenSampler

        print("Loading EagerSetSampler")

        return LoggedSetTokenSampler(
            EagerSetSampler(llm, bool_cfg, **sampler_args), log_stats=time_sampler
        )
    elif sampler_name == "swar":
        from genlm_control.experimental.vegas import GumbelMaxAdaptiveRejectionSampler

        return GumbelMaxAdaptiveRejectionSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join)
            * GoalInference(pddl, grammar).coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "clip":
        from genlm_control.experimental.vegas import ClippedAdaptiveRejectionSampler

        return ClippedAdaptiveRejectionSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join)
            * GoalInference(pddl, grammar).coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "swor":
        from genlm_control.experimental.vegas import WithoutReplacementSampler

        return WithoutReplacementSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join)
            * GoalInference(pddl, grammar).coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "top-k":
        from genlm_control.sampler import TopKSetSampler
        from benchmark.util import LoggedSetTokenSampler

        return LoggedSetTokenSampler(
            TopKSetSampler(llm, bool_cfg, **sampler_args), log_stats=time_sampler
        )
    elif sampler_name == "rejection":
        from genlm_control.experimental.vegas import RejectionSampler

        return RejectionSampler(
            llm,
            bool_cfg.coerce(llm, f=b"".join)
            * GoalInference(pddl, grammar).coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "lm":
        from genlm_control.sampler import DirectTokenSampler

        return DirectTokenSampler(llm)
    else:
        raise ValueError(f"Unknown sampler: {sampler_name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Goal Inference inference with potential."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Maximum number of tokens."
    )
    parser.add_argument(
        "--grammar_dir",
        type=str,
        default="data/grammars",
        help="Path to the Goal Inference grammar directory.",
    )
    parser.add_argument(
        "--n_particles",
        type=int,
        default=10,
        help="Number of particles.",
    )
    parser.add_argument(
        "--ess_threshold",
        type=float,
        default=0.9,
        help="ESS threshold for resampling.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to write the inference results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing inference results in the output directory.",
    )
    parser.add_argument(
        "--sampler_name",
        type=str,
        default="eager",
        help="Name of the sampler to use.",
    )
    parser.add_argument(
        "--sampler_args",
        type=str,
        default="{}",
        help="Arguments for the sampler.",
    )
    parser.add_argument(
        "--time_sampler",
        action="store_true",
        help="Time the sampler operations.",
    )
    parser.add_argument(
        "--use_critic",
        action="store_true",
        help="Use the critic to check the table column references.",
    )
    parser.add_argument(
        "--lm_args",
        type=str,
        default="{}",
        help="Arguments to pass to the language model, provided to PromptedLLM at initialization.",
    )
    parser.add_argument(
        "--cache_clear_interval",
        type=int,
        default=100,
        help="Interval to clear the parser caches.",
    )
    parser.add_argument(
        "--verbosity",
        type=int,
        default=0,
        help="Verbosity level for inference. When set to 1, particles are printed at each step.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout for the inference engine.",
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    arg_path = os.path.join(args.output_dir, "args.json")
    with open(arg_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    df = pl.read_parquet(
        "hf://datasets/BatsResearch/planetarium/data/train-00000-of-00001.parquet"
    )
    goal_suffix = "(:goal (and [BLANK]))\n)"

    df = df.filter(pl.col("problem_pddl").str.contains("(:goal (and", literal=True))
    df = df.with_columns(
        goal_pddl=pl.col("problem_pddl")
        .str.split(by="(:goal (and ")
        .list.last()
        .str.replace("))\n)", "", literal=True),
        masked_pddl=pl.concat_str(
            pl.col("problem_pddl").str.split(by="(:goal").list.first(),
            pl.lit(goal_suffix),
        ),
        prefix_pddl=pl.concat_str(
            pl.col("problem_pddl").str.split(by="(:goal (and").list.first(),
            pl.lit("(:goal (and"),
        ),
        goal_natural_language=pl.concat_str(
            pl.lit("Your goal"),
            pl.col("natural_language").str.split(by="Your goal").list.last(),
        ),
    )

    seed = 1234
    n_examples = 500
    max_n_objects = 10
    df = df.filter(
        (pl.col("domain") == "blocksworld")
        & (pl.col("num_objects") < max_n_objects)
        & (pl.col("init_is_abstract") == 0)
        & (pl.col("goal_is_abstract") == 0)
    )
    # Remove rows with duplicated goal_natural_language
    df = df.unique(subset=["goal_natural_language"])

    df = df.sample(fraction=1, shuffle=True, seed=seed)
    df = df.head(n_examples)

    messages = [{"role": "system", "content": problem_text}]

    prompts = []
    for row in df.to_dicts():
        prompt_message = messages[:]
        prompt_message += [
            {
                "role": "user",
                "content": (
                    "Natural Language goal description: \n\n"
                    + row["goal_natural_language"]
                    + "\n\n"
                ),
            },
            {
                "role": "assistant",
                "content": row["prefix_pddl"],
            },
        ]
        prompt = "".join([m["content"] for m in prompt_message])
        prompts.append(prompt)

    sampler_cache = {}
    llm = PromptedLLM.from_name(args.model_name, **json.loads(args.lm_args))
    eos_tokens = [t for t in llm.vocab if b"))" in t]
    llm = llm.spawn_new_eos(eos_tokens)

    pbar = tqdm(
        enumerate(
            zip(prompts, df["problem_pddl"].to_list(), df["masked_pddl"].to_list())
        ),
        total=len(prompts),
        desc="Running inference",
    )
    grammar = open(os.path.join(args.grammar_dir, f"goal_inference.lark"), "r").read()
    bool_cfg = BoolCFG.from_lark(grammar)
    critic = None

    for i, (prompt, problem_pddl, masked_pddl) in pbar:
        if args.use_critic:
            if args.sampler_name == "eager":
                critic = GoalInference(problem_pddl, grammar).coerce(llm, f=b"".join)
            else:
                critic = bool_cfg.coerce(llm, f=b"".join) * GoalInference(
                    problem_pddl, grammar
                ).coerce(llm, f=b"".join)

        result_file = os.path.join(args.output_dir, f"{i}_result.pkl")

        if not args.overwrite and os.path.exists(result_file):
            pbar.set_postfix(status=f"Skipped {i} (exists)")
            continue

        pbar.set_postfix(status=f"{i}")

        llm.prompt_ids = llm.model.tokenizer.encode(
            prompt,
        )

        sampler_key = (grammar, args.sampler_name, args.sampler_args, problem_pddl)

        if sampler_key not in sampler_cache:
            sampler_cache[sampler_key] = make_sampler(
                args.sampler_name,
                llm,
                problem_pddl,
                bool_cfg,
                grammar,
                json.loads(args.sampler_args),
                args.time_sampler,
            )
        sampler = sampler_cache[sampler_key]

        start_time = time.time()

        try:
            sequences = await asyncio.wait_for(
                InferenceEngine(sampler, critic=critic)(
                    n_particles=args.n_particles,
                    max_tokens=args.max_tokens,
                    ess_threshold=args.ess_threshold,
                    json_path=os.path.join(args.output_dir, f"{i}_record.json"),
                    verbosity=args.verbosity,
                ),
                timeout=args.timeout,
            )
        except asyncio.TimeoutError:
            print(f"Inference for example {i} timed out after {args.timeout} seconds")
            sequences = None

        metadata = {
            "model_name": args.model_name,
            "lm_args": args.lm_args,
            "max_tokens": args.max_tokens,
            "sampler_name": args.sampler_name,
            "sampler_args": args.sampler_args,
            "inference_time": time.time() - start_time,
            "masked_pddl": masked_pddl,
            "problem_pddl": problem_pddl,
        }

        with open(result_file, "wb") as f:
            pickle.dump(
                {
                    "metadata": metadata,
                    "contexts": sequences.contexts if sequences else None,
                    "log_weights": sequences.log_weights if sequences else None,
                },
                f,
            )

        # Save proposal timing stats.
        if args.time_sampler:
            if hasattr(sampler, "get_stats"):
                stats = sampler.get_stats()
                stats["metadata"] = metadata
                with open(os.path.join(args.output_dir, f"{i}_stats.pkl"), "wb") as f:
                    pickle.dump(stats, f)
                sampler._reset_stats()
            else:
                warnings.warn("Sampler does not support timing stats", RuntimeWarning)

        if hasattr(sampler, "_save_cache"):
            sampler._save_cache()


if __name__ == "__main__":
    asyncio.run(main())
