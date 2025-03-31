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
from functools import lru_cache

from genlm_control.potential import Potential
import partialsmiles as ps

from genlm_control import InferenceEngine, PromptedLLM, BoolCFG

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

warnings.filterwarnings("once", category=RuntimeWarning)

os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "360"


class PartialSMILES(Potential):
    def __init__(self):
        super().__init__(vocabulary=list(range(256)))

    @lru_cache(maxsize=None)
    def _parse(self, query):
        return self.parser.parse(query)

    async def prefix(self, context):
        string = bytes(context).decode("utf-8", errors="ignore")
        if len(string) > 0 and string[0] == " ":
            string = string[1:]
        return self._validate(string, partial=True)

    async def complete(self, context):
        string = bytes(context).decode("utf-8", errors="ignore")
        if len(string) > 0 and string[0] == " ":
            string = string[1:]
        return self._validate(string, partial=False)

    def _validate(self, smiles, partial):
        try:
            ps.ParseSmiles(smiles, partial=partial)
            return 0.0
        except Exception as e:
            return -np.inf


def make_sampler(sampler_name, llm, bool_cfg, sampler_args, time_sampler=False):
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
            PartialSMILES().coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "swor":
        from genlm_control.experimental.vegas import WithoutReplacementSampler

        return WithoutReplacementSampler(
            llm,
            PartialSMILES().coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )
    elif sampler_name == "top-k":
        from genlm_control.sampler import TopKSetSampler
        from benchmark.util import LoggedSetTokenSampler

        return LoggedSetTokenSampler(
            TopKSetSampler(llm, bool_cfg, **sampler_args), log_stats=time_sampler
        )
    elif sampler_name == "clip":
        from genlm_control.experimental.vegas import ClippedAdaptiveRejectionSampler

        return ClippedAdaptiveRejectionSampler(
            llm,
            PartialSMILES().coerce(llm, f=b"".join),
            **sampler_args,
            log_stats=time_sampler,
        )

    elif sampler_name == "rejection":
        from genlm_control.experimental.vegas import RejectionSampler

        return RejectionSampler(
            llm,
            PartialSMILES().coerce(llm, f=b"".join),
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
        description="Run Molecular Synthesis inference with PartialSMILES potential."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--raw_molecules_dir",
        type=str,
        default="data/molecules",
        help="Path to the raw molecules directory.",
    )
    parser.add_argument(
        "--max_tokens", type=int, default=100, help="Maximum number of tokens."
    )
    parser.add_argument(
        "--grammar_dir",
        type=str,
        default="data/grammars",
        help="Path to the Molecular Synthesis grammar directory.",
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

    seed = 1234
    n_molecules = 20
    n_prompts = 100

    import random

    molecules = open(
        os.path.join(args.raw_molecules_dir, "GDB17.50000000.smi")
    ).readlines()

    # %%
    prompt_format = {
        "instruction": "You are an expert in chemistry. You are given a list of molecules in SMILES format. You are asked to write another molecule in SMILES format with similar chemical properties.\n",
        "exemplar": lambda ex: f"Molecule: {ex}",
        "prediction": "Molecule:",
    }

    prompts = []
    random.seed(seed)
    for i in range(n_prompts):
        molecule_ids = random.sample(range(len(molecules)), n_molecules)
        sample_mols = [molecules[i] for i in molecule_ids]
        prompt = (
            prompt_format["instruction"]
            + "".join([prompt_format["exemplar"](m) for m in sample_mols])
            + prompt_format["prediction"]
        )
        prompts.append(prompt)

    sampler_cache = {}
    llm = PromptedLLM.from_name(args.model_name, **json.loads(args.lm_args))
    eos_tokens = [t for t in llm.vocab if b"\n" in t]
    llm = llm.spawn_new_eos(eos_tokens)

    pbar = tqdm(enumerate(prompts), total=len(prompts), desc="Running inference")
    grammar = open(os.path.join(args.grammar_dir, f"smiles.lark"), "r").read()
    bool_cfg = BoolCFG.from_lark(grammar)
    critic = None
    if args.use_critic:
        if args.sampler_name == "eager":
            critic = PartialSMILES().coerce(llm, f=b"".join)
        else:
            critic = PartialSMILES().coerce(llm, f=b"".join)

    for i, prompt in pbar:
        result_file = os.path.join(args.output_dir, f"{i}_result.pkl")

        if not args.overwrite and os.path.exists(result_file):
            pbar.set_postfix(status=f"Skipped {i} (exists)")
            continue

        pbar.set_postfix(status=f"{i}")

        # TODO check if this is correct
        llm.prompt_ids = llm.model.tokenizer.encode(
            prompt,
        )

        sampler_key = (grammar, args.sampler_name, args.sampler_args)

        if sampler_key not in sampler_cache:
            sampler_cache[sampler_key] = make_sampler(
                args.sampler_name,
                llm,
                bool_cfg,
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
