"""Diagnostic (NOT a gate): does the counter-based (threefry) picker make the on-device
burst draw RNG-match StepLoop?

Runs the gate-2 multitoken config (CASES["multitoken-boolfsa[a-z ]+"]) as burst vs live
StepLoop under two pickers:
  - gumbel_max      : torch.rand -> CUDA stream (burst) != CPU stream (StepLoop) => unpaired
  - threefry_gumbel : counter-based, device-agnostic => burst and StepLoop draw the same

Reports per-seed exact-context matches (n_match/N) + length/log_ml gaps. If the picker
works on-device, threefry's n_match jumps toward N and the gaps collapse to the warm-KV
residual only (a rare flip where a near-tied argmax moves under the warm-KV logit delta).

Box only. Run from tests/sampler/:
    VLLM_USE_FLASHINFER_SAMPLER=0 /root/genlm/genlm-venv/bin/python diag_multitoken_seeds.py [n_seeds]
"""

import sys
import time

import numpy as np

import test_engine_native as T  # _run_burst/_run_steploop/_controller (no module skip on box)
from gate2_cases import CASES, MODEL, PROMPT, EOS_BYTES
from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.util import set_draw_method

CASE = CASES["multitoken-boolfsa[a-z ]+"]
_BANK = [1234, 7, 99, 2024, 555, 31, 8, 17, 42, 123, 271, 314]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
SEEDS = _BANK[:N]


def build_llm():
    from genlm.backend.llm import AsyncVirtualLM

    print(f"loading {MODEL} vLLM engine ...", flush=True)
    t0 = time.perf_counter()
    model = AsyncVirtualLM.from_name(
        MODEL,
        engine_opts={
            "gpu_memory_utilization": 0.3,
            "max_model_len": 256,
            "enable_prefix_caching": True,
            "enforce_eager": True,
        },
    )
    print(f"engine ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return PromptedLLM(model, eos_byte_strings=EOS_BYTES)


def main():
    llm = build_llm()
    llm.set_prompt_from_str(PROMPT)
    Np = CASE.n_particles

    for method in ["gumbel_max", "threefry_gumbel"]:
        set_draw_method(method)
        nmatch, lgaps, mgaps = [], [], []
        for seed in SEEDS:
            make = lambda s=seed: CASE.sampler(llm, s)  # noqa: E731
            t0 = time.perf_counter()
            slow = T._run_steploop(make, Np, CASE.ess, CASE.max_tokens, seed)
            burst = T._run_burst(make, Np, CASE.ess, CASE.max_tokens, seed)
            nm = sum(a == b for a, b in zip(slow["contexts"], burst["contexts"]))
            sl = float(np.mean([len(c) for c in slow["contexts"]]))
            bl = float(np.mean([len(c) for c in burst["contexts"]]))
            nmatch.append(nm)
            lgaps.append(bl - sl)
            mgaps.append(burst["log_ml"] - slow["log_ml"])
            print(
                f"[{method:15s}] seed={seed:<5d} {time.perf_counter() - t0:5.1f}s "
                f"n_match={nm}/{Np} len_gap={bl - sl:+.2f} ml_gap={burst['log_ml'] - slow['log_ml']:+.3f}",
                flush=True,
            )
        print(
            f"=== {method}: mean n_match={np.mean(nmatch):.1f}/{Np} "
            f"| len_gap mean={np.mean(lgaps):+.3f} | ml_gap mean={np.mean(mgaps):+.3f} ===\n",
            flush=True,
        )


if __name__ == "__main__":
    main()
