"""Diagnostic (NOT a gate): per-lane repeatability of the token-grain LM-critic
config on a live engine. Runs each lane twice per seed in one process; a
same-seed same-lane log_ml mismatch = engine nondeterminism, located per lane.

Box/Mila only (real vLLM). Run:
    VLLM_USE_FLASHINFER_SAMPLER=0 python tests/sampler/diag_token_repeat.py
"""

import asyncio

import numpy as np

from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.sampler.token import DirectTokenSampler
from genlm.control.sampler.smc import Controller, StepLoop
from genlm.control.sampler.burst import BurstLoop
from genlm.control.util import set_draw_method

from _harness import seed_all
from gate2_cases import MODEL, PROMPT, EOS_BYTES


def controller(llm, prompt_ids):
    critic = PromptedLLM(llm.model, prompt_ids=prompt_ids,
                         eos_byte_strings=EOS_BYTES)
    return Controller(
        samplers=[DirectTokenSampler(llm)], critics=[critic],
        group_sizes=[8], ess_threshold=0.5, max_tokens=8,
        twist_with_critic=True,
    )


def log_ml(parts):
    lw = np.array([float(p.logw) for p in parts])
    return float(np.logaddexp.reduce(lw) - np.log(len(lw)))


def run(llm, prompt_ids, lane, seed):
    seed_all(seed)
    c = controller(llm, prompt_ids)
    driver = StepLoop(c) if lane == "slow" else BurstLoop(c)
    parts = asyncio.run(driver.run())
    ctxs = tuple(tuple(repr(t) for t in p.context) for p in parts)
    return log_ml(parts), ctxs


def main():
    set_draw_method("threefry_gumbel")
    llm = PromptedLLM.from_name(
        MODEL, backend="vllm", eos_byte_strings=EOS_BYTES,
        engine_opts={"gpu_memory_utilization": 0.45, "max_model_len": 1024},
    )
    llm.set_prompt_from_str(PROMPT)
    prompt_ids = llm.model.tokenizer.encode(PROMPT)

    for seed in (1234, 7, 99, 2024, 555, 31):
        vals = {}
        for lane in ("slow", "burst"):
            (m1, c1), (m2, c2) = (run(llm, prompt_ids, lane, seed) for _ in range(2))
            vals[lane] = (m1, m2)
            rep = "REPEATABLE" if c1 == c2 else "NONDETERMINISTIC"
            print(f"seed {seed} {lane:5s}: {m1:+.4f} / {m2:+.4f} "
                  f"ctx_equal={c1 == c2} {rep}", flush=True)
        print(f"seed {seed}: diff(run1)={vals['burst'][0] - vals['slow'][0]:+.3f} "
              f"diff(run2)={vals['burst'][1] - vals['slow'][1]:+.3f}", flush=True)


if __name__ == "__main__":
    main()
