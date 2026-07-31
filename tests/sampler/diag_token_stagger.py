"""Diagnostic (NOT a gate): are free-running resamples ordinal-aligned?

Token-grain twisted resampling assumes every live row's twist was computed at the
same context length. This logs the live rows' context lengths at every in-burst
resample and each step's scheduled-vs-live group counts, alongside the paired
log_ml diff per seed.

Box/Mila only (real vLLM). Run:
    VLLM_USE_FLASHINFER_SAMPLER=0 python tests/sampler/diag_token_stagger.py
"""

import asyncio

import numpy as np

from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.sampler.token import DirectTokenSampler
from genlm.control.sampler.smc import Controller, StepLoop
from genlm.control.sampler import burst as burst_mod
from genlm.control.util import set_draw_method

from _harness import seed_all
from gate2_cases import MODEL, PROMPT, EOS_BYTES


STAGGER = {"mixed_resamples": 0, "partial_steps": 0, "resamples": 0}


def patch_probes():
    orig_rr = burst_mod._Burst.resample_realize
    orig_draw = burst_mod._Burst.draw

    def rr(self):
        c = self.d.controller
        lens = sorted(len(c.particles[r].context) for r in self.row_handle)
        crossed = orig_rr(self)
        if crossed:
            STAGGER["resamples"] += 1
            if len(set(lens)) > 1:
                STAGGER["mixed_resamples"] += 1
                print(f"[stagger] MIXED resample: live ctx lens={lens}", flush=True)
        return crossed

    def draw(self, logits, handles):
        live = len(self.handle_row)
        if len(handles) < live:
            STAGGER["partial_steps"] += 1
            print(f"[stagger] partial step: scheduled={len(handles)} live={live}",
                  flush=True)
        return orig_draw(self, logits, handles)

    burst_mod._Burst.resample_realize = rr
    burst_mod._Burst.draw = draw


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


def main():
    set_draw_method("threefry_gumbel")
    llm = PromptedLLM.from_name(
        MODEL, backend="vllm", eos_byte_strings=EOS_BYTES,
        engine_opts={"gpu_memory_utilization": 0.45, "max_model_len": 1024},
    )
    llm.set_prompt_from_str(PROMPT)
    prompt_ids = llm.model.tokenizer.encode(PROMPT)
    patch_probes()

    for seed in (1234, 7, 99, 2024, 555, 31):
        for k in STAGGER:
            STAGGER[k] = 0
        seed_all(seed)
        slow = asyncio.run(StepLoop(controller(llm, prompt_ids)).run())
        seed_all(seed)
        fast = asyncio.run(burst_mod.BurstLoop(controller(llm, prompt_ids)).run())
        print(f"seed {seed}: diff={log_ml(fast) - log_ml(slow):+.3f} "
              f"resamples={STAGGER['resamples']} "
              f"mixed={STAGGER['mixed_resamples']} "
              f"partial_steps={STAGGER['partial_steps']}", flush=True)


if __name__ == "__main__":
    main()
