"""Diagnostic (NOT a gate): does the burst's banked twist serving match the truth?

At every critic boundary the burst serves the critic LM leaf's ``batch_prefix``
from banked per-token warm-logit sums (``twist_logp``). This intercepts the
override seam and prints served-vs-recomputed (cold, engine-drained) values per
particle, per boundary — the drift, if any, is the bias mechanism.

Box/Mila only (real vLLM). Run:
    VLLM_USE_FLASHINFER_SAMPLER=0 python tests/sampler/diag_twist_serving.py
"""

import asyncio

import numpy as np

from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.potential import base as pot_base
from genlm.control.potential.coerce import Coerced
from genlm.control.sampler.token import DirectTokenSampler
from genlm.control.sampler.unit import MultiTokenUnitSampler, flatten_units
from genlm.control.sampler.smc import Controller
from genlm.control.sampler.burst import BurstLoop
from genlm.control.util import set_draw_method

from _harness import seed_all
from gate2_cases import MODEL, PROMPT, EOS_BYTES, ByteLengthBoundary


def patch_serving_probe():
    """Wrap PromptedLLM.batch_prefix: when a burst_prefix override serves values,
    ALSO recompute them cold (override cleared) and report the drift."""
    orig = PromptedLLM.batch_prefix

    async def probed(self, contexts):
        override = pot_base._burst_prefix_overrides.get()
        if not (override and self in override):
            return await orig(self, contexts)
        served = np.asarray(override[self], dtype=float)
        token = pot_base._burst_prefix_overrides.set(None)
        try:
            true = np.asarray(await orig(self, contexts), dtype=float)
        finally:
            pot_base._burst_prefix_overrides.reset(token)
        drift = served - true
        flag = "OK " if np.allclose(served, true, atol=1e-3) else "DRIFT"
        print(f"[serve] {flag} n={len(contexts)} max|d|={np.abs(drift).max():.6f} "
              f"served={np.round(served, 4).tolist()} true={np.round(true, 4).tolist()}",
              flush=True)
        return served

    PromptedLLM.batch_prefix = probed


def main():
    set_draw_method("threefry_gumbel")
    base_llm = PromptedLLM.from_name(
        MODEL, backend="vllm", eos_byte_strings=EOS_BYTES,
        engine_opts={"gpu_memory_utilization": 0.45, "max_model_len": 1024},
    )
    base_llm.set_prompt_from_str(PROMPT)
    prompt_ids = base_llm.model.tokenizer.encode(PROMPT)

    def make():
        return MultiTokenUnitSampler(
            subunit_sampler=DirectTokenSampler(base_llm),
            boundary_predicate=ByteLengthBoundary(3),
        )

    def make_critic():
        lm = PromptedLLM(base_llm.model, prompt_ids=prompt_ids,
                         eos_byte_strings=EOS_BYTES)
        # Unit-grain contexts hold nested units: coerce through flatten_units.
        return Coerced(lm, lm.vocab, f=flatten_units, prune=False)

    patch_serving_probe()
    for seed in (1234, 7, 99):
        seed_all(seed)
        controller = Controller(
            samplers=[make()], critics=[make_critic()], group_sizes=[8],
            ess_threshold=0.5, max_tokens=6, twist_with_critic=True,
        )
        print(f"=== seed {seed}", flush=True)
        parts = asyncio.run(BurstLoop(controller).run())
        print(f"seed {seed}: logw={[round(float(p.logw), 3) for p in parts]}",
              flush=True)


if __name__ == "__main__":
    main()
