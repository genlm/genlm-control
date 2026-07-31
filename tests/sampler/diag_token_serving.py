"""Diagnostic (NOT a gate): token-grain critic serving drift, warm vs cold.

During a free-running burst the critic LM leaf's prefix/complete are served from
banked warm-row sums. Cold recompute is illegal mid-burst, so this records every
served (context, value) and, after the run drains, re-scores them cold and prints
the per-call drift -- prefix drift cancels at untwist, complete drift is permanent
in log_ml.

Box/Mila only (real vLLM). Run:
    VLLM_USE_FLASHINFER_SAMPLER=0 python tests/sampler/diag_token_serving.py
"""

import asyncio

import numpy as np

from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.potential import base as pot_base
from genlm.control.sampler.token import DirectTokenSampler
from genlm.control.sampler.smc import Controller
from genlm.control.sampler.burst import BurstLoop
from genlm.control.util import set_draw_method

from _harness import seed_all
from gate2_cases import MODEL, PROMPT, EOS_BYTES


RECORDED = []  # (kind, context tuple, served value)


def patch_recorder():
    for name, var in (
        ("batch_prefix", pot_base._burst_prefix_overrides),
        ("batch_complete", pot_base._burst_complete_overrides),
    ):
        orig = getattr(PromptedLLM, name)

        def probed(self, contexts, _orig=orig, _var=var, _kind=name):
            override = _var.get()
            if override and self in override:
                for ctx, val in zip(contexts, override[self]):
                    RECORDED.append((_kind, self, tuple(ctx), float(val)))
            return _orig(self, contexts)

        setattr(PromptedLLM, name, probed)


async def cold_drift():
    """Re-score every recorded serve cold (engine idle) and report drift."""
    out = {}
    for kind in ("batch_prefix", "batch_complete"):
        rec = [r for r in RECORDED if r[0] == kind]
        if not rec:
            continue
        drifts, by_len = [], {}
        for _, leaf, ctx, served in rec:
            cold = (
                await leaf.prefix(list(ctx))
                if kind == "batch_prefix"
                else await leaf.complete(list(ctx))
            )
            d = served - cold
            drifts.append(d)
            by_len.setdefault(len(ctx), []).append(d)
        drifts = np.array(drifts)
        out[kind] = drifts
        print(f"\n[{kind}] n={len(drifts)} drift mean={drifts.mean():+.4f} "
              f"std={drifts.std():.4f} max|d|={np.abs(drifts).max():.4f}")
        for n in sorted(by_len):
            ds = np.array(by_len[n])
            print(f"  len={n:2d}: n={len(ds):3d} mean={ds.mean():+.4f} "
                  f"max|d|={np.abs(ds).max():.4f}")
    return out


def main():
    set_draw_method("threefry_gumbel")
    llm = PromptedLLM.from_name(
        MODEL, backend="vllm", eos_byte_strings=EOS_BYTES,
        engine_opts={"gpu_memory_utilization": 0.45, "max_model_len": 1024},
    )
    llm.set_prompt_from_str(PROMPT)
    prompt_ids = llm.model.tokenizer.encode(PROMPT)
    patch_recorder()

    for seed in (1234, 7, 99):
        RECORDED.clear()
        seed_all(seed)
        critic = PromptedLLM(llm.model, prompt_ids=prompt_ids,
                             eos_byte_strings=EOS_BYTES)
        controller = Controller(
            samplers=[DirectTokenSampler(llm)], critics=[critic],
            group_sizes=[8], ess_threshold=0.5, max_tokens=8,
            twist_with_critic=True,
        )
        print(f"\n=== seed {seed}", flush=True)
        parts = asyncio.run(BurstLoop(controller).run())
        print(f"logw={[round(float(p.logw), 3) for p in parts]}", flush=True)
        asyncio.run(cold_drift())


if __name__ == "__main__":
    main()
