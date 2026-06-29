"""Generate gate-2's cached reference from the ORIGINAL genlm-control (main + llamppl
``smc_standard``), not our StepLoop.

The configs/critics/seeds are the SHARED ``gate2_cases.CASES`` the gate itself uses, so the
reference can never drift from the test. Only ``reference == "ref"`` cases are emitted (the
``"steploop"`` cases compare against a live StepLoop in the test, no cached key). main and
this branch share the potential/sampler/PromptedLLM API verbatim -- only the SMC engine
differs -- so importing ``gate2_cases`` under main's ``genlm.control`` builds the same
samplers, just run through main's llamppl SMC (no ``accelerate=``).

Run on the box, shadowing our genlm.control with main's:

    PYTHONPATH=/root/genlm/genlm-control-main VLLM_USE_FLASHINFER_SAMPLER=0 \
        /root/genlm-venv/bin/python tests/sampler/gen_original_reference.py

Writes ``gate2_snapshot.json`` next to this file (same path + key format the gate loads).
Pass ``--only <substr>`` to regenerate a subset.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os

from genlm.control.sampler.sequence import SMC

from _harness import seed_all, ctx_ids
from gate2_cases import MODEL, PROMPT, EOS_BYTES, CONFIG, CASES

_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "gate2_snapshot.json")


def _key(label, n_particles, ess_threshold, max_tokens, seed):
    return f"{label}|N={n_particles}|ess={ess_threshold}|mt={max_tokens}|seed={seed}"


async def _run(case, llm, seed):
    seed_all(seed)
    sampler = case.sampler(llm, seed)
    critic = case.critic(llm)
    try:
        seqs = await SMC(sampler, critic)(
            n_particles=case.n_particles,
            ess_threshold=case.ess,
            max_tokens=case.max_tokens,
        )
    finally:
        await sampler.cleanup()  # the Set sampler's trie task; no-op for the rest
    return {
        "contexts": [ctx_ids(c) for c in seqs.contexts],
        "logw": [float(w) for w in seqs.log_weights],
        "log_ml": float(seqs.log_ml),
    }


async def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="regen only configs whose label contains this")
    args = ap.parse_args()

    from genlm.backend.llm import AsyncVirtualLM
    from genlm.control.potential.built_in.llm import PromptedLLM

    model = AsyncVirtualLM.from_name(
        MODEL,
        engine_opts={"gpu_memory_utilization": 0.3, "max_model_len": 256, "enable_prefix_caching": True},
    )
    llm = PromptedLLM(model, eos_byte_strings=EOS_BYTES)
    llm.set_prompt_from_str(PROMPT)

    try:
        with open(_SNAPSHOT_PATH) as f:
            snap = json.load(f)
    except FileNotFoundError:
        snap = {}
    snap.pop("__config__", None)

    for label, case in CASES.items():
        if case.reference != "ref":
            continue
        if args.only and args.only not in label:
            continue
        for seed in case.seeds:
            key = _key(case.label, case.n_particles, case.ess, case.max_tokens, seed)
            snap[key] = await _run(case, llm, seed)
            print(f"  {key}: log_ml={snap[key]['log_ml']:.6f} n={len(snap[key]['contexts'])}")
            snap["__config__"] = CONFIG
            with open(_SNAPSHOT_PATH, "w") as f:
                json.dump(snap, f, indent=1, sort_keys=True)
    print(f"wrote {_SNAPSHOT_PATH} ({len(snap) - 1} reference entries) from ORIGINAL genlm-control")


if __name__ == "__main__":
    asyncio.run(main())
