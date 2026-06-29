"""Generate the TIGHT gate-2 reference: OUR StepLoop run with the counter-based (threefry)
picker, for every case in CASES. Stored in ``gate2_steploop_snapshot.json`` -- a SEPARATE
file that never touches the original-llamppl ``gate2_snapshot.json``.

Because the picker is device-agnostic, the burst (CUDA) draws the SAME tokens as this CPU
StepLoop reference, so gate-2's ``reference="steploop_cached"`` comparison is a near-byte-
exact paired check (warm-KV residual only) instead of divergent-path MC noise.

Box only (CUDA + vLLM). Run from tests/sampler/:
    VLLM_USE_FLASHINFER_SAMPLER=0 /root/genlm/genlm-venv/bin/python gen_steploop_reference.py [--only <substr>]
"""

import argparse
import json
import os

import test_engine_native as T  # _run_steploop (no module skip on box)
from gate2_cases import CASES, MODEL, PROMPT, EOS_BYTES, CONFIG
from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.util import set_draw_method

_SNAP = os.path.join(os.path.dirname(__file__), "gate2_steploop_snapshot.json")


def _key(label, n, ess, mt, seed):
    return f"{label}|N={n}|ess={ess}|mt={mt}|seed={seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="regen only labels containing this substring")
    args = ap.parse_args()

    from genlm.backend.llm import AsyncVirtualLM

    set_draw_method("threefry_gumbel")  # the tight reference draws with the counter-based picker
    model = AsyncVirtualLM.from_name(
        MODEL,
        engine_opts={"gpu_memory_utilization": 0.3, "max_model_len": 256, "enable_prefix_caching": True},
    )
    llm = PromptedLLM(model, eos_byte_strings=EOS_BYTES)
    llm.set_prompt_from_str(PROMPT)

    try:
        with open(_SNAP) as f:
            snap = json.load(f)
    except FileNotFoundError:
        snap = {}
    snap.pop("__config__", None)

    for label, case in CASES.items():
        if args.only and args.only not in label:
            continue
        mkc = (lambda: case.critic(llm)) if case.make_critic is not None else None
        for seed in case.seeds:
            make = lambda s=seed: case.sampler(llm, s)  # noqa: E731
            r = T._run_steploop(make, case.n_particles, case.ess, case.max_tokens, seed, mkc)
            key = _key(label, case.n_particles, case.ess, case.max_tokens, seed)
            snap[key] = {"contexts": r["contexts"], "logw": r["logw"], "log_ml": r["log_ml"]}
            print(f"  {key}: log_ml={r['log_ml']:.6f} n={len(r['contexts'])}", flush=True)
            snap["__config__"] = CONFIG
            with open(_SNAP, "w") as f:
                json.dump(snap, f, indent=1, sort_keys=True)
    print(f"wrote {_SNAP} ({len(snap) - 1} entries) from StepLoop+threefry")


if __name__ == "__main__":
    main()
