"""Generate the TIGHT gate-2 reference: every case in CASES run over plain per-context
forwards (a HuggingFace ``AsyncTransformer``) with the counter-based (threefry) picker.
Stored in ``gate2_forward_snapshot.json`` -- a SEPARATE file that never touches the
original-llamppl ``gate2_snapshot.json``.

Because the picker is device-agnostic, the engine-served gate draws the SAME tokens as this
reference, so ``reference="forward_cached"`` is a near-byte-exact paired check (engine
numeric residual only) instead of divergent-path MC noise.

Needs no engine -- run anywhere with torch. From tests/sampler/:
    python gen_reference.py [--only <substr>]
"""

import argparse
import json
import os

from _harness import run_case
from gate2_cases import CASES, MODEL, PROMPT, EOS_BYTES, CONFIG
from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.util import set_draw_method

_SNAP = os.path.join(os.path.dirname(__file__), "gate2_forward_snapshot.json")


def _key(label, n, ess, mt, seed):
    return f"{label}|N={n}|ess={ess}|mt={mt}|seed={seed}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="regen only labels containing this substring")
    args = ap.parse_args()

    from genlm.backend.llm import AsyncTransformer

    set_draw_method("threefry_gumbel")  # the tight reference draws with the counter-based picker
    llm = PromptedLLM(AsyncTransformer.from_name(MODEL), eos_byte_strings=EOS_BYTES)
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
            r = run_case(make, case.n_particles, case.ess, case.max_tokens, seed, mkc)
            key = _key(label, case.n_particles, case.ess, case.max_tokens, seed)
            snap[key] = {"contexts": r["contexts"], "logw": r["logw"], "log_ml": r["log_ml"]}
            print(f"  {key}: log_ml={r['log_ml']:.6f} n={len(r['contexts'])}", flush=True)
            snap["__config__"] = CONFIG
            with open(_SNAP, "w") as f:
                json.dump(snap, f, indent=1, sort_keys=True)
    print(f"wrote {_SNAP} ({len(snap) - 1} entries) from plain forwards + threefry")


if __name__ == "__main__":
    main()
