"""Generate the TIGHT gate-2 reference: every case in CASES run over plain per-context
forwards with the counter-based (threefry) picker. Stored in
``gate2_forward_snapshot.json`` -- a SEPARATE file that never touches the
original-llamppl ``gate2_snapshot.json``.

Because the picker is device-agnostic, the engine-served gate draws the SAME tokens as
this reference IF the rows agree to near-numeric precision -- which requires the
reference forwards to come from the SAME engine numerics the gate runs on. A reference
from another backend (HF fp32 on a laptop vs vLLM bf16 on an A100) flips enough draws
that the paired check degenerates to noise.

So: ``--backend vllm`` (on the GPU that runs the gate, engine opts matching the gate
fixture) generates the canonical reference -- every window starts from a cold
residency table, so each ask prefills fresh while the population still batches into
one frame, and the gate measures exactly the warm-KV-vs-reprefill residual.
``--backend hf`` needs no engine and exists for local smoke only; never commit a
snapshot from it. From tests/sampler/:

    python gen_reference.py --backend vllm [--only <substr>]
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


def _load(backend):
    if backend == "vllm":
        from genlm.backend.llm import AsyncVirtualLM

        class _FreshEngine(AsyncVirtualLM):
            """Every window starts from a cold residency table, so every ask
            prefills a fresh request — no row ever comes from warm KV — while
            the window still batches the population's asks into one frame,
            matching the gate run's batch composition."""

            def _execute(self, queries):
                reaps = self._reap_idle(len(self._requests))
                if reaps:
                    self._sched.genlm_submit([], [], reaps)
                super()._execute(queries)

        # Engine opts must MATCH the gate fixture (test_engine_native.py):
        # eager and cudagraph kernels differ numerically, and a row difference
        # flips paired threefry draws.
        return _FreshEngine.from_name(
            MODEL,
            engine_opts={
                "gpu_memory_utilization": 0.2,
                "max_model_len": 256,
                "enable_prefix_caching": True,
                "enforce_eager": True,
            },
        )
    from genlm.backend.llm import AsyncTransformer

    return AsyncTransformer.from_name(MODEL)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default=None, help="regen only labels containing this substring")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    args = ap.parse_args()

    set_draw_method("threefry_gumbel")  # the tight reference draws with the counter-based picker
    llm = PromptedLLM(_load(args.backend), eos_byte_strings=EOS_BYTES)
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
