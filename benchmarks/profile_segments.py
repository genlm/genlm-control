"""Per-segment profiling of the StepLoop and BurstLoop execution paths.

Monkeypatches each segment function with an accumulating ``perf_counter`` timer
(thread-safe — the burst callback runs in a worker thread). Measurement model:

* BurstLoop: ``Controller.draw`` (and everything it calls) runs synchronously in
  the engine's worker thread, so every wrapped segment's wall-time is exact and
  non-overlapping. ``engine forward+sched`` = ``burst_wall − Controller.draw total``.
* StepLoop: SYNC work serializes on the single event loop, so wrapped sync
  segments (``_process_logw_next``, ``fast_sample``) are exact; the engine round
  trip = ``step_wall − Σ(sync) − resample`` (the awaits release the loop).

Run on the box, one config per invocation:
    VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python benchmarks/profile_segments.py --model gpt2 --sampler direct [--critic]
"""

from __future__ import annotations

import argparse
import asyncio
import threading
import time
from collections import defaultdict

import numpy as np
from transformers import AutoTokenizer

PROF: dict = defaultdict(lambda: [0.0, 0])
_LK = threading.Lock()


def _wrap(cls, attr, name):
    orig = getattr(cls, attr)
    if asyncio.iscoroutinefunction(orig):
        async def w(*a, **k):
            t = time.perf_counter()
            try:
                return await orig(*a, **k)
            finally:
                dt = time.perf_counter() - t
                with _LK:
                    PROF[name][0] += dt
                    PROF[name][1] += 1
    else:
        def w(*a, **k):
            t = time.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                dt = time.perf_counter() - t
                with _LK:
                    PROF[name][0] += dt
                    PROF[name][1] += 1
    setattr(cls, attr, w)


def _reset():
    with _LK:
        PROF.clear()


def _dump(label, wall):
    print(f"\n===== {label}   wall={wall:.3f}s =====")
    print(f"  {'segment':30s} {'total':>9s} {'%wall':>7s} {'calls':>8s} {'ms/call':>9s}")
    rows = sorted(PROF.items(), key=lambda x: -x[1][0])
    for name, (tot, cnt) in rows:
        print(
            f"  {name:30s} {tot:8.3f}s {100 * tot / wall:6.1f}% {cnt:8d} "
            f"{1000 * tot / max(cnt, 1):8.2f}"
        )


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--sampler", default="direct", choices=["direct", "awrs", "set"])
    ap.add_argument("--critic", action="store_true")
    ap.add_argument("--n-particles", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from genlm.backend.llm import AsyncVirtualLM
    from genlm.control.constant import EndOfSequence
    from genlm.control.potential import Potential
    from genlm.control.potential.built_in.llm import PromptedLLM
    from genlm.control.potential.built_in.wfsa import BoolFSA
    from genlm.control.sampler import EagerSetSampler
    from genlm.control.sampler.sequence import SMC
    from genlm.control.sampler.token import AWRS, DirectTokenSampler, SetTokenSampler

    class TermCritic(Potential):
        def __init__(self, vocab):
            super().__init__(vocabulary=vocab)

        async def _ind(self, context):
            bs = [t for t in context if not isinstance(t, EndOfSequence)]
            try:
                return 0.0 if " " in b"".join(bs).decode("utf-8") else float("-inf")
            except UnicodeDecodeError:
                return float("-inf")

        async def complete(self, c):
            return await self._ind(c)

        async def prefix(self, c):
            return 0.0

        async def score(self, c):
            return await self._ind(c)

    model = AsyncVirtualLM.from_name(
        args.model,
        engine_opts={
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_mem,
            "enable_prefix_caching": True,
        },
    )
    eos_id = model.tokenizer.eos_token_id
    llm = PromptedLLM(model, eos_byte_strings=[model.byte_vocab[eos_id].byte_string])
    llm.set_prompt_from_str("The")
    critic = TermCritic(llm.vocab) if args.critic else None

    set_sampler = None
    if args.sampler == "direct":
        sampler = DirectTokenSampler(llm)
    elif args.sampler == "awrs":
        cond = BoolFSA.from_regex(r"[a-z ]+").coerce(llm, f=b"".join)
        sampler = AWRS(llm, cond)
    else:
        set_sampler = EagerSetSampler(
            iter_potential=llm, item_potential=BoolFSA.from_regex(r"[a-z ]+")
        )
        sampler = SetTokenSampler(set_sampler)

    # ---- monkeypatch the segments ----
    from genlm.control.sampler import controller as C
    from genlm.control.sampler import token as T

    _wrap(C.Controller, "draw", "burst.draw_TOTAL")
    _wrap(C.Controller, "_bank_burst_draw", "burst.bank(+criticHop)")
    _wrap(C.Controller, "_ess_crosses", "burst.ess")
    _wrap(C.Controller, "_maybe_resample", "resample")
    _wrap(C._Burst, "run_sync", "burst.hop.critic")
    _wrap(C.BurstContext, "run_sync", "burst.hop.sampler")
    _wrap(PromptedLLM, "_process_logw_next_batch", "burst.lm_batch[GPU]")
    _wrap(PromptedLLM, "_process_logw_next", "step.lm_proc[GPU+xfer]")
    _wrap(type(sampler), "burst_draw_batch", "burst.draw(sampler)")
    _wrap(type(sampler), "sample", "step.sample_TOTAL")
    # fast_sample_lazyweights is imported into token's namespace
    _orig_fs = T.fast_sample_lazyweights

    def _fs(*a, **k):
        t = time.perf_counter()
        try:
            return _orig_fs(*a, **k)
        finally:
            with _LK:
                PROF["fast_sample[numpy]"][0] += time.perf_counter() - t
                PROF["fast_sample[numpy]"][1] += 1

    T.fast_sample_lazyweights = _fs
    if args.sampler == "awrs":
        _wrap(AWRS, "_run_rejection", "burst.rejection")
    if set_sampler is not None:
        _wrap(type(set_sampler), "sample_set", "burst.set_construct")

    def _seed():
        import torch

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    async def estep(accelerate):
        _seed()
        t0 = time.perf_counter()
        seqs = await SMC(sampler, critic=critic)(
            n_particles=args.n_particles,
            ess_threshold=0.0,
            max_tokens=args.max_tokens,
            accelerate=accelerate,
        )
        return time.perf_counter() - t0, seqs

    await estep("require")  # warmup

    _reset()
    step_wall, _ = await estep("off")
    _dump(
        f"STEP  {args.model} sampler={args.sampler} critic={args.critic} N={args.n_particles}",
        step_wall,
    )

    _reset()
    burst_wall, _ = await estep("require")
    draw_total = PROF.get("burst.draw_TOTAL", [0, 0])[0]
    print(
        f"\n[derived] engine forward+sched (burst_wall - draw_TOTAL) "
        f"= {burst_wall - draw_total:.3f}s ({100 * (burst_wall - draw_total) / burst_wall:.1f}%)"
    )
    _dump(
        f"BURST {args.model} sampler={args.sampler} critic={args.critic} N={args.n_particles}",
        burst_wall,
    )

    if set_sampler is not None:
        await set_sampler.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
