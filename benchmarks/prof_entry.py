"""cProfile entry for ONE E-step config → .pstats (for gprof2dot call-graph SVG).

Captures BOTH threads and merges them, because the burst decode loop runs in a
worker thread that a plain ``python -m cProfile`` (main thread) would miss:
  * main thread  — the SMC driver + the hop coroutines (AWRS rejection / critic /
    factor eval run on the event loop, serviced while the main thread awaits the
    executor future);
  * worker thread — ``run_burst`` → ``ControlSampler.forward`` → ``Controller.draw``.

CAVEAT (read the graph with this in mind): CUDA is async. The engine forward
queues on the GPU stream and returns; its compute time is attributed to whichever
CPU frame first *synchronizes* on the result (a ``.cpu()`` / ``.tolist()`` / a read).
So a frame's cumulative time can include GPU waits, not just CPU work.

Run on the box (writes <out>.pstats):
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/prof_entry.py --mode burst --model gpt2 --sampler direct --out /tmp/p_burst_direct
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import pstats
import shutil

import numpy as np


async def _run(sampler, critic, accelerate, args):
    from genlm.control.sampler.sequence import SMC

    np.random.seed(args.seed)
    import torch

    torch.manual_seed(args.seed)
    await SMC(sampler, critic=critic)(
        n_particles=args.n_particles,
        ess_threshold=0.0,
        max_tokens=args.max_tokens,
        accelerate=accelerate,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["step", "burst"], required=True)
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--sampler", default="direct", choices=["direct", "awrs", "set"])
    ap.add_argument("--constraint", default="alpha", choices=["alpha", "json"])
    ap.add_argument("--critic", action="store_true")
    ap.add_argument("--n-particles", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--gpu-mem", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from genlm.backend.llm import AsyncVirtualLM
    from genlm.control.constant import EndOfSequence
    from genlm.control.potential import Potential
    from genlm.control.potential.built_in.llm import PromptedLLM
    from genlm.control.potential.built_in.wfsa import BoolFSA
    from genlm.control.sampler import EagerSetSampler
    from genlm.control.sampler.token import AWRS, DirectTokenSampler, SetTokenSampler

    class TermCritic(Potential):
        def __init__(self, vocab):
            super().__init__(vocabulary=vocab)

        async def _ind(self, c):
            bs = [t for t in c if not isinstance(t, EndOfSequence)]
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
    eid = model.tokenizer.eos_token_id
    llm = PromptedLLM(model, eos_byte_strings=[model.byte_vocab[eid].byte_string])
    llm.set_prompt_from_str("The")
    critic = TermCritic(llm.vocab) if args.critic else None

    regex = {
        "alpha": r"[a-z ]+",
        "json": r'\{("[a-z]+": ("[a-z ]*"|-?[0-9]+|true|false|null)(, "[a-z]+": ("[a-z ]*"|-?[0-9]+|true|false|null))*)?\}',
    }[args.constraint]

    set_sampler = None
    if args.sampler == "direct":
        sampler = DirectTokenSampler(llm)
    elif args.sampler == "awrs":
        sampler = AWRS(llm, BoolFSA.from_regex(regex).coerce(llm, f=b"".join))
    else:
        set_sampler = EagerSetSampler(
            iter_potential=llm, item_potential=BoolFSA.from_regex(regex)
        )
        sampler = SetTokenSampler(set_sampler)

    accelerate = "require" if args.mode == "burst" else "off"

    # warmup (untimed, unprofiled)
    asyncio.run(_run(sampler, critic, accelerate, args))

    # worker-thread profiler: wrap run_burst (the executor target)
    worker_prof = cProfile.Profile()
    if args.mode == "burst":
        _orig = model.run_burst

        def _wrapped(*a, **k):
            worker_prof.enable()
            try:
                return _orig(*a, **k)
            finally:
                worker_prof.disable()

        model.run_burst = _wrapped

    main_prof = cProfile.Profile()
    main_prof.enable()
    asyncio.run(_run(sampler, critic, accelerate, args))
    main_prof.disable()

    main_prof.dump_stats(args.out + ".main.pstats")
    if args.mode == "burst":
        worker_prof.dump_stats(args.out + ".worker.pstats")
        merged = pstats.Stats(args.out + ".main.pstats")
        merged.add(args.out + ".worker.pstats")
        merged.dump_stats(args.out + ".pstats")
    else:
        shutil.copy(args.out + ".main.pstats", args.out + ".pstats")
    print(f"wrote {args.out}.pstats")

    if set_sampler is not None:
        asyncio.run(set_sampler.cleanup())


if __name__ == "__main__":
    main()
