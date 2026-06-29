"""cProfile a single scenario's SMC run -> merged .pstats -> gprof2dot call-graph SVG.

Reuses bench.py's scenario builders (direct/awrs/set/lora/cot/ds1000), so the
profiled path is exactly the benchmarked one. Captures BOTH threads and merges:
the burst's vLLM decode loop runs in a worker thread (``run_burst``) that a plain
main-thread cProfile would miss; the SMC driver + awaited critics run on the loop.

CAVEAT: CUDA is async -- the engine forward queues on the GPU stream and returns;
its time is attributed to whichever CPU frame first synchronizes on the result
(.cpu()/.tolist()/a read). A frame's cumulative time can include GPU waits.

Run on the box (writes <out>.pstats and, unless --no-svg, <out>.svg):
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/prof_entry.py --scenario direct --mode burst \
      --model Qwen/Qwen2.5-7B-Instruct --out results/prof__direct__burst
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import pstats
import shutil
import subprocess

import bench
import bench_core as bc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario",
                   choices=["direct", "awrs", "set", "lora", "cot", "ds1000"],
                   default="direct")
    p.add_argument("--mode", choices=["burst", "step"], default="burst")
    p.add_argument("--out", required=True, help="output path stem (.pstats/.svg appended)")
    p.add_argument("--no-svg", action="store_true", help="write .pstats only")
    # scenario knobs (mirror bench.py; the sampler/critic LOGIC is reused from bench)
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--n-particles", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ess-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu-mem", type=float, default=0.6)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--prompt", default="The")
    p.add_argument("--eos", choices=["natural", "newline"], default="natural")
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--constraint", choices=["alpha", "json"], default="alpha")
    p.add_argument("--lora-adapter", default=bench.LORA_ADAPTER)
    p.add_argument("--question", default=(
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did she sell altogether?"))
    p.add_argument("--answer", default="72")
    p.add_argument("--library", default="Pandas")
    p.add_argument("--item", type=int, default=0)
    p.add_argument("--timeout", type=float, default=6.0)
    return p.parse_args()


def _render_svg(pstats_path: str, svg_path: str) -> None:
    """gprof2dot -f pstats <pstats> | dot -Tsvg -o <svg>  (default node format:
    name:module / total-time-incl-subcalls % / (self %) / num self calls)."""
    dot = subprocess.run(
        ["gprof2dot", "-f", "pstats", pstats_path], check=True, capture_output=True)
    subprocess.run(["dot", "-Tsvg", "-o", svg_path], check=True, input=dot.stdout)
    print(f"wrote {svg_path}")


def main() -> None:
    args = parse_args()
    from genlm.control.sampler.sequence import SMC

    model_name, engine_opts, post_engine = bench.scenario_engine(args)
    args.model = model_name
    model = bc.build_engine(model_name, engine_opts)
    if post_engine:
        post_engine(model)
    built = bench.scenario_build(args, model)
    accelerate = "require" if args.mode == "burst" else "off"

    async def run():
        bc.seed_all(args.seed)
        await SMC(built.sampler, critic=built.critic)(
            n_particles=args.n_particles, ess_threshold=args.ess_threshold,
            max_tokens=args.max_tokens, accelerate=accelerate)

    asyncio.run(run())  # warmup (untimed, unprofiled): cold prefill / CUDA graphs

    # worker-thread profiler: the burst decode loop runs in the run_burst executor target
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
    asyncio.run(run())
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

    asyncio.run(built.sampler.cleanup())

    if not args.no_svg:
        _render_svg(args.out + ".pstats", args.out + ".svg")


if __name__ == "__main__":
    main()
