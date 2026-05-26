"""Benchmark the engine-native burst (BurstLoop) vs the per-token StepLoop.

Measures the wall-clock speedup of riding vLLM's warm-KV decode loop instead of
re-prefilling the whole context every token, on the *production SMC pattern* used
by genlm-latent's E-step: ``DirectTokenSampler`` over a ``PromptedLLM`` prior with
a terminal 0/-inf correctness critic at ``ess_threshold=0`` (rollouts drawn from
the backend, the indicator applied once at termination).

It also asserts the two paths agree (``log_ml`` within the warm-KV residual) so a
speedup is never bought by changing the algorithm.

Run on the GPU box (vLLM is not installed on macOS):

    VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        /root/genlm-venv/bin/python benchmarks/bench_burst.py \
        --model gpt2 --n-particles 16 --max-tokens 128

Sweep model sizes by invoking once per model (a fresh engine each time avoids
accumulating GPU pools / OOM):

    for m in gpt2 meta-llama/Llama-3.2-1B Qwen/Qwen2.5-7B-Instruct; do
        ... python benchmarks/bench_burst.py --model "$m" ...
    done
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

from genlm.control.constant import EndOfSequence
from genlm.control.potential import Potential
from genlm.control.potential.built_in.llm import PromptedLLM
from genlm.control.sampler.sequence import SMC
from genlm.control.sampler.token import AWRS, DirectTokenSampler, SetTokenSampler


class ContainsCritic(Potential):
    """Terminal 0/-inf indicator critic: the completed text must contain a space.

    A cheap, deterministic stand-in for genlm-latent's ``CoTCritic`` (answer-match
    indicator): ``prefix`` is 0 throughout, all weight comes from the terminal
    ``score``. This is the regime ``ess_threshold=0`` is built for -- the burst
    draws whole rollouts from the backend and applies the indicator once at the
    end.
    """

    def __init__(self, vocab):
        super().__init__(vocabulary=vocab)

    async def _indicator(self, context):
        bs = [t for t in context if not isinstance(t, EndOfSequence)]
        try:
            text = b"".join(bs).decode("utf-8")
        except UnicodeDecodeError:
            return float("-inf")
        return 0.0 if " " in text else float("-inf")

    async def complete(self, context):
        return await self._indicator(context)

    async def prefix(self, context):
        return 0.0

    async def score(self, context):
        return await self._indicator(context)


def _seed(s):
    import torch

    np.random.seed(s)
    torch.manual_seed(s)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--prompt", default="The")
    p.add_argument("--n-particles", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--gpu-mem", type=float, default=0.6)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument(
        "--sampler",
        default="direct",
        choices=["direct", "awrs", "set"],
        help="Token sampler to benchmark (all ride the unified batched burst path).",
    )
    p.add_argument(
        "--constraint",
        default="alpha",
        choices=["alpha", "json"],
        help="AWRS/Set constraint: 'alpha' ([a-z ]+, context-free) or 'json' "
        "(flat JSON object, context-dependent -- exercises the rejection walk).",
    )
    p.add_argument(
        "--no-critic",
        action="store_true",
        help="Drop the terminal critic to isolate the pure batched-draw ceiling "
        "(no per-particle critic-eval event-loop hop).",
    )
    p.add_argument(
        "--eos-newline",
        action="store_true",
        help="Use b'\\n' as EOS (fires early -> short rollouts). Default: the "
        "model's natural EOS, so rollouts reach ~max_tokens (the long-rollout "
        "regime the burst targets).",
    )
    p.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable vLLM prefix caching. With it ON (default), the StepLoop's "
        "per-token re-prefill reuses the cached prefix KV, so the burst's warm-KV "
        "advantage is mostly per-token overhead; OFF exposes the full re-prefill cost.",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    from genlm.backend.llm import AsyncVirtualLM

    model = AsyncVirtualLM.from_name(
        args.model,
        engine_opts={
            "gpu_memory_utilization": args.gpu_mem,
            "max_model_len": args.max_model_len,
            "enable_prefix_caching": not args.no_prefix_cache,
        },
    )
    if args.eos_newline:
        eos_bytes = [b"\n"]  # fires early -> short rollouts (overhead-bound regime)
    else:
        # The model's natural EOS is rare in free continuation, so rollouts reach
        # ~max_tokens -- the long-rollout regime the burst is designed for (it
        # avoids re-prefilling a growing context every token).
        eos_id = model.tokenizer.eos_token_id
        eos_bytes = [model.byte_vocab[eos_id].byte_string]
    llm = PromptedLLM(model, eos_byte_strings=eos_bytes)
    llm.set_prompt_from_str(args.prompt)
    # --no-critic isolates the pure batched-draw ceiling (no per-particle critic
    # eval hop); default keeps the terminal critic (the genlm-latent regime).
    critic = None if args.no_critic else ContainsCritic(llm.vocab)

    # Build the chosen sampler ONCE (Set's trie init is heavy). AWRS/Set keep a
    # per-particle control loop (rejection / trie) over the batched proposal --
    # timing them shows how far that per-particle dance leaves them from the ceiling.
    from genlm.control.potential.built_in.wfsa import BoolFSA
    from genlm.control.sampler import EagerSetSampler

    # Constraint regexes. `alpha` ([a-z ]+) is context-FREE -- a token is valid by
    # its bytes alone, so the rejection accepts the top token immediately (~1 step)
    # and prune captures the whole constraint. `json` is a flat JSON object: it is
    # context-DEPENDENT (after `{` only `"`/`}`, after `:` a value, ...), so valid
    # tokens are few and state-dependent -- the regime where the rejection actually
    # walks and prune is only a static lower bound.
    CONSTRAINTS = {
        "alpha": r"[a-z ]+",
        "json": r'\{("[a-z]+": ("[a-z ]*"|-?[0-9]+|true|false|null)(, "[a-z]+": ("[a-z ]*"|-?[0-9]+|true|false|null))*)?\}',
    }
    regex = CONSTRAINTS[args.constraint]

    if args.sampler == "direct":
        sampler = DirectTokenSampler(llm)
    elif args.sampler == "awrs":
        condition = BoolFSA.from_regex(regex).coerce(llm, f=b"".join)
        sampler = AWRS(llm, condition)
        print(f"constraint={args.constraint}  |valid_idxs|={len(sampler.valid_idxs)} "
              f"of vocab+eos {len(llm.vocab) + 1}")
    elif args.sampler == "set":
        item = BoolFSA.from_regex(regex)
        sampler = SetTokenSampler(EagerSetSampler(iter_potential=llm, item_potential=item))
    else:
        raise ValueError(args.sampler)

    async def once(backend):
        # Same seed each run so StepLoop and BurstLoop consume the same RNG stream
        # (they then differ only by the warm-KV-vs-reprefill logit residual).
        _seed(args.seed)
        t0 = time.perf_counter()
        seqs = await SMC(sampler, critic=critic)(
            n_particles=args.n_particles,
            ess_threshold=0.0,
            max_tokens=args.max_tokens,
            backend=backend,
        )
        return time.perf_counter() - t0, seqs

    def raw_batch_decode(max_tokens):
        """CEILING: stock vLLM batch decode of N sequences for `max_tokens`, with
        NO SMC and NO control callback -- the engine's native decode throughput.
        The gap (burst - raw) is the per-step control-CPU overhead the burst
        serializes with the GPU decode (the thing to drive toward zero)."""
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        prompt_ids = list(llm.prompt_ids)
        sp = SamplingParams(
            n=1, max_tokens=max_tokens, ignore_eos=True, detokenize=False
        )
        prompts = [
            TokensPrompt(prompt_token_ids=prompt_ids) for _ in range(args.n_particles)
        ]
        t0 = time.perf_counter()
        model.llm_engine.generate(prompts, sp, use_tqdm=False)
        return time.perf_counter() - t0

    # Warm the engine (cold prefill / CUDA-graph capture) before timing.
    for _ in range(args.n_warmup):
        await once(None)
        await once(model)
        raw_batch_decode(args.max_tokens)

    async def trials(backend):
        dts, last = [], None
        for _ in range(args.n_trials):
            dt, seqs = await once(backend)
            dts.append(dt)
            last = seqs
        return float(np.median(dts)), last

    step_dt, step_seqs = await trials(None)
    burst_dt, burst_seqs = await trials(model)
    raw_dt = float(np.median([raw_batch_decode(args.max_tokens) for _ in range(args.n_trials)]))
    await sampler.cleanup()

    speedup = step_dt / burst_dt if burst_dt > 0 else float("nan")
    ml_diff = float(burst_seqs.log_ml - step_seqs.log_ml)
    step_len = float(np.mean([len(c) for c in step_seqs.contexts]))
    burst_len = float(np.mean([len(c) for c in burst_seqs.contexts]))

    print("=" * 64)
    print(
        f"model={args.model}  sampler={args.sampler}  N={args.n_particles}  "
        f"max_tokens={args.max_tokens}  trials={args.n_trials}  "
        f"prefix_cache={not args.no_prefix_cache}"
    )
    print(
        f"  StepLoop : {step_dt:7.3f}s  (median)  log_ml={step_seqs.log_ml:+.4f}  "
        f"mean_len={step_len:.1f}"
    )
    print(
        f"  BurstLoop: {burst_dt:7.3f}s  (median)  log_ml={burst_seqs.log_ml:+.4f}  "
        f"mean_len={burst_len:.1f}"
    )
    print(
        f"  RAW batch-N decode (ceiling, no control): {raw_dt:7.3f}s "
        f"({args.n_particles}x{args.max_tokens} tok)"
    )
    print(
        f"  SPEEDUP  : {speedup:6.2f}x (burst vs step)   |   "
        f"burst/raw = {burst_dt / raw_dt:5.2f}x  (how far the burst is above the "
        f"engine's native decode floor)   |   log_ml diff={ml_diff:+.4f}"
    )
    print("=" * 64)


if __name__ == "__main__":
    asyncio.run(main())
