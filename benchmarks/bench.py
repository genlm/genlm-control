"""Unified speedup benchmark: one modular harness over every SMC pattern the
engine-served SMC, with neatly persisted per-version results.

Supersedes the scattered scripts (direct/awrs/set + raw ceiling;
genlm-latent's ``bench_latent_estep.py`` CoT and ``bench_ds1000_estep.py`` exec
critic) and adds the previously-unbenchmarked LoRA K=2 multi-view scenario. Each
scenario plugs a (sampler, critic, prompt) into the shared smc/raw matrix
in ``bench_core``; nothing scenario-specific lives in the harness.

Scenarios
  direct  DirectTokenSampler + terminal 0/-inf critic   (the latent E-step pattern)
  awrs    AWRS over an [a-z ]+/JSON BoolFSA constraint
  set     SetTokenSampler (EagerSetSampler) over the same constraint
  lora    K=2 multi-view: base prior p0 + LoRA proposal q on ONE engine (p0/q reweight)
  cot     genlm-latent CoTCritic answer-match (GSM8K-style)            [needs genlm-latent]
  ds1000  genlm-latent CodeCorrectnessCritic (sandboxed exec)   [needs genlm-latent + genlm-eval]

Run on the box (vLLM not on macOS):
  VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    python benchmarks/bench.py --scenario direct --model gpt2 --label speedup-now

Compare recorded versions (no GPU needed):
  python benchmarks/bench.py --compare --store bench
"""

from __future__ import annotations

import argparse
import asyncio
import time

import numpy as np

import bench_core as bc

CONSTRAINTS = {
    "alpha": r"[a-z ]+",
    "json": r'\{("[a-z]+": ("[a-z ]*"|-?[0-9]+|true|false|null)(, "[a-z]+": '
    r'("[a-z ]*"|-?[0-9]+|true|false|null))*)?\}',
}
# Default LoRA adapter for the `lora` scenario: a real adapter on the benchmark
# model (Qwen2.5-7B-Instruct). Override with --lora-adapter; the base model + rank
# are read from its adapter_config.json.
LORA_ADAPTER = "qiaw99/Qwen2.5-7B-Instruct-LogiQA-DPO-D"


# --------------------------------------------------------------------------- #
# Small shared builders (lazy genlm imports -> file also imports on main)      #
# --------------------------------------------------------------------------- #
def _print_window_stats(model):
    """Batching breadcrumb histograms: control-side windows + backend residency."""
    from genlm.control.util import take_window_stats

    stats = take_window_stats()
    draw = {k[1]: v for k, v in stats.items() if k[0] == "draw"}
    if draw:
        print(f"    draw window   : {dict(sorted(draw.items()))}")
    ab = {}
    for k, v in stats.items():
        if k[0] == "autobatch":
            ab.setdefault(k[1], {})[k[2]] = v
    for method, hist in sorted(ab.items()):
        print(f"    autobatch {method:<16}: {dict(sorted(hist.items()))}")
    if hasattr(model, "take_stats"):
        b = model.take_stats()
        cohort = {k[1]: v for k, v in b.items() if isinstance(k, tuple) and k[0] == "cohort"}
        steps = {k[1]: v for k, v in b.items() if isinstance(k, tuple) and k[0] == "steps"}
        totals = {k: v for k, v in b.items() if isinstance(k, str)}
        print(f"    backend cohort: {dict(sorted(cohort.items()))}")
        print(f"    engine steps  : {dict(sorted(steps.items()))}")
        print(f"    totals        : {totals}")


def _terminal_critic(vocab):
    """Synthetic 0/-inf terminal indicator (completed text contains a space) -- a
    cheap, deterministic stand-in for a real answer/exec critic, all weight at
    termination (the ess=0 regime engine serving is built for)."""
    from genlm.control.constant import EndOfSequence
    from genlm.control.potential import Potential

    class ContainsCritic(Potential):
        async def _ind(self, ctx):
            bs = [t for t in ctx if not isinstance(t, EndOfSequence)]
            try:
                return 0.0 if " " in b"".join(bs).decode("utf-8") else float("-inf")
            except UnicodeDecodeError:
                return float("-inf")

        async def complete(self, ctx):
            return await self._ind(ctx)

        async def prefix(self, ctx):
            return 0.0

        async def score(self, ctx):
            return await self._ind(ctx)

    return ContainsCritic(vocabulary=vocab)


def _eos_bytes(model, mode):
    if mode == "newline":
        return [b"\n"]
    eid = model.tokenizer.eos_token_id
    return [model.byte_vocab[eid].byte_string]


def _lora_info(adapter: str):
    """Download the adapter; return (local_path, base_model, rank)."""
    import json
    import os

    from huggingface_hub import snapshot_download

    path = snapshot_download(adapter)
    cfg = json.load(open(os.path.join(path, "adapter_config.json")))
    return path, cfg["base_model_name_or_path"], int(cfg.get("r", 16))


# --------------------------------------------------------------------------- #
# Scenario -> (model, engine_opts, post_engine)                               #
# --------------------------------------------------------------------------- #
def scenario_engine(args):
    """Returns (model_name, engine_opts, post_engine|None)."""
    opts = {
        "gpu_memory_utilization": args.gpu_mem,
        "max_model_len": args.max_model_len,
        "enable_prefix_caching": not args.no_prefix_cache,
    }
    if args.scenario == "lora":
        path, base, rank = _lora_info(args.lora_adapter)
        opts.update(enable_lora=True, max_lora_rank=max(16, rank), max_loras=2,
                    enforce_eager=True)

        def post_engine(model):
            model.add_new_lora(path, "vk")

        return base, opts, post_engine
    return args.model, opts, None


# --------------------------------------------------------------------------- #
# Scenario -> Built(sampler, critic, prompt_ids, config, supports_raw)         #
# --------------------------------------------------------------------------- #
class Built:
    def __init__(self, sampler, critic, prompt_ids, config, supports_raw=True):
        self.sampler, self.critic = sampler, critic
        self.prompt_ids, self.config, self.supports_raw = prompt_ids, config, supports_raw


def scenario_build(args, model) -> Built:
    from genlm.control.potential.built_in.llm import PromptedLLM
    from genlm.control.sampler.token import AWRS, DirectTokenSampler, SetTokenSampler

    s = args.scenario
    cfg = {"model": args.model, "scenario": s, "N": args.n_particles,
           "max_tokens": args.max_tokens, "ess": args.ess_threshold}

    if s == "synth":
        # A context-only potential with a DIAL on its per-call cost. Sweeping that
        # against a fixed forward is what places a real potential on the curve:
        # engine serving can only hide host work behind the GPU, so the ratio is the
        # whole story. Pure-Python burn, because genlm's real potentials (trie/WFSA
        # walks) hold the GIL the same way.
        from genlm.control.potential.base import Potential, VocabTables

        class SynthPotential(Potential):
            def __init__(self, llm, micros):
                self.micros = micros
                self._row = np.zeros(len(llm.vocab) + 1)
                super().__init__(
                    llm.vocab,
                    tables=VocabTables(
                        llm.token_type, llm.eos, llm.vocab_eos, llm.lookup
                    ),
                )

            def _burn(self):
                t_end = time.perf_counter() + self.micros / 1e6
                x = 0
                while time.perf_counter() < t_end:
                    for _ in range(256):
                        x += 1
                return x

            async def prefix(self, context):
                return 0.0

            async def complete(self, context):
                return 0.0

            async def logw_next(self, context):
                self._burn()
                return self.make_lazy_weights(self._row.copy())

        llm = PromptedLLM(model, eos_byte_strings=_eos_bytes(model, args.eos))
        llm.set_prompt_from_str(args.prompt)
        sampler = DirectTokenSampler(llm * SynthPotential(llm, args.potential_us), autobatch=args.autobatch)
        critic = None if args.no_critic else _terminal_critic(llm.vocab)
        cfg.update(potential_us=args.potential_us, eos=args.eos,
                   critic=not args.no_critic)
        return Built(sampler, critic, llm.prompt_ids, cfg)

    if s == "product":
        # Direct sampler over a PRODUCT target: the constraint is a factor of the
        # distribution being drawn from, not a rejection test, so its `logw_next` is a
        # dense per-step CPU walk that depends only on the context.
        from genlm.control.potential.built_in.wfsa import BoolFSA

        llm = PromptedLLM(model, eos_byte_strings=_eos_bytes(model, args.eos))
        llm.set_prompt_from_str(args.prompt)
        fsa = BoolFSA.from_regex(CONSTRAINTS[args.constraint]).coerce(llm, f=b"".join)
        sampler = DirectTokenSampler(llm * fsa, autobatch=args.autobatch)
        critic = None if args.no_critic else _terminal_critic(llm.vocab)
        cfg.update(constraint=args.constraint, eos=args.eos,
                   critic=not args.no_critic)
        return Built(sampler, critic, llm.prompt_ids, cfg)

    if s in ("direct", "awrs", "set"):
        llm = PromptedLLM(model, eos_byte_strings=_eos_bytes(model, args.eos))
        llm.set_prompt_from_str(args.prompt)
        critic = None if args.no_critic else _terminal_critic(llm.vocab)
        cfg["critic"] = not args.no_critic
        cfg["eos"] = args.eos
        if s == "direct":
            sampler = DirectTokenSampler(llm, autobatch=args.autobatch)
        else:
            from genlm.control.potential.built_in.wfsa import BoolFSA

            regex = CONSTRAINTS[args.constraint]
            cfg["constraint"] = args.constraint
            if s == "awrs":
                sampler = AWRS(llm, BoolFSA.from_regex(regex).coerce(llm, f=b"".join), autobatch=args.autobatch)
            else:
                from genlm.control.sampler import EagerSetSampler

                sampler = SetTokenSampler(
                    EagerSetSampler(iter_potential=llm,
                                    item_potential=BoolFSA.from_regex(regex)))
        return Built(sampler, critic, llm.prompt_ids, cfg)

    if s == "lora":
        ids = model.tokenizer.encode(args.prompt)
        eos = _eos_bytes(model, args.eos)
        p0 = PromptedLLM(model, prompt_ids=ids, eos_byte_strings=eos)
        q = PromptedLLM(model, prompt_ids=ids, eos_byte_strings=eos, lora_name="vk")
        sampler = DirectTokenSampler(potential=p0, proposal=q, autobatch=args.autobatch)
        critic = None if args.no_critic else _terminal_critic(p0.vocab)
        cfg.update(sampler="direct-multiview", lora=args.lora_adapter,
                   critic=not args.no_critic)
        return Built(sampler, critic, ids, cfg)

    if s == "cot":
        from transformers import AutoTokenizer
        from genlm.latent.algorithm.cot import CoTCritic, trice_template

        tok = AutoTokenizer.from_pretrained(args.model)
        llm = PromptedLLM(model)
        template = trice_template(
            fewshot="Question: What is 2+3?\nAnswer: 2+3=5. The answer is 5\n\n")
        llm.prompt_ids = tok.encode(template.format_prompt(args.question))
        critic = CoTCritic(llm.vocab, target_answer=args.answer)
        sampler = DirectTokenSampler(llm, autobatch=args.autobatch)
        cfg.update(answer=args.answer, critic=True)
        return Built(sampler, critic, llm.prompt_ids, cfg)

    if s == "ds1000":
        from transformers import AutoTokenizer
        from genlm.latent.algorithm.code import CodeCorrectnessCritic
        from genlm.latent.training.ds1000 import ds1000_full

        rows = ds1000_full(split="test", libraries=[args.library])
        row = rows[args.item % len(rows)]
        tok = AutoTokenizer.from_pretrained(args.model)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        llm = PromptedLLM(model)
        llm.prompt_ids = list(tok.encode(row["prompt"]))
        critic = CodeCorrectnessCritic(llm.vocab, row["code_context"],
                                       timeout_seconds=args.timeout)
        sampler = DirectTokenSampler(llm, autobatch=args.autobatch)
        cfg.update(library=args.library, item=args.item, timeout=args.timeout,
                   critic=True)
        return Built(sampler, critic, llm.prompt_ids, cfg)

    raise ValueError(args.scenario)


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario",
                   choices=["direct", "synth", "product", "awrs", "set", "lora", "cot", "ds1000"],
                   default="direct")
    p.add_argument("--model", default="gpt2")
    p.add_argument("--backend", default="vllm", choices=["vllm", "mlx", "hf"],
                   help="engine under test; only vllm has a raw ceiling")
    p.add_argument("--label", default="dev",
                   help="version key for the results store (main/speedup-old/speedup-now)")
    p.add_argument("--store", default="bench", help="results store name (results/<store>.jsonl)")
    p.add_argument("--paths", default="smc,raw",
                   help="comma list of smc|raw to run")
    p.add_argument("--n-particles", type=int, default=16)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--ess-threshold", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--n-warmup", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=3)
    p.add_argument("--gpu-mem", type=float, default=0.6)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--prompt", default="The")
    p.add_argument("--eos", choices=["natural", "newline"], default="natural")
    p.add_argument("--no-critic", action="store_true")
    p.add_argument("--no-autobatch", dest="autobatch", action="store_false",
                   help="construct samplers/SMC with autobatch seat wrapping OFF (default on)")
    p.add_argument("--n-smcs", type=int, default=1,
                   help="concurrent SMC runs per trial (batched SMC via asyncio.gather)")
    p.add_argument("--window-stats", action="store_true",
                   help="print per-window cohort histograms after each smc path")
    p.add_argument("--no-prefix-cache", action="store_true")
    p.add_argument("--constraint", choices=["alpha", "json"], default="alpha")
    p.add_argument("--potential-us", type=int, default=500,
                   help="synth scenario: per-call cost of the CPU potential, microseconds")
    p.add_argument("--lora-adapter", default=LORA_ADAPTER,
                   help="HF LoRA adapter for the lora scenario (base model + rank read from it)")
    p.add_argument("--draw", default="gumbel_max",
                   choices=["gumbel_max", "multinomial", "inverse_cdf"],
                   help="token picker (set_draw_method); process-wide")
    p.add_argument("--critic-split", action="store_true",
                   help="extra no-critic run -> isolate critic vs rollout time")
    # cot / ds1000
    p.add_argument("--question", default=(
        "Natalia sold clips to 48 of her friends in April, and then she sold half "
        "as many clips in May. How many clips did she sell altogether?"))
    p.add_argument("--answer", default="72")
    p.add_argument("--library", default="Pandas")
    p.add_argument("--item", type=int, default=0)
    p.add_argument("--timeout", type=float, default=6.0)
    p.add_argument("--compare", action="store_true", help="print compare table and exit")
    return p.parse_args()


async def main():
    args = parse_args()
    if args.compare:
        print(bc.compare_table(args.store))
        return

    if args.draw != "gumbel_max":
        from genlm.control.util import set_draw_method

        set_draw_method(args.draw)  # process-wide picker

    model_name, engine_opts, post_engine = scenario_engine(args)
    args.model = model_name  # so cfg/tokenizer use the resolved (lora base) name
    model = bc.build_engine(model_name, engine_opts, args.backend)
    if post_engine:
        post_engine(model)

    built = scenario_build(args, model)
    version, env = bc.version_tag(args.label), bc.env_tag()

    if args.n_smcs != 1:
        built.config["n_smcs"] = args.n_smcs
    want = [p.strip() for p in args.paths.split(",") if p.strip()]
    print("=" * 72)
    print(f"scenario={args.scenario} model={model_name} N={args.n_particles} "
          f"max_tokens={args.max_tokens} label={args.label}")
    print(f"config={built.config}")

    async def smc_run():
        return await bc.run_smc(
            built.sampler, built.critic, n_particles=args.n_particles,
            max_tokens=args.max_tokens, ess_threshold=args.ess_threshold, seed=args.seed,
            autobatch=args.autobatch, n_smcs=args.n_smcs)

    for path in want:
        try:
            if path == "raw":
                if not built.supports_raw:
                    print("  raw      : skipped (no engine ceiling on this version)")
                    continue

                def raw():
                    dt = bc.raw_ceiling(model, built.prompt_ids,
                                        n_particles=args.n_particles,
                                        max_tokens=args.max_tokens)
                    return dt, None

                # raw is sync; wrap in the async trials contract
                med, _, dts = await bc.trials(lambda: _aw(raw()),
                                              n_warmup=args.n_warmup, n_trials=args.n_trials)
                bc.record(args.store, bc.Result(
                    scenario=args.scenario, config=built.config, path="raw",
                    dt_median=med, dt_all=dts, version=version, env=env,
                    extra={"draw": args.draw}))
                print(f"  raw      : {med:7.3f}s  (engine decode ceiling)")
                continue

            med, last, dts = await bc.trials(smc_run,
                                             n_warmup=args.n_warmup, n_trials=args.n_trials)
            mean_len = float(np.mean([len(c) for c in last.contexts]))
            extra = {"draw": args.draw}
            if args.critic_split and built.critic is not None:
                # rollout-only run (drop the critic) -> critic_s = with - without
                base_critic = built.critic
                built.critic = None
                nc_med, _, _ = await bc.trials(smc_run,
                                               n_warmup=0, n_trials=args.n_trials)
                built.critic = base_critic
                extra.update(rollout_s=nc_med, critic_s=med - nc_med)
            bc.record(args.store, bc.Result(
                scenario=args.scenario, config=built.config, path=path,
                dt_median=med, dt_all=dts, version=version, env=env,
                log_ml=float(last.log_ml), mean_len=mean_len, extra=extra))
            line = f"  {path:<8} : {med:7.3f}s  log_ml={last.log_ml:+.4f}  mean_len={mean_len:.1f}"
            if "rollout_s" in extra:
                line += f"  [rollout={extra['rollout_s']:.3f}s critic={extra['critic_s']:.3f}s]"
            print(line)
            if args.window_stats:
                _print_window_stats(model)
        except bc._Unsupported as e:
            print(f"  {path:<8} : skipped ({e})")

    await built.sampler.cleanup()
    print("=" * 72)
    print(f"recorded -> {bc.store_path(args.store)}")


async def _aw(value):
    """Adapt a already-computed (dt, payload) to the async trials contract."""
    return value


if __name__ == "__main__":
    asyncio.run(main())
