"""Gate-2 parity: the engine-accelerated BurstLoop vs the ground-truth StepLoop.

Both loops run the identical controller SMC algorithm; the ONLY intended difference is
the source of next-token logits:

* StepLoop re-prefills the full context every step (``PromptedLLM.logw_next`` ->
  ``AsyncVirtualLM.next_token_logprobs``);
* BurstLoop reads warm-KV decode logits inside the vLLM engine.

So the two should agree up to the warm-KV-vs-reprefill numerical residual
(~1e-2/logit), which may occasionally flip a single Gumbel-max draw but is small
and unbiased -- NOT a systematic divergence. A consistent ``log_ml`` bias or
sequences that systematically run longer in the burst is a BUG, not a tolerance.

Gate 2a: unconstrained ``DirectTokenSampler(llm)``.
Gate 2b: constrained ``DirectTokenSampler(llm * coerced_boolfsa)`` -- the additive
factor folded into the sampler target (the proposal IS the product).

**Slow-reference cache.** The StepLoop reference is deterministic per
(target, N, ess, max_tokens, seed) and is the expensive half (per-token re-prefill
+ O(V) factor ``logw_next``). It is cached in ``gate2_snapshot.json`` so a normal
run executes ONLY the burst and compares against the snapshot. Regenerate after
changing the slow algorithm / configs / model with::

    GATE2_REGEN=1 pytest tests/sampler/test_engine_native.py

(mirrors gate-1's ``parity_snapshot.json`` discipline). The comparison is the
no-bias check (shared prefix, log_ml within the warm-KV residual, no systematic
length gap) -- NOT byte-equality -- so caching the deterministic slow side is sound.

Requires CUDA + vLLM; uses gpt2. Run on the box with
``VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1``.
"""

import asyncio
import json
import os
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("engine-native parity needs CUDA + vLLM", allow_module_level=True)

from genlm.control.constant import EndOfSequence  # noqa: E402
from genlm.control.potential.built_in.llm import PromptedLLM  # noqa: E402
from genlm.control.potential.built_in.wfsa import BoolFSA  # noqa: E402
from genlm.control.sampler.token import DirectTokenSampler, AWRS  # noqa: E402
from genlm.control.sampler.controller import (  # noqa: E402
    Controller,
    StepLoop,
    BurstLoop,
    can_burst,
)

MODEL = "gpt2"
SEED = 1234

# Slow-reference snapshot (see module docstring). REGEN recomputes + rewrites it.
_SNAPSHOT_PATH = os.path.join(os.path.dirname(__file__), "gate2_snapshot.json")
_REGEN = bool(os.environ.get("GATE2_REGEN"))
try:
    with open(_SNAPSHOT_PATH) as _f:
        _SNAPSHOT = json.load(_f)
except FileNotFoundError:  # pragma: no cover
    _SNAPSHOT = {}


def _save_snapshot():  # pragma: no cover - only in REGEN
    with open(_SNAPSHOT_PATH, "w") as f:
        json.dump(_SNAPSHOT, f, indent=1, sort_keys=True)


def _seed(s=SEED):
    np.random.seed(s)
    torch.manual_seed(s)


def _ctx_ids(ctx):
    """Token-id view of a particle context (EOS -> the sentinel string)."""
    out = []
    for t in ctx:
        if isinstance(t, EndOfSequence):
            out.append("EOS")
        else:
            out.append(t.token_id)
    return out


def _controller(make_sampler, n_particles, ess_threshold, max_tokens):
    # `make_sampler` is a factory so the slow and burst runs each get a fresh
    # sampler (an AWRS carries its own RNG; a fresh one per run keeps the two
    # comparable, seeded by `_seed`).
    return Controller(
        unit_sampler=make_sampler(),
        critic=None,
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        twist_with_critic=False,
    )


def _key(label, n_particles, ess_threshold, max_tokens, seed):
    return f"{label}|N={n_particles}|ess={ess_threshold}|mt={max_tokens}|seed={seed}"


def _run_slow(make_sampler, n_particles, ess_threshold, max_tokens, seed):
    """The ground-truth StepLoop reference (the expensive, cached half)."""
    from genlm.control.sampler.sequence import Sequences, _unpack_particles

    _seed(seed)
    controller = _controller(make_sampler, n_particles, ess_threshold, max_tokens)
    t0 = time.perf_counter()
    parts = asyncio.run(StepLoop(controller).run())
    dt = time.perf_counter() - t0
    seq = Sequences(*_unpack_particles(parts))
    return {
        "contexts": [_ctx_ids(p.context) for p in parts],
        "logw": [float(p.logw) for p in parts],
        "log_ml": float(seq.log_ml),
        "wall": dt,
    }


def _slow_cached(label, make_sampler, n_particles, ess_threshold, max_tokens, seed):
    """Return the cached slow reference; (re)compute + persist it under REGEN or
    when the key is absent (first run with no snapshot)."""
    key = _key(label, n_particles, ess_threshold, max_tokens, seed)
    if _REGEN or key not in _SNAPSHOT:
        _SNAPSHOT[key] = _run_slow(
            make_sampler, n_particles, ess_threshold, max_tokens, seed
        )
        if _REGEN:
            _save_snapshot()
    return _SNAPSHOT[key]


def _run_burst(make_sampler, n_particles, ess_threshold, max_tokens, seed):
    from genlm.control.sampler.sequence import Sequences, _unpack_particles

    _seed(seed)
    controller = _controller(make_sampler, n_particles, ess_threshold, max_tokens)
    driver = BurstLoop(controller)
    t0 = time.perf_counter()
    parts = asyncio.run(driver.run())
    dt = time.perf_counter() - t0
    seq = Sequences(*_unpack_particles(parts))
    return {
        "contexts": [_ctx_ids(p.context) for p in parts],
        "logw": [float(p.logw) for p in parts],
        "log_ml": float(seq.log_ml),
        "wall": dt,
        "n_bursts": driver.n_bursts,
    }


def _compare(label, ess_threshold, n_particles, slow, burst):
    """Burst-vs-(cached)-slow stats + report. ``slow``/``burst`` are the dicts
    from ``_slow_cached`` / ``_run_burst``."""
    slow_ctx, burst_ctx = slow["contexts"], burst["contexts"]
    slow_w = np.array(slow["logw"])
    burst_w = np.array(burst["logw"])

    n_match = sum(a == b for a, b in zip(slow_ctx, burst_ctx))
    slow_lens = [len(c) for c in slow_ctx]
    burst_lens = [len(c) for c in burst_ctx]
    # First-divergence step per particle: isolates single-flip-then-cascade
    # (the warm-KV signature) from a step-1 wiring bug.
    first_div = []
    for a, b in zip(slow_ctx, burst_ctx):
        k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        first_div.append(k if k is not None else min(len(a), len(b)))

    matched = [
        abs(slow_w[i] - burst_w[i])
        for i in range(n_particles)
        if slow_ctx[i] == burst_ctx[i]
    ]
    signed = burst_w - slow_w
    finite = signed[np.isfinite(signed)]

    lines = [
        f"=== {label}  ess={ess_threshold}  N={n_particles} (slow cached) ===",
        f"contexts match: {n_match}/{n_particles}",
        f"first-divergence step per particle: {first_div}",
        f"mean len slow={np.mean(slow_lens):.3f} burst={np.mean(burst_lens):.3f}",
        f"max |logw diff| over matched contexts: "
        f"{max(matched) if matched else float('nan'):.4e}",
        f"signed logw diff (burst-slow): mean={np.mean(finite):+.4e} "
        f"std={np.std(finite):.4e} n={len(finite)}",
        f"slow log_ml={slow['log_ml']:.6f}  burst log_ml={burst['log_ml']:.6f}  "
        f"diff={burst['log_ml'] - slow['log_ml']:+.6f}",
        f"burst wall={burst['wall']:.2f}s  bursts opened={burst['n_bursts']}",
    ]
    print("\n" + "\n".join(lines))

    return {
        "n_bursts": burst["n_bursts"],
        "n_match": n_match,
        "log_ml_diff": float(burst["log_ml"] - slow["log_ml"]),
        "slow_log_ml": float(slow["log_ml"]),
        "burst_log_ml": float(burst["log_ml"]),
        "mean_len_slow": float(np.mean(slow_lens)),
        "mean_len_burst": float(np.mean(burst_lens)),
    }


@pytest.fixture(scope="module")
def llm():
    from genlm.backend.llm import AsyncVirtualLM

    model = AsyncVirtualLM.from_name(
        MODEL,
        engine_opts={
            "gpu_memory_utilization": 0.3,
            "max_model_len": 256,
            "enable_prefix_caching": True,
        },
    )
    return PromptedLLM(model, eos_byte_strings=[b"\n"])


def _boolfsa_target(llm, regex):
    return llm * BoolFSA.from_regex(regex).coerce(llm, f=b"".join)


# ----- gate 2a: unconstrained ----------------------------------------------


@pytest.mark.parametrize("ess_threshold", [0.0, 0.5])
def test_unconstrained_burst_vs_slow(llm, ess_threshold):
    llm.set_prompt_from_str("The")

    def make():
        return DirectTokenSampler(llm)

    assert can_burst(_controller(make, 8, ess_threshold, 12))
    slow = _slow_cached("unconstrained", make, 8, ess_threshold, 12, SEED)
    burst = _run_burst(make, 8, ess_threshold, 12, SEED)
    stats = _compare("unconstrained", ess_threshold, 8, slow, burst)
    # Pure warm-KV residual (no factor): the burst draws from the same normalized
    # LM logw_next as the slow path, so length and log_ml track tightly.
    assert abs(stats["mean_len_burst"] - stats["mean_len_slow"]) <= 2.0, (
        f"length diverged (slow {stats['mean_len_slow']:.2f} vs "
        f"burst {stats['mean_len_burst']:.2f})"
    )
    assert abs(stats["log_ml_diff"]) <= 0.2, (
        f"log_ml diverged (slow {stats['slow_log_ml']:.4f} vs "
        f"burst {stats['burst_log_ml']:.4f})"
    )


# ----- gate 2b: constrained (additive factor folded into target) -----------


@pytest.mark.parametrize("ess_threshold", [0.0, 0.5])
def test_constrained_boolfsa_burst_vs_slow(llm, ess_threshold):
    llm.set_prompt_from_str("The")
    target = _boolfsa_target(llm, r"[a-z ]+")

    def make():
        return DirectTokenSampler(target)

    assert can_burst(_controller(make, 16, ess_threshold, 12))
    slow = _slow_cached("boolfsa[a-z ]+", make, 16, ess_threshold, 12, SEED)
    burst = _run_burst(make, 16, ess_threshold, 12, SEED)
    stats = _compare("boolfsa[a-z ]+", ess_threshold, 16, slow, burst)
    # Low-variance constrained case (broad allowed set, full-length sequences):
    # the warm-KV residual is small and unbiased, so a single-seed log_ml diff
    # and the mean length stay tight.
    assert abs(stats["mean_len_burst"] - stats["mean_len_slow"]) <= 2.0, (
        f"length diverged (slow {stats['mean_len_slow']:.2f} vs "
        f"burst {stats['mean_len_burst']:.2f})"
    )
    assert abs(stats["log_ml_diff"]) <= 0.2, (
        f"log_ml diverged (slow {stats['slow_log_ml']:.4f} vs "
        f"burst {stats['burst_log_ml']:.4f})"
    )


# ----- gate 2b': a tighter constraint that FORCES resampling (exercises the
#       burst pop-out / relaunch path, which the looser constraint does not) ---


def test_constrained_forces_resample_burst_vs_slow(llm):
    """Vowels+space only: sharply disagrees with the LLM, so per-step weights
    spread and ESS crosses 0.5 -> the burst must pop out, resample, and relaunch.

    This constraint narrows the proposal to ~53 tokens and produces very short
    sequences (~2-3 tokens), so a single run is high-variance Monte Carlo. We
    therefore (1) assert the pop-out/relaunch path actually ran (n_bursts > 1),
    and (2) check that the burst log_ml is UNBIASED by averaging the burst-vs-
    slow log_ml difference across several seeds -- the mean should sit near 0
    even though any single seed is noisy.
    """
    llm.set_prompt_from_str("The")
    target = _boolfsa_target(llm, r"[aeiou ]+")

    def make():
        return DirectTokenSampler(target)

    seeds = (1234, 7, 99, 2024, 555, 31, 808, 42, 17, 6, 71, 900)
    diffs, len_gaps = [], []
    any_resample = False
    for seed in seeds:
        slow = _slow_cached("boolfsa[aeiou ]+", make, 16, 0.5, 10, seed)
        burst = _run_burst(make, 16, 0.5, 10, seed)
        s = _compare("boolfsa[aeiou ]+", 0.5, 16, slow, burst)
        diffs.append(s["log_ml_diff"])
        len_gaps.append(s["mean_len_burst"] - s["mean_len_slow"])
        any_resample = any_resample or s["n_bursts"] > 1
    diffs = np.array(diffs)
    len_gaps = np.array(len_gaps)
    sem = diffs.std() / np.sqrt(len(diffs))
    print(f"\n[aeiou ]+ over {len(seeds)} seeds:")
    print(f"  log_ml diff mean={diffs.mean():+.4f} std={diffs.std():.4f} sem={sem:.4f}")
    print(f"  len gap mean={len_gaps.mean():+.3f}")

    assert any_resample, (
        "ESS never crossed across any seed -> the pop-out/relaunch path was "
        "not exercised; pick a tighter constraint"
    )
    # Unbiasedness is the testable claim: the MEAN log_ml diff and MEAN length gap
    # across seeds sit near 0. A persistent same-sign gap would be a real bias bug.
    assert abs(diffs.mean()) <= max(0.3, 2.0 * sem), (
        f"burst log_ml is biased across seeds: mean diff {diffs.mean():+.4f} "
        f"(sem {sem:.4f})"
    )
    assert abs(len_gaps.mean()) <= 1.5, (
        f"burst length is systematically biased: mean gap {len_gaps.mean():+.3f}"
    )


# ----- gate 2c: AWRS (rejection over engine logits + stateful condition) ------


def test_awrs_burst_vs_slow(llm):
    """AWRS over an engine LM + a boolean condition, in the burst: the rejection
    runs over the engine LM logits with the condition checked from its carried
    state (no per-step gather over the vocab). AWRS's weight is a Monte-Carlo
    estimate, so -- like the resample test -- check the burst-vs-slow log_ml is
    UNBIASED across seeds (mean near 0); a persistent same-sign gap would be a bug.
    """
    llm.set_prompt_from_str("The")
    condition = BoolFSA.from_regex(r"[a-z ]+").coerce(llm, f=b"".join)

    def make_seed(s):
        # A fresh AWRS seeded per run so slow and burst share the rejection RNG,
        # and the seed varies the draws across the loop (AWRS uses its own RNG).
        def make():
            return AWRS(llm, condition, seed=s)

        return make

    assert can_burst(_controller(make_seed(SEED), 16, 0.0, 12))

    seeds = (1234, 7, 99, 2024, 555, 31)
    diffs, len_gaps = [], []
    for seed in seeds:
        make = make_seed(seed)
        slow = _slow_cached("awrs[a-z ]+", make, 16, 0.0, 12, seed)
        burst = _run_burst(make, 16, 0.0, 12, seed)
        s = _compare("awrs[a-z ]+", 0.0, 16, slow, burst)
        diffs.append(s["log_ml_diff"])
        len_gaps.append(s["mean_len_burst"] - s["mean_len_slow"])
    diffs = np.array(diffs)
    len_gaps = np.array(len_gaps)
    sem = diffs.std() / np.sqrt(len(diffs))
    print(
        f"\nAWRS [a-z ]+ over {len(seeds)} seeds: "
        f"log_ml diff mean={diffs.mean():+.4f} sem={sem:.4f} "
        f"len gap mean={len_gaps.mean():+.3f}"
    )
    assert abs(diffs.mean()) <= max(0.3, 2.0 * sem), (
        f"AWRS burst log_ml biased: mean diff {diffs.mean():+.4f} (sem {sem:.4f})"
    )
    assert abs(len_gaps.mean()) <= 1.5, (
        f"AWRS burst length biased: mean gap {len_gaps.mean():+.3f}"
    )
