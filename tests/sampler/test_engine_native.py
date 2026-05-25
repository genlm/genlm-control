"""Gate-2 parity: the engine-accelerated BurstLoop vs the ground-truth StepLoop.

Both loops run the identical hub SMC algorithm; the ONLY intended difference is
the source of next-token logits:

* StepLoop re-prefills the full context every step (``PromptedLLM.logw_next`` ->
  ``AsyncVirtualLM.next_token_logprobs``);
* BurstLoop reads warm-KV decode logits inside the vLLM engine.

So the two should agree up to the warm-KV-vs-reprefill numerical residual
(~1e-2/logit), which may occasionally flip a single Gumbel-max draw but is small
and unbiased -- NOT a systematic divergence. A consistent ``log_ml`` bias or
sequences that systematically run longer in the window is a BUG, not a tolerance.

Gate 2a: unconstrained ``DirectTokenSampler(llm)``.
Gate 2b: constrained ``DirectTokenSampler(llm * coerced_boolfsa)`` -- the additive
factor folded into the sampler target (the proposal IS the product).

Requires CUDA + vLLM; uses gpt2. Run on the box with
``VLLM_USE_FLASHINFER_SAMPLER=0 OMP_NUM_THREADS=1``.
"""

import asyncio
import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("engine-native parity needs CUDA + vLLM", allow_module_level=True)

from genlm.control.constant import EndOfSequence  # noqa: E402
from genlm.control.potential.built_in.llm import PromptedLLM  # noqa: E402
from genlm.control.potential.built_in.wfsa import BoolFSA  # noqa: E402
from genlm.control.sampler.token import DirectTokenSampler  # noqa: E402
from genlm.control.sampler.hub import (  # noqa: E402
    Controller,
    StepLoop,
    BurstLoop,
    can_burst,
)

MODEL = "gpt2"
SEED = 1234


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


def _hub(target, n_particles, ess_threshold, max_tokens):
    return Controller(
        unit_sampler=DirectTokenSampler(target),
        critic=None,
        n_particles=n_particles,
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        twist_with_critic=False,
    )


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


def _compare(label, target, n_particles, ess_threshold, max_tokens, seed=SEED):
    assert can_burst(_hub(target, n_particles, ess_threshold, max_tokens))

    _seed(seed)
    slow_hub = _hub(target, n_particles, ess_threshold, max_tokens)
    t0 = time.perf_counter()
    slow = asyncio.run(StepLoop(slow_hub).run())
    slow_t = time.perf_counter() - t0

    _seed(seed)
    win_hub = _hub(target, n_particles, ess_threshold, max_tokens)
    driver = BurstLoop(win_hub)
    t0 = time.perf_counter()
    win = asyncio.run(driver.run())
    win_t = time.perf_counter() - t0
    n_windows = driver.n_windows

    slow_ctx = [_ctx_ids(p.context) for p in slow]
    win_ctx = [_ctx_ids(p.context) for p in win]
    slow_w = np.array([p.logw for p in slow])
    win_w = np.array([p.logw for p in win])

    from genlm.control.sampler.sequence import Sequences, _unpack_particles

    slow_seq = Sequences(*_unpack_particles(slow))
    win_seq = Sequences(*_unpack_particles(win))

    n_match = sum(a == b for a, b in zip(slow_ctx, win_ctx))
    slow_lens = [len(c) for c in slow_ctx]
    win_lens = [len(c) for c in win_ctx]
    # First-divergence step per particle: isolates single-flip-then-cascade
    # (the warm-KV signature) from a step-1 wiring bug.
    first_div = []
    for a, b in zip(slow_ctx, win_ctx):
        k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        first_div.append(k if k is not None else min(len(a), len(b)))

    matched = [abs(slow_w[i] - win_w[i]) for i in range(n_particles)
               if slow_ctx[i] == win_ctx[i]]
    # Signed per-particle logw differences (window - slow): mean ~0 => unbiased.
    signed = win_w - slow_w
    finite = signed[np.isfinite(signed)]

    lines = [
        f"=== {label}  ess={ess_threshold}  N={n_particles} ===",
        f"contexts match: {n_match}/{n_particles}",
        f"first-divergence step per particle: {first_div}",
        f"slow lens: {slow_lens}",
        f"win  lens: {win_lens}",
        f"mean len slow={np.mean(slow_lens):.3f} win={np.mean(win_lens):.3f}",
        f"max |logw diff| over matched contexts: "
        f"{max(matched) if matched else float('nan'):.4e}",
        f"signed logw diff (win-slow) over finite particles: "
        f"mean={np.mean(finite):+.4e} std={np.std(finite):.4e} n={len(finite)}",
        f"slow log_ml={slow_seq.log_ml:.6f}  win log_ml={win_seq.log_ml:.6f}  "
        f"diff={win_seq.log_ml - slow_seq.log_ml:+.6f}",
        f"wall-clock: slow={slow_t:.2f}s  window={win_t:.2f}s  "
        f"speedup={slow_t / win_t:.2f}x",
        f"windows opened: {n_windows} (>1 => pop-out/relaunch ran)",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(f"/tmp/gate2_{label.replace(' ', '_').replace('[', '').replace(']', '')}"
              f"_ess{ess_threshold}.txt", "w") as f:
        f.write(report + "\n")

    return {"n_windows": n_windows, "n_match": n_match,
            "log_ml_diff": float(win_seq.log_ml - slow_seq.log_ml),
            "slow_log_ml": float(slow_seq.log_ml),
            "win_log_ml": float(win_seq.log_ml),
            "mean_len_slow": float(np.mean(slow_lens)),
            "mean_len_win": float(np.mean(win_lens))}


# ----- gate 2a: unconstrained ----------------------------------------------


@pytest.mark.parametrize("ess_threshold", [0.0, 0.5])
def test_unconstrained_window_vs_slow(llm, ess_threshold):
    llm.set_prompt_from_str("The")
    _compare("unconstrained", llm, 8, ess_threshold, 12)


# ----- gate 2b: constrained (additive factor folded into target) -----------


@pytest.mark.parametrize("ess_threshold", [0.0, 0.5])
def test_constrained_boolfsa_window_vs_slow(llm, ess_threshold):
    llm.set_prompt_from_str("The")
    # lowercase letters + space; coerced onto the gpt2 token vocab (prunes to
    # tokens whose bytes are all in [a-z ]).
    fsa = BoolFSA.from_regex(r"[a-z ]+")
    coerced = fsa.coerce(llm, f=b"".join)
    target = llm * coerced
    stats = _compare("boolfsa[a-z ]+", target, 16, ess_threshold, 12)
    # This is the low-variance constrained case (broad allowed set, full-length
    # sequences): the warm-KV residual is small and unbiased, so a single-seed
    # log_ml diff and the mean length stay tight.
    assert abs(stats["mean_len_win"] - stats["mean_len_slow"]) <= 2.0, (
        f"length diverged (slow {stats['mean_len_slow']:.2f} vs "
        f"win {stats['mean_len_win']:.2f})"
    )
    assert abs(stats["log_ml_diff"]) <= 0.2, (
        f"log_ml diverged (slow {stats['slow_log_ml']:.4f} vs "
        f"win {stats['win_log_ml']:.4f})"
    )


# ----- gate 2b': a tighter constraint that FORCES resampling (exercises the
#       window pop-out / relaunch path, which the looser constraint does not) ---


def test_constrained_forces_resample_window_vs_slow(llm):
    """Vowels+space only: sharply disagrees with the LLM, so per-step weights
    spread and ESS crosses 0.5 -> the window must pop out, resample, and relaunch.

    This constraint narrows the proposal to ~53 tokens and produces very short
    sequences (~2-3 tokens), so a single run is high-variance Monte Carlo. We
    therefore (1) assert the pop-out/relaunch path actually ran (n_windows > 1),
    and (2) check that the window log_ml is UNBIASED by averaging the window-vs-
    slow log_ml difference across several seeds -- the mean should sit near 0
    even though any single seed is noisy. A persistent same-sign gap would be a
    real bias bug; Monte Carlo noise cancels.
    """
    llm.set_prompt_from_str("The")
    fsa = BoolFSA.from_regex(r"[aeiou ]+")
    coerced = fsa.coerce(llm, f=b"".join)
    target = llm * coerced

    seeds = (1234, 7, 99, 2024, 555, 31, 808, 42, 17, 6, 71, 900)
    diffs, len_gaps = [], []
    any_resample = False
    for seed in seeds:
        s = _compare("boolfsa[aeiou ]+", target, 16, 0.5, 10, seed=seed)
        diffs.append(s["log_ml_diff"])
        len_gaps.append(s["mean_len_win"] - s["mean_len_slow"])
        any_resample = any_resample or s["n_windows"] > 1
    diffs = np.array(diffs)
    len_gaps = np.array(len_gaps)
    sem = diffs.std() / np.sqrt(len(diffs))
    print(f"\n[aeiou ]+ over {len(seeds)} seeds:")
    print(f"  log_ml diffs: {np.array2string(diffs, precision=3)}")
    print(f"  log_ml diff mean={diffs.mean():+.4f} std={diffs.std():.4f} sem={sem:.4f}")
    print(f"  len gaps (win-slow): {np.array2string(len_gaps, precision=2)}")
    print(f"  len gap mean={len_gaps.mean():+.3f}")

    assert any_resample, (
        "ESS never crossed across any seed -> the pop-out/relaunch path was "
        "not exercised; pick a tighter constraint"
    )
    # This is a deliberately pathological config: ~53 allowed tokens, ~2-3 token
    # sequences, so every draw is a near-tie that the warm-KV residual flips, and
    # any single seed is high-variance Monte Carlo (per-seed length/log_ml gaps
    # are large and bidirectional). Unbiasedness is the testable claim: the MEAN
    # log_ml diff and the MEAN length gap across seeds sit near 0. A persistent
    # same-sign gap would be a real bias bug.
    assert abs(diffs.mean()) <= max(0.3, 2.0 * sem), (
        f"window log_ml is biased across seeds: mean diff {diffs.mean():+.4f} "
        f"(sem {sem:.4f})"
    )
    assert abs(len_gaps.mean()) <= 1.5, (
        f"window length is systematically biased: mean gap {len_gaps.mean():+.3f}"
    )
