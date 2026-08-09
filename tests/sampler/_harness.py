"""Shared harness for the SMC parity gates (gate-1 byte-exact + gate-2 no-bias) and
their reference generators. One definition of the bits that MUST agree across the
gate, its generator, and the other gate -- drift here is a silent false-green.

- ``seed_all`` : the single seed call (was copied in 5 files).
- ``ctx_repr`` / ``ctx_ids`` / ``num`` : context + number serialization. TWO ctx
  flavors kept distinct on purpose -- ``ctx_repr`` is the byte/string view gate-1's
  Mock vocab needs; ``ctx_ids`` is the engine token-id view gate-2 needs. Both flatten
  nested unit lists and map EOS to a sentinel.
- ``load_snapshot`` : load a reference snapshot, optionally enforcing a ``__config__``.
- ``assert_unbiased`` : the ``|mean| <= max(floor, k*sem)`` no-bias check gate-2 repeats.
- ``make_controller`` / ``run_burst`` / ``run_steploop`` / ``compare_runs`` / ``log`` :
  the engine-agnostic run+compare loop. Shared by every no-bias gate, one per engine
  (vLLM in ``test_engine_native.py``, MLX in ``test_engine_mlx.py``); only the engine
  behind the ``PromptedLLM`` and the choice of reference differ.
"""

import json
import time

import numpy as np
import pytest

from genlm.control.constant import EndOfSequence


def seed_all(s):
    """Seed numpy + torch (in that order) -- the per-case seed every gate uses. Also
    seeds the counter-based picker so a threefry draw is identical across lanes/devices."""
    np.random.seed(s)
    import torch

    torch.manual_seed(s)
    from genlm.control.util import set_draw_seed

    set_draw_seed(s)


def ctx_repr(ctx):
    """Byte/string-level, comparison-stable repr of a context (gate-1: Mock vocab)."""

    def one(t):
        if hasattr(t, "type_"):  # EOS / any EndOfSequence (EOS = EndOfSequence("EOS"))
            return f"<EOS:{getattr(t, 'type_', 'EOS')}>"
        if isinstance(t, list):
            return [one(x) for x in t]
        if isinstance(t, bytes):
            return "b:" + t.hex()
        return repr(t)

    return [one(t) for t in ctx]


def ctx_ids(ctx):
    """Token-id view of a context (gate-2: engine tokens). EOS -> "EOS"; nested
    unit lists flattened so a unit run and a token run with the same tokens compare equal."""
    out = []

    def emit(t):
        if isinstance(t, EndOfSequence):
            out.append("EOS")
        elif isinstance(t, list):
            for s in t:
                emit(s)
        else:
            out.append(t.token_id)

    for t in ctx:
        emit(t)
    return out


def num(x):
    """nan/inf-safe scalar for JSON: special floats -> tagged strings, else float()."""
    if np.isnan(x):
        return "nan"
    if np.isneginf(x):
        return "-inf"
    if np.isposinf(x):
        return "inf"
    return float(x)


def load_snapshot(path, expected_config=None):
    """Load a reference snapshot JSON. If ``expected_config`` is given, pop the stored
    ``__config__`` and raise on mismatch (so a comparison against a snapshot generated
    under a different model/prompt/eos fails loudly, never silently). Returns the dict
    (without ``__config__``). Raises FileNotFoundError if missing -- the caller decides
    whether that is a skip (gate-1) or an empty-snapshot hard-error-per-key (gate-2)."""
    with open(path) as f:
        snap = json.load(f)
    stored = snap.pop("__config__", None)
    if expected_config is not None and stored is not None and stored != expected_config:
        raise RuntimeError(
            f"snapshot at {path} was generated under {stored} but the tests now run "
            f"under {expected_config}; regenerate it."
        )
    return snap


def assert_unbiased(diffs, *, floor, k, label):
    """No-bias check: |mean(diffs)| <= max(floor, k*sem). Returns (mean, sem) for logging.
    Used by every multi-seed gate-2 no-bias assertion (log_ml and length)."""
    diffs = np.asarray(diffs, dtype=float)
    sem = float(diffs.std() / np.sqrt(len(diffs)))
    mean = float(diffs.mean())
    assert abs(mean) <= max(floor, k * sem), (
        f"{label}: biased -- mean {mean:+.4f} exceeds max({floor}, {k}*sem={k * sem:.4f})"
    )
    return mean, sem


# --- the engine-agnostic run + compare loop -------------------------------------------

_T0 = time.perf_counter()


def log(msg):
    """Flushed, timestamped progress line. Streams live under ``pytest -s`` so a long
    gate run shows constant progress (per-run wall, bursts) instead of going dark."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def make_controller(make_sampler, n_particles, ess_threshold, max_tokens, make_critic=None):
    """A one-group controller for a no-bias case.

    ``make_sampler``/``make_critic`` are factories so each run gets a fresh sampler and
    critic (an AWRS carries its own RNG). ``twist_with_critic`` mirrors ``SMC.__call__``
    exactly: per-step twist iff ``ess_threshold > 0``."""
    from genlm.control.sampler.smc import Controller

    return Controller(
        samplers=[make_sampler()],
        critics=[make_critic() if make_critic is not None else None],
        group_sizes=[n_particles],
        ess_threshold=ess_threshold,
        max_tokens=max_tokens,
        twist_with_critic=ess_threshold > 0,
    )


def _result(controller, parts, wall, n_lane_rows):
    from genlm.control.sampler.sequence import Sequences, _unpack_particles

    seq = Sequences(*_unpack_particles(parts))
    return {
        "contexts": [ctx_ids(p.context) for p in parts],
        "logw": [float(p.logw) for p in parts],
        "log_ml": float(seq.log_ml),
        "wall": wall,
        "n_bursts": n_lane_rows,  # lane rows opened; >0 proves the lane path ran
        "n_resamples": controller.n_resamples,
    }


def run_burst(make_sampler, n_particles, ess_threshold, max_tokens, seed, make_critic=None):
    """One engine-accelerated (lanes-on) run at ``seed``."""
    import asyncio

    from genlm.control.lane_runner import LaneRunner, lane_blocker

    seed_all(seed)
    controller = make_controller(
        make_sampler, n_particles, ess_threshold, max_tokens, make_critic
    )
    reason = lane_blocker(controller)
    assert reason is None, f"lane path unavailable: {reason}"
    runner = LaneRunner(controller)
    opened = 0
    _open = runner.open_row

    def counting_open(p):
        nonlocal opened
        opened += 1
        _open(p)

    runner.open_row = counting_open
    t0 = time.perf_counter()
    parts = asyncio.run(controller.run(lanes=runner))
    return _result(controller, parts, time.perf_counter() - t0, opened)


def run_steploop(
    make_sampler, n_particles, ess_threshold, max_tokens, seed, make_critic=None
):
    """Same as :func:`run_burst` but lanes-off (``accelerate="off"`` ground truth,
    gate-1-pinned == original) -- the live reference for configs with no cached
    snapshot, and RNG-matched to the lane path so the paired diff is tight."""
    import asyncio

    seed_all(seed)
    controller = make_controller(
        make_sampler, n_particles, ess_threshold, max_tokens, make_critic
    )
    parts = asyncio.run(controller.run())
    return _result(controller, parts, 0.0, 0)


@pytest.fixture(scope="module", autouse=True)
def threefry_draw():
    """RNG-match burst<->StepLoop via the counter-based picker, so a comparison is a tight
    paired check (warm-KV residual only) rather than divergent-path MC noise. Import it
    into a no-bias gate module to arm it there; gate-1 must never see it, since its
    llamppl reference cannot set draw keys and has to stay on ``gumbel_max``."""
    from genlm.control.util import set_draw_method

    set_draw_method("threefry_gumbel")
    yield
    set_draw_method("gumbel_max")


def live_steploop_ref(case, seed, make, mkc):
    """A StepLoop at this seed on the engine under test -- the reference for a gate with
    no cached snapshot of its own."""
    return run_steploop(make, case.n_particles, case.ess, case.max_tokens, seed, mkc)


def assert_case_unbiased(
    case,
    llm,
    prompt,
    resolve_ref=live_steploop_ref,
    *,
    ml_floor=0.3,
    ml_k=2.5,
    len_bound=None,
    len_k=None,
    need_resample=False,
    need_rounds=False,
):
    """Drive one ``gate2_cases`` case and assert the burst is unbiased against a reference.

    Per seed: burst vs ``resolve_ref(case, seed, make_sampler, make_critic)``, accumulate
    the log_ml + length diffs, then assert the MEAN log_ml diff is within sampling noise
    of 0. ``len_bound`` bounds the mean length gap (sem-aware if ``len_k`` given, else
    absolute). ``need_resample`` / ``need_rounds`` guard that the resample / per-unit-round
    path actually fired (no vacuous pass); an ``n_bursts>0`` guard always runs.

    ``resolve_ref`` is the only engine-specific part: vLLM reads cached snapshots where it
    has them, everything else runs live.
    """
    from genlm.control.lane_runner import lane_blocker

    c = case
    floor = c.match_floor
    llm.set_prompt_from_str(prompt)
    mkc = (lambda: c.critic(llm)) if c.make_critic is not None else None
    blocker = lane_blocker(
        make_controller(
            lambda: c.sampler(llm, c.seeds[0]), c.n_particles, c.ess, c.max_tokens, mkc
        )
    )
    assert blocker is None, f"{c.label}: not acceleratable -- {blocker}"
    diffs, len_gaps, matches = [], [], 0
    any_resample, max_bursts = False, 0
    for seed in c.seeds:
        make = lambda s=seed: c.sampler(llm, s)  # noqa: E731
        slow = resolve_ref(c, seed, make, mkc)
        burst = run_burst(make, c.n_particles, c.ess, c.max_tokens, seed, mkc)
        s = compare_runs(c.label, c.ess, c.n_particles, slow, burst)
        diffs.append(s["log_ml_diff"])
        len_gaps.append(s["mean_len_burst"] - s["mean_len_slow"])
        matches += s["n_match"]
        any_resample = any_resample or s["n_resamples"] > 0
        max_bursts = max(max_bursts, s["n_bursts"])
    assert max_bursts > 0, f"{c.label}: burst never opened (n_bursts==0)"
    total = c.n_particles * len(c.seeds)
    log(f"{c.label}: contexts matching the reference {matches}/{total}")
    if floor is not None:
        # The no-bias check LOOSENS as the comparison degrades: it is `|mean| <=
        # max(floor, k*sem)`, and its tightness comes entirely from the burst drawing
        # the same threefry keys as the reference. Lose the pairing and `sem` inflates
        # until any mean passes. This floor is what notices.
        assert matches >= floor, (
            f"{c.label}: only {matches}/{total} contexts match the reference "
            f"(floor {floor}) -- the paired comparison has come apart, so the "
            "no-bias assertion above is no longer tight"
        )
    if need_resample:
        assert any_resample, f"{c.label}: ESS never crossed -- resample path unexercised"
    if need_rounds:
        assert max_bursts > 1, f"{c.label}: single unit round -- per-unit loop unexercised"
    m, sem = assert_unbiased(diffs, floor=ml_floor, k=ml_k, label=f"{c.label} log_ml")
    log(f"{c.label}: log_ml diff mean={m:+.4f} sem={sem:.4f}")
    if len_bound is not None:
        # k=0 is the absolute bound: |mean| <= max(len_bound, 0).
        assert_unbiased(
            len_gaps, floor=len_bound, k=len_k or 0.0, label=f"{c.label} length"
        )
    return np.array(diffs), np.array(len_gaps)


def compare_runs(label, ess_threshold, n_particles, ref, burst):
    """Burst-vs-reference stats + report. ``ref`` is the slow-path reference (cached or
    live), ``burst`` the engine-accelerated run."""
    log(
        f"{label} ess={ess_threshold} N={n_particles}: wall={burst['wall']:.1f}s "
        f"bursts={burst['n_bursts']} resamples={burst['n_resamples']} "
        f"log_ml={burst['log_ml']:.3f}"
    )
    slow = ref
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
        f"=== {label}  ess={ess_threshold}  N={n_particles} ===",
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
        "n_resamples": burst["n_resamples"],
        "n_match": n_match,
        "log_ml_diff": float(burst["log_ml"] - slow["log_ml"]),
        "slow_log_ml": float(slow["log_ml"]),
        "burst_log_ml": float(burst["log_ml"]),
        "mean_len_slow": float(np.mean(slow_lens)),
        "mean_len_burst": float(np.mean(burst_lens)),
    }
