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
- ``make_controller`` / ``run_case`` / ``compare_runs`` / ``log`` : the engine-agnostic
  run+compare loop. There is ONE control path; a gate compares the case served by the
  engine under test against a reference (a cached snapshot, or the same case served by
  a plain-forward backend).
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


def _result(controller, parts, wall):
    from genlm.control.sampler.sequence import Sequences, _unpack_particles

    seq = Sequences(*_unpack_particles(parts))
    return {
        "contexts": [ctx_ids(p.context) for p in parts],
        "logw": [float(p.logw) for p in parts],
        "log_ml": float(seq.log_ml),
        "wall": wall,
        "n_resamples": controller.n_resamples,
    }


def run_case(
    make_sampler, n_particles, ess_threshold, max_tokens, seed, make_critic=None
):
    """One run at ``seed``. Which engine serves it is decided by the potentials the
    sampler factory builds over -- the control path is the same either way."""
    import asyncio

    seed_all(seed)
    controller = make_controller(
        make_sampler, n_particles, ess_threshold, max_tokens, make_critic
    )
    t0 = time.perf_counter()
    parts = asyncio.run(controller.run())
    return _result(controller, parts, time.perf_counter() - t0)


@pytest.fixture(scope="module", autouse=True)
def threefry_draw():
    """RNG-match run<->reference via the counter-based picker, so a comparison is a tight
    paired check (engine numeric residual only) rather than divergent-path MC noise.
    Import it into a no-bias gate module to arm it there; gate-1 must never see it, since
    its llamppl reference cannot set draw keys and has to stay on ``gumbel_max``."""
    from genlm.control.util import set_draw_method

    set_draw_method("threefry_gumbel")
    yield
    set_draw_method("gumbel_max")


def live_ref_factory(ref_llm):
    """Reference resolver running each case live on ``ref_llm`` -- a ``PromptedLLM`` over
    a second backend, so the comparison measures the engine under test, not the loop."""

    def resolve(case, seed, make, mkc):
        return run_case(
            lambda: case.sampler(ref_llm, seed),
            case.n_particles,
            case.ess,
            case.max_tokens,
            seed,
            (lambda: case.critic(ref_llm)) if case.make_critic is not None else None,
        )

    return resolve


def assert_case_unbiased(
    case,
    llm,
    prompt,
    resolve_ref,
    *,
    ml_floor=0.3,
    ml_k=2.5,
    len_bound=None,
    len_k=None,
    need_resample=False,
):
    """Drive one ``gate2_cases`` case on ``llm`` and assert it is unbiased against
    ``resolve_ref(case, seed, make_sampler, make_critic)``.

    Per seed: run vs reference, accumulate the log_ml + length diffs, then assert the
    MEAN log_ml diff sits within sampling noise of 0. ``len_bound`` bounds the mean
    length gap (sem-aware if ``len_k`` given, else absolute). ``need_resample`` guards
    that the resample path actually fired (no vacuous pass).
    """
    c = case
    floor = c.match_floor
    llm.set_prompt_from_str(prompt)
    mkc = (lambda: c.critic(llm)) if c.make_critic is not None else None
    diffs, len_gaps, matches = [], [], 0
    any_resample = False
    for seed in c.seeds:
        make = lambda s=seed: c.sampler(llm, s)  # noqa: E731
        ref = resolve_ref(c, seed, make, mkc)
        run = run_case(make, c.n_particles, c.ess, c.max_tokens, seed, mkc)
        s = compare_runs(c.label, c.ess, c.n_particles, ref, run)
        diffs.append(s["log_ml_diff"])
        len_gaps.append(s["mean_len_run"] - s["mean_len_ref"])
        matches += s["n_match"]
        any_resample = any_resample or s["n_resamples"] > 0
    total = c.n_particles * len(c.seeds)
    log(f"{c.label}: contexts matching the reference {matches}/{total}")
    if floor is not None:
        # The no-bias check LOOSENS as the comparison degrades: it is `|mean| <=
        # max(floor, k*sem)`, and its tightness comes entirely from the run drawing
        # the same threefry keys as the reference. Lose the pairing and `sem` inflates
        # until any mean passes. This floor is what notices.
        assert matches >= floor, (
            f"{c.label}: only {matches}/{total} contexts match the reference "
            f"(floor {floor}) -- the paired comparison has come apart, so the "
            "no-bias assertion above is no longer tight"
        )
    if need_resample:
        assert any_resample, f"{c.label}: ESS never crossed -- resample path unexercised"
    m, sem = assert_unbiased(diffs, floor=ml_floor, k=ml_k, label=f"{c.label} log_ml")
    log(f"{c.label}: log_ml diff mean={m:+.4f} sem={sem:.4f}")
    if len_bound is not None:
        # k=0 is the absolute bound: |mean| <= max(len_bound, 0).
        assert_unbiased(
            len_gaps, floor=len_bound, k=len_k or 0.0, label=f"{c.label} length"
        )
    return np.array(diffs), np.array(len_gaps)


def compare_runs(label, ess_threshold, n_particles, ref, run):
    """Run-vs-reference stats + report. ``ref`` is the reference (cached or live on a
    second backend), ``run`` the case served by the engine under test."""
    log(
        f"{label} ess={ess_threshold} N={n_particles}: wall={run['wall']:.1f}s "
        f"resamples={run['n_resamples']} log_ml={run['log_ml']:.3f}"
    )
    ref_ctx, run_ctx = ref["contexts"], run["contexts"]
    ref_w = np.array(ref["logw"])
    run_w = np.array(run["logw"])

    n_match = sum(a == b for a, b in zip(ref_ctx, run_ctx))
    ref_lens = [len(c) for c in ref_ctx]
    run_lens = [len(c) for c in run_ctx]
    # First-divergence step per particle: isolates single-flip-then-cascade
    # (the engine-numeric signature) from a step-1 wiring bug.
    first_div = []
    for a, b in zip(ref_ctx, run_ctx):
        k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
        first_div.append(k if k is not None else min(len(a), len(b)))

    matched = [
        abs(ref_w[i] - run_w[i])
        for i in range(n_particles)
        if ref_ctx[i] == run_ctx[i]
    ]
    signed = run_w - ref_w
    finite = signed[np.isfinite(signed)]

    lines = [
        f"=== {label}  ess={ess_threshold}  N={n_particles} ===",
        f"contexts match: {n_match}/{n_particles}",
        f"first-divergence step per particle: {first_div}",
        f"mean len ref={np.mean(ref_lens):.3f} run={np.mean(run_lens):.3f}",
        f"max |logw diff| over matched contexts: "
        f"{max(matched) if matched else float('nan'):.4e}",
        f"signed logw diff (run-ref): mean={np.mean(finite):+.4e} "
        f"std={np.std(finite):.4e} n={len(finite)}",
        f"ref log_ml={ref['log_ml']:.6f}  run log_ml={run['log_ml']:.6f}  "
        f"diff={run['log_ml'] - ref['log_ml']:+.6f}",
        f"run wall={run['wall']:.2f}s",
    ]
    print("\n" + "\n".join(lines))

    return {
        "n_resamples": run["n_resamples"],
        "n_match": n_match,
        "log_ml_diff": float(run["log_ml"] - ref["log_ml"]),
        "ref_log_ml": float(ref["log_ml"]),
        "run_log_ml": float(run["log_ml"]),
        "mean_len_ref": float(np.mean(ref_lens)),
        "mean_len_run": float(np.mean(run_lens)),
    }
