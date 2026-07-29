# Gate-1 / Gate-2 parity-test review

A careful review of the two SMC parity gates against their stated responsibility, then a
proposed revision, subagent stress-test, and the implemented change. Phases below.

## 0. What the gates are responsible for
The acceptance bar (CLAUDE.md): **engine speedups must be EXACT re-implementations, never
algorithm changes; parity with the per-token path is the gate.** Delivered as a chain:
**gate-1** pins our `StepLoop` == the original llamppl `smc_standard` (byte-exact), and
**gate-2** pins the engine `BurstLoop` ≈ the exact path (no-bias, since warm-KV makes
byte-exactness impossible). The chain only holds where gate-1 actually covers the sampler
AND gate-2 routes through a pinned reference.

## 1. Review (per dimension)

### Gate-1 (`test_per_token_parity.py`, byte-exact vs frozen llamppl snapshot)
- **Validity** — STRONG where it covers: contexts exact, logw 1e-9, log_ml 1e-9, JSON
  record. The reference is a frozen capture of the genuine original (the generator's
  `_RefSequenceModel` is a transcription of the deleted `SequenceModel`, RNG-order
  preserved). One residual risk: it's a transcription, frozen.
- **Redundancy** — low; clean 3-sampler × {critic,no} × ess{0,.5,1} matrix.
- **Performance** — fast (mock potentials, no engine; ~0.1s local).
- **Coverage** — **GAP (F1):** only `DirectTokenSampler` (Mock/WeightedSet) and
  `MultiTokenUnitSampler`. **AWRS, Set, and batched (B>1) had NO byte-exact pinning
  anywhere** (test_awrs.py / test_set_sampler.py are MC-closeness, not vs-original-SMC).
- **Parity-fulfillment** — yes for the covered samplers; the gap (F1) means the
  "StepLoop==original" half of the chain was UNPROVEN for AWRS/Set — yet gate-2 leaned on
  it.
- **Rigor** — high (byte-exact) where covered; one seed is sufficient (determinism).

### Gate-2 (`test_engine_native.py`, no-bias vs reference)
- **Validity** — the no-bias approach is correct (warm-KV residual is real + unbiased; each
  test averages the log_ml/length diff across seeds, asserts |mean| ≤ max(floor,k·sem)).
- **Redundancy** — **F8:** `set[a-z]+` and `slowcadence-base` references were generated but
  read by NO test (orphans). unconstrained ess=0.5 generated, unused.
- **Performance** — **F6:** two tests dominate the 15-min wall: `constrained_boolfsa[0.0]`
  (803s, runs a LIVE StepLoop, only seed 1234 cached) and `multitoken` (854s, 12 seeds ×
  unit-grain burst).
- **Coverage** — broad (unconstrained / constrained / forces-resample / AWRS / 2 critics /
  multiview K=2 / batched-multiview / multitoken / multitoken-multiview / set / batched /
  LoRA / forward-free-gate routing). Good.
- **Parity-fulfillment** — **F2:** `batched_smc` was the lone UNPAIRED test (one-seed batched
  group-mean vs a noisy 12-seed original-ref mean) — it reported a phantom −1.47 "bias"
  (the ref mean is ~+0.9 off the true μ≈−12.1; verified with 60 solo runs). `unconstrained`
  is also single-seed/absolute-threshold. **F9:** Set/multiview compare burst vs *our*
  StepLoop, which is NOT gate-1-pinned → a bug shared by both lanes is invisible.
- **Rigor** — **F3:** power is reference-dependent — burst-vs-StepLoop@same-seed is
  RNG-matched (tight); burst-vs-original is ~independent (low power). **F7:** ~half the
  no-bias tests had NO "burst actually ran" guard (relied on `can_burst()`, which proves
  only capability) → vacuous-pass vector. **F10:** `_compare` computes per-particle weight
  diffs but asserts on none. **F4:** docstring + a stale comment were inaccurate.

## 2. Proposed revision
- Shared `_harness.py` (seed/serialize/load/assert) + `gate2_cases.py` (critics + constants
  + one `CASES` table consumed by gate AND generator) → kill drift, esp. the SILENT
  critic-duplication hazard.
- Collapse the 8 homogeneous gate-2 tests into a `_nobias` driver; add `n_bursts>0` guards
  (F7); switch Set to the cached original (F9); drop orphan refs (F8); fix docs (F4).
- Fix `batched_smc` to a paired burst-vs-StepLoop check (F2). [done first, commit 9b4b7e2]
- Add AWRS + Set to gate-1 byte-exact (F1), restricted to ess=0 (see §4).
- Rejected after stress-test: a B=1 batched byte-exact test (see §3).
- Deferred (need box-regen / llamppl-only / strategic): cache seed-7 to kill the 803s live
  StepLoop (F6); unconstrained multi-seed (F2); converge gate-2 onto RNG-matched StepLoop
  once gate-1 covers all samplers (F3/F9).

## 3. Subagent stress-test (4 adversarial agents)
- **Validity/stats agent** — CONFIRMED F1/F2/F3; FOUND F7 (worse than I'd stated), F8
  (orphans), F10, and that `unconstrained` shares F2's single-seed defect.
- **P1 stress-test agent** — REJECTED my proposed B=1-batched byte-exact test: at B=1 the
  group machinery is degenerate (global==per-group → tests nothing), AND it false-REDs
  (solo sets `record=True` → `np.sort` on resample ancestors; `SMC.batched` has no record →
  resampling cases return a permutation). B>1-vs-solo is infeasible (shared RNG). ⇒ batched
  isolation is a STATISTICAL claim only. Dropped P1.
- **Refactor-design agent** — found the AWRS factory must be `make_sampler(llm, seed)` (per-
  seed rng — a BLOCKER otherwise); `ctx_repr`/`ctx_ids` must stay two functions; Set factory
  fresh-per-call; `reference` is data; unconstrained absolute thresholds can't go through
  `assert_unbiased`. All folded in.
- **Migration-faithfulness agent** — verified the `_nobias` collapse preserved every
  original test's thresholds/guards/length-checks; the one semantic change (Set → cached
  ref) is the intended F9 strengthening.

## 4. Change made (committed) + validation
- `9b4b7e2` batched_smc paired fix; `03f4f95` untrack box.sh; `ba843cd` shared harness +
  gate-2 consolidation + AWRS/Set gate-1 coverage (net −194 LOC).
- **AWRS/Set gate-1:** byte-match the original at ess=0 (gate-1 18→22). Restricted to ess=0
  via per-sampler `SAMPLER_ESS`: AWRS draws from its OWN seeded rng (never the global
  stream), so at ess>0 the resample (which consumes the global stream) desyncs vs the
  original — RNG-order, not bias; the resample path is byte-pinned by the global-rng
  samplers. ess=0 is exactly the regime gate-2 uses AWRS/Set in, closing the chain for them.
- Validated: **gate-1 22/22** local, **gate-2 13/13** box (xdist 4).

## 5. Residual (honest)
- F1 batched: no byte-exact pin possible (no original grouped reference) — statistical only.
- F6 perf, F2 unconstrained multi-seed, F3/F9 gate-2→StepLoop convergence: deferred.
- Snapshots stay gitignored/regenerated; gate-1 only runs where the snapshot was generated.
