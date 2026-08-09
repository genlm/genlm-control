# Lane server

The backend is a persistent async request server; SMC is plain async client code. Two
request lifetimes — one-shot scores and resident decode lanes — replace the burst mode.
No engine-drives-control callback is visible outside the backend; threading is a backend
implementation detail. Groups pace independently: the object that coupled all B groups
to one cadence (the global round loop) does not exist in this shape.

Parity bar unchanged: exact SMC math, unbiased weights, per-group log_ml. Gates are
instruments — they re-aim at the new shape (references regenerate from its own off-path);
they do not constrain it.

## 1. Backend API (vLLM and MLX, one contract)

```python
# One-shot scoring — main's surface, unchanged
logprobs = await llm.next_token_logprobs(token_ids)     # [V] at last position
scores   = await llm.score_prompt(token_ids)            # per-position prompt logps

# Resident lane
lane = llm.open_lane(prompt_ids, lora_name=None, group=handle)
w    = await lane.next()      # this step's post-processor logits row [V]
lane.feed(token_id)           # commit; engine steps once every resident lane has fed
await lane.close()            # abort the engine request; lane is dead
```

Contract:

- `next()` resolves once per engine step the lane was scheduled in. A second `next()`
  before `feed()` raises; `feed()` without a pending `next()` raises; any call on a
  closed lane raises. Violations are immediate exceptions at the seam, not races.
- **`close` is a first-class answer to a step.** A row that will not read the next warm
  closes *instead of feeding its final token* — the engine only ever needed the feed to
  compute the step after it, and committed tokens live control-side. This kills, by
  construction: the placeholder commit, the one-wasted-forward-per-row-per-unit
  residual (today the abort races the drain and lands a step late), and the
  `terminate_when` blocker reason (a step appending `[tok, EOS]` closes at EOS; nothing
  is ever committed engine-side that control didn't bank).
- **Liveness**: feed-or-close promptly. vLLM steps all resident lanes together, so one
  withheld feed stalls every lane. A lane pausing past a step (unit boundary, group
  resample) must `close`; reopen with `open_lane(prefix + committed)` — prefix cache
  (`enable_prefix_caching=True`) prices the rejoin.
- One-shots may be submitted at any time, lanes resident or not. Spike P3: a
  `prompt_logprobs` request finished mid-decode with residents unperturbed.
- The engine loop runs for the server's lifetime. Zero live requests parks it; a
  queued add revives it (spike P2). There is no burst begin/end.
- `group=`: K lanes opened under one handle are atomic — the backend never delivers a
  step to a partial group. vLLM's scheduler may step a subset of a group's requests
  (physics: scheduler decides per-step membership, backend vllm.py:296-309); the
  backend absorbs it by stalling the group (flush + re-add at committed context),
  invisible above.
- LoRA per-request, as today: `lora_name` at open; rebind between runs only.

## 2. Backend implementation — vLLM

Engine thread (backend-owned, spike P1):

- Loops `engine.step()`; drains queued adds/aborts between steps; parks on empty.
- The custom sampler publishes each step's post-processor logits rows to lane futures
  (`call_soon_threadsafe`) and blocks on the feed queue until every scheduled resident
  lane has fed or closed. Same blocking the callback does today, different primitive:
  data crosses the thread boundary, control flow never does. No client code runs on
  the engine thread.
- Feed-or-close releases the barrier; a close mid-step is a defined release (the row's
  slot gets a placeholder, its request aborts on the next drain — today's one-drain-late
  residual, unchanged).

Identity (closed):

- The backend mints monotonic int ids, never reused across stalls (the lane ledger).
- Row → lane at sampler time: `int(req_id.rsplit("-", 1)[0])`. vLLM appends `-{8hex}`
  to every request id unconditionally (`InputProcessor.assign_request_id`, byte-identical
  v0.21.0 ↔ v0.26.0; suffix is hyphen-free, so last-dash split is safe). One line, one
  place, spike-validated (lane_spike.py used the same parse; P1 green).
- Rejected: BatchUpdate/logits-processor identity tracking (redundant — the sampler
  reads `input_batch.req_ids` directly), `extra_args` lookup via `model_runner.requests`
  (equally internal, adds per-row hops), `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION`
  (deprecated flag).

One-shot capture routing: today's `GlobalLogprobsCapture` is a single overwrite-on-call
slot — sufficient when calls are serialized, wrong under concurrent one-shots + resident
lanes. Route capture per request id (the LP sees post-processor logits and the batch's
req_ids; keyed dict, consumed on read).

Version: pin `vllm >= 0.26, < 0.27`. Seam surfaces diffed byte-identical 0.21 → 0.26
(Sampler.forward signature, SamplerOutput fields, logitsprocs argmax split, InputBatch
req_ids, request-id composition, LP init arg). Pin the V2 model runner off in engine
opts — 0.26 auto-enables it for some architectures and it bypasses the
Sampler/LogitsProcessor seam entirely. Keep `async_scheduling=False` for the port
(its rationale is group-table-tied; revisit after landing). The sampler install stays
`model_runner.sampler = ...` via the private chain — vLLM offers no public seam for
logits-out/token-in; it is the one hack the design keeps, in one file.

## 3. Backend implementation — MLX

Same contract, own loop. We own the scheduler, so:

- Subset-stepping is real: the engine may step only the lanes that have fed — no
  leave/rejoin dance for pausing lanes. Current `advance` requires rectangular
  same-delta extension (mlx.py); the lane loop assembles the step batch from fed
  lanes only.
- `burst_active` mutual exclusion dies with the mode: one-shots run between decode
  steps in the same loop.
- `_Ledger` and `_GroupTable` implement the same EngineControl-consumption contract
  twice today; the lane server collapses them into one ledger with two engine bindings.
- Thread affinity (Metal) and dtype constraints (no float64) unchanged.

## 4. Control reshaping

- `Controller.run` = `gather(run_group(g) for g)`. One loop shape per group:
  `round_start(g) → rows draw one step → critic settle(g) → ESS/resample(g) → flush(g)`.
  All the math is already per-group (`_maybe_resample`, `apply_critic_boundary`,
  `_round_start`, `serve_lanes`, `lane_logp`); the global while-loop is the one global
  structure and it dissolves.
- **Dead**: `sync_boundary`, `defers_critic`, `critic_deferred`, `burst_blocker`'s
  boundary half, `StepLoop`/`BurstLoop` as drivers, the executor hop, `_on_main`,
  `_drain_lock`, the burst-end settling, `view_prefixes`/ContextVar snapshotting for
  the worker thread.
- Row coroutines keep their shape: draw → bank → group barrier → repeat. Inside a run,
  `PromptedLLM.logw_next` routes to the row's lane (per-task binding, as the seam does
  today); `accelerate="off"` opens no lanes — every call is a one-shot, main's behavior.
  PromptedLLM processes its own pulled logits (temperature, EOS fold) — the
  `_maybe_temper`/`_process_logw_next_batch` reach-in from burst.py dies with the push
  delivery.
- **Pick rendezvous survives** (measured 1.64×): rows parked at the picker draw in one
  op per vocabulary, threefry-keyed (batch-composition-independent by construction).
  Trigger: every row holding open lanes has parked. Companion gather rides it:
  `draw_reweighted(proposal_logws, target_logws)` batches the target lookup at the
  picked indices, killing the per-row `.item()` sync in DirectTokenSampler
  (`logw = target[token] - logp`; the proposal lookup is algebraically the returned
  `logp`, bit-identical).
- **Device discipline** (the `__getitem__` rule): `LazyWeights.__getitem__` ends in
  `.item()` — a blocking device→host sync per bracket read. That is a cold-path
  convenience, banned from the step path. Hot-path reads are batched gathers crossing
  to host once per step per direction (tokens in, picked scalars out); the picker's
  three `.tolist()` transfers collapse to one; lane banking gathers its per-lane
  increments in the same crossing. A step that round-trips per row is a bug.
- **One seam variable.** The row's per-task binding (row → its lanes) is the only
  ContextVar; boundary lane-sum serving rides the same binding instead of a second var.
- Grain is only "decode steps per round" — **this is the step-lock fix, the motivating
  deliverable**: unit rows close lanes at the unit boundary, the group settles, lanes
  reopen; no group ever waits on another group's unit cadence. Token-grain lanes stay
  open across boundaries; a resample crossing closes + reopens the crossed group's
  lanes (today's `_flush`, per group). No other group notices either event.
- Critic: an engine-lane critic serves from banked lane sums (unchanged math, unchanged
  EOS-increment handling). A critic needing a real forward scores by `score_prompt`
  one-shots at its group's boundary — no drain, no blocker, no B=1 special case.
  Per-step non-lane LM leaves stay blocked: a per-row scoring round-trip per token
  would serialize the decode loop (latency bound, not a mode).
- Record: per group. `SMCRecord.step_num` is a single counter with no notion of
  interleaved cadences; each group records its own step stream. Viz rebuilds per group.
- RNG: draw keys stay `(row, ordinal)` — cadence-independent. gate-1's byte pin is to
  draw *order*, an implementation convention; the reference regenerates against the new
  loop's order.

## 5. Acceptance

- **Fake-lane choreography tests** (new, local, no engine): a scripted lane server
  exercises the group loops — barrier arithmetic, membership after resample, lane
  lifecycle, EOS/termination order, K-group atomicity under injected stalls.
- **MLX gate, local**: end-to-end no-bias on this machine before any remote run.
- **vLLM campaign, Mila**: paired no-bias vs references generated by the new shape's
  own `accelerate="off"` path, threefry-keyed pairing, per-group log_ml exactness.
  One campaign at the end, not a per-step crutch.
- Perf re-earned and re-measured: pick batching, forward-overlap window, burst
  wall-clock vs current shape (`benchmarks/bench.py`, off/require/raw matrix), plus
  the batch-cadence scenario that motivated this: B groups × ragged unit lengths.

## 6. Sequencing

1. Upgrade tranche lands (other session): transformers/peft/mlx-lm, vllm pinned 0.21,
   green. Lane branch starts from it.
2. First lane-branch commit: vllm → 0.26 + V2-runner pin; rebuild gpuvenv; re-fire
   `lane_spike.py` as seam smoke (probes are version-portable).
3. Lane API + fake-lane tests, pure local.
4. MLX lane server + control reshaping, validated by the local MLX gate.
5. vLLM lane server (port of the settled contract), one Mila campaign.
6. Old shape reachable in git only — no in-tree fallback, no legacy flag.

Spike record (Mila job 10323351, L40S, Qwen2.5-0.5B): P1 external feed PASS,
P2 park-on-empty PASS, P3 one-shot mid-decode PASS, P4 prefix-reuse instrument-saturated
(610-tok prefill ≈ one decode step at 0.5B; re-measure at 8B if the number is wanted).
