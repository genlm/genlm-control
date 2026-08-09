# Lane server

The backend is a persistent async request server; SMC is client code on the main event
loop. Two request lifetimes — one-shot scores and resident lanes — and one invariant:
**clients express per-item awaits; batches are formed by whoever owns the bottleneck**
(engine scheduler for one-shots, lane residency for decode, the loop-drain collector
for draws, vectorized kernels for symbolic potentials). Nothing control-side runs on
an engine thread; nothing backend-side calls into control.

Parity bar unchanged: exact SMC math, unbiased weights, per-group log_ml. Gates re-aim
at the new shape (references regenerate from its own off-path); they do not constrain it.

## 1. Backend primitives

### Server

One per engine (`AsyncVirtualLM` / MLX equivalent). Owns the engine, the engine thread,
the id ledger, and the capture router. Lifetime = process; there is no burst begin/end.

```python
await server.next_token_logprobs(ids: list[int]) -> Tensor[V]     # main's surface
await server.score_prompt(ids: list[int]) -> Tensor[len(ids)-1]   # logp(ids[t] | ids[:t])
server.open_lane(prompt_ids: list[int], *, lora_name: str | None = None,
                 group: GroupHandle) -> Lane
```

One-shots are stateless awaits, submittable at any time, lanes resident or not
(spike P3). On vLLM they batch emergently through the engine's own scheduler (chunked
prefill mixes them into decode steps); capture is routed per request id — a keyed
dict consumed on read, replacing the single overwrite-on-call capture slot.

### Lane

A resident decode request. State machine, all transitions raising on violation:

```
OPEN ──(engine schedules; sampler publishes row)──> WARM
WARM ──feed(tok)──> OPEN        # context += tok; warm cleared; engine may step
WARM ──close()───> CLOSED       # instead of feeding: engine never forwards again
OPEN ──close()───> CLOSED       # e.g. forced EOS: close without reading
```

```python
await lane.next() -> Tensor[V]   # this step's post-processor float32 logits row,
                                 # device-resident. Idempotent in WARM: re-reads
                                 # return the same row (shared leaves, product reads).
lane.feed(token_id: int)         # raises unless WARM
lane.close()                     # from any state; enqueued to the drain
lane.context -> list[int]        # prompt + fed tokens; reads verify against it
```

- **Context verification**: a read carries the caller's context; mismatch with
  `lane.context` raises (a `Normalized` critic walking proper prefixes fails here,
  by construction, before it can desynchronize anything).
- **Liveness**: feed-or-close promptly. vLLM steps all resident lanes together; one
  withheld feed stalls the batch. A lane pausing past a step (unit boundary, resample
  crossing) closes; reopening is `open_lane(old.context)` — priced by prefix cache.
- **Lanes are owned.** A lane belongs to the task that opened it; the owner's exit or
  exception auto-closes its lanes (async-context/RAII). A crashed row therefore
  releases the feed barrier and surfaces as a group error instead of stalling the
  engine. The barrier is the design's one hang surface; it is a single named wait
  and reports the owing lanes on timeout.
- **`close` is a first-class answer to a step.** The engine needs a feed only to
  compute the step after it; committed tokens live control-side. A row whose drawn
  token ends its participation (unit end, EOS, `terminate_when`, `max_tokens`
  force-EOS) closes instead of feeding. Because close is part of the step barrier,
  the drain retires the request before the next forward: no placeholder token is
  ever forwarded, no wasted step exists. (Today both exist: the abort races the
  drain and lands a step late.)

### Group handle

`K` lanes opened under one handle are atomic: the backend never resolves a step for a
partial group. vLLM's scheduler may schedule a subset of a group's requests in a step
(physics — backend vllm.py:296-309); the backend absorbs this by stalling the group
(flush + re-add at `lane.context`), invisible above. On MLX, we own the scheduler and
the situation is inexpressible.

### Engine thread (vLLM)

The one place that blocks. Loop (spike P1/P2):

```
drain adds/aborts → no live requests? park on submit-event
                  → engine.step() → sampler publishes rows to lane futures (by rid,
                    via call_soon_threadsafe) → blocks until every scheduled resident
                    lane is FED or CLOSED → returns fed tokens (closed rows get an
                    engine-side placeholder that the post-step drain retires unforwarded)
```

Data crosses the thread boundary (futures out, feed queue in); control flow never
does. The sampler install remains `model_runner.sampler = ...` through the private
attribute chain — vLLM offers no public logits-out/token-in seam; this is the one
kept hack, in one file, version-diffed 0.21 ↔ 0.26 byte-identical.

### Identity

The ledger mints monotonic ints, never reused across stalls or reopens. Row → rid at
sampler time: `int(req_id.rsplit("-", 1)[0])` — vLLM suffixes every request id with
hyphen-free `-{8hex}` (`InputProcessor.assign_request_id`, identical 0.21 ↔ 0.26);
spike-validated. Rejected alternatives: BatchUpdate identity tracking (redundant with
reading `input_batch.req_ids`), `extra_args` via `model_runner.requests` (equally
internal, adds per-row hops), the deprecated de-randomization env flag.

### MLX

Same contract, own loop, two simplifications: the step batch is assembled from
FED lanes only (true subset-stepping — pausing lanes are simply absent, no
close/reopen dance required, though the contract permits it), and one-shots run
between decode steps in the same loop, so `burst_active` mutual exclusion has no
referent. `_Ledger`/`_GroupTable` collapse to one ledger with two engine bindings.
Metal thread affinity and no-float64 unchanged.

### Version

Pin `vllm >= 0.26, < 0.27`. All consumed surfaces diffed byte-identical from 0.21
(Sampler.forward signature, SamplerOutput, logitsprocs split, InputBatch identity,
request-id composition, LP init arg). Pin the V2 model runner off (it bypasses the
Sampler seam; 0.26 auto-enables it for some architectures). Keep
`async_scheduling=False` at the port; revisit after landing.

## 2. Control shape

### Actors

Three, all on the main loop:

- **Row coroutine** (one per particle): `while not done: draw → bank → arrive`.
  Draw runs the sampler's real `transition`; leaf reads route through the binding.
  A row that terminates closes its lanes and arrives one last time.
- **Group coroutine** (one per group): awaits its rows' arrivals, then runs the
  boundary — critic settle, ESS/resample, membership re-derivation (a done row can
  inherit a live ancestor: respawn its coroutine, reopen its lanes), lane lifecycle
  for crossed rows (close + reopen at rewritten contexts) — and releases the round.
- **`Controller.run`** = `gather(group(g) for g)` after `start()`. No structure spans
  groups. The math is already per-group (`_maybe_resample`, `apply_critic_boundary`,
  `_round_start`, `serve_lanes`, `lane_logp`); the global while-loop was the only
  global object, and it is deleted, not parameterized.

### The binding (one seam variable)

Per row-task ContextVar holding `{id(leaf): Lane}`, built by the controller from
`group_lanes`' identity dedup (a target and proposal over the same `PromptedLLM`
share a lane). `PromptedLLM.logw_next` resolves itself through it: bound → pull the
lane, verify context, process the row itself (temperature, EOS fold — its own code
path, same as a fresh forward); unbound → one-shot, main's behavior. Boundary
lane-sum serving rides the same binding. `accelerate="off"` = never bind. One SMC
loop; acceleration is whether lanes exist.

Deleted with the push-delivery seam: the `[G, K, vocab]` injection block,
`_row_injection`, the `_maybe_temper`/`_process_logw_next_batch` reach-in, the
`EngineControl.draw` contract, `burst_serve`, `_RowSeat`, both old ContextVars.

### The draw collector

The pick batches emergently, not at a barrier. One engine step resolves all warms in
one loop pass, so the step cohort parks together: the first park schedules a fire via
`loop.call_soon`; parks landing in the same pass join it; the fire stacks per
vocabulary, draws with the threefry picker (keyed `(row, ordinal)` — results are
batch-composition-independent, so a straggler splitting the cohort changes nothing),
and resolves the parked futures. No membership state.

**One host crossing per fire**: picked token ids, their logps and logZs, companion
gathers, and per-lane bank increments at the fed ids leave the device together.
`LazyWeights.__getitem__` ends in `.item()` — cold-path convenience, banned from the
step path; a step that round-trips per row is a bug. `draw_reweighted(proposal_lws,
target_lws) -> (token, logw, logp)` is the importance-sampling seam riding the
collector: `logw = target[token] - logp` (the proposal lookup is algebraically the
returned `logp`; bit-identical), with `target[token]` as a companion gather.
Samplers with their own draw machinery (AWRS's per-instance rejection stream) read
warms and feed without the collector; nothing requires it.

### Grain

Grain is only "decode steps per round" — **the step-lock fix, the motivating
deliverable**: no group ever waits on another group's cadence.

- Token grain: a row feeds every step; the group boundary runs every round; lanes
  stay open across boundaries. A resample crossing closes + reopens the crossed
  group's lanes (today's `_flush`, scoped to that group).
- Unit grain: a row loops subunit draws inside one round, feeding each, and closes
  instead of feeding its unit-final token; the group settles; lanes reopen for the
  next round. Intra-group waiting at the boundary is the resample's data dependency,
  not scheduling.

### Critic

- Engine-lane critic: its leaf is a lane in the binding; per-step increments bank
  into `lane_logp` (EOS increment included when the row terminates — the serve and
  the bank must agree, as today); boundary serving reads banked sums. Math unchanged.
- Forward critic (no single engine leaf): `score_prompt` one-shots at its group's
  boundary. No drain, no engine-idle requirement, no B=1 special case.
- The static blocker shrinks to its per-step half: every LM leaf on a group's
  per-step path must be a lane (a per-row scoring round-trip per token would
  serialize the decode loop — latency bound, not a mode), plus engine homogeneity
  across groups. Boundary reasons and the `terminate_when` reason are dead.

### Record & RNG

Record is per group: `SMCRecord.step_num` is one counter with no notion of
interleaved cadences, so each group records its own stream; viz rebuilds per group.
Draw keys stay `(row, ordinal)` — cadence-independent. gate-1's byte pin is to draw
*order*, an implementation convention; the reference regenerates against the new
loop's order.

## 3. What is deleted

`StepLoop`/`BurstLoop` as drivers; `sync_boundary`, `defers_critic`,
`critic_deferred`; the global round loop; `run_burst` and burst attach/detach;
`EngineControl`; `_Burst`, `_RowSeat`, `burst_serve`, both seam ContextVars, the
injection block; the executor hop, `_on_main`, `_drain_lock`, `on_burst_end`,
`view_prefixes` snapshots; the placeholder-token commit and the
one-wasted-forward-per-unit-row residual; `burst_active` (MLX); the boundary and
`terminate_when` blocker reasons. The old shape stays reachable in git only — no
in-tree fallback, no legacy flag.

## 4. Acceptance

- **Fake-lane tests** (new, local, no engine): a scripted `Server` drives the group
  loops — arrival accounting, membership after resample, lane lifecycle per grain,
  close-instead-of-feed at every terminal (EOS, `terminate_when`, force-EOS, unit
  end), K-atomicity under injected stalls, context-verification raises.
- **MLX gate, local**: end-to-end no-bias on this machine before any remote run.
- **vLLM campaign, Mila**: paired no-bias against references generated by the new
  shape's own `accelerate="off"` path; threefry-keyed pairing; per-group log_ml.
  One campaign at the end, not a per-step crutch.
- **Perf re-measured** (`benchmarks/bench.py`, off/require/raw): pick batching,
  forward-overlap, burst wall-clock vs current shape, and the motivating scenario —
  B groups × ragged unit lengths.

## 5. Sequencing

1. Upgrade tranche lands (other session): transformers/peft/mlx-lm, vllm still 0.21.
   Lane branch starts from that tree.
2. First lane commit: vllm → 0.26 + V2-runner pin; rebuild gpuvenv; re-fire
   `lane_spike.py` as seam smoke (probes are version-portable).
3. Lane API + fake-lane tests, pure local.
4. MLX lane server + control reshaping, validated by the local MLX gate.
5. vLLM lane server (port of the settled contract), one Mila campaign.

Spike record (Mila job 10323351, L40S, Qwen2.5-0.5B): P1 external token feed PASS,
P2 park-on-empty PASS, P3 one-shot mid-decode PASS, P4 prefix-reuse
instrument-saturated (610-token prefill ≈ one decode step at 0.5B; re-measure at 8B
if the number is wanted).
