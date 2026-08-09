# MLX burst — design

Against `shepard/mlx-burst` @ `077f3bb` (burst-rows surface; the merge that produced it
touched only the potential layer, so `burst.py` / `burst_seam.py` /
`potential/built_in/llm.py` are byte-identical to `df90cee`) and backend
`shepard/speedups` @ `f150d5e`.

All dev/test is **local** (Apple silicon). No box, no vLLM.

## The delta

Control is engine-agnostic. `_is_burst_lm(p) = isinstance(p, PromptedLLM) and
p.model.supports_burst` (`potential/built_in/llm.py:34`) is the entire gate.

**Backend-only. Zero control changes.** On `AsyncMlxLM`:

- `supports_burst = True`
- `burst_active` set for the duration of `run_burst`
- `run_burst(control, max_steps)`
- `_SlotPool` gains two public verbs (below)

Everything the burst reads off the LM leaf — `token_maps.eos_idxs`, `prompt_ids`,
`_process_logw_next_batch`, `_maybe_temper`, `make_lazy_weights`, `lora_name` — is on
the control-side `PromptedLLM`, which already wraps MLX for the slow lane.

## Why MLX is the easy engine

vLLM needed `ControlSampler` + `_GroupTable` because **the engine owns the decode loop**:
the only interpose point is the `Sampler` vLLM calls inside `step()`, requests get
partially scheduled, and row→request identity has to be recovered from
`input_batch.req_ids`.

MLX has no engine and no scheduler. `model(tokens, cache=...)` is a plain call. So:

- `run_burst` **is** the loop; `control.draw` is a call in its body.
- Every ordered handle forwards every step. No stalls, no partial groups, no
  re-add-at-committed-context, no request ids. The ledger is ~30 lines, not 60.
- No logits processors, no penalties, no `sample()` to bypass.

## Loop

```python
supports_burst = True

def run_burst(self, control, max_steps):
    ledger = _Ledger()
    self.burst_active = True
    try:
        with wired_limit(self.mlx_lm_model, [self.generation_stream]):
            ledger.drain(control)                       # seed
            for _ in range(max_steps):
                if not ledger.order:
                    break
                logits = self._burst_step(ledger)
                ledger.commit(control.draw(logits, ledger.order).tolist())
                ledger.drain(control)
            control.on_burst_end()
            ledger.drain(control)
    finally:
        self.burst_active = False
```

`max_steps` is a **global** decode-step cap here, where vLLM's is per-request
(`SamplingParams.max_tokens`, refreshed for a row re-added after a crossing). Tighter,
and it costs nothing: length termination is control-side (the forced EOS at
`max_tokens`); the engine cap is only a runaway guard.

The ledger:

```python
class _Ledger:
    """run_burst's group ledger: handle -> per-lane committed prompt, in batch order."""

    def __init__(self):
        self.order = []     # handles, batch order
        self.prompts = {}   # handle -> [ids per lane]
        self.loras = None   # per lane, snapshotted at the first add
        self.laid = None    # the order the pools' rows currently hold

    def drain(self, control):
        for h in control.drain_aborts():
            if self.prompts.pop(h, None) is not None:
                self.order.remove(h)
        for h, prompts, loras in control.drain_adds():
            if self.loras is None:
                self.loras = list(loras)
            self.prompts[h] = [list(p) for p in prompts]
            self.order.append(h)

    def commit(self, tokens):
        for h, t in zip(self.order, tokens):
            for p in self.prompts[h]:
                p.append(t)
```

## Lanes

A group is K views in lockstep, and `_batch_blocker` guarantees **lane `l` carries one
LoRA name and one temperature for the whole burst** across every group. So a lane is a
batch of G rows under one adapter — one `_SlotPool` each, one forward each.

```python
def _burst_step(self, ledger):
    """One decode step across every lane: [G, K, vocab] fp32."""
    sources = ledger.sources(self.burst_slots[0])
    per_lane = []
    for lane, pool in enumerate(self.burst_slots):
        self.adapters.select(ledger.loras[lane])
        prompts = [ledger.prompts[h][lane] for h in ledger.order]
        per_lane.append(
            pool.advance(sources, [p[-1:] for p in prompts])
            if sources is not None
            else pool.logits(prompts)
        )
    ledger.laid = list(ledger.order)
    out = mx.stack(per_lane, axis=1).astype(mx.float32)
    mx.eval(out)
    return torch.from_dlpack(out)
```

K=1 is one forward — the common case pays nothing for the K machinery.

`sources` is computed **once**, off lane 0, and reused for every lane: a crossing
reindexes within a group, and a group's views share a prompt prefix, so the ancestor of
a row is the same row in every lane. That also keeps the pools in lockstep, which is
the invariant the whole thing rests on:

> After every forward, every pool holds exactly `ledger.order`'s prompts for its lane,
> in that order.

`_Adapters.select` should short-circuit on an unchanged name (track `self.current`,
invalidate in `add`); otherwise each lane pays a `tree_unflatten` + `model.update`
every step.

## KV: fork-by-gather **is** the resample

This is the part worth getting right, and the slot pool already does it — it just needs
its two motions named.

`_SlotPool.logits(prompts)` is a **discovery** API: "here are prompts, work out what to
reuse." Right for the slow lane (independent batched queries), wrong for a driver that
knows its own ancestry. But the machinery underneath is exactly what the driver wants:

```python
def logits(self, prompts):                       # discovery — slow lane
    sources, shared = zip(*map(self._source, prompts))
    deltas = [p[n:] for p, n in zip(prompts, shared)]
    if self._continues(sources, deltas):
        return self.advance(list(sources), deltas)
    return self._prefill(prompts)

def advance(self, sources, deltas):              # explicit — burst
    """Re-lay the rows as ``sources`` (repeat to fork, omit to drop), then forward one
    equal-length token block per row."""
    self._gather(sources)
    return self._extend(deltas)

def rows_holding(self, prefixes):
    """The row holding exactly each prefix, ``None`` where none does."""
    index = {tuple(s): i for i, s in enumerate(self.seqs)}
    return [index.get(tuple(p)) for p in prefixes]
```

`+8 lines net`, no second KV mechanism, `logits` keeps its meaning.

`_gather` is `BatchKVCache.filter(mx.array(sources))` — fancy indexing, so a **repeated
source forks that row and an omitted one drops it**. A resample crossing is one gather.
`filter` also re-shifts `left_padding` down, so raggedness doesn't accumulate. Verified
byte-identical to cold prefill in `test_kv_fork_matches_cold_prefill`.

### Where `sources` comes from

```python
def sources(self, pool):
    """Pool row each ordered handle continues, or ``None`` to rediscover."""
    if self.laid == self.order:
        return list(range(len(self.order)))
    if not pool.seqs:
        return None
    rows = pool.rows_holding([self.prompts[h][0][:-1] for h in self.order])
    return None if None in rows else rows
```

- **Steady step** (membership unchanged): identity, O(G). Every row extends by the token
  just committed.
- **Membership changed** (crossing, pop-out, revival): one index build, O(total tokens)
  — ~1–2 ms at G=32, L=1000, and only on steps that actually changed. The dict lookup
  *is* the exact-prefix check, so a wrong hint is impossible by construction.
- **Miss** → `None` → `pool.logits(prompts)` rediscovers, reprefilling if it must.

Why the `[:-1]` lookup is sound: adds only ever come from `_flush`/`_arrive`, which run
in `_release` **after** the token is banked, so a re-added prompt is exactly one token
past the pool. And `_flush` re-adds only `if not p.done` — a live child implies a live
ancestor, so the ancestor's row is still in the pool (aborts and adds resolve together
in the same gather, so the ancestor is never dropped before it is read).

The miss path is the honest fallback for the cases that violate this: a step that
committed more than one item (the control's context outruns the engine's by more than
the drawn token), and the unit-grain seed where rows advanced by ragged unit lengths
between bursts.

Burst pools persist on the instance across bursts (`self.burst_slots`), so a unit-grain
round continues instead of reprefilling when the units happen to be equal-length.
`clear_cache()` drops them alongside `self.slots`.

## Threading and overlap

`BurstLoop.round` already does `loop.run_in_executor(None, lambda:
llm.model.run_burst(...))`, so `run_burst` is on a **worker thread** and the main loop
is free. That is what the whole `_release` tail buys, and MLX gets it for free:

- `draw` computes the warm on the worker thread, then `_on_main(self._step(...))` blocks
  the worker while the main loop delivers/picks.
- `_step` returns at the pick and schedules `_release` (leading `await
  asyncio.sleep(0)`), so the lane bank, the row banking and the round boundary land
  **after** `draw` returns — i.e. during the next MLX forward.
- The worker blocks in `mx.eval`; that does not block the main loop, so the overlap is
  automatic.

The accepted "abort lands one drain late" property carries over verbatim: `ledger.drain`
runs on the worker right after `draw` while `_release` may still be running on the main
loop. Keep it — closing the race means joining the tail before the forward, which is
exactly the overlap `a1b8caa` bought.

### Measured

The assumption worth checking is that this overlaps at all. On CUDA the main-loop tail
is host work against a busy device; on Apple silicon the forward and the tail
(`_bank_lanes` gather + `_batch_draw` logsumexp/gumbel/argmax) are both GPU work on one
unified device, so they could just contend. Decomposed so rendezvous cost is separable
from contention (G=32, V=152k, a 24×4096² fp16 matmul chain standing in for the
forward):

| | ms/step |
|---|---|
| A forward alone on the worker | 5.16 |
| B + rendezvous (`run_coroutine_threadsafe` per step) | 5.13 |
| C + tail on MPS | 6.16 |
| D + tail on CPU | 10.61 |

- **Rendezvous is free** (B−A ≈ 0). The two thread hops per step do not register.
- **The MPS tail is 69% hidden**: 3.30 ms solo, 1.03 ms marginal.
- **A CPU tail is worse on both counts**: 9.96 ms solo (3× the MPS version over a
  `[32, 152k]` block) and only 45% hidden. Moving the tail off the GPU to dodge
  contention loses badly — leave it on MPS.

So the overlap is real, and the fraction hidden scales with the forward:tail ratio.
Sweep that ratio rather than trusting one point (same reasoning as
`bench.py --scenario synth --potential-us`).

Also add the vLLM-style guard (`vllm.py:546`): the slow-lane `next_token_logprobs*`
entry points must refuse while `burst_active`, or a stray leaf silently reprefills into
the pool the burst is driving.

## Verified locally

| Claim | Result |
|---|---|
| `torch.from_dlpack(mlx array)` device | `mps:0`, zero-copy |
| bf16 survives dlpack | yes, `torch.bfloat16` |
| threefry int64 ops on MPS | supported; **bit-identical to CPU** (same indices) |
| `torch.isin`, advanced indexing, `logsumexp` on MPS | all work (`_process_logw_next_batch` needs them) |
| `to_numpy` on MPS | `w.cpu().numpy()` — fine |
| `BatchKVCache` | has `filter` / `extend` / `trim`; `filter` forks on repeated index and re-shifts left padding |

threefry being bit-identical on MPS is the important one — it means an MLX gate-2 can be
a **tight paired check on-device**, not Monte Carlo, exactly as the CUDA one is.

## Numerics

Hand `control.draw` **raw logits as fp32** (`.astype(mx.float32)` in MLX before dlpack).
That sidesteps `_to_torch`'s bf16→fp16 narrowing (which exists only for numpy callers)
and makes `_maybe_temper` / `.float()` no-ops.

The slow lane normalizes in MLX (`logits - logsumexp`) and the burst normalizes in torch
(`_process_logw_next_batch`'s `log_softmax`), so the two differ in the last bits. Same
asymmetry the vLLM arm already has, and gate-2 tolerates it as part of the warm-KV
residual — but it means the MLX gate must be a paired no-bias check, not a byte pin.

## Measured: burst vs StepLoop

`benchmarks/bench.py --backend mlx` (gpt2, max_tokens=64, 3 trials, median). `raw` is
vLLM-only — `raw_ceiling` drives `model.llm_engine.generate` — so the MLX matrix is
`off` vs `require`.

| N | ess | off (StepLoop) | require (burst) | speedup |
|---|---|---|---|---|
| 4 | 0.0 | 0.820 s | 0.516 s | 1.6x |
| 16 | 0.0 | 3.433 s | 0.504 s | 6.8x |
| 32 | 0.0 | 7.067 s | 0.725 s | 9.7x |
| 16 | 0.5 | 3.246 s | 0.543 s | 6.0x |

StepLoop is linear in N (~0.22 s per particle); the burst is nearly flat (0.50 s at N=4,
0.73 s at N=32). StepLoop reprefills every particle every step; the burst puts all N rows
through one forward, so the win grows with the population and resampling costs almost
nothing (the fork is one `filter`).

`bench.py` defaults to `gumbel_max`, so an `off`/`require` pair is NOT RNG-matched and
their `log_ml` diverge by Monte Carlo noise (visibly at N=32). Bias is the gate's job, not
the bench's.

## Gate

Because it is all local, this is the first arm where gate-2 runs without a box.

- `tests/sampler/test_engine_mlx.py`, importing `gate2_cases.py` / `_harness.py` so it
  cannot drift from the CUDA gate.
- A 4-bit `mlx-community` 0.5B is enough; the check is burst-vs-StepLoop on the *same*
  MLX model, so model quality is irrelevant.
- Paired at the same seed with the threefry picker (autouse fixture, as gate-2 does),
  plus a `match_floor` — the sem-scaled no-bias check loosens silently as pairing
  degrades, the floor fails loudly.

Result: **9/9 in 6m29s**, all nine shared cases, locally.

| case | contexts exact | log_ml mean diff |
|---|---|---|
| unconstrained | 64/64 | +0.0002 |
| constrained-boolfsa[a-z ]+ | 32/32 | +0.0003 |
| boolfsa[aeiou ]+ | 192/192 | -0.0000 |
| terminal-critic | 96/96 | +0.0002 |
| terminal-critic-resample | 96/96 | +0.0002 |
| twist-critic | 130/192 | -0.6570 (sem 0.4699) |
| multitoken-boolfsa[a-z ]+ | 96/96 | +0.0002 |
| awrs[a-z ]+ | 3/96 | -0.1334 |
| set[a-z ]+ | 0/32 | -0.0736 |

`awrs` and `set` are the two cases that carry no `match_floor` because they are not
RNG-matched to their reference at all (measured 1/96 and 0/32 on vLLM) — MLX reproduces
that, so their assertions are unpaired Monte Carlo, as designed.

`twist-critic` is the one to keep an eye on: it passes, and 130/192 is far above the
vLLM floor of 34, but a sem of 0.47 makes the no-bias check loose there.

## Known weaknesses (stated, not fixed)

- `_continues` is all-or-nothing: one non-matching prompt or a non-uniform delta
  reprefills the whole batch. Pre-existing; the burst's `sources` path routes around it
  for every step it can, and falls into it otherwise.
- `self.generation_stream` is created but never entered (`wired_limit` takes it, no
  `with mx.stream(...)`). Pre-existing oddity; unrelated to the burst but worth a look
  while in here.
- K forwards per step at K>1. Unavoidable — MLX attaches adapters to the model, so two
  lanes under different adapters cannot share one forward.

## Built

All five steps landed. Three things the design did not predict:

**The warm must land on the host, not the device.** The design said to hand `control.draw`
device-resident fp32 rows and let the pick run on MPS. That deadlocks the moment a
context-only potential is in the product: `Product._compose` does
`torch.as_tensor(w2, device=w1.device)` with `w2` a **float64 numpy** row, and Metal has
no float64 — so every row coroutine dies with `TypeError`, never sets `settled`, and
`_Burst._step`'s gather waits forever. The crash presents as a hang because `_run_row`
has no handler and a dead task looks identical to a parked one.

`_compose` preserving `w2`'s dtype is deliberate ("dtype preserved -> same promotion"),
and the slow lane already serves host rows (`PromptedLLM.logw_next` ends `.float().cpu()`).
So `_burst_step` returns `.cpu()`: it matches the slow lane exactly, keeps both paths
promoting through float64, and costs one `[G, K, V]` copy per step over unified memory.
The measured 69%-hidden MPS tail does not apply — that work is now host-side.

Worth noting for the CUDA arm: it composes float64 *on the GPU*, which is 1/32 rate on
consumer silicon. Not touched here (it would move gate-2's numerics), but it is a real
cost hiding in `_compose`.

**MLX streams are thread-affine.** `AsyncMlxLM.generation_stream` was built in `__init__`
on the main thread, so `wired_limit`'s exit `mx.synchronize(s)` died on the burst worker
with `There is no Stream(gpu, 3) in current thread`. Measured: the default stream is
thread-portable, a stream created on another thread is not. The attribute was also a
no-op — it synchronized a stream nothing ever ran on, since the forwards use the default
stream and no `with mx.stream(...)` was ever entered. Deleted in favour of a `_wired(model)`
helper that resolves `mx.default_stream(mx.default_device())` per call.

**`advance` bypassed the `_continues` guard.** A reorder needs row-sliceable caches, and a
recurrent state (mamba) has none — the slow lane checks, the burst path did not, so a
resample on a mamba-class model would have raised `AttributeError`. `advance` now rebuilds
by prefilling what the rows would have held (`self.seqs[s] + d`), which it can compute
itself; the predicate is `_relayable`, shared with `_continues`.

## Order of work

1. ~~Smoke-test the threading assumption.~~ Overlap holds — see *Measured*.
2. ~~`_SlotPool`: `advance` / `rows_holding`, `logits` rewritten over them.~~
3. ~~`_Adapters.select` short-circuit.~~ Tracks the installed weight set by identity, so a
   rebind invalidates itself and needs no hook in `add`/`remove`.
4. ~~`run_burst` + `_Ledger` + `burst_active` guards.~~
5. ~~`test_engine_mlx.py` off the shared cases.~~ Reaching it meant moving the
   engine-agnostic run+compare loop (`make_controller`, `run_burst`, `run_steploop`,
   `compare_runs`, `assert_case_unbiased`, `log`) out of `test_engine_native.py`, which
   skips at module level without CUDA and so cannot be imported on a mac. Both gates now
   drive the same loop and differ only in `resolve_ref`: vLLM reads cached snapshots,
   MLX always runs a live StepLoop on the same model. 856 -> 688 lines there,
   104 -> 327 in `_harness.py`.

Not covered: the CUDA gate's K=2 multi-view and B>1 batched tests are written inline
rather than as `CASES` entries, so the MLX gate does not reach them. The K>1 lane logic
(one source list shared across lanes) is therefore implemented but unexercised.

## Where this work lives

Both repos are worktreed off their `shepard/speedups` tips onto `shepard/mlx-burst`:

- control `.claude/worktrees/mlx-burst` @ `077f3bb`
- backend `../genlm-backend-mlx-burst` @ `f150d5e`

The control worktree's `.venv` has both installed editable, so it resolves to the
worktree code, not the main checkout. Baseline in this tree: gate-1 22/22, backend
`test_mlx.py` + `test_mlx_lora.py` 41/41.
