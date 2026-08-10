# Engine serving — design

How SMC gets its next-token distributions fast, and why none of it is visible above
the potential layer.

## The shape

Control is llamppl-shaped SMC: one coroutine per group, `asyncio.gather` over the
live rows, per-group ESS/resample. A row's step awaits potentials; a `PromptedLLM`
awaits `next_token_logprobs(prompt_ids + context_ids)`. That call is the ONLY
crossing, and it is content-keyed: the backend sees byte contexts, never particles.

Everything that makes it fast lives under that call, inside genlm-backend:

- **The window.** Concurrent asks meet in one batch. The window's first caller holds
  it open until a full event-loop pass adds no new ask (a whole `gather` lands
  together), plus an optional `timeout` linger for late callers; then it executes
  the batch. Window state never outlives the loop its callers are on.
- **Residency by continuation.** A backend keeps `(context, lora) -> engine request`.
  An ask whose context is a resident's plus one token extends that resident (the
  diff IS the token to append); anything else prefills a new one. Resample clones
  ride one resident until their draws diverge; divergence simply misses the map.
- **Cadence by omission.** A resident whose next token has not arrived is skipped by
  vLLM's own scheduler (`num_new_tokens == 0`), resident and free. That is the whole
  pause mechanism: ragged units, per-group cadences and B-group independence cost
  nothing and need no protocol. The engine never waits on control; control only ever
  awaits rows.
- **Eviction.** Idle residents are CPU-free but pin KV blocks, so the scheduler reaps
  the oldest idle ones under block-pool pressure (and a recency cap). Freed blocks
  keep their prefix-cache hashes, so a reaped path that returns prefills warm.

## vLLM

Three pieces, all on sanctioned seams (`genlm/backend/llm/vllm.py`):

- **Capture shim** in the model runner's sampler slot, installed through MRv2's
  `ModelState.custom_sampler` hook. It captures each step's full-vocabulary
  `log_softmax` row (device-resident) and reports `num_sampled = 0`, so the engine
  appends nothing and checks no stop condition. vLLM's own sampler is unused: drawing
  is the algorithm's, never the engine's.
- **`GenlmScheduler`** (`scheduler_cls`) owns request life and death directly —
  births, token appends and reaps arrive through a thread-safe inbox drained at the
  top of `schedule()`. It zeroes `num_output_placeholders` for our requests, since
  their tokens materialize from appends rather than from sampling; without that,
  async scheduling's run-ahead accounting would corrupt.
- **The crank.** The window's executor turns `engine.step()` on a worker thread until
  every asked row is captured, single-flight: later windows reconcile their contexts
  and ride the running crank's frames. There is no daemon and no cadence of our own —
  the in-process engine has none either.

## MLX

Same window; the batch is the frame. `_SlotPool.logits([contexts])` discovers KV reuse
by content, so a batch that walks forward from the previous one continues live KV
instead of reprefilling. No scheduler, no request objects.

## Invariants

- Draws are control-side and keyed (`draw_key(row, ordinal)` → threefry), so a batch's
  composition never changes a draw.
- `prefix`/`complete` are served from a content-keyed running-sum cache in
  `PromptedLLM`: `cumsum(c) = cumsum(c[:-1]) + row(c[:-1])[c[-1]]`, O(1) amortized as
  contexts grow. A per-step critic's reads extend its own resident by content match —
  critics need no special handling anywhere.
- Engine-served rows differ from fresh-prefill rows only by the warm-KV residual
  (~5e-4 total variation on A100/SmolLM). Gates assert statistical unbiasedness, never
  per-row equality.
