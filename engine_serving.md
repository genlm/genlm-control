# Engine serving — design

How SMC gets its next-token distributions fast, and why none of it is visible above
the potential layer.

## The shape

Control is main-shaped SMC: `smc_standard` gathers `step()` over live particles; a
step awaits potentials; a `PromptedLLM` awaits `next_token_logprobs(prompt_ids +
context_ids)`. That call is the ONLY crossing, and it is content-keyed: the backend
sees byte contexts, never particles. Concurrent SMCs are plain `asyncio.gather` —
their asks meet below this seam like any others.

Everything that makes it fast lives under that call, inside genlm-backend:

- **The window.** Concurrent asks meet in one cohort. The window's first caller holds
  it open until a full event-loop pass adds no new ask (a whole `gather` lands
  together), then hands the cohort to the crank. Window state never outlives the
  loop its callers are on.
- **The crank.** One backend-owned thread owns all engine interaction. The in-process
  engine runs `schedule()` on its caller's thread, so residency, eviction and delivery
  bookkeeping are single-threaded by construction: a work queue is the only crossing
  in, `call_soon_threadsafe` per future the only crossing out.
- **Residency by continuation.** The crank keeps `(context, lora) -> engine request`.
  An ask whose context is a resident's plus one token extends that resident (the
  diff IS the token to append); anything else births a stock request through vLLM's
  front door. Resample clones ride one resident until their draws diverge; divergence
  simply misses the map.
- **Cadence by omission.** A resident whose next token has not arrived is skipped by
  vLLM's own scheduler (`num_new_tokens == 0`), resident and free. That is the whole
  pause mechanism: ragged units and concurrent-SMC independence cost nothing and need
  no protocol. The engine never waits on control; control only ever awaits rows.
- **Eviction.** Idle residents are CPU-free but pin KV blocks, so the crank reaps the
  oldest idle ones under block-pool pressure (and a recency cap), via stock
  `finish_requests`. Freed blocks keep their prefix-cache hashes, so a reaped path
  that returns prefills warm.

## vLLM

What remains custom, and why it can't ride a front door (`genlm/backend/llm/vllm.py`):

- **Capture shim** in the model runner's sampler slot, installed through MRv2's
  `ModelState.custom_sampler` hook and asserted installed at `from_name`. It captures
  each step's full-vocabulary `log_softmax` row (device-resident) and reports
  `num_sampled = 0`, so the engine appends nothing and checks no stop condition.
  vLLM's own sampler is unused: drawing is the algorithm's, never the engine's. Rows
  have no front door — vLLM's logprobs machinery materializes rows in Python at
  ruinous cost.
- **`BackendScheduler`** (`scheduler_cls`) keeps only what caller-fed tokens force. Our
  tokens come from appends rather than from sampling, which breaks two runner
  assumptions, and both must be neutralized: it ships each appended token on the
  `SchedulerOutput` so a worker-side wrap writes it into the runner's last-sampled
  buffer — the buffer the decode step reads its input token from, which the capture
  sampler leaves at zero (miss this and every decode forwards token 0 while all
  scheduler-side bookkeeping looks healthy); and it zeroes `num_output_placeholders`
  for our requests (async scheduling's run-ahead accounting). Token feed has no O(1)
  front door either: vLLM's streaming sessions accept caller tokens but rebuild the
  runner request per chunk — O(context) per token, chunk-grain machinery.

## MLX

Same window; the batch is the frame. `_SlotPool.logits([contexts])` discovers KV reuse
by content, so a batch that walks forward from the previous one continues live KV
instead of reprefilling. No scheduler, no request objects, no crank — the forward runs
in the flush.

## Invariants

- Rows stay on the backend's device end to end; control's draw window performs the one
  designed host crossing (the pick readback). Draws are unkeyed — batched rows draw
  independent noise, and a cohort's composition carries no RNG meaning.
- `prefix`/`complete` score by the dumb prefix walk, deliberately quadratic; the real
  fix is a backend scoring primitive, parked. No control-side scoring caches.
- Engine-served rows differ from fresh-prefill rows only by the warm-KV residual
  (~5e-4 total variation on A100/SmolLM). The serving contract is asserted at row
  grain by `tests/probe_vllm_server.py`, never laundered through SMC statistics.
