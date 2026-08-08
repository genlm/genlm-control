"""Engine-accelerated SMC: ``_Burst`` seam, ``BurstLoop`` driver, ``burst_blocker`` gate.
Resample/ESS/log_ml stay Controller-owned, never in the backend."""

import asyncio
import enum
import threading
from dataclasses import dataclass

import numpy as np
import torch

from genlm.control.constant import EndOfSequence
from genlm.control.potential.built_in.llm import find_engine_lm, lm_leaves
from genlm.control.burst_seam import burst_row
from genlm.control.util import draw_key, flatten_units, picker_indices, to_numpy


class NotAcceleratable(Exception):
    """Engine acceleration required but the config can't be driven by the burst."""


class BlockReason(enum.Enum):
    """Matchable category for why a config can't run the engine burst."""

    NO_ENGINE_LEAF = "engine_leaf"
    FORWARD_NOT_INJECTABLE = "forward"
    BATCH_HETEROGENEOUS = "batch"
    UNBANKABLE_STEP = "unbankable"
    LANE_OFF_ENGINE = "lane_engine"


@dataclass(frozen=True)
class BurstBlock:
    """``burst_blocker``'s verdict: a :class:`BlockReason` + human ``detail``."""

    reason: BlockReason
    detail: str


@dataclass
class BurstDraw:
    """One live row's result for one decode step.

    token: ``Token``/``EOS`` drawn this step. step: ``(to_append, logw, logp)`` or
    ``None`` (mid-step).
    """

    token: object
    step: tuple | None


class _RowChannel:
    """One parked row's handoff with the burst. The burst delivers a decode step's warm;
    the row's ``transition`` consumes it, draws, and parks at its next logits read. The
    context it parks with ends in the token it just drew."""

    def __init__(self):
        self._warm = None
        self._served_len = None  # None until this step's warm has been read
        self.context = None
        self._delivered = asyncio.Event()
        # Set while the row is parked or returned -- the burst waits on it per step.
        self.settled = asyncio.Event()
        # The row's published draw while it waits at the picker, else ``None``.
        self.pending = None
        self._drawn = asyncio.Event()
        self._result = None

    async def draw(self, lazyweights, slot, step):
        """Row side: publish this draw and wait for the burst to make it with the rest of
        the population. ``(slot, step)`` travel with it -- the key lives in the row's own
        task context, which the burst cannot read."""
        self.pending = (lazyweights, slot, step)
        self._drawn.clear()
        self.settled.set()
        await self._drawn.wait()
        return self._result

    def resolve(self, result):
        """Burst side: hand back one row's ``(token, logZ, logp)`` and let it run on."""
        self.pending = None
        self._result = result
        self.settled.clear()
        self._drawn.set()

    async def next_warm(self, context):
        """Row side: this read's warm, parking when the step's warm is spent. Reads at one
        context length are the same decode step (target and proposal share it), so only a
        read past a draw parks."""
        if self._warm is None or self._served_len not in (None, len(context)):
            self.context = context
            self._warm = None
            self._delivered.clear()
            self.settled.set()
            await self._delivered.wait()
        self._served_len = len(context)
        return self._warm

    def deliver(self, warm):
        """Burst side: hand this row the step's warm and let it run."""
        self.settled.clear()
        self._warm = warm
        self._served_len = None
        self._delivered.set()


class _Burst:
    """Per-burst engine state for one ``run_burst``. ``draw``/``drain_*``/``context_ids``/
    ``on_burst_end`` are the engine seam (the backend drives the control through them).
    One engine group per particle: K view requests the backend keeps in lockstep."""

    def __init__(self, d, live):
        self.d = d
        # Adapter names snapshotted at burst start: a lora_name rebind mid-run must
        # not split this burst across adapters.
        self.view_loras = [v.lora_name for v in d.views]
        # The drain streams are written by ``_settle`` on the main loop while the engine
        # forwards, and read from the engine thread's ``_drain``.
        self._drain_lock = threading.Lock()
        self.abort_handles = set()
        self.add_handles = []
        self.handle_row = {}  # engine group handle -> particle row
        self.row_handle = {}  # particle row -> engine group handle
        self.next_handle = 0
        for p in live:
            self._add_group(p)
        self.channels = {}  # particle row -> _RowChannel (parked-row lane)
        self.tasks = {}  # particle row -> its in-flight transition Task
        # The previous step's tail, in flight against the current forward.
        self._pending_settle = None

    def context_ids(self, p, view_idx):
        """Engine prompt for one (particle, view) substream: per-view prefix + the particle's
        drawn token ids (EOS dropped; drawn suffix shared across views)."""
        g = p.group
        ids = list(self.d.view_prefixes[g][view_idx])

        def _emit(item):
            if isinstance(item, EndOfSequence):
                return
            if isinstance(item, list):
                for sub in item:
                    _emit(sub)
            else:
                ids.append(item.token_id)

        for item in p.context:
            _emit(item)
        return ids

    def _add_group(self, p):
        """Mint+register the group handle for one particle, queue its K-view engine
        add. Sole add path (initial population + mid-burst re-add)."""
        h = self.next_handle
        self.next_handle += 1
        self.handle_row[h] = p.row
        self.row_handle[p.row] = h
        entry = (
            h,
            [self.context_ids(p, vi) for vi in range(len(self.d.views))],
            self.view_loras,
        )
        with self._drain_lock:
            self.add_handles.append(entry)

    def _row_injection(self, warm_batch, i, p):
        """Row ``i``'s ``{leaf: [V+1]}`` slice of the batched warm, keyed by the row's OWN
        group's lanes: groups carry their own leaves, so another group's would miss the
        override and forward inside the engine's step."""
        return {
            rv: rv.make_lazy_weights(warm_batch[gv].weights[i])
            for rv, gv in zip(self.d.controller.group_lanes[p.group], self.d.views)
            if rv is not None
        }

    def _spawn_row(self, p, row):
        """Start this row's whole ``Controller.draw_step`` as a parked task -- the same
        step shape the per-token driver runs, so the burst inherits the draw key, the
        ``max_tokens`` forced EOS, and ``terminate_when`` rather than restaging them.
        Its logits reads park on this channel like any other."""
        channel = _RowChannel()

        async def run():
            with burst_row(channel):
                return await self.d.controller.draw_step(p)

        task = asyncio.ensure_future(run())
        task.add_done_callback(lambda _: channel.settled.set())
        self.channels[row], self.tasks[row] = channel, task
        return channel

    def _release_row(self, row):
        """Drop a row's parked task, cancelling it if it is still mid-transition."""
        task = self.tasks.pop(row, None)
        self.channels.pop(row, None)
        if task is not None and not task.done():
            task.cancel()

    async def _rendezvous(self, rows):
        """Serve every row waiting at the picker in one op -- one logsumexp, one keyed draw,
        one gather over the stacked ``[N, V+1]``. Loops because a resumed row may reach the
        picker again before it parks (a sampler may draw more than once per decode step)."""
        while True:
            waiting = [row for row in rows if self.channels[row].pending is not None]
            if not waiting:
                return
            # One batch per vocabulary: groups may carry different constraints, so their
            # rows are different widths over different tokens and cannot stack together.
            by_vocab = {}
            for row in waiting:
                by_vocab.setdefault(
                    id(self.channels[row].pending[0].decode), []
                ).append(row)
            for batch in by_vocab.values():
                self._batch_draw(batch)
            await asyncio.gather(*(self.channels[row].settled.wait() for row in waiting))

    def _batch_draw(self, batch):
        """One vocabulary's parked rows, drawn together: one logsumexp, one keyed pick, one
        gather. Keying makes this byte-identical to the same rows drawn one at a time."""
        pend = [self.channels[row].pending for row in batch]
        W = torch.stack([torch.as_tensor(lw.weights) for lw, _, _ in pend])
        slots = torch.tensor([s for _, s, _ in pend], dtype=torch.int64)
        steps = torch.tensor([k for _, _, k in pend], dtype=torch.int64)
        logZ = torch.logsumexp(W, dim=-1)
        logps = W - logZ[:, None]
        with draw_key(slots, steps):
            idx = picker_indices(logps)
        ar = torch.arange(len(batch), device=W.device)
        picked, zs, ids = logps[ar, idx].tolist(), logZ.tolist(), idx.tolist()
        decode = pend[0][0].decode
        for j, row in enumerate(batch):
            self.channels[row].resolve((decode[ids[j]], zs[j], picked[j]))

    async def _parked_records(self, warm_batch, parts, rows):
        """One decode step through per-row parked ``transition`` tasks: deliver each row's
        warm, serve the population's draws in one op, then wait for each row to park again
        (mid-unit) or return (the SMC step)."""
        for i, (p, row) in enumerate(zip(parts, rows)):
            # Already running unless this is the burst's first step, or the engine got
            # one more step out of a row whose abort has not drained yet.
            channel = self.channels.get(row) or self._spawn_row(p, row)
            channel.deliver(self._row_injection(warm_batch, i, p))
        await asyncio.gather(*(self.channels[row].settled.wait() for row in rows))
        await self._rendezvous(rows)
        records = []
        for row in rows:
            task = self.tasks[row]
            if task.done():
                step = task.result()
                self._release_row(row)
                records.append(BurstDraw(token=flatten_units(step[0])[-1], step=step))
            else:
                records.append(
                    BurstDraw(token=self.channels[row].context[-1], step=None)
                )
        return records

    def _bank_lanes(self, warm_batch, records, rows):
        """Add this step's drawn-token logp to every lane, from the warm row each lane's
        leaf already produced. That running sum IS the leaf's own ``prefix``, and a
        terminating row's EOS increment closes it into its ``complete`` -- so a leaf
        reached inside the burst reads a banked number rather than forwarding."""
        L = self.d.controller.particles.lane_logp
        drawn = [k for k, r in enumerate(records)
                 if not isinstance(r.token, EndOfSequence)]
        eos = [k for k, r in enumerate(records) if isinstance(r.token, EndOfSequence)]
        if not drawn and not eos:
            return
        drawn_rows = np.array([rows[k] for k in drawn], dtype=np.int64)
        eos_rows = np.array([rows[k] for k in eos], dtype=np.int64)
        for lane, view in enumerate(self.d.views):
            W = warm_batch[view].weights  # [N, V+1] device tensor
            if drawn:
                lk = view.lookup
                ks = torch.tensor(drawn, device=W.device)
                idx = torch.tensor(
                    [lk[records[k].token] for k in drawn], device=W.device
                )
                L[lane, drawn_rows] += to_numpy(W[ks, idx])
            if eos:
                L[lane, eos_rows] += to_numpy(W[eos, -1])

    def drain_aborts(self):
        with self._drain_lock:
            handles = self.abort_handles
            self.abort_handles = set()
        return list(handles)

    def drain_adds(self):
        with self._drain_lock:
            adds = self.add_handles
            self.add_handles = []
        return adds

    def draw(self, logits, handles):
        """Engine per-step callback over the complete groups' ``[G, K, vocab]`` logits:
        one token per live group. The engine thread blocks here, so the step owes it
        nothing but the tokens -- everything after the draw is deferred to
        :meth:`_settle` and runs against the next forward. Dropped groups get a
        placeholder token."""
        c = self.d.controller
        # Per view: [G, V+1] warm log-weights (device tensors, no host xfer).
        processed = [
            view._process_logw_next_batch(view._maybe_temper(logits[:, vi].float()))
            for vi, view in enumerate(self.d.views)
        ]

        async def _step():
            await self._join_settle()
            live_k = [k for k, h in enumerate(handles) if h in self.handle_row]
            rows = [self.handle_row[handles[k]] for k in live_k]
            parts = [c.particles[row] for row in rows]
            if c.twist_with_critic:
                c.particles.untwist(rows)
            out = [0] * len(handles)
            if rows:
                # One batched warm per view ([N, V+1], rows-order).
                sel = torch.tensor(live_k, dtype=torch.int64, device=logits.device)
                warm_batch = {
                    view: view.make_lazy_weights(processed[vi][sel])
                    for vi, view in enumerate(self.d.views)
                }
                records = await self._parked_records(warm_batch, parts, rows)
                self._bank_lanes(warm_batch, records, rows)
                for k, rec in zip(live_k, records):
                    tok = rec.token
                    out[k] = (
                        self.d.eos_id if isinstance(tok, EndOfSequence) else tok.token_id
                    )
            else:  # no live groups this step (all drained/terminated)
                records = []
            if self.d.sync_boundary:
                # Unit grain settles in the block: a surviving row's pop-out abort has
                # to reach the engine's drain for THIS step, and there is no round to
                # close mid-burst -- the controller runs it between bursts.
                await self._bank_pop(parts, records)
                self._flag_after_bank(parts, rows, records)
            else:
                self._pending_settle = asyncio.ensure_future(
                    self._settle(parts, rows, records)
                )
            return out

        out = self._on_main(_step())
        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def on_burst_end(self):
        """Engine lifecycle hook: decode loop drained, so join the last step's tail --
        no next ``draw`` will. Rows still parked at the drain (mid-unit, or started
        ahead for a step the engine never ran) are cancelled with their partial work --
        on the main loop, since Tasks are not touchable from this worker thread."""

        async def _end():
            await self._join_settle()
            for row in list(self.tasks):
                self._release_row(row)

        self._on_main(_end())

    async def _settle(self, parts, rows, records):
        """One step's whole tail, off the engine's critical path: bank, evict, close the
        round, start every live row's next transition. Starting them here is what
        overlaps them -- a row walks its context-only potentials against the next
        forward and is parked at its logits read before that warm lands."""
        await self._bank_pop(parts, records)  # score/extend/critic, sets p.done
        self._flag_after_bank(parts, rows, records)
        # Token grain closes its round mid-burst: ESS is tested every step, but only a
        # step that advanced a row is recorded. Returns the rows a crossing invalidated.
        self._flush(
            self.d.controller.round_boundary(
                record=any(r.step is not None for r in records)
            )
        )
        for row in list(self.row_handle):  # survivors + whatever a crossing re-added
            if row not in self.tasks:
                self._spawn_row(self.d.controller.particles[row], row)

    async def _join_settle(self):
        """Await the previous step's tail, so this step draws over a banked, resampled
        population. No-op if none pending."""
        if self._pending_settle is None:
            return
        fut, self._pending_settle = self._pending_settle, None
        await fut

    def _flag_after_bank(self, parts, rows, records):
        """Per banked row: evict if terminated; at unit grain a surviving row pops out
        of the engine (its group aborts, the maps stay) to wait for the boundary."""
        for k_i, (p, row) in enumerate(zip(parts, rows)):
            if isinstance(records[k_i].token, EndOfSequence):
                assert p.done, "burst drew EOS for a particle that did not terminate"
            if p.done:
                self._drop_row(row)
            elif self.d.sync_boundary and records[k_i].step is not None:
                h = self.row_handle.get(row)
                if h is not None:
                    with self._drain_lock:
                        self.abort_handles.add(h)

    def _on_main(self, coro):
        """Run ``coro`` on the main loop (parked in ``run_in_executor``) from the burst
        worker thread, blocking for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.d.main_loop).result()

    async def _bank_pop(self, parts, records):
        """Bank one step's records into the population (score/extend/critic; sets
        p.done). A token-grain critic's LM leaf is served from its lane's banked sum --
        it must not forward mid-burst."""
        c = self.d.controller
        for p, rec in zip(parts, records):
            if rec.step is None:
                continue
            if self.d.defers_critic:  # settles at the boundary, nothing to serve
                await c.bank_row(p, *rec.step)
            else:
                with c.serve_lanes(p.group, [p], [p]):
                    await c.bank_row(p, *rec.step)

    def _drop_row(self, row):
        """Evict a particle's engine group: abort it, drop both maps, release its task."""
        self._release_row(row)
        h = self.row_handle.pop(row, None)
        if h is not None:
            self.handle_row.pop(h, None)
            with self._drain_lock:
                self.abort_handles.add(h)

    def _flush(self, crossed):
        """Rebuild the engine requests of every row a resample invalidated -- survivors
        too, since a crossing rewrites their contexts. Drop them all before re-adding
        any: a row's handle must be gone before its replacement mints one."""
        c = self.d.controller
        for rows in crossed:
            for row in rows:
                self._drop_row(int(row))
            for row in rows:
                p = c.particles[int(row)]
                if not p.done:
                    self._add_group(p)



def critic_deferred(sampler, controller):
    """Whether the critic settles at round boundaries (engine drained) rather than
    per step. False only for free-running in-burst resampling with
    ``twist_with_critic`` (it consumes twists mid-burst); true otherwise."""
    return not (sampler.burst_free_running() and controller.twist_with_critic)


def burst_blocker(controller):
    """Why this config can't run the engine burst, or ``None`` if it can. Needs a target with
    one engine-burst LM leaf, must be forward-free, and (if batched) burst-homogeneous
    (:func:`_batch_blocker`)."""
    s = controller.samplers[0]
    # The burst banks one warm-row increment per record, and a record carries one
    # committed item. `terminate_when` appends an EOS the sampler never drew, so that
    # step commits two -- the drawn token and the EOS -- and one of them goes unbanked
    # whichever way the record's token is read. Twisting is what consumes those sums.
    if controller.terminate_when is not None and controller.twist_with_critic:
        return BurstBlock(
            BlockReason.UNBANKABLE_STEP,
            "`terminate_when` closes a step with an undrawn EOS, which the per-step "
            "lane bank cannot represent alongside the drawn token",
        )
    if find_engine_lm(s.target) is None:
        return BurstBlock(
            BlockReason.NO_ENGINE_LEAF, "sampler target has no single engine-burst LM leaf"
        )
    # Forward-free invariant: every LM leaf on a group's per-step draw path (target/
    # proposal) must be an injected lane, or it would forward inside the burst (which
    # can't supply it). A deferred critic scores at the drain; a non-deferred
    # (token-grain) critic is served per step from its lane's bank, which covers
    # exactly its own engine leaf.
    for g, (samp, crit, lanes) in enumerate(
        zip(controller.samplers, controller.critics, controller.group_lanes)
    ):
        # Every lane is an engine request advanced by the drawn token ids and banked at
        # the drawn token's index, so a lane's leaf must sit on the draw path's own
        # engine. One over another tokenizer would be fed ids that mean something else.
        engine = lanes[0].model if lanes[0] is not None else None
        if engine is None or any(lf is None or lf.model is not engine for lf in lanes):
            return BurstBlock(
                BlockReason.LANE_OFF_ENGINE,
                f"group {g}: a lane's LM leaf is not on the draw path's engine, so the "
                "drawn token ids do not index its vocabulary",
            )
        injected = set(lanes)
        draw = samp.burst_draw_sampler()
        for pot in (draw.target, draw.proposal):
            if pot is None:
                continue
            if any(lm not in injected for lm in lm_leaves(pot)):
                return BurstBlock(
                    BlockReason.FORWARD_NOT_INJECTABLE,
                    f"group {g}: a draw-path LM leaf would forward inside the burst "
                    "(it is not an injected lane)",
                )
        if crit is not None and not critic_deferred(samp, controller):
            if any(lm not in injected for lm in lm_leaves(crit)):
                return BurstBlock(
                    BlockReason.FORWARD_NOT_INJECTABLE,
                    f"group {g}: the token-grain critic has an LM leaf beyond its own "
                    "lane; it cannot be served from a banked sum",
                )
    if len(controller.samplers) > 1:
        # The burst serves one critic lane per group, present for all or none; mixed
        # batches fall back to the per-token loop.
        if len({c is None for c in controller.critics}) != 1:
            return BurstBlock(
                BlockReason.BATCH_HETEROGENEOUS,
                "groups mix critic-present and critic-free problems",
            )
        return _batch_blocker(controller)
    return None


def _batch_blocker(controller):
    """Why a batched burst's groups can't share one forward, or ``None`` if
    burst-homogeneous. Groups must share grain (the driver reads it off group 0 for the
    whole population) and per-lane temperature/LoRA -- one engine request serves lane
    ``l`` for every group at once. They may differ in sampler kind, prompt, critic, and
    constraint: the parked lane draws each row through its OWN group's sampler."""
    samplers = controller.samplers
    s0, lanes0 = samplers[0], controller.group_lanes[0]

    def blocked(detail):
        return BurstBlock(BlockReason.BATCH_HETEROGENEOUS, f"group {g} {detail}")

    for g, (s, lanes) in enumerate(
        zip(samplers[1:], controller.group_lanes[1:]), start=1
    ):
        if s.burst_free_running() != s0.burst_free_running():
            return blocked("draws at a different grain from group 0")
        if len(lanes) != len(lanes0):
            return blocked(f"has {len(lanes)} lanes, not {len(lanes0)}")
        for li, (v, v0) in enumerate(zip(lanes, lanes0)):
            if v.model is not v0.model:
                return blocked(f"lane {li} uses a different engine")
            if getattr(v, "temperature", None) != getattr(v0, "temperature", None):
                return blocked(f"lane {li} temperature differs from group 0")
            if v.lora_name != v0.lora_name:
                return blocked(f"lane {li} uses a different LoRA adapter from group 0")
    return None


class BurstLoop:
    """Engine-accelerated SMC driver: each :meth:`round` runs the live rows as an
    engine burst. The burst never resamples; the controller-owned loop does. Only
    valid when :func:`burst_blocker` is ``None``."""

    def __init__(self, controller):
        self.controller = controller
        self.sampler = controller.samplers[0]
        # Unit grain hands back at the synced boundary (controller runs the round
        # boundary); token grain records/resamples in place inside the burst.
        self.sync_boundary = not self.sampler.burst_free_running()
        # A deferred critic (non-free-running, or ess=0 terminal-only) settles at the
        # round boundary — engine drained, its LM leaves may forward there.
        self.defers_critic = critic_deferred(self.sampler, controller)
        self.n_bursts = 0  # bursts opened -- for verifying the burst path ran
        # views: group 0's engine lanes, one request each -- the batched burst draws
        # every group through group 0's sampler. ``_batch_blocker`` is what guarantees
        # every group agrees on lane count; a lane's leaf still differs per group.
        self.views = controller.group_lanes[0]
        # The engine LM the burst drives (run_burst + eos id); views share its model.
        self.llm = self.views[0]
        if self.llm is None:  # pragma: no cover - guarded by burst_blocker
            raise ValueError("sampler target has no single engine-burst LM leaf")

        # Per-(group, lane) prompt prefix, snapshotted on the main thread (``prompt_ids``
        # is a ContextVar invisible on the ``run_burst`` worker thread).
        self.view_prefixes = [
            [list(v.prompt_ids) for v in lanes] for lanes in controller.group_lanes
        ]

        # Engine token id committed as the placeholder for an aborted/EOS row.
        eos_idxs = list(self.llm.token_maps.eos_idxs)
        if not eos_idxs:
            raise ValueError(
                'Engine LM has no EOS token id; the burst needs one. Use accelerate="off".'
            )
        self.eos_id = eos_idxs[0]

    async def round(self):
        """One burst over the live rows: a whole generation at token grain (resampling
        in place at ESS crossings), one synced unit per row at unit grain. Runs the
        engine decode loop in a worker thread; each step's draw hops back to this loop
        (parked in ``run_in_executor``) via ``run_coroutine_threadsafe`` (see
        ``_Burst.draw``)."""
        loop = self.main_loop = asyncio.get_running_loop()
        self.n_bursts += 1
        live = [p for p in self.controller.particles if not p.done]
        b = _Burst(self, live)
        # Every group's budget, not group 0's: at unit grain the groups may size
        # their units differently, and the engine cap must not cut the largest short.
        max_steps = max(s.burst_max_steps(live) for s in self.controller.samplers)
        await loop.run_in_executor(
            None,
            lambda: self.llm.model.run_burst(control=b, max_steps=max_steps),
        )

    async def run(self):
        return await self.controller.run(self)
