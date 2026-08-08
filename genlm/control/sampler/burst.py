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


class _RowSeat:
    """One particle's seat in the burst: its coroutine reads the engine's warm through
    the seat and publishes its draw there; the burst delivers, picks, banks, releases.

    Not to be confused with an engine LANE, which is a request slot shared across groups.
    A row occupies one seat and reads every lane."""

    def __init__(self):
        self._warm = None
        self._served_len = None  # None until this step's warm has been read
        self._delivered = asyncio.Event()
        # Set when the row is done with this step's warm: parked at the picker, parked
        # for the next subunit's warm, or returned. The burst's quiescence signal.
        self.settled = asyncio.Event()
        # The row's published draw while it waits at the picker, else ``None``.
        self.pending = None
        # The step a row returned WITHOUT drawing (a forced EOS at ``max_tokens``, every
        # AWRS row): the burst still owes the engine a token and takes it off the step.
        self.returned = None
        # One committed draw per delivered warm: the burst reads this step's token off
        # the pick, so a second draw would have nowhere to go.
        self._drawn = False
        self._resolved = asyncio.Event()
        self._result = None
        # Set once the burst has banked this step's lane increments. A row must not reach
        # ``bank_row`` before then -- an inline critic's ``serve_lanes`` reads those sums.
        self.banked = asyncio.Event()

    async def draw(self, lazyweights, slot, step):
        """Row side: publish this draw and wait for the burst to make it with the rest of
        the population. ``(slot, step)`` travel with it -- the key lives in the row's own
        task context, which the burst cannot read."""
        if self._drawn:
            raise RuntimeError(
                "burst row drew twice from one warm. The burst commits one token per "
                "decode step, so the second draw could not be committed; read fresh "
                'logits between draws, or run with accelerate="off".'
            )
        self._drawn = True
        self.pending = (lazyweights, slot, step)
        self._resolved.clear()
        self.settled.set()
        await self._resolved.wait()
        return self._result

    def resolve(self, result):
        """Burst side: hand back one row's ``(token, logZ, logp)`` and let it run on."""
        self.pending = None
        self._result = result
        self.settled.clear()
        self._resolved.set()

    async def next_warm(self, context):
        """Row side: this read's warm, parking when the step's warm is spent. Reads at one
        context length are the same decode step (target and proposal share it), so only a
        read past a draw parks."""
        if self._warm is None or self._served_len not in (None, len(context)):
            self._warm = None
            self._delivered.clear()
            self.settled.set()
            await self._delivered.wait()
        self._served_len = len(context)
        return self._warm

    def finish(self, step):
        """Row side: this SMC step is over. ``step`` is read only when the row never drew
        -- the burst is still owed a token and takes it off the last item."""
        self.returned = step
        self.settled.set()

    def deliver(self, warm):
        """Burst side: hand this row the step's warm and let it run."""
        self.settled.clear()
        self._warm = warm
        self._served_len = None
        self._drawn = False
        self.returned = None
        self._delivered.set()


class _RoundGate:
    """The population barrier a free-running burst's round closes on: every row the engine
    advanced this step banks, then the last arrival runs the round boundary (record +
    per-group ESS test + resample) and releases the rest.

    Membership is re-derived from the population at every close, never carried: a resample
    reindexes over a group's WHOLE row set, finished rows included, so a done row can
    inherit a live ancestor and needs its engine requests and its coroutine back."""

    def __init__(self, burst):
        self.b = burst
        self.remaining = 0
        self.open = asyncio.Event()

    def arm(self, rows):
        """Burst side: how many rows owe this round an arrival. Called with every row
        parked, so no arrival can race it."""
        self.remaining = len(rows)
        self.open.clear()

    async def arrive(self):
        """Row side: this row banked its step; the last arrival closes the round. Every
        row the engine advanced must arrive, terminating ones included -- one that
        returned without arriving would leave the round short forever."""
        self.remaining -= 1
        if self.remaining > 0:
            await self.open.wait()
            return
        self._close()

    def _close(self):
        """Close the round, then restore the population's engine state: rebuild whatever
        a crossing invalidated and give every revived row a coroutine again.

        ``record=True`` unconditionally: a free-running step is one SMC step per row, so
        every arriving row advanced."""
        b = self.b
        b._flush(b.d.controller.round_boundary(record=True))
        for p in b.d.controller.particles:
            if not p.done and p.row not in b.tasks:
                b._spawn_row(p)
        self.open.set()


class _Burst:
    """Per-burst engine state for one ``run_burst``. ``draw``/``drain_*``/``on_burst_end``
    are the engine seam (the backend drives the control through them). One engine group
    per particle: K view requests the backend keeps in lockstep."""

    def __init__(self, d, live):
        self.d = d
        # Adapter names snapshotted at burst start: a lora_name rebind mid-run must
        # not split this burst across adapters.
        self.view_loras = [v.lora_name for v in d.views]
        # The drain streams are written by the row coroutines on the main loop while the
        # engine forwards, and read from the engine thread's ``_drain``.
        self._drain_lock = threading.Lock()
        self.abort_handles = set()
        self.add_handles = []
        self.handle_row = {}  # engine group handle -> particle row
        self.row_handle = {}  # particle row -> engine group handle
        self.next_handle = 0
        self.seats = {}  # particle row -> _RowSeat
        self.tasks = {}  # particle row -> its coroutine, one per row for the whole burst
        self.gate = _RoundGate(self)
        self._tail = None  # the previous step's burst-side tail, in flight
        for p in live:
            self._add_group(p)
            self._spawn_row(p)

    def context_ids(self, p, view_idx):
        """Engine prompt for one (particle, view) substream: per-view prefix + the particle's
        drawn token ids (EOS dropped; drawn suffix shared across views)."""
        ids = list(self.d.view_prefixes[p.group][view_idx])

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

    def _row_injection(self, warm, i, p):
        """Row ``i``'s ``{leaf: [V+1]}`` slice of the step's warm, one entry per engine
        lane, keyed by the row's OWN group's leaf -- another group's would miss the
        override and forward inside the engine's step."""
        return {
            leaf: leaf.make_lazy_weights(warm[lane][i])
            for lane, leaf in enumerate(self.d.controller.group_lanes[p.group])
        }

    def _spawn_row(self, p):
        """Start this row's coroutine: one task for the whole burst, stepping the same
        ``Controller.draw_step`` the per-token driver runs. Its logits reads park on this
        row's seat."""
        seat = _RowSeat()
        self.seats[p.row], self.tasks[p.row] = seat, asyncio.ensure_future(
            self._run_row(p, seat)
        )

    async def _run_row(self, p, seat):
        """One particle's whole burst: draw a step, bank it, close the round, repeat.

        Runs against the engine's next forward, so the row walks its next step's
        context-only potentials there and is already parked at its logits read when the
        next warm lands."""
        c, row = self.d.controller, p.row
        while True:
            # The row owns its own per-step seat state: a burst-side clear could race
            # a row that has not yet woken from the previous step.
            seat.banked.clear()
            if c.twist_with_critic:
                c.particles.untwist(row)
            # Scoped to the draw ONLY: a bank-path potential that reached the seam would
            # park on a warm nobody will deliver instead of raising.
            with burst_row(seat):
                step = await c.draw_step(p)
            seat.finish(step)
            await seat.banked.wait()
            if self.d.defers_critic:  # settles at the boundary, nothing to serve
                await c.bank_row(p, *step)
            else:
                with c.serve_lanes(p.group, [p], [p]):
                    await c.bank_row(p, *step)
            if self.d.sync_boundary:
                # Unit grain: one unit per round. The row leaves the engine at its unit
                # boundary and the controller runs the boundary once the whole burst has
                # drained -- there is no round here for a gate to close.
                self._leave(row)
                return
            if p.done:
                # Leave BEFORE arriving: the round is still one arrival short and this
                # row owes it, but the next step must not deliver a warm to a row that
                # will never read it.
                self._leave(row)
                await self.gate.arrive()
                return
            await self.gate.arrive()
            if p.done:  # a crossing handed this row a finished ancestor
                self._leave(row)  # `_flush` already dropped the handle; the evict no-ops
                return

    def _pick(self, rows):
        """Draw every row waiting at the picker, returning ``{row: (token, logZ, logp)}``
        WITHOUT releasing them -- the lane bank has to land first. One pass: a row commits
        one draw per delivered warm (``_RowSeat.draw`` enforces it), so no row can reach
        the picker twice in a step."""
        waiting = [row for row in rows if self.seats[row].pending is not None]
        # One batch per vocabulary: groups may carry different constraints, so their
        # rows are different widths over different tokens and cannot stack together.
        by_vocab = {}
        for row in waiting:
            by_vocab.setdefault(id(self.seats[row].pending[0].decode), []).append(row)
        picked = {}
        for batch in by_vocab.values():
            picked.update(self._batch_draw(batch))
        return picked

    def _batch_draw(self, batch):
        """One vocabulary's parked rows, drawn together: one logsumexp, one keyed pick, one
        gather. Keying makes this byte-identical to the same rows drawn one at a time."""
        pend = [self.seats[row].pending for row in batch]
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
        return {
            row: (decode[ids[j]], zs[j], picked[j]) for j, row in enumerate(batch)
        }

    def _committed(self, rows, picked):
        """The item each row commits this step: the one it drew, or -- for a step that
        returned without reading the picker (a forced EOS at ``max_tokens``, AWRS) -- the
        last item of the step it returned."""
        out = {}
        for row in rows:
            drawn = picked.get(row)
            if drawn is not None:
                out[row] = drawn[0]
                continue
            step = self.seats[row].returned
            if step is None:
                raise RuntimeError(
                    f"burst row {row} took a warm without drawing or returning; it "
                    "would run ahead of the engine's lockstep"
                )
            out[row] = flatten_units(step[0])[-1]
        return out

    def _bank_lanes(self, warm, committed, rows):
        """Add this step's committed-token logp to every lane, from the warm row each
        lane's leaf already produced. That running sum IS the leaf's own ``prefix``, and a
        terminating row's EOS increment closes it into its ``complete`` -- so a leaf
        reached inside the burst reads a banked number rather than forwarding.

        Two device->host transfers per lane for the whole population. The group only
        picks the leaf whose vocabulary indexes the committed token, so it resolves per
        row rather than splitting the batch."""
        parts = self.d.controller.particles
        L, gl = parts.lane_logp, self.d.controller.group_lanes
        drawn = [(i, r) for i, r in enumerate(rows)
                 if not isinstance(committed[r], EndOfSequence)]
        eos = [(i, r) for i, r in enumerate(rows)
               if isinstance(committed[r], EndOfSequence)]
        drawn_rows = np.array([r for _, r in drawn], dtype=np.int64)
        eos_rows = np.array([r for _, r in eos], dtype=np.int64)
        for lane in range(len(self.d.views)):
            W = warm[lane]  # [N, V+1] device tensor, rows-order
            if drawn:
                ks = torch.tensor([i for i, _ in drawn], device=W.device)
                idx = torch.tensor(
                    [gl[parts.group[r]][lane].lookup[committed[r]] for _, r in drawn],
                    device=W.device,
                )
                L[lane, drawn_rows] += to_numpy(W[ks, idx])
            if eos:
                ks = torch.tensor([i for i, _ in eos], device=W.device)
                L[lane, eos_rows] += to_numpy(W[ks, -1])

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
        nothing but the tokens -- every row's bank and the round boundary run afterwards,
        on the main loop, against the next forward. Dropped groups get a placeholder."""
        c = self.d.controller
        # Per lane: [G, V+1] warm log-weights (device tensors, no host xfer). Indexed by
        # engine lane slot, not by a leaf -- a lane is one engine request across groups.
        warm = [
            view._process_logw_next_batch(view._maybe_temper(logits[:, vi].float()))
            for vi, view in enumerate(self.d.views)
        ]

        async def _step():
            await self._join_tail()
            # A row advances iff it has a coroutine. Anything else in ``handles`` is
            # popped out or dying and gets the placeholder token.
            live_k = [
                k
                for k, h in enumerate(handles)
                if h in self.handle_row and self.handle_row[h] in self.tasks
            ]
            rows = [self.handle_row[handles[k]] for k in live_k]
            out = [0] * len(handles)
            if not rows:
                return out
            if len(live_k) == len(handles):
                step_warm = warm  # already in ``handles`` order; the gather would copy
            else:
                sel = torch.tensor(live_k, dtype=torch.int64, device=logits.device)
                step_warm = [w[sel] for w in warm]
            for i, row in enumerate(rows):
                self.seats[row].deliver(self._row_injection(step_warm, i, c.particles[row]))
            # Quiescence: every row has consumed this step's warm and is parked at the
            # picker or has returned.
            await asyncio.gather(*(self.seats[row].settled.wait() for row in rows))
            picked = self._pick(rows)
            committed = self._committed(rows, picked)
            # Armed with every row parked, so no arrival can race it.
            if not self.d.sync_boundary:
                self.gate.arm(rows)
            for k, row in zip(live_k, rows):
                tok = committed[row]
                out[k] = self.d.eos_id if isinstance(tok, EndOfSequence) else tok.token_id
            self._tail = asyncio.ensure_future(
                self._release(step_warm, committed, picked, rows)
            )
            return out

        out = self._on_main(_step())
        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    async def _release(self, warm, committed, picked, rows):
        """One step's burst-side tail: bank the lanes, then let every row run on.

        The leading yield is the whole point. The engine thread is released when
        ``draw``'s coroutine resolves, and anything already queued runs before that -- so
        without it the lane bank, the rows' banking and the round boundary all land
        inside the engine's blocking window, which is GPU idle time."""
        await asyncio.sleep(0)
        self._bank_lanes(warm, committed, rows)
        for row in rows:
            seat = self.seats.get(row)
            if seat is None:  # evicted while the tail was queued
                continue
            seat.banked.set()
            if row in picked:
                seat.resolve(picked[row])

    async def _join_tail(self):
        """Await the previous step's tail, so this step delivers over banked, released
        seats. Normally already done -- it runs against the forward."""
        if self._tail is not None:
            fut, self._tail = self._tail, None
            await fut

    def on_burst_end(self):
        """Engine lifecycle hook: decode loop drained, so settle the last step's tail and
        cancel whatever is still parked -- mid-unit rows, and rows waiting on a round the
        engine will not run. On the main loop, since Tasks are not touchable from this
        worker thread."""

        async def _end():
            await self._join_tail()
            for row in list(self.tasks):
                self._forget(row, cancel=True)

        self._on_main(_end())

    def _forget(self, row, *, cancel=False):
        """Drop a row's coroutine and seat. ``cancel`` only from outside the coroutine --
        a row forgets itself as its last act, and cancelling would raise into the caller."""
        task = self.tasks.pop(row, None)
        self.seats.pop(row, None)
        if cancel and task is not None and not task.done():
            task.cancel()

    def _leave(self, row):
        """A row's last act: abort its engine group and forget it."""
        self._evict_handle(row)
        self._forget(row)

    def _on_main(self, coro):
        """Run ``coro`` on the main loop (parked in ``run_in_executor``) from the burst
        worker thread, blocking for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.d.main_loop).result()

    def _evict_handle(self, row):
        """Abort a row's engine group and forget its handle. Touches no coroutine: a
        reindexed row keeps its own, since a crossing rewrites its context but not its
        identity."""
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
                self._evict_handle(int(row))
            for row in rows:
                p = c.particles[int(row)]
                if not p.done:
                    self._add_group(p)


def critic_deferred(sampler, controller):
    """Whether the critic settles at round boundaries (engine drained) rather than per
    step.

    A free-running burst resamples mid-burst, so anything the critic puts into ``logw``
    must land before that resample -- ground truth settles the critic before every ESS
    test (``Controller.run`` orders ``apply_critic_boundary`` ahead of
    ``round_boundary``). Deferring is therefore only safe when nothing consumes the
    critic mid-burst: no twisting, and no resample that can cross."""
    if not sampler.burst_free_running():
        return True
    return not (controller.twist_with_critic or controller.ess_threshold > 0)


def burst_blocker(controller):
    """Why this config can't run the engine burst, or ``None`` if it can. Needs a target with
    one engine-burst LM leaf, must be forward-free, and (if batched) burst-homogeneous
    (:func:`_batch_blocker`)."""
    s = controller.samplers[0]
    # The burst banks one warm-row increment per committed item, and a step commits one.
    # `terminate_when` appends an EOS the sampler never drew, so that step commits two --
    # the drawn token and the EOS -- and one of them goes unbanked whichever way the
    # committed item is read. Twisting is what consumes those sums.
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
    constraint: each row is drawn, injected and banked through its OWN group's lanes."""
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
        # views: group 0's engine lanes, one request each. A lane slot is one engine
        # request across every group; ``_batch_blocker`` is what guarantees every group
        # agrees on lane count, and a row is always served through its own group's leaf.
        self.views = controller.group_lanes[0]
        # The engine LM the burst drives (run_burst + eos id); views share its model.
        self.llm = self.views[0]

        # Per-(group, lane) prompt prefix, snapshotted on the main thread: ``prompt_ids``
        # is a ContextVar, and a task the worker thread schedules does not inherit it.
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
