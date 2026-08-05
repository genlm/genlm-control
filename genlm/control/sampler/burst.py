"""Engine-accelerated SMC: ``_Burst`` seam, ``BurstLoop`` driver, ``burst_blocker`` gate.
Resample/ESS/log_ml stay Controller-owned, never in the backend."""

import asyncio
import enum
from dataclasses import dataclass, replace

import torch

from genlm.control.constant import EndOfSequence, EOS
from genlm.control.potential.base import burst_logw_next, burst_row
from genlm.control.potential.built_in.llm import (
    find_engine_lm,
    constraint_leaf_ids,
    lm_leaves,
)
from genlm.control.util import draw_key, draw_ordinal, flatten_units


class NotAcceleratable(Exception):
    """Engine acceleration required but the config can't be driven by the burst."""


class BlockReason(enum.Enum):
    """Matchable category for why a config can't run the engine burst."""

    UNSUPPORTED_SAMPLER = "sampler"
    NO_ENGINE_LEAF = "engine_leaf"
    FORWARD_NOT_INJECTABLE = "forward"
    BATCH_HETEROGENEOUS = "batch"


@dataclass(frozen=True)
class BurstBlock:
    """``burst_blocker``'s verdict: a :class:`BlockReason` + human ``detail``."""

    reason: BlockReason
    detail: str



@dataclass
class BurstDraw:
    """One live row's result from ``burst_draw_batch``.

    token: ``Token``/``EOS`` drawn this step. step: ``(to_append, logw, logp)`` or
    ``None`` (mid-step). pop: pop the row out without terminating (unit-boundary wait).
    """

    token: object
    step: tuple | None
    pop: bool = False


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
        self.views = d.views
        # Adapter names snapshotted at burst start: a lora_name rebind mid-run must
        # not split this burst across adapters.
        self.view_loras = [v.lora_name for v in d.views]
        self.abort_handles = set()
        self.add_handles = []
        self.handle_row = {}  # engine group handle -> particle row
        self.row_handle = {}  # particle row -> engine group handle
        self.next_handle = 0
        for p in live:
            self._add_group(p)
        self.channels = {}  # particle row -> _RowChannel (parked-row lane)
        self.tasks = {}  # particle row -> its in-flight transition Task
        self._pending_bank = None  # last step's deferred bank Future (overlaps next forward)

    def context_ids(self, p, view_idx):
        """Engine prompt for one (particle, view) substream: per-view prefix + the particle's
        drawn token ids (EOS dropped; drawn suffix shared across views)."""
        g = self.d.controller.particles.group[p._i]
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
        self.handle_row[h] = p._i
        self.row_handle[p._i] = h
        self.add_handles.append((
            h,
            [self.context_ids(p, vi) for vi in range(len(self.views))],
            self.view_loras,
        ))

    def _row_injection(self, warm_batch, i, p):
        """Row ``i``'s ``{view: [V+1]}`` slice of the batched warm, keyed by the row's OWN
        sampler's views: groups carry their own leaves, so another group's would miss the
        override and forward inside the engine's step."""
        return {
            rv: rv.make_lazy_weights(warm_batch[gv].weights[i])
            for rv, gv in zip(_views_of(self.d.controller._sampler_of(p)), self.views)
        }

    def _spawn_row(self, p, row):
        """Start this row's real ``transition`` as a parked task. One scalar draw key for
        the whole transition, exactly as ``Controller.draw_step`` scopes it, so a
        multi-draw transition advances its ordinal per draw on its own."""
        channel = _RowChannel()
        sampler = self.d.controller._sampler_of(p)

        async def run():
            with burst_row(channel), draw_key(row, draw_ordinal(p.context)):
                return await sampler.transition(p.context)

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

    async def _parked_records(self, warm_batch, parts, rows):
        """One decode step through per-row parked ``transition`` tasks: deliver each row's
        warm, then wait for it to park again (mid-unit) or return (the SMC step)."""
        for i, (p, row) in enumerate(zip(parts, rows)):
            channel = self.channels.get(row) or self._spawn_row(p, row)
            channel.deliver(self._row_injection(warm_batch, i, p))
        await asyncio.gather(*(self.channels[row].settled.wait() for row in rows))
        records = []
        for row in rows:
            task = self.tasks[row]
            if task.done():
                step = task.result()
                self._release_row(row)
                records.append(
                    BurstDraw(
                        token=flatten_units(step[0])[-1],
                        step=step,
                        pop=self.d.sync_boundary,
                    )
                )
            else:
                records.append(
                    BurstDraw(token=self.channels[row].context[-1], step=None)
                )
        return records

    def drain_aborts(self):
        handles = self.abort_handles
        self.abort_handles = set()
        return list(handles)

    def drain_adds(self):
        adds = self.add_handles
        self.add_handles = []
        return adds

    def draw(self, logits, handles):
        """Engine per-step callback over the complete groups' ``[G, K, vocab]``
        logits: one token per live group. Banking is deferred under free running to
        overlap the next forward; dropped groups get a placeholder token."""
        c = self.d.controller
        sampler = self.d.sampler
        # Per view: [G, V+1] warm log-weights (device tensors, no host xfer).
        processed = [
            view._process_logw_next_batch(view._maybe_temper(logits[:, vi].float()))
            for vi, view in enumerate(self.views)
        ]

        async def _step():
            # (1) Join prior deferred bank so select draws over the resampled population.
            await self._join_pending_bank()
            # (2) live groups still in handle_row.
            live_k = [k for k, h in enumerate(handles) if h in self.handle_row]
            rows = [self.handle_row[handles[k]] for k in live_k]
            parts = [c.particles[row] for row in rows]
            if c.twist_with_critic:
                c.particles.untwist_subset(rows)
            out = [0] * len(handles)
            if rows:
                # One batched warm per view ([N, V+1], rows-order).
                sel = torch.tensor(live_k, dtype=torch.int64, device=logits.device)
                warm_batch = {
                    view: view.make_lazy_weights(processed[vi][sel])
                    for vi, view in enumerate(self.views)
                }
                if sampler.burst_draws_batched():
                    records = await sampler.burst_draw_batch(
                        warm_batch, [p.context for p in parts], rows, self
                    )
                else:
                    records = await self._parked_records(warm_batch, parts, rows)
                # Settle each row through the controller's own step shape. At the
                # max_tokens boundary ``draw_step`` forces EOS via ``logw_eos``, whose
                # injection must be keyed by THAT row's sampler's views, not group 0's:
                # a mis-keyed injection forwards inside the engine's own step (deadlock).
                for k_i, p in enumerate(parts):
                    if p.max_tokens_left == 1:
                        self._release_row(rows[k_i])  # its in-flight unit is discarded
                        with burst_logw_next(self._row_injection(warm_batch, k_i, p)):
                            records[k_i] = BurstDraw(token=EOS, step=await c.draw_step(p))
                    elif records[k_i].step is not None:
                        step = c._close_if_stopped(p, records[k_i].step)
                        if step is not records[k_i].step:
                            records[k_i] = replace(records[k_i], step=step)
                # Bank the twist view: the drawn token's warm-row logp per particle —
                # the increments of the critic LM leaf's prefix (plus, under clip, the
                # contrast against the draw target; both rows are already in hand).
                if self.d.twist_view is not None:
                    tw = warm_batch[self.d.twist_view].weights  # [N, V+1] device tensor
                    lk = self.d.twist_view.lookup
                    ks = [k for k, r in enumerate(records)
                          if not isinstance(r.token, EndOfSequence)]
                    if ks:
                        rows_t = torch.tensor(ks, device=tw.device)
                        idx = torch.tensor([lk[records[k].token] for k in ks],
                                           device=tw.device)
                        tvals = None
                        if c.twist.clip is not None:
                            tvals = warm_batch[self.d.views[0]].weights[rows_t, idx].tolist()
                        c.twist.bank(c.particles, [rows[k] for k in ks],
                                     tw[rows_t, idx].tolist(), tvals)
                    # A terminating row's EOS increment completes the sum, so
                    # ``served_complete`` covers the full sequence in either lane --
                    # inline (token grain) or at the round boundary (deferred).
                    eos_ks = [k for k, r in enumerate(records)
                              if isinstance(r.token, EndOfSequence)]
                    if eos_ks:
                        c.twist.bank_eos(c.particles, [rows[k] for k in eos_ks],
                                         tw[eos_ks, -1].tolist())
                for k, rec in zip(live_k, records):
                    tok = rec.token
                    out[k] = (
                        self.d.eos_id if isinstance(tok, EndOfSequence) else tok.token_id
                    )
            else:  # no live groups this step (all drained/terminated)
                records = []
            # (3) Bank: free running defers (overlaps next forward); unit grain banks inline
            # (its pop-out abort must take effect this step).
            if sampler.burst_free_running():
                self._pending_bank = (
                    asyncio.ensure_future(self._bank_pop(parts, records)),
                    parts,
                    rows,
                    records,
                )
            else:
                await self._bank_pop(parts, records)
                self._flag_after_bank(parts, rows, records)
            return out

        out = self._on_main(_step())
        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def on_burst_end(self):
        """Engine lifecycle hook: decode loop drained, join the final deferred bank +
        resample (no next ``draw`` to do it). Rows still parked mid-unit at the drain
        (the engine hit ``burst_max_steps``) are cancelled with their partial unit --
        on the main loop, since Tasks are not touchable from this worker thread."""

        async def _end():
            await self._join_pending_bank()
            for row in list(self.tasks):
                self._release_row(row)

        self._on_main(_end())

    async def _join_pending_bank(self):
        """Await the deferred bank, then flag its rows and resample. No-op if none pending."""
        if self._pending_bank is None:
            return
        fut, parts, rows, records = self._pending_bank
        self._pending_bank = None
        await fut  # banking: score/extend/critic, sets p.done
        self._flag_after_bank(parts, rows, records)
        if self.d.sampler.burst_free_running():
            self.resample_realize()

    def _flag_after_bank(self, parts, rows, records):
        """Per banked row: evict if terminated; if pop, abort its engine group but
        keep maps."""
        for k_i, (p, row) in enumerate(zip(parts, rows)):
            if isinstance(records[k_i].token, EndOfSequence):
                assert p.done, "burst drew EOS for a particle that did not terminate"
            if p.done:
                self._drop_row(row)
            elif records[k_i].pop:
                h = self.row_handle.get(row)
                if h is not None:
                    self.abort_handles.add(h)

    def _on_main(self, coro):
        """Run ``coro`` on the main loop (parked in ``run_in_executor``) from the burst
        worker thread, blocking for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.d.main_loop).result()

    async def _bank_pop(self, parts, records):
        """Bank one step's records into the population (score/extend/critic; sets
        p.done). A token-grain critic's LM leaf is served from the banked twist
        sums (prefix and complete) -- it must not forward mid-burst."""
        c = self.d.controller
        for p, rec in zip(parts, records):
            if rec.step is None:
                continue
            if self.d.defers_critic:  # settles at the boundary, nothing to serve
                await c.bank_row(p, *rec.step)
            else:
                with c.serve_row(p, dlogp=rec.step[2]):
                    await c.bank_row(p, *rec.step)
        # Token grain records per step here; unit grain once per round boundary.
        if self.d.sampler.burst_free_running() and any(r.step is not None for r in records):
            c._record_step()

    def _drop_row(self, row):
        """Evict a particle's engine group: abort it, drop both maps, release its task."""
        self._release_row(row)
        h = self.row_handle.pop(row, None)
        if h is not None:
            self.handle_row.pop(h, None)
            self.abort_handles.add(h)

    def resample_realize(self):
        """Translate a completed per-group resample into engine abort/re-add; return whether
        anything crossed. Every row in a crossing group is flushed (survivors too)."""
        c = self.d.controller
        groups, _ = c._maybe_resample()
        for g in groups:
            for row in c._group_rows[g]:
                self._drop_row(int(row))
            for row in c._group_rows[g]:
                p = c.particles[int(row)]
                if not p.done:
                    self._add_group(p)
        return bool(groups)



def _views_of(sampler):
    """LM views the burst injects for ``sampler``: draw sampler's target leaf + its proposal's
    (if any). A view is ``None`` if its potential has no single engine-burst leaf."""
    s = sampler.burst_draw_sampler()
    proposal = s.proposal
    views = [find_engine_lm(s.target)]
    if proposal is not None:
        views.append(find_engine_lm(proposal))
    return views


def critic_deferred(sampler, controller):
    """Whether the critic settles at round boundaries (engine drained) rather than
    per step. False only for free-running in-burst resampling with
    ``twist_with_critic`` (it consumes twists mid-burst); true otherwise."""
    return not (sampler.burst_free_running() and controller.twist_with_critic)


def burst_blocker(controller):
    """Why this config can't run the engine burst, or ``None`` if it can. Needs a
    burst-capable sampler over a target with one engine-burst LM leaf, must be forward-free,
    and (if batched) burst-homogeneous (:func:`_batch_blocker`)."""
    s = controller.samplers[0]
    if not s.supports_burst():
        return BurstBlock(
            BlockReason.UNSUPPORTED_SAMPLER,
            f"{type(s).__name__} does not support the engine burst",
        )
    if find_engine_lm(s.target) is None:
        return BurstBlock(
            BlockReason.NO_ENGINE_LEAF, "sampler target has no single engine-burst LM leaf"
        )
    # Forward-free invariant: every LM leaf on a group's per-step draw path (target/
    # proposal) must be an injected view, or it would forward inside the burst (which
    # can't supply it). A deferred critic scores at the drain; a non-deferred
    # (token-grain) critic is served per step from banked twist sums, which covers
    # exactly its own engine leaf.
    for g, (samp, crit) in enumerate(zip(controller.samplers, controller.critics)):
        injected = set(_views_of(samp))
        draw = samp.burst_draw_sampler()
        for pot in (draw.target, draw.proposal):
            if pot is None:
                continue
            if any(lm not in injected for lm in lm_leaves(pot)):
                return BurstBlock(
                    BlockReason.FORWARD_NOT_INJECTABLE,
                    f"group {g}: a draw-path LM leaf would forward inside the burst "
                    "(it is not an injected view)",
                )
        if crit is not None and not critic_deferred(samp, controller):
            servable = {find_engine_lm(crit)}
            if any(lm not in servable for lm in lm_leaves(crit)):
                return BurstBlock(
                    BlockReason.FORWARD_NOT_INJECTABLE,
                    f"group {g}: the token-grain critic has an LM leaf beyond its own "
                    "engine leaf; it cannot be served from banked twist sums",
                )
    if len(controller.samplers) > 1:
        # The burst serves one twist view per group, present for all or none; mixed
        # batches fall back to the per-token loop.
        if len({c is None for c in controller.critics}) != 1:
            return BurstBlock(
                BlockReason.BATCH_HETEROGENEOUS,
                "groups mix critic-present and critic-free problems",
            )
        return _batch_blocker(controller.samplers)
    return None


def _batch_blocker(samplers):
    """Why a batched burst can't draw every group through group 0's sampler, or ``None`` if
    burst-homogeneous. Groups must share sampler kind, K views, per-view engine/temperature/
    LoRA, and constraint; they may differ only in prompt and critic. A sampler that
    ``burst_routes_groups`` draws each row through its own group's sampler, so groups may
    additionally differ in constraint."""
    s0 = samplers[0]
    views0 = _views_of(s0)
    constraint0 = constraint_leaf_ids(s0.target)

    def blocked(detail):
        return BurstBlock(BlockReason.BATCH_HETEROGENEOUS, f"group {g} {detail}")

    for g, s in enumerate(samplers[1:], start=1):
        if type(s) is not type(s0):
            return blocked(f"sampler is {type(s).__name__}, not {type(s0).__name__}")
        views = _views_of(s)
        if len(views) != len(views0):
            return blocked(f"has {len(views)} views, not {len(views0)}")
        for vi, (v, v0) in enumerate(zip(views, views0)):
            if v is None or v.model is not v0.model:
                return blocked(f"view {vi} uses a different engine")
            if getattr(v, "temperature", None) != getattr(v0, "temperature", None):
                return blocked(f"view {vi} temperature differs from group 0")
            if v.lora_name != v0.lora_name:
                return blocked(f"view {vi} uses a different LoRA adapter from group 0")
        routes = s0.burst_routes_groups() or not s0.burst_draws_batched()
        if not routes and constraint_leaf_ids(s.target) != constraint0:
            return blocked("has a different constraint")
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
        # views: LM leaves whose warm logits the burst injects (group 0's target+proposal,
        # plus the critic's engine leaf when twisting will read it -- at the boundary
        # if deferred, per step if token-grain); the batched burst draws every group
        # through group 0's sampler.
        self.twist_leaves = controller.twist_leaves
        self.twist_view = self.twist_leaves[0]
        # The burst banks per-token twist sums whenever a twist view is injected.
        self.banks_twist = self.twist_view is not None
        assert all(
            (lf is None) == (self.twist_view is None) for lf in self.twist_leaves
        ), "batched groups must agree on having an engine-LM critic leaf"
        self.views = _views_of(self.sampler) + (
            [self.twist_view] if self.twist_view is not None else []
        )
        # The engine LM the burst drives (run_burst + eos id); views share its model.
        self.llm = self.views[0]
        if self.llm is None:  # pragma: no cover - guarded by burst_blocker
            raise ValueError("sampler target has no single engine-burst LM leaf")

        # Per-(group, view) prompt prefix, snapshotted on the main thread (``prompt_ids`` is a
        # ContextVar invisible on the ``run_burst`` worker thread).
        self.view_prefixes = [
            [list(v.prompt_ids) for v in _views_of(s)]
            + ([list(lf.prompt_ids)] if lf is not None else [])
            for s, lf in zip(controller.samplers, self.twist_leaves)
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
        max_steps = self.sampler.burst_max_steps(live)
        await loop.run_in_executor(
            None,
            lambda: self.llm.model.run_burst(control=b, max_steps=max_steps),
        )

    async def run(self):
        return await self.controller.run(self)
