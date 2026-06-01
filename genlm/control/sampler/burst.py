"""Engine-accelerated SMC: the ``_Burst`` seam, the ``BurstLoop`` driver, and the
``burst_blocker`` capability gate. Banks into the ``Controller`` (smc.py) via the
instance it is handed; resample/ESS/log_ml stay Controller-owned, never in the backend."""

import asyncio
import enum
from dataclasses import dataclass

import torch

from genlm.control.constant import EndOfSequence
from genlm.control.potential.built_in.llm import (
    find_engine_lm,
    constraint_leaf_ids,
    lm_leaves,
)


class NotAcceleratable(Exception):
    """Engine acceleration was required but the config can't be driven by the burst.
    Message is ``burst_blocker``'s detail."""


class BlockReason(enum.Enum):
    """Why a config can't run the engine burst -- the matchable category."""

    UNSUPPORTED_SAMPLER = "sampler"
    NO_ENGINE_LEAF = "engine_leaf"
    FORWARD_NOT_INJECTABLE = "forward"
    BATCH_HETEROGENEOUS = "batch"


@dataclass(frozen=True)
class BurstBlock:
    """``burst_blocker``'s verdict: a matchable :class:`BlockReason` + human ``detail``."""

    reason: BlockReason
    detail: str



@dataclass
class BurstDraw:
    """One live row's result from ``burst_draw_batch``.

    Fields:
        token: the ``Token``/``EOS`` drawn this decode step (mapped to an engine id).
        step: ``(to_append, logw, logp)`` if an SMC step completed, else ``None``
            (mid-step, e.g. a unit still accumulating).
        pop: pop this row out of the burst after this step without terminating
            (the unit-boundary wait). Terminated rows are always popped.
    """

    token: object
    step: tuple | None
    pop: bool = False


# -- burst exit reasons: why one burst returned, so BurstLoop can dispatch. A burst
# never resamples. ESS is NOT an exit reason (a token-grain crossing resamples in
# place and the burst keeps running).
_EXIT_TERMINATED = "terminated"  # every flagged row finished on its own
_EXIT_UNIT_SYNC = "unit_sync"  # unit-grain round done; driver runs ESS/resample


class _Burst:
    """Per-burst engine state for one ``run_burst``: the K-substreams-per-particle handle
    maps (``handle_rv``/``row_handles``), abort/re-add queues, scratch, and exit reason.
    ``draw``/``drain_*``/``context_ids``/``on_burst_end`` are the engine seam."""

    def __init__(self, d, live):
        self.d = d
        self.particles = live
        self.views = d.views
        self.k = d.k
        self.abort_rows = set()
        self.add_rows = []
        # K substreams per particle: handle -> (row, view_idx) and row -> [handle/view].
        # The initial population and every mid-burst re-add go through _add_substream,
        # which mints a fresh handle and queues the engine add (drained by run_burst) --
        # one add path, no special-cased "initial requests".
        self.handle_rv = {}
        self.row_handles = {}
        self.next_handle = 0
        for p in live:
            self.row_handles[p._i] = [
                self._add_substream(p, vi, view) for vi, view in enumerate(self.views)
            ]
        self.scratch = {}
        self.exit_reason = _EXIT_TERMINATED
        self._pending_bank = None  # Future of the last step's deferred bank (overlaps next forward)

    def context_ids(self, p, view_idx):
        """Engine prompt for one (particle, view) substream: the view's prefix + the
        particle's drawn token ids (EOS dropped; the drawn suffix is shared across
        views, the prefix is per-view)."""
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

    def _add_substream(self, p, vi, view):
        """Mint a handle for one (particle, view) substream, register it, and queue its
        engine add. The sole add path (initial population + mid-burst re-add)."""
        h = self.next_handle
        self.next_handle += 1
        self.handle_rv[h] = (p._i, vi)
        self.add_rows.append((h, self.context_ids(p, vi), view.lora_name))
        return h

    def drain_aborts(self):
        rows = self.abort_rows
        self.abort_rows = set()
        return list(rows)

    def drain_adds(self):
        rows = self.add_rows
        self.add_rows = []
        return rows

    def draw(self, logits, handles):
        """The engine callback, once per decode step: (1) join the previous step's bank +
        resample; (2) select this step's token for the live rows; (3) kick this step's
        bank async to overlap the next forward. Rows popped from ``handle_rv`` are skipped."""
        c = self.d.controller
        sampler = self.d.sampler
        idx_of = {h: i for i, h in enumerate(handles)}

        # Warm logits for every forwarded substream.
        warm = {}  # handle -> LazyWeights
        for vi, view in enumerate(self.views):
            vh = [h for h in handles if (h in self.handle_rv and self.handle_rv[h][1] == vi)]
            if not vh:
                continue
            rowsidx = [idx_of[h] for h in vh]
            batch = view._process_logw_next_batch(view._maybe_temper(logits[rowsidx].float()))
            for h, row_w in zip(vh, batch):  # row_w: [V+1] device-tensor view, no host xfer
                warm[h] = view.make_lazy_weights(row_w)

        async def _step():
            # (1) Join the prior deferred bank so select draws over the resampled population.
            await self._join_pending_bank()
            # (2) Live rows still in handle_rv.
            rows, seen = [], set()
            for h in handles:
                rv = self.handle_rv.get(h)
                if rv is None:
                    continue
                row = rv[0]
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
            parts = [c.particles[row] for row in rows]
            if c.twist_with_critic:
                c.particles.untwist_subset([p._i for p in parts])
            injections = [
                {view: warm[self.row_handles[row][vi]] for vi, view in enumerate(self.views)}
                for row in rows
            ]
            records = await sampler.burst_draw_batch(
                injections, [p.context for p in parts], rows, self
            )
            # (3) Bank the step. Free running defers it to overlap the next forward; unit
            # grain banks inline (its pop-out abort must take effect this step).
            if sampler.burst_free_running():
                self._pending_bank = (
                    asyncio.ensure_future(self._bank_pop(parts, records)),
                    parts,
                    rows,
                    records,
                )
            else:
                await self._bank_pop(parts, records)

            out = [0] * len(handles)
            for k_i, (p, row) in enumerate(zip(parts, rows)):
                tok = records[k_i].token
                tok_id = self.d.eos_id if isinstance(tok, EndOfSequence) else tok.token_id
                for h in self.row_handles[row]:  # fan the drawn token to the K substreams
                    if h in idx_of:
                        out[idx_of[h]] = tok_id

            if not sampler.burst_free_running():
                self._flag_after_bank(parts, rows, records)
            return out

        out = self._on_main(_step())
        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def on_burst_end(self):
        """Engine lifecycle hook: the decode loop drained, so join the final deferred
        bank + resample (no next ``draw`` to do it). No-op if none pending."""
        self._on_main(self._join_pending_bank())

    async def _join_pending_bank(self):
        """Await the previous step's deferred bank, then flag its rows and resample.
        No-op if none pending; shared by ``draw`` and ``on_burst_end``."""
        if self._pending_bank is None:
            return
        fut, parts, rows, records = self._pending_bank
        self._pending_bank = None
        await fut  # population banking (score/extend/critic; sets p.done)
        self._flag_after_bank(parts, rows, records)
        if self.d.sampler.burst_free_running():
            self.resample_realize()

    def _flag_after_bank(self, parts, rows, records):
        """After a step is banked, tell the engine what to do with its rows: evict a
        terminated row; abort a unit-boundary row's engine rows but keep its maps."""
        for k_i, (p, row) in enumerate(zip(parts, rows)):
            if isinstance(records[k_i].token, EndOfSequence):
                assert p.done, "burst drew EOS for a particle that did not terminate"
            if p.done:
                self._drop_row(row)
            elif records[k_i].pop:
                for h in self.row_handles.get(row, ()):
                    self.abort_rows.add(h)

    def _on_main(self, coro):
        """Run ``coro`` on the main loop from the burst worker thread and block for its
        result (the loop is parked in ``run_in_executor`` and pumps it)."""
        return asyncio.run_coroutine_threadsafe(coro, self.d.main_loop).result()

    async def _bank_pop(self, parts, records):
        """Bank one step's records into the population (score/extend/critic; sets
        ``p.done``). Free running kicks this as a Future to overlap the next forward
        (the deferred bank); unit grain awaits it inline."""
        c = self.d.controller
        await c._bank_steps(parts, records)
        # Token grain records per step here; unit grain once per round in BurstLoop.run.
        if self.d.sampler.burst_free_running() and any(r.step is not None for r in records):
            c._record_step()

    def _drop_row(self, row):
        """Fully evict a row's K substreams: abort their engine requests and drop both
        handle maps."""
        for h in self.row_handles.pop(row, []):
            self.abort_rows.add(h)
            self.handle_rv.pop(h, None)

    def resample_realize(self):
        """Translate a completed per-group resample into engine abort/re-add; return
        whether anything crossed. Every row in a crossing group is flushed (survivors too).
        Only the crossing group is touched."""
        c = self.d.controller
        groups, _ = c._maybe_resample()
        for g in groups:
            for row in c._group_rows[g]:
                self._drop_row(int(row))
            for row in c._group_rows[g]:
                row = int(row)
                p = c.particles[row]
                if p.done:
                    continue
                self.row_handles[row] = [  # re-add all K substreams (per-view prefix + lora)
                    self._add_substream(p, vi, view) for vi, view in enumerate(self.views)
                ]
        return bool(groups)



def _views_of(sampler):
    """The LM views a burst injects for ``sampler``: its draw sampler's target leaf plus
    its proposal's (if any). K=1 (no proposal) is common; K=2 is proposal + prior. A view
    is ``None`` if its potential has no single engine-burst leaf."""
    s = sampler.burst_draw_sampler()
    proposal = s.proposal
    views = [find_engine_lm(s.target)]
    if proposal is not None:
        views.append(find_engine_lm(proposal))
    return views


def burst_blocker(controller):
    """Why this config can't run the engine burst, or ``None`` if it can. Needs a
    burst-capable sampler over a target with one engine-burst LM leaf, and must be
    forward-free: every LM leaf in each group's per-step path (target + proposal + critic)
    must be an injected view. A batched (B>1) population must be burst-homogeneous
    (:func:`_batch_blocker`)."""
    s = controller.unit_sampler
    if not s.supports_burst():
        return BurstBlock(
            BlockReason.UNSUPPORTED_SAMPLER,
            f"{type(s).__name__} does not support the engine burst",
        )
    if find_engine_lm(s.target) is None:
        return BurstBlock(
            BlockReason.NO_ENGINE_LEAF, "sampler target has no single engine-burst LM leaf"
        )
    # Forward-free invariant: every LM leaf in each group's per-step path (target +
    # proposal + critic) must be an injected view (warm logits), else it would do a real
    # engine forward inside the burst, which the burst can't supply. Route to slow lane.
    for g, (samp, crit) in enumerate(zip(controller.samplers, controller.critics)):
        injected = set(_views_of(samp))
        draw = samp.burst_draw_sampler()
        for pot in (draw.target, draw.proposal, crit):
            if pot is None:
                continue
            if any(lm not in injected for lm in lm_leaves(pot)):
                return BurstBlock(
                    BlockReason.FORWARD_NOT_INJECTABLE,
                    f"group {g}: an LM leaf would forward inside the burst (it is not an "
                    "injected view) -- e.g. an LM critic or a second engine LM",
                )
    if len(controller.samplers) > 1:
        return _batch_blocker(controller.samplers)
    return None


def _batch_blocker(samplers):
    """Why a batched burst can't draw every group through group 0's sampler, or ``None``
    if burst-homogeneous. Groups must share sampler kind, the same K views, and per view
    the same engine model, temperature, and LoRA, plus the same constraint; they may
    differ only in prompt and critic."""
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
        if constraint_leaf_ids(s.target) != constraint0:
            return blocked("has a different constraint")
    return None


class BurstLoop:
    """The engine-accelerated SMC driver: runs each step as an engine burst over the live
    contexts. The burst never resamples; :meth:`run` dispatches the controller-owned
    resample. Only valid when :func:`burst_blocker` is ``None``."""

    def __init__(self, controller):
        self.controller = controller
        self.n_bursts = 0  # bursts opened -- for verifying the burst path ran
        self.sampler = controller.unit_sampler
        # Views: the LM leaves whose warm logits the burst injects (group 0's target +
        # proposal). The batched burst draws every group through group 0's sampler.
        self.views = _views_of(self.sampler)
        self.k = len(self.views)
        # The engine LM the burst drives (run_burst + the eos id); views share its model.
        self.llm = self.views[0]
        if self.llm is None:  # pragma: no cover - guarded by burst_blocker
            raise ValueError("sampler target has no single engine-burst LM leaf")

        # Per-(group, view) prompt prefix, snapshotted on the main thread (``prompt_ids``
        # is a ContextVar invisible on the ``run_burst`` worker thread).
        self.view_prefixes = [
            [list(v.prompt_ids) for v in _views_of(s)] for s in controller.samplers
        ]

        # Engine token id committed as the placeholder for an aborted/EOS row.
        eos_idxs = list(self.llm.token_maps.eos_idxs)
        if not eos_idxs:
            raise ValueError(
                'Engine LM has no EOS token id; the burst needs one. Use accelerate="off".'
            )
        self.eos_id = eos_idxs[0]

    async def _run_burst(self, live, loop):
        """Run one stateless burst over ``live`` to a pop-out; return its ``_EXIT_*``
        reason. Runs the engine decode loop in a worker thread; each step's draw hops
        back to this loop via ``run_coroutine_threadsafe`` (see ``_Burst.draw``)."""
        self.n_bursts += 1
        b = _Burst(self, live)
        # Unit grain hands back at the synced boundary (fixed reason); token grain
        # stays terminated, resampling in place at ESS crossings.
        b.exit_reason = (
            _EXIT_TERMINATED if self.sampler.burst_free_running() else _EXIT_UNIT_SYNC
        )
        # Decode budget for one burst (token grain: longest token budget; unit grain:
        # one unit's subunits).
        max_steps = self.sampler.burst_max_steps(live)
        await loop.run_in_executor(
            None,
            lambda: self.llm.model.run_burst(control=b, max_steps=max_steps),
        )
        return b.exit_reason

    async def run(self):
        """The outer driver: each iteration runs the live rows' next step as a burst,
        then dispatches the controller-owned ESS/resample. Repeat until every particle
        is done."""
        controller = self.controller
        await controller.start()

        # The burst worker thread hops each step's SMC coroutine back to this loop
        # (parked in run_in_executor) via run_coroutine_threadsafe; stash it for _Burst.
        loop = self.main_loop = asyncio.get_running_loop()

        while any(not p.done for p in controller.particles):
            live = [p for p in controller.particles if not p.done]
            reason = await self._run_burst(live, loop)
            # Token grain resamples in place at ESS crossings (no exit); a unit-grain
            # round hands back here to record + resample at the synced boundary.
            if reason == _EXIT_UNIT_SYNC:
                controller._record_step()
                controller._maybe_resample()

        return controller.particles
