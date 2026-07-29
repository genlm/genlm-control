"""Engine-accelerated SMC: ``_Burst`` seam, ``BurstLoop`` driver, ``burst_blocker`` gate.
Resample/ESS/log_ml stay Controller-owned, never in the backend."""

import asyncio
import enum
from dataclasses import dataclass, replace

import torch

from genlm.control.constant import EndOfSequence, EOS
from genlm.control.potential.base import burst_logw_next
from genlm.control.potential.built_in.llm import (
    find_engine_lm,
    constraint_leaf_ids,
    lm_leaves,
)


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


class _Burst:
    """Per-burst engine state for one ``run_burst``. ``draw``/``drain_*``/``context_ids``/
    ``on_burst_end`` are the engine seam (the backend drives the control through them)."""

    def __init__(self, d, live):
        self.d = d
        self.views = d.views
        # Adapter names snapshotted at burst start (like view_prefixes): a rebind
        # of a view's lora_name mid-run must not split this burst across adapters.
        self.view_loras = [v.lora_name for v in d.views]
        self.abort_rows = set()
        self.add_rows = []
        # K substreams per particle: handle -> (row, view_idx); row -> [handle/view].
        self.handle_rv = {}
        self.row_handles = {}
        self.next_handle = 0
        for p in live:
            self.row_handles[p._i] = [
                self._add_substream(p, vi) for vi in range(len(self.views))
            ]
        self.scratch = {}
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

    def _add_substream(self, p, vi):
        """Mint+register a handle for one (particle, view) substream, queue its engine add.
        Sole add path (initial population + mid-burst re-add)."""
        h = self.next_handle
        self.next_handle += 1
        self.handle_rv[h] = (p._i, vi)
        self.add_rows.append((h, self.context_ids(p, vi), self.view_loras[vi]))
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
        """Engine callback, once per decode step: (1) join prior step's deferred bank +
        resample; (2) select this step's token for live rows; (3) kick this step's bank
        async to overlap the next forward. Popped rows (gone from ``handle_rv``) skipped."""
        c = self.d.controller
        sampler = self.d.sampler
        idx_of = {h: i for i, h in enumerate(handles)}

        # warm: handle -> [V+1] device tensor of warm logits per forwarded substream.
        warm = {}
        for vi, view in enumerate(self.views):
            vh = [h for h in handles if (h in self.handle_rv and self.handle_rv[h][1] == vi)]
            if not vh:
                continue
            rowsidx = [idx_of[h] for h in vh]
            batch = view._process_logw_next_batch(view._maybe_temper(logits[rowsidx].float()))
            for h, row_w in zip(vh, batch):  # row_w: [V+1] device-tensor view, no host xfer
                warm[h] = row_w

        async def _step():
            # (1) Join prior deferred bank so select draws over the resampled population.
            await self._join_pending_bank()
            # (2) live rows still in handle_rv.
            rows, seen = [], set()
            for h in handles:
                rv = self.handle_rv.get(h)
                if rv is None:
                    continue
                row = rv[0]
                if row not in seen:
                    seen.add(row)
                    rows.append(row)
            # Lockstep guard: the engine can sample a strict subset of a row's K
            # substreams in one step (prefill-chunk boundary / KV preemption landing
            # between siblings). The sampled sibling would advance while the other
            # never receives the token — permanent stream desync. Stall such rows:
            # flush their engine requests and re-add all K substreams at the current
            # context (prefix-cached), so the pair re-enters together next step.
            # No token is banked for a stalled row, so the SMC math is untouched.
            stalled = {row for row in rows
                       if any(h not in warm for h in self.row_handles[row])}
            if stalled:
                for row in stalled:
                    self._drop_row(row)
                    p = c.particles[row]
                    self.row_handles[row] = [
                        self._add_substream(p, vi) for vi in range(len(self.views))
                    ]
                print(f"[burst] lockstep stall: flushed+readded rows "
                      f"{sorted(stalled)}", flush=True)
                rows = [row for row in rows if row not in stalled]
            parts = [c.particles[row] for row in rows]
            if c.twist_with_critic:
                c.particles.untwist_subset([p._i for p in parts])
            if rows:
                # One batched warm per view ([N, V+1], rows-order).
                warm_batch = {
                    view: view.make_lazy_weights(
                        torch.stack([warm[self.row_handles[row][vi]] for row in rows])
                    )
                    for vi, view in enumerate(self.views)
                }
                records = await sampler.burst_draw_batch(
                    warm_batch, [p.context for p in parts], rows, self
                )
                # Settle each row through the controller's own step shape. At the
                # max_tokens boundary ``draw_step`` forces EOS via ``logw_eos``, whose
                # injection must be keyed by THAT row's sampler's views, not group 0's:
                # a mis-keyed injection forwards inside the engine's own step (deadlock).
                for k_i, p in enumerate(parts):
                    if p.max_tokens_left == 1:
                        inj = {rv: rv.make_lazy_weights(warm_batch[gv].weights[k_i])
                               for rv, gv in zip(_views_of(c._sampler_of(p)), self.views)}
                        with burst_logw_next(inj):
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
            else:  # no live rows this step (all drained/terminated)
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

            out = [0] * len(handles)
            for k_i, (p, row) in enumerate(zip(parts, rows)):
                tok = records[k_i].token
                tok_id = self.d.eos_id if isinstance(tok, EndOfSequence) else tok.token_id
                for h in self.row_handles[row]:  # fan the token to the K substreams
                    if h in idx_of:
                        out[idx_of[h]] = tok_id

            if not sampler.burst_free_running():
                self._flag_after_bank(parts, rows, records)
            return out

        out = self._on_main(_step())
        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def on_burst_end(self):
        """Engine lifecycle hook: decode loop drained, join the final deferred bank +
        resample (no next ``draw`` to do it)."""
        self._on_main(self._join_pending_bank())

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
        """Per banked row: evict if terminated; if pop, abort its engine rows but keep maps."""
        for k_i, (p, row) in enumerate(zip(parts, rows)):
            if isinstance(records[k_i].token, EndOfSequence):
                assert p.done, "burst drew EOS for a particle that did not terminate"
            if p.done:
                self._drop_row(row)
            elif records[k_i].pop:
                for h in self.row_handles.get(row, ()):
                    self.abort_rows.add(h)

    def _on_main(self, coro):
        """Run ``coro`` on the main loop (parked in ``run_in_executor``) from the burst
        worker thread, blocking for its result."""
        return asyncio.run_coroutine_threadsafe(coro, self.d.main_loop).result()

    async def _bank_pop(self, parts, records):
        """Bank one step's records into the population (score/extend/critic; sets p.done)."""
        c = self.d.controller
        for p, rec in zip(parts, records):
            if rec.step is not None:
                await c.bank_row(p, *rec.step)
        # Token grain records per step here; unit grain once per round boundary.
        if self.d.sampler.burst_free_running() and any(r.step is not None for r in records):
            c._record_step()

    def _drop_row(self, row):
        """Evict a row's K substreams: abort their engine requests, drop both maps."""
        for h in self.row_handles.pop(row, []):
            self.abort_rows.add(h)
            self.handle_rv.pop(h, None)

    def resample_realize(self):
        """Translate a completed per-group resample into engine abort/re-add; return whether
        anything crossed. Every row in a crossing group is flushed (survivors too)."""
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
                self.row_handles[row] = [  # re-add all K substreams
                    self._add_substream(p, vi) for vi in range(len(self.views))
                ]
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
    """Whether critic math settles at round boundaries (engine drained) rather than
    being consumed per step. Single source of truth shared by :func:`burst_blocker`
    (legality: a deferred critic's LM leaves may forward at the drain) and
    :class:`BurstLoop` (routing: a non-deferred critic scores inline in ``bank_row``'s
    pre-boundary path — engine-free there by ``burst_blocker``). Free-running in-burst
    resampling consumes twists mid-burst, so there the critic is NOT deferrable;
    everywhere else (unit grain; ess=0 terminal-only) it is."""
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
    # Forward-free invariant: every LM leaf in a group's per-step DRAW path (target/
    # proposal) must be an injected view, else it would forward inside the burst (which
    # can't supply it). The critic is boundary-scored (``apply_critic_boundary`` runs at
    # the engine drain), so its LM leaves are legal — EXCEPT token-grain in-burst
    # resampling (free-running + twist_with_critic), which consumes twists mid-burst.
    for g, (samp, crit) in enumerate(zip(controller.samplers, controller.critics)):
        injected = set(_views_of(samp))
        draw = samp.burst_draw_sampler()
        deferred = critic_deferred(samp, controller)
        pots = (draw.target, draw.proposal) + ((crit,) if not deferred else ())
        for pot in pots:
            if pot is None:
                continue
            if any(lm not in injected for lm in lm_leaves(pot)):
                return BurstBlock(
                    BlockReason.FORWARD_NOT_INJECTABLE,
                    f"group {g}: an LM leaf would forward inside the burst (it is not an "
                    "injected view) -- e.g. a second engine LM, or an LM critic under "
                    "token-grain in-burst resampling",
                )
    if len(controller.samplers) > 1:
        # The burst serves one twist view per group, present for all or none; a mixed
        # batch runs the exact per-token loop (which handles it per-row).
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
        if not s0.burst_routes_groups() and constraint_leaf_ids(s.target) != constraint0:
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
        self.n_bursts = 0  # bursts opened -- for verifying the burst path ran
        # views: LM leaves whose warm logits the burst injects (group 0's target+proposal,
        # plus the critic's engine leaf when boundary twisting will read it); the batched
        # burst draws every group through group 0's sampler.
        serve = critic_deferred(self.sampler, controller) and controller.twist_with_critic
        self.twist_leaves = [
            find_engine_lm(c) if (serve and c is not None) else None
            for c in controller.critics
        ]
        self.twist_view = self.twist_leaves[0]
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

    def defers_critic(self, controller):
        # A non-deferred critic (free-running in-burst resampling) scores inline per
        # step in ``bank_row`` — engine-free there by ``burst_blocker``, so the inline
        # await cannot deadlock.
        return critic_deferred(self.sampler, controller)

    async def round(self, c):
        """One burst over the live rows: a whole generation at token grain (resampling
        in place at ESS crossings), one synced unit per row at unit grain. Runs the
        engine decode loop in a worker thread; each step's draw hops back to this loop
        (parked in ``run_in_executor``) via ``run_coroutine_threadsafe`` (see
        ``_Burst.draw``)."""
        loop = self.main_loop = asyncio.get_running_loop()
        self.n_bursts += 1
        live = [p for p in c.particles if not p.done]
        b = _Burst(self, live)
        max_steps = self.sampler.burst_max_steps(live)
        await loop.run_in_executor(
            None,
            lambda: self.llm.model.run_burst(control=b, max_steps=max_steps),
        )

    async def run(self):
        return await self.controller.run(self)
