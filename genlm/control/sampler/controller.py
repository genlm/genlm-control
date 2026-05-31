"""The SMC controller: the single owner of the SMC algorithm (population, per-step
transition, ESS, resample/fork, log_ml).

Two drivers turn the population: ``StepLoop`` (per-token, the byte-exact ground
truth) and ``BurstLoop`` (engine-accelerated; bursts expressible steps). Resample/ESS/
log_ml are always controller-owned, never delegated to the engine.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
import torch

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.util import logsumexp
from genlm.control.potential.built_in.llm import (
    find_engine_lm,
    constraint_leaf_ids,
    lm_leaves,
    factor_leaves,
)
from genlm.control.sampler.resampling import get_resampling_fn
from genlm.control.sampler.smc_record import SMCRecord, string_for_serialization


class NotAcceleratable(Exception):
    """Engine acceleration was required but the config can't be driven by the burst.
    Message is ``burst_blocker``'s reason."""


# ---------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------


class Population:
    """The SMC particle population, stored columnar: bulk scalars (``logw``,
    ``logp``, ``twist_amount``, ``done``, ``max_tokens_left``) are parallel numpy
    arrays; ``contexts`` are Python lists. Indexing/iterating yields :class:`Particle`
    views onto a row (no second copy to sync)."""

    __slots__ = (
        "n",
        "logw",
        "logp",
        "twist_amount",
        "done",
        "max_tokens_left",
        "contexts",
        "group",
        "_views",
    )

    def __init__(self, n, max_tokens, group=None):
        self.n = n
        # Sub-population id per row (batched runs); ESS/resample/log_ml are per-group.
        self.group = (
            np.zeros(n, dtype=np.int64) if group is None
            else np.asarray(group, dtype=np.int64)
        )
        self.logw = np.zeros(n)
        self.logp = np.zeros(n)
        self.twist_amount = np.zeros(n)
        self.done = np.zeros(n, dtype=bool)
        self.max_tokens_left = np.full(n, max_tokens, dtype=np.int64)
        self.contexts = [[] for _ in range(n)]
        # Row views built once and reused (a view is a stateless (pop, row) pointer;
        # reindex mutates the arrays in place, so row i staying row i keeps them valid).
        self._views = [Particle(self, i) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self._views[i]

    def __iter__(self):
        return iter(self._views)

    def untwist_all(self):
        """Vectorized untwist over the whole population (done rows carry 0)."""
        self.logw -= self.twist_amount
        self.twist_amount[:] = 0.0

    def untwist_subset(self, idx):
        """Vectorized untwist of rows ``idx`` (the burst's live rows; ``idx`` distinct)."""
        self.logw[idx] -= self.twist_amount[idx]
        self.twist_amount[idx] = 0.0

    def reindex(self, ancestor_indices):
        """Resample/fork: reindex every column by ``ancestor_indices`` (contexts
        shallow-copied; their token elements are immutable)."""
        idx = ancestor_indices
        self.logw = self.logw[idx]
        self.logp = self.logp[idx]
        self.twist_amount = self.twist_amount[idx]
        self.done = self.done[idx]
        self.max_tokens_left = self.max_tokens_left[idx]
        self.contexts = [list(self.contexts[i]) for i in idx]
        self.group = self.group[idx]


class Particle:
    """A thin view onto one row of a :class:`Population`; scalar reads/writes go
    through to the population arrays (no second copy to sync)."""

    __slots__ = ("_pop", "_i")

    def __init__(self, pop, i):
        self._pop = pop
        self._i = i

    @property
    def logw(self):
        return self._pop.logw[self._i]

    @logw.setter
    def logw(self, v):
        self._pop.logw[self._i] = v

    @property
    def logp(self):
        return self._pop.logp[self._i]

    @logp.setter
    def logp(self, v):
        self._pop.logp[self._i] = v

    @property
    def twist_amount(self):
        return self._pop.twist_amount[self._i]

    @property
    def done(self):
        return self._pop.done[self._i]

    @property
    def max_tokens_left(self):
        return self._pop.max_tokens_left[self._i]

    @max_tokens_left.setter
    def max_tokens_left(self, v):
        self._pop.max_tokens_left[self._i] = v

    @property
    def context(self):
        return self._pop.contexts[self._i]

    # -- weight bookkeeping --

    def score(self, amt):
        self._pop.logw[self._i] += amt

    def twist(self, amt):
        self._pop.twist_amount[self._i] += amt
        self._pop.logw[self._i] += amt

    def untwist(self):
        self._pop.logw[self._i] -= self._pop.twist_amount[self._i]
        self._pop.twist_amount[self._i] = 0.0

    def finish(self):
        self.untwist()
        self._pop.done[self._i] = True

    # -- viz adapter (the record reads .weight + .string_for_serialization()) --

    @property
    def weight(self):
        return self._pop.logw[self._i]

    def string_for_serialization(self):
        return string_for_serialization(self._pop.contexts[self._i])


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
        self._precomputed_factor = {}  # row -> {factor_leaf: LazyWeights}, pre-computed in the bank

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
            for h, row_w in zip(vh, batch.cpu().numpy()):
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
                {
                    **{
                        view: warm[self.row_handles[row][vi]]
                        for vi, view in enumerate(self.views)
                    },
                    **self._precomputed_factor.get(row, {}),  # factor pre-computed in the bank
                }
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
        ``p.done``) and, for free running, pre-compute the next step's constraint factors."""
        c = self.d.controller
        await c._bank_burst_steps(parts, records)
        if not self.d.sampler.burst_free_running():
            return  # unit grain banks inline; BurstLoop.run records, no overlap to prep
        # Token grain records per step here; unit grain once per round in BurstLoop.run.
        if any(r.step is not None for r in records):
            c._record_step()
        # Pre-compute the constraint factors at the live rows' new context so the next
        # draw's Product.logw_next serves them from the injection (overlaps the next forward).
        leaves = self.d.factor_leaves
        live = [p for p in parts if not p.done]
        if leaves and live:
            results = await asyncio.gather(
                *(leaf.logw_next(p.context) for p in live for leaf in leaves)
            )
            n = len(leaves)
            for i, p in enumerate(live):
                self._precomputed_factor[p._i] = dict(
                    zip(leaves, results[i * n : (i + 1) * n])
                )

    def _drop_row(self, row):
        """Fully evict a row's K substreams: abort their engine requests, drop both handle
        maps, and drop any pre-computed factor."""
        self._precomputed_factor.pop(row, None)
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


# ---------------------------------------------------------------------------
# The transition (shared by all samplers)
# ---------------------------------------------------------------------------


class Controller:
    """Owns the SMC algorithm: population, transition, ESS, resample, log_ml. A
    sampler collapses to one per-step ``transition``; no sampler reimplements the loop.

    Args:
        unit_sampler (TokenSampler): produces ``(to_append, logw, logp)`` per step.
        critic (Potential, optional): reweights/twists particles.
        n_particles (int): number of particles.
        ess_threshold (float): ESS fraction below which we resample.
        max_tokens (int): per-particle token budget.
        twist_with_critic (bool): whether the critic twists during stepping.
        resampling_method (str): multinomial/stratified/systematic/residual.
        record (bool): build an :class:`SMCRecord`.
        verbosity (int): 0 silent, 1 prints particles per step.
    """

    def __init__(
        self,
        unit_sampler,
        critic,
        n_particles,
        ess_threshold,
        max_tokens,
        twist_with_critic,
        resampling_method="multinomial",
        record=False,
        verbosity=0,
        group_sizes=None,
        samplers=None,
        critics=None,
    ):
        assert max_tokens > 0
        # Batched runs: B groups in one population; default is a single group.
        if group_sizes is None:
            group_sizes = [n_particles]
            samplers = [unit_sampler]
            critics = [critic]
        assert n_particles == sum(group_sizes), "n_particles must be the TOTAL row count"
        assert len(samplers) == len(critics) == len(group_sizes)

        self.samplers = samplers
        self.critics = critics
        self.group_sizes = group_sizes
        # Representatives for the capability check / BurstLoop (groups share structure).
        self.unit_sampler = samplers[0]
        self.critic = critics[0]
        self.n_particles = n_particles
        self.n_resamples = 0  # resample events (any lane); guards verify the resample path fired
        self.ess_threshold = ess_threshold
        self.max_tokens = max_tokens
        self.twist_with_critic = twist_with_critic
        # A terminal-only critic carries no per-step signal, so reweight only at
        # termination (skips the per-step critic call; same result).
        if twist_with_critic and all(
            c is not None and c.is_terminal_only() for c in critics
        ):
            self.twist_with_critic = False
        self.resample_fn = get_resampling_fn(resampling_method)
        self.verbosity = verbosity

        group = np.concatenate(
            [np.full(ng, g, dtype=np.int64) for g, ng in enumerate(group_sizes)]
        )
        self.particles = Population(n_particles, max_tokens, group=group)
        # Per-group row indices (invariant across reindex; resample is group-local).
        self._group_rows = [
            np.nonzero(self.particles.group == g)[0] for g in range(len(group_sizes))
        ]
        self.record = SMCRecord(n_particles) if record else None
        # Resample tag: ``_maybe_resample`` sets it so the next ``_record_step`` is
        # tagged ``add_resample`` (lazy tag-at-next-step).
        self._pending_resample = False
        self._pending_ancestors = list(range(n_particles))

        # log(ess_threshold); the per-group ESS test adds log(group_size).
        with np.errstate(divide="ignore"):
            self._log_ess_threshold = np.log(ess_threshold)

    # -- the per-step transition for ONE particle --

    async def _draw_and_score(self, p):
        """Slow per-step transition: draw from the sampler, then apply the shared
        score/twist/terminate math."""
        to_append, logw, logp = await self._sampler_of(p).transition(p.context)
        await self._score_advance_terminate(p, to_append, logw, logp)

    def _sampler_of(self, p):
        """The sampler for particle ``p``'s group."""
        return self.samplers[self.particles.group[p._i]]

    def _critic_of(self, p):
        """The critic for particle ``p``'s group."""
        return self.critics[self.particles.group[p._i]]

    def _advance_no_critic(self, p, to_append, logw, logp):
        """Sync score + advance + terminate for the no-critic path (shared by the slow
        path and the burst)."""
        p.score(logw)
        p.logp += logp
        p.context.extend(to_append)
        if p.logw == float("-inf"):
            p.finish()
            return
        if self.verbosity > 0:
            print(self._repr_particle(p))
        p.max_tokens_left -= 1
        if p.max_tokens_left == 0 or self._is_terminal(p):
            p.finish()

    async def _score_advance_terminate(self, p, to_append, logw, logp):
        """The post-draw SMC math (one implementation, shared by both lanes): score,
        advance the context, critic-twist per step, reweight + terminate. The caller
        untwists ``p`` before the draw."""
        if not self.critic:
            self._advance_no_critic(p, to_append, logw, logp)
            return

        p.score(logw)
        p.logp += logp
        p.context.extend(to_append)

        if p.logw == float("-inf"):
            assert p.twist_amount != float("-inf")
            p.finish()
            return

        if self.twist_with_critic:
            twist_amt = await self._critic_of(p).score(p.context)
            if twist_amt != float("-inf"):
                p.twist(twist_amt)
            else:
                p.score(twist_amt)
                p.finish()
                return

        if self.verbosity > 0:
            print(self._repr_particle(p))

        p.max_tokens_left -= 1
        if p.max_tokens_left == 0 or self._is_terminal(p):
            p.finish()
            if not self.twist_with_critic:
                twist_amt = await self._critic_of(p).score(p.context)
            p.score(twist_amt)
            return

    def _is_terminal(self, p):
        return bool(p.context) and p.context[-1] is EOS

    def _repr_particle(self, p):
        from arsenal import colors
        from genlm.control.util import escape

        return (
            f"{p.logw:.2f}:\t"
            + colors.magenta % "["
            + (colors.magenta % "|").join(escape(y) for y in p.context)
            + colors.magenta % "]"
        )

    # -- controller-owned SMC primitives the drivers turn (start, per-step banking,
    #    ESS+resample, record); the per-step loop itself lives in the drivers ----------

    async def start(self):
        """Score every particle by its group's empty-sequence prefix weight."""
        start_ws = [await s.start_weight() for s in self.samplers]
        for g, start_w in enumerate(start_ws):
            if start_w == float("-inf"):
                raise ValueError(
                    f"Start weight is -inf (log(0)) for group {g}. This is likely "
                    "because a potential assigns zero weight to the empty sequence "
                    "under `prefix`, which violates the potential contract."
                )
        for p in self.particles:
            p.score(start_ws[self.particles.group[p._i]])

    def _maybe_resample(self):
        """Per-group ESS test + resample (the only ESS/resample impl; both lanes call
        it). Group-local: each crossing group reindexes within its own rows. Mutates
        ``self.particles`` on a resample. Returns ``(crossing_groups, ancestors)`` --
        ``([], None)`` when nothing crossed (no global vector built)."""
        W = self.particles.logw
        crossings = []  # (g, rows, local_ancestors, reset_logw) per crossing group
        for g, rows in enumerate(self._group_rows):
            Wg = W[rows]
            if np.all(Wg == -np.inf):
                continue
            w_sum = logsumexp(Wg)
            nw = Wg - w_sum
            if -logsumexp(nw * 2) < self._log_ess_threshold + np.log(len(rows)):
                local = np.asarray(self.resample_fn(np.exp(nw)))  # ancestors in 0..ng-1
                if self.record is not None:
                    local = np.sort(local)  # only matters for a reproducible record
                crossings.append((g, rows, local, w_sum - np.log(len(rows))))

        if not crossings:
            return [], None  # no-cross early-out: no arange / reindex / tolist

        ancestors = np.arange(self.n_particles)
        for _g, rows, local, _t in crossings:
            ancestors[rows] = rows[local]  # group-local -> global rows
        self.n_resamples += 1
        self.particles.reindex(ancestors)
        for _g, rows, _l, target in crossings:
            self.particles.logw[rows] = target
        ancestors = ancestors.tolist()
        if self.record is not None:
            # Tag the NEXT recorded step as a resample (lazy tag-at-next-step,
            # consumed by ``_record_step``).
            self._pending_resample = True
            self._pending_ancestors = ancestors
        return [c[0] for c in crossings], ancestors

    def save_record(self, json_path):
        """Write the SMC record JSON."""
        if self.record is None:
            return
        with open(json_path, "w") as f:
            f.write(self.record.to_json())
        print(f"Saved record to {json_path}")

    def _record_step(self):
        """Record one completed SMC step (lane-neutral): ``add_init`` first,
        ``add_resample`` if a resample preceded it, else ``add_smc_step``. No-op
        without a record."""
        if self.record is None:
            return
        if len(self.record.history) == 0:
            self.record.add_init(self.particles)
        elif self._pending_resample:
            self.record.add_resample(self._pending_ancestors, self.particles)
        else:
            self.record.add_smc_step(self.particles)
        self._pending_resample = False

    # -- burst-lane math the engine seam (_Burst) calls into --
    #
    # The seam (draw / drain_* / token-id mapping) lives on _Burst; what stays here is
    # the SMC math it banks into: per-step banking + ESS/resample.

    async def _bank_burst_steps(self, parts, records):
        """Bank every row's completed SMC step (``rec.step``), same math as the slow
        loop. Runs on the main loop (hopped from the burst worker), so an awaiting
        critic composes as a normal gather."""
        steps = [
            (p, rec.step) for p, rec in zip(parts, records) if rec.step is not None
        ]
        if not steps:
            return
        if self.critic is None:
            for p, (to_append, logw, logp) in steps:
                self._advance_no_critic(p, to_append, logw, logp)
            return
        for p, (to_append, logw, logp) in steps:
            await self._score_advance_terminate(p, to_append, logw, logp)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


class StepLoop:
    """Per-token round-trip driver (the byte-exact ground truth); recomputes logprobs
    from the full context every step."""

    def __init__(self, controller):
        self.controller = controller

    async def run(self):
        """Turn the population per-token to completion: each step draws + scores every
        live particle (concurrently), records the step, then runs the controller-owned
        ESS/resample. Symmetric with ``BurstLoop.run`` -- the controller owns the SMC
        math, the driver owns the loop."""
        c = self.controller
        await c.start()
        while any(not p.done for p in c.particles):
            if c.twist_with_critic:
                c.particles.untwist_all()
            await asyncio.gather(
                *[c._draw_and_score(p) for p in c.particles if not p.done]
            )
            c._record_step()
            c._maybe_resample()
        return c.particles


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


def _factor_leaves_of(sampler):
    """The non-LM constraint leaves a burst can pre-compute for ``sampler`` -- the
    non-LM leaves of the draw sampler's target (+ proposal), deduped. Empty for an
    unconstrained target; pre-computing them overlaps the engine forward."""
    s = sampler.burst_draw_sampler()
    leaves, seen = [], set()
    for pot in (s.target, s.proposal):
        if pot is None:
            continue
        for leaf in factor_leaves(pot):
            if id(leaf) not in seen:
                seen.add(id(leaf))
                leaves.append(leaf)
    return leaves


def burst_blocker(controller):
    """Why this config can't run the engine burst, or ``None`` if it can. Needs a
    burst-capable sampler over a target with one engine-burst LM leaf, and must be
    forward-free: every LM leaf in each group's per-step path (target + proposal + critic)
    must be an injected view. A batched (B>1) population must be burst-homogeneous
    (:func:`_batch_blocker`)."""
    s = controller.unit_sampler
    if not s.supports_burst():
        return f"{type(s).__name__} does not support the engine burst"
    if find_engine_lm(s.target) is None:
        return "sampler target has no single engine-burst LM leaf"
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
                return (
                    f"group {g}: an LM leaf can't be assembled into the burst as an "
                    "injected view (it would forward) -- e.g. an LM critic or a "
                    "second engine LM"
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
    for g, s in enumerate(samplers[1:], start=1):
        if type(s) is not type(s0):
            return f"group {g} sampler is {type(s).__name__}, not {type(s0).__name__}"
        views = _views_of(s)
        if len(views) != len(views0):
            return f"group {g} has {len(views)} views, not {len(views0)}"
        for vi, (v, v0) in enumerate(zip(views, views0)):
            if v is None or v.model is not v0.model:
                return f"group {g} view {vi} uses a different engine"
            if getattr(v, "temperature", None) != getattr(v0, "temperature", None):
                return f"group {g} view {vi} temperature differs from group 0"
            if v.lora_name != v0.lora_name:
                return f"group {g} view {vi} uses a different LoRA adapter from group 0"
        if constraint_leaf_ids(s.target) != constraint0:
            return f"group {g} has a different constraint"
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
        # Non-LM constraint leaves to pre-compute during the forward. Empty == unconstrained.
        self.factor_leaves = _factor_leaves_of(self.sampler)
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
