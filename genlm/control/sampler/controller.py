"""The SMC controller: the single owner of the SMC algorithm (population, per-step
transition, ESS, resample/fork, log_ml).

Two drivers turn the population: ``StepLoop`` (per-token, the byte-exact ground
truth) and ``BurstLoop`` (engine-accelerated; bursts expressible steps, drops to the
slow lane for a ``slow_cadence`` step). Resample/ESS/log_ml are always
controller-owned, never delegated to the engine.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
import torch

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.util import logsumexp
from genlm.control.potential.built_in.llm import find_engine_lm, constraint_leaf_ids
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
        "pop_anchor",
        "group",
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
        # Context length at the last slow-lane step; the cadence predicate sees
        # ``context[pop_anchor:]`` as its buffer, ``context[:pop_anchor]`` as context.
        self.pop_anchor = np.zeros(n, dtype=np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return Particle(self, i)

    def __iter__(self):
        return (Particle(self, i) for i in range(self.n))

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
        self.pop_anchor = self.pop_anchor[idx]
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

    # -- slow-lane cadence run buffer (see Population.pop_anchor) --

    @property
    def run_prefix(self):
        """Tokens before the current run -- the cadence predicate's ``unit_context``."""
        return self._pop.contexts[self._i][: self._pop.pop_anchor[self._i]]

    @property
    def run_buffer(self):
        """Tokens drawn since the last slow step -- the cadence predicate's buffer."""
        return self._pop.contexts[self._i][self._pop.pop_anchor[self._i] :]

    def reset_run(self):
        """Close the current run (anchor at context end) so the cadence predicate re-arms."""
        self._pop.pop_anchor[self._i] = len(self._pop.contexts[self._i])

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
_EXIT_SLOW_STEP = "slow_step"  # next step is inexpressible; driver runs it slow


class _Burst:
    """Per-burst engine state, set on the Controller for one ``run_burst``: the live
    row<->particle handle maps, the abort/re-add queues, the sampler scratch, and the
    exit reason. Run config (the engine LM, eos, the event-loop hop) is borrowed from
    the ``BurstLoop`` ``d``. The sampler-facing surface is ``engine_lm``/``warm_logws``
    (inject + draw) and ``scratch`` (unit accumulation); ``draw``/``drain_*``/
    ``run_sync`` are the internal engine seam."""

    def __init__(self, d, live):
        self.d = d
        self.particles = live
        self.abort_rows = set()
        self.add_rows = []
        # handle <-> population row. The engine hands ``draw`` the handle owning each
        # row; ``next_handle`` issues a fresh one per mid-burst re-add.
        self.handle_to_row = {i: p._i for i, p in enumerate(live)}
        self.row_to_handle = {p._i: i for i, p in enumerate(live)}
        self.next_handle = len(live)
        self.scratch = {}
        self.exit_reason = _EXIT_TERMINATED

    @property
    def engine_lm(self):
        return self.d.llm

    def warm_logws(self, logits):
        """The engine's warm logits -> one per-row LazyWeights == the engine LM's
        ``logw_next`` for each row's context (folded once on-device, transferred once).
        The sampler injects these as the LM's ``logw_next`` and runs the real draw."""
        llm = self.d.llm
        batch = llm._process_logw_next_batch(llm._maybe_temper(logits.float()))
        return [llm.make_lazy_weights(row) for row in batch.cpu().numpy()]

    def run_sync(self, coro):
        """Run an async helper to completion on the driver loop (one hop)."""
        return self.d.run_async(coro)

    def particle_of(self, handle):
        return self.d.controller.particles[self.handle_to_row[handle]]

    def context_ids(self, p):
        """Engine prompt: the row's group prefix + its drawn token ids (EOS dropped)."""
        ids = list(self.d.group_prefixes[self.d.controller.particles.group[p._i]])

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

    def drain_aborts(self):
        rows = self.abort_rows
        self.abort_rows = set()
        return list(rows)

    def drain_adds(self):
        rows = self.add_rows
        self.add_rows = []
        return rows

    def draw(self, logits, handles):
        """The engine callback (``run_burst`` calls this), one token per live row: the
        engine's warm logits become each row's engine-LM ``logw_next``, the sampler runs
        its REAL ``burst_draw_batch`` (inject + ``transition``), and the Controller banks
        the completed SMC steps. ``handles[i]`` owns row ``i``. Pop-out is out-of-band
        (``drain_aborts``); a token-grain ESS crossing resamples in place (no exit)."""
        c = self.d.controller
        sampler = self.d.sampler
        # Every row here is live (aborted/terminated rows were dropped from the engine).
        parts = [self.particle_of(h) for h in handles]

        # Untwist last step's critic twist before this step's score (no-op w/o critic).
        c.particles.untwist_subset([p._i for p in parts])

        # Engine warm logits -> per-row LM logw_next; the sampler injects them and runs
        # the real per-step draw. ONE event-loop hop for the whole batch.
        warm = self.warm_logws(logits)
        records = self.run_sync(
            sampler.burst_draw_batch(warm, [p.context for p in parts], handles, self)
        )
        # Bank steps before the token map: scoring sets p.done on termination.
        c._bank_burst_steps(parts, records, self.run_sync)
        # Token grain records per step here; unit grain once per round in BurstLoop.run.
        if sampler.burst_free_running() and any(rec.step is not None for rec in records):
            c._record_step()
        out = [0] * len(parts)
        for k, p in enumerate(parts):
            out[k] = self._burst_token_id(p, records[k])
            if p.done or records[k].pop:
                # Drop the row: terminated, or waiting at a unit boundary.
                self.abort_rows.add(handles[k])
                if p.done:
                    self.handle_to_row.pop(handles[k], None)
                    self.row_to_handle.pop(p._i, None)

        if sampler.burst_free_running() and c._ess_crosses():
            self.resample_realize()  # token-grain crossing: resample in place, no exit
        elif c._slow_cadence_due(parts):
            self.abort_rows.update(handles)  # next step is slow-lane: pop all
            self.exit_reason = _EXIT_SLOW_STEP

        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def _burst_token_id(self, p, rec):
        """Map a banked row's drawn token to the engine token id (after banking set
        ``p.done`` on termination). EOS has no token_id -> the eos placeholder."""
        token = rec.token
        if isinstance(token, EndOfSequence):
            assert p.done, "burst drew EOS for a particle that did not terminate"
            return self.d.eos_id
        return token.token_id

    def resample_realize(self):
        """Translate a completed per-group resample into engine abort/re-add: drop the
        crossing groups' current rows, re-add their still-live rows at the resampled
        context (fresh handle). ESS/resample stay Controller-owned (``_maybe_resample``)."""
        c = self.d.controller
        c._maybe_resample()
        for g in c._last_resampled_groups:
            rows = c._group_rows[g]
            for row in rows:
                h = self.row_to_handle.pop(int(row), None)
                if h is not None:
                    self.abort_rows.add(h)
                    self.handle_to_row.pop(h, None)
            for row in rows:
                p = c.particles[int(row)]
                if p.done:
                    continue
                h = self.next_handle
                self.next_handle += 1
                self.handle_to_row[h] = int(row)
                self.row_to_handle[int(row)] = h
                self.add_rows.append((h, self.context_ids(p)))


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
        slow_cadence (BoundaryPredicate, optional): marks steps the burst must run on
            the slow lane (evaluated per row on its run buffer). Performance hint only;
            the per-step math is identical on both lanes. ``None`` = burst every step.
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
        slow_cadence=None,
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
        self.slow_cadence = slow_cadence

        group = np.concatenate(
            [np.full(ng, g, dtype=np.int64) for g, ng in enumerate(group_sizes)]
        )
        self.particles = Population(n_particles, max_tokens, group=group)
        # Per-group row indices (invariant across reindex; resample is group-local).
        self._group_rows = [
            np.nonzero(self.particles.group == g)[0] for g in range(len(group_sizes))
        ]
        # Group indices the last ``_maybe_resample`` resampled (read by the burst).
        self._last_resampled_groups = []
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
        path and the burst; no event-loop hop)."""
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

    # -- the controller-owned loop --------------------------------------------------

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

    async def run(self):
        """Run the slow (ground-truth) SMC loop to completion."""
        while any(not p.done for p in self.particles):
            self.particles.untwist_all()

            await asyncio.gather(
                *[self._draw_and_score(p) for p in self.particles if not p.done]
            )

            self._record_step()
            self._maybe_resample()

        return self.particles

    def _ess_below_threshold(self, normalized_weights, ng):
        """Per-group ESS resample predicate (ESS_g < ess_threshold * ng); shared by
        ``_ess_crosses`` and ``_maybe_resample``."""
        return -logsumexp(normalized_weights * 2) < self._log_ess_threshold + np.log(ng)

    def _ess_crosses(self):
        """Whether the ESS test triggers a resample for any group (test only, no
        mutation) -- the burst's pop-out trigger."""
        W = self.particles.logw
        for rows in self._group_rows:
            Wg = W[rows]
            if np.all(Wg == -np.inf):
                continue
            normalized_weights = Wg - logsumexp(Wg)
            if self._ess_below_threshold(normalized_weights, len(rows)):
                return True
        return False

    def _maybe_resample(self):
        """Per-group ESS test + resample (the only ESS/resample impl; both lanes call
        it). Group-local: each crossing group reindexes within its own rows. Mutates
        ``self.particles`` on a resample. Returns ``(did_resample, ancestor_indices)``."""
        n = self.n_particles
        sort_ancestors = self.record is not None  # only matters for a reproducible record
        W = self.particles.logw
        global_ancestors = np.arange(n)
        resets = []  # (rows, target_logw) for the groups that resampled
        resampled_groups = []  # group indices that crossed (for the in-place handler)

        for g, rows in enumerate(self._group_rows):
            Wg = W[rows]
            if np.all(Wg == -np.inf):
                continue
            w_sum = logsumexp(Wg)
            normalized_weights = Wg - w_sum
            if self._ess_below_threshold(normalized_weights, len(rows)):
                probs = np.exp(normalized_weights)
                local = np.asarray(self.resample_fn(probs))  # ancestors in 0..ng-1
                if sort_ancestors:
                    local = np.sort(local)
                global_ancestors[rows] = rows[local]  # map group-local -> global rows
                resets.append((rows, w_sum - np.log(len(rows))))
                resampled_groups.append(g)

        self._last_resampled_groups = resampled_groups
        if resampled_groups:
            self.particles.reindex(global_ancestors)
            for rows, target in resets:
                self.particles.logw[rows] = target
            if self.record is not None:
                # Tag the NEXT recorded step as a resample (the lazy tag-at-next-step
                # the record cadence relies on); consumed by ``_record_step``.
                self._pending_resample = True
                self._pending_ancestors = global_ancestors.tolist()

        return bool(resampled_groups), global_ancestors.tolist()

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
    # the SMC math it banks into: per-step banking, the cadence test, ESS/resample.

    def _slow_cadence_due(self, parts):
        """Whether the next step is a slow-lane cadence for any still-live row
        (mid-burst check). When it fires, the driver runs one slow-lane step for the
        whole live population."""
        return self._cadence_due(p for p in parts if not p.done)

    def _cadence_due(self, live):
        """Boundary-predicate cadence test over ``live`` (shared by the mid-burst and
        burst-entry checks). ``False`` without a cadence."""
        if self.slow_cadence is None:
            return False
        return any(self.slow_cadence(p.run_prefix, p.run_buffer) for p in live)

    def _bank_burst_steps(self, parts, records, run_sync):
        """Bank every row's completed SMC step (``rec.step``), same math as the slow
        loop. No-critic: sync, no hop. With a critic: all rows gathered into ONE
        ``run_sync`` hop (passed in by the seam) so the critic LM autobatches."""
        steps = [
            (p, rec.step) for p, rec in zip(parts, records) if rec.step is not None
        ]
        if not steps:
            return
        if self.critic is None:
            for p, (to_append, logw, logp) in steps:
                self._advance_no_critic(p, to_append, logw, logp)
            return

        async def _score_all():
            await asyncio.gather(
                *(
                    self._score_advance_terminate(p, to_append, logw, logp)
                    for p, (to_append, logw, logp) in steps
                )
            )

        run_sync(_score_all())


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


class StepLoop:
    """Per-token round-trip driver (the ground-truth path); recomputes logprobs from
    the full context every step."""

    def __init__(self, controller):
        self.controller = controller

    async def run(self):
        await self.controller.start()
        return await self.controller.run()


def burst_blocker(controller):
    """Why this config can't run the engine burst, or ``None`` if it can. The burst
    needs a burst-capable sampler (``supports_burst()``) over a target with exactly
    one engine-burst LM leaf (:func:`find_engine_lm`). A critic does not disqualify
    it. A batched (B>1) population must be burst-homogeneous (:func:`_batch_blocker`).
    ``auto`` falls back to ``StepLoop`` on a non-None reason; ``require`` raises it."""
    s = controller.unit_sampler
    if not s.supports_burst():
        return f"{type(s).__name__} does not support the engine burst"
    if find_engine_lm(s.target) is None:
        return "sampler target has no single engine-burst LM leaf"
    if len(controller.samplers) > 1:
        return _batch_blocker(controller.samplers)
    return None


def _batch_blocker(samplers):
    """Why a batched burst can't draw every group through group 0's sampler, or
    ``None`` if burst-homogeneous. The burst runs one engine forward over all B*N rows
    and draws them with ``samplers[0]``, so groups must share sampler kind, engine
    model, temperature, and constraint -- they may differ only in prompt (the warm
    logits carry it) and critic (scored per group)."""
    s0 = samplers[0]
    lm0 = find_engine_lm(s0.target)
    constraint0 = constraint_leaf_ids(s0.target)
    for g, s in enumerate(samplers[1:], start=1):
        if type(s) is not type(s0):
            return f"group {g} sampler is {type(s).__name__}, not {type(s0).__name__}"
        lm = find_engine_lm(s.target)
        if lm is None or lm.model is not lm0.model:
            return f"group {g} uses a different engine"
        if getattr(lm, "temperature", None) != getattr(lm0, "temperature", None):
            return f"group {g} temperature differs from group 0"
        if constraint_leaf_ids(s.target) != constraint0:
            return f"group {g} has a different constraint"
    return None


class BurstLoop:
    """The engine-accelerated SMC driver. Runs the next step on the fast lane (an
    engine burst over the live contexts) when it's expressible, dropping to the slow
    lane for a ``slow_cadence`` step. The burst never resamples -- it pops at a tagged
    boundary and :meth:`run` dispatches the controller-owned resample. :class:`StepLoop`
    is the all-slow degenerate case. Only valid when :func:`burst_blocker` is ``None``."""

    def __init__(self, controller):
        self.controller = controller
        # Counts (bursts opened, slow-lane steps) -- for verifying which paths ran.
        self.n_bursts = 0
        self.n_slow_steps = 0
        self.sampler = controller.unit_sampler
        # The single engine LM the burst drives (run_burst + warm-logits injection).
        self.llm = find_engine_lm(self.sampler.target)
        if self.llm is None:  # pragma: no cover - guarded by burst_blocker
            raise ValueError("sampler target has no single engine-burst LM leaf")

        # Groups share one engine, differing only in ``prompt_ids``. Snapshot each
        # prefix on the main thread: ``prompt_ids`` is a ContextVar, invisible on the
        # ``run_burst`` worker thread where the mid-burst re-add runs.
        llms = [find_engine_lm(s.target) for s in controller.samplers]
        self.group_prefixes = [list(lm.prompt_ids) for lm in llms]

        # Engine token id committed as the placeholder for an aborted/EOS row.
        eos_idxs = list(self.llm.token_maps.eos_idxs)
        if not eos_idxs:
            raise ValueError(
                'Engine LM has no EOS token id; the burst needs one. Use accelerate="off".'
            )
        self.eos_id = eos_idxs[0]
        self.run_async = None  # the worker-thread -> loop hop, bound in run()

    async def _run_burst(self, live, loop):
        """Run one stateless burst over ``live`` to a pop-out; return its ``_EXIT_*``
        reason. Runs the engine decode loop in a worker thread so this loop stays free
        for the ``draw`` hops."""
        self.n_bursts += 1
        b = _Burst(self, live)
        # Unit grain hands back at the synced boundary (fixed reason); token grain
        # stays terminated, resampling in place at ESS crossings.
        b.exit_reason = (
            _EXIT_TERMINATED if self.sampler.burst_free_running() else _EXIT_UNIT_SYNC
        )
        prompts = [b.context_ids(p) for p in live]
        # Decode budget for one burst (token grain: longest token budget; unit grain:
        # one unit's subunits).
        max_steps = self.sampler.burst_max_steps(live)
        # The _Burst IS the engine seam (draw / drain_* the engine calls back into);
        # the Controller stays pure SMC math that the seam banks into.
        await loop.run_in_executor(
            None,
            lambda: self.llm.model.run_burst(
                prompts=prompts,
                control=b,
                max_steps=max_steps,
            ),
        )
        return b.exit_reason

    async def run(self):
        """The outer driver: each iteration run the live rows' next step on the slow
        lane (a ``slow_cadence`` step) or the fast lane (a burst), then dispatch the
        controller-owned ESS/resample. Repeat until every particle is done."""
        controller = self.controller
        await controller.start()

        loop = asyncio.get_running_loop()
        self.run_async = lambda coro: asyncio.run_coroutine_threadsafe(coro, loop).result()

        while any(not p.done for p in controller.particles):
            live = [p for p in controller.particles if not p.done]
            if self._next_step_is_slow(live):
                # Next step is inexpressible -- run it slow, then ESS.
                await self._slow_step(live)
                controller._maybe_resample()
                continue

            reason = await self._run_burst(live, loop)
            # Resample at a burst exit (unit-grain boundary, or before a cadence step).
            # A token-grain ESS crossing resamples in place, no exit.
            if reason == _EXIT_UNIT_SYNC:
                # One record entry for the completed unit round, before its resample.
                # Not on _EXIT_SLOW_STEP (recorded by the next _slow_step).
                controller._record_step()
            if reason in (_EXIT_UNIT_SYNC, _EXIT_SLOW_STEP):
                controller._maybe_resample()

        return controller.particles

    def _next_step_is_slow(self, live):
        """Whether any live row's next step is a slow-lane cadence (the burst-entry
        check; same test as mid-burst)."""
        return self.controller._cadence_due(live)

    async def _slow_step(self, live):
        """Run one slow-lane transition for the live rows (the inexpressible step,
        engine free): untwist, the shared per-step transition, then ``reset_run`` so
        the cadence predicate re-arms (next run starts empty)."""
        controller = self.controller
        self.n_slow_steps += 1
        controller.particles.untwist_subset([p._i for p in live])
        await asyncio.gather(*[controller._draw_and_score(p) for p in live])
        controller._record_step()
        for p in live:
            p.reset_run()
