"""SMC population and the ``Controller``: algorithm owner and the per-group run
loop. Engine acceleration is whether rows hold lanes (``lane_seam``); the loop is
the same either way."""

import asyncio
import contextlib

import numpy as np
from arsenal import colors

from genlm.control.constant import EOS
from genlm.control.potential.built_in.llm import find_engine_lm
from genlm.control.util import logsumexp, draw_key, draw_ordinal, escape
from genlm.control.sampler.resampling import get_resampling_fn
from genlm.control.sampler.smc_record import SMCRecord


class Population:
    """Columnar SMC particle store: scalars are parallel numpy arrays, ``contexts``
    are Python lists. Indexing yields :class:`Particle` row views."""

    __slots__ = (
        "n",
        "logw",
        "twist_amount",
        "done",
        "max_tokens_left",
        "contexts",
        "group",
        "_views",
    )

    def __init__(self, n, max_tokens, group):
        self.n = n
        # Per-row group id; ESS/resample/log_ml are per-group.
        self.group = np.asarray(group, dtype=np.int64)
        self.logw = np.zeros(n)
        self.twist_amount = np.zeros(n)
        self.done = np.zeros(n, dtype=bool)
        self.max_tokens_left = np.full(n, max_tokens, dtype=np.int64)
        self.contexts = [[] for _ in range(n)]
        # Views reused: reindex mutates arrays in place so row i stays valid.
        self._views = [Particle(self, i) for i in range(n)]

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self._views[i]

    def __iter__(self):
        return iter(self._views)

    def untwist(self, idx=slice(None)):
        """Untwist rows ``idx`` (the whole population by default)."""
        self.logw[idx] -= self.twist_amount[idx]
        self.twist_amount[idx] = 0.0

    def reindex(self, ancestor_indices):
        """Reindex every column by ``ancestor_indices`` (resample/fork)."""
        idx = ancestor_indices
        self.logw = self.logw[idx]
        self.twist_amount = self.twist_amount[idx]
        self.done = self.done[idx]
        self.max_tokens_left = self.max_tokens_left[idx]
        self.contexts = [list(self.contexts[i]) for i in idx]
        self.group = self.group[idx]


class Particle:
    """A view onto one row of a :class:`Population`; reads/writes pass through to the
    population arrays."""

    __slots__ = ("_pop", "_i")

    def __init__(self, pop, i):
        self._pop = pop
        self._i = i

    @property
    def row(self):
        """This particle's row in the population -- its identity across per-row
        bookkeeping (lanes, draw keys), and stable under reindex."""
        return self._i

    @property
    def group(self):
        """The SMC problem this row belongs to; ESS/resample/log_ml are per-group."""
        return self._pop.group[self._i]

    @property
    def logw(self):
        return self._pop.logw[self._i]

    @logw.setter
    def logw(self, v):
        self._pop.logw[self._i] = v

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

    def score(self, amt):
        self._pop.logw[self._i] += amt

    def twist(self, amt):
        self._pop.twist_amount[self._i] += amt
        self._pop.logw[self._i] += amt

    def untwist(self):
        self._pop.untwist(self._i)

    def finish(self):
        self.untwist()
        self._pop.done[self._i] = True

    # record viz adapter
    @property
    def weight(self):
        return self._pop.logw[self._i]


class Controller:
    """Owns the SMC algorithm: population, transition, ESS, resample, log_ml. Every
    sampler collapses to one per-step ``transition``. The population is B independent
    SMC problems ("groups") in one flat row space — ESS/resample/log_ml are per-group,
    and B=1 is the plain single-problem case.

    Args:
        samplers (list[TokenSampler]): per-group sampler producing
            ``(to_append, logw, logp)`` per step.
        critics (list[Potential | None]): per-group critic reweighting/twisting
            that group's particles.
        group_sizes (list[int]): particles per group.
        ess_threshold (float): per-group ESS fraction below which that group resamples.
        max_tokens (int): per-particle token budget.
        twist_with_critic (bool): whether the critic twists during stepping.
        terminate_when (callable, optional): ``context -> bool`` stop condition. When it
            fires, EOS closes the sequence in that same step. The context is in the
            sampler's own representation, so unit nesting is the caller's business.
        resampling_method (str): multinomial/stratified/systematic/residual.
        record (bool): build an :class:`SMCRecord`.
        verbosity (int): 0 silent, 1 prints particles per step.
    """

    def __init__(
        self,
        samplers,
        critics,
        group_sizes,
        ess_threshold,
        max_tokens,
        twist_with_critic,
        terminate_when=None,
        resampling_method="multinomial",
        record=False,
        verbosity=0,
    ):
        assert max_tokens > 0
        assert len(samplers) == len(critics) == len(group_sizes) > 0
        n_particles = sum(group_sizes)

        self.samplers = samplers
        self.critics = critics
        self.n_particles = n_particles
        self.n_resamples = 0
        self.ess_threshold = ess_threshold
        self.twist_with_critic = twist_with_critic
        self.terminate_when = terminate_when
        # A terminal-only critic has no per-step signal: reweight only at termination.
        if twist_with_critic and all(
            c is not None and c.is_terminal_only() for c in critics
        ):
            self.twist_with_critic = False
        # Engine lane leaves, per group: the draw path's views, then the critic's
        # when the run consumes one. Each leaf's lane banks its own per-token
        # logp, which serves that leaf's ``prefix``/``complete`` without a forward.
        self.group_lanes = [
            s.lane_views() + [lf for lf in [self._critic_lane(c)] if lf is not None]
            for s, c in zip(samplers, critics)
        ]
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
        # One record stream per group: cadences are independent, so there is no
        # meaningful global step counter to share.
        self.records = [
            SMCRecord(ng) if record else None for ng in group_sizes
        ]
        # Group-local ancestors a crossing leaves for the next ``_record_step``.
        self._pending_resample: list = [None] * len(group_sizes)
        # Terminal-only critic settles, deferred to the row's round end (the
        # dying row's lanes must release the engine before the critic forwards).
        self._terminal_pending: list = [[] for _ in group_sizes]

        # log(ess_threshold); the per-group ESS test adds log(group_size).
        with np.errstate(divide="ignore"):
            self._log_ess_threshold = np.log(ess_threshold)

    def _critic_lane(self, critic):
        """A consumed critic's own engine lane leaf, or ``None``: no critic, nothing
        that consumes it mid-run, or no single engine leaf (a multi-LM critic scores
        by one-shot forwards at a boundary instead).

        A twist consumes the critic every step; so does a resample, which reweights on
        sums the critic has to be inside. Either way the leaf must serve from its
        lane's bank rather than a forward."""
        if critic is None or not (self.twist_with_critic or self.ess_threshold > 0):
            return None
        return find_engine_lm(critic)

    async def draw_step(self, p):
        """One row's step ``(to_append, logw, logp)``: forced EOS at the ``max_tokens``
        boundary, else the sampler's transition (closed by ``terminate_when``). The
        (slot, ordinal) draw key makes a counter-based picker batch-independent.

        Untwists ``p`` first: a twist is a bet on the resample the row has now passed."""
        if self.twist_with_critic:
            self.particles.untwist(p.row)
        if p.max_tokens_left == 1:
            return await self._force_eos_step(p, self.sampler_of(p))
        with draw_key(p.row, draw_ordinal(p.context)):
            step = await self.sampler_of(p).transition(p.context)
        return self._close_if_stopped(p, step)

    async def step_row(self, p):
        """Draw + bank one live row: a per-token driver's whole per-row step."""
        await self.bank_row(p, *(await self.draw_step(p)))

    def _close_if_stopped(self, p, step):
        """``step`` with EOS appended if ``terminate_when`` fires on the context it
        produces, so the particle terminates in the step that wrote the stop.

        No ``logw_eos``: the stop condition defines what a complete sequence *is*,
        not a deviation from the proposal, so charging it would penalize exactly
        the particles that close."""
        to_append, logw, logp = step
        if self.terminate_when is None or not to_append:
            return step
        context = p.context + list(to_append)
        if context[-1] is EOS or not self.terminate_when(context):
            return step
        return [*to_append, EOS], logw, logp

    async def _force_eos_step(self, p, sampler):
        """Forced-EOS step ``(to_append, logw, logp)`` at the ``max_tokens`` boundary."""
        return [EOS], await sampler.logw_eos(p.context), 0.0

    def sampler_of(self, p):
        """The sampler owning ``p``'s group -- a driver routes each row through its
        own group's sampler, never group 0's."""
        return self.samplers[p.group]

    def _critic_of(self, p):
        return self.critics[p.group]

    async def bank_row(self, p, to_append, logw, logp):
        """Post-draw SMC math: score, advance, critic-twist, reweight + terminate.
        Critic-free rows bank without awaiting; an inline critic twists/reweights
        here.

        ``logp`` is the step's own choice log-prob. Nothing banks it -- each engine
        leaf's lane holds its own -- but it stays in the step tuple callers splat."""
        p.score(logw)
        p.context.extend(to_append)

        critic = self._critic_of(p)
        if critic is None:
            if p.logw == float("-inf"):
                p.finish()
            else:
                if self.verbosity > 0:
                    print(self._repr_particle(p))
                p.max_tokens_left -= 1
                if p.max_tokens_left == 0 or self._is_terminal(p):
                    p.finish()
            return

        if p.logw == float("-inf"):
            p.finish()
            return

        if self.twist_with_critic:
            # batch_score so a lane-served critic LM leaf reads its bank.
            twist_amt = float((await critic.batch_score([p.context]))[0])
            if twist_amt == float("-inf"):
                p.score(twist_amt)
                p.finish()
                return
            p.twist(twist_amt)

        if self.verbosity > 0:
            print(self._repr_particle(p))

        p.max_tokens_left -= 1
        if p.max_tokens_left == 0 or self._is_terminal(p):
            p.finish()
            if not self.twist_with_critic:
                # Terminal-only settle. Under lanes it defers to the round's end:
                # the critic may forward (one-shot), and the dying row's lanes
                # must release the engine before anything awaits one.
                self._terminal_pending[p.group].append(p)
                return
            p.score(twist_amt)

    def _is_terminal(self, p):
        return bool(p.context) and p.context[-1] is EOS

    def _repr_particle(self, p):
        return (
            f"{p.logw:.2f}:\t"
            + colors.magenta % "["
            + (colors.magenta % "|").join(escape(y) for y in p.context)
            + colors.magenta % "]"
        )

    # controller-owned SMC primitives the drivers turn

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
            p.score(start_ws[p.group])

    def _maybe_resample(self, g):
        """Group ``g``'s ESS test + group-local resample. Mutates the group's rows
        of ``self.particles`` in place (other groups untouched). Returns ``True``
        on a crossing."""
        rows = self._group_rows[g]
        Wg = self.particles.logw[rows]
        if np.all(Wg == -np.inf):
            return False
        w_sum = logsumexp(Wg)
        nw = Wg - w_sum
        if not (-logsumexp(nw * 2) < self._log_ess_threshold + np.log(len(rows))):
            return False
        probs = np.exp(nw)
        probs /= probs.sum()  # np.random.choice is strict on sum==1
        local = np.asarray(self.resample_fn(probs))  # ancestors in 0..ng-1
        if self.records[g] is not None:
            local = np.sort(local)  # reproducible record
        ancestors = np.arange(self.n_particles)
        ancestors[rows] = rows[local]  # group-local -> global rows
        self.n_resamples += 1
        self.particles.reindex(ancestors)
        self.particles.logw[rows] = w_sum - np.log(len(rows))
        if self.records[g] is not None:
            self._pending_resample[g] = local.tolist()
        return True

    def save_record(self, json_path):
        """Write group 0's record (the whole run for a single-problem ``SMC``)."""
        if self.records[0] is None:
            return
        with open(json_path, "w") as f:
            f.write(self.records[0].to_json())
        print(f"Saved record to {json_path}")

    def _record_step(self, g):
        """Record one of group ``g``'s completed steps: ``add_init`` first,
        ``add_resample`` if one preceded it, else ``add_smc_step``."""
        record = self.records[g]
        if record is None:
            return
        group = [self.particles[i] for i in self._group_rows[g]]
        if len(record.history) == 0:
            record.add_init(group)
        elif self._pending_resample[g] is not None:
            record.add_resample(self._pending_resample[g], group)
        else:
            record.add_smc_step(group)
        self._pending_resample[g] = None

    def round_boundary(self, g):
        """Close one of group ``g``'s rounds: record the step, then the ESS
        test/resample. Returns ``True`` on a crossing — a caller holding per-row
        state outside the population (open lanes) must rebuild the group's rows,
        survivors included, since a resample rewrites their contexts."""
        self._record_step(g)
        return self._maybe_resample(g)

    def group_rows(self, g):
        """Row indices of group ``g``, invariant across reindex (resample is
        group-local)."""
        return self._group_rows[g]

    async def run(self, lanes=None):
        """The SMC loop: one coroutine per group, gathered. Each group paces its
        own rounds — draw + bank every live row, then its boundary — and no
        structure spans groups. ``lanes`` (a ``LaneRunner``) is the only
        difference between accelerated and plain runs: rows hold engine lanes,
        and the boundary reconciles them."""
        await self.start()
        try:
            if lanes is not None:
                for p in self.particles:
                    if not p.done:
                        lanes.open_row(p)
            await asyncio.gather(
                *[self._run_group(g, lanes) for g in range(len(self._group_rows))]
            )
        finally:
            if lanes is not None:
                lanes.close_all()
        return self.particles

    async def _run_group(self, g, lanes=None):
        rows = self._group_rows[g]
        sampler = self.samplers[g]
        while True:
            live = [self.particles[i] for i in rows if not self.particles.done[i]]
            if not live:
                await self._settle_terminals(g, lanes)
                return
            await sampler.round_start([p.context for p in live])
            await asyncio.gather(*[self._step_row(p, lanes) for p in live])
            await self._settle_terminals(g, lanes)
            crossed = self.round_boundary(g)
            if lanes is not None:
                lanes.after_round(g, crossed)

    async def _settle_terminals(self, g, lanes):
        """Settle deferred terminal-only critic scores for group ``g``'s newly
        finished rows, after their lanes release the engine. One batched score
        per round."""
        parts, self._terminal_pending[g] = self._terminal_pending[g], []
        if not parts:
            return
        if lanes is not None:
            for p in parts:
                lanes.close_row(p)
        amts = await self.critics[g].batch_score([p.context for p in parts])
        for p, amt in zip(parts, amts):
            p.score(float(amt))

    async def _step_row(self, p, lanes):
        if lanes is None:
            return await self.step_row(p)
        from genlm.control.lane_seam import row_binding

        with row_binding(lanes.binding_of(p)):
            await self.step_row(p)



