"""SMC population, the ``Controller`` (algorithm owner), and ``StepLoop`` (per-token
driver). Engine acceleration lives in ``burst.py``."""

import asyncio
import contextlib

import numpy as np
from arsenal import colors

from genlm.control.constant import EOS
from genlm.control.burst_seam import burst_prefix, burst_complete
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
        "lane_logp",
        "twist_amount",
        "done",
        "max_tokens_left",
        "contexts",
        "group",
        "_views",
    )

    def __init__(self, n, max_tokens, group, n_lanes):
        self.n = n
        # Per-row group id; ESS/resample/log_ml are per-group.
        self.group = np.asarray(group, dtype=np.int64)
        self.logw = np.zeros(n)
        # ``lane_logp[l, i]``: lane ``l``'s engine LM leaf's own ``prefix`` along row
        # ``i``'s path -- one drawn-token logp per committed token, EOS included once
        # the row terminates, which makes it that leaf's ``complete``.
        self.lane_logp = np.zeros((n_lanes, n))
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
        self.lane_logp = self.lane_logp[:, idx]
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
        """This particle's row in the population -- its identity across a burst's
        per-row bookkeeping, and stable under reindex."""
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
        # Engine lanes, per group: the LM leaves a burst injects for that group --
        # the draw path's, then the critic's when it twists through one. Slot ``l`` is
        # one engine lane across every group, and ``lane_logp[l]`` banks that slot's
        # leaf's own per-token logp, which is what serves its ``prefix``/``complete``
        # inside a burst instead of a forward.
        self.group_lanes = [
            s.burst_views() + [lf for lf in [self._critic_lane(c)] if lf is not None]
            for s, c in zip(samplers, critics)
        ]
        self.resample_fn = get_resampling_fn(resampling_method)
        self.verbosity = verbosity

        group = np.concatenate(
            [np.full(ng, g, dtype=np.int64) for g, ng in enumerate(group_sizes)]
        )
        # Groups that disagree on lane count cannot share a burst (``_batch_blocker``
        # says so and the run falls back), but the column block is sized before the
        # driver is chosen, so it covers the widest group.
        self.particles = Population(
            n_particles,
            max_tokens,
            group=group,
            n_lanes=max(len(lanes) for lanes in self.group_lanes),
        )
        # Per-group row indices (invariant across reindex; resample is group-local).
        self._group_rows = [
            np.nonzero(self.particles.group == g)[0] for g in range(len(group_sizes))
        ]
        self.record = SMCRecord(n_particles) if record else None
        # True when critic math defers to the round boundary (``bank_row`` pends
        # instead of awaiting; ``apply_critic_boundary`` settles). Set by ``run``.
        self.defer_critic = False
        self._critic_pending: list = []
        # ``_maybe_resample`` sets these so the next ``_record_step`` tags ``add_resample``.
        self._pending_resample = False
        self._pending_ancestors = list(range(n_particles))

        # log(ess_threshold); the per-group ESS test adds log(group_size).
        with np.errstate(divide="ignore"):
            self._log_ess_threshold = np.log(ess_threshold)

    def _critic_lane(self, critic):
        """A per-step critic's own engine lane, or ``None``: no critic, nothing that
        consumes it mid-burst, or no single engine leaf to bank (a multi-LM critic
        scores by forwarding at a boundary instead).

        A twist consumes the critic every step; so does a resample, which reweights on
        sums the critic has to be inside. Either way the leaf must be servable from a
        bank rather than a forward."""
        if critic is None or not (self.twist_with_critic or self.ess_threshold > 0):
            return None
        return find_engine_lm(critic)

    async def draw_step(self, p):
        """One row's step ``(to_append, logw, logp)``: forced EOS at the ``max_tokens``
        boundary, else the sampler's transition (closed by ``terminate_when``). The
        (slot, ordinal) draw key lets a counter-based picker match the burst draw."""
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

    @contextlib.contextmanager
    def serve_lanes(self, g, live, done):
        """Serve group ``g``'s lanes their own banked sums as their leaves'
        ``batch_prefix`` (rows ``live``) and ``batch_complete`` (rows ``done``).

        Values are positional against the contexts the caller scores, so ``live`` and
        ``done`` must be the same lists in the same order. Keyed by the group's OWN
        lanes: another group's leaves would miss the override and forward."""
        L, lanes = self.particles.lane_logp, self.group_lanes[g]

        def over(ps):
            return {
                leaf: [float(L[lane, p.row]) for p in ps]
                for lane, leaf in enumerate(lanes)
                if leaf is not None
            }

        with burst_prefix(over(live)), burst_complete(over(done)):
            yield

    async def bank_row(self, p, to_append, logw, logp):
        """Post-draw SMC math: score, advance, critic-twist, reweight + terminate.
        Caller untwists ``p`` before the draw. Critic-free rows (no critic, or the
        critic deferred to the round boundary) bank without awaiting; an inline
        critic twists/reweights here.

        ``logp`` is the step's own choice log-prob. Nothing banks it -- each engine
        leaf's lane holds its own -- but it stays in the step tuple callers splat."""
        p.score(logw)
        p.context.extend(to_append)

        critic = self._critic_of(p)
        if critic is None or self.defer_critic:
            if p.logw == float("-inf"):
                p.finish()
            else:
                if self.verbosity > 0:
                    print(self._repr_particle(p))
                p.max_tokens_left -= 1
                if p.max_tokens_left == 0 or self._is_terminal(p):
                    p.finish()
            if critic is not None and (p.done or self.twist_with_critic):
                self._critic_pending.append(p)
            return

        if p.logw == float("-inf"):
            p.finish()
            return

        if self.twist_with_critic:
            # batch_score so a burst-served critic LM leaf reads its overrides.
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
                # batch_score, like the twist branch: the scalar `score` dispatches to
                # `complete`/`prefix`, which `serve_lanes` does not override, so a
                # burst-served critic leaf would forward here instead of reading its bank.
                twist_amt = float((await critic.batch_score([p.context]))[0])
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

    def _maybe_resample(self):
        """Per-group ESS test + group-local resample. Mutates ``self.particles`` on a
        resample. Returns ``(crossing_groups, ancestors)`` or ``([], None)``."""
        W = self.particles.logw
        crossings = []  # (g, rows, local_ancestors, reset_logw) per crossing group
        for g, rows in enumerate(self._group_rows):
            Wg = W[rows]
            if np.all(Wg == -np.inf):
                continue
            w_sum = logsumexp(Wg)
            nw = Wg - w_sum
            if -logsumexp(nw * 2) < self._log_ess_threshold + np.log(len(rows)):
                probs = np.exp(nw)
                probs /= probs.sum()  # np.random.choice is strict on sum==1
                local = np.asarray(self.resample_fn(probs))  # ancestors in 0..ng-1
                if self.record is not None:
                    local = np.sort(local)  # reproducible record
                crossings.append((g, rows, local, w_sum - np.log(len(rows))))

        if not crossings:
            return [], None

        ancestors = np.arange(self.n_particles)
        for _g, rows, local, _t in crossings:
            ancestors[rows] = rows[local]  # group-local -> global rows
        self.n_resamples += 1
        self.particles.reindex(ancestors)
        for _g, rows, _l, target in crossings:
            self.particles.logw[rows] = target
        ancestors = ancestors.tolist()
        if self.record is not None:
            self._pending_resample = True
            self._pending_ancestors = ancestors
        return [c[0] for c in crossings], ancestors

    def save_record(self, json_path):
        if self.record is None:
            return
        with open(json_path, "w") as f:
            f.write(self.record.to_json())
        print(f"Saved record to {json_path}")

    def _record_step(self):
        """Record one completed step: ``add_init`` first, ``add_resample`` if one
        preceded it, else ``add_smc_step``."""
        if self.record is None:
            return
        if len(self.record.history) == 0:
            self.record.add_init(self.particles)
        elif self._pending_resample:
            self.record.add_resample(self._pending_ancestors, self.particles)
        else:
            self.record.add_smc_step(self.particles)
        self._pending_resample = False

    def round_boundary(self, record=True):
        """Close a round: record the step, then the per-group ESS test/resample.

        Returns the rows of every group that crossed, empty when none did. A driver
        holding per-row state outside the population (the burst's engine requests)
        must flush exactly those rows -- survivors included, since a resample rewrites
        their contexts. ``record=False`` for a round that advanced no row: the ESS
        test still runs, but there is no new step to put in the record."""
        if record:
            self._record_step()
        groups, _ = self._maybe_resample()
        return [self._group_rows[g] for g in groups]

    def group_rows(self, g):
        """Row indices of group ``g``, invariant across reindex (resample is
        group-local)."""
        return self._group_rows[g]

    async def run(self, driver):
        """The SMC loop, driver-agnostic: each iteration the driver turns every live
        row's next step (one token per round for the per-token driver, a whole burst
        for the engine driver), then deferred critic math settles and the round
        boundary runs. The driver owns scheduling; the controller owns the math."""
        self.defer_critic = driver.defers_critic
        await self.start()
        while any(not p.done for p in self.particles):
            await self._round_start()
            await driver.round()
            await self.apply_critic_boundary()
            if driver.sync_boundary:
                self.round_boundary()
        return self.particles

    async def _round_start(self):
        """Hand each group's sampler its live contexts before the round's draws. One
        driver round is one unit per row at unit grain, so this is unit start; a
        free-running (token-grain) burst rounds once per burst, so it fires there
        instead. Runs with the engine idle -- a forward here is legal."""
        by_group = {}
        for p in self.particles:
            if not p.done:
                by_group.setdefault(p.group, []).append(p.context)
        await asyncio.gather(
            *[self.samplers[g].round_start(ctxs) for g, ctxs in by_group.items()]
        )

    async def apply_critic_boundary(self):
        """The deferred critic math, at the round boundary (engine drained; forwards are
        legal). Same math as the inline path: a finished particle scores ``complete``
        permanently; a live one twists for the upcoming resample (the caller untwists at
        the next round's draw)."""
        parts, self._critic_pending = self._critic_pending, []
        if not parts:
            return
        # One settle per particle per boundary: a duplicate entry would re-run the
        # terminal critic (an exec, or an LM forward) on the same context.
        assert len({id(p) for p in parts}) == len(parts)
        # Score the whole pending population through each critic's batched path,
        # one call per critic (groups concurrent).
        by_group = {}
        for p in parts:
            by_group.setdefault(p.group, []).append(p)

        async def _settle(g, ps):
            critic = self.critics[g]
            contexts = [p.context for p in ps]
            # batch_score routes non-EOS contexts to batch_prefix and EOS contexts
            # to batch_complete, each in list order; serve the banked sums for
            # exactly those subsets.
            live = [p for p, ctx in zip(ps, contexts)
                    if not (ctx and ctx[-1] == critic.eos)]
            done = [p for p, ctx in zip(ps, contexts)
                    if ctx and ctx[-1] == critic.eos]
            with self.serve_lanes(g, live, done):
                return ps, await critic.batch_score(contexts)

        for ps, amts in await asyncio.gather(
            *[_settle(g, ps) for g, ps in by_group.items()]
        ):
            for p, amt in zip(ps, amts):
                amt = float(amt)
                if p.done:
                    p.score(amt)
                elif amt == float("-inf"):
                    p.score(amt)
                    p.finish()
                else:
                    p.twist(amt)


class StepLoop:
    """Per-token driver (byte-exact ground truth): each round draws + banks every live
    row concurrently, recomputing logprobs from the full context every step."""

    sync_boundary = True
    defers_critic = False

    def __init__(self, controller):
        self.controller = controller

    async def round(self):
        """One token for every live row."""
        c = self.controller
        if c.twist_with_critic:
            c.particles.untwist()
        await asyncio.gather(*[c.step_row(p) for p in c.particles if not p.done])

    async def run(self):
        return await self.controller.run(self)


