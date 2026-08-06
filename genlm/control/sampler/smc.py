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
        "logp",
        "twist_logp",
        "twist_clip_sum",
        "twist_amount",
        "done",
        "max_tokens_left",
        "contexts",
        "group",
        "_views",
    )

    def __init__(self, n, max_tokens, group=None):
        self.n = n
        # Per-row group id; ESS/resample/log_ml are per-group.
        self.group = (
            np.zeros(n, dtype=np.int64) if group is None
            else np.asarray(group, dtype=np.int64)
        )
        self.logw = np.zeros(n)
        self.logp = np.zeros(n)
        # Critic LM leaf's banked per-token logp sum, and the clipped contrast sum
        # used under ``Twist(clip=...)``. Written only by ``Twist``.
        self.twist_logp = np.zeros(n)
        self.twist_clip_sum = np.zeros(n)
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
        self.logp = self.logp[idx]
        self.twist_logp = self.twist_logp[idx]
        self.twist_clip_sum = self.twist_clip_sum[idx]
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
    def logp(self):
        return self._pop.logp[self._i]

    @logp.setter
    def logp(self, v):
        self._pop.logp[self._i] = v

    @property
    def twist_amount(self):
        return self._pop.twist_amount[self._i]

    @property
    def twist_logp(self):
        return self._pop.twist_logp[self._i]

    @property
    def twist_clip_sum(self):
        return self._pop.twist_clip_sum[self._i]

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


class Twist:
    """Critic-twist policy: the twist value formula plus the burst-banked per-token
    serving sums. The columns live in :class:`Population`; every formula that reads
    or writes them lives here.

    Args:
        contrast (bool): twist with ``critic.score(context) - particle.logp`` (the
            log-ratio against the proposal) instead of ``critic.score(context)``.
            Terminal scores are unaffected — what lands at termination is the
            critic's own ``complete`` (a zeroed one cancels the twist entirely).
        temperature (float): scales the twist; 0 disables twisting entirely.
        clip (tuple, optional): ``(neg, pos)`` per-token caps for the critic-LM
            contrast: the twist becomes ``temperature * sum_t clip(v_t - t_t, -neg,
            +pos)`` over per-token critic/target logp increments, in both lanes.
            Terminal scores stay unclipped.
    """

    def __init__(self, contrast=False, temperature=1.0, clip=None):
        assert clip is None or contrast, "clip requires contrast"
        self.contrast = contrast
        self.temperature = temperature
        self.clip = clip

    def value(self, p, amt):
        """The twist applied for critic score ``amt`` on particle ``p``."""
        if self.contrast:
            amt -= p.logp
        return self.temperature * amt

    def served_row(self, p, dlogp=0.0):
        """One particle's banked prefix sum, served as the critic LM leaf's
        ``prefix``. Under clip, ``logp`` is folded back in so the contrast
        subtraction in :meth:`value` yields the clipped sum; ``dlogp`` covers a
        step increment not yet applied to ``p.logp``."""
        if self.clip is not None:
            return p.logp + dlogp + p.twist_clip_sum
        return p.twist_logp

    def served_complete(self, p):
        """One particle's banked sum (EOS increment included) served as the critic
        LM leaf's ``complete``. Never clipped."""
        return p.twist_logp

    def served_prefix(self, ps):
        """Banked warm-row sums served as the critic LM leaf's ``batch_prefix``."""
        return np.array([self.served_row(p) for p in ps])

    def bank(self, pop, rows, vals, tvals):
        """Accumulate one step's drawn-token increments: the critic leaf's logp per
        row (``vals``), and under ``clip`` the clipped contrast against the draw
        target's logp (``tvals``)."""
        if self.clip is None:
            for row, v in zip(rows, vals):
                pop.twist_logp[row] += v
        else:
            neg, pos = self.clip
            for row, v, t in zip(rows, vals, tvals):
                pop.twist_logp[row] += v
                pop.twist_clip_sum[row] += min(max(v - t, -neg), pos)

    def bank_eos(self, pop, rows, vals):
        """Accumulate terminating rows' EOS logp increments (completes the sum
        :meth:`served_complete` serves; the clip column is terminal-irrelevant)."""
        for row, v in zip(rows, vals):
            pop.twist_logp[row] += v


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
        twist (Twist, optional): the twist policy (contrast/temperature/clip);
            defaults to the bare critic score.
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
        twist=None,
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
        self.twist = twist if twist is not None else Twist()
        self.terminate_when = terminate_when
        # A terminal-only critic has no per-step signal: reweight only at termination.
        if twist_with_critic and all(
            c is not None and c.is_terminal_only() for c in critics
        ):
            self.twist_with_critic = False
        # Per-group critic LM leaf, served from banked twist sums instead of forwarding.
        # ``None`` where that group's critic has no engine leaf (or isn't twisting).
        self.twist_leaves = [
            find_engine_lm(c) if (self.twist_with_critic and c is not None) else None
            for c in critics
        ]
        # Per-group draw-path engine leaf -- the last view a burst injects for that
        # group. Its accumulated logp IS ``p.logp``, so its ``complete`` can be served
        # from the bank: a terminal view scoring against the draw distribution (e.g. a
        # tempered terminal's student side) then never forwards mid-burst.
        self.proposal_leaves = [s.burst_views()[-1] for s in samplers]
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
        # True when critic math defers to the round boundary (``bank_row`` pends
        # instead of awaiting; ``apply_critic_boundary`` settles). Set by ``run``.
        self.defer_critic = False
        # Whether the driver banks the per-token twist sums itself (the burst's
        # warm rows); False means the inline clip path banks from critic scores.
        self.twist_banked = False
        self._critic_pending: list = []
        # ``_maybe_resample`` sets these so the next ``_record_step`` tags ``add_resample``.
        self._pending_resample = False
        self._pending_ancestors = list(range(n_particles))

        # log(ess_threshold); the per-group ESS test adds log(group_size).
        with np.errstate(divide="ignore"):
            self._log_ess_threshold = np.log(ess_threshold)

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
    def serve_row(self, p, dlogp=0.0):
        """Serve one row's banked sums as its engine leaves' ``prefix``/``complete``:
        the critic LM leaf its twist sums, the draw leaf its accumulated logp
        (``dlogp`` covers this step's increment, not yet applied to ``p.logp``).
        Keyed by the row's OWN group: groups carry their own leaves, so another
        group's would miss the override and forward."""
        g = p.group
        leaf = self.twist_leaves[g]
        if leaf is None:
            yield
            return
        prefix = {leaf: [self.twist.served_row(p, dlogp)]}
        complete = {leaf: [self.twist.served_complete(p)]}
        prop = self.proposal_leaves[g]
        if prop is not None and prop is not leaf:
            complete[prop] = [float(p.logp) + dlogp]
        with burst_prefix(prefix), burst_complete(complete):
            yield

    @contextlib.contextmanager
    def serve_prefix(self, g, ps):
        """Serve rows ``ps`` their banked prefix sums as group ``g``'s critic LM leaf's
        ``batch_prefix``."""
        leaf = self.twist_leaves[g]
        if leaf is None:
            yield
            return
        with burst_prefix({leaf: self.twist.served_prefix(ps)}):
            yield

    @contextlib.contextmanager
    def serve_complete(self, g, ps):
        """Serve terminating rows ``ps`` their banked sums as ``batch_complete``:
        group ``g``'s critic LM leaf its twist sums, the draw leaf its accumulated
        logp (fully applied by the boundary settle)."""
        leaf = self.twist_leaves[g]
        if leaf is None:
            yield
            return
        over = {leaf: [self.twist.served_complete(p) for p in ps]}
        prop = self.proposal_leaves[g]
        if prop is not None and prop is not leaf:
            over[prop] = [float(p.logp) for p in ps]
        with burst_complete(over):
            yield

    async def bank_row(self, p, to_append, logw, logp):
        """Post-draw SMC math: score, advance, critic-twist, reweight + terminate.
        Caller untwists ``p`` before the draw. Critic-free rows (no critic, or the
        critic deferred to the round boundary) bank without awaiting; an inline
        critic twists/reweights here."""
        p.score(logw)
        p.logp += logp
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
            if self.twist.clip is not None and not self._is_terminal(p):
                # Clip lives on per-token increments: bank this step's (the burst
                # already banked it from warm rows), then twist off the served sum.
                if not self.twist_banked:
                    self.twist.bank(
                        self.particles, [p.row], [twist_amt - p.twist_logp], [logp]
                    )
                twist_amt = self.twist.served_row(p)
            p.twist(self.twist.value(p, twist_amt))

        if self.verbosity > 0:
            print(self._repr_particle(p))

        p.max_tokens_left -= 1
        if p.max_tokens_left == 0 or self._is_terminal(p):
            p.finish()
            if not self.twist_with_critic:
                twist_amt = await critic.score(p.context)
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
        self.twist_banked = driver.banks_twist
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
            with self.serve_prefix(g, live), self.serve_complete(g, done):
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
                    p.twist(self.twist.value(p, amt))


class StepLoop:
    """Per-token driver (byte-exact ground truth): each round draws + banks every live
    row concurrently, recomputing logprobs from the full context every step."""

    sync_boundary = True
    defers_critic = False
    banks_twist = False

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


