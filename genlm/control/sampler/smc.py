"""SMC population, the ``Controller`` (algorithm owner), and ``StepLoop`` (per-token
driver). Engine acceleration lives in ``burst.py``."""

import asyncio

import numpy as np
from arsenal import colors

from genlm.control.constant import EOS
from genlm.control.potential.base import burst_prefix
from genlm.control.potential.built_in.llm import find_engine_lm
from genlm.control.util import logsumexp, draw_key, draw_ordinal, escape
from genlm.control.sampler.resampling import get_resampling_fn
from genlm.control.sampler.smc_record import SMCRecord, string_for_serialization


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

    def untwist_all(self):
        """Vectorized untwist over the whole population."""
        self.logw -= self.twist_amount
        self.twist_amount[:] = 0.0

    def untwist_subset(self, idx):
        """Vectorized untwist of rows ``idx`` (must be distinct)."""
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
        self._pop.logw[self._i] -= self._pop.twist_amount[self._i]
        self._pop.twist_amount[self._i] = 0.0

    def finish(self):
        self.untwist()
        self._pop.done[self._i] = True

    # record viz adapter
    @property
    def weight(self):
        return self._pop.logw[self._i]

    def string_for_serialization(self):
        return string_for_serialization(self._pop.contexts[self._i])



class Twist:
    """Critic-twist policy: the twist value formula plus the burst-banked per-token
    serving sums. The columns live in :class:`Population`; every formula that reads
    or writes them lives here.

    Args:
        contrast (bool): twist with ``critic.score(context) - particle.logp`` (the
            log-ratio against the proposal) instead of ``critic.score(context)``.
            Terminal scores are unaffected, so the twist still cancels at termination.
        temperature (float): scales the twist; 0 disables twisting entirely.
        clip (tuple, optional): ``(neg, pos)`` per-token caps for the critic-LM
            contrast. When set, the burst banks ``sum_t clip(delta_t, -neg, +pos)``
            and serving folds ``logp`` back in so the contrast subtraction in
            :meth:`value` yields the clipped sum. Burst lane only; the cold path
            scores the unclipped prefix.
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

    def served_prefix(self, ps):
        """Banked warm-row sums served as the critic LM leaf's ``batch_prefix``."""
        return np.array([
            (p.logp + p.twist_clip_sum) if self.clip is not None else p.twist_logp
            for p in ps
        ])

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
            return await self._force_eos_step(p, self._sampler_of(p))
        with draw_key(p._i, draw_ordinal(p.context)):
            step = await self._sampler_of(p).transition(p.context)
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

    def _sampler_of(self, p):
        return self.samplers[self.particles.group[p._i]]

    def _critic_of(self, p):
        return self.critics[self.particles.group[p._i]]

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
            twist_amt = await critic.score(p.context)
            if twist_amt != float("-inf"):
                p.twist(self.twist.value(p, twist_amt))
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
            p.score(start_ws[self.particles.group[p._i]])

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

    def round_boundary(self):
        """Close a round: record the step, then the per-group ESS test/resample."""
        self._record_step()
        self._maybe_resample()

    async def run(self, driver):
        """The SMC loop, driver-agnostic: each iteration the driver turns every live
        row's next step (one token per round for the per-token driver, a whole burst
        for the engine driver), then deferred critic math settles and the round
        boundary runs. The driver owns scheduling; the controller owns the math."""
        self.defer_critic = driver.defers_critic
        await self.start()
        while any(not p.done for p in self.particles):
            await driver.round()
            await self.apply_critic_boundary()
            if driver.sync_boundary:
                self.round_boundary()
        return self.particles

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
            by_group.setdefault(self.particles.group[p._i], []).append(p)

        async def _settle(g, ps):
            critic = self.critics[g]
            contexts = [p.context for p in ps]
            leaf = find_engine_lm(critic) if self.twist_with_critic else None
            if leaf is None:
                return ps, await critic.batch_score(contexts)
            # batch_score routes non-EOS contexts to batch_prefix in list order;
            # serve the banked sums for exactly that subset.
            served = self.twist.served_prefix([
                p for p, ctx in zip(ps, contexts)
                if not (ctx and ctx[-1] == critic.eos)
            ])
            with burst_prefix({leaf: served}):
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

    def __init__(self, controller):
        self.controller = controller

    async def round(self):
        """One token for every live row."""
        c = self.controller
        if c.twist_with_critic:
            c.particles.untwist_all()
        await asyncio.gather(*[c.step_row(p) for p in c.particles if not p.done])

    async def run(self):
        return await self.controller.run(self)


