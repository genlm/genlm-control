"""SMC population, the ``Controller`` (algorithm owner), and ``StepLoop`` (per-token
driver). Engine acceleration lives in ``burst.py``."""

import asyncio

import numpy as np
from arsenal import colors

from genlm.control.constant import EOS
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



class Controller:
    """Owns the SMC algorithm: population, transition, ESS, resample, log_ml. Every
    sampler collapses to one per-step ``transition``.

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
        # B groups in one population; default is one group.
        if group_sizes is None:
            group_sizes = [n_particles]
            samplers = [unit_sampler]
            critics = [critic]
        assert n_particles == sum(group_sizes), "n_particles must be the TOTAL row count"
        assert len(samplers) == len(critics) == len(group_sizes)

        self.samplers = samplers
        self.critics = critics
        # Representatives for the capability check / BurstLoop (groups share structure).
        self.unit_sampler = samplers[0]
        self.critic = critics[0]
        self.n_particles = n_particles
        self.n_resamples = 0
        self.ess_threshold = ess_threshold
        self.twist_with_critic = twist_with_critic
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
        # ``_maybe_resample`` sets these so the next ``_record_step`` tags ``add_resample``.
        self._pending_resample = False
        self._pending_ancestors = list(range(n_particles))

        # log(ess_threshold); the per-group ESS test adds log(group_size).
        with np.errstate(divide="ignore"):
            self._log_ess_threshold = np.log(ess_threshold)

    async def _draw_and_bank(self, p):
        """Per-step transition: draw, then bank. The (slot, ordinal) draw key lets a
        counter-based picker match the burst draw."""
        with draw_key(p._i, draw_ordinal(p.context)):
            to_append, logw, logp = await self._sampler_of(p).transition(p.context)
        await self._bank_step(p, to_append, logw, logp)

    async def _step_particle(self, p):
        """One SMC step for ``p``: force EOS at the ``max_tokens`` boundary, else draw + bank."""
        if p.max_tokens_left == 1:
            logw = await self._sampler_of(p).logw_eos(p.context)
            await self._bank_step(p, [EOS], logw, 0.0)
        else:
            await self._draw_and_bank(p)

    def _sampler_of(self, p):
        return self.samplers[self.particles.group[p._i]]

    def _critic_of(self, p):
        return self.critics[self.particles.group[p._i]]

    def _bank_step_no_critic(self, p, to_append, logw, logp):
        """Sync score + advance + terminate, no-critic path."""
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

    async def _bank_step(self, p, to_append, logw, logp):
        """Post-draw SMC math: score, advance, critic-twist, reweight + terminate.
        Caller untwists ``p`` before the draw."""
        if not self.critic:
            self._bank_step_no_critic(p, to_append, logw, logp)
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

    # burst-lane math the engine seam (_Burst) calls into

    async def _bank_steps(self, parts, records):
        """Bank every row's completed step (``rec.step``), same math as the slow loop.
        Runs on the main loop (hopped from the burst worker)."""
        steps = [
            (p, rec.step) for p, rec in zip(parts, records) if rec.step is not None
        ]
        if not steps:
            return
        if self.critic is None:
            for p, (to_append, logw, logp) in steps:
                self._bank_step_no_critic(p, to_append, logw, logp)
            return
        for p, (to_append, logw, logp) in steps:
            await self._bank_step(p, to_append, logw, logp)



class StepLoop:
    """Per-token round-trip driver (byte-exact ground truth); recomputes logprobs from
    the full context every step."""

    def __init__(self, controller):
        self.controller = controller

    async def run(self):
        """Turn the population per-token to completion: each step draws + scores every
        live particle concurrently, records, then runs the controller-owned
        ESS/resample."""
        c = self.controller
        await c.start()
        while any(not p.done for p in c.particles):
            if c.twist_with_critic:
                c.particles.untwist_all()
            await asyncio.gather(
                *[c._step_particle(p) for p in c.particles if not p.done]
            )
            c._record_step()
            c._maybe_resample()
        return c.particles


