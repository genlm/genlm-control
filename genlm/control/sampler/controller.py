"""The SMC controller: a single, engine-independent owner of the entire SMC algorithm.

This module replaces the previous llamppl ``smc_standard`` + ``SequenceModel``
coupling. The controller owns the particle population, the per-step transition, the
ESS test, resampling/forking and the log marginal likelihood accumulation --
*always*. It is exact per token: there is no segment-graining and no
hard/soft constraint fork. ``logw_next`` is one operation.

Two drivers turn the population. ``StepLoop`` is the slow per-token Python loop --
it fuses shaping + drawing inside the sampler's ``transition`` and is the byte-exact
ground truth. ``BurstLoop`` is the engine-accelerated outer/inner driver: it runs
each SMC step on the FAST lane (an engine burst, :meth:`Controller.draw` row-wise on
raw warm-KV logits) when the step is engine-expressible, and drops to the SLOW lane
(one ``StepLoop`` transition) for an inexpressible step (a ``slow_cadence``, e.g. a
periodic critic-LM forward). The burst handles both the token grain (one SMC step
per decode step, ESS every step) and the synchronized unit grain (one whole unit per
step, ESS once per unit round). Either way, resample / fork / ESS / log_ml are
controller-owned and NEVER delegated to the engine; the burst and the slow-lane
transition are the two step-runners, nothing more.
"""

import asyncio
from dataclasses import dataclass

import numpy as np
import torch

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.util import logsumexp
from genlm.control.sampler.resampling import get_resampling_fn
from genlm.control.sampler.smc_record import SMCRecord, string_for_serialization
from genlm.control.sampler.util import _drive_or_hop


class NotAcceleratable(Exception):
    """Raised when engine acceleration was *required* but the configuration can't
    be driven by the engine burst.

    The message is the same human-readable ``reason`` that
    :func:`burst_capability` reports and that an ``accelerate="auto"`` fallback
    logs, so the explanation a user sees on an explicit failure matches the one
    they get from introspection (:meth:`SMC.acceleration_report`).
    """


# ---------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------


class Population:
    """The SMC particle population, stored columnar.

    The per-particle scalars that the algorithm reads in bulk (ESS, resample,
    log_ml) live as parallel numpy arrays -- the *single source of truth* -- so
    those tests are array ops, not a fresh ``np.array([p.logw for p in ...])``
    rebuilt every step. The ragged per-particle token ``contexts`` stay as Python
    lists.

    A :class:`Particle` is a thin *view* onto row ``i`` (see below); there is no
    second copy of ``logw`` to keep in sync. ``Population`` is iterable / indexable
    / ``len``-able and yields views, so the rest of the controller (and the
    vendored ``SMCRecord``, which reads ``p.weight`` / ``p.string_for_serialization``
    at call time) is unchanged.

    Attributes (arrays are length ``n``):
        logw: current (twisted) log importance weights.
        logp: accumulated log-probabilities of the sampler's random choices.
        twist_amount: amount currently added to ``logw`` by the critic twist;
            subtracted back out (``untwist``) before each step and on termination,
            exactly as llamppl's ``Model.twist``/``untwist``.
        done: whether each particle has finished stepping.
        max_tokens_left: remaining per-particle token budget.
        contexts: list of token lists sampled so far.
    """

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
        # Which example (sub-population) each row belongs to. A *batched* E-step runs
        # B examples in one population of ``B*N`` rows; ESS / resample / log_ml are
        # per-group (array-masked), the per-row hot paths are group-agnostic. The
        # default -- one group -- makes a single-example run byte-identical to before.
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
        # Per-particle context length at the last slow-lane step. The slow-lane
        # :class:`BoundaryPredicate` cadence is evaluated on the *run buffer*
        # ``context[pop_anchor:]`` (tokens drawn since the last slow step) with the
        # completed run ``context[:pop_anchor]`` as its unit-context -- so
        # ``FixedLengthBoundary(N)`` fires a slow step every N tokens,
        # ``TokenSetBoundary({b"."})`` after a period, etc. Derived from ``contexts``
        # (no parallel buffer to keep in sync); reindexed on resample like the rest.
        self.pop_anchor = np.zeros(n, dtype=np.int64)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return Particle(self, i)

    def __iter__(self):
        return (Particle(self, i) for i in range(self.n))

    def untwist_all(self):
        """Vectorized per-step untwist over the whole population (the slow loop's
        ``for p: p.untwist()``). Done particles carry ``twist_amount == 0`` (their
        ``finish`` already untwisted), so this is a no-op for them -- identical to
        looping."""
        self.logw -= self.twist_amount
        self.twist_amount[:] = 0.0

    def untwist_subset(self, idx):
        """Vectorized untwist of the rows ``idx`` (the burst's live, not-popped
        particles); ``idx`` are distinct so the in-place scatter is well-defined.
        The burst untwists only live rows (done/popped rows already finished),
        matching the old per-row ``for i in live: parts[i].untwist()``."""
        self.logw[idx] -= self.twist_amount[idx]
        self.twist_amount[idx] = 0.0

    def reindex(self, ancestor_indices):
        """Resample/fork: reindex every column by ``ancestor_indices`` (the
        columnar form of the old per-particle ``clone()``). Contexts are
        shallow-copied (their token elements are immutable), matching the old
        ``clone``."""
        idx = ancestor_indices
        self.logw = self.logw[idx]
        self.logp = self.logp[idx]
        self.twist_amount = self.twist_amount[idx]
        self.done = self.done[idx]
        self.max_tokens_left = self.max_tokens_left[idx]
        self.contexts = [list(self.contexts[i]) for i in idx]
        self.pop_anchor = self.pop_anchor[idx]
        # Group labels reindex like every other column. Resample is group-local (a
        # row's ancestor is always in its own group), so this preserves the labels;
        # carried for consistency / correctness under any global ancestor vector.
        self.group = self.group[idx]


class Particle:
    """A thin view onto one row of a :class:`Population`.

    All scalar reads/writes go through to the population's arrays (single source
    of truth, no sync), so the existing per-particle bookkeeping
    (``score``/``twist``/``untwist``/``finish`` -- mirroring ``llamppl.modeling.Model``)
    reads exactly as before. The vectorized paths (ESS, resample, untwist) operate
    on the arrays directly and never go through a view.
    """

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
        """Completed runs before the current one -- the ``unit_context`` arg of the
        slow-lane :class:`BoundaryPredicate`."""
        return self._pop.contexts[self._i][: self._pop.pop_anchor[self._i]]

    @property
    def run_buffer(self):
        """Tokens drawn since the last slow step -- the ``subunit_buffer`` the
        slow-lane :class:`BoundaryPredicate` decides the next cadence step on."""
        return self._pop.contexts[self._i][self._pop.pop_anchor[self._i] :]

    def reset_run(self):
        """Close the current run (anchor at the live context end) -- called right
        after a slow step so the cadence predicate starts the next run empty."""
        self._pop.pop_anchor[self._i] = len(self._pop.contexts[self._i])

    # -- weight bookkeeping, mirroring llamppl.modeling.Model exactly --

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


# ---------------------------------------------------------------------------
# The burst context (the only thing a sampler's burst_draw_batch sees)
# ---------------------------------------------------------------------------


class BurstContext:
    """The per-burst context handed to a burst-capable sampler's ``burst_draw_batch``.

    This is the burst analog of the slow path handing a sampler a ``context``
    and letting it call its potentials' async methods. In the burst, the engine
    supplies the LM half (the batched ``lm_batch``) and the particle's token
    ``context`` is passed to ``burst_draw_batch``; this object exposes the two
    substitutions that stand in for the slow path's async potential calls:

    * :meth:`product_logws` -- reconstruct ``Product(llm, factor).logw_next`` over
      the target vocab from the engine LM weights + ``factor.logw_next(context)``
      (the DirectTokenSampler proposal);
    * :meth:`run_sync` -- run a sampler's own async helper (e.g. AWRS's rejection)
      to completion on the driver's event loop.

    ``factor`` (the additive control-side factor / AWRS boolean condition, or
    ``None``) is exposed directly so the sampler can score it from a context. The
    Controller's row-map / pop-out / engine-LM runtime is deliberately NOT on this
    object -- a sampler only ever sees this narrow, named interface, never the
    Controller's private burst state.

    ``scratch`` is a fresh per-burst dict the sampler may use to carry its own
    state across the burst's engine steps (the unit sampler keeps each row's
    in-progress subunit buffer there, keyed by the row's external index). It is
    re-created for every burst, so nothing leaks across resamples; token samplers
    ignore it.
    """

    __slots__ = (
        "factor",
        "scratch",
        "_target",
        "_llm_idxs",
        "_factor_idxs",
        "_eval_factor",
        "_run_async",
    )

    def __init__(self, factor, target, llm_idxs, factor_idxs, eval_factor, run_async):
        self.factor = factor
        self.scratch = {}
        self._target = target
        self._llm_idxs = llm_idxs
        self._factor_idxs = factor_idxs
        self._eval_factor = eval_factor
        self._run_async = run_async

    def factor_logws(self, context):
        """The factor's ``logw_next`` over ``context``, evaluated on the driver's
        event loop -- the burst stand-in for the slow path's
        ``factor.logw_next(context)`` (the wrapped potential's ``_consume`` cache
        carries it incrementally across the burst, so this is not a full replay)."""
        return self._eval_factor(self.factor, context)

    def product_logws(self, lm_weights, context):
        """Reconstruct ``Product(llm, factor).logw_next`` over the target vocab
        from the engine LM weights (``lm_weights``, a length-``V+1`` array in the
        LM's vocab-eos order -- one row of the batched ``_process_logw_next_batch``)
        and ``factor.logw_next(context)``.

        Gathers through ``Product``'s own ``v1_idxs``/``v2_idxs`` (resolved here
        as ``llm_idxs``/``factor_idxs``) -- the exact slow-path index maps -- with
        the engine LM half substituting the reprefilled one, so vocab narrowing
        matches the slow path.
        """
        factor_logws = self.factor_logws(context)
        return self._target.make_lazy_weights(
            lm_weights[self._llm_idxs] + factor_logws.weights[self._factor_idxs]
        )

    def run_sync(self, coro):
        """Run a sampler's async helper to completion on the driver's event loop
        and block for the result -- one event-loop hop, no inner gather."""
        return self._run_async(coro)


@dataclass
class BurstDraw:
    """One live row's result from a sampler's ``burst_draw_batch`` -- the burst
    analog of a slow ``transition`` outcome, plus the engine-continuation token.

    The single per-row contract between a sampler and the Controller's ``draw``,
    uniform across grains: a token sampler completes one SMC step every decode
    step; a unit sampler accumulates subunits and completes one only at a unit
    boundary. ``draw`` banks the step (if any), emits the token to the engine, and
    pops the row (if asked) without ever branching on the sampler's type.

    Fields:
        token: the ``Token`` (or ``EOS``) drawn this decode step; the Controller
            maps it to the engine token id that extends this row's warm KV. For a
            unit sampler mid-unit it is the latest subunit; ``EOS`` maps to the
            engine eos placeholder (never banked, only fed to the engine, and the
            row is dropped immediately after).
        step: ``(to_append, logw, logp)`` when an SMC step COMPLETED this decode
            step -- the Controller banks it (score / advance the context by
            ``to_append`` / critic-twist / terminate), exactly the slow
            transition's payload. ``None`` when the row is mid-step (a unit still
            accumulating): only the engine token is emitted, nothing is banked.
        pop: pop this row OUT of the burst after this step even though it did not
            terminate -- the synchronized unit-boundary wait (the row is relaunched
            next burst). A terminated row is always popped regardless of ``pop``.
    """

    token: object
    step: tuple | None
    pop: bool = False


# -- burst exit reasons -------------------------------------------------------
#
# Why one engine burst returned, so the outer driver (:class:`BurstLoop`) can
# dispatch. A burst is a stateless enter-run-exit unit; it never resamples. It
# ends for one of:
#
# * ``_EXIT_TERMINATED`` -- every flagged row finished on its own (drawn EOS /
#   budget / -inf weight). Nothing for the driver to do; the population shrinks
#   and the loop either re-enters or exits.
# * (ESS is NOT an exit reason: a token-grain ESS crossing resamples IN PLACE
#   mid-burst -- drop + re-add the crossing group's rows -- and the burst keeps
#   running; a single population is just one group.)
# * ``_EXIT_UNIT_SYNC`` -- a synchronized (unit-grain) burst ran exactly one SMC
#   step (one unit) for every live row and they have all popped at their unit
#   boundaries; the driver runs the controller-owned ESS test + resample over the
#   synced population, identical timing to the slow per-unit loop, then relaunches.
#
# * ``_EXIT_SLOW_STEP`` -- the next step is engine-INEXPRESSIBLE for the live rows
#   (a cadence: e.g. a periodic critic-LM forward). The burst popped every live row
#   BEFORE drawing that step; the driver runs one per-token slow-lane transition for
#   it (engine free) and then re-enters the burst. Like ESS it is synchronized -- in
#   a free-running burst all live rows advance in lockstep, so a length-based cadence
#   is due for all of them at once.
_EXIT_TERMINATED = "terminated"
_EXIT_UNIT_SYNC = "unit_sync"
_EXIT_SLOW_STEP = "slow_step"


class _Burst:
    """The Controller's private per-engine-burst runtime (set by the BurstLoop
    for the duration of one ``run_burst``; ``None`` outside a burst).

    Holds what the Controller's ``draw`` needs -- the live particle row-map, the
    engine LM (to batch the rows' ``lm_batch``), the EOS id, ``abort_rows``
    (external row indices the draw has flagged to drop from the burst), and
    ``exit_reason`` (why the burst ended, read by the driver to dispatch) -- plus
    the sampler-facing :class:`BurstContext` (``ctx``). Samplers never see this
    object; only ``ctx``.

    Pop-out is out-of-band: ``draw`` adds a row's external index to ``abort_rows``
    (when it terminates, or when an in-place resample drops it for re-add), and
    ``run_burst`` drains them via :meth:`Controller.drain_aborts` and calls
    ``abort_request`` -- no forced-EOS, no discard forward. ``exit_reason`` stays
    ``_EXIT_TERMINATED`` for a free-running burst (ESS resamples in place mid-burst,
    no exit); a unit-grain burst sets ``_EXIT_UNIT_SYNC``.
    """

    __slots__ = (
        "particles",
        "llm",
        "eos_id",
        "abort_rows",
        "add_rows",
        "ext_to_row",
        "row_to_ext",
        "next_ext",
        "context_ids",
        "ctx",
        "exit_reason",
    )

    def __init__(self, particles, llm, eos_id, ctx, context_ids):
        self.particles = particles
        self.llm = llm
        self.eos_id = eos_id
        # BurstLoop._context_ids: builds a particle's engine prompt (its group's
        # prompt + drawn tokens). The Controller's in-place resample needs it to
        # re-prefill re-added rows, but it lives on BurstLoop -> passed in here.
        self.context_ids = context_ids
        self.abort_rows = set()
        # In-place per-group resample: rows the controller asks the engine to
        # (re-)add mid-burst, as ``(ext_id, prompt_ids)``. ``ext_to_row`` /
        # ``row_to_ext`` map an engine request's external id <-> its POPULATION row
        # (``p._i``); ``next_ext`` hands out a FRESH id per re-add (so an abort and a
        # re-add of the same slot never collide on an id). Initial rows: ext i is the
        # entry-order index, mapping to that live particle's population row.
        self.add_rows = []
        self.ext_to_row = {i: p._i for i, p in enumerate(particles)}
        self.row_to_ext = {p._i: i for i, p in enumerate(particles)}
        self.next_ext = len(particles)
        self.ctx = ctx
        self.exit_reason = _EXIT_TERMINATED


# ---------------------------------------------------------------------------
# The transition (shared by all samplers)
# ---------------------------------------------------------------------------


class Controller:
    """Owns the SMC algorithm: population, transition, ESS, resample, log_ml.

    A "sampler" collapses to a single per-step transition
    ``state -> (token, logw[, logp])`` that this controller calls. The controller is owned
    once; no sampler reimplements the loop.

    Args:
        unit_sampler (TokenSampler): Produces ``(token, logw, logp)`` per step.
        critic (Potential, optional): Reweights/twists particles.
        n_particles (int): Number of particles.
        ess_threshold (float): ESS fraction below which we resample.
        max_tokens (int): Per-particle token budget.
        twist_with_critic (bool): Whether the critic twists during stepping
            (True iff ``ess_threshold > 0``), matching the old SequenceModel.
        resampling_method (str): One of multinomial/stratified/systematic/residual.
        record (bool): Whether to build an :class:`SMCRecord`.
        verbosity (int): 0 silent, 1 prints particles per step.
        slow_cadence (BoundaryPredicate, optional): a
            :class:`~genlm.control.sampler.unit.BoundaryPredicate` marking when "the
            NEXT step is engine-INEXPRESSIBLE and must run on the slow lane" (a
            cadence, e.g. a periodic critic-LM forward). It is evaluated per row on
            the row's *run buffer* -- the tokens drawn since its last slow step --
            with the same ``(unit_context, subunit_buffer)`` signature the unit
            sampler uses: ``FixedLengthBoundary(N)`` -> a slow step every N tokens,
            ``TokenSetBoundary({b"."})`` -> a slow step after each period,
            ``CFGBoundary`` -> on grammar completion. When set, the ``BurstLoop``
            runs the cadence steps as per-token transitions (engine free) and bursts
            the rest; ``None`` (default) means every step is engine-expressible (the
            burst never drops to the slow lane). Only the ``BurstLoop`` reads it --
            the all-slow ``StepLoop`` runs every step slow regardless. When it fires
            for any live row the whole population takes one synced slow-lane step (the
            non-firing rows run the identical per-step math), so the cadence is
            math-neutral: a pure performance hint, never an algorithm change.
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
        # Batched (B-example) SMC runs B independent sub-populations ("groups") in
        # ONE population of ``sum(group_sizes)`` rows -- ESS / resample / log_ml are
        # per-group (array-masked), the per-row paths dispatch the group's sampler /
        # critic. The default -- a single group of ``n_particles`` with the lone
        # ``unit_sampler`` / ``critic`` -- is byte-identical to the unbatched path.
        if group_sizes is None:
            group_sizes = [n_particles]
            samplers = [unit_sampler]
            critics = [critic]
        assert n_particles == sum(group_sizes), "n_particles must be the TOTAL row count"
        assert len(samplers) == len(critics) == len(group_sizes)

        self.samplers = samplers
        self.critics = critics
        self.group_sizes = group_sizes
        # B-agnostic representatives for the burst-capability check / BurstLoop seam
        # (all groups share sampler/target STRUCTURE; only prompt + critic differ).
        self.unit_sampler = samplers[0]
        self.critic = critics[0]
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.max_tokens = max_tokens
        self.twist_with_critic = twist_with_critic
        # A terminal-only critic (``prefix == 0`` for every proper prefix) carries no
        # per-step twist signal -- its per-step twist is ``twist(0)``, a no-op -- so
        # reweight ONLY at termination. Byte-identical to twisting per step (the
        # provisional ``twist(0)`` adds then untwists 0), but skips the per-step
        # critic call. ``is_terminal_only()`` defaults to ``False``, so this never
        # fires unless a critic explicitly opts in. Population-wide flag -> require
        # every group's critic to be terminal-only.
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
        # Per-group row indices. Resample is group-local (a row's ancestor is always
        # in its own group), so group labels -- and therefore these index sets -- are
        # invariant across reindex; precompute once.
        self._group_rows = [
            np.nonzero(self.particles.group == g)[0] for g in range(len(group_sizes))
        ]
        # Set by ``_maybe_resample`` each call to the group indices it resampled,
        # read by the in-place burst resample handler.
        self._last_resampled_groups = []
        self.record = SMCRecord(n_particles) if record else None
        # Record cadence (lane-neutral): a resample sets these so the NEXT recorded
        # step is tagged ``add_resample`` (the lazy tag-at-next-step). Set by
        # ``_maybe_resample``, consumed by ``_record_step`` -- every lane uses it.
        self._pending_resample = False
        self._pending_ancestors = list(range(n_particles))

        # ``log(ess_threshold)`` -- the per-group ESS RHS adds ``log(group_size)`` (so
        # the test is ``log(ESS_g) < log(ess_threshold * N_g)``). For one group of
        # ``n_particles`` this is exactly the old ``log(ess_threshold)+log(n)``.
        with np.errstate(divide="ignore"):
            self._log_ess_threshold = np.log(ess_threshold)

        # The :class:`_Burst` runtime set by the BurstLoop for the duration of a
        # single engine burst; read by the ``draw`` callback the engine invokes
        # row-wise. ``None`` outside a burst.
        self._burst = None

    # -- the per-step transition for ONE particle ----------------------------
    #
    # In the slow path drawing is fused inside the sampler's ``transition``
    # coroutine (which computes logw_next, draws, and returns the importance
    # weight). The burst path instead calls ``draw`` row-wise; both paths apply
    # the identical SMC weight/termination math via the ``_terminate_*`` helpers
    # below.

    async def _draw_and_score(self, p):
        """The slow per-step transition for one particle.

        Draws a token + weight increment from the sampler, then applies the
        shared SMC scoring/twist/termination math (:meth:`_score_advance_terminate`).

        ``unit_sampler.transition`` returns ``(to_append, logw, logp)`` where
        ``to_append`` is the list of items to extend the particle context with
        (a single token for token samplers; a unit -- possibly split around a
        trailing EOS -- for the multi-token unit sampler).
        """
        to_append, logw, logp = await self._sampler_of(p).transition(p.context)
        await self._score_advance_terminate(p, to_append, logw, logp)

    def _sampler_of(self, p):
        """The unit sampler for particle ``p``'s group (``samplers[0]`` when
        unbatched). Per-group dispatch is the only group-awareness on the hot path."""
        return self.samplers[self.particles.group[p._i]]

    def _critic_of(self, p):
        """The critic for particle ``p``'s group (``critics[0]`` when unbatched)."""
        return self.critics[self.particles.group[p._i]]

    def _advance_no_critic(self, p, to_append, logw, logp):
        """Sync score + advance + terminate for the NO-CRITIC path. Shared by the
        slow :meth:`_score_advance_terminate` (its no-critic branch) and the burst
        bookkeeping (:meth:`_bank_burst_steps`) -- the burst calls it directly so a
        no-critic step needs NO event-loop hop, which would otherwise be the new
        per-particle bottleneck once the LM draw is batched."""
        p.score(logw)
        p.logp += logp
        p.context.extend(to_append)
        if p.logw == float("-inf"):
            p.finish()
            return
        # Post-draw termination (the no-critic tail): verbosity, budget decrement,
        # and finish on budget exhaustion or a terminal EOS.
        if self.verbosity > 0:
            print(self._repr_particle(p))
        p.max_tokens_left -= 1
        if p.max_tokens_left == 0 or self._is_terminal(p):
            p.finish()

    async def _score_advance_terminate(self, p, to_append, logw, logp):
        """The post-draw SMC math: score, advance the context, apply the critic
        twist (per-step when ``twist_with_critic``) and reweight at termination,
        and terminate. This is the ONE implementation of the per-step math; both
        the slow loop (:meth:`_draw_and_score`, ``await``) and the engine burst
        (:meth:`draw`, via ``run_sync``) call it after obtaining their draw, so
        the critic is handled identically regardless of where the logits came
        from -- no per-driver critic logic, no critic-category dispatch.

        The caller is responsible for untwisting ``p`` before the draw (the slow
        ``run`` loop and the burst ``draw`` both untwist each live particle at the
        start of the step, mirroring llamppl's ``smc_standard``).
        """
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

        # From here on the critic is non-None (the no-critic case returned above).
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
        """Initialize particles from each group's start weight.

        Mirrors ``SequenceModel.start``: scores every particle by the empty
        sequence's prefix weight under ITS GROUP's target potential. For a single
        group this is one weight applied to all (byte-identical to the unbatched
        path); for a batch each group uses its own sampler's ``start_weight`` (the
        empty-prefix weight can differ per example).
        """
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
        """Run the slow (ground-truth) SMC loop to completion.

        Exact per-token re-implementation of llamppl's ``smc_standard``: untwist
        all particles, step the live ones (batched), record, then ESS-test and
        resample. Returns the final particle population.
        """
        while any(not p.done for p in self.particles):
            self.particles.untwist_all()

            await asyncio.gather(
                *[self._draw_and_score(p) for p in self.particles if not p.done]
            )

            self._record_step()
            self._maybe_resample()

        return self.particles

    def _ess_below_threshold(self, normalized_weights, ng):
        """The per-group ESS resample predicate, shared by the test-only
        ``_ess_crosses`` and the mutating ``_maybe_resample`` so both decide
        identically.

        Given a group's (log) normalized weights and its size ``ng``, returns
        whether its effective sample size has fallen below ``ess_threshold * ng``.
        ``-logsumexp(2w)`` is ``log(ESS)``; the RHS ``log(ess_threshold)+log(ng)``.
        For one group of ``n_particles`` this is the old whole-population test.
        """
        return -logsumexp(normalized_weights * 2) < self._log_ess_threshold + np.log(ng)

    def _ess_crosses(self):
        """Whether the ESS test triggers a resample for ANY group on the current
        population (the burst's pop-out trigger).

        Test only -- does not mutate. Uses the identical per-group predicate as
        ``_maybe_resample`` so the burst's pop-out decision matches the slow path's
        resample decision exactly. A group whose weights are all ``-inf`` never
        crosses (matching the slow path's ``continue``). One group -> the old test.
        """
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
        """Run the per-group ESS test and resample each group that crosses.

        This is the *only* implementation of the ESS/resample math; both the slow
        ``run`` loop and the ``BurstLoop`` call it so the two paths are bit-identical
        here. Resampling is group-LOCAL: a group that crosses is reindexed within its
        own rows (its ancestors are drawn only from itself), other groups are left
        as-is. Builds ONE global ancestor vector (identity outside crossing groups)
        and reindexes once. For a single group this reduces exactly to the old
        whole-population resample. Mutates ``self.particles`` on a resample.

        Returns ``(did_resample, ancestor_indices)`` where ``did_resample`` is True
        iff ANY group resampled and ``ancestor_indices`` is the global vector.
        """
        n = self.n_particles
        # ``sort_ancestors`` only matters for a reproducible record; derive it here so
        # every caller (slow loop, burst, cadence) gets identical behavior.
        sort_ancestors = self.record is not None
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
        """Write the SMC record JSON, matching the old smc_standard json_file path."""
        if self.record is None:
            return
        with open(json_path, "w") as f:
            f.write(self.record.to_json())
        print(f"Saved record to {json_path}")

    def _record_step(self):
        """Record one completed SMC step (lane-neutral): ``add_init`` for the first
        entry, ``add_resample`` if a resample preceded it (the pending tag set by
        ``_maybe_resample``), else ``add_smc_step``. No-op without a record. Called
        wherever a step completes -- the slow ``run`` loop, the token-grain ``draw``,
        the unit-grain round boundary, and the cadence ``_slow_step``."""
        if self.record is None:
            return
        if len(self.record.history) == 0:
            self.record.add_init(self.particles)
        elif self._pending_resample:
            self.record.add_resample(self._pending_ancestors, self.particles)
        else:
            self.record.add_smc_step(self.particles)
        self._pending_resample = False

    # -- the engine callback draw (used only by the BurstLoop) ---------------
    #
    # This reproduces, on raw engine logits, exactly what ``_draw_and_score``
    # does for a burst-capable sampler over an engine LM (and, with an additive
    # control-side factor, the constrained extension). The engine IS the language
    # model; the draw works in the same control-side V+1 vocabulary as the slow
    # path, turning the batched proposal into one :class:`BurstDraw` per live row
    # via the sampler's ``burst_draw_batch`` (which adds any non-LM factor's
    # per-token log-weights). Each sampler's draw picks its RNG stream -- the slow
    # path's numpy ``fast_sample_lazyweights`` or a batched torch draw -- so parity
    # stays the no-bias check the warm-KV residual already requires (see the
    # per-sampler docstrings in ``token.py``).
    #
    # A live burst is described by ``self._burst`` (a :class:`_Burst` set by
    # the BurstLoop): ``particles`` (the live row-map), ``llm`` (the engine
    # PromptedLLM, supplying ``_process_logw_next_batch`` / ``_maybe_temper``),
    # ``eos_id`` (the engine token id used as the committed placeholder for an
    # aborted row), ``abort_rows`` (external indices ``draw`` flags to drop), and
    # ``ctx`` (the sampler-facing :class:`BurstContext`). Pop-out is the explicit
    # ``abort_request`` ``run_burst`` issues from :meth:`drain_aborts` -- no
    # forced-EOS, no discard forward.

    def _burst_external(self, request_id):
        """The external particle index for a vLLM internal request id
        (``"{external}-{8 chars}"``); the BurstLoop assigned ``str(index)``."""
        return int(request_id.rsplit("-", 1)[0])

    def _burst_particle(self, request_id):
        """Map a vLLM internal request id to its particle via its external id ->
        population row (so in-place re-added rows, which carry fresh ids, resolve to
        the right resampled slot)."""
        return self.particles[self._burst.ext_to_row[self._burst_external(request_id)]]

    def drain_adds(self):
        """Rows the controller asks ``run_burst`` to (re-)add to the live engine batch this
        step, as ``(ext_id, prompt_ids)`` (consumed). The in-place per-group resample
        (:meth:`_resample_burst`) queues a resampled group's surviving rows
        here so they rejoin WITHOUT draining the engine; empty otherwise."""
        rows = self._burst.add_rows
        self._burst.add_rows = []
        return rows

    def _resample_burst(self, w):
        """The burst's mid-stream per-group resample: reindex the groups that crossed
        ESS, then -- WITHOUT draining the burst -- drop each resampled group's current
        engine rows and re-add its surviving rows at their resampled contexts (fresh
        ids). Other groups never pause; the engine re-prefills the re-added rows
        (prefix-cache-warm). A single population is just one group. ESS / resample /
        weights stay controller-owned -- this only translates a completed resample
        into engine abort/add plumbing.

        Resampling can flip a row done<->live (a slot reindexed to a finished vs an
        unfinished ancestor), so re-adds are decided over each group's FULL post-
        reindex state: every still-live row is re-added, finished rows are not.
        """
        self._maybe_resample()  # reindexes crossing groups; sets _last_resampled_groups + record tag
        for g in self._last_resampled_groups:
            rows = self._group_rows[g]
            # Drop the group's current engine rows -- their KV holds pre-resample
            # tokens. (A row already aborted on termination is harmless to re-abort.)
            for row in rows:
                ext = w.row_to_ext.pop(int(row), None)
                if ext is not None:
                    w.abort_rows.add(ext)
                    w.ext_to_row.pop(ext, None)
            # Re-add the still-live rows at their resampled context (fresh id).
            for row in rows:
                p = self.particles[int(row)]
                if p.done:
                    continue
                ext = w.next_ext
                w.next_ext += 1
                w.ext_to_row[ext] = int(row)
                w.row_to_ext[int(row)] = ext
                w.add_rows.append((ext, w.context_ids(p)))

    def drain_aborts(self):
        """External row indices ``draw`` has flagged to abort since the last call
        (consumed). ``run_burst`` calls this after each engine step and issues
        ``abort_request`` for them -- the out-of-band pop-out that replaces the old
        forced-EOS. A row is flagged when its particle terminates, when it completes
        a unit and must wait at the synced boundary (unit grain), when the ESS test
        crosses (all live rows, ending the burst for resample), or just before a
        cadence step the slow lane must run (all live rows)."""
        rows = self._burst.abort_rows
        self._burst.abort_rows = set()
        return list(rows)

    def draw(self, logits, request_ids):
        """Draw one token per live row, reproducing the slow path's per-step draw.

        The expensive 50k-vocab LM processing runs ONCE for the whole batch
        on-device (``_process_logw_next_batch``); the sampler's
        :meth:`~TokenSampler.burst_draw_batch` then turns that batched proposal
        into one :class:`BurstDraw` per live row -- the burst analog of ``sample``
        (see the per-sampler docstrings in ``token.py`` for how each reconstructs
        ``Product(llm, factor).logw_next`` from the carried factor state and which
        RNG stream it draws with). The Controller banks any completed SMC step,
        advances, and terminates each particle (:meth:`_bank_burst_steps`). Pop-out
        is out-of-band -- a row is added to ``abort_rows`` for ``run_burst`` to
        ``abort_request`` (:meth:`drain_aborts`) when its particle terminates, when
        it asks to wait at a unit boundary (``BurstDraw.pop``), when the ESS test
        crosses (every live row, token grain), or just before a cadence step (every
        live row) -- no forced EOS, no discard forward.

        The engine's ``SamplingMetadata`` is intentionally not taken: this draw
        works entirely in the control-side V+1 vocab (temperature via
        ``_maybe_temper``, no top-k/p), so the engine's per-row sampling params
        play no role.
        """
        w = self._burst
        sampler = self.unit_sampler
        # Every row in request_ids is live: an aborted/terminated row was dropped
        # from the engine, so it never reappears here.
        parts = [self._burst_particle(rid) for rid in request_ids]
        externals = [self._burst_external(rid) for rid in request_ids]

        # Untwist last step's provisional critic twist before this step's score
        # (mirrors the slow loop's per-step untwist; no-op without a critic).
        self.particles.untwist_subset([p._i for p in parts])

        # The expensive 50k-vocab LM processing happens ONCE for the whole batch,
        # on-device -- never per row. The sampler then draws all rows from this
        # batched proposal (Direct vectorizes; AWRS/Set loop their cheap
        # per-particle control over the [V+1] rows), returning one BurstDraw per
        # row. ``externals`` are the rows' stable particle indices, so a unit
        # sampler can key its per-burst accumulator across the shrinking live set.
        llm = w.llm
        lm_batch = llm._process_logw_next_batch(llm._maybe_temper(logits.float()))
        records = sampler.burst_draw_batch(
            lm_batch,
            [p.context for p in parts],
            externals,
            w.ctx,
        )
        # Score every row's completed SMC step. With a critic, ALL rows are gathered
        # into ONE event-loop hop so the critic LM autobatches across the population
        # (mirrors the slow loop's per-step ``gather``) instead of a hop + serial
        # critic forward per particle. Runs before the token map + abort checks
        # because scoring sets ``p.done`` on termination.
        self._bank_burst_steps(parts, records)
        # Record this engine step as one SMC step (when any row banked one) -- the
        # burst lane's equivalent of the slow loop's per-step record entry.
        # Token grain completes one SMC step per decode step -> record here, every
        # step. Unit grain completes a step only at the synced boundary, so it records
        # once per round in ``BurstLoop.run`` (not per decode-step-with-a-completion).
        if sampler.burst_free_running() and any(rec.step is not None for rec in records):
            self._record_step()
        out = [0] * len(parts)
        for k, p in enumerate(parts):
            out[k] = self._burst_token_id(p, records[k])
            if p.done or records[k].pop:
                # Staggered pop-out: drop the row when the particle terminated, or
                # when the sampler asks it to wait at a step boundary (a completed
                # unit, synchronized grain) -- both leave the engine the same way.
                w.abort_rows.add(externals[k])
                # The burst continues past this terminated row, so drop its ext<->row
                # entries to keep the maps tight rather than leaning on run_burst's
                # gone-filter to mask a stale lookup.
                if p.done:
                    w.ext_to_row.pop(externals[k], None)
                    w.row_to_ext.pop(p._i, None)

        # End-of-step ESS test (the predicate the slow path applies after every
        # token) -- the TOKEN-grain (free-running) pop-out only. If it crosses,
        # abort every live row so the burst ends and the driver resamples; tag the
        # reason. ``draw`` uses the identical predicate (``_ess_crosses``) the
        # driver's ``_maybe_resample`` re-tests on the unchanged population, so the
        # tag is exact. A synchronized (unit-grain) sampler does NOT test ESS
        # mid-unit -- it pops every row at its unit boundary and the driver runs
        # ESS once per round (``_EXIT_UNIT_SYNC``, set in ``_run_burst``).
        if sampler.burst_free_running() and self._ess_crosses():
            # Resample the crossing groups IN PLACE and re-add their surviving rows
            # mid-burst -- the burst keeps running (no engine drain), other groups
            # never pause, and a single population is just one group (no special
            # "drain + relaunch" case). No ``exit_reason``: the burst ends only when
            # every particle is done.
            self._resample_burst(w)
        elif self._slow_cadence_due(parts):
            # The next step is engine-inexpressible (a cadence) -- pop every live
            # row BEFORE the burst draws it, so the driver runs it on the slow lane
            # (ESS takes precedence above: if it crossed, resample first, then the
            # driver's burst-entry check runs this same cadence step next).
            w.abort_rows.update(externals)
            w.exit_reason = _EXIT_SLOW_STEP

        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def _slow_cadence_due(self, parts):
        """Whether the NEXT step is engine-inexpressible (a slow-lane cadence) for
        ANY still-live row. ``parts`` are this step's rows just banked.

        The cadence is a :class:`~genlm.control.sampler.unit.BoundaryPredicate`
        evaluated per row on its run buffer (``run_prefix`` as ``unit_context``,
        ``run_buffer`` as ``subunit_buffer``) -- the same boundary library the unit
        sampler uses, reused as the cadence. So it covers a length cadence
        (``FixedLengthBoundary(N)``, due for every lockstep-synced row at the same
        step) and a CONTENT cadence (``TokenSetBoundary``/``CFGBoundary``, due for
        different rows at different steps) uniformly. When it fires for any row, the
        driver runs ONE slow-lane step for the WHOLE live population -- the rows that
        did not hit the cadence simply run their (still expressible) transition on
        the slow lane that step, which is the identical per-step math, so the
        population stays synced and the result is unchanged."""
        return self._cadence_due(p for p in parts if not p.done)

    def _cadence_due(self, live):
        """Shared boundary-predicate cadence test over ``live`` particles (used by
        both the mid-burst :meth:`_slow_cadence_due` and the driver's burst-entry
        check). ``False`` without a cadence (the burst owns every step)."""
        if self.slow_cadence is None:
            return False
        return any(self.slow_cadence(p.run_prefix, p.run_buffer) for p in live)

    def _bank_burst_steps(self, parts, records):
        """Bank every row's completed SMC step this engine step, with the SAME math
        the slow loop uses. A mid-step row (``rec.step is None``: a unit still
        accumulating subunits) banks nothing; only its subunit extends the warm KV.

        Without a critic the advance is sync (:meth:`_advance_no_critic`, no hop).
        With a critic, ALL rows' :meth:`_score_advance_terminate` are gathered into
        ONE ``run_sync`` hop, so the critic's per-particle ``score`` calls are
        concurrent and the critic LM's autobatcher coalesces them into a single
        forward -- the same per-step ``gather`` the slow ``run`` loop uses, instead
        of a hop (and a serial critic forward) per particle."""
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

        self._burst.ctx.run_sync(_score_all())

    def _burst_token_id(self, p, rec):
        """Map a banked row's drawn token to the engine token id to emit. Must run
        after :meth:`_bank_burst_steps` (which sets ``p.done`` on termination)."""
        token = rec.token
        if isinstance(token, EndOfSequence):
            # A drawn EOS always completes the step and terminates the particle
            # (token grain: the token IS the step; unit grain: EOS ends the unit).
            # EOS has no token_id, so commit the engine eos id as the (unused)
            # placeholder -- the assert keeps a slow/burst termination divergence loud.
            assert p.done, "burst drew EOS for a particle that did not terminate"
            return self._burst.eos_id
        return token.token_id


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


class StepLoop:
    """Per-token round-trip driver (the ground-truth path).

    Each step batches ``unit_sampler.sample(context)`` over the live particles
    via ``asyncio.gather`` -- the same batching shape as the old
    ``smc_standard`` ``asyncio.gather(p.step())`` -- so the per-token logprobs
    are recomputed from the full context every step.
    """

    def __init__(self, controller):
        self.controller = controller

    async def run(self):
        await self.controller.start()
        return await self.controller.run()


def _is_engine_llm(potential):
    """Whether ``potential`` is a single LM backed by a vLLM engine."""
    model = getattr(potential, "model", None)
    return model is not None and hasattr(model, "llm_engine")


def split_engine_target(potential):
    """Split a sampler target into ``(llm, factor, target)`` for the burst.

    Recognizes two shapes:

    * a bare engine LM -> ``(llm, None, llm)`` (unconstrained);
    * ``Product(llm, factor)`` / ``Product(factor, llm)`` where exactly one side
      is an engine LM and the other is an additive control-side ``factor`` with
      the same token type -> ``(llm, factor, product)``.

    For the product case, ``draw`` reconstructs ``Product.logw_next`` over the
    product's (possibly narrowed) vocabulary by gathering the engine-derived LM
    weights and the factor's ``logw_next`` through ``Product``'s own
    ``v1_idxs``/``v2_idxs`` -- i.e. the exact slow-path index maps -- so vocab
    narrowing (a coerced ``BoolFSA`` prunes to its accepted bytes) is handled the
    same way the slow path handles it. Returns ``None`` if not expressible.
    """
    if _is_engine_llm(potential):
        return potential, None, potential

    p1 = getattr(potential, "p1", None)
    p2 = getattr(potential, "p2", None)
    if p1 is None or p2 is None:
        return None

    if _is_engine_llm(p1) and not _is_engine_llm(p2):
        llm, factor = p1, p2
    elif _is_engine_llm(p2) and not _is_engine_llm(p1):
        llm, factor = p2, p1
    else:
        return None

    return llm, factor, potential


@dataclass
class BurstCapability:
    """Whether a configuration can be engine-accelerated, and (if not) why.

    Args:
        ok (bool): True iff the :class:`BurstLoop` can drive this configuration.
        reason (str | None): Human-readable blocker, naming the exact reason the
            burst is unavailable. ``None`` when ``ok`` is True.

    This single result backs both the runtime fallback/raise text and the
    user-facing :meth:`SMC.acceleration_report`, so the message a user sees on
    fallback is the same one introspection reports.
    """

    ok: bool
    reason: str | None = None


def burst_capability(controller):
    """Whether the ``BurstLoop`` can drive this configuration, and why not.

    The burst fast path requires:

    * the unit sampler is **burst-capable** -- it implements ``burst_draw_batch``,
      declared via ``supports_burst()`` (DirectTokenSampler with no separate
      proposal, or AWRS with no separate proposal). The Controller asks the
      sampler; it does not branch on the sampler's type.
    * the sampler target decomposes (:func:`split_engine_target`) into a single
      engine LM, optionally times one additive same-vocab factor (e.g. a coerced
      ``BoolFSA``).

    A critic does NOT disqualify the burst: it is scored/twisted/terminated by the
    same :meth:`Controller._score_advance_terminate` the slow loop uses (driven
    via ``run_sync`` from the engine-thread draw), so a per-step twist
    (``ess_threshold > 0``) or a terminal reweight (``ess_threshold == 0``) is
    handled identically to the slow path -- no critic-category gate.

    Anything else (e.g. a two-LM proposal the engine can't express) falls back to
    :class:`StepLoop`, with the ``reason`` naming the blocker.

    Returns:
        (BurstCapability): ``ok`` plus a human-readable ``reason`` on failure.
    """
    sampler = controller.unit_sampler

    if not getattr(sampler, "supports_burst", lambda: False)():
        return BurstCapability(False, _sampler_burst_blocker(sampler))

    decomposed = split_engine_target(sampler.target)
    if decomposed is None:
        return BurstCapability(False, _target_burst_blocker(sampler.target))

    # Batched (B>1): the burst draws the WHOLE population through the group-0
    # representative -- one engine forward, ``unit_sampler.burst_draw_batch``, the
    # group-0 factor fold (``draw`` / ``_run_burst``). That is correct only if every
    # group shares the same engine model, temperature, sampler kind, and in-logit
    # factor; groups may differ ONLY in prompt and critic (both handled per-group in
    # scoring). Reject a heterogeneous batch so ``accelerate='require'`` fails loud
    # and ``'auto'`` falls back to the exact per-token StepLoop instead of silently
    # drawing later groups against group 0's configuration.
    if len(controller.samplers) > 1:
        reason = _batch_homogeneity_blocker(controller.samplers, decomposed)
        if reason is not None:
            return BurstCapability(False, reason)

    return BurstCapability(True)


def _batch_homogeneity_blocker(samplers, ref_decomposed):
    """Reason a batched burst can't use the group-0 representative for all groups,
    or ``None`` if the batch is burst-homogeneous (see :func:`burst_capability`)."""
    ref_llm, ref_factor, _ = ref_decomposed
    ref_type = type(samplers[0])
    for g, s in enumerate(samplers[1:], start=1):
        if type(s) is not ref_type:
            return (
                f"group {g} sampler is {type(s).__name__} but group 0 is "
                f"{ref_type.__name__}; batched burst needs one sampler kind"
            )
        dec = split_engine_target(s.target)
        if dec is None:
            return f"group {g} target is not burst-expressible"
        llm, factor, _ = dec
        if llm.model is not ref_llm.model:
            return f"group {g} uses a different engine; batched burst needs one shared engine"
        if getattr(llm, "temperature", None) != getattr(ref_llm, "temperature", None):
            return f"group {g} temperature differs from group 0; batched burst uses one temperature"
        if (factor is None) != (ref_factor is None) or (
            factor is not None and factor is not ref_factor
        ):
            return f"group {g} has a different in-logit factor; batched burst folds one shared factor"
    return None


def _sampler_burst_blocker(sampler):
    """Human-readable reason the unit sampler is not burst-capable.

    Mirrors the per-sampler ``supports_burst()`` conditions (which the
    Controller does not otherwise branch on) so the blocker text is precise."""
    # Decision 2: the Set sampler's in-engine path is marginal/at-best and can
    # regress, so it is reported as not accelerated until the trie is vectorized.
    if type(sampler).__name__ == "SetTokenSampler":
        return "SetTokenSampler is not engine-accelerated"
    # A separate importance-sampling proposal makes the per-step weight
    # non-engine-expressible (DirectTokenSampler / AWRS with `proposal=`).
    if getattr(sampler, "proposal", None) is not None:
        return (
            "sampler uses a separate proposal; the importance weight isn't "
            "engine-expressible"
        )
    # MultiToken rides the burst iff its subunit sampler does (the subunits ARE the
    # burst draws); point at the subunit's blocker rather than the wrapper.
    subunit = getattr(sampler, "subunit_sampler", None)
    if subunit is not None and not subunit.supports_burst():
        return f"multi-token subunit sampler is not burst-capable: {_sampler_burst_blocker(subunit)}"
    return "this sampler is not engine-accelerated (no burst draw)"


def _target_burst_blocker(target):
    """Human-readable reason the sampler target does not decompose for the burst."""
    p1 = getattr(target, "p1", None)
    p2 = getattr(target, "p2", None)
    if p1 is not None and p2 is not None:
        if _is_engine_llm(p1) and _is_engine_llm(p2):
            return "two-LM product can't be expressed in one decode stream"
        if not _is_engine_llm(p1) and not _is_engine_llm(p2):
            return (
                "neither side of the product is a vLLM-engine LM — acceleration "
                "is vLLM-only"
            )
    return (
        f"target is not a single vLLM-engine LM (got {type(target).__name__}) — "
        "acceleration is vLLM-only"
    )


class BurstLoop:
    """The outer/inner SMC driver: the slow lane's burst-accelerated counterpart.

    The driver runs SMC steps over the population, entering the **fast lane** -- an
    engine *burst* = one ``AsyncVirtualLM.run_burst`` over the live particles' token
    contexts -- whenever the next step is engine-expressible. For an inexpressible
    step (a ``slow_cadence``, e.g. a periodic critic-LM forward) it drops to the
    SLOW lane: one ``StepLoop`` transition with the engine free, re-entering the
    burst after. The controller's :meth:`Controller.draw` runs per engine decode
    step, advancing/banking each particle exactly as the slow path's transition
    would.

    A burst is a **stateless enter-run-exit** unit and NEVER resamples: it pops at a
    tagged boundary -- the ESS crossing (token grain), the synced unit boundary
    (unit grain), or just before a cadence step -- by aborting the relevant rows
    (``abort_request``, not a forced EOS -- nothing extra is banked). ``run_burst``
    returns and :meth:`run` dispatches the controller-owned resample over the full
    population (and any slow-lane step), then relaunches the next burst from each
    survivor's token prefix. ESS / resample / log_ml stay entirely controller-owned
    -- the engine only produces logits.

    :class:`StepLoop` is this driver's all-slow degenerate case (the byte-exact
    parity ground truth, ``accelerate="off"``).

    Only valid when :func:`burst_capability` is ``ok`` (checked by the caller).
    """

    def __init__(self, controller):
        self.controller = controller
        # Number of engine bursts opened (>1 iff a resample / unit-sync / cadence
        # popped a burst out mid-generation) and number of slow-lane steps run (the
        # cadence handoffs). Useful for verifying the pop-out / slow-lane paths ran.
        self.n_bursts = 0
        self.n_slow_steps = 0
        self.sampler = controller.unit_sampler
        # `target` decomposes uniformly: DirectTokenSampler.target == its
        # potential; AWRS.target == potential * condition (so factor == the
        # condition). split_engine_target peels the engine LM off either.
        decomposed = split_engine_target(self.sampler.target)
        if decomposed is None:  # pragma: no cover - guarded by burst_capability
            raise ValueError("target is not burst-expressible")
        self.llm, self.factor, self.target = decomposed

        # Per-group engine LMs. All groups share ONE engine (same model / vocab /
        # eos / factor structure -- so ``self.llm`` above is a fine representative
        # for those); they differ ONLY in ``prompt_ids``. ``_context_ids`` picks the
        # row's group's prompt. For a single group this is just ``[self.llm]``.
        self.llms = [split_engine_target(s.target)[0] for s in controller.samplers]
        # Snapshot each group's prompt prefix HERE (main thread). ``prompt_ids`` is a
        # thread-local ``ContextVar`` (PromptedLLM): the initial prefill reads it on
        # the main thread, but the mid-burst in-place re-add (``_resample_burst``)
        # runs on the ``run_burst`` worker thread, where the override is invisible and
        # the live read would silently fall back to the default prompt. The prefix is
        # fixed for the whole run, so one main-thread snapshot is exact for both.
        self._group_prefixes = [list(llm.prompt_ids) for llm in self.llms]

        # Gather maps onto the product's vocab_eos (the same maps Product.logw_next
        # uses, but resolved explicitly against llm/factor so p1/p2 order is
        # irrelevant). None in the unconstrained case.
        if self.factor is None:
            self.llm_idxs = None
            self.factor_idxs = None
        else:
            self.llm_idxs = np.array(
                [self.llm.lookup[t] for t in self.target.vocab_eos]
            )
            self.factor_idxs = np.array(
                [self.factor.lookup[t] for t in self.target.vocab_eos]
            )

        # The engine token id used as the committed placeholder for an aborted /
        # EOS row (EOS carries no token_id). Fixed for the run -> resolved once. The
        # row is aborted regardless, so any of the model's EOS ids serves; we take
        # the first. A model with no EOS configured can't burst (the placeholder is
        # undefined) -> fail loud rather than IndexError deep in a burst.
        eos_idxs = list(self.llm.token_maps.eos_idxs)
        if not eos_idxs:
            raise ValueError(
                "Engine LM has no EOS token id; the burst path needs one as the "
                'committed placeholder for aborted rows. Use accelerate="off".'
            )
        self.eos_id = eos_idxs[0]

    def _context_ids(self, p):
        """Engine prompt for a particle: prompt prefix + its drawn token ids.

        A token-grain context is a flat list of ``Token``s; a unit-grain context is
        a list of units, each a list of subunit ``Token``s. Both flatten the same
        way: recurse into unit lists, take each token's id, and drop EOS sentinels
        (a finished particle is never relaunched, but a context may carry a trailing
        EOS)."""
        # Per-group prompt prefix (the row's example); the engine itself is shared.
        # Read from the main-thread snapshot, NOT the live ``prompt_ids`` ContextVar
        # (this runs on the worker thread during in-place re-add).
        ids = list(self._group_prefixes[self.controller.particles.group[p._i]])

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

    def _make_engine_callbacks(self, loop):
        """The two event-loop hops the worker-thread ``draw`` uses.

        * ``eval_factor`` -- the factor's ``logw_next(context)``: ``draw`` runs in a
          worker thread, so it schedules the coroutine back onto this loop and
          blocks for the result. The wrapped potential's ``_consume`` cache carries
          the chart incrementally across the burst, so each call is one arc-step,
          not a full context replay.
        * ``run_async`` -- run a sampler's async helper (e.g. AWRS's rejection)
          from the worker-thread draw and block -- one event-loop hop, no inner
          gather.
        """

        def run_async(coro):
            return asyncio.run_coroutine_threadsafe(coro, loop).result()

        def eval_factor(factor, context):
            # Inline if `factor.logw_next` is non-suspending (an FSA: pure-CPU
            # async); hop to the loop only if it actually awaits (a critic LM /
            # autobatched / IPC factor). Same value either way -- the hop is just
            # threading -- so this drops the per-step hop for the FSA-factor case,
            # using the SAME inline-or-hop primitive AWRS's draw uses.
            return _drive_or_hop(lambda: factor.logw_next(context), run_async)

        return eval_factor, run_async

    async def _run_burst(self, live, eval_factor, run_async, loop):
        """Enter one stateless engine burst over the ``live`` particles, run it to
        a pop-out, and return WHY it exited (an ``_EXIT_*`` reason) so the driver
        can dispatch.

        Installs the per-burst :class:`BurstContext` (the sampler-facing surface)
        wrapped in the Controller's private :class:`_Burst` runtime, runs the
        engine decode loop in a worker thread (so this event loop stays free to
        service the ``draw``'s ``run_coroutine_threadsafe`` hops), and reads the
        ``_Burst.exit_reason`` ``draw`` set before clearing the runtime.
        """
        controller = self.controller
        self.n_bursts += 1
        prompts = [self._context_ids(p) for p in live]
        # The engine decode budget for ONE burst, asked of the sampler: a token
        # sampler bursts until the longest particle's token budget (free-running,
        # pops at the ESS crossing); a unit sampler bursts one unit round (capped
        # at the subunit budget, pops each row at its boundary).
        max_steps = self.sampler.burst_max_steps(live)

        ctx = BurstContext(
            factor=self.factor,
            target=self.target,
            llm_idxs=self.llm_idxs,
            factor_idxs=self.factor_idxs,
            eval_factor=eval_factor,
            run_async=run_async,
        )
        burst = _Burst(
            particles=live,
            llm=self.llm,
            eos_id=self.eos_id,
            ctx=ctx,
            context_ids=self._context_ids,
        )
        # A synchronized (unit-grain) burst always runs exactly one SMC step (one
        # unit) per row and hands back at the synced boundary for the controller-
        # owned ESS test -- so its exit reason is fixed up front. A free-running
        # (token-grain) burst stays ``_EXIT_TERMINATED``: it resamples in place at
        # ESS crossings without exiting, ending only when every row terminates (or a
        # cadence pops it via ``_EXIT_SLOW_STEP``).
        if not self.sampler.burst_free_running():
            burst.exit_reason = _EXIT_UNIT_SYNC
        controller._burst = burst
        try:
            await loop.run_in_executor(
                None,
                lambda: self.llm.model.run_burst(
                    prompts=prompts,
                    control=controller,
                    max_steps=max_steps,
                ),
            )
            return controller._burst.exit_reason
        finally:
            controller._burst = None

    async def run(self):
        """The outer driver: per iteration, run the next step on the fast lane (an
        engine burst) or the slow lane (one per-token transition), then dispatch the
        controller-owned ESS/resample. Repeat until every particle is done.

        Each iteration picks the lane for the live rows' NEXT step:

        * **inexpressible** (the ``slow_cadence`` is due -- e.g. a periodic
          critic-LM forward): run ONE slow-lane transition for the live rows
          (:meth:`_slow_step`), engine free, then the controller-owned ESS test.
        * **expressible** (the common case): enter a stateless burst that runs a
          run of expressible steps and pops at a tagged boundary -- the ESS
          crossing (token grain), the synced unit boundary (unit grain), or just
          before the next cadence step (``_EXIT_SLOW_STEP``, handled by the next
          iteration's burst-entry check). Then the controller-owned ESS test.

        A burst that ends because all its rows terminated needs no dispatch
        (``_maybe_resample`` would be a no-op). ESS / resample / log_ml stay
        entirely controller-owned -- the burst and the slow-lane transition are the
        two step-runners, nothing more.
        """
        controller = self.controller
        await controller.start()

        loop = asyncio.get_running_loop()
        eval_factor, run_async = self._make_engine_callbacks(loop)

        while any(not p.done for p in controller.particles):
            live = [p for p in controller.particles if not p.done]
            if self._next_step_is_slow(live):
                # The next step is engine-inexpressible for the (lockstep) live
                # rows -- run it on the slow lane, then the controller-owned ESS.
                await self._slow_step(live)
                controller._maybe_resample()
                continue

            reason = await self._run_burst(live, eval_factor, run_async, loop)
            # Resample triggers at a burst exit: the synced boundary of a unit-grain
            # round, or just before a cadence step (the next iteration runs that step
            # on the slow lane). (A token-grain ESS crossing does NOT exit -- it
            # resamples in place mid-burst.) A terminated exit needs no dispatch.
            if reason == _EXIT_UNIT_SYNC:
                # One record entry for the completed unit round (the grain's single
                # SMC step), before its resample -- keeps the lazy resample tag order.
                # Not on _EXIT_SLOW_STEP: that step is recorded by the next
                # iteration's _slow_step.
                controller._record_step()
            if reason in (_EXIT_UNIT_SYNC, _EXIT_SLOW_STEP):
                controller._maybe_resample()

        return controller.particles

    def _next_step_is_slow(self, live):
        """Whether ANY live row's next step is engine-inexpressible (a slow-lane
        cadence) -- the burst-entry check. Delegates to the same
        :meth:`Controller._cadence_due` boundary-predicate test ``draw`` applies
        mid-burst, so entry and pop-out fire on the identical condition. ``False``
        without a cadence (the burst owns every step)."""
        return self.controller._cadence_due(live)

    async def _slow_step(self, live):
        """Run ONE per-token slow-lane transition for the live rows -- the
        inexpressible step (a critic-LM forward + reweight, a second LM, ...) with
        the engine free. Mirrors one iteration of the slow ``StepLoop``: untwist the
        live rows, then their shared per-step transition (which scores / advances /
        twists / terminates identically); the driver runs ESS after, so the cadence
        step has the exact per-step timing of the all-slow loop.

        Closing each row's cadence run (``reset_run``) after the draw is what lets a
        :class:`~genlm.control.sampler.unit.BoundaryPredicate` like
        ``FixedLengthBoundary`` re-arm: the next run starts empty, so the predicate
        does not immediately re-fire on the same buffer."""
        controller = self.controller
        self.n_slow_steps += 1
        controller.particles.untwist_subset([p._i for p in live])
        await asyncio.gather(*[controller._draw_and_score(p) for p in live])
        controller._record_step()
        for p in live:
            p.reset_run()
