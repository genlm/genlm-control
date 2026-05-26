"""The SMC controller: a single, engine-independent owner of the entire SMC algorithm.

This module replaces the previous llamppl ``smc_standard`` + ``SequenceModel``
coupling. The controller owns the particle population, the per-step transition, the
ESS test, resampling/forking and the log marginal likelihood accumulation --
*always*. It is exact per token: there is no segment-graining and no
hard/soft constraint fork. ``logw_next`` is one operation.

Two drivers turn the population: the slow per-token Python loop (this file's
``StepLoop``) fuses shaping + drawing inside the sampler's ``transition``; the
engine-accelerated ``BurstLoop`` instead calls :meth:`Controller.draw` row-wise
on raw engine logits (``shape`` is currently an identity pre-pass). Either way,
resample / fork / ESS / log_ml are controller-owned and NEVER delegated to the
engine.
"""

import asyncio

import numpy as np
import torch

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.sampler.resampling import get_resampling_fn
from genlm.control.sampler.smc_record import SMCRecord, string_for_serialization


def logsumexp(nums):
    """logsumexp matching llamppl.util.logsumexp exactly (for parity)."""
    nums = np.asarray(nums)
    if np.all(nums == -np.inf):
        return -np.inf
    m = np.max(nums)
    return np.log(np.sum(np.exp(nums - m))) + m


# ---------------------------------------------------------------------------
# The population
# ---------------------------------------------------------------------------


class Population:
    """The SMC particle population, stored columnar.

    The per-particle scalars that the algorithm reads in bulk (ESS, resample,
    log_ml) live as parallel numpy arrays -- the *single source of truth* -- so
    those tests are array ops, not a fresh ``np.array([p.logw for p in ...])``
    rebuilt every step. The ragged per-particle state (token ``contexts`` and the
    carried factor ``states``) stays as Python lists.

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
        states: carried stateful-potential state for the burst's factor (the
            chart of the in-engine product's control-side factor), advanced one
            token at a time instead of replaying the context; ``None`` on the slow
            path / when there is no factor. Kept in lockstep with ``contexts``.
    """

    __slots__ = (
        "n",
        "logw",
        "logp",
        "twist_amount",
        "done",
        "max_tokens_left",
        "contexts",
        "states",
    )

    def __init__(self, n, max_tokens):
        self.n = n
        self.logw = np.zeros(n)
        self.logp = np.zeros(n)
        self.twist_amount = np.zeros(n)
        self.done = np.zeros(n, dtype=bool)
        self.max_tokens_left = np.full(n, max_tokens, dtype=np.int64)
        self.contexts = [[] for _ in range(n)]
        self.states = [None] * n

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
        shallow-copied (their token elements are immutable); factor states are
        immutable per step so they can be shared by reference -- both matching the
        old ``clone``."""
        idx = ancestor_indices
        self.logw = self.logw[idx]
        self.logp = self.logp[idx]
        self.twist_amount = self.twist_amount[idx]
        self.done = self.done[idx]
        self.max_tokens_left = self.max_tokens_left[idx]
        self.contexts = [list(self.contexts[i]) for i in idx]
        self.states = [self.states[i] for i in idx]


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

    @property
    def factor_state(self):
        return self._pop.states[self._i]

    @factor_state.setter
    def factor_state(self, v):
        self._pop.states[self._i] = v

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
    supplies the LM half (the batched ``lm_batch``) and each particle carries the
    factor's state (``factor_state``), both passed to ``burst_draw_batch``; this
    object exposes the two substitutions that stand in for the slow path's async
    potential calls:

    * :meth:`product_logws` -- reconstruct ``Product(llm, factor).logw_next`` over
      the target vocab from the engine LM weights + carried factor state (the
      DirectTokenSampler proposal);
    * :meth:`run_sync` -- run a sampler's own async helper (e.g. AWRS's rejection)
      to completion on the driver's event loop.

    ``factor`` (the additive control-side factor / AWRS boolean condition, or
    ``None``) is exposed directly so the sampler can advance it. The Controller's
    row-map / pop-out / engine-LM runtime is deliberately NOT on this object --
    a sampler only ever sees this narrow, named interface, never the Controller's
    private burst state.
    """

    __slots__ = ("factor", "_target", "_llm_idxs", "_factor_idxs", "_eval_factor", "_run_async")

    def __init__(self, factor, target, llm_idxs, factor_idxs, eval_factor, run_async):
        self.factor = factor
        self._target = target
        self._llm_idxs = llm_idxs
        self._factor_idxs = factor_idxs
        self._eval_factor = eval_factor
        self._run_async = run_async

    def factor_logws(self, state):
        """The factor's stateful ``logw_next``, evaluated synchronously from the
        carried ``state`` -- the burst stand-in for the slow path's
        ``factor.logw_next(context)``."""
        return self._eval_factor(self.factor, state)

    def product_logws(self, lm_weights, state):
        """Reconstruct ``Product(llm, factor).logw_next`` over the target vocab
        from the engine LM weights (``lm_weights``, a length-``V+1`` array in the
        LM's vocab-eos order -- one row of the batched ``_process_logw_next_batch``)
        and the carried factor ``state``.

        Gathers through ``Product``'s own ``v1_idxs``/``v2_idxs`` (resolved here
        as ``llm_idxs``/``factor_idxs``) -- the exact slow-path index maps -- with
        the engine LM half substituting the reprefilled one, so vocab narrowing
        matches the slow path.
        """
        factor_logws = self.factor_logws(state)
        return self._target.make_lazy_weights(
            lm_weights[self._llm_idxs] + factor_logws.weights[self._factor_idxs]
        )

    def run_sync(self, coro):
        """Run a sampler's async helper to completion on the driver's event loop
        and block for the result -- one event-loop hop, no inner gather."""
        return self._run_async(coro)


class _Burst:
    """The Controller's private per-engine-burst runtime (set by the BurstLoop
    for the duration of one ``run_burst``; ``None`` outside a burst).

    Holds what the Controller's ``draw`` needs -- the live particle row-map, the
    engine LM (to batch the rows' ``lm_batch``), the EOS id, and the mutable
    pop-out flag -- plus the sampler-facing :class:`BurstContext` (``ctx``).
    Samplers never see this object; only ``ctx``.
    """

    __slots__ = ("particles", "llm", "eos_id", "pop_out", "ctx", "_run_async")

    def __init__(self, particles, llm, eos_id, ctx, run_async):
        self.particles = particles
        self.llm = llm
        self.eos_id = eos_id
        self.pop_out = False
        self.ctx = ctx
        self._run_async = run_async

    def run_sync(self, coro):
        """Drive a coroutine to completion on the driver's event loop and block
        for the result (one event-loop hop). The Controller uses this to evaluate
        the async critic from the synchronous engine-thread ``draw``, the same hop
        ``BurstContext.run_sync`` gives samplers."""
        return self._run_async(coro)


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
    ):
        assert max_tokens > 0
        self.unit_sampler = unit_sampler
        self.critic = critic
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold
        self.max_tokens = max_tokens
        self.twist_with_critic = twist_with_critic
        self.resample_fn = get_resampling_fn(resampling_method)
        self.verbosity = verbosity

        self.particles = Population(n_particles, max_tokens)
        self.record = SMCRecord(n_particles) if record else None

        # Constant RHS of the ESS resample predicate (``log(ess_threshold * n)``),
        # compared against ``log(ESS)`` every step -- precomputed once here instead
        # of recomputing two ``np.log`` calls on the burst hot path.
        with np.errstate(divide="ignore"):
            self._ess_log_threshold = np.log(ess_threshold) + np.log(n_particles)

        # The :class:`_Burst` runtime set by the BurstLoop for the duration of a
        # single engine burst; read by the ``shape`` / ``draw`` callbacks the
        # engine invokes row-wise. ``None`` outside a burst.
        self._burst = None

    # -- the per-step transition for ONE particle ----------------------------
    #
    # In the slow path shaping + drawing are fused inside the sampler's
    # ``transition`` coroutine (which computes logw_next, draws, and returns the
    # importance weight). The burst path instead calls ``shape`` / ``draw``
    # row-wise; both paths apply the identical SMC weight/termination math via
    # the ``_terminate_*`` helpers below.

    async def _draw_and_score(self, p):
        """The slow per-step transition for one particle.

        Draws a token + weight increment from the sampler, then applies the
        shared SMC scoring/twist/termination math (:meth:`_score_advance_terminate`).

        ``unit_sampler.transition`` returns ``(to_append, logw, logp)`` where
        ``to_append`` is the list of items to extend the particle context with
        (a single token for token samplers; a unit -- possibly split around a
        trailing EOS -- for the multi-token unit sampler).
        """
        to_append, logw, logp = await self.unit_sampler.transition(p.context)
        await self._score_advance_terminate(p, to_append, logw, logp)

    def _advance_no_critic(self, p, to_append, logw, logp):
        """Sync score + advance + terminate for the NO-CRITIC path. Shared by the
        slow :meth:`_score_advance_terminate` (its no-critic branch) and the burst
        bookkeeping (:meth:`_bank_burst_draw`) -- the burst calls it directly so a
        no-critic step needs NO event-loop hop, which would otherwise be the new
        per-particle bottleneck once the LM draw is batched."""
        p.score(logw)
        p.logp += logp
        p.context.extend(to_append)
        if p.logw == float("-inf"):
            p.finish()
        else:
            self._terminate_if_done(p)

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
            twist_amt = await self.critic.score(p.context)
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
                twist_amt = await self.critic.score(p.context)
            p.score(twist_amt)
            return

    def _terminate_if_done(self, p):
        """Synchronous post-draw termination for the NO-CRITIC path.

        Exactly the tail of ``_draw_and_score`` when ``self.critic`` is None:
        verbosity print, budget decrement, and finish on budget exhaustion or a
        terminal EOS. Shared with the burst driver's synchronous ``draw`` so the
        two paths terminate particles identically.
        """
        assert not self.critic
        if self.verbosity > 0:
            print(self._repr_particle(p))
        p.max_tokens_left -= 1
        if p.max_tokens_left == 0 or self._is_terminal(p):
            p.finish()

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
        """Initialize particles from the sampler's start weight.

        Mirrors ``SequenceModel.start``: scores every particle by the empty
        sequence's prefix weight under the target potential.
        """
        start_w = await self.unit_sampler.start_weight()
        if start_w == float("-inf"):
            raise ValueError(
                "Start weight is -inf (log(0)). This is likely because a potential assigns zero weight to "
                "the empty sequence under `prefix`, which violates the potential contract."
            )
        for p in self.particles:
            p.score(start_w)

    async def run(self):
        """Run the slow (ground-truth) SMC loop to completion.

        Exact per-token re-implementation of llamppl's ``smc_standard``: untwist
        all particles, step the live ones (batched), record, then ESS-test and
        resample. Returns the final particle population.
        """
        n = self.n_particles
        ancestor_indices = list(range(n))
        did_resample = False

        while any(not p.done for p in self.particles):
            self.particles.untwist_all()

            await asyncio.gather(
                *[self._draw_and_score(p) for p in self.particles if not p.done]
            )

            if self.record is not None:
                if len(self.record.history) == 0:
                    self.record.add_init(self.particles)
                elif did_resample:
                    self.record.add_resample(ancestor_indices, self.particles)
                else:
                    self.record.add_smc_step(self.particles)

            did_resample, ancestor_indices = self._maybe_resample(
                sort_ancestors=self.record is not None
            )

        return self.particles

    def _ess_below_threshold(self, normalized_weights):
        """The ESS resample predicate, shared by the test-only ``_ess_crosses``
        and the mutating ``_maybe_resample`` so both decide identically.

        Given the (log) normalized weights, returns whether the effective sample
        size has fallen below ``ess_threshold * n_particles``. ``-logsumexp(2w)``
        is ``log(ESS)``; the RHS ``log(ess_threshold * n)`` is the constant
        precomputed in ``__init__``.
        """
        return -logsumexp(normalized_weights * 2) < self._ess_log_threshold

    def _ess_crosses(self):
        """Whether the ESS test triggers a resample on the current population.

        Test only -- does not mutate. Uses the identical predicate as
        ``_maybe_resample`` so the burst's pop-out decision matches the slow
        path's resample decision exactly. Returns ``False`` when all weights are
        ``-inf`` (matching the slow path's ``continue``).
        """
        W = self.particles.logw
        if np.all(W == -np.inf):
            return False
        normalized_weights = W - logsumexp(W)
        return self._ess_below_threshold(normalized_weights)

    def _maybe_resample(self, sort_ancestors=False):
        """Run the ESS test on the current population and resample if it crosses.

        This is the *only* implementation of the ESS/resample math; both the slow
        ``run`` loop and the ``BurstLoop`` call it so the two paths are
        bit-identical here. Mutates ``self.particles`` on a resample.

        Returns ``(did_resample, ancestor_indices)``.
        """
        n = self.n_particles
        W = self.particles.logw
        if np.all(W == -np.inf):
            return False, list(range(n))

        w_sum = logsumexp(W)
        normalized_weights = W - w_sum

        if self._ess_below_threshold(normalized_weights):
            probs = np.exp(normalized_weights)
            ancestor_indices = self.resample_fn(probs).tolist()

            if sort_ancestors:
                ancestor_indices.sort()

            self.particles.reindex(ancestor_indices)
            self.particles.logw[:] = w_sum - np.log(n)

            return True, ancestor_indices

        return False, list(range(n))

    def save_record(self, json_path):
        """Write the SMC record JSON, matching the old smc_standard json_file path."""
        if self.record is None:
            return
        with open(json_path, "w") as f:
            f.write(self.record.to_json())
        print(f"Saved record to {json_path}")

    # -- the engine callbacks shape / draw (used only by the BurstLoop) ------
    #
    # These reproduce, on raw engine logits, exactly what ``_draw_and_score``
    # does for a burst-capable sampler over an engine LM (and, with an additive
    # control-side factor, the constrained extension). The engine IS the language
    # model, so ``shape`` adds only the non-LM factor's per-token log-weights; the
    # draw works in the same control-side V+1 vocabulary as the slow path. Each
    # sampler's ``burst_draw_batch`` decides whether it reuses the slow path's
    # numpy RNG (tight warm-KV-only parity) or a batched torch draw (no-bias) --
    # see the per-sampler docstrings in ``token.py``.
    #
    # A live burst is described by ``self._burst`` (a :class:`_Burst` set by
    # the BurstLoop): ``particles`` (the live row-map), ``llm`` (the engine
    # PromptedLLM, supplying ``_process_logw_next_batch`` / ``_maybe_temper``),
    # ``eos_id`` (the engine token id to force/observe EOS), the mutable
    # ``pop_out`` flag (once True, every subsequent draw forces EOS -- a control
    # signal, never banked -- so the burst ends for resampling), and ``ctx``
    # (the sampler-facing :class:`BurstContext`).

    def _burst_particle(self, request_id):
        """Map a vLLM internal request id (``"{external}-{8 chars}"``) to its
        particle via the external index the BurstLoop assigned."""
        external = request_id.rsplit("-", 1)[0]
        return self._burst.particles[int(external)]

    def shape(self, logits, request_ids) -> None:
        """Shape each row's raw engine logits into its proposal in place.

        For the unconstrained ``DirectTokenSampler`` there is no control-side
        factor, so shaping is the identity and the per-step weight increment is
        zero.

        For a ``DirectTokenSampler(Product(llm, factor))`` with an additive
        factor sharing the LLM's vocabulary, the proposal is the product
        ``llm.logw_next + factor.logw_next``. That sum is computed in the
        control-side V+1 space inside :meth:`draw` (the unambiguous reference --
        it *is* ``Product.logw_next``), so ``shape`` is also a no-op here: there
        is no engine-token-id-order inverse mapping to get wrong, and no
        double-counting. The per-step weight (``logsumexp`` of the product) is
        banked once in :meth:`draw`.
        """
        # Identity for both the unconstrained and additive-factor cases; the
        # full proposal (LM + factor) and its normalizer are formed in ``draw``.

    def draw(self, logits, request_ids):
        """Draw one token per live row, reproducing the slow path's per-step draw.

        The expensive 50k-vocab LM processing runs ONCE for the whole batch
        on-device (``_process_logw_next_batch``); the sampler's
        :meth:`~TokenSampler.burst_draw_batch` then turns that batched proposal
        into one ``(token, logw, logp, new_factor_state)`` per live row -- the
        burst analog of ``sample`` (see the per-sampler docstrings in
        ``token.py`` for how each reconstructs ``Product(llm, factor).logw_next``
        from the carried factor state and which RNG stream it draws with). The
        Controller then banks the weight, advances, and terminates each particle
        (:meth:`_bank_burst_draw`). When popping out, force EOS for every row
        WITHOUT banking it.

        The engine's ``SamplingMetadata`` is intentionally not taken: this draw
        works entirely in the control-side V+1 vocab (temperature via
        ``_maybe_temper``, no top-k/p), so the engine's per-row sampling params
        play no role.
        """
        w = self._burst
        sampler = self.unit_sampler
        parts = [self._burst_particle(rid) for rid in request_ids]
        # Rows still being driven (not popped-out for resample, not finished).
        # Popped-out / done rows are forced to EOS (control signal, never banked).
        out = [w.eos_id] * len(parts)
        live = [i for i, p in enumerate(parts) if not (w.pop_out or p.done)]

        if live:
            # Untwist last step's provisional critic twist before this step's score
            # (mirrors the slow loop's per-step untwist; no-op without a critic).
            # Vectorized over the live population rows -- no per-particle loop.
            self.particles.untwist_subset([parts[i]._i for i in live])

            # The expensive 50k-vocab LM processing happens ONCE for the whole
            # batch, on-device -- never per row (the per-row Python loop was
            # ~5-19x the engine's native decode cost). The sampler then draws all
            # rows from this batched proposal (Direct vectorizes; AWRS/Set loop
            # their cheap per-particle control over the [V+1] rows).
            llm = w.llm
            lm_batch = llm._process_logw_next_batch(llm._maybe_temper(logits[live].float()))
            tokens, logws, logps, new_states = sampler.burst_draw_batch(
                lm_batch,
                [parts[i].factor_state for i in live],
                [parts[i].context for i in live],
                w.ctx,
            )
            for k, i in enumerate(live):
                out[i] = self._bank_burst_draw(
                    parts[i], tokens[k], logws[k], logps[k], new_states[k]
                )

        # End-of-step ESS test (same predicate the slow path applies after every
        # token). If it crosses, arm pop-out: the next engine step forces EOS for
        # all live rows so the burst ends and the BurstLoop resamples. Skip while
        # already popping out (no real step happened).
        if not w.pop_out and self._ess_crosses():
            w.pop_out = True

        return torch.tensor(out, dtype=torch.int64, device=logits.device)

    def _bank_burst_draw(self, p, token, logw, logp, new_state):
        """Per-particle SMC bookkeeping after a burst draw (batched or per-row);
        returns the engine token id to emit for this row. No-critic uses the sync
        :meth:`_advance_no_critic` (no event-loop hop); the critic path drives the
        shared async :meth:`_score_advance_terminate` via the burst's hop."""
        p.factor_state = new_state
        if self.critic is None:
            self._advance_no_critic(p, [token], logw, logp)
        else:
            self._burst.run_sync(self._score_advance_terminate(p, [token], logw, logp))

        if isinstance(token, EndOfSequence):
            # Emitting EOS ends the engine request, so the particle must have
            # terminated in lockstep -- turns a non-singleton-EOS slow/burst
            # divergence into a loud failure rather than a silent length/log_ml gap.
            assert p.done, "burst emitted EOS for a particle that did not terminate"
            return self._burst.eos_id
        if p.done:
            # Critic/budget terminated on a non-EOS token: pop the row out now (the
            # drawn token is already in the context -- bit-identical to the slow
            # path simply ceasing to step a finished particle).
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


def can_burst(controller):
    """Whether the ``BurstLoop`` can drive this configuration.

    The burst fast path requires:

    * the unit sampler is **burst-capable** -- it implements ``burst_draw_batch``,
      declared via ``supports_burst()`` (DirectTokenSampler with no separate
      proposal, or AWRS over a sync-stateful boolean condition). The Controller
      asks the sampler; it does not branch on the sampler's type.
    * the sampler target decomposes (:func:`split_engine_target`) into a single
      engine LM, optionally times one additive same-vocab factor (e.g. a coerced
      ``BoolFSA``).

    A critic does NOT disqualify the burst: it is scored/twisted/terminated by the
    same :meth:`Controller._score_advance_terminate` the slow loop uses (driven
    via ``run_sync`` from the engine-thread draw), so a per-step twist
    (``ess_threshold > 0``) or a terminal reweight (``ess_threshold == 0``) is
    handled identically to the slow path -- no critic-category gate.

    Anything else (e.g. a two-LM proposal the engine can't express) falls back to
    :class:`StepLoop`.
    """
    sampler = controller.unit_sampler
    if not getattr(sampler, "supports_burst", lambda: False)():
        return False
    return split_engine_target(sampler.target) is not None


class BurstLoop:
    """Drives consecutive SMC steps inside a vLLM engine via the engine callbacks.

    Opens a *burst* = one ``AsyncVirtualLM.run_burst`` call over the live
    particles' token contexts. The controller's :meth:`Controller.shape` /
    :meth:`Controller.draw` run per engine
    decode step, advancing each particle exactly as the slow path's transition
    would. The burst itself NEVER resamples: when the controller's end-of-step ESS test
    crosses the threshold it arms pop-out, the next step forces EOS for every live
    row (a control signal, never banked), and ``run_burst`` returns. The driver
    then runs the controller-owned ESS test + resample/fork and relaunches the next
    burst from each surviving particle's token prefix. ESS / resample / log_ml
    stay entirely controller-owned -- the engine only produces logits.

    Only valid when :func:`can_burst` is True (checked by the caller).
    """

    def __init__(self, controller):
        self.controller = controller
        # Number of engine bursts opened (>1 iff a resample popped a burst out
        # mid-generation). Useful for verifying the pop-out/relaunch path ran.
        self.n_bursts = 0
        self.sampler = controller.unit_sampler
        # `target` decomposes uniformly: DirectTokenSampler.target == its
        # potential; AWRS.target == potential * condition (so factor == the
        # condition). split_engine_target peels the engine LM off either.
        decomposed = split_engine_target(self.sampler.target)
        if decomposed is None:  # pragma: no cover - guarded by can_burst
            raise ValueError("target is not burst-expressible")
        self.llm, self.factor, self.target = decomposed

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

    def _context_ids(self, p):
        """Engine prompt for a particle: prompt prefix + its drawn token ids.

        Drops a trailing control EOS sentinel (a finished particle is never
        relaunched, but a context may carry EOS as its last element)."""
        ids = list(self.llm.prompt_ids)
        for tok in p.context:
            if isinstance(tok, EndOfSequence):
                continue
            ids.append(tok.token_id)
        return ids

    async def run(self):
        controller = self.controller
        await controller.start()

        # Seed each particle's carried factor state (empty context == state0).
        # Across bursts the state is carried (cloned on resample), never
        # re-derived, so only freshly-started particles need seeding.
        if self.factor is not None:
            for p in controller.particles:
                if p.factor_state is None:
                    p.factor_state = self.factor.state0()

        eos_idxs = list(self.llm.token_maps.eos_idxs)
        eos_id = eos_idxs[0]
        loop = asyncio.get_running_loop()

        # A factor is evaluated from its CARRIED state (advanced one token at a
        # time), not by replaying the context: `logw_next_from_state` has a sync
        # body (no `asyncio.gather` over the vocabulary -- the dominant per-step
        # cost), and `draw` runs in the worker thread below, so it schedules this
        # one coroutine back onto the event loop and blocks for the result.
        def eval_factor(factor, state):
            fut = asyncio.run_coroutine_threadsafe(
                factor.logw_next_from_state(state), loop
            )
            return fut.result()

        def run_async(coro):
            # Run a sampler's async helper (e.g. AWRS's rejection) from the
            # worker-thread draw and block for the result -- one event-loop hop,
            # no inner gather.
            return asyncio.run_coroutine_threadsafe(coro, loop).result()

        while any(not p.done for p in controller.particles):
            self.n_bursts += 1
            live = [p for p in controller.particles if not p.done]
            prompts = [self._context_ids(p) for p in live]
            # Each particle keeps its own budget; the engine burst just needs to
            # be long enough to reach the longest particle's budget (plus one
            # step to let a budget-exhausted row pop out via forced EOS).
            max_steps = max(p.max_tokens_left for p in live) + 1

            # The sampler-facing context (named methods, no Controller runtime)
            # wrapped in the Controller's private per-burst runtime. The
            # Controller reads `_Burst`; the sampler only ever sees `.ctx`.
            ctx = BurstContext(
                factor=self.factor,
                target=self.target,
                llm_idxs=self.llm_idxs,
                factor_idxs=self.factor_idxs,
                eval_factor=eval_factor,
                run_async=run_async,
            )
            controller._burst = _Burst(
                particles=live, llm=self.llm, eos_id=eos_id, ctx=ctx,
                run_async=run_async,
            )
            try:
                # Run the engine step loop in a worker thread so this event loop
                # stays free to service the factor's run_coroutine_threadsafe.
                await loop.run_in_executor(
                    None,
                    lambda: self.llm.model.run_burst(
                        prompts=prompts,
                        control=controller,
                        max_steps=max_steps,
                        eos_token_ids=eos_idxs,
                        temperature=self.llm.temperature,
                    ),
                )
            finally:
                controller._burst = None

            # Controller-owned ESS test + resample/fork over the FULL population.
            controller._maybe_resample(
                sort_ancestors=controller.record is not None
            )

        return controller.particles
