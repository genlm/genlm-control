"""The SMC hub: a single, engine-independent owner of the entire SMC algorithm.

This module replaces the previous llamppl ``smc_standard`` + ``SequenceModel``
coupling. The hub owns the particle population, the per-step transition, the
ESS test, resampling/forking and the log marginal likelihood accumulation --
*always*. It is exact per token: there is no segment-graining and no
hard/soft constraint fork. ``logw_next`` is one operation.

The hub is designed so that the per-step work splits into two phases that can
be invoked either by the slow Python loop (this file's ``StepLoop``) or,
later, row-wise by in-engine arms keyed on a ``request_id -> particle`` map:

1. **shape the proposal**: turn a particle's state into the proposal
   log-distribution it will sample its next token from.
2. **draw + weight + advance**: sample a token, compute the importance-weight
   increment, advance the sampler/critic transition state.

Resample / fork / ESS / log_ml are hub-owned and are NEVER delegated.

See :class:`EngineControl` for the protocol the future window driver / engine
arms will call back into.
"""

import asyncio
from typing import Protocol, runtime_checkable

import numpy as np

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


class Particle:
    """One SMC particle: an engine-independent record of a partial sequence.

    Attributes:
        context (list): The tokens (or units) sampled so far. Token objects are
            immutable, so resampling can shallow-copy the list.
        logw (float): The particle's current (twisted) log importance weight.
        logp (float): Accumulated log-probability of the sampler's random choices.
        twist_amount (float): The amount currently *added* to ``logw`` by twisting
            with the critic; subtracted back out (``untwist``) before each step and
            on termination, exactly as in the llamppl ``Model.twist``/``untwist``.
        done (bool): Whether this particle has finished stepping.
        max_tokens_left (int): Remaining token budget for this particle.
    """

    __slots__ = (
        "context",
        "logw",
        "logp",
        "twist_amount",
        "done",
        "max_tokens_left",
        "factor_state",
    )

    def __init__(self, max_tokens):
        self.context = []
        self.logw = 0.0
        self.logp = 0.0
        self.twist_amount = 0.0
        self.done = False
        self.max_tokens_left = max_tokens
        # Carried stateful-potential state for the window's factor (the chart of
        # the in-engine product's control-side factor), advanced one token at a
        # time instead of replaying the context. None on the slow path / when
        # there is no factor. Kept in lockstep with `context` (cloned together).
        self.factor_state = None

    # -- weight bookkeeping, mirroring llamppl.modeling.Model exactly --

    def score(self, amt):
        self.logw += amt

    def twist(self, amt):
        self.twist_amount += amt
        self.score(amt)

    def untwist(self):
        self.score(-self.twist_amount)
        self.twist_amount = 0.0

    def finish(self):
        self.untwist()
        self.done = True

    # -- viz adapter (the record reads .weight + .string_for_serialization()) --

    @property
    def weight(self):
        return self.logw

    def string_for_serialization(self):
        return string_for_serialization(self.context)

    def clone(self):
        """Shallow clone for resampling/forking.

        The context list is shallow-copied (its token elements are immutable),
        and all scalar bookkeeping is copied. This is the hub analogue of the
        ``copy.deepcopy`` llamppl performed per resampled particle, but without
        deep-copying immutable tokens.
        """
        cpy = Particle.__new__(Particle)
        cpy.context = list(self.context)
        cpy.logw = self.logw
        cpy.logp = self.logp
        cpy.twist_amount = self.twist_amount
        cpy.done = self.done
        cpy.max_tokens_left = self.max_tokens_left
        # The factor state is immutable per step (advance returns a fresh chart),
        # so it can be shared by reference across the clone -- it stays in lockstep
        # with the (copied) context.
        cpy.factor_state = self.factor_state
        return cpy


# ---------------------------------------------------------------------------
# The engine-control contract (shape so the future window driver plugs in)
# ---------------------------------------------------------------------------


@runtime_checkable
class EngineControl(Protocol):
    """Protocol the future in-engine window driver / engine arms call back into.

    The slow driver does NOT use this protocol (it fuses shaping + drawing
    inside the sampler's ``sample`` coroutine). It exists so that the window
    driver can hand consecutive steps to the engine, which will then invoke
    these two methods row-wise on raw logits, keyed by ``request_ids`` mapping
    engine rows to hub particles.

    Implementations mutate logits in place (``shape``) and return a sampled
    token id per row (``draw``); ``draw`` may force EOS for pop-out. The hub's
    resample / fork / ESS / log_ml machinery is never delegated through this
    protocol.
    """

    def shape(self, logits, request_ids) -> None:  # pragma: no cover - contract
        """Mutate ``logits[i]`` in place into the proposal log-distribution for
        the particle mapped to ``request_ids[i]``."""
        ...

    def draw(self, logits, request_ids, sampling_metadata):  # pragma: no cover
        """Return a sampled token id per row for ``request_ids``; may force EOS
        for pop-out. Weight/advance bookkeeping is applied to the mapped
        particles as a side effect."""
        ...


# ---------------------------------------------------------------------------
# The transition (shared by all samplers)
# ---------------------------------------------------------------------------


class Controller:
    """Owns the SMC algorithm: population, transition, ESS, resample, log_ml.

    A "sampler" collapses to a single per-step transition
    ``state -> (token, logw[, logp])`` that this hub calls. The hub is owned
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

        self.particles = [Particle(max_tokens) for _ in range(n_particles)]
        self.record = SMCRecord(n_particles) if record else None

        # Set by the BurstLoop for the duration of a single engine window;
        # read by the EngineControl ``shape`` / ``draw`` callbacks. ``None``
        # outside a window.
        self._window = None

    # -- phase 1: shape + draw + weight + advance for ONE particle -----------
    #
    # In the slow path these two phases are fused inside the sampler's
    # ``sample`` coroutine (which computes logw_next, draws, and returns the
    # importance weight). The window driver will instead call EngineControl
    # .shape / .draw row-wise; both paths converge on ``_advance_particle``
    # below, which applies the identical SMC math to the population.

    async def _draw_and_score(self, p):
        """The shared per-step transition for one particle.

        Calls the sampler's transition to draw a token + weight increment,
        scores the particle, advances the critic twist, and handles
        termination. This is the SMC math, identical regardless of where the
        next-token logprobs come from.

        ``unit_sampler.transition`` returns ``(to_append, logw, logp)`` where
        ``to_append`` is the list of items to extend the particle context with
        (a single token for token samplers; a unit -- possibly split around a
        trailing EOS -- for the multi-token unit sampler).
        """
        to_append, logw, logp = await self.unit_sampler.transition(p.context)
        p.score(logw)
        p.logp += logp
        p.context.extend(to_append)

        if p.logw == float("-inf"):
            if self.critic:
                assert p.twist_amount != float("-inf")
            p.finish()
            return

        if not self.critic:
            self._terminate_if_done(p)
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
        terminal EOS. Shared with the window driver's synchronous ``draw`` so the
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

    # -- the hub-owned loop --------------------------------------------------

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
            for p in self.particles:
                p.untwist()

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
        size has fallen below ``ess_threshold * n_particles``.
        """
        return -logsumexp(normalized_weights * 2) < np.log(self.ess_threshold) + np.log(
            self.n_particles
        )

    def _ess_crosses(self):
        """Whether the ESS test triggers a resample on the current population.

        Test only -- does not mutate. Uses the identical predicate as
        ``_maybe_resample`` so the window's pop-out decision matches the slow
        path's resample decision exactly. Returns ``False`` when all weights are
        ``-inf`` (matching the slow path's ``continue``).
        """
        W = np.array([p.logw for p in self.particles])
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
        W = np.array([p.logw for p in self.particles])
        if np.all(W == -np.inf):
            return False, list(range(n))

        w_sum = logsumexp(W)
        normalized_weights = W - w_sum

        if self._ess_below_threshold(normalized_weights):
            probs = np.exp(normalized_weights)
            ancestor_indices = self.resample_fn(probs).tolist()

            if sort_ancestors:
                ancestor_indices.sort()

            self.particles = [self.particles[i].clone() for i in ancestor_indices]
            avg_weight = w_sum - np.log(n)
            for p in self.particles:
                p.logw = avg_weight

            return True, ancestor_indices

        return False, list(range(n))

    def save_record(self, json_path):
        """Write the SMC record JSON, matching the old smc_standard json_file path."""
        if self.record is None:
            return
        with open(json_path, "w") as f:
            f.write(self.record.to_json())
        print(f"Saved record to {json_path}")

    # -- EngineControl phases (used only by the BurstLoop) ----------------
    #
    # These reproduce, row-wise on raw engine logits, exactly what
    # ``_draw_and_score`` does for an UNCONSTRAINED ``DirectTokenSampler`` (and,
    # with an additive control-side factor, for the constrained extension). The
    # engine IS the language model, so ``shape`` adds only the non-LM factor's
    # per-token log-weights; the draw reuses the slow path's ``fast_sample_*``
    # over the same control-side V+1 vocabulary so the same RNG stream selects
    # the same token (up to the warm-KV-vs-reprefill logit residual).
    #
    # A live window is described by ``self._window`` (set by the BurstLoop):
    #   particles   -- list of live particles, indexed by external request id
    #   llm         -- the engine PromptedLLM (the LM the engine evaluates;
    #                  supplies ``_process_logw_next`` / ``_maybe_temper``)
    #   factor      -- an additive control-side Potential sharing the LLM's V+1
    #                  vocab (e.g. a coerced BoolFSA), or None for the
    #                  unconstrained case
    #   eval_factor -- callable ``(factor, context) -> LazyWeights`` that runs the
    #                  factor's async ``logw_next`` synchronously on the driver's
    #                  event loop (the slow path's exact call)
    #   eos_id      -- an engine token id to return when forcing/observing EOS
    #   pop_out     -- once True, every subsequent draw forces EOS (control
    #                  signal; never banked) so the window ends for resampling

    def _window_particle(self, request_id):
        """Map a vLLM internal request id (``"{external}-{8 chars}"``) to its
        particle via the external index the BurstLoop assigned."""
        external = request_id.rsplit("-", 1)[0]
        return self._window["particles"][int(external)]

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
        return

    def draw(self, logits, request_ids, sampling_metadata):
        """Draw one token per row using the SLOW PATH's draw function.

        Reproduces ``DirectTokenSampler.sample``:

        * unconstrained -- build the control-side V+1 ``LazyWeights`` from the
          engine logits exactly as ``PromptedLLM._process_logw_next`` does;
        * additive factor -- reconstruct ``Product(llm, factor).logw_next`` over
          the product's vocabulary, substituting the engine-derived LM weights
          for the reprefilled ones and gathering through ``Product``'s own
          ``v1_idxs``/``v2_idxs`` index maps (so vocab narrowing matches slow).

        Then normalize and draw with ``fast_sample_lazyweights`` (the same
        Gumbel-max consuming the same numpy RNG as the slow path), bank
        ``logws.sum()`` and ``logps[token]``, advance the particle, and run the
        no-critic termination bookkeeping. The factor's ``logw_next`` is the
        slow path's exact (async) call, evaluated synchronously via the driver's
        ``eval_factor`` hook on the BurstLoop's event loop. When popping out,
        force EOS for every row WITHOUT banking it.
        """
        import torch

        from genlm.control.util import fast_sample_lazyweights

        w = self._window
        llm = w["llm"]
        factor = w["factor"]
        target = w["target"]
        eval_factor = w["eval_factor"]
        eos_id = w["eos_id"]
        out = []

        for i, rid in enumerate(request_ids):
            p = self._window_particle(rid)

            if w["pop_out"] or p.done:
                # Forced pop-out (control signal) or an already-finished row:
                # return EOS, advance nothing, bank nothing.
                out.append(eos_id)
                continue

            # LM half: temper, then _process_logw_next -- == llm.logw_next(ctx)
            # (modulo the warm-KV-vs-reprefill residual).
            row = logits[i]
            if row.dtype != torch.float32:
                row = row.float()
            lm_logws = llm._process_logw_next(llm._maybe_temper(row))

            if factor is None:
                logws = lm_logws
            else:
                # Factor half: read from the particle's CARRIED factor state
                # (advanced one token at a time), not by replaying the context.
                # Reconstruct Product.logw_next over the product vocab by gathering
                # the LM and factor weight vectors onto the product's vocab_eos
                # (the same gather Product.logw_next performs), with the
                # engine-derived LM weights substituted for the reprefilled ones.
                factor_logws = eval_factor(factor, p.factor_state)
                logws = target.make_lazy_weights(
                    lm_logws.weights[w["llm_idxs"]]
                    + factor_logws.weights[w["factor_idxs"]]
                )

            logps = logws.normalize()
            token = fast_sample_lazyweights(logps)

            # Bank exactly what the slow path's transition banks, then run the
            # identical synchronous (no-critic) termination bookkeeping.
            p.score(logws.sum())
            p.logp += logps[token]
            p.context.append(token)
            # Advance the carried factor state by the drawn token (EOS has no
            # symbols and ends the particle, so nothing to advance).
            if factor is not None and not isinstance(token, EndOfSequence):
                p.factor_state = factor.advance(p.factor_state, token)

            if p.logw == float("-inf"):
                p.finish()
            else:
                self._terminate_if_done(p)

            if isinstance(token, EndOfSequence):
                out.append(eos_id)
            else:
                out.append(token.token_id)

        # End-of-step ESS test (same predicate the slow path applies after every
        # token). If it crosses, arm pop-out: the next engine step forces EOS for
        # all live rows so the window ends and the BurstLoop resamples. Skip
        # while already popping out (no real step happened).
        if not w["pop_out"] and self._ess_crosses():
            w["pop_out"] = True

        return torch.tensor(out, dtype=torch.int64, device=logits.device)


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

    def __init__(self, hub):
        self.hub = hub

    async def run(self):
        await self.hub.start()
        return await self.hub.run()


def _is_engine_llm(potential):
    """Whether ``potential`` is a single LM backed by a vLLM engine."""
    model = getattr(potential, "model", None)
    return model is not None and hasattr(model, "llm_engine")


def split_engine_target(potential):
    """Split a sampler target into ``(llm, factor, target)`` for the window.

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


def can_burst(hub):
    """Whether the ``BurstLoop`` can drive this hub's configuration.

    The window fast path requires:

    * the unit sampler is engine-native (``supports_engine_native()``) -- i.e. an
      unbiased ``DirectTokenSampler`` with no separate proposal, so drawing from
      the backend LM's normalized ``logw_next`` equals its per-step ``sample``;
    * the sampler target decomposes (:func:`split_engine_target`) into a
      single engine LM, optionally times one additive same-vocab factor (e.g. a
      coerced ``BoolFSA``);
    * no critic. A non-trivial critic twists per step / reweights at termination
      and is not yet handled by the window's synchronous draw, so it stays on the
      slow path.

    Anything else falls back to :class:`StepLoop`.
    """
    sampler = hub.unit_sampler
    if not getattr(sampler, "supports_engine_native", lambda: False)():
        return False

    potential = getattr(sampler, "potential", None)
    if potential is None or split_engine_target(potential) is None:
        return False

    if hub.critic is not None:
        return False

    return True


class BurstLoop:
    """Drives consecutive SMC steps inside a vLLM engine via EngineControl.

    Opens a *window* = one ``AsyncVirtualLM.run_burst`` call over the live
    particles' token contexts. The hub's :meth:`Controller.shape` /
    :meth:`Controller.draw` run per engine
    decode step, advancing each particle exactly as the slow path's transition
    would. The window itself NEVER resamples: when the hub's end-of-step ESS test
    crosses the threshold it arms pop-out, the next step forces EOS for every live
    row (a control signal, never banked), and ``run_burst`` returns. The driver
    then runs the hub-owned ESS test + resample/fork and relaunches the next
    window from each surviving particle's token prefix. ESS / resample / log_ml
    stay entirely hub-owned -- the engine only produces logits.

    Only valid when :func:`can_burst` is True (checked by the caller).
    """

    def __init__(self, hub):
        self.hub = hub
        # Number of engine windows opened (>1 iff a resample popped a window out
        # mid-generation). Useful for verifying the pop-out/relaunch path ran.
        self.n_windows = 0
        decomposed = split_engine_target(hub.unit_sampler.potential)
        if decomposed is None:  # pragma: no cover - guarded by can_burst
            raise ValueError("target is not window-expressible")
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

    def _eos_idxs(self):
        return list(self.llm.token_maps.eos_idxs)

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
        hub = self.hub
        await hub.start()

        # Seed each particle's carried factor state (empty context == state0).
        # Across windows the state is carried (cloned on resample), never
        # re-derived, so only freshly-started particles need seeding.
        if self.factor is not None:
            for p in hub.particles:
                if p.factor_state is None:
                    p.factor_state = self.factor.state0()

        eos_idxs = self._eos_idxs()
        eos_id = eos_idxs[0]
        loop = asyncio.get_running_loop()

        # The factor is evaluated from its CARRIED state (advanced one token at a
        # time), not by replaying the context: `logw_next_from_state` has a sync
        # body (no `asyncio.gather` over the vocabulary -- the dominant per-step
        # cost), and `draw` runs in the worker thread below, so it schedules this
        # one coroutine back onto the event loop and blocks for the result.
        def eval_factor(factor, state):
            fut = asyncio.run_coroutine_threadsafe(
                factor.logw_next_from_state(state), loop
            )
            return fut.result()

        while any(not p.done for p in hub.particles):
            self.n_windows += 1
            live = [p for p in hub.particles if not p.done]
            prompts = [self._context_ids(p) for p in live]
            # Each particle keeps its own budget; the engine window just needs to
            # be long enough to reach the longest particle's budget (plus one
            # step to let a budget-exhausted row pop out via forced EOS).
            max_steps = max(p.max_tokens_left for p in live) + 1

            hub._window = {
                "particles": live,
                "llm": self.llm,
                "factor": self.factor,
                "target": self.target,
                "llm_idxs": self.llm_idxs,
                "factor_idxs": self.factor_idxs,
                "eval_factor": eval_factor,
                "eos_id": eos_id,
                "pop_out": False,
            }
            try:
                # Run the engine step loop in a worker thread so this event loop
                # stays free to service the factor's run_coroutine_threadsafe.
                await loop.run_in_executor(
                    None,
                    lambda: self.llm.model.run_burst(
                        prompts=prompts,
                        control=hub,
                        max_steps=max_steps,
                        eos_token_ids=eos_idxs,
                        temperature=self.llm.temperature,
                    ),
                )
            finally:
                hub._window = None

            # Controller-owned ESS test + resample/fork over the FULL population.
            hub._maybe_resample(sort_ancestors=hub.record is not None)

        return hub.particles
