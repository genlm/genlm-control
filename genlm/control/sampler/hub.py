"""The SMC hub: a single, engine-independent owner of the entire SMC algorithm.

This module replaces the previous llamppl ``smc_standard`` + ``SequenceModel``
coupling. The hub owns the particle population, the per-step transition, the
ESS test, resampling/forking and the log marginal likelihood accumulation --
*always*. It is exact per token: there is no segment-graining and no
hard/soft constraint fork. ``logw_next`` is one operation.

The hub is designed so that the per-step work splits into two phases that can
be invoked either by the slow Python loop (this file's ``SlowDriver``) or,
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
    )

    def __init__(self, max_tokens):
        self.context = []
        self.logw = 0.0
        self.logp = 0.0
        self.twist_amount = 0.0
        self.done = False
        self.max_tokens_left = max_tokens

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


class Hub:
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

        # Set by the WindowDriver for the duration of a single engine window;
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

        if self.critic and self.twist_with_critic:
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
            if self.critic:
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

    def _ess_crosses(self):
        """Whether the ESS test triggers a resample on the current population.

        Test only -- does not mutate. Uses the identical predicate as
        ``_maybe_resample`` so the window's pop-out decision matches the slow
        path's resample decision exactly. Returns ``False`` when all weights are
        ``-inf`` (matching the slow path's ``continue``).
        """
        n = self.n_particles
        W = np.array([p.logw for p in self.particles])
        if np.all(W == -np.inf):
            return False
        normalized_weights = W - logsumexp(W)
        return -logsumexp(normalized_weights * 2) < np.log(self.ess_threshold) + np.log(
            n
        )

    def _maybe_resample(self, sort_ancestors=False):
        """Run the ESS test on the current population and resample if it crosses.

        This is the *only* implementation of the ESS/resample math; both the slow
        ``run`` loop and the ``WindowDriver`` call it so the two paths are
        bit-identical here. Mutates ``self.particles`` on a resample.

        Returns ``(did_resample, ancestor_indices)``.
        """
        n = self.n_particles
        W = np.array([p.logw for p in self.particles])
        if np.all(W == -np.inf):
            return False, list(range(n))

        w_sum = logsumexp(W)
        normalized_weights = W - w_sum

        if -logsumexp(normalized_weights * 2) < np.log(self.ess_threshold) + np.log(n):
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

    # -- EngineControl phases (used only by the WindowDriver) ----------------
    #
    # These reproduce, row-wise on raw engine logits, exactly what
    # ``_draw_and_score`` does for an UNCONSTRAINED ``DirectTokenSampler`` (and,
    # with an additive control-side factor, for the constrained extension). The
    # engine IS the language model, so ``shape`` adds only the non-LM factor's
    # per-token log-weights; the draw reuses the slow path's ``fast_sample_*``
    # over the same control-side V+1 vocabulary so the same RNG stream selects
    # the same token (up to the warm-KV-vs-reprefill logit residual).
    #
    # A live window is described by ``self._window`` (set by the WindowDriver):
    #   particles  -- list of live particles, indexed by external request id
    #   potential  -- the engine PromptedLLM (the LM the engine evaluates)
    #   factor     -- an additive control-side factor with ``logw_next`` over the
    #                 engine vocab, or None for the unconstrained case (stage a).
    #                 The constrained extension (stage b) lives in ``shape`` and
    #                 is gated off by ``window_eligible`` until verified.
    #   eos_id     -- an engine token id to return when forcing/observing EOS
    #   pop_out    -- once True, every subsequent draw forces EOS (control
    #                 signal; never banked) so the window ends for resampling

    def _window_particle(self, request_id):
        """Map a vLLM internal request id (``"{external}-{8 chars}"``) to its
        particle via the external index the WindowDriver assigned."""
        external = request_id.rsplit("-", 1)[0]
        return self._window["particles"][int(external)]

    def shape(self, logits, request_ids) -> None:
        """Shape each row's raw engine logits into its proposal in place.

        For the unconstrained ``DirectTokenSampler`` (the only configuration
        ``window_eligible`` currently admits) there is no control-side factor, so
        shaping is the identity and the per-step weight increment is zero.

        The constrained extension (stage b) belongs here: add the additive
        factor's per-token log-weights -- mapped into engine-token-id order as the
        exact inverse of ``PromptedLLM._process_logw_next`` -- and bank
        ``logsumexp(shaped) - logsumexp(raw)`` on the particle so the draw matches
        ``DirectTokenSampler(Product(llm, factor))``. It is intentionally not
        wired yet (see ``window_eligible``); doing so requires advancing the
        factor's per-token state synchronously inside this callback, which must be
        verified against the slow path before being enabled.
        """
        w = self._window
        if w["pop_out"]:
            # We are forcing EOS this step; do not touch logits or weights.
            return

        if w["factor"] is not None:  # pragma: no cover - stage b, not yet enabled
            raise NotImplementedError(
                "constrained engine-native shaping (stage b) is not wired; "
                "window_eligible should have routed this to SlowDriver"
            )
        # Unconstrained: identity shape, zero weight increment.

    def draw(self, logits, request_ids, sampling_metadata):
        """Draw one token per row using the SLOW PATH's draw function.

        Reproduces ``DirectTokenSampler.sample``: build the control-side V+1
        ``LazyWeights`` from the (already shaped) engine logits exactly as
        ``PromptedLLM._process_logw_next`` does, normalize, and draw with
        ``fast_sample_lazyweights`` (the same Gumbel-max consuming the same numpy
        RNG as the slow path). Advance the particle (append token, accumulate
        ``logp`` and the start/normalizer weight), and return the engine token id
        per row. When popping out, force EOS for every row WITHOUT banking it.
        """
        import torch

        from genlm.control.util import fast_sample_lazyweights

        w = self._window
        potential = w["potential"]
        eos_id = w["eos_id"]
        out = []

        for i, rid in enumerate(request_ids):
            p = self._window_particle(rid)

            if w["pop_out"] or p.done:
                # Forced pop-out (control signal) or an already-finished row:
                # return EOS, advance nothing, bank nothing.
                out.append(eos_id)
                continue

            # Reproduce DirectTokenSampler.sample over the V+1 control vocab,
            # exactly as PromptedLLM.logw_next: temper, then _process_logw_next.
            row = logits[i]
            if row.dtype != torch.float32:
                row = row.float()
            logws = potential._process_logw_next(potential._maybe_temper(row))
            logps = logws.normalize()
            token = fast_sample_lazyweights(logps)

            # Bank exactly what the slow path's transition banks, then run the
            # identical synchronous (no-critic) termination bookkeeping.
            p.score(logws.sum())
            p.logp += logps[token]
            p.context.append(token)

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
        # all live rows so the window ends and the WindowDriver resamples. Skip
        # while already popping out (no real step happened).
        if not w["pop_out"] and self._ess_crosses():
            w["pop_out"] = True

        return torch.tensor(out, dtype=torch.int64, device=logits.device)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


class SlowDriver:
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


def window_eligible(hub):
    """Whether the ``WindowDriver`` can drive this hub's configuration.

    The window fast path requires:

    * the unit sampler is engine-native (``supports_engine_native()``) -- i.e. an
      unbiased ``DirectTokenSampler`` with no separate proposal, so drawing from
      the backend LM's normalized ``logw_next`` equals its per-step ``sample``;
    * the target potential is a single engine LM exposing a vLLM engine
      (``potential.model.llm_engine``);
    * no critic (stage a). A terminal-only critic (``is_terminal_only()``) is a
      natural future extension -- it never twists mid-generation, so only the
      end-of-window terminal reweight differs -- but the window's synchronous
      ``draw`` does not yet perform that terminal critic ``score``, so it stays
      excluded until wired and verified.

    Anything else falls back to :class:`SlowDriver`. This is intentionally
    conservative; it grows as more factors become additively expressible over the
    engine vocab.
    """
    sampler = hub.unit_sampler
    if not getattr(sampler, "supports_engine_native", lambda: False)():
        return False

    potential = getattr(sampler, "potential", None)
    model = getattr(potential, "model", None)
    if model is None or not hasattr(model, "llm_engine"):
        return False

    if hub.critic is not None:
        return False

    return True


class WindowDriver:
    """Drives consecutive SMC steps inside a vLLM engine via EngineControl.

    Opens a *window* = one ``backend.run_window`` call over the live particles'
    token contexts. The hub's :meth:`Hub.shape` / :meth:`Hub.draw` run per engine
    decode step, advancing each particle exactly as the slow path's transition
    would. The window itself NEVER resamples: when the hub's end-of-step ESS test
    crosses the threshold it arms pop-out, the next step forces EOS for every live
    row (a control signal, never banked), and ``run_window`` returns. The driver
    then runs the hub-owned ESS test + resample/fork and relaunches the next
    window from each surviving particle's token prefix. ESS / resample / log_ml
    stay entirely hub-owned -- the engine only produces logits.

    Only valid when :func:`window_eligible` is True (checked by the caller).
    """

    def __init__(self, hub, backend):
        self.hub = hub
        self.backend = backend

    def _potential(self):
        return self.hub.unit_sampler.potential

    def _engine(self):
        return self._potential().model.llm_engine

    def _prompt_ids(self):
        return list(self._potential().prompt_ids)

    def _eos_idxs(self):
        return list(self._potential().token_maps.eos_idxs)

    def _context_ids(self, p):
        """Engine prompt for a particle: prompt prefix + its drawn token ids.

        Drops a trailing control EOS sentinel (a finished particle is never
        relaunched, but a context may carry EOS as its last element)."""
        ids = self._prompt_ids()
        for tok in p.context:
            if isinstance(tok, EndOfSequence):
                continue
            ids.append(tok.token_id)
        return ids

    async def run(self):
        hub = self.hub
        await hub.start()

        potential = self._potential()
        eos_idxs = self._eos_idxs()
        eos_id = eos_idxs[0]

        while any(not p.done for p in hub.particles):
            live = [p for p in hub.particles if not p.done]
            prompts = [self._context_ids(p) for p in live]
            # Each particle keeps its own budget; the engine window just needs to
            # be long enough to reach the longest particle's budget (plus one
            # step to let a budget-exhausted row pop out via forced EOS).
            max_steps = max(p.max_tokens_left for p in live) + 1

            hub._window = {
                "particles": live,
                "potential": potential,
                "factor": None,
                "eos_id": eos_id,
                "pop_out": False,
            }
            try:
                self.backend.run_window(
                    self._engine(),
                    prompts=prompts,
                    control=hub,
                    max_steps=max_steps,
                    eos_token_ids=eos_idxs,
                    temperature=potential.temperature,
                )
            finally:
                hub._window = None

            # Hub-owned ESS test + resample/fork over the FULL population.
            hub._maybe_resample(sort_ancestors=hub.record is not None)

        return hub.particles
