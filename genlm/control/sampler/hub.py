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

from genlm.control.constant import EOS
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

            W = np.array([p.logw for p in self.particles])
            if np.all(W == -np.inf):
                did_resample = False
                continue

            w_sum = logsumexp(W)
            normalized_weights = W - w_sum

            if -logsumexp(normalized_weights * 2) < np.log(self.ess_threshold) + np.log(
                n
            ):
                probs = np.exp(normalized_weights)
                ancestor_indices = self.resample_fn(probs).tolist()

                if self.record is not None:
                    ancestor_indices.sort()

                self.particles = [self.particles[i].clone() for i in ancestor_indices]
                avg_weight = w_sum - np.log(n)
                for p in self.particles:
                    p.logw = avg_weight

                did_resample = True
            else:
                did_resample = False

        return self.particles

    def save_record(self, json_path):
        """Write the SMC record JSON, matching the old smc_standard json_file path."""
        if self.record is None:
            return
        with open(json_path, "w") as f:
            f.write(self.record.to_json())
        print(f"Saved record to {json_path}")


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


class WindowDriver:
    """Delegates consecutive steps to the engine via the EngineControl contract.

    NOT IMPLEMENTED here -- the engine arms are built by a separate effort. The
    hub already exposes everything this driver needs: it implements the
    EngineControl shape/draw phases over a ``request_id -> particle`` map, and
    keeps resample/fork/ESS/log_ml hub-owned (those are never delegated).
    """

    def __init__(self, hub, backend):
        self.hub = hub
        self.backend = backend

    async def run(self):  # pragma: no cover - engine arms not built yet
        await self.hub.start()
        # TODO(engine-native): drive the engine over windows of consecutive
        # steps, e.g.
        #
        #   prompts = [p.context for p in self.hub.particles]
        #   await self.backend.run_window(
        #       prompts, control=self.hub, max_steps=...
        #   )
        #
        # The backend calls back into the hub via EngineControl.shape /
        # EngineControl.draw, keyed on a request_id -> particle map, to shape
        # the proposal and draw+weight+advance each row. After each engine
        # window returns, the hub performs the ESS test, resampling/forking and
        # log_ml accumulation (never delegated to the engine).
        raise NotImplementedError(
            "WindowDriver requires the in-engine arms (backend.run_window); "
            "use SlowDriver until those land."
        )
