"""Conditional Sequential Monte Carlo (C-SMC).

This module implements C-SMC (Andrieu, Doucet & Holenstein 2010, §4.3): a
variant of SMC in which slot 0 of the particle population is a *retained
particle* -- extended deterministically along a fixed pre-specified token
sequence and exempt from resampling. The remaining ``n_particles - 1``
slots evolve as in standard SMC.

The module exposes two pieces that communicate through a writeable
``is_retained`` attribute that :func:`csmc_standard` sets on each
particle. Retainedness is a *slot* property, not a particle property:
after resampling, only slot 0 has ``is_retained = True``, even if a free
slot happened to inherit slot 0's trajectory via the multinomial draw.

* :func:`csmc_standard` -- the SMC loop with the modified resampling
  rule (slot 0 survives unchanged; the remaining slots are drawn
  multinomially from all ``n_particles`` normalized weights). Sets and
  re-establishes the ``is_retained`` tag.

* :class:`RetainedTokenSampler` -- a transparent wrapper around any
  :class:`TokenSampler`. When its host particle has
  ``is_retained == True`` and the retained sequence is not exhausted,
  it forces the next token to be the next entry of the retained
  sequence; the underlying base sampler is still invoked to compute
  the importance weight at that forced token, so the C-SMC weight
  semantics match standard SMC exactly. Otherwise it is a no-op
  delegate to the base sampler -- which makes it safe to always wrap.
"""

import copy
import asyncio
import numpy as np
from arsenal.maths import logsumexp

from genlm.control.sampler.token import TokenSampler


async def csmc_standard(model, n_particles, ess_threshold=0.5):
    """Conditional sequential Monte Carlo with multinomial resampling.

    Slot 0 is the retained particle: pinned across resampling rounds and
    extended deterministically by the model whenever its ``is_retained``
    attribute is ``True``. Slots 1..``n_particles``-1 are free.

    Args:
        model (llamppl.Model): The particle model. Must expose a writeable
            ``is_retained`` attribute; its ``step`` method (typically via
            a :class:`RetainedTokenSampler` unit sampler) is expected to
            branch on this tag.
        n_particles (int): Total number of particles, including the
            retained one. Must be ``>= 1``; ``n_particles >= 2`` is
            required for nontrivial mixing (Andrieu et al. 2010,
            Theorem 5(b)).
        ess_threshold (float): Fraction of ``n_particles`` below which
            resampling is triggered.

    Returns:
        (list[llamppl.Model]): The completed particles. Slot 0 carries
            the retained-particle trajectory; the remaining
            ``n_particles - 1`` carry trajectories that may have
            inherited from any slot (including slot 0).
    """
    if n_particles < 1:
        raise ValueError(f"n_particles must be >= 1, got {n_particles}")

    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    _tag_slots(particles)

    while any(not p.done_stepping() for p in particles):
        for p in particles:
            p.untwist()
        await asyncio.gather(
            *[p.step() for p in particles if not p.done_stepping()]
        )

        W = np.array([p.weight for p in particles])
        w_sum = logsumexp(W)
        normalized_log_weights = W - w_sum

        ess_log = -logsumexp(2 * normalized_log_weights)
        if ess_log < np.log(ess_threshold) + np.log(n_particles):
            probs = np.exp(normalized_log_weights)
            # Slot 0 survives unchanged; the remaining N-1 are drawn from
            # the multinomial over all N normalized weights.
            new_particles = [particles[0]]
            for _ in range(n_particles - 1):
                idx = np.random.choice(n_particles, p=probs)
                new_particles.append(copy.deepcopy(particles[idx]))
            particles = new_particles

            avg_weight = w_sum - np.log(n_particles)
            for p in particles:
                p.weight = avg_weight

            _tag_slots(particles)

    return particles


def _tag_slots(particles):
    """Set ``is_retained``: ``True`` on slot 0, ``False`` on slots 1..N-1.

    Called once at startup and after every resample, so that retainedness
    stays a slot property: a free slot that just inherited its trajectory
    from slot 0 via multinomial resampling is correctly marked as free
    going forward, and will sample (not force) at its next step.
    """
    particles[0].is_retained = True
    for p in particles[1:]:
        p.is_retained = False


class RetainedTokenSampler(TokenSampler):
    """Transparent retained-particle wrapper around a :class:`TokenSampler`.

    When the host particle has ``is_retained == True`` and the retained
    sequence is not yet exhausted, the next token is forced to be the
    next entry of ``retained_sequence``. The underlying base sampler is
    still invoked to compute the importance weight at that forced
    token, so the C-SMC weight semantics match standard SMC exactly.
    Otherwise (free particle, or retained sequence exhausted), this
    wrapper is a no-op delegate to the base sampler -- so it is safe
    to wrap *every* particle's sampler and let the wrapper itself
    decide whether to force on a per-call basis.

    ``retained_idx`` is held per-wrapper-instance, so each particle's
    deepcopy of the wrapper tracks its own position into the retained
    sequence. Only retained particles ever advance the index; free
    particles never advance it, regardless of which slot they were
    resampled from.

    Args:
        base (TokenSampler): The underlying token sampler used in
            free-particle (or post-exhaustion) mode.
        retained_sequence (list): The pre-specified retained-particle
            tokens, in order. The wrapper forces
            ``retained_sequence[t]`` at step ``t`` for any host particle
            with ``is_retained == True``.
    """

    def __init__(self, base, retained_sequence):
        super().__init__(target=base.target)
        self.base = base
        self.retained_sequence = list(retained_sequence)
        self.retained_idx = 0

    async def forward(self):
        parent = self.parent
        draw = None
        if (
            getattr(parent, "is_retained", False)
            and self.retained_idx < len(self.retained_sequence)
        ):
            forced = self.retained_sequence[self.retained_idx]
            draw = lambda _probs, _forced=forced: _forced  # noqa: E731
            self.retained_idx += 1
        token, logw, logp = await self.base.sample(parent.token_ctx, draw=draw)
        parent.score(logw)
        parent.logp += logp
        return token

    async def sample(self, context, draw=None):
        return await self.base.sample(context, draw=draw)

    async def start_weight(self):
        return await self.base.start_weight()

    async def cleanup(self):
        await self.base.cleanup()
