"""Tests for Conditional SMC: csmc_standard, RetainedTokenSampler, and
the SMC.__call__ retained_sequence dispatch.

These tests exercise the contract described in
``genlm.control.sampler.csmc``: that slot 0 of the C-SMC particle
population is retained across resampling rounds; that the
``is_retained`` tag is a slot property re-established after every
resample; and that :class:`RetainedTokenSampler` is a transparent
delegate when its host particle is not retained or the retained
sequence is exhausted.
"""

import pytest

from genlm.control.sampler.csmc import (
    csmc_standard,
    RetainedTokenSampler,
)
from genlm.control.sampler.sequence import SMC
from genlm.control.sampler.token import DirectTokenSampler

from conftest import WeightedSet


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockModel:
    """A duck-typed mock of the subset of ``llamppl.Model`` that
    :func:`csmc_standard` actually touches: ``untwist``, ``step``,
    ``done_stepping``, the writeable ``weight`` attribute, and the
    ``is_retained`` tag the loop manipulates.

    ``step`` records whether the slot is retained at the time of the
    call and scores the weight differently for retained vs. free
    particles, so that weights diverge enough for the ESS check to fire
    resampling on every round.
    """

    def __init__(self, n_steps=3, score_retained=10.0, score_free=0.0):
        self.n_steps = n_steps
        self.step_idx = 0
        self.weight = 0.0
        self.slot_history = []  # list of "retained"/"free" per call to step
        self.is_retained = False
        self._score_retained = score_retained
        self._score_free = score_free

    def untwist(self):
        pass

    def done_stepping(self):
        return self.step_idx >= self.n_steps

    async def step(self):
        self.slot_history.append("retained" if self.is_retained else "free")
        self.weight += (
            self._score_retained if self.is_retained else self._score_free
        )
        self.step_idx += 1


class _MockParent:
    """The minimum surface a TokenSampler's ``forward`` reads off its
    ``parent``: ``token_ctx``, ``logp``, and a ``score`` method."""

    def __init__(self, is_retained=False, token_ctx=None):
        self.token_ctx = list(token_ctx or [])
        self.is_retained = is_retained
        self.logp = 0.0
        self.weight = 0.0

    def score(self, amt):
        self.weight += amt


# ---------------------------------------------------------------------------
# csmc_standard
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_csmc_standard_runs_to_termination():
    """Smoke test: with a model that finishes after ``n_steps`` calls,
    the loop terminates and returns ``n_particles`` particles."""
    model = _MockModel(n_steps=3)
    particles = await csmc_standard(model, n_particles=4, ess_threshold=0.5)
    assert len(particles) == 4
    for p in particles:
        assert p.done_stepping()


@pytest.mark.asyncio
async def test_csmc_standard_slot_zero_history_all_retained():
    """Slot 0's recorded history is "retained" at every step, regardless
    of how many resamples happened in between."""
    model = _MockModel(n_steps=4)
    particles = await csmc_standard(model, n_particles=8, ess_threshold=1.0)
    assert particles[0].slot_history == ["retained"] * 4


@pytest.mark.asyncio
async def test_csmc_standard_other_slots_free_at_termination():
    """Non-zero slots have ``is_retained == False`` at termination.

    Note: we do *not* assert anything about ``slot_history`` of non-zero
    slots, because the loop's post-last-step resample (after the final
    step but before the while-check exits) may overwrite their
    ``slot_history`` with deepcopied content from slot 0. The invariant
    that survives that final inheritance is on the ``is_retained`` tag
    itself, which the resample block re-establishes (slot 0 stays
    retained, slots 1..N-1 are explicitly marked free) immediately
    after every resample.
    """
    model = _MockModel(n_steps=4)
    particles = await csmc_standard(model, n_particles=8, ess_threshold=1.0)
    for p in particles[1:]:
        assert p.is_retained is False


@pytest.mark.asyncio
async def test_csmc_standard_n_particles_one():
    """M=1 degenerate case: only the retained slot exists, no
    resampling possible, all steps record "retained"."""
    model = _MockModel(n_steps=2)
    particles = await csmc_standard(model, n_particles=1, ess_threshold=0.5)
    assert len(particles) == 1
    assert particles[0].is_retained is True
    assert particles[0].slot_history == ["retained", "retained"]


@pytest.mark.asyncio
async def test_csmc_standard_n_particles_zero_raises():
    with pytest.raises(ValueError, match="n_particles must be >= 1"):
        await csmc_standard(_MockModel(), n_particles=0)


@pytest.mark.asyncio
async def test_csmc_standard_skips_resample_when_all_particles_dead():
    """If every particle's weight is -inf, the resample block must skip
    rather than feed NaN into ``np.random.choice``. Mirrors the same
    guard in llamppl's ``smc_standard``."""
    model = _MockModel(
        n_steps=2, score_retained=float("-inf"), score_free=float("-inf")
    )
    particles = await csmc_standard(model, n_particles=4, ess_threshold=1.0)
    assert len(particles) == 4
    for p in particles:
        assert p.done_stepping()


# ---------------------------------------------------------------------------
# RetainedTokenSampler
# ---------------------------------------------------------------------------


def _make_unit_sampler():
    """A tiny ``DirectTokenSampler`` over a deterministic vocab."""
    sequences = [["a", "b", "c"], ["b", "a", "c"]]
    weights = [1.0, 1.0]
    pot = WeightedSet(sequences, weights)
    return DirectTokenSampler(pot), pot


@pytest.mark.asyncio
async def test_retained_sampler_forces_tokens_when_retained():
    """When the host particle is retained, the wrapper returns the
    retained-sequence tokens in order, regardless of base-sampler
    probabilities."""
    base, _ = _make_unit_sampler()
    try:
        z_star = ["b", "a", "c"]
        wrapper = RetainedTokenSampler(base, retained_sequence=z_star)
        for expected in z_star:
            wrapper.parent = _MockParent(is_retained=True)
            token = await wrapper.forward()
            assert token == expected
    finally:
        await base.cleanup()


@pytest.mark.asyncio
async def test_retained_sampler_passthrough_when_not_retained():
    """When the host particle is not retained, the wrapper delegates
    to the base sampler. On a degenerate single-token potential, the
    base sampler is deterministic, so we can verify the delegated
    token."""
    pot = WeightedSet([["x"]], [1.0])
    base = DirectTokenSampler(pot)
    try:
        wrapper = RetainedTokenSampler(base, retained_sequence=["y"])
        wrapper.parent = _MockParent(is_retained=False)
        token = await wrapper.forward()
        # The wrapper did NOT force "y"; instead the base sampler chose
        # the only positive-weight token.
        assert token == "x"
    finally:
        await base.cleanup()


@pytest.mark.asyncio
async def test_retained_sampler_passthrough_after_exhaustion():
    """When the retained sequence is empty (or already consumed), the
    wrapper delegates to the base sampler even for a retained host."""
    pot = WeightedSet([["x"]], [1.0])
    base = DirectTokenSampler(pot)
    try:
        wrapper = RetainedTokenSampler(base, retained_sequence=[])
        wrapper.parent = _MockParent(is_retained=True)
        token = await wrapper.forward()
        assert token == "x"
    finally:
        await base.cleanup()


@pytest.mark.asyncio
async def test_retained_sampler_sample_delegates_to_base():
    pot = WeightedSet([["x"]], [1.0])
    base = DirectTokenSampler(pot)
    try:
        wrapper = RetainedTokenSampler(base, retained_sequence=[])
        token, _w, _p = await wrapper.sample([], draw=lambda _: "x")
        assert token == "x"
    finally:
        await base.cleanup()


@pytest.mark.asyncio
async def test_retained_sampler_start_weight_delegates_to_base():
    pot = WeightedSet([["x"]], [1.0])
    base = DirectTokenSampler(pot)
    try:
        wrapper = RetainedTokenSampler(base, retained_sequence=[])
        assert await wrapper.start_weight() == await base.start_weight()
    finally:
        await base.cleanup()


@pytest.mark.asyncio
async def test_retained_sampler_cleanup_delegates_to_base():
    pot = WeightedSet([["x"]], [1.0])
    base = DirectTokenSampler(pot)
    calls: list[str] = []
    orig_cleanup = base.cleanup

    async def tracking_cleanup():
        calls.append("base")
        await orig_cleanup()

    base.cleanup = tracking_cleanup
    wrapper = RetainedTokenSampler(base, retained_sequence=[])
    await wrapper.cleanup()
    assert calls == ["base"]


@pytest.mark.asyncio
async def test_retained_sampler_idx_only_advances_when_forcing():
    """``retained_idx`` advances iff the wrapper actually forces a
    token (host is retained AND sequence not exhausted)."""
    pot = WeightedSet([["x"]], [1.0])
    base = DirectTokenSampler(pot)
    try:
        # Use 'x' as the retained token so a retained-and-not-exhausted
        # call (which forces "x") and a non-retained call (which samples
        # the base, also "x") both produce "x"; the test of intent is
        # purely in retained_idx.
        wrapper = RetainedTokenSampler(base, retained_sequence=["x", "x", "x"])

        # Three retained, in-bounds calls: idx 0 -> 1 -> 2 -> 3.
        for expected_idx in (1, 2, 3):
            wrapper.parent = _MockParent(is_retained=True)
            await wrapper.forward()
            assert wrapper.retained_idx == expected_idx

        # Now exhausted; a retained call must NOT advance further.
        wrapper.parent = _MockParent(is_retained=True)
        await wrapper.forward()
        assert wrapper.retained_idx == 3

        # And a non-retained call must NOT advance, even mid-sequence.
        wrapper.retained_idx = 1
        wrapper.parent = _MockParent(is_retained=False)
        await wrapper.forward()
        assert wrapper.retained_idx == 1
    finally:
        await base.cleanup()


# ---------------------------------------------------------------------------
# SMC.__call__ dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_smc_with_retained_sequence_runs():
    """End-to-end: ``SMC(...)(retained_sequence=z)`` returns a
    ``Sequences`` object and slot 0's prefix equals ``z``."""
    sequences = [["a", "b", "c"], ["a", "c", "b"], ["b", "a", "c"]]
    weights = [1.0, 2.0, 3.0]
    pot = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(pot)
    smc = SMC(unit_sampler)
    try:
        z_star = ["a", "b", "c"]
        result = await smc(
            n_particles=4,
            ess_threshold=0.5,
            max_tokens=10,
            retained_sequence=z_star,
        )
        assert len(result) == 4
        # Slot 0's tokens, sliced to the length of the retained sequence,
        # match z_star exactly. (Anything generated after that, if any,
        # is from the free-sampling tail.)
        assert result.contexts[0][: len(z_star)] == z_star
    finally:
        await unit_sampler.cleanup()


@pytest.mark.asyncio
async def test_smc_without_retained_sequence_unchanged():
    """Regression: with ``retained_sequence=None`` the existing
    standard-SMC path runs to completion."""
    sequences = [["a", "b"]]
    weights = [1.0]
    pot = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(pot)
    smc = SMC(unit_sampler)
    try:
        result = await smc(n_particles=4, ess_threshold=0.5, max_tokens=10)
        assert len(result) == 4
    finally:
        await unit_sampler.cleanup()
