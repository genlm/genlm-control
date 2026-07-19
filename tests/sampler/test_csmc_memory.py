"""Tests for :class:`RetainedTokenSampler` and the
``SMC.__call__(retained_sequence=...)`` dispatch.

The C-SMC inference loop itself lives in
:func:`llamppl.csmc_standard` and is tested upstream; here we cover the
genlm-control sampler-layer wrapper that implements the "force the next
token to be the next retained entry" contract that the loop relies on.
"""

import pytest

from genlm.control.sampler.csmc_memory import RetainedTokenSampler
from genlm.control.sampler.sequence import SMC
from genlm.control.sampler.token import DirectTokenSampler

from conftest import WeightedSet


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


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
