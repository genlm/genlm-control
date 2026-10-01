"""Unit tests for `SequenceModel` step semantics and the `smc_standard` population loop."""

import numpy as np
import pytest

from genlm.control.constant import EOS
from genlm.control.sampler import resampling
from genlm.control.sampler.smc import SequenceModel, smc_standard
from genlm.control.sampler.token import DirectTokenSampler

from conftest import WeightedSet


# -- deterministic test doubles ----------------------------------------------


class ScriptedSampler:
    """Unit sampler returning the same fixed token and weight on every call."""

    def __init__(self, token=1, step_logw=0.0, eos_logw=0.0, start_logw=0.0):
        self.token = token
        self.step_logw = step_logw
        self.eos_logw = eos_logw
        self.start_logw = start_logw

    async def start_weight(self):
        return self.start_logw

    async def logw_eos(self, context):
        return self.eos_logw

    async def sample(self, context, draw=None):
        return self.token, self.step_logw, 0.0


class ScheduledWeightSampler:
    """Unit sampler returning EOS with the next weight from `weight_schedule` on each call.

    Particles consume the schedule in order only because `sample` never awaits.
    """

    def __init__(self, weight_schedule):
        self._weights = list(weight_schedule)
        self._i = 0

    async def start_weight(self):
        return 0.0

    async def sample(self, context, draw=None):
        w = self._weights[self._i]
        self._i += 1
        return EOS, w, 0.0

    async def logw_eos(self, context):
        return 0.0


class AllDeadSampler:
    """Every draw is immediately -inf; every particle dies on its first step."""

    async def start_weight(self):
        return 0.0

    async def sample(self, context, draw=None):
        return 1, float("-inf"), 0.0

    async def logw_eos(self, context):
        return float("-inf")


def argmax_draw(chart):
    """Pick the highest-probability token (`materialize()` sorts descending)."""
    return next(iter(chart))


class PinnedDrawSampler:
    """Wraps a token sampler so every `sample` call uses `draw`, whatever the caller passes."""

    def __init__(self, inner, draw):
        self._inner = inner
        self._draw = draw

    async def start_weight(self):
        return await self._inner.start_weight()

    async def logw_eos(self, context):
        return await self._inner.logw_eos(context)

    async def sample(self, context, draw=None):
        return await self._inner.sample(context, draw=self._draw)


async def run_to_done(model):
    await model.start()
    while not model.done:
        await model.step()
    return model


# -- forced EOS at the max_tokens boundary ------------------------------------


@pytest.mark.asyncio
async def test_forced_eos_at_max_tokens_boundary():
    """A particle that never samples EOS gets EOS forced at `max_tokens`, weighted by `logw_eos`."""
    sampler = ScriptedSampler(token=1, step_logw=-0.3, eos_logw=-2.0, start_logw=0.1)
    model = SequenceModel(sampler, max_tokens=4)

    await run_to_done(model)

    assert model.context == [1, 1, 1, EOS]
    assert model.done
    expected = 0.1 + 3 * (-0.3) + (-2.0)
    assert np.isclose(model.weight, expected)


@pytest.mark.asyncio
async def test_forced_eos_population_all_end_in_eos():
    """Every particle `smc_standard` returns ends in EOS exactly at the budget."""
    sampler = ScriptedSampler(token=1, step_logw=0.0, eos_logw=-1.0)
    model = SequenceModel(sampler, max_tokens=3)

    particles = await smc_standard(model, n_particles=3, ess_threshold=0)

    assert len(particles) == 3
    for p in particles:
        assert p.context[-1] is EOS
        assert len(p.context) == 3


# -- terminate_when ------------------------------------------------------------


@pytest.mark.asyncio
async def test_terminate_when_appends_eos_same_step_no_correction():
    """`terminate_when` appends EOS in the step that satisfies it, with no weight correction."""
    sampler = ScriptedSampler(token=1, step_logw=-0.4, eos_logw=-99.0, start_logw=0.2)
    model = SequenceModel(
        sampler, max_tokens=100, terminate_when=lambda ctx: len(ctx) >= 3
    )

    await run_to_done(model)

    assert model.context == [1, 1, 1, EOS]
    assert np.isclose(model.weight, 0.2 + 3 * (-0.4))


# -- twist accounting -----------------------------------------------------------


@pytest.mark.asyncio
async def test_twist_matches_terminal_only_scoring():
    """Without resampling, per-step twisting nets to the terminal critic score."""
    potential = WeightedSet(["aab", "b"], [3.0, 1.0])
    critic = WeightedSet(["aab", "b"], [2.0, 5.0])
    sampler = PinnedDrawSampler(
        DirectTokenSampler(potential, autobatch=False), argmax_draw
    )

    twisted = SequenceModel(
        sampler, critic=critic, max_tokens=10, twist_with_critic=True
    )
    terminal_only = SequenceModel(
        sampler, critic=critic, max_tokens=10, twist_with_critic=False
    )

    await run_to_done(twisted)
    await run_to_done(terminal_only)

    assert twisted.context == terminal_only.context
    assert np.isclose(twisted.weight, terminal_only.weight)
    assert np.isclose(twisted.weight, np.log(8))  # hand-derived: log4 (sampler) + log2 (critic)


# -- SequenceModel.start: -inf start weight raises ------------------------------


@pytest.mark.asyncio
async def test_start_raises_on_neg_inf_start_weight():
    class NegInfStartSampler:
        async def start_weight(self):
            return float("-inf")

    model = SequenceModel(NegInfStartSampler())
    with pytest.raises(ValueError, match="Start weight.*"):
        await model.start()


# -- clone: shared sampler/critic, copied state ---------------------------------


def test_clone_shares_sampler_and_critic_copies_state():
    sampler = ScriptedSampler()
    critic = object()  # identity is all clone() cares about for the critic
    parent = SequenceModel(sampler, critic=critic, max_tokens=10)
    parent.context = [1, 2, 3]
    parent.weight = -0.5
    parent.twist_amount = 0.25

    child = parent.clone()

    assert child.unit_sampler is sampler
    assert child.critic is critic
    assert child.context == parent.context and child.context is not parent.context
    assert child.weight == parent.weight
    assert child.twist_amount == parent.twist_amount

    child.context.append(4)
    child.weight = 99.0
    child.twist_amount = -1.0
    assert parent.context == [1, 2, 3]
    assert parent.weight == -0.5
    assert parent.twist_amount == 0.25


# -- smc_standard: population-level plumbing ------------------------------------


@pytest.mark.asyncio
async def test_smc_standard_all_dead_returns_without_crashing():
    """An all -inf population is returned in full instead of crashing in the resample step."""
    model = SequenceModel(AllDeadSampler(), max_tokens=5)

    particles = await smc_standard(model, n_particles=4, ess_threshold=0.5)

    assert len(particles) == 4
    assert all(p.weight == float("-inf") for p in particles)
    assert all(p.done for p in particles)


@pytest.mark.asyncio
async def test_smc_standard_default_resampling_is_multinomial(monkeypatch):
    """An ESS-triggered resample defaults to multinomial resampling."""
    calls = []
    real_multinomial = resampling.RESAMPLING_METHODS["multinomial"]

    def spy(weights):
        calls.append(np.asarray(weights).copy())
        return real_multinomial(weights)

    monkeypatch.setitem(resampling.RESAMPLING_METHODS, "multinomial", spy)

    # Half the population is far heavier than the other half, so ESS collapses
    # well below the threshold and a resample is forced.
    schedule = [0.0, 0.0, 0.0, 0.0, -20.0, -20.0, -20.0, -20.0]
    model = SequenceModel(ScheduledWeightSampler(schedule), max_tokens=5)

    particles = await smc_standard(model, n_particles=8, ess_threshold=0.9)

    assert len(particles) == 8
    assert calls, "the default 'multinomial' entry was never invoked"


def test_nan_weight_folds_to_neg_inf():
    """A NaN weight increment folds the particle's weight to -inf."""
    model = SequenceModel(unit_sampler=None)
    model.score(-2.0)
    model.score(float("nan"))
    assert model.weight == float("-inf")

    model.twist(float("nan"))
    model.untwist()
    assert model.weight == float("-inf")


def test_positive_inf_weight_is_rejected():
    """`score` rejects a +inf log-weight."""
    model = SequenceModel(unit_sampler=None)
    with pytest.raises(ValueError, match=r"\+inf"):
        model.score(float("inf"))


@pytest.mark.asyncio
async def test_nan_weights_do_not_reach_the_resampler():
    """A population whose draws come back NaN dies, rather than resampling off
    NaN probabilities."""
    schedule = [float("nan")] * 4
    model = SequenceModel(ScheduledWeightSampler(schedule), max_tokens=5)

    particles = await smc_standard(model, n_particles=4, ess_threshold=0.9)

    assert len(particles) == 4
    assert all(p.weight == float("-inf") for p in particles)
