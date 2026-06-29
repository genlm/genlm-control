import pytest
import numpy as np


from genlm.control.potential import Potential
from genlm.control.sampler.sequence import SMC
from genlm.control.sampler.smc import Controller
from genlm.control.sampler.smc_record import string_for_serialization
from genlm.control.sampler.token import DirectTokenSampler

from hypothesis import strategies as st, settings, given
from conftest import (
    weighted_set,
    weighted_sequence,
    double_weighted_sequence,
    WeightedSet,
)


@pytest.fixture
def default_unit_sampler():
    sequences = ["a", "b", "c"]
    weights = [1, 2, 3]
    p = WeightedSet(sequences, weights)
    return DirectTokenSampler(p)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence))
async def test_importance(S):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = SMC(unit_sampler)

    sequences = await sampler(n_particles=n_particles, ess_threshold=0, max_tokens=10)
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(double_weighted_sequence))
async def test_importance_with_critic(S):
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 10
    sampler = SMC(unit_sampler, critic=critic)
    sequences = await sampler(n_particles=n_particles, ess_threshold=0, max_tokens=10)

    logeps = await p.prefix([])
    for seq, logw in sequences:
        logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(len(seq))])
        assert np.isclose(logw, logZ + logeps + await critic.score(seq))


@pytest.mark.asyncio
@settings(deadline=None)
@given(weighted_set(weighted_sequence), st.floats(min_value=0, max_value=1))
async def test_smc(S, ess_threshold):
    sequences, weights = zip(*S)

    p = WeightedSet(sequences, weights)
    unit_sampler = DirectTokenSampler(p)

    n_particles = 100
    sampler = SMC(unit_sampler)

    sequences = await sampler(
        n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=10
    )
    assert len(sequences) == n_particles
    assert np.isclose(sequences.log_ml, np.log(sum(weights)), atol=1e-3, rtol=1e-5)


@pytest.mark.asyncio
@settings(deadline=None)
@given(st.floats(min_value=0, max_value=1))
async def test_smc_with_critic(ess_threshold):
    seqs = ["0", "00", "1"]
    weights1 = [3.0, 2.0, 1.0]
    weights2 = [1.0, 2.0, 3.0]

    p = WeightedSet(seqs, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(seqs, weights2)

    n_particles = 500
    sampler = SMC(unit_sampler, critic=critic)

    sequences = await sampler(
        n_particles=n_particles, ess_threshold=ess_threshold, max_tokens=10
    )

    intersection_ws = [w1 * w2 for w1, w2 in zip(weights1, weights2)]
    assert len(sequences) == n_particles
    assert np.isclose(
        np.exp(sequences.log_ml), sum(intersection_ws), atol=0.5, rtol=0.05
    )


@st.composite
def smc_params(draw, item_sampler, max_seq_len=5, max_size=5):
    S = draw(weighted_set(item_sampler, max_seq_len, max_size))
    stop_point = draw(st.integers(min_value=1, max_value=max_seq_len))
    return S, stop_point


@pytest.mark.asyncio
@settings(deadline=None)
@given(smc_params(double_weighted_sequence))
async def test_smc_weights(params):
    S, stop_point = params
    sequences, weights1, weights2 = zip(*S)

    p = WeightedSet(sequences, weights1)
    unit_sampler = DirectTokenSampler(p)
    critic = WeightedSet(sequences, weights2)

    n_particles = 10
    sampler = SMC(unit_sampler, critic=critic)

    sequences = await sampler(
        n_particles=n_particles,
        ess_threshold=0,  # don't resample since that would reset weights
        max_tokens=stop_point,
    )

    logeps = await p.prefix([])
    for seq, logw in sequences:
        L = len(seq)
        # Sequences hitting the max_tokens boundary have EOS deterministically
        # appended, so the final-position IS correction is the unnormalized
        # target log-weight on EOS (not the partition sum used elsewhere).
        if L < stop_point:
            logZ = sum([(await p.logw_next(seq[:n])).sum() for n in range(L)])
        else:
            natural = sum(
                [(await p.logw_next(seq[:n])).sum() for n in range(L - 1)]
            )
            boundary = (await p.logw_next(seq[:-1]))[p.eos]
            logZ = natural + boundary
        twist = await critic.score(seq)
        assert np.isclose(logw, logZ + logeps + twist)


@pytest.mark.asyncio
async def test_max_tokens_boundary_forces_eos():
    """ We check each particle's importance weight is correct for both
    termination modes: natural EOS (L < max_tokens) and EOS forced at the
    boundary (L == max_tokens). """
    seqs = ["", "a", "ab"]  # "" lets EOS fire at the start; "a" hits the boundary
    p = WeightedSet(seqs, [1.0, 2.0, 3.0])
    unit_sampler = DirectTokenSampler(p)
    sampler = SMC(unit_sampler)

    out = await sampler(n_particles=64, ess_threshold=0, max_tokens=2)
    logeps = await p.prefix([])

    for seq, logw in out:
        assert seq[-1] == p.eos
        L = len(seq)
        if L < 2:
            expected = (
                logeps
                + sum([(await p.logw_next(seq[:n])).sum() for n in range(L)])
            )
        else:
            expected = (
                logeps
                + sum(
                    [(await p.logw_next(seq[:n])).sum() for n in range(L - 1)]
                )
                + (await p.logw_next(seq[:-1]))[p.eos]
            )
        assert np.isclose(logw, expected)


@pytest.mark.asyncio
async def test_max_tokens_one_forces_eos():
    """ We check that if we hit the max tokens, the importance weight
    of the EOS forced particle is correct. """
    seqs = ["", "a", "ab"]  # "" gives the empty completion positive mass
    weights = [1.0, 2.0, 3.0]
    p = WeightedSet(seqs, weights)
    unit_sampler = DirectTokenSampler(p)
    sampler = SMC(unit_sampler)

    out = await sampler(n_particles=8, ess_threshold=0, max_tokens=1)

    logeps = await p.prefix([])
    expected = logeps + (await p.logw_next([]))[p.eos]
    for seq, logw in out:
        assert seq == [p.eos]
        assert np.isclose(logw, expected)
    # Only the empty completion fits |y| <= 1, so the partition is its weight.
    assert np.isclose(out.log_ml, np.log(weights[0]))


@pytest.mark.asyncio
async def test_controller_invalid_start_weight():
    class MockPotential(Potential):
        async def prefix(self, context):
            if not context:
                return -np.inf
            return 0

        async def complete(self, context):
            return 0

    unit_sampler = DirectTokenSampler(MockPotential([0]))
    controller = Controller(
        unit_sampler=unit_sampler,
        critic=None,
        n_particles=1,
        ess_threshold=0.5,
        max_tokens=10,
        twist_with_critic=True,
    )
    with pytest.raises(ValueError, match="Start weight.*"):
        await controller.start()


def test_string_for_serialization():
    out = string_for_serialization([b"a", b"b"])
    assert isinstance(out, str) and "|" in out
