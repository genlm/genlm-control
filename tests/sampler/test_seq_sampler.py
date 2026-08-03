import json
import pathlib
import pytest
import tempfile
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

    # `.smc()` (the TokenSampler convenience wrapper) must thread `json_path`
    # through to `SMC.__call__` without crashing, critic and all.
    with tempfile.NamedTemporaryFile() as tmp:
        via_smc = await unit_sampler.smc(
            n_particles=10,
            ess_threshold=ess_threshold,
            max_tokens=10,
            critic=critic,
            json_path=tmp.name,
        )
        assert len(via_smc) == 10
        assert all(len(seq) <= 10 for seq in via_smc)


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
@pytest.mark.parametrize("max_tokens, n_particles", [(2, 64), (1, 8)])
async def test_max_tokens_boundary_forces_eos(max_tokens, n_particles):
    """ We check each particle's importance weight is correct for both
    termination modes: natural EOS (L < max_tokens) and EOS forced at the
    boundary (L == max_tokens). At max_tokens=1, EOS is forced for every
    particle, and the empty completion is the only sequence fitting the
    boundary, so it also holds the entire partition. """
    seqs = ["", "a", "ab"]  # "" lets EOS fire at the start; "a" hits the boundary
    weights = [1.0, 2.0, 3.0]
    p = WeightedSet(seqs, weights)
    unit_sampler = DirectTokenSampler(p)
    sampler = SMC(unit_sampler)

    out = await sampler(n_particles=n_particles, ess_threshold=0, max_tokens=max_tokens)
    logeps = await p.prefix([])

    for seq, logw in out:
        assert seq[-1] == p.eos
        L = len(seq)
        if L < max_tokens:
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

    if max_tokens == 1:
        assert all(seq == [p.eos] for seq, _ in out)
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
        samplers=[unit_sampler],
        critics=[None],
        group_sizes=[1],
        ess_threshold=0.5,
        max_tokens=10,
        twist_with_critic=True,
    )
    with pytest.raises(ValueError, match="Start weight.*"):
        await controller.start()


def test_string_for_serialization():
    assert string_for_serialization([b"a", b"b"]) == "a|b"
    assert string_for_serialization([]) == ""


@pytest.mark.asyncio
async def test_record_increments_rebuild_each_context():
    """A step records only what it appended, so a particle's context is the
    concatenation of its increments along the ancestor chain (the walk the viewer
    does). Resampling is on, so the fork bookkeeping is exercised."""
    p = WeightedSet(["0", "00", "1"], [3.0, 2.0, 1.0])
    sampler = SMC(DirectTokenSampler(p))

    with tempfile.NamedTemporaryFile(suffix=".json") as tmp:
        out = await sampler(
            n_particles=8, ess_threshold=0.9, max_tokens=6, json_path=tmp.name
        )
        history = json.loads(pathlib.Path(tmp.name).read_text())

    rebuilt = None  # per-row token lists, carried forward step to step
    for step in history:
        parts = step["particles"]
        assert all("contents" not in rec for rec in parts), "record stores increments"
        if rebuilt is None:
            parent = [[] for _ in parts]
        elif step["mode"] == "resample":
            parent = [list(rebuilt[a]) for a in step["ancestors"]]
        else:
            parent = [list(row) for row in rebuilt]
        assert len(parent) == len(parts)
        rebuilt = [
            row + ([] if not rec["contents_incr"] else rec["contents_incr"].split("|"))
            for row, rec in zip(parent, parts)
        ]

    # Every rebuilt row is a prefix of that particle's final serialized context;
    # a dropped increment, a mis-keyed ancestor, or a bad separator all break this.
    for row, (context, _) in zip(rebuilt, out):
        final = string_for_serialization(context).split("|") if context else []
        assert row == final[: len(row)], (row, final)
