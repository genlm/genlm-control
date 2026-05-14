import asyncio
import pytest
import tempfile
import numpy as np
from arsenal.maths import logsumexp

from genlm.control.sampler import DirectTokenSampler, SetTokenSampler, EagerSetSampler
from conftest import (
    mock_params,
    iter_item_params,
    MockPotential,
    trace_swor,
    mock_vocab,
)

from hypothesis import given, settings, strategies as st


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_params())
async def test_direct_token_sampler(params):
    vocab, next_token_ws, context = params
    mock_potential = MockPotential(vocab, np.log(next_token_ws))
    sampler = DirectTokenSampler(mock_potential)

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()


@st.composite
def _mock_params_with_proposal(draw, max_w=1e3):
    """Vocab + two independent positive weight vectors (target / proposal) + a context."""
    vocab = draw(mock_vocab())
    n = len(vocab) + 1
    target_ws = draw(
        st.lists(st.floats(1e-3, max_w), min_size=n, max_size=n)
    )
    proposal_ws = draw(
        st.lists(st.floats(1e-3, max_w), min_size=n, max_size=n)
    )
    context = draw(st.lists(st.sampled_from(vocab), min_size=0, max_size=5))
    return vocab, target_ws, proposal_ws, context


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(_mock_params_with_proposal())
async def test_direct_token_sampler_with_proposal_swor(params):
    """SWOR enumeration recovers `target.logw_next` under an arbitrary
    same-vocab, fully-supported proposal."""
    vocab, target_ws, proposal_ws, context = params
    target = MockPotential(vocab, np.log(target_ws))
    proposal = MockPotential(vocab, np.log(proposal_ws))
    sampler = DirectTokenSampler(target, proposal=proposal)

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()


@pytest.mark.asyncio
async def test_direct_token_sampler_with_proposal_monte_carlo():
    """IS weighting under a skewed proposal recovers `target.logw_next` via
    Monte-Carlo; also exercises the Gumbel-max (`draw=None`) path."""
    vocab = [bytes([i]) for i in range(4)]
    target_ws = np.array([0.1, 0.2, 0.3, 0.3, 0.1])
    # Deliberately skewed proposal — supports same tokens but different mass.
    proposal_ws = np.array([0.4, 0.1, 0.1, 0.3, 0.1])

    target = MockPotential(vocab, np.log(target_ws))
    proposal = MockPotential(vocab, np.log(proposal_ws))
    sampler = DirectTokenSampler(target, proposal=proposal)

    N = 20_000
    samples = await asyncio.gather(*[sampler.sample([]) for _ in range(N)])

    logws = sampler.target.alloc_logws()
    for tok, logw, _ in samples:
        if logw == float("-inf"):
            continue
        tid = sampler.target.lookup[tok]
        logws[tid] = (
            logw - np.log(N) if logws[tid] == float("-inf")
            else logsumexp([logws[tid], logw - np.log(N)])
        )

    want = await sampler.target.logw_next([])
    have = sampler.target.make_lazy_weights(logws)
    np.testing.assert_allclose(
        np.exp(have.weights), np.exp(want.weights), rtol=5e-2, atol=5e-2
    )


def test_direct_token_sampler_proposal_vocab_mismatch():
    target = MockPotential([bytes([i]) for i in range(3)], np.log([0.3, 0.3, 0.3, 0.1]))
    different_vocab = MockPotential(
        [bytes([i]) for i in range(2)], np.log([0.4, 0.4, 0.2])
    )
    with pytest.raises(ValueError, match="vocab_eos"):
        DirectTokenSampler(target, proposal=different_vocab)


def test_direct_token_sampler_proposal_must_be_potential():
    target = MockPotential([bytes([i]) for i in range(3)], np.log([0.3, 0.3, 0.3, 0.1]))
    with pytest.raises(TypeError, match="Potential"):
        DirectTokenSampler(target, proposal=object())


def test_direct_token_sampler_factory_threads_proposal():
    """`direct_token_sampler` forwards `proposal` to `DirectTokenSampler`."""
    from genlm.control.sampler import direct_token_sampler

    vocab = [bytes([i]) for i in range(3)]
    target = MockPotential(vocab, np.log([0.3, 0.3, 0.3, 0.1]))
    proposal = MockPotential(vocab, np.log([0.4, 0.3, 0.2, 0.1]))

    s_default = direct_token_sampler(target)
    s_with_proposal = direct_token_sampler(target, proposal=proposal)

    assert s_default.proposal is None
    assert s_with_proposal.proposal is proposal


@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params())
async def test_set_token_sampler(params):
    iter_vocab, iter_next_token_ws, item_vocab, item_next_token_ws, context = params

    mock_iter = MockPotential(iter_vocab, np.log(iter_next_token_ws))
    mock_item = MockPotential(item_vocab, np.log(item_next_token_ws))

    sampler = SetTokenSampler(
        set_sampler=EagerSetSampler(
            iter_potential=mock_iter,
            item_potential=mock_item,
        )
    )

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()


@st.composite
def mock_vocab_and_logws(draw, max_w=1e3):
    vocab = draw(mock_vocab())
    ws = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    ws2 = draw(
        st.lists(
            st.floats(1e-5, max_w),
            min_size=len(vocab) + 1,
            max_size=len(vocab) + 1,
        )
    )
    logws = [np.log(w) if w > 0 else -np.inf for w in ws]
    logws2 = [np.log(w) if w > 0 else -np.inf for w in ws2]
    return vocab, logws, logws2


@pytest.mark.asyncio
@settings(deadline=None)
@given(mock_vocab_and_logws())
async def test_smc_token_sampler(params):
    vocab, logws, logws_critic = params
    mock_potential = MockPotential(vocab, logws)
    sequences = await DirectTokenSampler(mock_potential).smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)

    mock_critic = MockPotential(vocab, logws_critic)
    sequences = await DirectTokenSampler(mock_potential).smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
        critic=mock_critic,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)

    with tempfile.NamedTemporaryFile() as tmp:
        sequences = await DirectTokenSampler(mock_potential).smc(
            n_particles=10,
            ess_threshold=0.5,
            max_tokens=10,
            json_path=tmp.name,
        )
        assert len(sequences) == 10
        assert all(len(seq) <= 10 for seq in sequences)

        sequences = await DirectTokenSampler(mock_potential).smc(
            n_particles=10,
            ess_threshold=0.5,
            max_tokens=10,
            critic=mock_critic,
            json_path=tmp.name,
        )
        assert len(sequences) == 10
        assert all(len(seq) <= 10 for seq in sequences)
