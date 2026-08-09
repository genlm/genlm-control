import pytest
import numpy as np
from arsenal.maths import logsumexp

from genlm.control.sampler import (
    DirectTokenSampler,
    SetTokenSampler,
    EagerSetSampler,
    AWRS,
)
from conftest import (
    mock_params,
    iter_item_params,
    MockPotential,
    ContextSensitiveMockPotential,
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
    same-vocab, fully-supported proposal.

    SWOR only proves that each token's `logw + logp` sums to the right
    `target.logw_next` value; it can't catch a bug where `logw` and `logp`
    are each wrong but compensate in the sum. So we additionally force every
    token directly and check the exact weight formula
    `logw = target_logws[t] - proposal_logws[t] + log_Z_proposal` and
    `logp = proposal_logps[t]` independently.
    """
    vocab, target_ws, proposal_ws, context = params
    target = MockPotential(vocab, np.log(target_ws))
    proposal = MockPotential(vocab, np.log(proposal_ws))
    sampler = DirectTokenSampler(target, proposal=proposal)

    try:
        have = await trace_swor(sampler, context)
        want = await sampler.target.logw_next(context)
        have.assert_equal(want, atol=1e-5, rtol=1e-5)

        target_logws = np.log(target_ws)
        proposal_logws = np.log(proposal_ws)
        log_Z_proposal = logsumexp(proposal_logws)
        proposal_logps = proposal_logws - log_Z_proposal

        for forced_idx in range(len(target_ws)):
            forced_token = sampler.target.vocab_eos[forced_idx]

            def draw(p, _tok=forced_token):
                return _tok

            tok, logw, logp = await sampler.sample(context, draw=draw)
            tid = sampler.target.lookup[tok]
            assert tid == forced_idx

            expected_logw = target_logws[tid] - proposal_logws[tid] + log_Z_proposal
            np.testing.assert_allclose(logw, expected_logw, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(logp, proposal_logps[tid], rtol=1e-10, atol=1e-12)
    finally:
        await sampler.cleanup()


@pytest.mark.parametrize(
    "proposal, exc_type, match",
    [
        (
            MockPotential([bytes([i]) for i in range(2)], np.log([0.4, 0.4, 0.2])),
            ValueError,
            "different tokenizers",
        ),
        (object(), TypeError, "Potential"),
    ],
    ids=["vocab_mismatch", "not_a_potential"],
)
def test_direct_token_sampler_proposal_vocab_mismatch(proposal, exc_type, match):
    target = MockPotential([bytes([i]) for i in range(3)], np.log([0.3, 0.3, 0.3, 0.1]))
    with pytest.raises(exc_type, match=match):
        DirectTokenSampler(target, proposal=proposal)


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


# Unit-level logw_eos checks; end-to-end accumulation is covered in test_seq_sampler.py.
@pytest.mark.asyncio
@settings(deadline=None)
@given(iter_item_params())
async def test_set_token_sampler_logw_eos(params):
    """ We check SetTokenSampler and AWRS both compute the forced-EOS
    importance weight correctly: their cheap logw_eos(context) equals
    logw_next(context)[eos] of the respective target. """
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
        got = await sampler.logw_eos(context)
        want = (await sampler.target.logw_next(context))[sampler.target.eos]
        assert np.isclose(got, want, atol=1e-5, rtol=1e-5)
    finally:
        await sampler.cleanup()

    vocab = [bytes([i]) for i in range(3)]
    potential = MockPotential(vocab, np.log([0.3, 0.3, 0.3, 0.1]))
    condition = MockPotential(vocab, [0.0, float("-inf"), 0.0, 0.0])  # bans vocab[1]
    awrs = AWRS(potential, condition, seed=0)

    awrs_context = [vocab[0], vocab[2]]
    got = await awrs.logw_eos(awrs_context)
    want = (await awrs.target.logw_next(awrs_context))[awrs.target.eos]
    assert np.isclose(got, want, atol=1e-5, rtol=1e-5)


@pytest.mark.asyncio
async def test_direct_token_sampler_proposal_different_distributions():
    """When target and proposal have different context-dependent distributions
    (simulating e.g. different prompt lengths / conditioning), the IS weight
    formula still holds exactly for each token."""
    vocab = [bytes([i]) for i in range(3)]
    target_base = np.log([0.3, 0.4, 0.2, 0.1])
    proposal_base = np.log([0.1, 0.1, 0.5, 0.3])

    target = ContextSensitiveMockPotential(vocab, target_base, context_scale=0.3)
    proposal = ContextSensitiveMockPotential(vocab, proposal_base, context_scale=0.7)
    sampler = DirectTokenSampler(target, proposal=proposal)

    # Test with several different context lengths
    for ctx in [[], [vocab[0]], [vocab[0], vocab[1]], [vocab[0], vocab[1], vocab[2]]]:
        target_logws_ctx = await target.logw_next(ctx)
        proposal_logws_ctx = await proposal.logw_next(ctx)

        proposal_log_Z = logsumexp(proposal_logws_ctx.weights)
        proposal_logps = proposal_logws_ctx.weights - proposal_log_Z

        for forced_idx in range(len(target_base)):
            forced_token = sampler.target.vocab_eos[forced_idx]

            def draw(p, _tok=forced_token):
                return _tok

            tok, logw, logp = await sampler.sample(ctx, draw=draw)
            tid = sampler.target.lookup[tok]
            assert tid == forced_idx

            expected_logw = (
                target_logws_ctx.weights[tid]
                - proposal_logws_ctx.weights[tid]
                + proposal_log_Z
            )
            np.testing.assert_allclose(logw, expected_logw, rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(logp, proposal_logps[tid], rtol=1e-10, atol=1e-12)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "target_ws, proposal_ws",
    [
        # cross-distribution — IS weight differs from logZ_target per step.
        (np.log([0.3, 0.4, 0.2, 0.1]), np.log([0.1, 0.1, 0.5, 0.3])),
        # proposal == target — IS correction collapses to logZ_target.
        (np.log([0.3, 0.4, 0.2, 0.1]), np.log([0.3, 0.4, 0.2, 0.1])),
        # peaked target, near-uniform proposal.
        (np.log([0.6, 0.1, 0.1, 0.2]), np.log([0.25, 0.25, 0.25, 0.25])),
        # skewed 5-way proposal (formerly its own Monte-Carlo recovery test).
        (np.log([0.1, 0.2, 0.3, 0.3, 0.1]), np.log([0.4, 0.1, 0.1, 0.3, 0.1])),
    ],
)
async def test_sis_with_proposal_weights_match_manual_computation(
    target_ws, proposal_ws
):
    """SIS (ess_threshold=0) with a proposal: each particle's final log-weight
    equals start_weight + sum_t (target_logws[tok_t] - proposal_logws[tok_t]
    + logsumexp(proposal_logws)).

    `ess_threshold=0` disables resampling so weights propagate intact through
    SMC, letting us check the IS formula exactly. Parametrized over several
    (target, proposal) pairs so a silently-dropped proposal or wrong-sign
    correction is caught regardless of which token happens to be drawn.
    """
    from genlm.control.constant import EOS

    vocab = [bytes([i]) for i in range(len(target_ws) - 1)]
    target = MockPotential(vocab, target_ws)
    proposal = MockPotential(vocab, proposal_ws)

    proposal_logZ = logsumexp(proposal_ws)

    max_tokens = 5
    seqs = await DirectTokenSampler(target, proposal=proposal).smc(
        n_particles=10,
        ess_threshold=0.0,
        max_tokens=max_tokens,
    )

    for ctx, actual_logw in zip(seqs.contexts, seqs.log_weights):
        # start_weight = target.prefix([]) = 0 for MockPotential.
        # Sequences of length == max_tokens had EOS forced at the boundary, so
        # their final-position IS correction is target.logw_next[EOS] rather
        # than the natural-sampling formula used at every other position.
        expected_logw = 0.0
        L = len(ctx)
        boundary_forced = L == max_tokens
        for i, tok in enumerate(ctx):
            tid = len(vocab) if tok is EOS else target.lookup[tok]
            if boundary_forced and i == L - 1:
                expected_logw += target_ws[tid]
            else:
                expected_logw += target_ws[tid] - proposal_ws[tid] + proposal_logZ

        np.testing.assert_allclose(
            actual_logw, expected_logw, rtol=1e-10, atol=1e-12,
            err_msg=f"Weight mismatch for sequence {ctx}",
        )


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
async def test_smc_token_sampler_with_proposal(params):
    """Smoke test: proposal-equipped DirectTokenSampler runs end-to-end
    through SMC with resampling (ess_threshold=0.5) and with a critic.
    Weight correctness is verified by `test_sis_with_proposal_weights_match_manual_computation`
    (which uses ess_threshold=0.0); this test only guards against crashes
    in the resampling + critic code paths when a proposal is in play."""
    vocab, logws, logws2 = params
    target = MockPotential(vocab, logws)
    proposal = MockPotential(vocab, logws2)

    sampler = DirectTokenSampler(target, proposal=proposal)
    sequences = await sampler.smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)

    mock_critic = MockPotential(vocab, logws2)
    sampler = DirectTokenSampler(target, proposal=proposal)
    sequences = await sampler.smc(
        n_particles=10,
        ess_threshold=0.5,
        max_tokens=10,
        critic=mock_critic,
    )
    assert len(sequences) == 10
    assert all(len(seq) <= 10 for seq in sequences)
