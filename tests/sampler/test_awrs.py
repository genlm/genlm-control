import pytest
import asyncio
import numpy as np
from arsenal.maths import logsumexp
from conftest import MockPotential
from hypothesis import given, strategies as st, settings, reject

from genlm.control.sampler.token import AWRS


async def monte_carlo(sampler, context, N, **kwargs):
    # Used for testing.
    samples = await asyncio.gather(
        *[sampler.sample(context, **kwargs) for _ in range(N)]
    )
    logws = sampler.target.alloc_logws()
    for tok, logw, _ in samples:
        if logw == float("-inf"):
            continue

        token_id = sampler.target.lookup[tok]

        if logws[token_id] == float("-inf"):
            logws[token_id] = logw - np.log(N)
        else:
            logws[token_id] = logsumexp([logws[token_id], logw - np.log(N)])

    return sampler.target.make_lazy_weights(logws)


async def assert_monte_carlo_close(
    sampler_cls, params, N, equality_opts={}, sampler_opts={}
):
    vocab, b_weights, c_weights = params
    potential = MockPotential(
        vocab,
        np.array([np.log(w) if w > 0 else float("-inf") for w in c_weights]),
    )
    condition = MockPotential(
        vocab,
        np.array([np.log(w) if w > 0 else float("-inf") for w in b_weights]),
    )

    sampler = sampler_cls(potential, condition, **sampler_opts)

    want = await sampler.target.logw_next([])
    have = await monte_carlo(sampler, [], N)

    assert np.isclose(np.exp(want.sum()), np.exp(have.sum()), **equality_opts)


# async def assert_variance_reduction(sampler_cls, params, N1, N2, K, sampler_opts={}):
#     # Check that the variance of the logZ estimate is reduced when using
#     # a larger number of samples.
#     assert N1 < N2

#     vocab, b_weights, c_weights = params
#     potential = MockPotential(vocab, np.log(c_weights))
#     condition = MockPotential(vocab, np.log(b_weights))

#     sampler = sampler_cls(potential, condition, **sampler_opts)

#     N1s = await asyncio.gather(*[monte_carlo(sampler, [], N1) for _ in range(K)])
#     Zs_N1 = np.array([np.exp(have.sum()) for have in N1s])
#     N2s = await asyncio.gather(*[monte_carlo(sampler, [], N2) for _ in range(K)])
#     Zs_N2 = np.array([np.exp(have.sum()) for have in N2s])

#     var_N1 = np.var(Zs_N1)
#     var_N2 = np.var(Zs_N2)

#     # If both variances are extremely small (close to machine epsilon),
#     # the test should pass regardless of their relative values
#     epsilon = 1e-30
#     if var_N1 < epsilon and var_N2 < epsilon:
#         return

#     assert var_N1 > var_N2


@st.composite
def V_size(draw):
    # Generate a vocabulary of size <=4.
    return draw(st.integers(min_value=1, max_value=4))


@st.composite
def cont_weights(draw, V_size, min_p=1e-3):
    # Generate a list of floats for each token in the vocabulary (and EOS).
    ws = draw(st.lists(st.floats(min_p, 1), min_size=V_size + 1, max_size=V_size + 1))
    Z = sum(ws)
    ps = [w / Z for w in ws]
    return ps


@st.composite
def bool_weights(draw, V_size):
    # Generate a list of booleans for each token in the vocabulary (and EOS).
    bws = draw(st.lists(st.booleans(), min_size=V_size + 1, max_size=V_size + 1))
    if not any(bws):
        # Need at least one valid token.
        reject()
    return bws


@st.composite
def params(draw, min_p=1e-3):
    vocab_size = draw(V_size())
    b_weights = draw(bool_weights(vocab_size))
    c_weights = draw(cont_weights(vocab_size, min_p))
    return [bytes([i]) for i in range(vocab_size)], b_weights, c_weights


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(params())
async def test_awrs(params):
    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
    )


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(params())
async def test_awrs_no_pruning(params):
    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
        sampler_opts={"prune_logws": False},
    )


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(params())
async def test_awrs_improper_weights_no_pruning(params):
    params = (params[0], [True] * len(params[1]), params[2])

    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
        sampler_opts={"proper_weights": False, "prune_logws": False},
    )


@pytest.fixture
def potential():
    return MockPotential(
        [bytes([i]) for i in range(4)],
        np.log([0.1, 0.2, 0.2, 0.1, 0.4]),
    )


@pytest.fixture
def zero_condition():
    return MockPotential(
        [bytes([i]) for i in range(4)],
        [float("-inf")] * 4,
    )


@pytest.mark.asyncio
async def test_verbosity(potential):
    condition = MockPotential(
        [bytes([i]) for i in range(4)],
        [0, 0, float("-inf"), float("-inf"), 0],
    )
    sampler = AWRS(potential=potential, condition=condition)
    await sampler.sample([], verbosity=1)


@pytest.mark.asyncio
async def test_awrs_no_valid_tokens(potential, zero_condition):
    sampler = AWRS(potential=potential, condition=zero_condition)
    tok, logw, _ = await sampler.sample([])
    assert tok == potential.eos
    assert logw == float("-inf")


@pytest.mark.asyncio
async def test_awrs_improper_weights_no_valid_tokens(potential, zero_condition):
    sampler = AWRS(
        potential=potential,
        condition=zero_condition,
        proper_weights=False,
    )
    tok, logw, _ = await sampler.sample([])
    assert tok == potential.eos
    assert logw == float("-inf")


@pytest.mark.asyncio
async def test_awrs_with_different_vocabs():
    potential = MockPotential(
        [bytes([i]) for i in range(4)],
        np.log([0.4, 0.3, 0.1, 0.1, 0.1]),
    )
    condition = MockPotential(
        [bytes([i]) for i in range(3)],
        [0, 0, float("-inf"), float("-inf")],
    )

    sampler = AWRS(potential, condition, prune_logws=True)

    want = await sampler.target.logw_next([])
    have = await monte_carlo(sampler, [], 10000)

    assert np.isclose(np.exp(want.sum()), np.exp(have.sum()), rtol=5e-3, atol=5e-3)


@pytest.mark.asyncio
async def test_awrs_with_no_pruning_and_different_vocabs():
    potential = MockPotential(
        [bytes([i]) for i in range(4)],
        np.log([0.4, 0.3, 0.1, 0.1, 0.1]),
    )
    condition = MockPotential(
        [bytes([i]) for i in range(3)],
        [0, 0, float("-inf"), float("-inf")],
    )

    sampler = AWRS(potential, condition, prune_logws=False)

    want = await sampler.target.logw_next([])
    have = await monte_carlo(sampler, [], 10000)

    assert np.isclose(np.exp(want.sum()), np.exp(have.sum()), rtol=5e-3, atol=5e-3)


@pytest.mark.asyncio
async def test_can_clip_sample_and_remain_unbiased():
    potential = MockPotential(
        list(range(100)),
        np.log([0.01] * 101),
    )
    condition = MockPotential(
        list(range(100)),
        [-float("inf")] * 40 + [0] + [-float("inf")] * 60,
    )

    sampler_unclipped = AWRS(potential, condition, prune_logws=False)
    sampler_clipped = AWRS(potential, condition, prune_logws=False, clip=0.1)

    unclipped_weight = await monte_carlo(sampler_unclipped, [], 10000)
    clipped_weight = await monte_carlo(sampler_clipped, [], 10000)

    for _ in range(100):
        _, log_weight, _ = await sampler_clipped.sample([])
        if log_weight == -float("inf"):
            break
    else:
        raise AssertionError("Clipped sampler never sampled zero weight")

    assert np.isclose(
        np.exp(unclipped_weight.sum()),
        np.exp(clipped_weight.sum()),
        rtol=5e-3,
        atol=5e-3,
    )
