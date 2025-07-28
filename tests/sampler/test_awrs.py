import pytest
import asyncio
import numpy as np
from arsenal.maths import logsumexp
from conftest import MockPotential
from hypothesis import given, strategies as st, settings, example

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
def V_size(draw, max_size=256):
    return draw(st.integers(min_value=1, max_value=min(max_size, 256)))


@st.composite
def cont_weights(draw, V_size, min_p=1e-3):
    ws = [draw(st.floats(min_p, 1))] * (V_size + 1)

    for ixs, f in draw(
        st.lists(
            st.tuples(st.sets(st.integers(0, V_size)), st.floats(min_p, 1)),
        )
    ):
        for i in ixs:
            ws[i] = f

    # Maybe boost some weights in order to create more peaked distributions.
    boost = draw(st.floats(min_value=1, max_value=1000))
    to_boost = draw(st.sets(st.integers(min_value=0, max_value=V_size - 1)))
    for i in to_boost:
        ws[i] *= boost

    Z = sum(ws)
    ps = [w / Z for w in ws]
    return ps


@st.composite
def bool_weights(draw, V_size):
    # Generate a list of booleans for each token in the vocabulary (and EOS).
    n_false = draw(st.integers(min_value=0, max_value=V_size))
    n_true = V_size + 1 - n_false
    weights = draw(st.permutations([True] * n_true + [False] * n_false))
    return weights


@st.composite
def params(draw, max_size=5, min_p=1e-3):
    vocab_size = draw(V_size(max_size=max_size))
    b_weights = draw(bool_weights(vocab_size))
    c_weights = draw(cont_weights(vocab_size, min_p))
    return [bytes([i]) for i in range(vocab_size)], b_weights, c_weights


@pytest.mark.asyncio
@example(([b"\x00"], [False, True], [0.5, 0.5]))
@settings(deadline=None, max_examples=25)
@given(params())
async def test_awrs_is_unbiased(params):
    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
    )


@pytest.mark.asyncio
@settings(deadline=None, max_examples=25)
@given(params(), st.floats(min_value=0.01, max_value=2.0))
async def test_awrs_unnormalized_weights(params, normalizing_constant):
    vocab, b_weights, c_weights = params
    c_weights = [w * normalizing_constant for w in c_weights]
    params = (vocab, b_weights, c_weights)

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
    assert logw == float("-inf")


@pytest.mark.asyncio
async def test_awrs_improper_weights_no_valid_tokens(potential, zero_condition):
    sampler = AWRS(
        potential=potential,
        condition=zero_condition,
        proper_weights=False,
    )
    tok, logw, _ = await sampler.sample([])
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
@example(
    params=([b"\x00"], [False, True], [0.5, 0.5]),
    max_accepts=2,
    max_rejects=2,
    n_monte_carlo_samples=1,
)
@settings(deadline=None, max_examples=25)
@given(
    max_accepts=st.integers(min_value=2, max_value=5),
    max_rejects=st.integers(min_value=2, max_value=5),
    n_monte_carlo_samples=st.integers(min_value=1, max_value=5),
    params=params(),
)
async def test_awrs_with_different_limits(
    params, max_accepts, max_rejects, n_monte_carlo_samples
):
    await assert_monte_carlo_close(
        sampler_cls=AWRS,
        params=params,
        N=10000,
        equality_opts={"rtol": 2e-2, "atol": 2e-2},
        sampler_opts={
            "max_accepts": max_accepts,
            "max_rejects": max_rejects,
            "n_monte_carlo_samples": n_monte_carlo_samples,
        },
    )


@pytest.mark.asyncio
@settings(deadline=None, max_examples=100)
@given(
    max_accepts=st.integers(min_value=2, max_value=2),
    max_rejects=st.integers(min_value=2, max_value=500),
    n_monte_carlo_samples=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=100),
    params=params(max_size=256),
)
async def test_awrs_does_not_return_zero_weight_token_is_valid(
    params, max_accepts, max_rejects, n_monte_carlo_samples, seed
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

    sampler = AWRS(
        max_accepts=max_accepts,
        max_rejects=max_rejects,
        n_monte_carlo_samples=n_monte_carlo_samples,
        seed=seed,
        potential=potential,
        condition=condition,
    )

    tok, logp, _ = await sampler.sample([])

    if tok == potential.eos:
        weight = b_weights[-1]
    else:
        assert isinstance(tok, bytes)
        assert len(tok) == 1
        weight = b_weights[tok[0]]

    if weight > 0:
        assert logp > float("-inf")
    else:
        assert logp == float("-inf")


@pytest.mark.asyncio
@settings(deadline=None, max_examples=100)
@given(
    max_accepts=st.integers(min_value=2, max_value=2),
    n_monte_carlo_samples=st.integers(min_value=1, max_value=5),
    seed=st.integers(min_value=0, max_value=100),
    params=params(max_size=256),
)
async def test_awrs_does_not_return_zero_weight_in_default_configuration(
    params, max_accepts, n_monte_carlo_samples, seed
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

    sampler = AWRS(
        max_accepts=max_accepts,
        n_monte_carlo_samples=n_monte_carlo_samples,
        seed=seed,
        potential=potential,
        condition=condition,
    )

    tok, logp, _ = await sampler.sample([])

    if tok == potential.eos:
        weight = b_weights[-1]
    else:
        assert isinstance(tok, bytes)
        assert len(tok) == 1
        weight = b_weights[tok[0]]

    if weight > 0:
        assert logp > float("-inf")
    else:
        assert logp == float("-inf")


@pytest.mark.parametrize(
    "params",
    [
        {"max_accepts": 1},
        {"max_rejects": 1},
        {"n_monte_carlo_samples": 0},
    ],
)
def test_invalid_arguments(params):
    potential = MockPotential(
        [bytes([i]) for i in range(4)],
        np.log([0.4, 0.3, 0.1, 0.1, 0.1]),
    )
    condition = MockPotential(
        [bytes([i]) for i in range(4)],
        [0, 0, float("-inf"), float("-inf"), 0],
    )
    with pytest.raises(ValueError):
        AWRS(potential, condition, **params)


@pytest.mark.asyncio
async def test_awrs_example_with_underflow_error():
    vocab = [bytes([i]) for i in range(182)]
    b_weights = [False] * 56 + [True] + [False] * 126
    c_weights = [0.23929169657812532] * 4 + [0.00023929169657812532] * 179

    potential = MockPotential(
        vocab,
        np.log(c_weights),
    )
    condition = MockPotential(
        vocab,
        [0 if b else -float("inf") for b in b_weights],
    )

    sampler = AWRS(
        max_accepts=2,
        max_rejects=183,
        n_monte_carlo_samples=1,
        seed=17,
        potential=potential,
        condition=condition,
    )

    for _ in range(1000):
        tok, logp, _ = await sampler.sample([])
        if tok == bytes([56]):
            assert logp > float("-inf")
        else:
            assert logp == float("-inf")


@pytest.mark.asyncio
async def test_awrs_example_with_underflow_error_never_zero_in_default_configuration():
    vocab = [bytes([i]) for i in range(182)]
    b_weights = [False] * 56 + [True] + [False] * 126
    c_weights = [0.23929169657812532] * 4 + [0.00023929169657812532] * 179

    potential = MockPotential(
        vocab,
        np.log(c_weights),
    )
    condition = MockPotential(
        vocab,
        [0 if b else -float("inf") for b in b_weights],
    )

    sampler = AWRS(
        max_accepts=2,
        n_monte_carlo_samples=1,
        seed=17,
        potential=potential,
        condition=condition,
    )

    for _ in range(1000):
        tok, logp, _ = await sampler.sample([])
        assert tok == bytes([56])
        assert logp > float("-inf")


@pytest.mark.asyncio
async def test_can_sample_reliably_with_rounding_to_one():
    vocab = [bytes([i]) for i in range(10)]

    # Chosen because although these sum to 1, they also sum to 1 with
    # one of them removed. This potentially triggers rounding to one
    # in the running sum of rejection probabilities unless we're careful.
    c_weights = [0.999999999999999] + [9.992007221626409e-17] * 10
    b_weights = [False, True] + [False] * 9

    potential = MockPotential(vocab, np.log(c_weights))
    condition = MockPotential(vocab, np.log(b_weights))

    sampler = AWRS(potential, condition, max_accepts=2, max_rejects=11)

    for _ in range(1000):
        tok, logp, _ = await sampler.sample([])
        assert tok == bytes([1]) or logp == -float("inf")
