import pytest
import numpy as np
import torch
from collections import defaultdict
from genlm.backend import load_model_by_name
from genlm.bytes import BeamParams
from genlm.control import (
    Ensemble,
    ByteLLM,
    PromptedLLM,
    convert_to_weighted_logop,
    direct_token_sampler,
    EOS,
)
from genlm.control.potential.built_in import WCFG
from conftest import MockPotential

OPS = ["sum", "prod", "harmonic", "min", "max", 2.5]

# Two toy LMs: weighted CFGs over strings of length 3 (vocabulary: bytes a, b, c), so
# the ensemble's target op(P(x), Q(x)) and its normalizer can be computed exactly.
P_GRAMMAR = """
0.20: S -> a b c
0.20: S -> a b a
0.20: S -> b b c
0.10: S -> b a c
0.10: S -> c c b
0.10: S -> a c a
0.10: S -> c b a
"""
Q_GRAMMAR = """
0.20: S -> a b c
0.20: S -> a b a
0.25: S -> b b c
0.10: S -> b a c
0.10: S -> c c b
0.10: S -> a c a
0.05: S -> c b a
"""
A, B, C = 97, 98, 99


@pytest.fixture
def p1():
    return WCFG.from_string(P_GRAMMAR)


@pytest.fixture
def p2():
    return WCFG.from_string(Q_GRAMMAR)


def exact_target(p1, p2, op, a):
    """{string: op(P(string), Q(string))} over both languages, and its sum Z."""
    P, Q = p1.cfg.language(10), p2.cfg.language(10)
    combine = convert_to_weighted_logop(op, a)

    def log(d, x):
        return np.log(d[x]) if d.get(x, 0) > 0 else -np.inf

    target = {x: float(np.exp(combine(log(P, x), log(Q, x)))) for x in set(P) | set(Q)}
    return target, sum(target.values())


def _normalized_mock(vocab, probs):
    """A context-independent mock LM with a normalized next-token distribution."""
    return MockPotential(vocab=vocab, next_token_logws=np.log(probs))


# ============================================================================
# Operations
# ============================================================================

X = np.array([0.2, 0.5, 0.3])
Y = np.array([0.6, 0.1, 0.3])


@pytest.mark.parametrize(
    "op, mean",
    [
        ("sum", lambda x, y, a: a * x + (1 - a) * y),
        ("prod", lambda x, y, a: x**a * y ** (1 - a)),
        ("harmonic", lambda x, y, a: 1 / (a / x + (1 - a) / y)),
    ],
)
@pytest.mark.parametrize("a", [0.3, 0.5])
def test_named_ops_match_definitions(op, mean, a):
    result = convert_to_weighted_logop(op, a)(np.log(X), np.log(Y))
    np.testing.assert_allclose(np.exp(result), mean(X, Y, a), rtol=1e-12)


@pytest.mark.parametrize("p", [-2.5, -0.5, 0.5, 2.5])
@pytest.mark.parametrize("a", [0.3, 0.7])
def test_power_mean_matches_definition(p, a):
    result = convert_to_weighted_logop(p, a)(np.log(X), np.log(Y))
    expected = (a * X**p + (1 - a) * Y**p) ** (1 / p)
    np.testing.assert_allclose(np.exp(result), expected, rtol=1e-12)


@pytest.mark.parametrize("op, func", [("min", np.minimum), ("max", np.maximum)])
@pytest.mark.parametrize("a", [0.3, 0.7])
def test_weighted_extrema(op, func, a):
    x, y = np.log(X), np.log(Y)
    if a <= 0.5:
        expected = (1 - 2 * a) * x + 2 * a * func(x, y)
    else:
        expected = (2 * a - 1) * y + 2 * (1 - a) * func(x, y)
    np.testing.assert_allclose(convert_to_weighted_logop(op, a)(x, y), expected)


@pytest.mark.parametrize(
    "op, a", [("sum", 0.0), ("sum", 1.0), ("pm5", 0.5), (True, 0.5)]
)
def test_invalid_op_or_weight(op, a):
    with pytest.raises(ValueError):
        convert_to_weighted_logop(op, a)


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("a", [0.3, 0.5, 0.7])
def test_zero_weights_give_neginf_not_nan(op, a):
    """Combining -inf (zero weight) entries must not produce nan."""
    fn = convert_to_weighted_logop(op, a)
    x = np.array([-1.0, -np.inf, -np.inf, -1.0])
    y = np.array([-2.0, -np.inf, -1.0, -np.inf])
    result = fn(x, y)
    assert not np.any(np.isnan(result))
    assert result[1] == -np.inf
    assert fn(-np.inf, -np.inf) == -np.inf


def test_weighted_max_equal_weights_ignores_zero_term():
    """At a=0.5 the weighted max is the plain max, even with a -inf input."""
    fn = convert_to_weighted_logop("max", a=0.5)
    np.testing.assert_allclose(fn(np.array([-np.inf]), np.array([-1.0])), [-1.0])


# ============================================================================
# Ensemble potential, on the toy LMs
# ============================================================================


@pytest.mark.asyncio
@pytest.mark.parametrize("op", OPS)
async def test_logw_next_matches_prefix(p1, p2, op):
    """prefix = op(p1.prefix, p2.prefix) and logw_next = prefix(ctx + [x]) - prefix(ctx)."""
    ensemble = Ensemble(p1, p2, op=op, a=0.3)
    combine = convert_to_weighted_logop(op, a=0.3)
    # After [B] and [C] the two grammars differ in prefix mass and the next token is
    # uncertain, so global and per-step combination differ (except for prod).
    contexts = [[], [A], [B], [C], [B, B]]
    batch = await ensemble.batch_logw_next(contexts)
    for row, context in zip(batch.weights, contexts):
        base = await ensemble.prefix(context)
        assert base == pytest.approx(
            combine(await p1.prefix(context), await p2.prefix(context)), abs=1e-12
        )
        for token in ensemble.vocab:
            expected = await ensemble.prefix(context + [token]) - base
            assert row[ensemble.lookup[token]] == pytest.approx(expected, abs=1e-10)
        expected_eos = await ensemble.complete(context) - base
        assert row[ensemble.lookup[ensemble.eos]] == pytest.approx(
            expected_eos, abs=1e-10
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("op", OPS)
async def test_next_token_weights_reproduce_exact_target(p1, p2, op):
    """Accumulating logw_next along every path gives op(P(x), Q(x)) for each string,
    and their total is the exact normalizer Z."""
    ensemble = Ensemble(p1, p2, op=op, a=0.3)
    found = defaultdict(float)

    async def walk(context, logw):
        row = (await ensemble.logw_next(context)).weights
        for token, w in zip(ensemble.vocab_eos, row):
            if w == -np.inf:
                continue
            if token is EOS:
                found[tuple(context)] += np.exp(logw + w)
            elif len(context) < 5:
                await walk(context + [token], logw + w)

    await walk([], await ensemble.prefix([]))
    target, Z = exact_target(p1, p2, op, a=0.3)
    assert set(found) == {x for x, w in target.items() if w > 0}
    for x, w in target.items():
        assert found.get(x, 0.0) == pytest.approx(w, abs=1e-12)
    assert sum(found.values()) == pytest.approx(Z, abs=1e-12)


@pytest.mark.asyncio
@pytest.mark.parametrize("op", ["sum", "prod", "max"])
async def test_smc_matches_exact_target(p1, p2, op):
    """Mean Zhat over runs matches Z, and the pooled posterior matches the target."""
    torch.manual_seed(0)  # token draws
    np.random.seed(0)  # resampling
    ensemble = Ensemble(p1, p2, op=op, a=0.3)
    target, Z = exact_target(p1, p2, op, a=0.3)
    sampler = direct_token_sampler(ensemble)
    z_hats, pooled = [], defaultdict(float)
    for _ in range(100):
        sequences = await sampler.smc(n_particles=10, ess_threshold=0.5, max_tokens=10)
        z_hats.append(np.exp(sequences.log_ml))
        for context, logw in zip(sequences.contexts, sequences.log_weights):
            assert context[-1] is EOS
            pooled[tuple(context[:-1])] += np.exp(logw)
        if op == "sum":
            # A sum of normalized LMs keeps every weight at log 1.
            np.testing.assert_allclose(sequences.log_weights, 0.0, atol=1e-12)
    se = np.std(z_hats) / np.sqrt(len(z_hats))
    assert abs(np.mean(z_hats) - Z) <= 4 * se + 1e-12
    total = sum(pooled.values())
    tv = 0.5 * sum(abs(pooled.get(x, 0) / total - w / Z) for x, w in target.items())
    assert tv < 0.05


@pytest.mark.asyncio
@pytest.mark.parametrize("warm", [False, True])
async def test_component_logws(p1, p2, warm):
    """Test per-model weights: prefix weights, and complete weights after EOS."""
    ensemble = Ensemble(p1, p2, op="sum", a=0.4)
    if warm:
        await ensemble.batch_logw_next([[], [A], [A, B]])
    w1, w2 = await ensemble.component_logws([A, B])
    assert w1 == pytest.approx(await p1.prefix([A, B]), abs=1e-12)
    assert w2 == pytest.approx(await p2.prefix([A, B]), abs=1e-12)
    w1, w2 = await ensemble.component_logws([A, B, C, EOS])
    assert w1 == pytest.approx(np.log(0.2), abs=1e-12)
    assert w2 == pytest.approx(np.log(0.2), abs=1e-12)


@pytest.mark.asyncio
async def test_smc_with_forced_eos():
    """Particles cut off at max_tokens end in EOS and keep per-model weights.
    (Mock LMs: the toy grammars give EOS zero mass before length 3.)"""
    p1 = _normalized_mock(["a", "b", "c"], [0.5, 0.2, 0.2, 0.1])
    p2 = _normalized_mock(["a", "b", "c"], [0.1, 0.3, 0.4, 0.2])
    ensemble = Ensemble(p1, p2, op="sum", a=0.4)
    sequences = await direct_token_sampler(ensemble).smc(
        n_particles=5, ess_threshold=0.5, max_tokens=3
    )
    for context, logw in zip(sequences.contexts, sequences.log_weights):
        assert context[-1] is EOS and len(context) <= 3
        assert np.isfinite(logw)
        w1, w2 = await ensemble.component_logws(context)
        assert w1 == pytest.approx(await p1.complete(context[:-1]), abs=1e-12)
        assert w2 == pytest.approx(await p2.complete(context[:-1]), abs=1e-12)


def test_different_vocabularies_raise():
    p1 = _normalized_mock(["a", "b"], [0.5, 0.3, 0.2])
    p2 = _normalized_mock(["a", "c"], [0.5, 0.3, 0.2])
    with pytest.raises(ValueError, match="same vocabulary"):
        Ensemble(p1, p2, op="prod")


@pytest.mark.asyncio
async def test_permuted_vocabularies():
    """Same tokens in a different order are matched by token, not by position."""
    probs = {"a": (0.5, 0.1), "b": (0.2, 0.3), "c": (0.2, 0.4), EOS: (0.1, 0.2)}
    v1, v2 = ["a", "b", "c"], ["c", "a", "b"]
    p1 = _normalized_mock(v1, [probs[t][0] for t in v1 + [EOS]])
    p2 = _normalized_mock(v2, [probs[t][1] for t in v2 + [EOS]])
    ensemble = Ensemble(p1, p2, op="sum", a=0.4)
    assert ensemble.p1_vocab_idxs != ensemble.p2_vocab_idxs
    await ensemble.assert_contract([[], ["a"], ["c", "b"]])


@pytest.mark.asyncio
@pytest.mark.parametrize("op", OPS)
async def test_potential_contract(p1, p2, op):
    """The library's checks, which also cover batch_prefix and batch_complete."""
    ensemble = Ensemble(p1, p2, op=op, a=0.3)
    await ensemble.assert_contract([[], [A], [B], [C], [B, B]])


@pytest.mark.asyncio
async def test_cache_eviction(p1, p2):
    """A one-entry cache evicts, and cache hits and misses give the same weights."""
    full = Ensemble(p1, p2, op="prod", a=0.3)
    tiny = Ensemble(p1, p2, op="prod", a=0.3, cache_size=1)
    contexts = [[], [A], [A, B], [B, B]]
    rows_full = (await full.batch_logw_next(contexts)).weights
    rows_tiny = (await tiny.batch_logw_next(contexts)).weights
    np.testing.assert_array_equal(rows_full, rows_tiny)
    assert len(tiny._rows) == 1
    for context in contexts:
        want = await full.prefix(context + [C])
        assert await tiny.prefix(context + [C]) == pytest.approx(want, abs=1e-12)


@pytest.mark.asyncio
async def test_zero_prefix_raises():
    p1 = MockPotential(["a", "b"], [-np.inf, np.log(0.7), np.log(0.3)])
    p2 = _normalized_mock(["a", "b"], [0.5, 0.3, 0.2])
    ensemble = Ensemble(p1, p2, op="prod")
    with pytest.raises(ValueError, match="weight zero"):
        await ensemble.logw_next(["a"])


# ============================================================================
# Real model (GPT-2): integration with PromptedLLM and ByteLLM
# ============================================================================


@pytest.mark.asyncio
async def test_token_ensemble():
    """Ensemble of two prompted GPT-2s matches the ensemble formula at a real context."""
    llm1 = PromptedLLM.from_name("openai-community/gpt2")
    llm2 = llm1.spawn()
    llm1.set_prompt_from_str("Task: Generate structured SQL.\n")
    llm2.set_prompt_from_str("Task: Generate correct SQL.\n")
    ensemble = Ensemble(llm1, llm2, op="sum", a=0.3)
    context = llm1.tokenize(" SELECT name")

    logws = (await ensemble.logw_next(context)).weights
    l1 = np.asarray((await llm1.logw_next(context)).weights)[ensemble.p1_vocab_idxs]
    l2 = np.asarray((await llm2.logw_next(context)).weights)[ensemble.p2_vocab_idxs]
    w1, w2 = await llm1.prefix(context), await llm2.prefix(context)
    combine = convert_to_weighted_logop("sum", a=0.3)
    assert not np.allclose(l1, l2)
    np.testing.assert_allclose(
        logws, combine(w1 + l1, w2 + l2) - combine(w1, w2), atol=1e-4
    )


@pytest.mark.asyncio
async def test_byte_ensemble():
    """Ensemble of two prompted GPT-2 byte-level LMs: a sum ensemble keeps every
    sampling weight at log 1, and per-model weights are available."""
    llm = load_model_by_name("openai-community/gpt2", backend="hf")
    params = BeamParams(K=3, eos_byte_strings=[b"<|endoftext|>"])
    b1, b2 = ByteLLM(llm, params), ByteLLM(llm, params)
    b1.set_prompt_from_str("My favorite physicist is")
    b2.set_prompt_from_str("My favorite author is")
    ensemble = Ensemble(b1, b2, op="sum", a=0.3)
    sampler = direct_token_sampler(ensemble)
    context = []
    for _ in range(8):
        token, logw, _ = await sampler.sample(context)
        assert logw == pytest.approx(0.0, abs=1e-8)
        if token is EOS:
            break
        context.append(token)
    w1, w2 = await ensemble.component_logws(context)
    assert np.isfinite(w1) and np.isfinite(w2)
