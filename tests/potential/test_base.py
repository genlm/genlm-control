import pytest
import asyncio
import numpy as np
from genlm.control.typing import Atomic
from genlm.control.potential.base import Potential, EOS


class SimplePotential(Potential):
    async def complete(self, context):
        return -float(len(context))  # Scoring based on length

    async def prefix(self, context):
        return -0.5 * float(len(context))  # Different scoring for prefixes


@pytest.fixture
def potential():
    return SimplePotential([b"a", b"b", b"c"])


def test_token_type(potential):
    assert potential.token_type == Atomic(bytes)


@pytest.mark.asyncio
async def test_score(potential):
    context = [b"b", b"c"]

    have = await potential.score(context)
    want = await potential.prefix(context)
    assert want == -1.0
    assert have == want

    have = await potential.score(context + [EOS])
    want = await potential.complete(context)
    assert want == -2.0
    assert have == want

    have = await potential.score([])
    want = await potential.prefix([])
    assert want == 0.0
    assert have == want


@pytest.mark.asyncio
async def test_logw_next(potential):
    context = [b"b", b"c"]
    have = (await potential.logw_next(context)).materialize()
    for token in potential.vocab_eos:
        want = await potential.score(context + [token]) - await potential.prefix(
            context
        )
        assert have[token] == want


@pytest.mark.asyncio
async def test_batch_score(potential):
    seq1 = [b"a"]
    seq2 = [b"a", b"b"]
    seq3 = [b"a", b"b", EOS]

    have = await potential.batch_score([seq1, seq2, seq3])
    want = await asyncio.gather(
        potential.score(seq1), potential.score(seq2), potential.score(seq3)
    )

    np.testing.assert_array_equal(have, want)
    np.testing.assert_array_equal(have, [-0.5, -1.0, -2.0])


@pytest.mark.asyncio
async def test_batch_logw_next(potential):
    seq1 = [b"a"]
    seq2 = [b"b", b"c"]

    haves = await potential.batch_logw_next([seq1, seq2])  # one batched LazyWeights [N, V+1]
    wants = await asyncio.gather(potential.logw_next(seq1), potential.logw_next(seq2))

    for i, want in enumerate(wants):
        haves.spawn(haves.weights[i]).assert_equal(want)


@pytest.mark.asyncio
async def test_empty(potential):
    with pytest.raises(ValueError):
        await potential.batch_logw_next([])

    with pytest.raises(ValueError):
        await potential.batch_score([])

    with pytest.raises(ValueError):
        await potential.batch_prefix([])

    with pytest.raises(ValueError):
        await potential.batch_complete([])


@pytest.mark.asyncio
async def test_properties(potential):
    await potential.assert_logw_next_consistency([b"b", b"c"], verbosity=1)
    await potential.assert_autoreg_fact([b"b", b"c"], verbosity=1)
    await potential.assert_batch_consistency([[b"b", b"c"], [b"a"]], verbosity=1)


def test_initialization_errors():
    # Test empty vocabulary
    with pytest.raises(ValueError, match="vocabulary cannot be empty"):
        SimplePotential([])

    # Test invalid token_type
    with pytest.raises(ValueError, match="token_type must be a TokenType"):
        SimplePotential([b"a"], token_type="not a token type")

    # Test wrong token types in vocabulary
    wrong_type = Atomic(str)  # Using str instead of bytes
    with pytest.raises(TypeError, match="Tokens in vocabulary must be of type"):
        SimplePotential([b"a", b"b"], token_type=wrong_type)

    # Test invalid EOS type
    with pytest.raises(ValueError, match="EOS must be an instance of EndOfSequence"):
        SimplePotential([b"a"], eos="not an EndOfSequence")

    # Test duplicate tokens
    with pytest.raises(ValueError, match="Duplicate token.*found in vocabulary"):
        SimplePotential([b"a", b"a"])


@pytest.mark.asyncio
async def test_zero_weight_context():
    class ZeroWeightPotential(SimplePotential):
        async def prefix(self, context):
            return float("-inf")

    potential = ZeroWeightPotential([b"a", b"b"])
    with pytest.raises(ValueError, match="Context.*has weight zero under `prefix`."):
        await potential.logw_next([b"a"])


def test_spawn_not_implemented():
    potential = SimplePotential([b"a"])
    with pytest.raises(
        NotImplementedError,
        match="Potential.spawn\\(\\) must be implemented by subclasses.",
    ):
        potential.spawn()
