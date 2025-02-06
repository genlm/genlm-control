import pytest
import numpy as np
from itertools import chain
from genlm_control.constant import EOS
from genlm_control.potential.lifted import Lifted
from genlm_control.potential.base import Potential


class MockPotential(Potential):
    """A simple mock potential for testing the Lifted potential."""

    def __init__(self):  # individual bytes
        super().__init__([b"a"[0], b"b"[0], b"c"[0]])

    async def complete(self, context):
        return len(context)

    async def prefix(self, context):
        return len(context) / 2


@pytest.fixture
def mock_potential():
    return MockPotential()


@pytest.fixture
def target_vocab():  # byte sequences
    return [b"aa", b"bb", b"aab", b"dd"]


@pytest.fixture
def f():
    return lambda x: list(chain(*x))


@pytest.fixture
def g():
    return lambda x: bytes(x)


@pytest.mark.asyncio
async def test_lifted_initialization(mock_potential, target_vocab, f, g):
    lifted = Lifted(mock_potential, target_vocab, f=f, g=g)
    assert lifted.potential == mock_potential
    assert set(lifted.decode) == {b"aa", b"bb", b"aab"}


@pytest.mark.asyncio
async def test_lifted_invalid_vocab(f, g):
    with pytest.raises(ValueError):
        Lifted(MockPotential(), [b"xx", b"yy"], f=f, g=g)  # Invalid tokens


@pytest.mark.asyncio
async def test_lifted_prefix(mock_potential, target_vocab, f, g):
    lifted = Lifted(mock_potential, target_vocab, f=f, g=g)
    result = await lifted.prefix([b"aa", b"bb"])
    assert result == 2


@pytest.mark.asyncio
async def test_lifted_complete(mock_potential, target_vocab, f, g):
    lifted = Lifted(mock_potential, target_vocab, f=f, g=g)
    result = await lifted.complete([b"aa", b"bb"])
    assert result == 4


@pytest.mark.asyncio
async def test_lifted_score(mock_potential, target_vocab, f, g):
    lifted = Lifted(mock_potential, target_vocab, f=f, g=g)
    result = await lifted.score([b"aa", b"bb", EOS])
    assert result == 4


@pytest.mark.asyncio
async def test_lifted_logw_next(mock_potential, target_vocab, f, g):
    lifted = Lifted(mock_potential, target_vocab, f=f, g=g)
    have = await lifted.logw_next([b"aa", b"bb"])
    want = await mock_potential.batch_logw_next_seq(b"aabb", lifted.decode_eos)
    for i, x in enumerate(lifted.decode_eos):
        assert have[x] == want[i], [have[x], want[i], x]


@pytest.mark.asyncio
async def test_lifted_batch_operations(mock_potential, target_vocab, f, g):
    lifted = Lifted(mock_potential, target_vocab, f=f, g=g)
    sequences = [[b"aa"], [b"bb"]]

    have = await lifted.batch_complete(sequences)
    want = np.array([await lifted.complete(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await lifted.batch_prefix(sequences)
    want = np.array([await lifted.prefix(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await lifted.batch_score(sequences)
    want = np.array([await lifted.score(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    haves = await lifted.batch_logw_next(sequences)
    wants = [await lifted.logw_next(sequence) for sequence in sequences]
    for have, want in zip(haves, wants):
        have.assert_equal(want)

    have = await lifted.batch_logw_next_seq([b"aa"], sequences)
    want = np.array(
        [await lifted.logw_next_seq([b"aa"], sequences) for sequence in sequences]
    )
    np.testing.assert_array_equal(have, want)


@pytest.mark.asyncio
async def test_lifted_custom(mock_potential):
    lifted = Lifted(
        mock_potential,
        target_vocab=[b"aa", b"bb"],
        f=lambda x: [item[0] for item in x],  # Take first byte of each token
        g=lambda x: bytes(x),
    )

    assert len(lifted.decode) == 2
    assert set(lifted.decode) == {b"a", b"b"}

    have = await lifted.complete([b"aa", b"bb"])
    want = await mock_potential.complete(b"ab")
    assert have == want

    have = await lifted.prefix([b"aa", b"bb"])
    want = await mock_potential.prefix(b"ab")
    assert have == want

    have = await lifted.score([b"aa", b"bb", EOS])
    want = await mock_potential.score(b"ab" + EOS)
    assert have == want
