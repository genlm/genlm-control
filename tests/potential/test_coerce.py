import pytest
import numpy as np
from genlm.control.typing import Atomic
from genlm.control.constant import EOS
from genlm.control.potential import Coerced, Potential


class MockPotential(Potential):
    def __init__(self, V):
        super().__init__(V)

    def bytes_to_int(self, byte_seq):
        return int.from_bytes(byte_seq, byteorder="big")

    async def complete(self, context):
        return self.bytes_to_int(context)

    async def prefix(self, context):
        return self.bytes_to_int(context) / 2


@pytest.mark.asyncio
async def test_simple():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    c = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join)

    assert c.token_type == Atomic(bytes)
    assert set(c.vocab) == {b"aa", b"bb", b"aab"}

    have = await c.complete([b"aa", b"bb"])
    want = await p.complete(b"aabb")
    assert have == want

    have = await c.prefix([b"aa", b"bb"])
    want = await p.prefix(b"aabb")
    assert have == want

    have = await c.score([b"aa", b"bb", EOS])
    want = await p.score(b"aabb" + EOS)
    assert have == want

    have = await c.logw_next([b"aa", b"bb"])
    for x in c.vocab_eos:
        want = await p.score(b"aabb" + x) - await p.prefix(b"aabb")
        assert have[x] == want, [have[x], want, x]


@pytest.mark.asyncio
async def test_coerced_batch_operations():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    coerced = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join)
    sequences = [[b"aa", b"aab"], [b"bb"]]

    have = await coerced.batch_complete(sequences)
    want = np.array([await coerced.complete(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await coerced.batch_prefix(sequences)
    want = np.array([await coerced.prefix(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    have = await coerced.batch_score(sequences)
    want = np.array([await coerced.score(sequence) for sequence in sequences])
    np.testing.assert_array_equal(have, want)

    haves = await coerced.batch_logw_next(sequences)
    wants = [await coerced.logw_next(sequence) for sequence in sequences]
    for have, want in zip(haves, wants):
        have.assert_equal(want)


@pytest.mark.asyncio
async def test_coerced_invalid_vocab():
    with pytest.raises(ValueError):
        Coerced(MockPotential([b"a"[0], b"b"[0], b"c"[0]]), [b"xx", b"yy"], f=b"".join)


@pytest.mark.asyncio
async def test_coerced_custom():
    mock_potential = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    coerced = Coerced(
        mock_potential,
        target_vocab=[b"aa", b"bb"],
        f=lambda seq: [item[0] for item in seq],  # Take first byte of each token
    )

    assert coerced.token_type == Atomic(bytes)

    assert len(coerced.vocab) == 2
    assert set(coerced.vocab) == {b"aa", b"bb"}

    have = await coerced.complete([b"aa", b"bb"])
    want = await mock_potential.complete(b"ab")
    assert have == want

    have = await coerced.prefix([b"aa", b"bb"])
    want = await mock_potential.prefix(b"ab")
    assert have == want

    have = await coerced.score([b"aa", b"bb", EOS])
    want = await mock_potential.score(b"ab" + EOS)
    assert have == want


def test_coerced_repr():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    c = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join)
    repr(c)


def test_coerced_no_prune():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    c = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join, prune=False)
    assert len(c.vocab) == 4
    assert set(c.vocab) == {b"aa", b"bb", b"aab", b"aad"}


class MockToken:
    """Mock Token object with byte_string attribute."""

    def __init__(self, token_id, byte_string):
        self.token_id = token_id
        self.byte_string = byte_string

    def __repr__(self):
        return f"MockToken({self.token_id}, {self.byte_string!r})"

    def __hash__(self):
        return hash((self.token_id, self.byte_string))

    def __eq__(self, other):
        if isinstance(other, MockToken):
            return (
                self.token_id == other.token_id
                and self.byte_string == other.byte_string
            )
        return False


class MockTokenPotential(Potential):
    """Mock potential with Token-based vocabulary."""

    def __init__(self, tokens):
        super().__init__(tokens)

    async def complete(self, context):
        return len(context)

    async def prefix(self, context):
        return len(context) / 2


def test_coerced_with_token_vocab():
    """Test Coerced with Token-based potential vocabulary (exercises byte_string extraction)."""
    tokens = [
        MockToken(0, b"a"),
        MockToken(1, b"b"),
        MockToken(2, b"c"),
    ]
    p = MockTokenPotential(tokens)
    target = [b"aa", b"bb", b"aab", b"aad"]
    c = Coerced(p, target, f=b"".join, prune=True)

    assert len(c.vocab) == 3
    assert set(c.vocab) == {b"aa", b"bb", b"aab"}
    assert hasattr(p.vocab[0], "byte_string")
