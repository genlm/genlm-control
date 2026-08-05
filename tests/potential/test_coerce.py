import pytest
import numpy as np
from genlm.control.typing import Atomic
from genlm.control.constant import EOS
from genlm.control.potential import Coerced, Potential
from genlm.backend.tokenization import Token


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


@pytest.mark.asyncio
async def test_properties():
    p = MockPotential([b"a"[0], b"b"[0], b"c"[0]])
    c = Coerced(p, [b"aa", b"bb", b"aab", b"aad"], f=b"".join)

    await c.assert_logw_next_consistency([b"aa", b"bb"], verbosity=1)
    await c.assert_autoreg_fact([b"aa", b"bb"], verbosity=1)
    await c.assert_batch_consistency([[b"aa", b"bb"], [b"aa"]], verbosity=1)


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

    # batch_score/batch_logw_next vs their non-batch counterparts, via the shared
    # testing.py helper (rtol/atol=0 since MockPotential's arithmetic is exact).
    await coerced.assert_batch_consistency(sequences, rtol=0, atol=0, verbosity=1)


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


class TokenPotential(Potential):
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
        Token(0, b"a"),
        Token(1, b"b"),
        Token(2, b"c"),
    ]
    p = TokenPotential(tokens)
    target = [b"aa", b"bb", b"aab", b"aad"]
    c = Coerced(p, target, f=b"".join, prune=True)

    assert len(c.vocab) == 3
    assert set(c.vocab) == {b"aa", b"bb", b"aab"}


class ChartPotential(Potential):
    """Byte potential on the `_consume` chart lane: the chart is the walk's landing
    (position, alive), and the support is the set of prefixes of `words`."""

    def __init__(self, words, advance=False):
        super().__init__(sorted({b for w in words for b in w}))
        self.words = list(words)
        self.consumed = 0
        if not advance:
            self._advance = None  # coerce reads it with a default; None = no lane

    def _live(self, syms):
        return any(bytes(w).startswith(bytes(syms)) for w in self.words)

    def _consume(self, syms):
        self.consumed += 1
        return (len(syms), self._live(syms), tuple(syms))

    def _advance(self, chart, sym):
        n, alive, syms = chart
        nxt = syms + (sym,)
        return (n + 1, True, nxt) if self._live(nxt) else None

    def prefix_logw(self, chart):
        return -float(chart[0]) if chart[1] else float("-inf")

    def complete_logw(self, chart):
        return -float(chart[0]) if bytes(chart[2]) in map(bytes, self.words) else float("-inf")

    async def prefix(self, context):
        return self.prefix_logw(self._consume(tuple(context)))

    async def complete(self, context):
        return self.complete_logw(self._consume(tuple(context)))


@pytest.mark.asyncio
async def test_advance_lane_matches_consume_lane():
    """`_advance` threads the chart down the vocab trie and prunes dead subtrees;
    it must produce exactly the rows the path-rebuilding walk does."""
    words = [b"abc", b"abd", b"axy"]
    vocab = [b"a", b"ab", b"abc", b"abd", b"ax", b"axy", b"b", b"zz", b"abz"]
    slow = Coerced(ChartPotential(words), vocab, f=b"".join, prune=False)
    fast = Coerced(ChartPotential(words, advance=True), vocab, f=b"".join, prune=False)
    assert slow.potential._advance is None

    for context in ([], [b"a"], [b"ab"], [b"ax"]):
        want = await slow.logw_next(context)
        got = await fast.logw_next(context)
        np.testing.assert_array_equal(np.asarray(want.weights), np.asarray(got.weights))

    # The point of threading the chart: dead subtrees are never consumed.
    assert fast.potential.consumed < slow.potential.consumed
