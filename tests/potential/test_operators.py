import pytest
from genlm.control.potential import (
    Potential,
    Coerced,
    Product,
    AutoBatchedPotential,
    MultiProcPotential,
)
from genlm.control.potential.autobatch import autobatched


class SimplePotential(Potential):
    """A simple potential for testing operators."""

    def __init__(self, vocabulary):
        super().__init__(vocabulary)

    async def complete(self, context):
        return 0

    async def prefix(self, context):
        return 0

    def spawn(self):
        return SimplePotential(self.vocab)


@pytest.fixture
def vocab():
    return [b"a"[0], b"b"[0], b"c"[0]]


@pytest.fixture
def p1(vocab):
    return SimplePotential(vocab)


@pytest.fixture
def p2(vocab):
    return SimplePotential(vocab)


@pytest.mark.asyncio
async def test_product_operator(p1, p2):
    have = p1 * p2
    want = Product(p1, p2)
    assert have.p1 == want.p1
    assert have.p2 == want.p2
    assert have.vocab == want.vocab


@pytest.mark.asyncio
async def test_coerce_operator(p1):
    target_vocab = [b"aa", b"bb", b"cc"]

    # Test with default transformations
    def f(seq):
        return [x for xs in seq for x in xs]

    coerced = p1.coerce(SimplePotential(target_vocab), f=f)
    assert set(coerced.vocab) == set(target_vocab)

    # Test with custom transformations
    def f(seq):
        return [xs[0] for xs in seq]

    have = p1.coerce(SimplePotential(target_vocab), f=f)
    want = Coerced(p1, target_vocab, f=f)
    assert have.potential == want.potential
    assert have.vocab == want.vocab


@pytest.mark.asyncio
async def test_to_autobatched(p1):
    have = p1.to_autobatched()
    assert isinstance(have, AutoBatchedPotential)
    assert have.potential is p1
    # One door: every route to the wrapper yields THE wrapper (same window).
    assert p1.to_autobatched() is have
    assert autobatched(p1) is have

    await have.cleanup()


@pytest.mark.asyncio
async def test_to_multiprocess(p1):
    num_workers = 2
    have = p1.to_multiprocess(num_workers=num_workers)
    want = MultiProcPotential(p1.spawn, (), num_workers=num_workers)
    assert have.vocab == want.vocab


@pytest.mark.asyncio
async def test_operator_chaining(p1, p2):
    have = (p1 * p2).to_autobatched()
    want = AutoBatchedPotential(Product(p1, p2))
    assert have.potential.p1 == want.potential.p1
    assert have.potential.p2 == want.potential.p2
    assert have.vocab == want.vocab

    await have.cleanup()
    await want.cleanup()

    V = [b"aa", b"bb", b"cc"]

    def f(seq):
        return [x for xs in seq for x in xs]

    have = (p1 * p2).coerce(SimplePotential(V), f=f)
    want = Coerced(Product(p1, p2), V, f=f)
    assert have.potential.p1 == want.potential.p1
    assert have.potential.p2 == want.potential.p2
    assert have.vocab == want.vocab
