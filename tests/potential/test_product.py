import re
import pytest
import numpy as np
from genlm.control.potential import Product, Potential
from genlm.control.typing import Atomic


class SimplePotential(Potential):
    def __init__(self, vocabulary, scale=1.0):
        super().__init__(vocabulary)
        self.scale = scale

    async def complete(self, context):
        return -float(len(context)) * self.scale

    async def prefix(self, context):
        return -0.5 * float(len(context)) * self.scale

    def spawn(self):
        return SimplePotential(self.vocab, scale=self.scale)


VOCAB_CASES = {
    "same_vocab": ([b"a", b"b", b"c"], [b"a", b"b", b"c"]),
    "different_vocabs": ([b"a", b"b", b"c"], [b"a", b"b", b"d"]),
}


@pytest.fixture(params=list(VOCAB_CASES.keys()))
def product(request):
    v1, v2 = VOCAB_CASES[request.param]
    p1 = SimplePotential(v1, scale=1.0)
    p2 = SimplePotential(v2, scale=2.0)
    return Product(p1, p2)


def test_initialization_same_vocab():
    base_vocab = [b"a", b"b", b"c"]
    product = Product(
        SimplePotential(base_vocab, scale=1.0), SimplePotential(base_vocab, scale=2.0)
    )
    assert product.token_type == Atomic(bytes)
    assert len(product.vocab) == len(base_vocab)
    assert product.vocab == base_vocab
    assert product.v1_idxs == ...
    assert product.v2_idxs == ...


def test_initialization_different_vocab():
    product = Product(
        SimplePotential([b"a", b"b", b"c"], scale=1.0),
        SimplePotential([b"a", b"b", b"d"], scale=2.0),
    )
    assert product.token_type == Atomic(bytes)
    assert len(product.vocab) == 2
    assert product.v1_idxs != ...
    assert product.v2_idxs != ...
    assert len(product.v1_idxs) == 3  # (2 + eos)
    assert len(product.v2_idxs) == 3  # (2 + eos)


def test_vocab_errors():
    p1 = SimplePotential([b"a", b"b", b"c"], scale=1.0)

    # Test mismatched token types
    class DifferentPotential(SimplePotential):
        def __init__(self):
            super().__init__([1, 2, 3])  # Different token type (int)

    with pytest.raises(
        ValueError, match="Potentials in product must have the same token type"
    ):
        Product(p1, DifferentPotential())

    # Test non-overlapping vocabularies
    p3 = SimplePotential([b"e", b"f", b"g"])
    with pytest.raises(
        ValueError, match="Potentials in product must share a common vocabulary"
    ):
        Product(p1, p3)


@pytest.mark.asyncio
async def test_prefix(product):
    context = [b"a", b"b"]
    result = await product.prefix(context)
    # Should be sum of both potentials' prefix values
    expected = -0.5 * len(context) * (1.0 + 2.0)
    assert result == expected


@pytest.mark.asyncio
async def test_complete(product):
    context = [b"a", b"b"]
    result = await product.complete(context)
    # Should be sum of both potentials' complete values
    expected = -len(context) * (1.0 + 2.0)
    assert result == expected


@pytest.mark.asyncio
async def test_logw_next(product):
    context = [b"a", b"b"]
    result = await product.logw_next(context)

    # Test that weights are properly combined
    weights = result.weights
    assert len(weights) == len(product.vocab_eos)

    # Test individual token weights
    for token in product.vocab:
        extended = context + [token]
        score = await product.score(extended)
        prefix_score = await product.prefix(context)
        expected_weight = score - prefix_score
        assert np.isclose(result.weights[product.lookup[token]], expected_weight)


@pytest.mark.asyncio
async def test_batch_operations(product):
    contexts = [[b"a"], [b"a", b"b"]]

    # Test batch_complete
    complete_results = await product.batch_complete(contexts)
    expected = [-3.0, -6.0]  # Combined scales (1.0 + 2.0) * -len(context)
    np.testing.assert_array_almost_equal(complete_results, expected)

    # Test batch_prefix
    prefix_results = await product.batch_prefix(contexts)
    expected = [-1.5, -3.0]  # Combined scales (1.0 + 2.0) * -0.5 * len(context)
    np.testing.assert_array_almost_equal(prefix_results, expected)


@pytest.mark.asyncio
async def test_properties(product):
    # Test the inherited property checks
    await product.assert_logw_next_consistency([b"b", b"a"], verbosity=1)
    await product.assert_autoreg_fact([b"b", b"a"], verbosity=1)
    await product.assert_batch_consistency([[b"b", b"a"], [b"a"]], verbosity=1)


def test_product_repr(product):
    repr(product)


def test_product_spawn(product):
    spawn = product.spawn()
    assert spawn.p1.vocab == product.p1.vocab and isinstance(spawn.p1, type(product.p1))
    assert spawn.p2.vocab == product.p2.vocab and isinstance(spawn.p2, type(product.p2))


def test_product_vocab_overlap():
    vocab = list(range(0, 11))
    p1 = SimplePotential(vocab, scale=1.0)
    p2 = SimplePotential(vocab[:1], scale=2.0)
    # Common vocabulary is less than 10% of p1's vocabulary
    with pytest.warns(RuntimeWarning):
        Product(p1, p2)

    with pytest.warns(RuntimeWarning):
        Product(p2, p1)


@pytest.mark.asyncio
async def test_product_laziness():
    class InfiniteAndCounterPotential(Potential):
        def __init__(self):
            super().__init__([b"a", b"b", b"c"])
            self.prefix_calls = 0
            self.complete_calls = 0

        async def complete(self, context):
            self.complete_calls += 1
            return float("-inf")

        async def prefix(self, context):
            self.prefix_calls += 1
            return float("-inf")

    p1 = InfiniteAndCounterPotential()
    p2 = InfiniteAndCounterPotential()
    product = Product(p1, p2)

    await product.prefix([])
    assert product.p1.prefix_calls == 1
    assert product.p2.prefix_calls == 0

    await product.complete([])
    assert product.p1.complete_calls == 1
    assert product.p2.complete_calls == 0


def test_product_token_type_mismatch():
    p1 = SimplePotential([b"a", b"b", b"c"], scale=1.0)
    p2 = SimplePotential([0, 1, 2], scale=2.0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Potentials in product must have the same token type. "
            + "Got Atomic(bytes) and Atomic(int)."
            + "\nMaybe you forgot to coerce the potentials to the same token type? See `Coerce`."
        ),
    ):
        Product(p1, p2)

    p1 = SimplePotential([b"a", b"b", b"c"], scale=1.0)
    p2 = SimplePotential(["a", "b", "c"], scale=2.0)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Potentials in product must have the same token type. "
            + "Got Atomic(bytes) and Atomic(str)."
        ),
    ):
        Product(p1, p2)
