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


class _IntVocabPotential(SimplePotential):
    """A potential with a different token type (int), via a subclass rather
    than a direct constructor call, to exercise that construction path too."""

    def __init__(self):
        super().__init__([1, 2, 3])


@pytest.mark.parametrize(
    "make_p1_p2, expected_match",
    [
        pytest.param(
            lambda: (SimplePotential([b"a", b"b", b"c"], scale=1.0), _IntVocabPotential()),
            "Potentials in product must have the same token type",
            id="bytes_vs_int_subclass_loose",
        ),
        pytest.param(
            lambda: (
                SimplePotential([b"a", b"b", b"c"], scale=1.0),
                SimplePotential([0, 1, 2], scale=2.0),
            ),
            re.escape(
                "Potentials in product must have the same token type. "
                + "Got Atomic(bytes) and Atomic(int)."
                + "\nMaybe you forgot to coerce the potentials to the same token type? See `Coerce`."
            ),
            id="bytes_vs_int_exact_with_hint",
        ),
        pytest.param(
            lambda: (
                SimplePotential([b"a", b"b", b"c"], scale=1.0),
                SimplePotential(["a", "b", "c"], scale=2.0),
            ),
            re.escape(
                "Potentials in product must have the same token type. "
                + "Got Atomic(bytes) and Atomic(str)."
            ),
            id="bytes_vs_str_exact_no_hint",
        ),
        pytest.param(
            lambda: (
                SimplePotential([b"a", b"b", b"c"], scale=1.0),
                SimplePotential([b"e", b"f", b"g"]),
            ),
            "Potentials in product must share a common vocabulary",
            id="no_common_vocabulary",
        ),
    ],
)
def test_vocab_errors(make_p1_p2, expected_match):
    p1, p2 = make_p1_p2()
    with pytest.raises(ValueError, match=expected_match):
        Product(p1, p2)


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
@pytest.mark.parametrize("context", [[b"a", b"b"], [b"b", b"a"]], ids=["ab", "ba"])
async def test_properties(product, context):
    # Test that weights are properly combined
    logw_next = await product.logw_next(context)
    assert len(logw_next.weights) == len(product.vocab_eos)

    # Test the inherited property checks
    await product.assert_logw_next_consistency(context, verbosity=1)
    await product.assert_autoreg_fact(context, verbosity=1)
    await product.assert_batch_consistency([context, [b"a"]], verbosity=1)


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
