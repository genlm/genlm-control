"""``p ** beta`` and ``p.normalize()``: the two single-parent reweightings."""

import numpy as np
import pytest

from genlm.control.potential import Potential


class Weighted(Potential):
    """A potential with an arbitrary per-context weight, so the wrappers are checked
    against something whose ``prefix`` is not itself a sum of normalized steps."""

    def __init__(self, vocab, seed=0):
        super().__init__(vocab)
        self.rng = np.random.default_rng(seed)
        self._w = {}

    def _score(self, context, terminal):
        key = (bytes(b"".join(context)), terminal)
        if key not in self._w:
            self._w[key] = float(self.rng.normal())
        return self._w[key]

    async def complete(self, context):
        return self._score(context, True)

    async def prefix(self, context):
        return self._score(context, False)


VOCAB = [b"a", b"b", b"c"]
CONTEXTS = [[], [b"a"], [b"b", b"c"], [b"a", b"a", b"b"]]


@pytest.fixture
def p():
    return Weighted(VOCAB)


@pytest.mark.asyncio
@pytest.mark.parametrize("beta", [0.5, 1.0, 2.0, -1.0])
async def test_tempered_contract(p, beta):
    q = p**beta
    for ctx in CONTEXTS:
        await q.assert_logw_next_consistency(ctx, top=None)
        await q.assert_autoreg_fact(ctx)
    await q.assert_batch_consistency(CONTEXTS)


def test_tempered_preserves_terminal_only(p):
    """beta * 0 == 0: a terminal-only potential stays terminal-only tempered."""

    class TerminalOnly(Weighted):
        def is_terminal_only(self):
            return True

    assert (TerminalOnly(VOCAB) ** 0.5).is_terminal_only()
    assert not (p**0.5).is_terminal_only()


@pytest.mark.asyncio
async def test_tempered_scales(p):
    q = p**2.5
    for ctx in CONTEXTS:
        assert await q.prefix(ctx) == pytest.approx(2.5 * await p.prefix(ctx))
        assert await q.complete(ctx) == pytest.approx(2.5 * await p.complete(ctx))


@pytest.mark.asyncio
async def test_tempered_keeps_hard_zeros(p):
    """``-inf * 0`` must stay ``-inf``, not become ``nan``: a mask is not resurrected."""

    class Masked(Potential):
        async def complete(self, context):
            return 0.0 if all(t == b"a" for t in context) else float("-inf")

        async def prefix(self, context):
            return await self.complete(context)

    m = Masked(VOCAB)
    w = (await (m**0.0).logw_next([b"a"])).weights
    assert not np.isnan(w).any()
    assert w[m.lookup[b"b"]] == float("-inf")


@pytest.mark.asyncio
@pytest.mark.parametrize("beta", [0.0, 0.5, -1.0])
async def test_tempered_scalar_and_batched_agree_on_minus_inf(beta):
    """``beta * -inf`` is ``nan`` at ``beta == 0``, so the two lanes must share one
    guard or a masked context reads a different score depending on which is called."""

    class Masked(Potential):
        async def complete(self, context):
            return 0.0 if all(t == b"a" for t in context) else float("-inf")

        async def prefix(self, context):
            return await self.complete(context)

    q = Masked(VOCAB) ** beta
    dead = [b"b"]
    assert await q.prefix(dead) == float("-inf")
    assert await q.complete(dead) == float("-inf")
    assert float((await q.batch_prefix([dead]))[0]) == float("-inf")
    assert float((await q.batch_complete([dead]))[0]) == float("-inf")


@pytest.mark.asyncio
async def test_normalized_contract(p):
    q = p.normalize()
    for ctx in CONTEXTS:
        await q.assert_logw_next_consistency(ctx, top=None)
        await q.assert_autoreg_fact(ctx)
    await q.assert_batch_consistency(CONTEXTS)


@pytest.mark.asyncio
async def test_normalized_rows_sum_to_one(p):
    q = p.normalize()
    for ctx in CONTEXTS:
        w = (await q.logw_next(ctx)).weights
        assert float(np.log(np.exp(w).sum())) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.asyncio
async def test_normalized_start_weight_is_the_inner_prefix(p):
    """``prefix([])`` spans no step, so it is untouched -- what a sampler's
    ``start_weight`` reads."""
    assert await p.normalize().prefix([]) == pytest.approx(await p.prefix([]))


@pytest.mark.asyncio
async def test_normalized_honours_an_overridden_prefix():
    """``normalize()`` subtracts the spanned normalizers from the potential's OWN
    ``prefix``; it must not rebuild that prefix out of per-step weights."""

    class OwnPrefix(Weighted):
        async def prefix(self, context):
            return 100.0 + len(context)

    p = OwnPrefix(VOCAB)
    q = p.normalize()
    ctx = [b"a", b"b"]
    spanned = 0.0
    for t in range(len(ctx)):
        w = (await p.logw_next(ctx[:t])).weights
        spanned += float(np.log(np.exp(w).sum()))
    assert await q.prefix(ctx) == pytest.approx(await p.prefix(ctx) - spanned)


@pytest.mark.asyncio
async def test_collapsed_support_carries_no_weight():
    """The point of the local product: a step where the mask leaves one live token
    contributes nothing, instead of the LM's log-probability for that token."""

    class Peaked(Potential):
        async def complete(self, context):
            return -3.0 * len(context)

        async def prefix(self, context):
            return -3.0 * len(context)

    class OnlyA(Potential):
        """Forces ``b"a"`` and never terminates, so exactly one token stays live."""

        async def complete(self, context):
            return float("-inf")

        async def prefix(self, context):
            return 0.0 if all(t == b"a" for t in context) else float("-inf")

    raw = Peaked(VOCAB) * OnlyA(VOCAB)
    ctx = [b"a"]
    unnormalized = (await raw.logw_next(ctx)).weights
    live = np.isfinite(unnormalized)
    assert live.sum() == 1
    # Unnormalized, the forced step still charges the peaked factor's log-probability.
    assert float(unnormalized[live][0]) == pytest.approx(-3.0)
    # Locally normalized, it charges nothing.
    normalized = (await raw.normalize().logw_next(ctx)).weights
    assert float(normalized[live][0]) == pytest.approx(0.0, abs=1e-9)
