from genlm.control.potential.base import Potential, VocabTables
from genlm.control.util import logsumexp


class Normalized(Potential):
    """A potential with every next-token row renormalized to sum to one.

    The locally normalized factor: `logw_next` sums to 0 at every context, so a step
    whose support collapses to a single token carries no weight. `(llm * mask).normalize()`
    is the local product-of-experts, as opposed to the unnormalized `llm * mask`.

    `prefix` and `complete` subtract the normalizers of the prefixes they span, which
    costs one `batch_logw_next` over `len(context)` prefixes. Nothing on a sampler's
    generation path reads them -- `logw_eos` routes through `logw_next`, and
    `start_weight` only ever asks for `prefix([])` -- so the sweep is a cold path,
    hot only if this is used as a per-step critic.

    Attributes:
        p (Potential): The normalized potential.
    """

    def __init__(self, p):
        self.p = p
        super().__init__(
            p.vocab, tables=VocabTables(p.token_type, p.eos, p.vocab_eos, p.lookup)
        )

    @property
    def children(self):
        return [self.p]

    def alloc_rows(self, n, default=float("-inf")):
        return self.p.alloc_rows(n, default)

    async def logw_next(self, context):
        return (await self.p.logw_next(context)).normalize()

    async def batch_logw_next(self, contexts):
        return (await self.p.batch_logw_next(contexts)).normalize()

    async def _cum_logZ(self, context):
        """`sum_t logsumexp(p.logw_next(x_<t))` over `context`'s proper prefixes."""
        if not context:
            return 0.0
        w = await self.p.batch_logw_next([context[:t] for t in range(len(context))])
        return float(logsumexp(w.weights).sum())

    async def prefix(self, context):
        return await self.p.prefix(context) - await self._cum_logZ(context)

    async def complete(self, context):
        w = await self.p.logw_next(context)
        return (
            await self.p.complete(context)
            - await self._cum_logZ(context)
            - float(logsumexp(w.weights))
        )

    async def cleanup(self):
        await self.p.cleanup()

    def __repr__(self):
        return f"{self.p!r}.normalize()"
