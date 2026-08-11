import numpy as np
import torch

from genlm.control.potential.base import Potential, VocabTables


class Tempered(Potential):
    """A potential with every log-weight scaled by `beta`, written `p ** beta`.

    Unnormalized: the next-token rows do not renormalize, so `p ** beta` is a raw
    reweighting. Compose with `.normalize()` for a locally normalized temper --
    `(p ** (1/tau)).normalize()` is `p` at temperature `tau`.

    A hard zero stays hard: `-inf` weights survive any `beta`, so a support mask is
    never resurrected.

    Attributes:
        p (Potential): The tempered potential.
        beta (float): The exponent. `1/temperature`.
    """

    def __init__(self, p, beta):
        self.p = p
        self.beta = float(beta)
        super().__init__(
            p.vocab, tables=VocabTables(p.token_type, p.eos, p.vocab_eos, p.lookup)
        )

    def alloc_rows(self, n, default=float("-inf")):
        return self.p.alloc_rows(n, default)

    def _scale(self, w):
        """``beta * w``, in ``w``'s own backend, leaving ``-inf`` untouched -- scaling
        it would make ``nan`` at ``beta <= 0`` and resurrect a masked token."""
        if torch.is_tensor(w):
            out = w.clone()
            live = ~w.isneginf()
            out[live] = w[live] * self.beta
            return out
        w = np.asarray(w)
        out = w.copy()
        live = ~np.isneginf(w)
        out[live] = w[live] * self.beta
        return out

    def _scale_one(self, v):
        """:meth:`_scale` for a single score -- the scalar and batched lanes must agree
        on ``-inf``, or a masked context reads ``nan`` through one and ``-inf`` through
        the other."""
        v = float(v)
        return v if v == float("-inf") else self.beta * v

    async def prefix(self, context):
        return self._scale_one(await self.p.prefix(context))

    async def complete(self, context):
        return self._scale_one(await self.p.complete(context))

    async def batch_prefix(self, contexts):
        return self._scale(await self.p.batch_prefix(contexts))

    async def batch_complete(self, contexts):
        return self._scale(await self.p.batch_complete(contexts))

    async def logw_next(self, context):
        w = await self.p.logw_next(context)
        return w.spawn(self._scale(w.weights))

    async def batch_logw_next(self, contexts):
        w = await self.p.batch_logw_next(contexts)
        return w.spawn(self._scale(w.weights))

    async def cleanup(self):
        await self.p.cleanup()

    def __repr__(self):
        return f"({self.p!r} ** {self.beta})"
