import numpy as np
import torch

from genlm.control.potential.base import Potential


class Tempered(Potential):
    """A potential with every log weight of `p` scaled by `beta`, written `p ** beta`.

    The result is unnormalized; `(p ** (1/tau)).normalize()` is `p` at temperature `tau`.
    `-inf` weights stay `-inf` for any `beta`.

    Attributes:
        p (Potential): The tempered potential.
        beta (float): The exponent. `1/temperature`.
    """

    def __init__(self, p, beta):
        self.p = p
        self.beta = float(beta)
        super().__init__(p.vocab, tables=p.tables)

    def alloc_rows(self, n, default=float("-inf")):
        return self.p.alloc_rows(n, default)

    def is_terminal_only(self) -> bool:
        # beta * 0 == 0: tempering preserves the prefix == 0 invariant.
        return self.p.is_terminal_only()

    def _scale(self, w):
        """`beta * w` in `w`'s own backend, with `-inf` entries left untouched."""
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
        """`_scale` for a single score."""
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
