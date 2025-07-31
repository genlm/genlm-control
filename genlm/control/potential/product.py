import asyncio
import warnings
from genlm.control.potential.base import Potential


class Product(Potential):
    """
    Combine two potential instances via element-wise multiplication (sum in log space).

    This class creates a new potential that is the element-wise product of two potentials:
    ```
    prefix(xs) = p1.prefix(xs) + p2.prefix(xs)
    complete(xs) = p1.complete(xs) + p2.complete(xs)
    logw_next(x | xs) = p1.logw_next(x | xs) + p2.logw_next(x | xs)
    ```

    The new potential's vocabulary is the intersection of the two potentials' vocabularies.

    This class inherits all methods from [`Potential`][genlm.control.potential.base.Potential],
    see there for method documentation.

    Attributes:
        p1 (Potential): The first potential instance.
        p2 (Potential): The second potential instance.
        token_type (str): The type of tokens that this product potential operates on.
        vocab (list): The common vocabulary shared between the two potentials.

    Warning:
        Be careful when taking products of potentials with minimal vocabulary overlap.
        The resulting potential will only operate on tokens present in both vocabularies.
    """

    def __init__(self, p1, p2):
        """Initialize a Product potential.

        Args:
            p1 (Potential): First potential
            p2 (Potential): Second potential
        """
        self.p1 = p1
        self.p2 = p2

        if self.p1.token_type == self.p2.token_type:
            self.token_type = self.p1.token_type
        else:
            raise ValueError(
                "Potentials in product must have the same token type. "
                f"Got {self.p1.token_type} and {self.p2.token_type}."
                + (
                    "\nMaybe you forgot to coerce the potentials to the same token type? See `Coerce`."
                    if (
                        self.p1.token_type.is_iterable_of(self.p2.token_type)
                        or self.p2.token_type.is_iterable_of(self.p1.token_type)
                    )
                    else ""
                )
            )

        if self.p1.vocab == self.p2.vocab:
            self._v1_idxs = ...
            self._v2_idxs = ...
            super().__init__(self.p1.vocab, token_type=self.token_type)
        else:
            common_vocab = list(set(p1.vocab) & set(p2.vocab))
            if not common_vocab:
                raise ValueError("Potentials in product must share a common vocabulary")

            # Check for small vocabulary overlap
            threshold = 0.1
            for potential, name in [(p1, "p1"), (p2, "p2")]:
                overlap_ratio = len(common_vocab) / len(potential.vocab)
                if overlap_ratio < threshold:
                    warnings.warn(
                        f"Common vocabulary ({len(common_vocab)} tokens) is less than {threshold * 100}% "
                        f"of {name}'s ({potential!r}) vocabulary ({len(potential.vocab)} tokens). "
                        "This Product potential only operates on this relatively small subset of tokens.",
                        RuntimeWarning,
                    )

            self._v1_idxs = None
            self._v2_idxs = None
            super().__init__(common_vocab, token_type=self.token_type)

    @property
    def v1_idxs(self):
        if self._v1_idxs is None:
            self._v1_idxs = [self.p1.lookup[token] for token in self.vocab_eos]
        return self._v1_idxs

    @property
    def v2_idxs(self):
        if self._v2_idxs is None:
            self._v2_idxs = [self.p2.lookup[token] for token in self.vocab_eos]
        return self._v2_idxs

    async def prefix(self, context):
        w1 = await self.p1.prefix(context)
        if w1 == float("-inf"):
            return float("-inf")
        w2 = await self.p2.prefix(context)
        return w1 + w2

    async def complete(self, context):
        w1 = await self.p1.complete(context)
        if w1 == float("-inf"):
            return float("-inf")
        w2 = await self.p2.complete(context)
        return w1 + w2

    async def batch_complete(self, contexts):
        W1, W2 = await asyncio.gather(
            self.p1.batch_complete(contexts), self.p2.batch_complete(contexts)
        )
        return W1 + W2

    async def batch_prefix(self, contexts):
        W1, W2 = await asyncio.gather(
            self.p1.batch_prefix(contexts), self.p2.batch_prefix(contexts)
        )
        return W1 + W2

    async def logw_next(self, context):
        W1, W2 = await asyncio.gather(
            self.p1.logw_next(context), self.p2.logw_next(context)
        )
        return self.make_lazy_weights(
            W1.weights[self.v1_idxs] + W2.weights[self.v2_idxs]
        )

    async def batch_logw_next(self, contexts):
        Ws1, Ws2 = await asyncio.gather(
            self.p1.batch_logw_next(contexts), self.p2.batch_logw_next(contexts)
        )
        return [
            self.make_lazy_weights(
                Ws1[n].weights[self.v1_idxs] + Ws2[n].weights[self.v2_idxs]
            )
            for n in range(len(contexts))
        ]

    def spawn(self, p1_opts=None, p2_opts=None):
        return Product(
            self.p1.spawn(**(p1_opts or {})),
            self.p2.spawn(**(p2_opts or {})),
        )

    def __repr__(self):
        return f"Product({self.p1!r}, {self.p2!r})"
