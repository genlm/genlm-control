import asyncio
import warnings
import torch
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
            token_type = self.p1.token_type
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
            super().__init__(self.p1.vocab, token_type=token_type)

        else:
            common_vocab = list(set(self.p1.vocab) & set(self.p2.vocab))
            if not common_vocab:
                raise ValueError("Potentials in product must share a common vocabulary")

            self._check_vocab_overlap(common_vocab, self.p1, self.p2, threshold=0.1)

            self._v1_idxs = None
            self._v2_idxs = None

            super().__init__(common_vocab, token_type=token_type)

    def _check_vocab_overlap(self, common_vocab, p1, p2, threshold=0.1):
        for potential, name in [(p1, "p1"), (p2, "p2")]:
            overlap_ratio = len(common_vocab) / len(potential.vocab)
            if overlap_ratio < threshold:
                warnings.warn(
                    f"Common vocabulary ({len(common_vocab)} tokens) is less than {threshold * 100}% "
                    f"of {name}'s ({potential!r}) vocabulary ({len(potential.vocab)} tokens). "
                    "This Product potential only operates on this relatively small subset of tokens.",
                    RuntimeWarning,
                )

    @property
    def children(self):
        return [self.p1, self.p2]

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

    def _compose(self, w1_full, w2_full):
        """Sum the operands' weights over the shared vocab, slicing the vocab on the LAST
        axis so the same code composes a single ``[V]`` draw and a batched ``[N, V]`` one
        (``v*_idxs`` may be ``...`` when vocabs already match, so index the vocab axis
        directly rather than prepend an ellipsis). The reconcile edge: a burst composes the
        engine-LM operand (GPU torch) with a factor mask (numpy or CPU torch) -- lift the
        numpy/CPU operand to the LM's backend + device. Both-numpy (slow lane) stays numpy,
        byte-identical to the per-token path."""
        w1 = w1_full[self.v1_idxs] if w1_full.ndim == 1 else w1_full[:, self.v1_idxs]
        w2 = w2_full[self.v2_idxs] if w2_full.ndim == 1 else w2_full[:, self.v2_idxs]
        t1, t2 = torch.is_tensor(w1), torch.is_tensor(w2)
        if t1 and t2:
            if w1.device != w2.device:
                dev = w1.device if w1.device.type != "cpu" else w2.device
                w1, w2 = w1.to(dev), w2.to(dev)
        elif t1:  # numpy w2 -> lift to w1's device (dtype preserved -> same promotion)
            w2 = torch.as_tensor(w2, device=w1.device)
        elif t2:
            w1 = torch.as_tensor(w1, device=w2.device)
        return w1 + w2

    async def logw_next(self, context):
        W1, W2 = await asyncio.gather(
            self.p1.logw_next(context), self.p2.logw_next(context)
        )
        return self.make_lazy_weights(self._compose(W1.weights, W2.weights))

    async def logw_eos(self, context) -> float:
        """Sum of the factors' eos log-weights."""
        e1, e2 = await asyncio.gather(
            self.p1.logw_eos(context), self.p2.logw_eos(context)
        )
        return float(e1 + e2)

    async def batch_logw_next(self, contexts):
        W1, W2 = await asyncio.gather(
            self.p1.batch_logw_next(contexts), self.p2.batch_logw_next(contexts)
        )  # each is one batched LazyWeights [N, V_i]; compose over the vocab axis -> [N, V]
        return self.make_lazy_weights(self._compose(W1.weights, W2.weights))

    def spawn(self, p1_opts=None, p2_opts=None):
        return Product(
            self.p1.spawn(**(p1_opts or {})),
            self.p2.spawn(**(p2_opts or {})),
        )

    def __repr__(self):
        return f"Product({self.p1!r}, {self.p2!r})"
