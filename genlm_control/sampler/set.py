import numpy as np
from genlm_grammar import Float
from arsenal.maths import sample_dict, logsumexp
from arsenal.datastructures import LocatorMaxHeap
from abc import ABC, abstractmethod

from genlm_control.util import load_async_trie


class SetSampler(ABC):
    """Base class for set samplers.

    A set sampler samples a weighted set of tokens from a the vocabulary of a `target` potential.

    Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary and a sampled set of tokens $S \\subseteq \\textsf{target.vocab_eos}$,
    the log-weight associated with each token $x \in S$ must correspond to:

    $$
        \\textsf{target.logw_next}(x | x_1, \\ldots, x_{n-1}) - \\log \\Pr(x \\in S)
    $$

    where $\\Pr(x \\in S)$ is the probability the token was included in a sampled set.

    Attributes:
        target (Potential): The target potential with respect to which the set's weights are computed.
    """

    def __init__(self, target):
        self.target = target

    @abstractmethod
    async def sample_set(self, context):
        """Sample a weighted set of tokens from the target potential's vocabulary."""
        pass

    async def trace_swor(self, context):
        from genlm_control.tracer import TraceSWOR

        tracer = TraceSWOR()
        logws = self.target.alloc_logws()
        while tracer.root.mass > 0:
            with tracer:
                set_logws, logp = await self.sample_set(context, draw=tracer)
                for token_id, logw in enumerate(set_logws.weights):
                    if logw == float("-inf"):
                        continue
                    logws[token_id] = logsumexp([logws[token_id], logw + logp])

        return self.target.make_lazy_weights(logws)

    async def cleanup(self):
        pass


class TrieSetSampler(SetSampler):
    """
    TrieSetSampler is a specialized set sampler that utilizes a trie data structure to efficiently sample a weighted set of tokens.

    This sampler is designed to work with two potentials:\n
    - a potential over a vocabulary of iterables (`iter_potential`) and\n
    - a potential over a vocabulary of items which are the elements of the iterables (`item_potential`).

    For example, if `iter_potential` is a potential over `bytes`, then `item_potential` is a potential over `int`s.

    The target potential is the product of `iter_potential` and the `item_potential` coerced to operate on the token type of `iter_potential`. Thus,
    `TrieSetSampler`s sample tokens from the `iter_potential`'s vocabulary.
    """

    def __init__(self, iter_potential, item_potential):
        """
        Initialize the `TrieSetSampler`.

        Args:
            iter_potential (Potential): The potential defined over a vocabulary of iterables.
            item_potential (Potential): The potential defined over a vocabulary of items.

        Raises:
            ValueError: If the token type of `iter_potential` is not an iterable of the token type of `item_potential`.
        """
        if not iter_potential.token_type.is_iterable_of(item_potential.token_type):
            raise ValueError(
                "Token type of `iter_potential` must be an iterable of token type of `item_potential`. "
                f"Got {iter_potential.token_type} and {item_potential.token_type}."
            )
        self.iter_potential = iter_potential
        self.item_potential = item_potential
        self.f = lambda context: [item for items in context for item in items]

        super().__init__(
            iter_potential * item_potential.coerce(iter_potential, f=self.f)
        )

        self.trie_executor = load_async_trie(
            self.iter_potential.vocab_eos, backend="parallel"
        )
        self.trie = self.trie_executor.trie

        vocab_eos = self.target.vocab_eos
        word2leaf = self.trie.word2leaf
        lookup = self.target.lookup

        common_tokens = set(vocab_eos) & set(word2leaf)

        self.leaf_to_token_id = dict(
            (word2leaf[token], lookup[token]) for token in common_tokens
        )

    async def sample_set(self, context):
        """
        Sample a weighted set of tokens given a context.

        Args:
            context (list): The sequence to condition on.

        Returns:
            (LazyWeights, float): A weighted set of tokens and the log-probability of the sampled set.

        Raises:
            NotImplementedError: If the method is not implemented in subclasses.
        """
        raise NotImplementedError("Subclasses must implement sample_set")

    async def cleanup(self):
        """
        Cleanup the TrieSetSampler. It is recommended to call this method at the end of usage.
        """
        await self.trie_executor.cleanup()


class EagerSetSampler(TrieSetSampler):
    """
    A trie-based set sampler that implements an eager sampling strategy
    for generating a set of tokens.

    An `EagerSetSampler` samples tokens by incrementally sampling items from the item-wise product of the `iter_potential` and `item_potential`.
    The sampled set is the set of sequences of items that correspond to valid tokens in `iter_potential`'s vocabulary.
    """

    async def sample_set(self, context, draw=None):
        """
        Sample a set of tokens given a context.

        Args:
            context (list): A sequence of tokens in the `iter_potential`'s vocabulary.

        Returns:
            (LazyWeights, float): A weighted set of tokens and the log-probability of the sampled set.
        """
        if draw is None:
            draw = sample_dict
        iter_logws = await self.iter_potential.logw_next(context)
        item_ws = await self.trie_executor.weight_sum(iter_logws.exp().weights)

        logws = self.target.alloc_logws()
        curr = self.trie.root
        coerced_ctx = self.f(context)
        subtokens = []
        logp, logw = 0, 0

        while True:
            children = self.trie.children[curr]
            item_w_curr = item_ws[curr]
            item_ws1 = Float.chart(
                {a: item_ws[c] / item_w_curr for a, c in children.items()}
            )

            if None in item_ws1:
                leaf = children[None]
                token = self.trie.leaf2word[leaf]
                token_id = self.leaf_to_token_id[leaf]
                logws[token_id] = iter_logws[token] + logw - logp

            item_logws2 = await self.item_potential.logw_next(coerced_ctx + subtokens)
            item_ws2 = item_logws2.exp().materialize()
            w_next = (item_ws1 * item_ws2).trim()

            if not w_next:
                break

            ps = w_next.normalize()
            b = draw(ps)
            logp += np.log(ps[b])
            logw += item_logws2[b]

            if b == self.target.eos:
                assert not subtokens, "subtokens should be empty at EOS."
                logws[-1] = iter_logws[self.target.eos] + logw - logp
                break

            subtokens.append(b)
            curr = children[b]

        return self.target.make_lazy_weights(logws), logp


class TopKSetSampler(TrieSetSampler):
    """
    A trie-based set sampler that lazily enumerates the top K tokens by weight in the target,
    and samples an additional "wildcard" token to ensure absolute continuity.

    Warning:
        This sampler is not guaranteed to be correct if the `item_potential`'s
        prefix weights do not monotonically decrease with the length of the context.
        That is, $\\textsf{item_potential.prefix}(x) \\leq \\textsf{item_potential.prefix}(xy)$ for all sequences of items $x, y$.
    """

    def __init__(self, iter_potential, item_potential, K):
        """
        Initialize the TopKSetSampler.

        Args:
            iter_potential (Potential): The potential defined over a vocabulary of iterables.
            item_potential (Potential): The potential defined over a vocabulary of items.
            K (int|None): The number of top tokens to enumerate. If None, all tokens are enumerated.
        """
        if K is not None and K <= 0:
            raise ValueError("K must be greater than 0 or None")
        super().__init__(iter_potential, item_potential)
        self.K = K

    async def sample_set(self, context, draw=None):
        """
        Sample a set of tokens given a context.

        Args:
            context (list): A sequence of tokens in the `iter_potential`'s vocabulary.

        Returns:
            (LazyWeights, float): A weighted set of tokens and the log-probability of the sampled set.
        """
        if draw is None:
            draw = sample_dict
        iter_logws = await self.iter_potential.logw_next(context)
        max_logws = await self.trie_executor.weight_max(iter_logws.weights)

        k = 0
        logws = self.target.alloc_logws()
        sampled = self.target.alloc_logws(default=False)

        async for token_id, logw in self._lazy_enum(context, max_logws):
            logws[token_id] = logw
            sampled[token_id] = True
            k += 1
            if self.K is not None and k >= self.K:
                break

        logp_wc = 0
        if self.K is not None and k == self.K:
            # Get the distribution over wildcard tokens
            iter_ws = iter_logws.exp()
            W_wc = Float.chart(
                {
                    token_id: iter_ws[token]
                    for token_id, token in enumerate(self.target.vocab_eos)
                    if not sampled[token_id]
                }
            )

            # if W_wc is non-empty, sample a wildcard token to ensure absolute continuity
            if W_wc:
                P_wc = W_wc.normalize()
                wc_id = draw(P_wc)
                logp_wc = np.log(P_wc[wc_id])
                wc = self.target.vocab_eos[wc_id]
                item_ctx = self.f(context)
                prefix_w = await self.item_potential.prefix(item_ctx)
                if wc == self.target.eos:
                    w_guide_wc = await self.item_potential.complete(item_ctx) - prefix_w
                else:
                    w_guide_wc = (
                        await self.item_potential.prefix(self.f(context + [wc]))
                        - prefix_w
                    )
                logws[wc_id] = np.log(W_wc[wc_id]) + w_guide_wc - logp_wc

        return self.target.make_lazy_weights(logws), logp_wc

    async def _lazy_enum(self, context, max_logws):
        agenda = LocatorMaxHeap()

        W = Float.chart()

        # initial conditions
        (token, node) = ((), self.trie.root)
        agenda[token, node, False] = max_logws[node]
        W[node] = 0

        children = self.trie.children
        coerced_ctx = self.f(context)

        curr_priority = float("inf")
        prev_best = float("inf")
        while agenda:
            (token, node, done), score = agenda.popitem()

            assert score <= curr_priority, (
                "Monotonicity assumption violated. "
                "`item_potential` prefix weight must be monotonically decreasing."
            )
            curr_priority = score

            # terminal state
            if done:
                value = W[node] + max_logws[node]
                assert prev_best >= value
                prev_best = value
                yield (self.leaf_to_token_id[node], value)
                continue

            logws = None
            for x, y in children[node].items():
                if x is None:
                    W_y = W[node]
                    W[y] = W_y
                    agenda[token, y, True] = W_y + max_logws[y]
                else:
                    if logws is None:
                        logws = await self.item_potential.logw_next(
                            coerced_ctx + list(token)
                        )
                    W_y = W[node] + logws[x]
                    if W_y == float("-inf"):
                        continue
                    W[y] = W_y
                    agenda[(*token, x), y, False] = W_y + max_logws[y]
