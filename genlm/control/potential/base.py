import asyncio
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import NamedTuple

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.util import LazyWeights, stack_weights
from genlm.control.typing import TokenType, infer_vocabulary_type
from genlm.control.potential.operators import PotentialOps
from genlm.control.potential.testing import PotentialTests


def _as_ints(idx, dtype):
    """An index run as an integer array, without a per-element pass when it already is
    one. `np.fromiter` and not `torch.tensor(list)`, which is several times slower."""
    if isinstance(idx, np.ndarray):
        return idx.astype(dtype, copy=False)
    return np.fromiter(idx, dtype=dtype)


class VocabTables(NamedTuple):
    """What a vocabulary determines, built once and shareable by every potential over
    it (see `Potential.build_tables`)."""

    token_type: TokenType
    eos: EndOfSequence
    vocab_eos: list
    lookup: dict


class Potential(ABC, PotentialOps, PotentialTests):
    """Abstract base class for potentials.

    A Potential is a function that maps sequences of tokens in a vocabulary to non-negative real numbers (weights).

    Potentials assign weights to sequences of tokens based on whether they are complete sequences or prefixes of complete sequences.

    - `complete`: Assess the log weight of a sequence of tokens in the vocabulary as a complete sequence.
    - `prefix`: Assess the log weight of a sequence of tokens in the vocabulary as a prefix.

    Potentials additionally implement a `logw_next` method:

    - `logw_next`: Compute the next-token log weights of each token in the vocabulary and a special EOS (end-of-sequence) token given a context.

    Subclasses must minimally implement `complete` and `prefix`. `logw_next` and batched versions of the above methods
    come with default implementations, but may be overridden by subclasses for improved performance.

    All Potentials must satisfy a set of properties which can be tested using [PotentialTests][genlm.control.potential.testing.PotentialTests].

    Attributes:
        token_type (TokenType): The type of tokens in the vocabulary.
        vocab (list): List of tokens making up the vocabulary.
        eos (EndOfSequence): Special token to use as end-of-sequence.
        vocab_eos (list): List of tokens in `vocab` and `eos`. `eos` is assumed to be the last token in `vocab_eos`.
        lookup (dict): Mapping from tokens and `eos` to their indices in `vocab_eos`.
    """

    @staticmethod
    def build_tables(vocabulary, token_type=None, eos=None):
        """The tables a vocabulary determines: `(token_type, eos, vocab_eos, lookup)`.

        A function of `(vocabulary, token_type, eos)` alone and O(len(vocabulary)) to
        build, so potentials sharing a vocabulary should build it once and pass it to
        each of them via `tables=` rather than each paying for its own. Validation
        (token types, duplicate tokens) happens here, once.
        """
        if not vocabulary:
            raise ValueError("vocabulary cannot be empty")

        if token_type is None:
            token_type = infer_vocabulary_type(vocabulary)
        elif not isinstance(token_type, TokenType):
            raise ValueError(f"token_type must be a TokenType, got {token_type!r}.")

        if not all(token_type.check(x) for x in vocabulary):
            raise TypeError(f"Tokens in vocabulary must be of type {token_type}.")

        if eos is not None and not isinstance(eos, EndOfSequence):
            raise ValueError("EOS must be an instance of EndOfSequence")
        eos = eos if eos is not None else EOS

        lookup = {}
        for i, x in enumerate(vocabulary):
            if x in lookup:
                raise ValueError(f"Duplicate token {x!r} found in vocabulary")
            lookup[x] = i
        lookup[eos] = len(vocabulary)

        return VocabTables(token_type, eos, vocabulary + [eos], lookup)

    def __init__(self, vocabulary, token_type=None, eos=None, tables=None):
        """
        Initialize the potential.

        Args:
            vocabulary (list): List of tokens that make up the vocabulary.
            token_type (TokenType, optional): Optional TokenType of all elements of the vocabulary.
                If None, will be inferred from vocabulary.
            eos (EndOfSequence, optional): Special token to use as end-of-sequence. Defaults to `EOS` sentinel.
            tables (VocabTables, optional): Prebuilt tables for this exact `vocabulary`,
                as returned by :meth:`build_tables` -- skips the O(len(vocabulary))
                construction and its validation. Mutually exclusive with `token_type`
                and `eos`, which the tables already carry.

        Raises:
            ValueError: If vocabulary is empty, or `tables` does not match `vocabulary`.
            TypeError: If vocabulary contains tokens which are not of `token_type`.
        """
        if tables is None:
            tables = self.build_tables(vocabulary, token_type, eos)
        else:
            if token_type is not None or eos is not None:
                raise ValueError(
                    "`tables` already carries `token_type` and `eos`; pass them to "
                    "`build_tables` instead"
                )
            # Cheap arity check only -- comparing the vocabularies elementwise would
            # cost exactly what injecting the tables is meant to avoid.
            if len(vocabulary) + 1 != len(tables.vocab_eos):
                raise ValueError(
                    f"`tables` covers {len(tables.vocab_eos) - 1} tokens but "
                    f"`vocabulary` has {len(vocabulary)}; they must be built together"
                )

        self.eos = tables.eos
        self.token_type = tables.token_type
        self.vocab = vocabulary
        self.vocab_eos = tables.vocab_eos
        self.lookup = tables.lookup

    ####################
    # Instance methods #
    ####################

    @abstractmethod
    async def complete(self, context) -> float:
        """Assess the weight of `context` as a complete sequence.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context under the language.
        """
        return 0.0  # pragma: no cover

    @abstractmethod
    async def prefix(self, context) -> float:
        """Assess the weight of `context` as a prefix.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context as a prefix.
        """
        return 0.0  # pragma: no cover

    async def logw_eos(self, context) -> float:
        """Assess the log-weight of terminating (EOS) after `context`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of terminating after `context`.
        """
        return float((await self.logw_next(context))[self.eos])

    async def score(self, context):
        """Assess the weight of `context` based on EOS-termination.

        This is a convenience method which dispatches to `complete` if `context` ends with `self.eos`, otherwise to `prefix`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context, either as a prefix or complete sequence.
        """
        if context and context[-1] == self.eos:
            return await self.complete(context[:-1])
        else:
            return await self.prefix(context)

    def is_terminal_only(self) -> bool:
        """Whether this potential contributes weight only at sequence termination.

        A terminal-only potential has ``prefix(context) == 0`` for every proper
        prefix, so it never reweights mid-generation; all of its weight comes from
        ``complete`` (equivalently, ``score`` at EOS). Indicator critics like
        ``1[f(z) == y]`` are the canonical example.

        When used as an SMC critic, returning ``True`` lets the Controller skip
        the per-step critic twist and reweight
        only at termination. In a batched run this fires only when EVERY group's
        critic is terminal-only (the flag is population-wide). Default ``False``;
        override in subclasses that satisfy the ``prefix == 0`` invariant.
        """
        return False

    @property
    def children(self):
        """Sub-potentials this composes (``[]`` for a leaf; ``Product`` -> ``[p1, p2]``)."""
        return []

    async def live_logws(self, context):
        """Live next-token weights as `(indices, values, eos)`: the vocabulary indices
        carrying finite weight, their weights, and the EOS weight -- or `None` when
        this potential has no sparse enumeration.

        `indices` and `values` may be any array-like; a numpy array reaches the scatter
        without a per-element pass, anything else is consumed once. `values` may instead
        be a single float when every live token shares a weight (a support mask), which
        is written as a scalar and builds no value array at all.

        Implementing this replaces the dense per-context build in `logw_next` AND lets
        `batch_logw_next` scatter the whole population into one `alloc_rows` block. An
        override of `logw_next` must consult this first (as `Coerced` does, to keep its
        own fallback), or the scalar and batched lanes disagree. Answer `None` per
        instance, never per context: the batched scatter is all-or-nothing, so a
        sometimes-`None` implementation is walked a second time for the whole batch.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (tuple | None): `(indices, values, eos)`, or `None` for no sparse path.
        """
        return None

    def _rows_from_live(self, lives):
        """Scatter `live_logws` triples into one `[N, len(vocab_eos)]` block from
        `alloc_rows` -- one allocation and one flat write for the whole batch,
        never a dense row per context."""
        V1 = len(self.vocab_eos)
        W = self.alloc_rows(len(lives))
        # Flat indices run to `len(lives) * V1`, which int32 carries for any real
        # vocabulary -- half of what crosses to a device block. Both backends index
        # from it directly, so nothing widens on the way in.
        dtype = np.int32 if len(lives) * V1 < 2**31 else np.int64
        idxs, vals, eoss = [], [], []
        for idx, val, eos in lives:
            idxs.append(_as_ints(idx, dtype) + len(idxs) * V1)
            vals.append(val)
            eoss.append(eos)
        eos_col = np.asarray(eoss, dtype=np.float64)
        flat = packed = None
        if any(len(a) for a in idxs):
            flat = np.concatenate(idxs) if len(idxs) > 1 else idxs[0]
            # One shared weight over the whole block (a support mask) writes as a
            # scalar: no value array is built and none is shipped to the device.
            if all(isinstance(v, (int, float)) for v in vals) and len(set(vals)) == 1:
                packed = float(vals[0])
            else:
                packed = np.concatenate(
                    [
                        np.full(len(a), float(v))
                        if isinstance(v, (int, float))
                        else np.asarray(v, dtype=np.float64)
                        for a, v in zip(idxs, vals)
                    ]
                )
        # Every write to the block happens here, the EOS column in one go rather than
        # one store per row -- on a device block those are a store apiece.
        if torch.is_tensor(W):
            W[:, -1] = torch.from_numpy(eos_col).to(dtype=W.dtype, device=W.device)
            if flat is not None:
                if not isinstance(packed, float):
                    packed = torch.from_numpy(packed).to(
                        dtype=W.dtype, device=W.device
                    )
                W.view(-1)[torch.from_numpy(flat).to(W.device)] = packed
        else:
            W[:, -1] = eos_col
            if flat is not None:
                W.reshape(-1)[flat] = packed
        return W

    async def logw_next(self, context):
        """Compute the next-token weights of each token in `self.vocab_eos` given `context`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.
        """
        live = await self.live_logws(context)
        if live is not None:
            return self.make_lazy_weights(self._rows_from_live([live])[0])
        return await self._logw_next_dense(context)

    async def _logw_next_dense(self, context):
        """The full row, computed without the sparse enumeration. Override this rather
        than `logw_next` when the potential also implements `live_logws`, so the sparse
        lane cannot be bypassed."""
        ctx_log_w = await self.prefix(context)

        if ctx_log_w == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")

        scores = await self.batch_score([[*context, x] for x in self.vocab_eos])
        logws = scores - ctx_log_w

        return self.make_lazy_weights(logws)

    ###################
    # Batched methods #
    ###################

    async def batch_complete(self, contexts):
        """Batched equivalent to `complete`.

        Assess the weight of each context as a complete sequence.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return np.array(
            await asyncio.gather(*[self.complete(context) for context in contexts])
        )

    async def batch_prefix(self, contexts):
        """Batched equivalent to `prefix`.

        Assess the weight of each context as a prefix.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return np.array(
            await asyncio.gather(*[self.prefix(context) for context in contexts])
        )

    async def batch_score(self, contexts):
        """Batched equivalent to `score`.

        Assess the weight of each context based on EOS-termination.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (np.array): Array of log weights for each context.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        complete, prefix = [], []
        complete_indices, prefix_indices = [], []

        for i, context in enumerate(contexts):
            # We want == here instead of `is`.
            if context and context[-1] == self.eos:
                complete.append(context[:-1])
                complete_indices.append(i)
            else:
                prefix.append(context)
                prefix_indices.append(i)

        complete_scores = (
            await self.batch_complete(complete) if complete else np.array([])
        )
        prefix_scores = await self.batch_prefix(prefix) if prefix else np.array([])

        results = np.empty(len(contexts))
        if len(complete_scores) > 0:
            results[complete_indices] = complete_scores
        if len(prefix_scores) > 0:
            results[prefix_indices] = prefix_scores

        return results

    async def batch_logw_next(self, contexts):
        """Batched equivalent to `logw_next`: the next-token weights for every context in the
        batch, as ONE `LazyWeights` whose weights are `[N, V+1]` (the batch dim leads). Row
        `i` is `result.weights[i]`. Default: stack the per-particle `logw_next` (preserving
        backend); `Product` composes batched, `PromptedLLM` serves its injected warm batch.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (LazyWeights): one batched `LazyWeights`, `.weights` shape `[N, V+1]`.

        Raises:
            ValueError: If any context has zero weight (log weight of -inf) under `prefix`.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        lives = await asyncio.gather(*[self.live_logws(c) for c in contexts])
        if all(live is not None for live in lives):
            return self.make_lazy_weights(self._rows_from_live(lives))

        lws = await asyncio.gather(*[self.logw_next(context) for context in contexts])
        return self.make_lazy_weights(stack_weights([lw.weights for lw in lws]))

    #############
    # Utilities #
    #############

    def make_lazy_weights(self, weights, log=True):
        """Helper method to create a LazyWeights object over the potential's vocabulary and EOS.

        Args:
            weights (np.array): Array of weights.
            log (bool, optional): Whether the weights are in log space. Defaults to True.

        Returns:
            (LazyWeights): LazyWeights object defined over `self.vocab_eos`.
        """
        return LazyWeights(
            weights=weights, encode=self.lookup, decode=self.vocab_eos, log=log
        )

    def alloc_logws(self, default=float("-inf")):
        """Allocate a new array of log weights for the potential's vocabulary and EOS.

        One row of :meth:`alloc_rows`, so a potential that overrides that to place its
        weights on a device gets this lane too.

        Args:
            default (float, optional): Default log weight. Defaults to -inf.

        Returns:
            Array of length `len(self.vocab_eos)` filled with `default`.
        """
        return self.alloc_rows(1, default)[0]

    def alloc_rows(self, n, default=float("-inf")):
        """Allocate an `[n, len(vocab_eos)]` weight block. Override to place the
        block on a device (or in another backend); `live_logws` assembly and
        `LazyWeights` both follow whatever this returns.

        Args:
            n (int): Number of rows.
            default (float, optional): Fill value. Defaults to -inf.

        Returns:
            Array of shape `[n, len(self.vocab_eos)]` filled with `default`.
        """
        return np.full((n, len(self.vocab_eos)), default)

    def spawn(self):
        """
        Spawn a fresh instance of the potential.

        This method is not required by default, but may be implemented by subclasses
        to support CPU-parallelism using (`MultiProcPotential`)[genlm.control.potential.multi_proc.MultiProcPotential].
        """
        raise NotImplementedError(
            "Potential.spawn() must be implemented by subclasses."
        )

    async def cleanup(self):
        """
        Cleanup the potential.

        This method may be implemented by subclasses to release resources.
        """
        pass
