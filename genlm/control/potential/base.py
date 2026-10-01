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
    """`idx` as an integer array of `dtype`, copied only when it is not one already."""
    if isinstance(idx, np.ndarray):
        return idx.astype(dtype, copy=False)
    return np.fromiter(idx, dtype=dtype)


class VocabTables(NamedTuple):
    """The vocabulary-derived tables a potential needs.

    Built by [build_tables][genlm.control.potential.base.Potential.build_tables] and
    shareable by every potential over the same vocabulary.

    Attributes:
        token_type (TokenType): The type of tokens in the vocabulary.
        eos (EndOfSequence): Special token to use as end-of-sequence.
        vocab_eos (list): List of tokens in the vocabulary and `eos`, `eos` last.
        lookup (dict): Mapping from tokens and `eos` to their indices in `vocab_eos`.
    """

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
        """Build the vocabulary tables a potential is initialized from.

        Potentials over the same `vocabulary`, `token_type` and `eos` can share one result
        via `tables=`.

        Args:
            vocabulary (list): List of tokens that make up the vocabulary.
            token_type (TokenType, optional): Optional TokenType of all elements of the vocabulary.
                If None, will be inferred from vocabulary.
            eos (EndOfSequence, optional): Special token to use as end-of-sequence. Defaults to `EOS` sentinel.

        Returns:
            (VocabTables): The tables `(token_type, eos, vocab_eos, lookup)`.

        Raises:
            ValueError: If vocabulary is empty or contains duplicate tokens.
            TypeError: If vocabulary contains tokens which are not of `token_type`.
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
            tables (VocabTables, optional): Prebuilt tables for `vocabulary`, as returned by
                `build_tables`; `vocabulary` is then not validated. Mutually exclusive with
                `token_type` and `eos`.

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

    @property
    def tables(self):
        """The `VocabTables` this potential was built from."""
        return VocabTables(self.token_type, self.eos, self.vocab_eos, self.lookup)

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

        A terminal-only potential has `prefix(context) == 0` for every context; as an SMC
        critic it is applied only at termination. Override only in subclasses which satisfy
        this invariant.

        Returns:
            (bool): Whether the potential is terminal-only. Defaults to `False`.
        """
        return False

    async def sparse_logw_next(self, context):
        """Compute the finite next-token log weights given `context`, or `None` if unsupported.

        The result is `(indices, values, eos)`: the vocabulary indices with finite weight,
        their log weights (array-like, or one float shared by all of them), and the EOS log
        weight. Potentials which implement this override `_logw_next_dense` rather than
        `logw_next`. A `None` for any context sends the whole batch in `batch_logw_next` to
        the dense path.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (tuple | None): `(indices, values, eos)`, or `None` for no sparse path.
        """
        return None

    def _rows_from_sparse(self, lives):
        """Scatter `sparse_logw_next` triples into one `[N, len(vocab_eos)]` block from `alloc_rows`."""
        V1 = len(self.vocab_eos)
        W = self.alloc_rows(len(lives))
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
        # Bulk writes only: on a device block every store is a separate device op.
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
        live = await self.sparse_logw_next(context)
        if live is not None:
            return self.make_lazy_weights(self._rows_from_sparse([live])[0])
        return await self._logw_next_dense(context)

    async def _logw_next_dense(self, context):
        """Compute the next-token weights given `context` when `sparse_logw_next` returns `None`."""
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
        """Batched equivalent to `logw_next`.

        Computes the next-token weights of each token in `self.vocab_eos` given each context in the batch.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (LazyWeights): Batched weights, `.weights` of shape `[N, V+1]`.

        Raises:
            ValueError: If any context has zero weight (log weight of -inf) under `prefix`.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        lives = await asyncio.gather(*[self.sparse_logw_next(c) for c in contexts])
        if all(live is not None for live in lives):
            return self.make_lazy_weights(self._rows_from_sparse(lives))

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

        Args:
            default (float, optional): Default log weight. Defaults to -inf.

        Returns:
            (array): Array of length `len(self.vocab_eos)` filled with `default`.
        """
        return self.alloc_rows(1, default)[0]

    def alloc_rows(self, n, default=float("-inf")):
        """Allocate a block of log weights for `n` contexts.

        Override to place the block on a device or in another backend.

        Args:
            n (int): Number of rows.
            default (float, optional): Default log weight. Defaults to -inf.

        Returns:
            (array): Array of shape `[n, len(self.vocab_eos)]` filled with `default`.
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
