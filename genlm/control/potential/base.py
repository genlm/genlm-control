import asyncio
import contextlib
import contextvars
import numpy as np
from abc import ABC, abstractmethod

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.util import LazyWeights, stack_weights
from genlm.control.typing import TokenType, infer_vocabulary_type
from genlm.control.potential.operators import PotentialOps
from genlm.control.potential.testing import PotentialTests


# Per-burst override: {potential: LazyWeights} a potential's ``logw_next`` returns for
# itself instead of computing. Read by ``PromptedLLM`` and ``Product``.
_burst_logw_next_overrides: contextvars.ContextVar = contextvars.ContextVar(
    "genlm_control_burst_logw_next", default=None
)


@contextlib.contextmanager
def burst_logw_next(overrides):
    """Inject ``{potential: LazyWeights}`` for one burst step (set per particle task)."""
    token = _burst_logw_next_overrides.set(overrides)
    try:
        yield
    finally:
        _burst_logw_next_overrides.reset(token)


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

    def __init__(self, vocabulary, token_type=None, eos=None):
        """
        Initialize the potential.

        Args:
            vocabulary (list): List of tokens that make up the vocabulary.
            token_type (TokenType, optional): Optional TokenType of all elements of the vocabulary.
                If None, will be inferred from vocabulary.
            eos (EndOfSequence, optional): Special token to use as end-of-sequence. Defaults to `EOS` sentinel.

        Raises:
            ValueError: If vocabulary is empty.
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

        self.eos = eos if eos is not None else EOS

        self.token_type = token_type
        self.vocab = vocabulary
        self.vocab_eos = self.vocab + [self.eos]
        self.lookup = {}
        for i, x in enumerate(vocabulary):
            if x in self.lookup:
                raise ValueError(f"Duplicate token {x!r} found in vocabulary")
            self.lookup[x] = i
        self.lookup[self.eos] = len(self.vocab)

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

        When used as an SMC critic, returning ``True`` makes the Controller skip
        the per-step critic twist (both the slow and burst lanes) and reweight only
        at termination -- byte-identical to twisting per step, since a terminal-only
        critic's per-step twist is ``twist(0)``, but it avoids the per-step critic
        call. In a batched run this fires only when EVERY group's critic is
        terminal-only (the flag is population-wide). The default is ``False``: a
        potential is assumed to reweight per step unless it explicitly opts in.
        Override in subclasses that satisfy the ``prefix == 0`` invariant.
        """
        return False

    @property
    def children(self):
        """Sub-potentials this composes (``[]`` for a leaf; ``Product`` -> ``[p1, p2]``)."""
        return []

    async def logw_next(self, context):
        """Compute the next-token weights of each token in `self.vocab_eos` given `context`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.
        """
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
            (np.array): Array of length `len(self.vocab_eos)` filled with `default`.
        """
        return np.full((len(self.vocab_eos),), default)

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
