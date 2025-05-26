import asyncio
import numpy as np
from abc import ABC, abstractmethod, abstractproperty

from genlm.control.constant import EOS, EndOfSequence
from genlm.control.util import LazyWeights
from genlm.control.typing import TokenType, infer_vocabulary_type
from genlm.control.potential.operators import PotentialOps
from genlm.control.potential.testing import PotentialTests


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
            eos (EndOfSequence, optional): Special token to use as end-of-sequence. Defaults to `EOS`.
                In general, this should not be set by users.

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

        if eos and not isinstance(eos, EndOfSequence):
            raise ValueError(f"EOS must be an instance of EndOfSequence, got {eos!r}.")

        self.eos = eos or EOS

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
    async def complete(self, context):
        """Assess the weight of `context` as a complete sequence.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context under the language.
        """
        pass  # pragma: no cover

    @abstractmethod
    async def prefix(self, context):
        """Assess the weight of `context` as a prefix.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (float): Log weight of the context as a prefix.
        """
        pass  # pragma: no cover

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
        """Batched equivalent to `logw_next`.

        Computes the next-token weights of each token in `self.vocab_eos` given each context in the batch.

        Args:
            contexts (list): List of sequences of tokens.

        Returns:
            (list): List of LazyWeights objects, one for each context.

        Raises:
            ValueError: If any context has zero weight (log weight of -inf) under `prefix`.
        """
        if not contexts:
            raise ValueError("Contexts must be non-empty.")

        return await asyncio.gather(*[self.logw_next(context) for context in contexts])

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


class ParticleState(ABC):
    def __init__(self, owner):
        self.owner = owner
        self.finished = False
        self.context = []

    async def update_context(self, incremental_context):
        """Update the context with more data that has come in."""
        if self.finished:
            return
        self.context.extend(incremental_context)
        await self.impl_update_context(incremental_context)

    async def finish(self):
        """Mark this state as finished, clearing up any associated
        state, and updating the current score to reflect whether
        this is a valid string in the associated language."""
        if self.finished:
            return
        self.finished = True
        await self.impl_finish()

    @abstractproperty
    def current_score(self):
        """The current score associated with this potential, which
        will reflect whether the current context is a suitable member
        of the language if this has been finished, or whether it is a
        suitable prefix if it has not."""

    @abstractmethod
    async def impl_update_context(self, incremental_context): ...

    @abstractmethod
    async def impl_finish(self): ...

    async def clone(self):
        if self.finished:
            return self
        result = self.owner.new_state()
        await result.update_context(self.context)
        assert self.context == result.context
        assert self.current_score == result.current_score
        return result


class StatefulPotential(Potential):
    def __init__(self, vocabulary, token_type=None, eos=None, state_class=None):
        super().__init__(vocabulary=vocabulary, token_type=token_type, eos=eos)
        self.__state_class = state_class

        self.__previous_states = []

    def new_state(self) -> ParticleState:
        if self.__state_class is None:
            raise NotImplementedError()
        return self.__state_class(self)

    def __look_up_state(self, context):
        # TODO: This is a horrible algorithm and is only here as a placeholder.
        context = list(context)
        for i, state in enumerate(self.__previous_states):
            if context[: len(state.context)] == state.context:
                return i

    async def prefix(self, context):
        i = self.__look_up_state(context)
        if i is None:
            state = self.new_state()
        else:
            state = self.__previous_states[i]
            if state.context == list(context):
                return state.current_score
            else:
                del self.__previous_states[i]

        await state.update_context(context[len(state.context) :])

        # FIXME: Temporary code for debugging some resource leaks,
        # which means we never return the state object to the pool.
        result = state.current_score
        await state.finish()
        return result
        self.__previous_states.append(state)
        return state.current_score

    async def complete(self, context):
        i = self.__look_up_state(context)
        if i is None:
            state = self.new_state()
        else:
            state = self.__previous_states[i]
            del self.__previous_states[i]

        await state.update_context(context[len(state.context) :])
        await state.finish()
        return state.current_score

    async def logw_next(self, context):
        """Compute the next-token weights of each token in `self.vocab_eos` given `context`.

        Args:
            context (list): Sequence of tokens.

        Returns:
            (LazyWeights): Weights of each token in the vocabulary and EOS.
        """
        i = self.__look_up_state(context)
        if i is None:
            state = self.new_state()
            await state.update_context(context)
        else:
            state = await self.__previous_states[i].clone()

        assert not state.finished
        ctx_log_w = state.current_score

        if ctx_log_w == float("-inf"):
            raise ValueError(f"Context {context!r} has weight zero under `prefix`.")

        async def step_score(x):
            local_state = await state.clone()
            await local_state.update_context([x])
            if x == self.eos:
                await local_state.finish()
                return local_state.current_score
            else:
                await local_state.update_context([x])
                result = local_state.current_score
                await local_state.finish()
                return result

        scores = np.array(
            await asyncio.gather(*[step_score(x) for x in self.vocab_eos])
        )

        logws = scores - ctx_log_w

        return self.make_lazy_weights(logws)
