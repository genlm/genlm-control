import asyncio
import numbers
import warnings
import numpy as np
from typing import Any, Callable, List, Tuple, Union

from arsenal.maths import logsumexp
from cachetools import LRUCache

from genlm.control.potential.base import Potential
from genlm.control.util import to_numpy
from genlm.bytes import ByteBeamState, BeamParams


class Ensemble(Potential):
    """An ensemble potential combining two language models using a weighted operation.

    The ensemble's weight of a sequence combines the two potentials' weights of it
    with a weighted operation (e.g., weighted arithmetic or geometric mean, min, max):
    `prefix(x) = op(p1.prefix(x), p2.prefix(x))`, and likewise for `complete`.
    `logw_next` is consistent with `prefix`, so sampling tokens from it (e.g. with
    `direct_token_sampler`) targets this combined distribution over sequences.

    Args:
        p1 (Potential): First potential (language model)
        p2 (Potential): Second potential (language model)
        op (str | float): "sum", "prod", "harmonic", "min", "max", or a power-mean
            exponent p (see `convert_to_weighted_logop`)
        a (float): Weighting parameter between 0 and 1 (default 0.5 for equal weighting).
            When a=0.5, models are weighted equally. For a != 0.5, the combination
            is weighted: a * model1 + (1-a) * model2
        cache_size (int): Number of contexts whose next-token weights are kept, so that
            the potentials' prefix weights of their extensions need no recomputation.
            Set it to at least the number of particles. Defaults to 256.

    Attributes:
        p1: First potential
        p2: Second potential
        op: Weighted log operation function
        p1_vocab_idxs: Indices mapping unified vocabulary to p1's vocabulary
        p2_vocab_idxs: Indices mapping unified vocabulary to p2's vocabulary

    Example:
        ```python
        from genlm.control import PromptedLLM, Ensemble

        # Create two language model potentials
        p1 = PromptedLLM.from_name("gpt2")
        p2 = PromptedLLM.from_name("gpt2")

        # Create an ensemble with weighted geometric mean (a=0.5)
        ensemble = Ensemble(p1, p2, op="prod", a=0.5)

        # Sample from the ensemble with SMC
        sequences = await direct_token_sampler(ensemble).smc(
            n_particles=10, ess_threshold=0.5, max_tokens=20
        )
        ```

    Note:
        Both potentials must have the same vocabulary (typically the same tokenizer).
        To ensemble models with different tokenizers, ensemble them at the byte level
        with `ByteLLM` potentials.
    """

    def __init__(
        self,
        p1: Potential,
        p2: Potential,
        op: Union[str, float],
        a: float = 0.5,
        cache_size: int = 256,
    ):
        self.p1 = p1
        self.p2 = p2
        self.op = convert_to_weighted_logop(op, a)

        # Warn if potentials have different vocabularies
        if set(p1.vocab) != set(p2.vocab):
            warnings.warn(
                "Ensemble is being used with potentials that have different vocabularies. "
                "Consider using ByteEnsemble instead.",
                UserWarning,
                stacklevel=2,
            )

        vocab = list(dict.fromkeys(p1.vocab + p2.vocab))
        super().__init__(vocabulary=vocab)

        self.p1_vocab_idxs = [self.p1.lookup[x] for x in self.vocab_eos]
        self.p2_vocab_idxs = [self.p2.lookup[x] for x in self.vocab_eos]
        assert self.p1_vocab_idxs == self.p2_vocab_idxs

        # context -> (p1 prefix, p2 prefix, p1 next-token row, p2 next-token row)
        self._rows = LRUCache(maxsize=cache_size)

    async def _component_prefixes(self, contexts):
        """Each potential's prefix log weight of each context.

        Read off the cached next-token weights of the context's parent when available,
        and computed with the potentials' `prefix` otherwise.

        Returns:
            (np.ndarray): Shape `[N, 2]`, the two potentials' prefix log weights.
        """
        W = np.empty((len(contexts), 2))
        missing = []
        for n, context in enumerate(contexts):
            parent = self._rows.get(tuple(context[:-1])) if context else None
            if parent is not None:
                w1, w2, row1, row2 = parent
                i = self.lookup[context[-1]]
                W[n] = w1 + row1[i], w2 + row2[i]
            else:
                missing.append(n)
        if missing:
            ctxs = [contexts[n] for n in missing]
            W1, W2 = await asyncio.gather(
                self.p1.batch_prefix(ctxs), self.p2.batch_prefix(ctxs)
            )
            W[missing, 0] = to_numpy(W1)
            W[missing, 1] = to_numpy(W2)
        return W

    async def component_logws(self, context: List[str]) -> Tuple[float, float]:
        """Each potential's log weight of `context`.

        For a context ending in EOS these are the potentials' `complete` weights of
        the sequence before it, otherwise their `prefix` weights. Together with the
        ensemble's own weight, e.g. to record per-model log weights of SMC samples.

        Args:
            context (List[str]): The context tokens, optionally ending in EOS.

        Returns:
            Tuple[float, float]: The log weights under `p1` and `p2`.
        """
        if context and context[-1] is self.eos:
            parent = self._rows.get(tuple(context[:-1]))
            if parent is not None:
                w1, w2, row1, row2 = parent
                i = self.lookup[self.eos]
                return float(w1 + row1[i]), float(w2 + row2[i])
            w1, w2 = await asyncio.gather(
                self.p1.complete(context[:-1]), self.p2.complete(context[:-1])
            )
            return float(w1), float(w2)
        ((w1, w2),) = await self._component_prefixes([context])
        return float(w1), float(w2)

    async def prefix(self, context: List[str]) -> float:
        """Compute log weights for the prefix using both potentials.

        Args:
            context (List[str]): The context tokens

        Returns:
            float: Combined log weight from both potentials using the ensemble operation
        """
        ((w1, w2),) = await self._component_prefixes([context])
        return self.op(w1, w2)

    async def complete(self, context: List[str]) -> float:
        """Compute completion log weights using both potentials.

        Args:
            context (List[str]): The context tokens

        Returns:
            float: Combined completion log weight from both potentials
        """
        p1_logw, p2_logw = await asyncio.gather(
            self.p1.complete(context), self.p2.complete(context)
        )
        return self.op(p1_logw, p2_logw)

    async def logw_next(self, context: List[str]):
        """Next-token log weights, `prefix(context + [x]) - prefix(context)`.

        Args:
            context (List[str]): The context tokens

        Returns:
            (LazyWeights): Log weights over `self.vocab_eos`.
        """
        batch = await self.batch_logw_next([context])
        return batch.spawn(batch.weights[0])

    async def batch_logw_next(self, contexts: List[List[str]]):
        """Batched version of logw_next for Ensemble.

        Args:
            contexts (List[List[str]]): List of context token sequences

        Returns:
            (LazyWeights): Batched log weights, `.weights` of shape `[N, V+1]`, row `n`
                being `prefix(contexts[n] + [x]) - prefix(contexts[n])`.
        """
        (Ws1, Ws2), W = await asyncio.gather(
            asyncio.gather(
                self.p1.batch_logw_next(contexts), self.p2.batch_logw_next(contexts)
            ),
            self._component_prefixes(contexts),
        )
        rows1 = to_numpy(Ws1.weights)[:, self.p1_vocab_idxs]
        rows2 = to_numpy(Ws2.weights)[:, self.p2_vocab_idxs]
        for n, context in enumerate(contexts):
            self._rows[tuple(context)] = (W[n, 0], W[n, 1], rows1[n], rows2[n])
        W1, W2 = W[:, :1], W[:, 1:]
        return self.make_lazy_weights(self.op(W1 + rows1, W2 + rows2) - self.op(W1, W2))


class ByteEnsemble(Potential):
    """
    An ensemble potential combining two language models at the byte level using beam search.

    ByteEnsemble manages synchronized beam states for two language models, enabling efficient
    byte-level ensemble sampling. Unlike the standard Ensemble class that works with any
    Potential, ByteEnsemble provides direct access to beam states for specialized sampling
    strategies like ByteEnsembleTokenSampler.

    Attributes:
        p1, p2: The base LM objects (not Potentials, but raw model objects).
        op: A function to combine log-probabilities.
        data_dict_1, data_dict_2: Beam state caches keyed by context (bytes).
        vocabulary: Byte-level vocabulary (list of integers 0-255).
        eos_tokens: EOS byte strings of the two models.

    Note:
        ByteEnsemble is designed to work with ByteEnsembleTokenSampler for specialized
        byte-level ensemble sampling. The prefix() and complete() methods are not fully
        implemented as this class is meant to be used with custom sampling strategies
        that directly access beam states via get_beam_states().

    Example:
        ```python
        from genlm.backend import load_model_by_name
        from genlm.bytes import BeamParams
        from genlm.control.potential.built_in import ByteEnsemble

        llm1 = load_model_by_name("openai-community/gpt2")
        llm2 = load_model_by_name("openai-community/gpt2")

        ensemble = await ByteEnsemble.create(
            llm1, llm2,
            op="prod",
            prompt1=b"Hello ",
            prompt2=b"Hello ",
            a=0.5
        )

        # Use with ByteEnsembleTokenSampler for sampling
        ```
    """

    def __init__(
        self,
        p1: Any,
        p2: Any,
        op: Callable,
        data_dict_1: dict,
        data_dict_2: dict,
        vocab: List[int],
        eos_tokens: List[bytes],
    ):
        self.p1 = p1
        self.p2 = p2
        self.op = op
        self.data_dict_1 = data_dict_1
        self.data_dict_2 = data_dict_2
        self.eos_tokens = eos_tokens
        super().__init__(vocabulary=vocab)

    @classmethod
    async def create(
        cls,
        llm1: Any,
        llm2: Any,
        op: Union[str, float],
        prompt1: bytes,
        prompt2: bytes,
        a: float = 0.5,
        K: int = 5,
        prune_threshold: float = 0.0,
        verbose: bool = False,
    ) -> "ByteEnsemble":
        """Factory method to initialize beam states from prompts and return a ByteEnsemble instance.

        Args:
            llm1 (Any): First language model (from genlm.backend)
            llm2 (Any): Second language model (from genlm.backend)
            op (str | float): 'sum', 'prod', 'harmonic', 'min', 'max', or a power-mean
                exponent p (see `convert_to_weighted_logop`)
            prompt1 (bytes): Prompt bytes for first model
            prompt2 (bytes): Prompt bytes for second model
            a (float): Weighting parameter between 0 and 1 (default 0.5 for equal weighting)
            K (int): Beam width for beam search (default 5)
            prune_threshold (float): Threshold for pruning low-probability beams (default 0.0)
            verbose (bool): Whether to print verbose beam search output (default False)

        Returns:
            ByteEnsemble: Initialized ensemble with beam states ready for sampling

        Raises:
            RuntimeError: If beam states become empty after prefill
        """

        eos_tokens = [
            llm1.byte_vocab[llm1.tokenizer.eos_token_id].byte_string,
            llm2.byte_vocab[llm2.tokenizer.eos_token_id].byte_string,
        ]

        def beam_params(eos):
            return BeamParams(
                K=K,
                prune_threshold=prune_threshold,
                verbose=verbose,
                eos_byte_strings=[eos],
            )

        data_dict_1 = {}
        data_dict_2 = {}

        async def setup():
            # Initialize beams sequentially to avoid overwhelming vLLM with concurrent requests
            beam1 = await ByteBeamState.initial(llm1, beam_params(eos_tokens[0]))
            beam2 = await ByteBeamState.initial(llm2, beam_params(eos_tokens[1]))
            # Prefill sequentially as well to reduce concurrent load
            beam_state_1 = await beam1.prefill(prompt1)
            beam_state_2 = await beam2.prefill(prompt2)
            return beam_state_1, beam_state_2

        beam_state_1, beam_state_2 = await setup()

        # Check if beams are empty after initialization
        if len(beam_state_1) == 0:
            raise RuntimeError(
                f"Beam1 is empty after prefill with prompt of length {len(prompt1)} bytes"
            )
        if len(beam_state_2) == 0:
            raise RuntimeError(
                f"Beam2 is empty after prefill with prompt of length {len(prompt2)} bytes"
            )

        data_dict_1[b""] = beam_state_1
        data_dict_2[b""] = beam_state_2

        return cls(
            llm1,
            llm2,
            convert_to_weighted_logop(op, a),
            data_dict_1,
            data_dict_2,
            vocab=list(range(256)),
            eos_tokens=eos_tokens,
        )

    async def _cleanup_cache(self):
        """Remove old entries to avoid cache bloat."""
        max_len = max((len(k) for k in self.data_dict_1), default=0)
        min_len = max_len - 2
        for d in [self.data_dict_1, self.data_dict_2]:
            for k in list(d.keys()):
                if len(k) < min_len:
                    del d[k]

    async def get_beam_states(
        self, context: List[int]
    ) -> Tuple["ByteBeamState", "ByteBeamState"]:
        """Fetch beam states for the current context.

        This method provides direct access to the underlying beam states, which
        is used by ByteEnsembleTokenSampler for synchronized beam advancement.

        Args:
            context (List[int]): Context as list of byte values

        Returns:
            Tuple[ByteBeamState, ByteBeamState]: Beam states from both models

        Raises:
            KeyError: If context not found in cache (beam states must be populated
                by ByteEnsembleTokenSampler during sampling)
        """
        ctx_bytes = bytes(context)

        await self._cleanup_cache()
        beam1 = self.data_dict_1[ctx_bytes]
        beam2 = self.data_dict_2[ctx_bytes]
        return beam1, beam2

    async def prefix(self, context: List[int]) -> None:
        """Compute prefix weight (not fully implemented).

        ByteEnsemble is designed to be used with ByteEnsembleTokenSampler which
        manages weights separately. This method is a stub to satisfy the Potential interface.

        Args:
            context (List[int]): The context as list of byte values

        Returns:
            None
        """
        return None  # pragma: no cover

    async def complete(self, context: List[int]) -> None:
        """Compute completion weight (not fully implemented).

        ByteEnsemble is designed to be used with ByteEnsembleTokenSampler which
        manages weights separately. This method is a stub to satisfy the Potential interface.

        Args:
            context (List[int]): The context as list of byte values

        Returns:
            None
        """
        return None  # pragma: no cover


def _power_mean(p: float, a: float) -> Callable:
    """Create a weighted power mean operator in log space.

    M_p(x, y; a) = (a * exp(p*x) + (1-a) * exp(p*y))^(1/p)
    In log space: (1/p) * logsumexp([log(a) + p*x, log(1-a) + p*y])
    p = 0 is the limit, the weighted geometric mean a*x + (1-a)*y.

    Args:
        p (float): Power parameter for the power mean
        a (float): Weighting parameter between 0 and 1

    Returns:
        Callable: Function that computes weighted power mean in log space
    """
    if p == 0:
        return lambda x, y: a * x + (1 - a) * y
    log_a, log_1_minus_a = np.log(a), np.log(1 - a)
    return lambda x, y: (1.0 / p) * logsumexp(
        [log_a + p * x, log_1_minus_a + p * y], axis=0
    )


def _weighted_extremum(func, a: float):
    """Create a weighted min/max operator.

    Args:
        func (Callable): The extremum function (np.minimum or np.maximum)
        a (float): Weighting parameter between 0 and 1

    Returns:
        Callable: Function that computes weighted extremum
    """

    def extremum(x, y, a):
        if a <= 0.5:
            other, coef, ext = x, 1 - 2 * a, 2 * a * func(x, y)
        else:
            other, coef, ext = y, 2 * a - 1, 2 * (1 - a) * func(x, y)
        # At a=0.5 the other term vanishes; skip it so 0 * -inf doesn't give nan.
        return ext if coef == 0 else coef * other + ext

    return lambda x, y: extremum(x, y, a)


def _neginf_for_nan(op: Callable):
    """Map nan results of `op` to -inf.

    The log-space means produce nan only by combining infinities, e.g. two -inf
    inputs, where the mean of zero weights is zero (-inf in log space).
    """

    def safe(x, y):
        with np.errstate(invalid="ignore"):
            result = op(x, y)
        if np.ndim(result) == 0:
            return float("-inf") if np.isnan(result) else result
        return np.where(np.isnan(result), -np.inf, result)

    return safe


# Named means, as power-mean exponents.
_NAMED_POWERS = {"sum": 1.0, "prod": 0.0, "harmonic": -1.0}


def convert_to_weighted_logop(
    op: Union[str, float],
    a: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Convert an operation to its weighted log-space equivalent.

    This function takes an operation and a weighting parameter and returns
    a function that combines two log-probability arrays using the specified
    weighted operation.

    Args:
        op (str | float): The operation:
            - a number p: the weighted power mean
              M_p(x, y) = (a * x^p + (1-a) * y^p)^(1/p), with p = 0 the geometric mean
            - "sum" (arithmetic mean), "prod" (geometric mean), "harmonic": the
              power means with p = 1, 0 and -1
            - "min", "max": weighted extrema
        a (float): Weighting parameter between 0 and 1. When a=0.5, equal weighting.
            For weighted operations: a * model1 + (1-a) * model2

    Returns:
        Callable[[np.ndarray, np.ndarray], np.ndarray]: A function that takes two
            log-probability arrays and returns their weighted combination in log space.

    Raises:
        ValueError: If a is not between 0 and 1, or if op is not recognized.

    Examples:
        >>> op_func = convert_to_weighted_logop("sum", a=0.5)
        >>> x = np.log(np.array([0.3, 0.7]))
        >>> y = np.log(np.array([0.6, 0.4]))
        >>> result = op_func(x, y)  # Weighted arithmetic mean in log space
        >>> result = convert_to_weighted_logop(2.5, a=0.5)(x, y)  # Power mean, p=2.5
    """
    if not 0 < a < 1:
        raise ValueError("variable a should be between 0 and 1")

    if op == "min":
        return _neginf_for_nan(_weighted_extremum(np.minimum, a))
    if op == "max":
        return _neginf_for_nan(_weighted_extremum(np.maximum, a))

    p = _NAMED_POWERS.get(op, op) if isinstance(op, str) else op
    if not isinstance(p, numbers.Real) or isinstance(p, bool):
        valid = ", ".join(repr(o) for o in [*_NAMED_POWERS, "min", "max"])
        raise ValueError(
            f"Invalid operation: {op!r}. Must be one of {valid}, or a number p "
            "for the power mean."
        )
    return _neginf_for_nan(_power_mean(float(p), a))
