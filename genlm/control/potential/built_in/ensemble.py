import asyncio
import warnings
import numpy as np
from typing import Any, Callable, List, Literal, Tuple

from arsenal.maths import logsumexp

from genlm.control.potential.base import Potential
from genlm.control.util import to_numpy
from genlm.bytes import ByteBeamState, BeamParams


class Ensemble(Potential):
    """An ensemble potential combining two language models using a weighted operation.

    The Ensemble class creates a potential that combines log-probabilities from two
    base potentials (typically language models) using a specified weighted operation
    (e.g., weighted geometric mean, arithmetic mean, min, max, etc.).

    Args:
        p1 (Potential): First potential (language model)
        p2 (Potential): Second potential (language model)
        op (str): Operation name (e.g., "sum", "prod", "min", "max", "harmonic", or power means)
        a (float): Weighting parameter between 0 and 1 (default 0.5 for equal weighting).
            When a=0.5, models are weighted equally. For a != 0.5, the combination
            is weighted: a * model1 + (1-a) * model2

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

        # Use ensemble in sampling
        logw = await ensemble.prefix(context)
        ```

    Note:
        The Ensemble class handles vocabulary alignment automatically. Both potentials
        must have compatible vocabularies (typically the same tokenizer). The logw_next
        method is not implemented for Ensemble; instead, use logws_next to get separate
        log weights from each model, or use batch_logw_next for batched operations.
    """

    def __init__(
        self,
        p1: Potential,
        p2: Potential,
        op: str,
        a: float = 0.5,
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

    async def prefix(self, context: List[str]) -> float:
        """Compute log weights for the prefix using both potentials.

        Args:
            context (List[str]): The context tokens

        Returns:
            float: Combined log weight from both potentials using the ensemble operation
        """
        p1_logw, p2_logw = await asyncio.gather(
            self.p1.prefix(context), self.p2.prefix(context)
        )
        return self.op(p1_logw, p2_logw)

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

    async def logws_next(self, context: List[str]) -> Tuple[Any, Any]:
        """Get log weights from both potentials separately.

        This method returns the log weights from both underlying potentials
        without combining them. Useful for custom combination logic.

        Args:
            context (List[str]): The context tokens

        Returns:
            Tuple[Any, Any]: Tuple of (p1_logw_next, p2_logw_next)
        """
        return await asyncio.gather(
            self.p1.logw_next(context), self.p2.logw_next(context)
        )

    async def logw_next(self, context: List[str]):
        """Not implemented for Ensemble class.

        Raises:
            NotImplementedError: Always raised. Use logws_next or batch_logw_next instead.
        """
        raise NotImplementedError("logw_next is not implemented for Ensemble class.")

    async def batch_logw_next(self, contexts: List[List[str]]):
        """Batched version of logw_next for Ensemble.

        This enables batching when multiple particles need to be extended during SMC,
        which can significantly improve performance when using PromptedLLM with
        batch_logw_next support.

        Args:
            contexts (List[List[str]]): List of context token sequences

        Returns:
            (LazyWeights): Batched combined log weights, `.weights` of shape `[N, V+1]`

        Note:
            This method is only used if the Ensemble is wrapped in AutoBatchedPotential or
            called directly with multiple contexts. EnsembleTokenSampler calls p1.logw_next()
            and p2.logw_next() directly, so for batching in EnsembleTokenSampler, wrap p1
            and p2 in AutoBatchedPotential before creating the Ensemble.
        """
        # Get batched log weights from both potentials
        Ws1, Ws2 = await asyncio.gather(
            self.p1.batch_logw_next(contexts), self.p2.batch_logw_next(contexts)
        )
        # Combine using the ensemble operation
        return self.make_lazy_weights(
            self.op(
                to_numpy(Ws1.weights)[:, self.p1_vocab_idxs],
                to_numpy(Ws2.weights)[:, self.p2_vocab_idxs],
            )
        )

    async def logw_eos(self, context: List[str]) -> float:
        """Combined log weight of terminating (EOS) after `context`.

        Args:
            context (List[str]): The context tokens

        Returns:
            float: The ensemble operation applied to both potentials' EOS weights
        """
        p1_logw, p2_logw = await asyncio.gather(
            self.p1.logw_eos(context), self.p2.logw_eos(context)
        )
        return float(self.op(p1_logw, p2_logw))


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
        op: str,
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
            op (str): Operation name ('sum', 'prod', 'min', 'max', 'harmonic', or power means)
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


def _power_mean(p: float, a: float):
    """Create a weighted power mean operator in log space.

    M_p(x, y; a) = (a * exp(p*x) + (1-a) * exp(p*y))^(1/p)
    In log space: (1/p) * logsumexp([log(a) + p*x, log(1-a) + p*y])

    Args:
        p (float): Power parameter for the power mean
        a (float): Weighting parameter between 0 and 1

    Returns:
        Callable: Function that computes weighted power mean in log space
    """
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
            return (1 - 2 * a) * x + 2 * a * func(x, y)
        else:
            return (2 * a - 1) * y + 2 * (1 - a) * func(x, y)

    return lambda x, y: extremum(x, y, a)


_POWER_MEANS = {
    "pm5": -5.0,
    "pm2.5": -2.5,
    "p-2": -2.0,
    "pm1.5": -1.5,
    "pm0.5": -0.5,
    "pm0.25": -0.25,
    "p0.25": 0.25,
    "p0.5": 0.5,
    "p1.5": 1.5,
    "p2": 2.0,
    "p2.5": 2.5,
    "p3": 3.0,
    "p5": 5.0,
}


def convert_to_weighted_logop(
    op: Literal[
        "sum",
        "prod",
        "min",
        "max",
        "harmonic",
        "pm5",
        "pm2.5",
        "p-2",
        "pm1.5",
        "pm0.5",
        "pm0.25",
        "p0.25",
        "p0.5",
        "p1.5",
        "p2",
        "p2.5",
        "p3",
        "p5",
    ],
    a: float = 0.5,
) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Convert a string operation to its weighted log-space equivalent.

    This function takes an operation name and a weighting parameter and returns
    a function that combines two log-probability arrays using the specified
    weighted operation.

    Args:
        op (str): Operation name. Supported operations include:
            - Means: "sum" (arithmetic), "prod" (geometric), "harmonic"
            - Extrema: "min", "max"
            - Power means: "pm5", "pm2.5", "p-2", "pm1.5", "pm0.5", "pm0.25",
                        "p0.25", "p0.5", "p1.5", "p2", "p2.5", "p3", "p5"
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
    """
    if not 0 < a < 1:
        raise ValueError("variable a should be between 0 and 1")

    log_a, log_1_minus_a = np.log(a), np.log(1 - a)

    # Ensemble operations
    if op in _POWER_MEANS:
        return _power_mean(_POWER_MEANS[op], a)

    operations = {
        "sum": lambda x, y: logsumexp([x + log_a, y + log_1_minus_a], axis=0),
        "prod": lambda x, y: a * x + (1 - a) * y,
        "harmonic": lambda x, y: -logsumexp([-x + log_a, -y + log_1_minus_a], axis=0),
        "min": _weighted_extremum(np.minimum, a),
        "max": _weighted_extremum(np.maximum, a),
    }

    if op in operations:
        return operations[op]

    valid_ops = list(operations.keys()) + list(_POWER_MEANS.keys())
    raise ValueError(
        f"Invalid operation: {op}. Must be one of {', '.join(repr(o) for o in valid_ops)}."
    )
