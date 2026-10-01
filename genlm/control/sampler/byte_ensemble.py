from typing import Any, List, Literal, Tuple
from collections import defaultdict

from cachetools import LRUCache
from arsenal.maths import logsumexp

from genlm.control.sampler.token import TokenSampler
from genlm.control.util import draw_indices
from genlm.control.constant import EOS
from genlm.control.sampler.sequence import EnsembleSMC

# Slot of EOS in genlm-bytes' next-byte distribution: 256 bytes, then EOT, then EOS.
EOS_IDX = 257


class ByteEnsembleTokenSampler(TokenSampler):
    """
    Token sampler for byte-level ensemble using synchronized beam search.

    This sampler draws from an ensemble of two language models by advancing both
    beam states synchronously with the same sampled token. This enables efficient
    exploration with proper importance weighting for SMC.

    Unlike standard token samplers, ByteEnsembleTokenSampler:
    - Directly accesses and manipulates beam states from ByteEnsemble
    - Advances both beams with the same token (synchronized exploration)
    - Tracks separate log probabilities for each model
    - Uses shaping weights for proper SMC proposals

    Args:
        potential (ByteEnsemble): The target byte-level ensemble potential.
        proposal (Literal["linear", "abs", "square", "soft n"]): Proposal strategy.
            Currently only "linear" is implemented.
        n_particles (int): Number of particles for SMC sampling. Defaults to 10.
        models_equal (bool): Flag indicating whether the two models are identical.
            Defaults to False.

    Example:
        ```python
        from genlm.backend import load_model_by_name
        from genlm.bytes import BeamParams
        from genlm.control.potential.built_in import ByteEnsemble
        from genlm.control.sampler.byte_ensemble import ByteEnsembleTokenSampler

        # Load models
        llm1 = load_model_by_name("openai-community/gpt2")
        llm2 = load_model_by_name("openai-community/gpt2")

        # Create ensemble
        ensemble = await ByteEnsemble.create(
            llm1, llm2,
            op="prod",
            prompt1=b"Hello ",
            prompt2=b"Hello ",
            a=0.5
        )

        # Create sampler
        sampler = ByteEnsembleTokenSampler(ensemble, n_particles=10)

        # Run SMC sampling
        result = await sampler.smc(
            n_particles=10,
            ess_threshold=0.5,
            max_tokens=100
        )
        ```
    """

    def __init__(
        self,
        potential,
        proposal: Literal["linear", "abs", "square", "soft n"] = "linear",
        n_particles: int = 10,
        models_equal: bool = False,
    ):
        super().__init__(target=potential)
        self.potential = potential
        self.proposal = proposal
        self.n_particles = n_particles
        self.models_equal = models_equal

        # LRU caches for prefix weights
        self.prefix_cache_1 = LRUCache(maxsize=3 * n_particles)
        self.prefix_cache_2 = LRUCache(maxsize=3 * n_particles)

        # Track final particle probabilities
        self.particle_prefix_log_prob_1 = defaultdict(lambda: float("-inf"))
        self.particle_prefix_log_prob_2 = defaultdict(lambda: float("-inf"))

        # Init empty context weights
        self.prefix_cache_1[()] = 0.0
        self.prefix_cache_2[()] = 0.0

    async def start_weight(self) -> float:
        """Compute the weight of the empty sequence.

        Returns:
            float: Log weight of the empty sequence (always 0.0)
        """
        return 0.0

    async def _next_weights(self, context: List[int]):
        """Per-model and combined next-byte log weights at `context`.

        Returns:
            Tuple of (beam1, beam2, logws1, logws2, proposal_weights), the weights
            each over genlm-bytes' 258 slots (256 bytes, EOT, EOS). `logws1` and
            `logws2` are each model's prefix weight of `context + [b]`;
            `proposal_weights` combines them with the ensemble operation, relative
            to the combined weight of `context`.
        """
        beam1, beam2 = await self.potential.get_beam_states(context)
        logp_1, logp_2 = await beam1.logp_next(), await beam2.logp_next()

        # Get cached prefix weights
        ctx_tuple = tuple(context)
        log_context_weight_1 = self.prefix_cache_1[ctx_tuple]
        log_context_weight_2 = self.prefix_cache_2[ctx_tuple]

        # Compute next-token weights
        logws1 = log_context_weight_1 + logp_1.ps
        logws2 = log_context_weight_2 + logp_2.ps

        # Compute shaping weight from previous context
        log_shaping_weight_prev = (
            0
            if not context
            else self.potential.op(log_context_weight_1, log_context_weight_2)
        )

        proposal_weights = self.potential.op(logws1, logws2) - log_shaping_weight_prev
        return beam1, beam2, logws1, logws2, proposal_weights

    def _record_final(self, context: List[int], logw1: float, logw2: float):
        """Store each model's weight for a sequence terminated by EOS."""
        final = tuple(context) + (EOS,)
        self.particle_prefix_log_prob_1[final] = logw1
        self.particle_prefix_log_prob_2[final] = logw2

    async def sample(self, context: List[int], draw=None) -> Tuple[int, float, float]:
        """Sample one token from the ensemble distribution.

        This method:
        1. Fetches beam states for both models at the current context
        2. Gets next-byte distributions from both beams
        3. Combines distributions using the ensemble operation
        4. Samples a byte (or EOS) from the combined distribution
        5. Advances both beams synchronously with the sampled byte
        6. Updates caches with new beam states and weights

        Args:
            context (List[int]): Current context as list of byte values
            draw (callable, optional): Not supported; the configured draw method
                (see `set_draw_method`) is used.

        Returns:
            Tuple[int, float, float]: (token, log_weight, log_prob)
                - token: Sampled byte value (or EOS)
                - log_weight: Log importance weight for SMC
                - log_prob: Log probability under proposal distribution

        Raises:
            NotImplementedError: If `draw` is given.
        """
        if draw is not None:
            raise NotImplementedError(
                "ByteEnsembleTokenSampler does not support a custom `draw`."
            )

        beam1, beam2, logws1, logws2, proposal_weights = await self._next_weights(
            context
        )
        logps = proposal_weights - logsumexp(proposal_weights)

        # Sample from the proposal distribution
        token_idx = int(draw_indices(logps))
        logw = proposal_weights[token_idx] - logps[token_idx]

        if token_idx == EOS_IDX:
            self._record_final(context, logws1[token_idx], logws2[token_idx])
            return EOS, logw, logps[token_idx]

        # Advance both beams synchronously with the sampled byte
        token = token_idx
        next_context = bytes(context + [token])
        self.potential.data_dict_1[next_context] = await (beam1.prune() << token)
        self.potential.data_dict_2[next_context] = await (beam2.prune() << token)

        # Update prefix caches
        new_ctx_tuple = tuple(context) + (token,)
        self.prefix_cache_1[new_ctx_tuple] = logws1[token_idx]
        self.prefix_cache_2[new_ctx_tuple] = logws2[token_idx]

        return token, logw, logps[token_idx]

    async def logw_eos(self, context: List[int]) -> float:
        """EOS log-weight at the `max_tokens` boundary, where SMC forces termination.

        Records each model's weight of the terminated sequence, as `sample` does
        when it draws EOS.

        Args:
            context (List[int]): Current context as list of byte values

        Returns:
            float: Log weight of terminating after `context`.
        """
        _, _, logws1, logws2, proposal_weights = await self._next_weights(context)
        self._record_final(context, logws1[EOS_IDX], logws2[EOS_IDX])
        return float(proposal_weights[EOS_IDX])

    async def smc(
        self,
        n_particles: int,
        ess_threshold: float,
        max_tokens: int,
        critic=None,
        **kwargs: Any,
    ):
        """Run Sequential Monte Carlo inference with byte-level ensemble.

        Args:
            n_particles (int): Number of particles to maintain
            ess_threshold (float): ESS threshold for resampling (0-1)
            max_tokens (int): Maximum tokens per sequence
            critic (Potential): Critic potential for guided sampling
            **kwargs: Additional arguments passed to SMC
        """
        return await EnsembleSMC(self, critic)(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            **kwargs,
        )
