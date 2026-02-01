from typing import List, Literal, Tuple
from collections import defaultdict

from cachetools import LRUCache
from arsenal.maths import logsumexp

from genlm.control.sampler.token import TokenSampler
from genlm.control.util import fast_sample_logprobs
from genlm.control.constant import EOS


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
        eos_tokens (List[int]): List of end-of-sequence tokens (as byte values).
        max_tokens (int, optional): Maximum number of tokens to generate.
        models_equal (bool): Flag indicating whether the two models are identical.
            Defaults to False.

    Example:
        ```python
        from genlm.backend import load_model_by_name
        from genlm.bytes import BeamParams
        from genlm.control.potential.built_in import ByteEnsemble
        from genlm.control.sampler.byte_ensemble import ByteEnsembleTokenSampler

        # Load models
        llm1 = load_model_by_name("gpt2")
        llm2 = load_model_by_name("gpt2")

        # Create ensemble
        ensemble = await ByteEnsemble.create(
            llm1, llm2,
            op="prod",
            prompt1=b"Hello ",
            prompt2=b"Hello ",
            a=0.5
        )

        # Create sampler
        eos_tokens = [llm1.byte_vocab[llm1.tokenizer.eos_token_id]]
        sampler = ByteEnsembleTokenSampler(
            ensemble,
            max_tokens=100,
            eos_tokens=eos_tokens,
            n_particles=10
        )

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
        eos_tokens: List[int] = None,
        max_tokens: int = None,
        models_equal: bool = False,
    ):
        super().__init__(target=potential)
        self.potential = potential
        self.proposal = proposal
        self.n_particles = n_particles
        self.eos_tokens = eos_tokens or []
        self.max_tokens = max_tokens
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
        """Compute the weight of the empty sequence."""
        return 0.0

    async def sample(self, context: List[int], draw=None) -> Tuple[int, float, float]:
        """Sample one token from the ensemble distribution.

        This method:
        1. Fetches beam states for both models at the current context
        2. Gets next-token distributions from both beams
        3. Combines distributions using the ensemble operation
        4. Samples a token using the combined distribution
        5. Advances both beams synchronously with the sampled token
        6. Updates caches with new beam states and weights

        Args:
            context (List[int]): Current context as list of byte values
            draw (callable, optional): Drawing function (not used, for compatibility)

        Returns:
            Tuple[int, float, float]: (token, log_weight, log_prob)
                - token: Sampled byte value (or EOS)
                - log_weight: Log importance weight for SMC
                - log_prob: Log probability under proposal distribution
        """
        # Get beam states
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

        # Combine weights using ensemble operation and compute proposal
        proposal_weights = self.potential.op(logws1, logws2) - log_shaping_weight_prev
        logps = proposal_weights - logsumexp(proposal_weights)

        # Sample token from proposal distribution
        token_idx = fast_sample_logprobs(logps)[0]

        # Decode token from trie
        token = beam1.states[0].trie.trie.decode[token_idx]
        assert (
            token == beam2.states[0].trie.trie.decode[token_idx]
        ), "Models must have aligned vocabularies"

        # Advance both beams synchronously with sampled token
        next_context = (
            bytes(context + [token])
            if isinstance(token, int)
            else bytes(context) + token
        )
        self.potential.data_dict_1[next_context] = await (beam1.prune() << token)
        self.potential.data_dict_2[next_context] = await (beam2.prune() << token)

        # Update prefix caches
        new_ctx_tuple = ctx_tuple + (token,)
        self.prefix_cache_1[new_ctx_tuple] = logws1[token_idx]
        self.prefix_cache_2[new_ctx_tuple] = logws2[token_idx]

        # Handle EOS tokens
        if token in self.eos_tokens:
            token = EOS

        # Store final particle weights if sequence is complete
        if token == EOS or (self.max_tokens and len(ctx_tuple) + 1 == self.max_tokens):
            self.particle_prefix_log_prob_1[ctx_tuple + (token,)] = logws1[token_idx]
            self.particle_prefix_log_prob_2[ctx_tuple + (token,)] = logws2[token_idx]

        # Return token, importance weight, and proposal log prob
        return token, proposal_weights[token_idx] - logps[token_idx], logps[token_idx]

    async def smc(
        self,
        n_particles: int,
        ess_threshold: float,
        max_tokens: int,
        critic=None,
        **kwargs,
    ):
        """Run Sequential Monte Carlo inference with byte-level ensemble.

        This method requires EnsembleSMC to be available in the sampler.sequence module.
        If not available, falls back to standard SMC.

        Args:
            n_particles (int): Number of particles to maintain
            ess_threshold (float): ESS threshold for resampling (0-1)
            max_tokens (int): Maximum tokens per sequence
            critic (Potential, optional): Critic potential for guided sampling
            **kwargs: Additional arguments passed to SMC

        Returns:
            Sequences or SequencesExt: Generated sequences with weights

        Raises:
            ImportError: If required SMC components are not available
        """
        from genlm.control.sampler.sequence import EnsembleSMC

        return await EnsembleSMC(self, critic)(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            **kwargs,
        )
