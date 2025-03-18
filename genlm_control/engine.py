from genlm_control.potential import Potential
from genlm_control.sampler.token import TokenSampler
from genlm_control.sampler.sequence import SequenceModel, _unpack_particles, Sequences

from hfppl import smc_standard


class InferenceEngine:
    """High-level entrypoint for controlled generation.

    This class implements sequential Monte Carlo (SMC) inference for controlled text generation.
    The generation process works as follows:

    1. Token Sampling: At each step, the `unit_sampler` is used to extend each particle (candidate sequence)
       by sampling a new token. This grows all sequences by one token at a time. The sampler also outputs
       an importance weight with each extension to correct for the myopic nature of token-by-token sampling.

    2. Critic Evaluation: If a `critic` is provided, it scores the updated sequences (via it's `score` method),
       reweighting the particles based on how well they satisfy the constraints encoded by the critic.

    3. Resampling: When the effective sample size (ESS) falls below the threshold,
       particles are resampled according to their weights. This helps focus computation
       on more promising sequences.

    4. Termination: The process continues until either:\n
        - All sequences reach an end-of-sequence (EOS) token\n
        - The maximum token length is reached

    If a critic is provided, the resulting sequences are properly weighted with respect to the product of the unit sampler's
    target potential and the critic potential (`unit_sampler.target * critic`). If a critic is not provided,
    the resulting sequences are weighted with respect to the unit sampler's target potential.

    Args:
        unit_sampler (TokenSampler): The sampler that generates tokens.
        critic (Potential, optional): A potential function that guides the generation process
            by scoring candidate sequences. Must have the same token type as the unit_sampler.

    Raises:
        ValueError: If unit_sampler is not a TokenSampler, if critic is not a Potential,
            or if the token types of unit_sampler and critic don't match.
    """

    def __init__(self, unit_sampler, critic=None):
        if not isinstance(unit_sampler, TokenSampler):
            raise ValueError("`unit_sampler` must be a TokenSampler")

        if critic:
            if not isinstance(critic, Potential):
                raise ValueError("`critic` must be a Potential")
            if not unit_sampler.token_type == critic.token_type:
                raise ValueError(
                    "`critic` must have the same token type as the `unit_sampler`. "
                    f"Got {unit_sampler.token_type} and {critic.token_type}."
                    + (
                        "\nMaybe you forgot to coerce the critic to the token type of the unit sampler? See `Coerce`."
                        if unit_sampler.token_type.is_iterable_of(critic.token_type)
                        else ""
                    )
                )

        self.unit_sampler = unit_sampler
        self.critic = critic
        self.model = SequenceModel(
            unit_sampler=unit_sampler, critic=critic, max_tokens=float("inf")
        )

    async def __call__(
        self,
        n_particles,
        ess_threshold,
        max_tokens,
        verbosity=0,
        json_path=None,
        **kwargs,
    ):
        """Generate sequences using sequential Monte Carlo inference.

        Args:
            n_particles (int): Number of particles (candidate sequences) to maintain during
                generation. Higher values provide better exploration but require more
                computation.
            ess_threshold (float): Effective sample size threshold for resampling,
                expressed as a fraction of the number of particles. When ESS falls below
                this value, particles are resampled according to their weights. Should be between 0 and 1.
                Higher values lead to more frequent resampling. Note that when ess_threshold = 0,
                the critic is only applied at the end of the generation (if it is provided).
            max_tokens (int): Maximum number of tokens to generate per sequence. Generation
                may terminate earlier if all sequences reach an EOS token.
            verbosity (int, optional): Verbosity level for the SMC algorithm. 0 is silent, 1 prints the
                particles at each step. Default is 0.
            json_path (str, optional): JSON file path for saving a record of the inference run.
                This can be used in conjunction with the `InferenceVisualizer` to visualize the inference run.
            **kwargs (dict): Additional keyword arguments to pass to the SMC algorithm.
                See the `hfppl.inference.smc_standard` documentation for more details.

        Returns:
            (Sequences): A container holding the generated sequences, their importance weights, and
                other metadata from the generation process.
        """
        try:
            original_max_tokens = self.model.max_tokens
            original_verbosity = self.model.verbosity
            original_twist_with_critic = self.model.twist_with_critic
            self.model.max_tokens = max_tokens
            self.model.verbosity = verbosity
            self.model.twist_with_critic = ess_threshold > 0

            particles = await smc_standard(
                model=self.model,
                n_particles=n_particles,
                ess_threshold=ess_threshold,
                json_file=json_path,
                **kwargs,
            )
        finally:
            self.model.max_tokens = original_max_tokens
            self.model.verbosity = original_verbosity
            self.model.twist_with_critic = original_twist_with_critic

        return Sequences(*_unpack_particles(particles))

    async def cleanup(self):
        """Clean up resources used by the inference engine.

        This method should be called when the InferenceEngine is no longer needed.

        Example:
            ```python
            engine = InferenceEngine(sampler, critic)
            try:
                sequences = await engine(n_particles=10, ess_threshold=0.5, max_tokens=20)
            finally:
                await engine.cleanup()
            ```
        """
        await self.unit_sampler.cleanup()
        if self.critic:
            await self.critic.cleanup()
