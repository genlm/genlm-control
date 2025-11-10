import numpy as np
from genlm.grammar import Float
from arsenal.maths import logsumexp
from functools import cached_property
from dataclasses import dataclass
from arsenal import colors

from llamppl import Model
from llamppl import smc_standard

from genlm.control.potential import Potential
from genlm.control.constant import EOS, EndOfSequence
from genlm.control.sampler.token import TokenSampler
from genlm.control.util import escape


class SMC:
    """This class implements sequential Monte Carlo (SMC) inference for controlled text generation.
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
                See the `llamppl.inference.smc_standard` documentation for more details.

        Returns:
            (Sequences): A container holding the generated sequences, their importance weights, and
                other metadata from the generation process.
        """

        # Slightly hacky way to adjust the sequence model based on the token sampler.
        sm = kwargs.pop("sequence_model", None)
        ModelClass = sm if sm else SequenceModel

        model_kwargs = dict(
            unit_sampler=self.unit_sampler,
            critic=self.critic,
            max_tokens=max_tokens,
            verbosity=verbosity,
            twist_with_critic=ess_threshold > 0,
        )

        if ModelClass is MultiTokenSequenceModel or issubclass(
            ModelClass, MultiTokenSequenceModel
        ):
            model_kwargs["stop_tokens"] = kwargs.pop("stop_tokens", set())

        model = ModelClass(**model_kwargs)

        particles = await smc_standard(
            model=model,
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            json_file=json_path,
            **kwargs,
        )
        return Sequences(*_unpack_particles(particles))

    async def cleanup(self):
        """Clean up resources used by the inference engine.

        This method should be called when the InferenceEngine is no longer needed.

        Example:
            ```python
            sampler = SequenceSampler(unit_sampler, critic)
            try:
                sequences = await sampler(n_particles=10, ess_threshold=0.5, max_tokens=20)
            finally:
                await sampler.cleanup()
            ```
        """
        await self.unit_sampler.cleanup()
        if self.critic:
            await self.critic.cleanup()


@dataclass
class Sequences:
    """Container for sequence samples with their weights and probabilities.

    Args:
        contexts (list): List of token sequences generated by the sampler.
        log_weights (list): Log importance weights for each sequence.

    Attributes:
        size (int): Number of sequences in the container.
        logp (float): Sum of log probabilities across all sequences.
        log_total (float): Log of the sum of importance weights.
        log_ml (float): Log marginal likelihood estimate.
        log_normalized_weights (list): Log weights normalized to sum to 1.
        log_ess (float): Log of the effective sample size.
        ess (float): Effective sample size of the particle population.
    """

    contexts: list
    log_weights: list

    def __post_init__(self):
        assert len(self.contexts) == len(self.log_weights)

        if not isinstance(self.log_weights, np.ndarray):
            self.log_weights = np.array(self.log_weights)

        self.size = len(self.contexts)

        # Handle case where all weights are -inf
        if np.all(np.isneginf(self.log_weights)):
            self.log_total = float("-inf")
            self.log_ml = float("-inf")
            self.log_normalized_weights = np.full_like(self.log_weights, float("-inf"))
            self.log_ess = float("-inf")
            self.ess = 0.0
            return

        self.log_total = logsumexp(self.log_weights)
        max_weight = max(self.log_weights)
        self.log_ml = (
            np.log(np.mean(np.exp(self.log_weights - max_weight))) + max_weight
        )
        self.log_normalized_weights = self.log_weights - self.log_total
        self.log_ess = -logsumexp(2 * self.log_normalized_weights)
        self.ess = np.exp(self.log_ess)

    @cached_property
    def posterior(self):
        """Compute the estimated posterior distribution over sequences.

        The probability of a sequence corresponds to its normalized weight. The probabilities
        of duplicate sequences are summed.

        Returns:
            (Float.chart): A normalized chart mapping sequences to their posterior probabilities,
                sorted in descending order by probability.
        """
        posterior = Float.chart()
        for sequence, prob in zip(self.contexts, self.normalized_weights):
            posterior[tuple(sequence)] += prob
        return posterior.normalize().sort_descending()

    @cached_property
    def decoded_posterior(self):
        """Compute posterior distribution over completed UTF-8 decodable sequences.

        Filters for sequences that:\n
        1. End with an EndOfSequence token\n
        2. Can be decoded as UTF-8 strings

        The probability of each sequence corresponds to its normalized weight among completed and decodable sequences.
        Probabilities of duplicate sequences (after decoding) are summed.

        To obtain the posterior distribution over all byte sequences, use `self.posterior`.

        Returns:
            (Float.chart): A normalized chart mapping decoded string sequences to their
                posterior probabilities, sorted in descending order by probability.
                Only includes sequences that meet both filtering criteria.
        """
        posterior = Float.chart()
        for sequence, w in zip(self.contexts, np.exp(self.log_weights)):
            if sequence and isinstance(sequence[-1], EndOfSequence):
                try:
                    string_sequence = b"".join(sequence[:-1]).decode("utf-8")
                    posterior[string_sequence] += w
                except UnicodeDecodeError:
                    pass
        return posterior.normalize().sort_descending()

    @property
    def normalized_weights(self):
        """Return exponential of normalized log weights."""
        if np.all(np.isneginf(self.log_weights)):
            return np.full_like(self.log_weights, 0.0)
        return np.exp(self.log_normalized_weights)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(zip(self.contexts, self.log_weights))

    def __getitem__(self, i):
        return self.contexts[i], self.log_weights[i]

    def __str__(self):
        return str(self.decoded_posterior)

    def _repr_html_(self):
        return self.decoded_posterior._repr_html_()

    def __repr__(self):
        return str(self.decoded_posterior)

    def show(self):
        for p in sorted(self, reverse=True):
            print(p)


class SequenceModel(Model):
    def __init__(
        self,
        unit_sampler,
        critic=None,
        max_tokens=float("inf"),
        verbosity=0,
        twist_with_critic=True,
    ):
        assert max_tokens > 0

        super().__init__()
        self.token_ctx = []
        self.unit_sampler = unit_sampler
        self.max_tokens = max_tokens
        self.critic = critic
        self.logp = 0
        self.verbosity = verbosity
        self.twist_with_critic = twist_with_critic

    async def start(self):
        start_w = await self.unit_sampler.start_weight()
        if start_w == float("-inf"):
            raise ValueError(
                "Start weight is -inf (log(0)). This is likely because a potential assigns zero weight to "
                "the empty sequence under `prefix`, which violates the potential contract."
            )
        self.score(start_w)

    async def step(self):
        unit = await self.call(self.unit_sampler)
        self.token_ctx.append(unit)

        inf_weight = self.weight == float("-inf")
        if inf_weight:
            if self.critic:
                assert self.twist_amount != float("-inf")
            self.finish()
            return

        if self.critic and self.twist_with_critic:
            twist_amt = await self.critic.score(self.token_ctx)
            if twist_amt != float("-inf"):
                self.twist(twist_amt)
            else:
                self.score(twist_amt)
                self.finish()
                return

        if self.verbosity > 0:
            print(self.__repr__())

        self.max_tokens -= 1
        if self.max_tokens == 0 or self.token_ctx[-1] is EOS:
            self.finish()
            if self.critic:
                if not self.twist_with_critic:
                    twist_amt = await self.critic.score(self.token_ctx)
                self.score(twist_amt)
            return

    def __repr__(self):
        return (
            f"{self.weight:.2f}:\t"
            + colors.magenta % "["
            + (colors.magenta % "|").join(escape(y) for y in self.token_ctx)
            + colors.magenta % "]"
        )

    def string_for_serialization(self):
        return "|".join(escape(y) for y in self.token_ctx)

    def immutable_properties(self):
        return set(["unit_sampler", "critic"])


class MultiTokenSequenceModel(Model):
    def __init__(
        self,
        unit_sampler,
        critic=None,
        max_tokens=float("inf"),
        verbosity=0,
        twist_with_critic=True,
        stop_tokens=(),
        patience_init=5,
        check_syntax=True,
    ):
        """
        Sequence model that grows tokens until a boundary word or grammar terminal, then applies a critic.

        Args:
        unit_sampler: TokenSampler used to propose next tokens.
        critic: Optional boundary critic returning.
        max_tokens: Hard cap on total tokens.
        stop_tokens: Tokens that define evaluation boundaries (e.g., newline).
        patience_init: Syntax-error budget before killing the particle.
        twist_with_critic: If True, apply critic at every boundary; else only at the end.
        check_syntax: If True, check syntax at each stop token.
        """
        super().__init__()
        self.unit_sampler = unit_sampler
        self.critic = critic
        self.max_tokens = int(max_tokens) if max_tokens != float("inf") else 10**9
        self.verbosity = verbosity
        self.twist_with_critic = bool(twist_with_critic)
        self.stop_tokens = set(stop_tokens)
        self.token_ctx = []
        self.logp = 0
        self.patience_init = int(patience_init)
        self.patience = self.patience_init
        self._boundary_update = False
        self.check_syntax = check_syntax

    async def start(self):
        self.score(0.0)

    async def step(self):
        self._boundary_update = False

        # Grow until boundary
        while True:
            tok = await self.call(self.unit_sampler)
            self.token_ctx.append(tok)
            self.max_tokens -= 1
            if (tok is EOS) or (tok in self.stop_tokens) or (self.max_tokens <= 0):
                self._boundary_update = True
                break

        # If boundary is EOS/max-length
        if (self.token_ctx[-1] is EOS) or (self.max_tokens <= 0):
            if self.critic:
                twist_amt = await self.critic.score(self.token_ctx)
                self.score(twist_amt)
            self.finish()
            return

        if self.critic and self.twist_with_critic:

            # TODO: We need to generalize additional boundary conditions here.
            if self.check_syntax:
                # this will run prefix() at non-EOS and set last_was_syntax_error
                twist_amt = await self.critic.score(self.token_ctx)
                if getattr(self.critic, "last_was_syntax_error", False):
                    self.patience -= 1
                    if self.patience <= 0:
                        self.score(float("-inf"))
                        self.finish()
                        return
                    return  # keep growing

                # no syntax error use the score
                if twist_amt != float("-inf"):
                    self.patience = self.patience_init
                    self.twist(twist_amt)
                else:
                    self.score(twist_amt)
                    self.finish()
                return
            else:
                # if not checking syntax, proceed as before
                twist_amt = await self.critic.score(self.token_ctx)
                if twist_amt != float("-inf"):
                    self.patience = self.patience_init
                    self.twist(twist_amt)
                else:
                    self.score(twist_amt)
                    self.finish()
                return

    def __repr__(self):
        return (
            f"{self.weight:.2f}:\t"
            + colors.magenta % "["
            + (colors.magenta % "|").join(escape(y) for y in self.token_ctx)
            + colors.magenta % "]"
        )

    def string_for_serialization(self):
        return "|".join(escape(y) for y in self.token_ctx)

    def immutable_properties(self):
        return set(["unit_sampler", "critic"])


def _unpack_particles(particles):
    contexts, logws = map(
        list,
        zip(
            *[
                (p.token_ctx, float("-inf") if np.isnan(p.weight) else p.weight)
                for p in particles
            ]
        ),
    )
    return contexts, logws
