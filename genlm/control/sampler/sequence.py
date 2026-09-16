import numpy as np
from arsenal import colors
from genlm.grammar import Float
from llamppl import Model, smc_standard
from functools import cached_property
from dataclasses import dataclass

from genlm.control.potential import Potential
from genlm.control.potential.autobatch import autobatched
from genlm.control.constant import EOS, EndOfSequence
from genlm.control.sampler.token import TokenSampler
from genlm.control.util import escape, logsumexp


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

    4. Termination: Each sequence terminates either by naturally sampling an
       end-of-sequence (EOS) token, or by hitting the ``max_tokens`` boundary.
       In the latter case, EOS is deterministically appended to the sequence
       and the particle's importance weight is corrected by
       ``unit_sampler.target.logw_next(context)[EOS]`` (see
       :meth:`TokenSampler.logw_eos`). This makes the resulting particles
       properly weighted with respect to the target distribution conditioned
       on ``|y| <= max_tokens``; every returned sequence ends with EOS.

    If a critic is provided, the resulting sequences are properly weighted with respect to the product of the unit sampler's
    target potential and the critic potential (`unit_sampler.target * critic`). If a critic is not provided,
    the resulting sequences are weighted with respect to the unit sampler's target potential.

    Args:
        unit_sampler (TokenSampler): The sampler that generates tokens.
        critic (Potential, optional): A potential function that guides the generation process
            by scoring candidate sequences. Must have the same token type as the unit_sampler.
        autobatch (bool): Whether to wrap the critic in
            [`AutoBatchedPotential`][genlm.control.potential.autobatch.AutoBatchedPotential],
            so that concurrent per-particle scores execute as one batched call.
            Default True. The unit sampler's own seats take a separate `autobatch`
            flag, at the sampler's construction.

    Raises:
        ValueError: If unit_sampler is not a TokenSampler, if critic is not a Potential,
            or if the token types of unit_sampler and critic don't match.
    """

    def __init__(self, unit_sampler, critic=None, autobatch=True):
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
        self.critic = autobatched(critic) if autobatch else critic

    async def __call__(
        self,
        n_particles,
        ess_threshold,
        max_tokens,
        *,
        verbosity=0,
        json_path=None,
        resampling_method="multinomial",
        terminate_when=None,
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
            max_tokens (int): Maximum sequence length (including the terminal EOS token).
                Sequences that haven't naturally sampled EOS by the boundary have EOS
                deterministically appended, with an importance-weight correction so the
                particles target the length-conditioned distribution.
            verbosity (int, optional): Verbosity level for the SMC algorithm. 0 is silent, 1 prints the
                particles at each step. Default is 0.
            json_path (str, optional): JSON file path for saving a record of the inference run.
                This can be used in conjunction with the `InferenceVisualizer` to visualize the inference run.
            resampling_method (str, optional): One of 'multinomial', 'stratified',
                'systematic', 'residual'. Defaults to 'multinomial'.
            terminate_when (callable, optional): A `context -> bool` stop condition.
                When it fires, EOS closes the sequence in that same step, with no
                importance correction: the condition defines which sequences are
                complete, so it is part of the target rather than a truncation of it.
                Contrast `max_tokens`, which cuts a sequence the model would have
                continued and therefore does correct.

        Returns:
            (Sequences): A container holding the generated sequences, their importance weights, and
                other metadata from the generation process.
        """
        assert max_tokens > 0
        # A terminal-only critic has no per-step signal: reweight only at termination.
        twist_with_critic = (
            ess_threshold > 0
            and self.critic is not None
            and not self.critic.is_terminal_only()
        )
        model = SequenceModel(
            unit_sampler=self.unit_sampler,
            critic=self.critic,
            max_tokens=max_tokens,
            twist_with_critic=twist_with_critic,
            terminate_when=terminate_when,
            verbosity=verbosity,
        )

        particles = await smc_standard(
            model=model,
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            resampling_method=resampling_method,
            json_file=json_path,
        )

        return Sequences(*_unpack_particles(particles))

    async def cleanup(self):
        """Clean up resources used by the inference engine.

        This method should be called when the InferenceEngine is no longer needed.

        Example:
            ```python
            sampler = SMC(unit_sampler, critic)
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
    """One particle: a candidate sequence's state and its per-step semantics.

    The per-particle state is ``context`` and ``max_tokens``, alongside ``Model``'s
    weight, twist and finished flags; the sampler, critic and configuration are
    shared across siblings (see ``immutable_properties``).

    Args:
        unit_sampler (TokenSampler): Draws one unit per step via ``sample``.
        critic (Potential, optional): Reweights and twists the particle.
        max_tokens (int): Per-particle token budget; EOS is forced at the boundary.
        twist_with_critic (bool): Whether the critic twists during stepping, rather
            than scoring once at termination.
        terminate_when (callable, optional): ``context -> bool`` stop condition.
            When it fires, EOS closes the sequence in that same step.
        verbosity (int): 0 is silent, 1 prints the particle at each step.
    """

    def __init__(
        self,
        unit_sampler,
        critic=None,
        max_tokens=float("inf"),
        twist_with_critic=True,
        terminate_when=None,
        verbosity=0,
    ):
        super().__init__()
        self.unit_sampler = unit_sampler
        self.critic = critic
        self.max_tokens = max_tokens
        self.twist_with_critic = twist_with_critic
        self.terminate_when = terminate_when
        self.verbosity = verbosity
        self.context = []

    def immutable_properties(self):
        """Properties shared by every particle; the rest is deep-copied per particle."""
        return {
            "unit_sampler",
            "critic",
            "twist_with_critic",
            "terminate_when",
            "verbosity",
        }

    def __deepcopy__(self, memo):
        # Tokens are immutable, so the context copies shallowly and EOS keeps its identity.
        memo[id(self.context)] = list(self.context)
        return super().__deepcopy__(memo)

    def score(self, amt):
        """Add ``amt`` to the log-weight. A ``+inf`` log-weight violates the potential
        contract; NaN folds to ``-inf``."""
        if amt == float("inf"):
            raise ValueError(
                "A potential returned a log-weight of +inf, which violates the "
                "potential contract."
            )
        super().score(amt)

    async def start(self):
        """Score the empty sequence's prefix weight."""
        start_w = await self.unit_sampler.start_weight()
        if start_w == float("-inf"):
            raise ValueError(
                "Start weight is -inf (log(0)). This is likely because a potential "
                "assigns zero weight to the empty sequence under `prefix`, which "
                "violates the potential contract."
            )
        self.score(start_w)

    async def step(self):
        """Advance the particle by one unit, forcing EOS at the token budget.
        The caller untwists first, as ``smc_standard`` does."""
        if self.max_tokens == 1:
            logw = await self.unit_sampler.logw_eos(self.context)
            unit = EOS
        else:
            unit, logw, _ = await self.unit_sampler.sample(self.context)

        self.score(logw)
        self._append(unit)

        if self.weight == float("-inf"):
            self.finish()
            return

        twist_amt = None
        if self.critic is not None and self.twist_with_critic:
            twist_amt = float(await self.critic.score(self.context))
            if twist_amt == float("-inf"):
                self.score(twist_amt)
                self.finish()
                return
            self.twist(twist_amt)

        if self.verbosity > 0:
            print(self.__repr__())

        self.max_tokens -= 1
        if self.max_tokens == 0 or self.context[-1] is EOS:
            self.finish()
            if self.critic is None:
                return
            if twist_amt is None:
                # Terminal-only critic: reweight once, at termination.
                self.score(float(await self.critic.score(self.context)))
            else:
                # `finish` took the twist back; at termination the critic's score
                # is real weight.
                self.score(twist_amt)

    def _append(self, unit):
        """Extend the context by one drawn unit.

        A multi-token unit ending in EOS is split, so that ``context[-1] is EOS``
        whenever the sequence is terminal. ``terminate_when`` appends EOS in the
        step that satisfied it and carries no weight correction: the stop
        condition defines what a complete sequence is.
        """
        if isinstance(unit, list) and unit and unit[-1] is EOS:
            if len(unit) > 1:
                self.context.append(unit[:-1])
            self.context.append(EOS)
        else:
            self.context.append(unit)
        if (
            self.terminate_when is not None
            and self.context[-1] is not EOS
            and self.terminate_when(self.context)
        ):
            self.context.append(EOS)

    def string_for_serialization(self):
        """The particle's context as the inference record stores it."""
        return "|".join(escape(y) for y in self.context)

    def __repr__(self):
        return (
            f"{self.weight:.2f}:\t"
            + colors.magenta % "["
            + (colors.magenta % "|").join(escape(y) for y in self.context)
            + colors.magenta % "]"
        )


def _unpack_particles(particles):
    contexts, logws = map(
        list, zip(*[(p.context, p.weight) for p in particles])
    )
    return contexts, logws
