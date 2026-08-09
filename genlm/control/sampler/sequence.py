import logging
import numpy as np
from genlm.grammar import Float
from genlm.control.util import logsumexp
from functools import cached_property
from dataclasses import dataclass

from genlm.control.potential import Potential
from genlm.control.constant import EOS, EndOfSequence  # noqa: F401 (re-exported)
from genlm.control.sampler.token import TokenSampler
from genlm.control.sampler.smc import Controller

logger = logging.getLogger("genlm.control")


class NotAcceleratable(Exception):
    """Raised by ``accelerate="require"`` when the configuration cannot run with
    engine lanes."""


def _normalize_accelerate(accelerate):
    """Map ``accelerate`` to the canonical "auto"/"off"/"require"; ``True``/``False``
    alias "auto"/"off"."""
    if accelerate is True:
        return "auto"
    if accelerate is False:
        return "off"
    if accelerate in ("auto", "off", "require"):
        return accelerate
    raise ValueError(
        f"`accelerate` must be one of 'auto', 'off', 'require' (or True/False); "
        f"got {accelerate!r}"
    )


async def _drive(controller, mode):
    """Run the controller's loop, with lanes ("auto"/"require") or without
    ("off"). Acceleration is whether rows hold engine lanes; the loop is the
    same either way."""
    if mode != "off":
        from genlm.control.lane_runner import LaneRunner, lane_blocker

        reason = lane_blocker(controller)
        if reason is None:
            if mode == "auto":
                logger.info("running with engine lanes.")
            return await controller.run(lanes=LaneRunner(controller))
        if mode == "require":
            raise NotAcceleratable(reason)
        logger.info(
            "running without engine lanes -- %s. "
            'Pass accelerate="off" to silence, or accelerate="require" to make '
            "this an error.",
            reason,
        )
    return await controller.run()


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
        *,
        accelerate="auto",
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
            max_tokens (int): Maximum sequence length (including the terminal EOS token).
                Sequences that haven't naturally sampled EOS by the boundary have EOS
                deterministically appended, with an importance-weight correction so the
                particles target the length-conditioned distribution.
            accelerate (str | bool, optional): The single engine-acceleration knob,
                keyword-only. One of:\n
                - ``"auto"`` (default, also ``True``): run the engine-accelerated
                  `BurstLoop` when the configuration is burst-capable, else the
                  exact per-token `StepLoop`. Logs (INFO) which path ran, and on
                  fallback the reason it was not accelerated.\n
                - ``"off"`` (also ``False``): always run the exact per-token
                  `StepLoop`, byte-reproducible given a seed.\n
                - ``"require"``: run the engine path, or raise
                  `NotAcceleratable` with the reason if not burst-capable.\n
                Acceleration is vLLM-only for now; the engine is derived from the
                sampler's `PromptedLLM`. The burst is statistically identical to
                `"off"` (same target, unbiased weights) but not byte-identical
                (warm-KV residual + batched-draw RNG); use `"off"` for exact
                reproducibility.
            verbosity (int, optional): Verbosity level for the SMC algorithm. 0 is silent, 1 prints the
                particles at each step. Default is 0.
            json_path (str, optional): JSON file path for saving a record of the inference run.
                This can be used in conjunction with the `InferenceVisualizer` to visualize the inference run.
            **kwargs (dict): Additional keyword arguments to pass to the SMC controller.
                Currently ``resampling_method`` (one of 'multinomial', 'stratified',
                'systematic', 'residual'; defaults to 'multinomial').

        Returns:
            (Sequences): A container holding the generated sequences, their importance weights, and
                other metadata from the generation process.

        Raises:
            NotAcceleratable: If ``accelerate="require"`` but the configuration is
                not burst-capable.
        """
        mode = _normalize_accelerate(accelerate)

        controller = Controller(
            samplers=[self.unit_sampler],
            critics=[self.critic],
            group_sizes=[n_particles],
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            twist_with_critic=ess_threshold > 0,
            record=json_path is not None,
            verbosity=verbosity,
            **kwargs,
        )

        particles = await _drive(controller, mode)

        if json_path is not None:
            controller.save_record(json_path)

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

    @classmethod
    async def batched(
        cls,
        smcs,
        n_particles,
        ess_threshold,
        max_tokens,
        *,
        accelerate="auto",
        verbosity=0,
        **kwargs,
    ):
        """Run ``B = len(smcs)`` :class:`SMC` problems as one batched population.

        ``smcs`` is a list of :class:`SMC` instances (each its own
        ``unit_sampler`` + ``critic``). They run as B independent sub-populations
        ("groups") of ``n_particles`` each in one ``Controller``; ESS / resample /
        log_ml are computed per-group, so each group is statistically identical to
        running that ``SMC`` alone (no cross-group coupling). Returns a list of B
        :class:`Sequences`, one per problem, in ``smcs`` order.

        Run params and ``accelerate`` carry the same meaning as
        :meth:`__call__`; the burst lane needs the batch to be burst-homogeneous
        (one shared forward over all B*N rows -- see
        :func:`~genlm.control.sampler.burst._batch_blocker`), else it falls
        back to the exact per-token loop.
        """
        B = len(smcs)
        controller = Controller(
            samplers=[s.unit_sampler for s in smcs],
            critics=[s.critic for s in smcs],
            group_sizes=[n_particles] * B,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            twist_with_critic=ess_threshold > 0,
            verbosity=verbosity,
            **kwargs,
        )
        await _drive(controller, _normalize_accelerate(accelerate))
        seqs = [
            Sequences(*_unpack_particles([controller.particles[i] for i in rows]))
            for rows in map(controller.group_rows, range(B))
        ]
        for s, record in zip(seqs, controller.records):
            if record is not None:
                s.record = record  # this group's own record stream
        return seqs


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


def _unpack_particles(particles):
    contexts, logws = map(
        list,
        zip(
            *[
                (p.context, float("-inf") if np.isnan(p.logw) else p.logw)
                for p in particles
            ]
        ),
    )
    return contexts, logws
