import logging
import numpy as np
from genlm.grammar import Float
from arsenal.maths import logsumexp
from functools import cached_property
from dataclasses import dataclass

from genlm.control.potential import Potential
from genlm.control.constant import EOS, EndOfSequence  # noqa: F401 (re-exported)
from genlm.control.sampler.token import TokenSampler
from genlm.control.sampler.controller import (
    Controller,
    StepLoop,
    BurstLoop,
    burst_capability,
    NotAcceleratable,
)

logger = logging.getLogger("genlm.control")


def _normalize_accelerate(accelerate):
    """Map the ``accelerate`` argument onto the canonical "auto"/"off"/"require".

    Accepts the bare booleans as friendly aliases: ``True`` -> "auto" (so the
    common ``accelerate=True`` never silently *requires* and errors) and
    ``False`` -> "off".
    """
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
            max_tokens (int): Maximum number of tokens to generate per sequence. Generation
                may terminate earlier if all sequences reach an EOS token.
            accelerate (str | bool, optional): The single engine-acceleration knob,
                keyword-only. One of:\n
                - ``"auto"`` (default, also ``True``): run the engine-accelerated
                  `BurstLoop` when the configuration is burst-capable, else the
                  exact per-token `StepLoop`. Logs (INFO) which path ran, and on
                  fallback the reason it was not accelerated.\n
                - ``"off"`` (also ``False``): always run the exact per-token
                  `StepLoop` -- byte-reproducible given a seed (the ground truth).\n
                - ``"require"``: run the engine path, or raise
                  `NotAcceleratable` with the reason if not burst-capable
                  (guarantees the fast path; use for benchmarks / production E-steps).\n
                Acceleration is vLLM-only for now. The engine is derived from the
                sampler's `PromptedLLM` -- you do not pass it. The burst is
                statistically identical to `"off"` (same target, unbiased weights)
                but not byte-identical (warm-KV residual + batched-draw RNG); use
                `"off"` for exact reproducibility.
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
            unit_sampler=self.unit_sampler,
            critic=self.critic,
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            twist_with_critic=ess_threshold > 0,
            record=json_path is not None,
            verbosity=verbosity,
            **kwargs,
        )

        if mode == "off":
            use_burst = False
        else:
            cap = burst_capability(controller)
            if mode == "require" and not cap.ok:
                raise NotAcceleratable(cap.reason)
            use_burst = cap.ok
            if mode == "auto":
                if cap.ok:
                    logger.info("running the engine-accelerated burst path.")
                else:
                    logger.info(
                        "running the exact per-token path -- acceleration "
                        "unavailable: %s. Pass accelerate=\"off\" to silence, or "
                        "accelerate=\"require\" to make this an error.",
                        cap.reason,
                    )

        if use_burst:
            particles = await BurstLoop(controller).run()
        else:
            particles = await StepLoop(controller).run()

        if json_path is not None:
            controller.save_record(json_path)

        return Sequences(*_unpack_particles(particles))

    def acceleration_report(self):
        """One-line human summary of whether this config engine-accelerates, and why.

        Backed by :func:`~genlm.control.sampler.controller.burst_capability` -- the
        same source of truth as the runtime fallback/raise text -- so a user can
        ask the object what it will do before committing a long run, without an
        engine.

        Returns:
            (str): e.g. "Engine-accelerated (vLLM): runs the in-engine burst." or
                "Not accelerated: <reason> -> exact per-token path."
        """
        controller = Controller(
            unit_sampler=self.unit_sampler,
            critic=self.critic,
            n_particles=1,
            ess_threshold=0.0,
            max_tokens=1,
            twist_with_critic=False,
            record=False,
            verbosity=0,
        )
        cap = burst_capability(controller)
        sampler_name = type(self.unit_sampler).__name__
        if cap.ok:
            critic_note = " + terminal critic" if self.critic is not None else ""
            return (
                f"Engine-accelerated (vLLM · {sampler_name}{critic_note}): "
                "runs the in-engine burst, near the raw-decode ceiling."
            )
        return f"Not accelerated: {cap.reason} → exact per-token path."

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
