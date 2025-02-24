import numpy as np
from abc import ABC, abstractmethod
from genlm_grammar import Float
from arsenal.maths import logsumexp, sample_dict
from functools import cached_property
from genlm_control import EOS
from contextlib import contextmanager
from dataclasses import dataclass

from hfppl import Model
from hfppl import smc_standard


@dataclass
class Sequences:
    """Container for sequence samples with their weights and probabilities."""

    contexts: list
    log_weights: list
    log_probs: list

    def __post_init__(self):
        if not (len(self.contexts) == len(self.log_weights) == len(self.log_probs)):
            raise ValueError(
                "Contexts, weights, and probabilities must have the same length"
            )

        self.size = len(self.contexts)
        self.logp = sum(self.log_probs)
        self.log_total = logsumexp(self.log_weights)
        max_weight = max(self.log_weights)
        if np.isfinite(max_weight):
            self.log_ml = (
                np.log(np.mean(np.exp(self.log_weights - max_weight))) + max_weight
            )
        else:
            self.log_ml = float("-inf")
        self.log_normalized_weights = self.log_weights - self.log_total
        self.log_ess = -logsumexp(2 * self.log_normalized_weights)
        self.ess = np.exp(self.log_ess)

    @cached_property
    def posterior(self):
        posterior = Float.chart()
        for sequence, prob in zip(self.contexts, np.exp(self.log_normalized_weights)):
            posterior[tuple(sequence)] += prob
        return posterior.normalize().sort_descending()

    @property
    def normalized_weights(self):
        """Return exponential of normalized log weights."""
        return np.exp(self.log_normalized_weights)

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(zip(self.contexts, self.log_weights))

    def __getitem__(self, i):
        return self.contexts[i], self.log_weights[i]

    def __str__(self):
        return str(self.posterior)

    def _repr_html_(self):
        return self.posterior._repr_html_()

    def show(self):
        for p in sorted(self, reverse=True):
            print(p)


class SequenceModel(Model):
    def __init__(self, unit_sampler, critic=None, max_tokens=float("inf")):
        if max_tokens < 0:
            raise ValueError("max_tokens must be greater than 0")

        super().__init__()
        self.context = []
        self.unit_sampler = unit_sampler
        self.max_tokens = max_tokens
        self.critic = critic
        self.logp = 0
        self.target = (
            unit_sampler.target * critic if critic is not None else unit_sampler.target
        )

    async def start(self):
        # Correct for discrepancy between autoregressive factorization of logw_next
        # and complete.
        self.score(await self.unit_sampler.target.prefix([]))

    async def step(self):
        unit, logw, logp = await self.unit_sampler.sample(self.context)
        self.score(logw)
        self.context.append(unit)
        self.logp += logp

        if self.critic:
            twist_amt = await self.critic.score(self.context)
            self.twist(twist_amt)

        self.max_tokens -= 1
        if self.max_tokens == 0 or self.context[-1] is EOS:
            self.finish()
            if self.critic:
                self.score(twist_amt)
            return

    def immutable_properties(self):
        return set(["unit_sampler", "critic", "target"])

    @contextmanager
    def contextualize(self, context):
        previous_context = self.context
        self.context = context
        try:
            yield self
        finally:
            self.context = previous_context


def _unpack_particles(particles):
    contexts, logws, logps = map(
        list,
        zip(
            *[
                (
                    p.context,
                    float("-inf") if np.isnan(p.weight) else p.weight,
                    p.logp,
                )
                for p in particles
            ]
        ),
    )
    return contexts, logws, logps


class SequenceSampler(ABC):
    """Abstract base class for sequence samplers."""

    def __init__(self, unit_sampler, critic=None, max_tokens=float("inf")):
        self.unit_sampler = unit_sampler
        self.critic = critic
        self.model = SequenceModel(unit_sampler, critic, max_tokens)
        self.model_no_critic = (
            SequenceModel(unit_sampler, None, max_tokens)
            if critic is not None
            else self.model
        )

    @property
    def max_tokens(self):
        return self.model.max_tokens

    @max_tokens.setter
    def max_tokens(self, value):
        self.model.max_tokens = value
        self.model_no_critic.max_tokens = value

    @abstractmethod
    async def sample(self, context=None, draw=sample_dict):
        pass

    @abstractmethod
    async def infer(self):
        pass


class Importance(SequenceSampler):
    def __init__(self, unit_sampler, n_particles, critic=None, max_tokens=float("inf")):
        if n_particles < 1:
            raise ValueError("n_particles must be greater than 0")
        super().__init__(unit_sampler, critic, max_tokens)
        self.n_particles = n_particles

    def sample(self, context=None, draw=sample_dict):
        raise NotImplementedError("SMC does not support sampling")

    async def infer(self, **kwargs):
        particles = await smc_standard(
            model=self.model_no_critic,
            n_particles=self.n_particles,
            ess_threshold=0,
            **kwargs,
        )

        contexts, logws, logps = _unpack_particles(particles)
        assert len(contexts) == len(logws) == len(logps)

        if self.critic is not None:
            # Since we didn't run inference with the critic,
            # we need to add in the critic weight here.
            twist_amts = await self.critic.batch_score(contexts)
            for i in range(len(contexts)):
                logws[i] += twist_amts[i]

        return Sequences(contexts, logws, logps)


class SMC(SequenceSampler):
    def __init__(
        self,
        unit_sampler,
        n_particles,
        ess_threshold,
        critic=None,
        max_tokens=float("inf"),
    ):
        if n_particles < 0:
            raise ValueError("n_particles must be greater than 0")
        if not 0 <= ess_threshold <= 1.0:
            raise ValueError("ess_threshold must be between 0 and 1.0")

        super().__init__(unit_sampler, critic, max_tokens)
        self.n_particles = n_particles
        self.ess_threshold = ess_threshold

    def sample(self, context=None, draw=sample_dict):
        # Eventually implement to support nested SMC.
        raise NotImplementedError("SMC does not support sampling")

    async def infer(self, **kwargs):
        particles = await smc_standard(
            model=self.model,
            n_particles=self.n_particles,
            ess_threshold=self.ess_threshold,
            **kwargs,
        )

        contexts, logws, logps = _unpack_particles(particles)
        return Sequences(contexts, logws, logps)
