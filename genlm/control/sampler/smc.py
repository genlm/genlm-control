"""Sequential Monte Carlo over ``SequenceModel`` particles.

Nothing in this module knows an engine exists; serving lives below the potential
layer. One ``smc_standard`` call is one SMC problem; several problems are batched
by running them concurrently under ``asyncio.gather``, and their asks meet below
the potential seam.
"""

import asyncio

import numpy as np
from arsenal import colors

from genlm.control.constant import EOS
from genlm.control.util import logsumexp, escape
from genlm.control.sampler.resampling import get_resampling_fn
from genlm.control.sampler.smc_record import SMCRecord


class SequenceModel:
    """One particle: a candidate sequence's state and its per-step semantics.

    The mutable run state is ``context``, ``weight``, ``twist_amount``,
    ``max_tokens`` and ``done``. ``clone`` copies that state alone; the sampler,
    critic and configuration are shared across siblings.

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
        self.unit_sampler = unit_sampler
        self.critic = critic
        self.max_tokens = max_tokens
        self.twist_with_critic = twist_with_critic
        self.terminate_when = terminate_when
        self.verbosity = verbosity

        self.context = []
        self.weight = 0.0
        self.twist_amount = 0.0
        self.done = False

    def clone(self):
        """Return a particle with copied state and a shared sampler and critic."""
        new = SequenceModel(
            unit_sampler=self.unit_sampler,
            critic=self.critic,
            max_tokens=self.max_tokens,
            twist_with_critic=self.twist_with_critic,
            terminate_when=self.terminate_when,
            verbosity=self.verbosity,
        )
        new.context = list(self.context)
        new.weight = self.weight
        new.twist_amount = self.twist_amount
        new.done = self.done
        return new

    def _add_weight(self, amt):
        """The one place ``weight`` changes. A ``+inf`` log-weight violates the
        potential contract; a NaN weight means impossible, which is ``-inf``."""
        if amt == float("inf"):
            raise ValueError(
                "A potential returned a log-weight of +inf, which violates the "
                "potential contract."
            )
        w = self.weight + amt
        self.weight = float("-inf") if np.isnan(w) else w

    def score(self, amt):
        self._add_weight(amt)

    def twist(self, amt):
        """Add ``amt`` to the weight provisionally; ``untwist`` takes it back."""
        self.twist_amount += amt
        self._add_weight(amt)

    def untwist(self):
        self._add_weight(-self.twist_amount)
        self.twist_amount = 0.0

    def finish(self):
        self.untwist()
        self.done = True

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
        """Advance the particle by one unit, forcing EOS at the token budget."""
        self.untwist()

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

    def __repr__(self):
        return (
            f"{self.weight:.2f}:\t"
            + colors.magenta % "["
            + (colors.magenta % "|").join(escape(y) for y in self.context)
            + colors.magenta % "]"
        )


async def smc_standard(
    model,
    n_particles,
    ess_threshold=0.5,
    resampling_method="multinomial",
    json_path=None,
):
    """Run standard SMC over clones of ``model``.

    One call is one SMC problem; run several concurrently to batch them.

    Args:
        model (SequenceModel): The particle template, cloned ``n_particles`` times.
        n_particles (int): Number of particles.
        ess_threshold (float): Resample when ESS falls below this fraction of
            ``n_particles``.
        resampling_method (str): One of 'multinomial', 'stratified', 'systematic',
            'residual'.
        json_path (str, optional): Path to write the inference record to, for use
            with the ``InferenceVisualizer``.

    Returns:
        (list[SequenceModel]): The completed particles.
    """
    resample_fn = get_resampling_fn(resampling_method)
    particles = [model.clone() for _ in range(n_particles)]
    await asyncio.gather(*[p.start() for p in particles])

    record = SMCRecord(n_particles) if json_path is not None else None
    ancestor_indices = None

    while any(not p.done for p in particles):
        await asyncio.gather(*[p.step() for p in particles if not p.done])

        if record is not None:
            if not record.history:
                record.add_init(particles)
            elif ancestor_indices is not None:
                record.add_resample(ancestor_indices, particles)
            else:
                record.add_smc_step(particles)

        ancestor_indices = None
        W = np.array([p.weight for p in particles])
        if np.all(W == -np.inf):
            continue
        w_sum = logsumexp(W)
        nw = W - w_sum
        with np.errstate(divide="ignore"):
            log_ess = -logsumexp(nw * 2)
            if log_ess >= np.log(ess_threshold) + np.log(n_particles):
                continue

        probs = np.exp(nw)
        probs /= probs.sum()  # np.random.choice is strict on sum==1
        ancestor_indices = list(resample_fn(probs))
        if record is not None:
            ancestor_indices.sort()  # reproducible record
        avg_weight = w_sum - np.log(n_particles)
        particles = [particles[i].clone() for i in ancestor_indices]
        for p in particles:
            p.weight = avg_weight

    if json_path is not None:
        with open(json_path, "w") as f:
            f.write(record.to_json())
        print(f"Saved record to {json_path}")

    return particles
