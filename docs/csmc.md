# Conditional SMC

Conditional Sequential Monte Carlo (C-SMC) is a variant of SMC in which one slot in the particle population is *retained* across resampling rounds. This page covers the math, the API, and how to use it.

The C-SMC inference loop lives upstream in [`llamppl.csmc_standard`][llamppl.csmc_standard]; the retained-particle adapter that plugs it into genlm-control's token-sampler layer lives in [`genlm.control.sampler.csmc_memory`][genlm.control.sampler.csmc_memory]. The whole thing is surfaced through `SMC.__call__(retained_sequence=...)`.

## The target

For a `PromptedLLM` proposal $r = p_\theta$, a critic $\kappa$, and observation $y$, the target posterior over latent token sequences $z \in \Sigma^{*}$ is

$$
\pi_\theta(z \mid x, y) \;=\; \frac{\phi(z)}{Z}, \qquad
\phi(z) \;=\; p_\theta(z \mid x)\,\kappa(z, y),
$$

with $Z = \sum_{z'} \phi(z')$. The library treats $\kappa$ as a `Potential`; in the SMC sweep, an optional shaping function $\psi$ telescopes to $\phi$ at termination and gives the per-token incremental weight $w_t = \psi(z_t \mid z_{<t}) / r(z_t \mid z_{<t})$.

## The C-SMC kernel

Given:

- a **retained particle** $z^{\ast} \in \Sigma^{*}$ — interpreted as the current state of a Markov chain on $\Sigma^{*}$ targeting $\pi_\theta(\cdot \mid x, y)$;
- a particle budget $M \geq 1$;
- a per-token proposal $r$ (in our setting, $r = p_\theta$);

C-SMC runs the SMC sweep of standard `SMC.__call__` with one structural constraint: a designated slot $m^{\ast} \in \{1, \ldots, M\}$ (conventionally $m^{\ast} = 1$) is **force-extended along $z^{\ast}$** at every time step and is **exempt from resampling**. The other $M - 1$ particles are extended freely from $r$ and participate normally in resampling. At termination, an index $R \sim \mathrm{Categorical}\bigl(\widetilde{w}^{(\cdot)}\bigr)$ is drawn from the normalized terminal weights, and the corresponding particle $z^{(R)}$ becomes the new state of the chain.

In the library, the retained slot's role is played by `RetainedTokenSampler`: a transparent wrapper around any `TokenSampler` that, when the host particle has `is_retained == True` and the retained sequence isn't exhausted, forces the next token via the base sampler's `draw` callback. Weights are computed by the base sampler at the forced token, so weight semantics match standard SMC exactly. *Retainedness is a slot property, not a particle property*: after multinomial resampling, only slot 0 has `is_retained == True`, regardless of which particle's trajectory was deepcopied into the other slots.

## Invariance

For any fixed $\theta$, the C-SMC kernel admits $\pi_\theta(\cdot \mid x, y)$ as its invariant distribution, for any $M \geq 1$ and for any choice of intermediate targets (equivalently, shaping function $\psi$) satisfying the absolute-continuity condition $\phi(z) > 0 \Rightarrow \psi(z_{<t}) > 0$ — Andrieu, Doucet & Holenstein (2010), Assumption 1 — together with unbiased multinomial resampling (their Assumption 2).

The original construction wraps C-SMC inside a *Particle Gibbs* sampler that also samples $\theta$ conditional on $(z, y)$, and the joint chain on $(\theta, z)$ is $\pi(\theta, z)$-invariant (their Theorem 5(a)). We use only the conditional-on-$\theta$ kernel; $\theta$ is updated by stochastic gradient ascent on the marginal log-likelihood rather than by sampling.

## Usage

```python
from genlm.control import PromptedLLM, BoolFSA, AWRS

# Set up a standard SMC sampler as usual.
llm = PromptedLLM.from_name("gpt2")
llm.set_prompt_from_str("Here is my honest opinion:")
fsa = BoolFSA.from_regex(r" SMC is (🔥🔥|😍😍|🤌🤌) with LMs")
sampler = AWRS(llm, fsa)

# Standard SMC (no retained particle): retained_sequence=None.
sequences = await sampler.smc(
    n_particles=8, ess_threshold=0.5, max_tokens=32,
)

# C-SMC: pass a retained sequence (as a list of Tokens).
# Typical pattern: the trainer holds the current memory as token IDs
# and converts to Tokens just before the call.
memory_ids = [...]                              # list[int], current memory
retained = llm.decode_tokens(memory_ids)         # list[Token]
sequences = await sampler.smc(
    n_particles=8, ess_threshold=0.5, max_tokens=32,
    retained_sequence=retained,                 # <- triggers C-SMC
)

# To get the next memory e.g. in Gibbs sampling (Andrieu et al., 2010), sample R ~ Categorical(w̃)
# over particles with positive weight:
import numpy as np
weights = sequences.normalized_weights
r = np.random.choice(len(weights), p=weights / weights.sum())
new_memory_ids = llm.encode_tokens(sequences.contexts[r])
```

## Notes

- **`ess_threshold = 0` and a binary terminal critic.** If your critic returns $-\infty$ for partial trajectories (i.e., the answer isn't yet emitted), set `ess_threshold = 0` — `SMC` then skips mid-trajectory critic application and only applies it at termination. Otherwise the partial-trajectory $-\infty$ would kill every particle on the first step.
- **`M = 1`** is the TRICE-degenerate case: the retained particle is the sole anchor and no exploratory particles exist. The kernel is trivially $\pi$-invariant but doesn't mix; useful only as a sanity check.
- **Initialization.** At the start of any C-SMC chain there is no previous trajectory yet. The standard idiom is to bootstrap by running standard SMC once (`retained_sequence=None`), then run C-SMC from the next step onward.

## References

- **Andrieu, Doucet & Holenstein (2010).** "Particle Markov chain Monte Carlo methods." *Journal of the Royal Statistical Society B*, 72(3):269–342. C-SMC is defined in §4.3; Theorem 5(a) establishes joint $(\theta, z)$ invariance for the full Particle Gibbs sampler.
- **Chan et al. (2026).** "Ensembling Sequential Monte Carlo." Provides the proposal / shaping / SIS / SMC framework used throughout `genlm.control`.
