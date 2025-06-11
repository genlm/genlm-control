import numpy as np
from arsenal import colors
from llamppl import SubModel
from arsenal.maths import log1mexp, logsumexp

from genlm.control.util import fast_sample_lazyweights
from genlm.control.sampler.set import SetSampler


class TokenSampler(SubModel):
    """Base class for sampling a token from a potential's vocabulary.

    `TokenSampler`s generate properly weighted samples with respect to a `target` potential.

    Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
    a `TokenSampler` samples a token $x_n \\in \\textsf{target.vocab_eos}$ and weight $w$.

    The sampled token and weight are properly weighted with respect to
    $$
    \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
    $$

    Args:
        target (Potential): The potential that samples are properly weighted with respect to.
    """

    def __init__(self, target):
        super().__init__()
        self.target = target
        self.token_type = self.target.token_type

    async def start_weight(self):
        """Compute the weight of the empty sequence under the target potential."""
        return await self.target.prefix([])

    async def forward(self):
        parent = self.parent  # For some reason, need to hold onto this reference.
        token, logw, logp = await self.sample(parent.token_ctx)
        parent.score(logw)
        parent.logp += logp
        return token

    async def sample(self, context, draw):
        """Sample a token and weight from the `target`potential's vocabulary.

        Args:
            context (list[int]): A sequence of tokens in the `target` potential's vocabulary.
            draw (callable): A callable that draws a sample from a distribution.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the sampled token.
        """
        raise NotImplementedError(
            "Subclasses must implement sample method"
        )  # pragma: no cover

    async def cleanup(self):
        pass  # pragma: no cover

    async def smc(self, n_particles, ess_threshold, max_tokens, critic=None, **kwargs):
        """Generate sequences using sequential Monte Carlo (SMC) inference with this token sampler and an optional critic.

        This method is a convenience wrapper around [`SMC`][genlm.control.sampler.sequence.SMC].
        See [`SMC`][genlm.control.sampler.sequence.SMC] for more details on the generation process.

        Args:
            n_particles (int): The number of particles to use in the SMC algorithm.
            ess_threshold (float): The threshold for the effective sample size (ESS).
            max_tokens (int): The maximum number of tokens to generate.
            critic (Potential, optional): A potential function that guides the generation process
                by scoring candidate sequences. Must have the same token type as the token sampler.
            **kwargs (dict): Additional keyword arguments to pass to `SMC`'s `__call__` method.
        """
        from genlm.control.sampler.sequence import SMC

        return await SMC(self, critic)(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            **kwargs,
        )


class DirectTokenSampler(TokenSampler):
    """Samples individual tokens directly from the log-normalized `logw_next` function
    of a potential.

    Args:
        potential (Potential): The potential function to sample from

    Warning:
        Only use this sampler if the potential's `logw_next` method is efficient. This is the case
        for potentials like `PromptedLLM`, but for custom potentials with a large vocabulary size,
        the default implementation of `logw_next` generally will not be efficient, and thus this
        sampler will be slow.
    """

    def __init__(self, potential):
        super().__init__(target=potential)
        self.potential = potential

    async def sample(self, context, draw=None):
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the target potential's vocabulary,
        this method samples a token $x_n \\in \\textsf{target.vocab_eos}$ and weight $w$.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        The returned weight corresponds to the log normalizing constant of $\\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})$.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the sampled token.
        """
        logws = await self.potential.logw_next(context)
        logps = logws.normalize()
        if draw is None:
            # fast sampling from logps using gumbel-max trick
            token = fast_sample_lazyweights(logps)
        else:
            token = draw(logps.exp().materialize())
        return token, logws.sum(), logps[token]

    async def cleanup(self):
        pass  # pragma: no cover


class SetTokenSampler(TokenSampler):
    """Samples individual tokens by sampling a weighted set of tokens and then selecting one
    proportional to its weight.

    This class wraps a `SetSampler`.

    Args:
        set_sampler (SetSampler): The set sampler to sample from
    """

    def __init__(self, set_sampler):
        assert isinstance(set_sampler, SetSampler)
        super().__init__(set_sampler.target)
        self.set_sampler = set_sampler

    async def sample(self, context, draw=None):
        """Sample a token and weight by sampling a weighted set of tokens from the `set_sampler`
        and then selecting one proportional to its weight.

        Given a context of tokens $x_1, \\ldots, x_{n-1}$ in the vocabulary of the set sampler's target potential,
        this method samples a token $x_n \\in \\textsf{set_sampler.target.vocab_eos}$ and a weight.

        The sampled token and weight are properly weighted with respect to
        $$
        \\textsf{set_sampler.target.logw_next}(x_n | x_1, \\ldots, x_{n-1})
        $$

        The returned weight corresponds to the sum of the weights of the sampled set.

        Args:
            context (list[int]): A sequence of tokens in the vocabulary of the set sampler's target potential.

        Returns:
            (token, weight, logp): A tuple containing the sampled token, weight, and log-probability of the random
                choices made in sampling that token.

        Note:
            For properly weighted sampling, the `set_sampler` must assign correct weights to each token. See
            `SetSampler` for more details.
        """
        logws, logp = await self.set_sampler.sample_set(context, draw=draw)
        logps = logws.normalize()
        if draw is None:
            token = fast_sample_lazyweights(logps)
        else:
            token = draw(logps.exp().materialize())
        return token, logws.sum(), logp + logps[token]

    async def cleanup(self):
        """Clean up the sampler.

        This method should be called when the sampler is no longer needed.
        """
        await self.set_sampler.cleanup()


class AWRS(TokenSampler):
    """Samples individual tokens through an adaptive weighted rejection sampling algorithm.

    This sampler is based on the algorithm described in [Fast Controlled Generation from Language Models with Adaptive Weighted Rejection Sampling](https://arxiv.org/abs/2504.05410)

    It draws properly weighted samples from the product of a non-boolean potential and a boolean condition.

    Args:
        potential (Potential): The non-boolean potential.
        condition (Potential): The boolean condition. This potential must only output boolean values (0 or -inf in log-space).
        seed (int): The seed for the random number generator.
        prune_logws (bool): Whether to prune the logws to only include the tokens in the intersection of the potential and condition vocabularies
        proper_weights (bool): Whether to return properly weighted samples.
            If False, the sampler will only run one round of adaptive rejection sampling.
        clip: Parameter that allows low-acceptance probabilities to be rounded to zero. Pick `clip` to be roughly
            the smallest probability of acceptance that you care about.
    """

    def __init__(
        self,
        potential,
        condition,
        seed=42,
        prune_logws=True,
        proper_weights=True,
        clip=0.0,
    ):
        super().__init__(target=potential * condition)
        self.potential = potential
        self.condition = condition

        self.prune_logws = prune_logws
        self.proper_weights = proper_weights
        self.clip = clip
        self.valid_idxs = np.array(
            [self.potential.lookup[t] for t in self.target.vocab_eos]
        )

        self.vocab_eos_set = set(self.target.vocab_eos)
        self.V = len(self.potential.vocab_eos)
        self.rng = np.random.default_rng(seed=seed)

    def _prune_logws(self, logws):
        # Prune the logws to only include the tokens in the
        # target vocabulary. (This zeros-out tokens which we know a priori
        # will be rejected.) Note: We need an additional correction term
        # to account for the fact that we're throwing away some probability mass.
        # This should be handled in `sample`.
        pruned = self.potential.alloc_logws()
        pruned[self.valid_idxs] = logws.weights[self.valid_idxs]
        logws.weights = pruned
        return logws

    async def _accept(self, context, token, verbosity=0):
        if self.prune_logws or token in self.vocab_eos_set:
            if token is self.target.eos:
                logscore = await self.condition.complete(context)
            else:
                logscore = await self.condition.prefix(context + [token])
            assert logscore in {-np.inf, 0}, "`condition` must be Boolean"
        else:
            logscore = -np.inf

        do_accept = logscore == 0

        if verbosity > 0:
            if do_accept:
                print(colors.green % f". {repr(token)}")
            else:
                print(colors.red % ".", end="")

        return do_accept

    async def sample(self, context, verbosity=0):
        """Sample a token and weight that are properly weighted with respect to the target potential's `logw_next` method via adaptive weighted rejection sampling.

        The returned weight corresponds to the log normalizing constant of $\\textsf{target.logw_next}(x_n | x_1, \\ldots, x_{n-1})$.

        Returns:
            (token, weight, np.nan): A tuple containing the sampled token, weight, and a dummy value for the log-probability of the sampled token.
        """
        logws = await self.potential.logw_next(context)
        if self.prune_logws:
            logws = self._prune_logws(logws)

        weights = logws.weights
        logZ = logsumexp(weights)
        logps = weights - logZ
        toks = logws.decode

        # If we are clipping small Z then we add an artificial token at the end
        # that we always accept. If this is the only accepted token we see during
        # sampling, we are allowed to return a weight of zero.
        if self.clip > 0:
            logps = np.concatenate([logps + np.log(1 - self.clip), [np.log(self.clip)]])

        tok, nrej, logp0 = None, 0, []
        for _ in range(2):
            keys = logps - np.log(-np.log(self.rng.random((len(logps)))))
            order = np.argsort(-keys)
            for rank in range(logps.size):
                item = order[rank]
                if keys[item] == -np.inf:
                    break
                if item == len(toks):
                    assert self.clip > 0
                    break
                elif await self._accept(context, toks[item], verbosity):
                    if tok is None:
                        tok = toks[item]
                    break
                else:
                    nrej += 1
                    if tok is None:
                        logp0.append(logps[item])
                    logps[item] = -np.inf

            if not self.proper_weights:
                if tok is None:
                    return self.target.eos, float("-inf"), np.nan
                return tok, 0, np.nan

        if tok is None:  # No token was accepted, return EOS and kill the particle.
            if self.clip > 0:
                # This is a bit fussy, but technically the correctness of this
                # method relies on the branch where we never successfully sampled
                # also being an unbiased estimate of Z, so in this case we do a
                # single loop of the rejection sample, check if it's accepted,
                # and return either 0 or 1 (in logspace) appropriately.
                #
                # Most of the time unless clip is set quite high (e.g. 0.5) this
                # has little benefit.
                keys = logps[:-1] - np.log(-np.log(self.rng.random((len(logps) - 1))))
                tok = toks[np.argmax(-keys)]
                if await self._accept(context, tok, verbosity):
                    return tok, 0.0, np.nan
                else:
                    return tok, -float("inf"), np.nan

            return self.target.eos, float("-inf"), np.nan

        if not logp0:  # Success on first try.
            logw = logZ - np.log(nrej + 1)
        else:
            logw = logZ + log1mexp(logsumexp(logp0)) - np.log(nrej + 1)

        return tok, logw / (1 - self.clip), np.nan
