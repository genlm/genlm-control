import numpy as np
from arsenal import colors
from llamppl import SubModel
from arsenal.maths import logsumexp

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
        seed (int or None): The seed for the random number generator.
        prune_logws (bool): Whether to prune the logws to only include the tokens in the intersection of the potential and condition vocabularies
        proper_weights (bool): Whether to return properly weighted samples.
            If False, the sampler will only run one round of adaptive rejection sampling.
        max_accepts (int): The maximum number of tokens to accept - higher values will decrease the variance of the weight estimate.
        max_rejects (int or float('inf')): The maximum number of tokens to reject - lower values will run faster, but at the cost of returning a weight of zero for some samples where there are tokens that would be accepted if tested.
        n_monte_carlo_samples (int): The number of Monte Carlo samples to use to estimate the weight. Higher values will decrease the variance of the weight estimate, but will run slower.
    """

    def __init__(
        self,
        potential,
        condition,
        seed=None,
        prune_logws=True,
        proper_weights=True,
        max_accepts=2,
        max_rejects=float("inf"),
        n_monte_carlo_samples=10,
    ):
        super().__init__(target=potential * condition)
        self.potential = potential
        self.condition = condition

        self.prune_logws = prune_logws
        self.proper_weights = proper_weights

        if max_accepts < 2:
            raise ValueError("`max_accepts` must be at least 2")

        if max_rejects < 2:
            raise ValueError("`max_rejects` must be at least 2")

        if n_monte_carlo_samples < 1:
            raise ValueError("`n_monte_carlo_samples` must be at least 1")

        self.max_accepts = max_accepts
        self.max_rejects = max_rejects or len(self.potential.vocab_eos)
        self.n_monte_carlo_samples = n_monte_carlo_samples

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

        logZ = logsumexp(logws.weights)
        logps = logws.weights - logZ
        toks = logws.decode

        # Note that this is a different algorithm than the one described
        # in the paper.
        #
        # Rather than use a RAVI-based estimator for the weight, we reduce
        # the sampling without replacement process to a sampling with
        # replacement process.
        #
        # This works by imagining that each token produced by the sampling
        # without replacement is preceded by some number of tokens that have
        # previously been seen. We don't need to know what those tokens are,
        # only how many of them there are, which can be calculated by sampling
        # from a geometric distribution with parameter equal to the total mass
        # of tokens that have previously been removed from our distribution.
        #
        # This will have a significantly lower variance than the estimator
        # from the paper. It might in fact be the Minimum Variance Unbiased
        # Estimator (MVUE) for the weight, but we're not 100% sure of the
        # details. Informal experiments suggest that it has about 2-5x lower
        # variance than the paper's estimator. It also allows a clean
        # implementation of the max_rejects parameter.

        replacement_probabilities = [0.0]
        accepted = []

        n_accepts = 0
        n_rejects = 0

        tok = None
        tok_logp = -float("inf")
        rejected_tok = None
        progress = True

        while (
            n_accepts < self.max_accepts and n_rejects < self.max_rejects and progress
        ):
            progress = False
            keys = logps - np.log(-np.log(self.rng.random((self.V,))))
            order = np.argsort(-keys)
            for item in order:
                if keys[item] == -np.inf:
                    break
                progress = True
                if await self._accept(context, toks[item], verbosity):
                    accepted.append(True)
                    replacement_probabilities.append(replacement_probabilities[-1])
                    if tok is None:
                        tok = toks[item]
                        tok_logp = logps[item]
                    n_accepts += 1
                    break
                else:
                    rejected_tok = toks[item]
                    accepted.append(False)
                    replacement_probabilities.append(
                        replacement_probabilities[-1] + np.exp(logps[item])
                    )
                    logps[item] = -np.inf
                    n_rejects += 1
                    if n_rejects == self.max_rejects:
                        break

            if not self.proper_weights:
                if tok is None:
                    return self.target.eos, float("-inf"), np.nan
                return tok, 0, np.nan

        # No token was accepted, return a rejected tokenand kill the particle.
        if tok is None:
            return rejected_tok, float("-inf"), np.nan

        if n_rejects == 0:
            return tok, logZ, np.nan

        def calc_estimator(local_accepts, local_rejects):
            # This is an estimator for the probability of acceptance,
            # from a random variable that samples with replacement until
            # it sees either a certain number of accepted tokens or a certain
            # number of rejected tokens.
            #
            # You can work out this estimator by applying the Rao-Blackwell
            # theorem, starting from the estimator that returns 1 if the first
            # sample is accepted, and 0 otherwise, conditioned on the total number
            # of accepted and rejected tokens seen. When you do this, it reduces
            # to a sequence counting problem, by looking at the fraction of sequences
            # that have those counts which start with 1.

            denominator = local_rejects + local_accepts - 1

            if local_accepts == self.max_accepts:
                return (self.max_accepts - 1) / denominator
            else:
                assert local_rejects == self.max_rejects
                return local_accepts / denominator

        novel_probabilities = 1 - np.array(replacement_probabilities[:-1])
        # The novel probability should never be less than the probability of
        # the token we accepted, but it can be for numerical stability reasons.
        # We boost it to avoid zero probabilities.
        novel_probabilities = np.maximum(novel_probabilities, np.exp(tok_logp))

        for i, x in enumerate(novel_probabilities):
            if x < 1:
                novel_start = i
                break
        sub_one_probabilities = novel_probabilities[novel_start:]

        # If we have successfully found a token but are very close to
        # the maximum number of rejects, it's possible for the simulation
        # of the sampling with replacement to always exceed the maximum
        # number of rejections, which gives us a weight of zero despite
        # having successfully found a token. We don't want to do that.
        #
        # So we split the estimator up into two parts: One where all
        # of the geometric distributions rolled zero, which we can calculate
        # the probability of exactly and can estimate in that case, and do
        # the monte-carlo simulation conditional on at least one of the
        # geometric samples rolling non-zero. We then combine the two
        # estimators at the end.
        logp_all_zero = np.log(novel_probabilities).sum()

        # Note: In many cases this rounds to zero, but that's actually fine.
        # The difference between 1 and 1 - epsilon is negligible for these
        # calculations.
        logp_non_zero = np.log1p(-np.exp(logp_all_zero))

        # This is the estimator we get when every geometric distribution
        # rolled zero. It is guaranteed to be > 0 because we've seeen
        # at least one accepted token.
        base_estimator = calc_estimator(n_accepts, n_rejects)

        def gen_monte_carlo_samples(n_samples):
            # We want geometric samples for the monte carlo simulation, but
            # annoyingly numpy's geometric distribution doesn't allow us to
            # set p=1, so we create the initial samples as zeros, then
            # concatenate them with the geometric samples to get the whole
            # sample for the simulation.
            initial_zeros = np.zeros(
                (n_samples, len(novel_probabilities[:novel_start]))
            )
            geometric_samples = (
                self.rng.geometric(
                    sub_one_probabilities,
                    size=(n_samples, len(sub_one_probabilities)),
                )
                - 1
            )

            samples = np.concatenate((initial_zeros, geometric_samples), axis=1)

            assert samples.shape == (
                n_samples,
                len(novel_probabilities),
            )
            return samples

        estimators = []

        # Because the monte carlo simulation is done conditionally on
        # not all of the geometric samples rolling zero, we may need
        # to run it a few times to get the number of samples we need.
        n_monte_carlo_samples_done = 0
        while n_monte_carlo_samples_done < self.n_monte_carlo_samples:
            for sample in gen_monte_carlo_samples(
                self.n_monte_carlo_samples - n_monte_carlo_samples_done
            ):
                # Simulate the sampling with replacement process, by modelling
                # a number of discarded tokens that were previously seen,
                # inserted before each novel token in the rejection sample.

                if not sample.any():
                    continue

                n_monte_carlo_samples_done += 1

                if sample.sum() + n_rejects < self.max_rejects:
                    local_rejects = sample.sum() + n_rejects
                    local_accepts = self.max_accepts
                else:
                    local_accepts = 0
                    local_rejects = 0

                    for i in range(len(accepted)):
                        if (
                            local_accepts == self.max_accepts
                            or local_rejects == self.max_rejects
                        ):
                            break

                        local_rejects += sample[i]
                        if local_rejects >= self.max_rejects:
                            local_rejects = self.max_rejects
                            break

                        if accepted[i]:
                            local_accepts += 1
                        else:
                            local_rejects += 1
                estimators.append(calc_estimator(local_accepts, local_rejects))

        estimators_prediction = np.mean(estimators)

        # p = p_all_zero * base_estimator + p_non_zero * estimators_prediction
        # The following calculation just does this in log-space.

        # This can be zero if `max_rejects` is finite and the rejection
        # sampling has thrown away a very large amount of probability mass.
        if estimators_prediction > 0:
            logp = logsumexp(
                (
                    logp_all_zero + np.log(base_estimator),
                    logp_non_zero + np.log(estimators_prediction),
                )
            )
        else:
            logp = logp_all_zero + np.log(base_estimator)

        assert 0 >= logp > -float("inf"), logp

        logw = logp + logZ

        return tok, logw, np.nan
