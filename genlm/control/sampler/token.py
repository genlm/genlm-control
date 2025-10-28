import numpy as np
from arsenal import colors
from llamppl import SubModel
from arsenal.maths import log1mexp, logsumexp
import warnings

from genlm.control.util import fast_sample_lazyweights
from genlm.control.sampler.set import SetSampler
import codeop
from collections import defaultdict


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
        ###
        try:
            parent.last_logw_incr = logw
        except Exception:
            pass
        ###
        parent.score(logw)
        parent.logp += logp
        return token

    def on_particle_cloned(self, new_parent, src_parent):
        return

    def on_particle_removed(self, parent):
        return

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


class SampleUntil(TokenSampler):
    def __init__(
        self, lm_potential, stop_tokens=(), patience_init=5, aggregate_weights=False
    ):
        """Sample until a stop token or EOS is reached, then aggregate the weights of the sampled tokens.

        Args:
            lm_potential (Potential): The language model potential to sample from.
            stop_tokens (list): The tokens that define the stop condition.
            patience_init (int): The initial patience value.
            aggregate_weights (bool): Whether to aggregate the weights of the sampled tokens.
        """
        super().__init__(target=lm_potential)
        self._proposal = DirectTokenSampler(lm_potential)
        self._stop_tokens = set(stop_tokens)
        self._patience_init = int(patience_init or 0)
        self.aggregate_weights = aggregate_weights

    def _is_boundary(self, tok):
        return (tok is self.target.eos) or (tok in self._stop_tokens)

    async def sample(self, context, draw=None):
        parent = self.parent
        # Lazily create per-particle fields
        if not hasattr(parent, "_pending_logZ"):
            parent._pending_logZ = 0.0
        if not hasattr(parent, "_patience"):
            parent._patience = self._patience_init

        tok, logZ_i, logp_i = await self._proposal.sample(context, draw=draw)
        if self._is_boundary(tok):

            if self.aggregate_weights:
                # Aggregate the log-weights of the sampled tokens until the boundary
                out_logw = parent._pending_logZ + logZ_i
                parent._pending_logZ = 0.0
            else:
                # Neutral logweight
                out_logw = 0.0
        else:
            parent._pending_logZ += logZ_i
            out_logw = 0.0

        parent.last_logw_incr = out_logw
        return tok, out_logw, logp_i

    async def smc(self, n_particles, ess_threshold, max_tokens, critic=None, **kwargs):
        """
        Override of the SMC method to use the StopTokenSequenceModel.
        """
        from genlm.control.sampler.sequence import SMC, StopTokenSequenceModel

        return await SMC(self, critic)(
            n_particles=n_particles,
            ess_threshold=ess_threshold,
            max_tokens=max_tokens,
            stop_tokens=self._stop_tokens,
            sequence_model=StopTokenSequenceModel,
            **kwargs,
        )

    async def cleanup(self):
        return


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
        n_monte_carlo_samples=None,
    ):
        super().__init__(target=potential * condition)
        self.potential = potential
        self.condition = condition

        self.prune_logws = prune_logws
        self.proper_weights = proper_weights

        if max_accepts < 2 and proper_weights:
            raise ValueError("`max_accepts` must be at least 2")

        if max_rejects < 2 and proper_weights:
            raise ValueError("`max_rejects` must be at least 2")

        if n_monte_carlo_samples is not None:
            warnings.warn(
                "n_monte_carlo_samples no longer does anything.",
                DeprecationWarning,
            )

        self.max_accepts = max_accepts
        self.max_rejects = max_rejects or float("inf")

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

        # We cache successful calls, as algorithms may want to see the
        # same successful token more than once (currently just geometric_awrs)
        cache = {}

        async def accept(tok):
            try:
                return cache[tok]
            except KeyError:
                pass
            result = await self._accept(context, tok, verbosity)
            if result:
                cache[tok] = result
            return result

        if not self.proper_weights:
            return await improper_sample(
                logps=logps,
                toks=toks,
                accept=accept,
                rng=self.rng,
                max_rejects=self.max_rejects,
            )
        # We pick which algorithm to use based on parameters and the
        # shape of the distribution, as this lets us pick the most
        # effective option.
        elif (
            # If max_accepts is large then recursive_awrs (which
            # does not currently support this parameter) isn't very
            # useful, because the recursive step means that you never
            # revisit the same value, so will often throw away most
            # of the accepted mass if you were to continue. Also
            # this parameter is only really relevant if you want to
            # lower the variance, and geometric_awrs is lower variance.
            self.max_accepts > 2
            or
            # If the distribution is strongly peaked around a single value
            # then geometric_awrs will be more efficient. See below
            # for specific derivation.
            logps.max() >= GEOMETRIC_THRESHOLD
        ):
            tok, w, _ = await geometric_awrs(
                logps=logps,
                toks=toks,
                accept=accept,
                rng=self.rng,
                max_rejects=self.max_rejects,
                max_accepts=self.max_accepts,
            )
            return tok, w + logZ, np.nan
        else:
            tok, w, _ = await recursive_awrs(
                logps=logps,
                toks=toks,
                accept=accept,
                rng=self.rng,
                max_rejects=self.max_rejects,
            )
            return tok, w + logZ, np.nan


# If the top log probability exceeds this value, then it will be
# more efficient to use geometric_awrs. This is because the
# expected number of distinct calls is bounded above by 1 +
# a negative binomial distribution with parameters 2, p (
# the number of calls that would be made by sampling with
# replacement before seeing two of the top probability), so
# has expected value 1 + 2(1 - p) / p, and so is < 2 whenever
# p > 2/ 3. As recursive_awrs always makes at least two calls,
# geometric_awrs dominates here.
GEOMETRIC_THRESHOLD = np.log(2 / 3)


async def improper_sample(*, logps, toks, accept, rng, max_rejects):
    """Implements a single rejection sampling loop which returns
    the first value found with no attempt to make a properly
    weighted sample."""
    keys = logps - np.log(-np.log(rng.random((len(logps),))))
    order = np.argsort(-keys)
    if len(order) > max_rejects:
        order = order[:max_rejects]
    for item in order:
        if keys[item] == -np.inf:
            break
        tok = toks[item]
        if await accept(tok):
            return tok, 0.0, np.nan
    return tok, -float("inf"), np.nan


async def recursive_awrs(*, logps, toks, accept, rng, max_rejects):
    """Implements Recursive AWRS.

    This uses the observation that

    E(f(X)) = P(X = x) f(x) + (1 - P(X = x)) E(f(X)|X != x)

    To construct a recursive estimator of the weight from a single
    sampling-with-rejection run. The first time accept(x) passes,
    we use a simple coin flip estimator for the tail.
    """
    n_accepts = 0
    n_rejects = 0

    rejected_mass = 0.0
    log_multiplier = 0.0

    # We treat any number smaller than this as "effetively" zero.
    # This causes us to terminate early in some cases, but those
    # cases are all ones where the remaining weight is very bad.
    error_tolerance = 10e-6

    keys = logps - np.log(-np.log(rng.random((len(logps),))))
    order = np.argsort(-keys)
    for index_into_list, item in enumerate(order):
        assert n_accepts == 0
        tok = toks[item]
        last = (
            index_into_list + 1 == len(order)
            or keys[order[index_into_list + 1]] == -np.inf
        )

        log_q = logps[item] - np.log1p(-rejected_mass)

        # The last check is because in the case where there is a single
        # accepted token with very low log probability, numerical stability
        # issues make it very hard to get this calculation right.
        assert not last or log_q >= -error_tolerance or logps[item] < -32
        assert log_q <= error_tolerance
        assert log_multiplier <= error_tolerance
        assert rejected_mass <= 1

        # Fix some minor numerical stability errors that can come up.
        if last:
            log_q = 0
        log_q = min(log_q, 0)
        log_multiplier = min(log_multiplier, 0)

        if await accept(toks[item]):
            n_accepts += 1
            if n_rejects == max_rejects - 1:
                return tok, log_multiplier, np.nan
            elif last:
                final_estimator = 0.0
            else:
                next_token = toks[order[index_into_list + 1]]
                if await accept(next_token):
                    final_estimator = 0
                else:
                    final_estimator = log_q
            logp = log_multiplier + final_estimator
            return tok, logp, np.nan
        elif last or n_rejects == max_rejects - 1:
            # No token was accepted, return a rejected token and kill the particle.
            return tok, float("-inf"), np.nan
        else:
            n_rejects += 1
            rejected_mass += np.exp(logps[item])
            if rejected_mass >= 1 - error_tolerance:
                # We've explored all the probability mass and still found no
                # accepted token.
                return tok, float("-inf"), np.nan
            m = log1mexp(log_q)
            assert not np.isnan(m)
            log_multiplier += m
        assert not last

    raise AssertionError("Unreachable")


async def geometric_awrs(*, logps, toks, accept, rng, max_rejects, max_accepts):
    """Implements Geometric AWRS.

    This simulates a single run of sampling with replacement from a sampling
    without replacement run, reconstructing the counts of "phantom" elements
    discarded from the without-replacement run as a series of draws from
    geometric distributions. We can then use an appropriate estimator
    for the with-replacement run at the end.
    """
    n_accepts = 0
    n_rejects = 0

    rejected_mass = 0.0
    result = None
    rejected_token = None

    for _ in range(max_accepts):
        if n_rejects >= max_rejects:
            break
        keys = logps - np.log(-np.log(rng.random((len(logps),))))
        order = np.argsort(-keys)
        for item in order:
            if keys[item] == -np.inf:
                break

            tok = toks[item]

            if rejected_mass >= 1:
                # If rejected mass is >= 1 but we have a non-zero probability
                # we've really had numerical precision issues that rounded us to 1.
                # However, this means that the correct estimator is ridiculously
                # small, and we'd exceed any reasonable `max_rejects`, so we just
                # immediately terminate in this case.
                #
                # This can technically happen after we've seen an accepted token
                # but this only happens if the distribution / constraint has gone
                # very wrong.
                assert rejected_token is not None
                return rejected_token, -float("inf"), np.nan
            elif rejected_mass > 0:
                # Add a geometric distribution with parameter 1 - rejected_mass
                # to the number of rejects, account for the phantom tokens
                # "hidden" by sampling without replacement.
                phantom_tokens = rng.geometric(1 - rejected_mass) - 1
                assert phantom_tokens >= 0
                n_rejects += phantom_tokens

            if n_rejects >= max_rejects:
                break

            if await accept(tok):
                n_accepts += 1
                if result is None:
                    result = tok
                break
            else:
                if rejected_token is None:
                    rejected_token = tok
                n_rejects += 1
                rejected_mass += np.exp(logps[item])
                logps[item] = -float("inf")

    if n_accepts == 0:
        assert rejected_token is not None
        return rejected_token, -np.inf, np.nan

    # If we stopped in the middle of a sequence of phantom tokens,
    # n_rejects may have gone over max_rejects.
    n_rejects = min(n_rejects, max_rejects)

    # The correctness of this estimator can be verified by applying
    # the Rao-Blackwell theorem to the estimator that just returns
    # 1 if the first sample was accepted and 0 if it was rejected
    # to the sufficient statistic (n_accepts, n_rejects). Some
    # straightforward sequence counting gives you this estimator.
    estimator = min(max_accepts - 1, n_accepts) / (n_accepts + n_rejects - 1)

    assert estimator > 0 or result is None

    return result, np.log(estimator), np.nan
