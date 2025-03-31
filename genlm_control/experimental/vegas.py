import time
import asyncio
import numpy as np
from arsenal import colors
from arsenal.maths import log1mexp, logsumexp, sample_dict
from genlm_control.sampler.token import TokenSampler


class LasVegasTokenSampler(TokenSampler):
    def __init__(self, potential, condition, prune_logws=True, log_stats=False):
        super().__init__(target=potential * condition)
        self.potential = potential
        self.condition = condition

        self.prune_logws = prune_logws
        self.valid_idxs = np.array(
            [self.potential.lookup[t] for t in self.target.vocab_eos]
        )

        self.vocab_eos_set = set(self.target.vocab_eos)
        self.V = len(self.potential.vocab_eos)

        self.log_stats = log_stats
        if self.log_stats:
            self._reset_stats()

    def _get_sample_id(self):
        if self.log_stats:
            self._sample_id += 1
            return self._sample_id
        else:
            return None

    async def sample(self, *args, **kwargs):
        if self.log_stats:
            start_time = time.process_time()
            sample_id = self._get_sample_id()
            self._calls[sample_id] = []
            kwargs["_sample_id"] = sample_id

        result = await self._sample(*args, **kwargs)

        if self.log_stats:
            self._total_times[sample_id] = time.process_time() - start_time
            self._results[sample_id] = result

        return result

    async def _sample(self, *args, **kwargs):
        raise NotImplementedError("`_sample` must be implemented by subclasses")

    async def accept(self, context, token, verbosity=0, _sample_id=None):
        if self.log_stats and _sample_id is None:
            raise ValueError("`_sample_id` must be provided when logging is enabled")

        if self.prune_logws or token in self.vocab_eos_set:
            if token is self.target.eos:
                logscore = await self.condition.complete(context)
                if self.log_stats:
                    self._calls[_sample_id].append(("complete", logscore, token))
            else:
                logscore = await self.condition.prefix(context + [token])
                if self.log_stats:
                    self._calls[_sample_id].append(("prefix", logscore, token))
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

    async def get_logws(self, context, _sample_id=None):
        if self.log_stats:
            assert _sample_id is not None
            self._contexts[_sample_id] = list(context)
            start_time = time.process_time()

        logws = await self.potential.logw_next(context)

        if self.log_stats:
            self._logws_times[_sample_id] = time.process_time() - start_time

        return self._prune_logws(logws) if self.prune_logws else logws

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

    async def _monte_carlo(self, context, N, **kwargs):
        # Used for testing.
        samples = await asyncio.gather(
            *[self.sample(context, **kwargs) for _ in range(N)]
        )
        logws = self.target.alloc_logws()
        for tok, logw, _ in samples:
            if logw == float("-inf"):
                continue
            token_id = self.target.lookup[tok]
            logws[token_id] = logsumexp([logws[token_id], logw - np.log(N)])
        return self.target.make_lazy_weights(logws)

    def _reset_stats(self):
        self._sample_id = -1
        self._calls = {}
        self._contexts = {}
        self._total_times = {}
        self._logws_times = {}
        self._results = {}

    def get_stats(self):
        if not self.log_stats:
            raise ValueError("Logging is not enabled")
        keys = ["calls", "contexts", "logws_times", "total_times", "results"]
        stats = {k: [] for k in keys}
        for sample_id in range(self._sample_id + 1):
            for key in keys:
                stats[key].append(getattr(self, f"_{key}")[sample_id])
        return stats


class RejectionSampler(LasVegasTokenSampler):
    # Run M + 1 rejection sampling loops.
    def __init__(self, potential, condition, M=1, **kwargs):
        super().__init__(potential, condition, **kwargs)
        assert M >= 1, "`M` must be at least 1"
        self.M = M

    async def _sample(self, context, verbosity=0, _sample_id=None):
        logws = await self.get_logws(context, _sample_id)
        logZ = logsumexp(logws.weights)
        ps = logws.normalize().exp().materialize()

        tok, logp, nrej = None, 0, 0
        for _ in range(self.M + 1):
            while True:
                token = sample_dict(ps)
                logp += np.log(ps[token])
                if await self.accept(context, token, verbosity, _sample_id):
                    if tok is None:
                        tok = token
                    break
                nrej += 1

        assert tok is not None, "No token was accepted"
        logw = logZ + np.log(self.M) - np.log(nrej + self.M)
        return tok, logw, logp


class WithoutReplacementSampler(LasVegasTokenSampler):
    # Sampling without replacement.
    def __init__(self, potential, condition, seed=42, **kwargs):
        super().__init__(potential, condition, **kwargs)
        self.rng = np.random.default_rng(seed=seed)

    async def _sample(self, context, verbosity=0, _sample_id=None):
        logws = await self.get_logws(context, _sample_id)
        logZ = logsumexp(logws.weights)
        logps = logws.normalize()

        toks = logps.decode
        keys = logps.weights - np.log(-np.log(self.rng.random((self.V,))))
        order = np.argsort(-keys)
        tok, logtau = None, None
        for rank in range(self.V):
            item = order[rank]
            token = toks[item]
            if await self.accept(context, token, verbosity, _sample_id):
                if tok is None:
                    tok = token
                else:
                    logtau = keys[item]
                    break  # Break when we've accepted two tokens.

        assert tok is not None, "No token was accepted"  # TODO: return EOS, -inf

        logp0 = logps[tok]
        logw = logZ + logp0
        if logtau is not None:
            # Multiple tokens were accepted.
            logw -= log1mexp(-np.exp(logp0 - logtau))

        return tok, logw, logp0


GumbelMaxRejectionSampler = WithoutReplacementSampler


class AdaptiveRejectionSampler(LasVegasTokenSampler):
    # Same as RejectionTokenSampler, but we adaptively zero-out rejected tokens.

    async def _sample(self, context, verbosity=0, _sample_id=None):
        logws = await self.get_logws(context, _sample_id)
        logZ = logsumexp(logws.weights)
        logps = logws.normalize()
        ps = logps.exp().materialize()

        tok, nrej, logp0 = None, 0, []
        for _ in range(2):
            while True:
                token = sample_dict(ps)
                if await self.accept(context, token, verbosity, _sample_id):
                    if tok is None:
                        tok = token
                    break
                elif tok is None:
                    logp0.append(logps[token])

                nrej += 1
                ps[token] = 0
                ps = ps.normalize()

        assert tok is not None, "No token was accepted"

        if not logp0:
            logw = logZ - np.log(nrej + 1)
        else:
            logw = logZ + log1mexp(logsumexp(logp0)) - np.log(nrej + 1)

        return tok, logw, np.nan


class GumbelMaxAdaptiveRejectionSampler(LasVegasTokenSampler):
    """Sampling with adaptive replacement"""

    def __init__(self, potential, condition, seed=42, proper_weights=True, **kwargs):
        super().__init__(potential, condition, **kwargs)
        self.rng = np.random.default_rng(seed=seed)
        self.proper_weights = proper_weights

    async def _sample(self, context, verbosity=0, _sample_id=None):
        logws = await self.get_logws(context, _sample_id)
        logZ = logsumexp(logws.weights)
        logps = logws.weights - logZ
        toks = logws.decode

        tok, nrej, logp0 = None, 0, []
        for _ in range(2):
            keys = logps - np.log(
                -np.log(self.rng.random((self.V,)))
            )  # throw this into init
            order = np.argsort(-keys)
            for rank in range(logps.size):
                item = order[rank]
                if keys[item] == -np.inf:
                    break
                if await self.accept(context, toks[item], verbosity, _sample_id):
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
                    # Check if all logps are -np.inf
                    non_inf_indices = np.where(logps != -np.inf)[0]
                    assert len(non_inf_indices) == 0, (
                        f"Expected all logps to be -np.inf, but found values at indices: {non_inf_indices}"
                    )
                    return self.target.eos, float("-inf"), np.nan
                return tok, 0, np.nan

        if tok is None:  # No token was accepted, return EOS and kill the particle.
            return self.target.eos, float("-inf"), np.nan

        if not logp0:  # Success on first try.
            logw = logZ - np.log(nrej + 1)
        else:
            logw = logZ + log1mexp(logsumexp(logp0)) - np.log(nrej + 1)

        return tok, logw, np.nan


class ClippedAdaptiveRejectionSampler(LasVegasTokenSampler):
    def __init__(
        self,
        potential,
        condition,
        top_p1=0.95,
        top_p2=0.99,
        seed=42,
        proper_weights=True,
        **kwargs,
    ):
        if not (0 < top_p1 <= 1 and 0 < top_p2 <= 1):
            raise ValueError("`top_p1` and `top_p2` must be between 0 and 1")
        super().__init__(potential, condition, **kwargs)
        self.rng = np.random.default_rng(seed=seed)
        self.top_logps = [np.log(top_p1), np.log(top_p2)]
        self.proper_weights = proper_weights

    async def _sample(self, context, verbosity=0, _sample_id=None):
        logws = await self.get_logws(context, _sample_id)
        logZ = logsumexp(logws.weights)
        logps = logws.weights - logZ
        toks = logws.decode

        tok, nrejs, logp0s = None, [0, 0], [-np.inf, -np.inf]
        for i in range(2):
            keys = logps - np.log(-np.log(self.rng.random((self.V,))))
            order = np.argsort(-keys)
            for rank in range(logps.size):
                item = order[rank]
                if await self.accept(context, toks[item], verbosity, _sample_id):
                    if tok is None:
                        assert i == 0
                        tok = toks[item]
                    break
                else:
                    nrejs[i] += 1
                    logp0s[i] = logsumexp([logp0s[i], logps[item]])
                    logps[item] = -np.inf
                    if logp0s[i] > self.top_logps[i]:
                        if i == 0:
                            assert tok is None
                            if (
                                rank + 1 < logps.size
                            ):  # check if there are more tokens to sample
                                tok = toks[order[rank + 1]]
                                if not await self.accept(
                                    context, tok, verbosity, _sample_id
                                ):
                                    return tok, float("-inf"), np.nan
                        break

            # Initialize second round with first round's accumulated mass
            if i == 0:
                logp0s[1] = logp0s[0]

        if tok is None:  # No token was accepted, return EOS and kill the particle.
            return self.target.eos, float("-inf"), np.nan

        if not self.proper_weights:
            return tok, 0, np.nan

        log1mp = log1mexp(logp0s[0]) if logp0s[0] != -np.inf else 0

        if logp0s[0] < self.top_logps[0]:
            logw = logZ + log1mp - np.log(np.sum(nrejs) + 1)
        else:
            logw = logZ + log1mp - np.log(np.sum(nrejs) + 1) + np.log(nrejs[1] + 1)

        return tok, logw, np.nan
